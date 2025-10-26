from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from pocketoptionapi.stable_api import PocketOption
except Exception:
    PocketOption = None

@dataclass
class BotParams:
    ema_fast: int = 19
    ema_slow: int = 21
    cci_period: int = 20
    cci_overbought: float = 100.0
    cci_oversold: float = -100.0
    cci_lookback: int = 6
    hold_bars: int = 1
    atr_period: int = 14
    atr_multiplier: float = 0.0


def _validate_df(df: pd.DataFrame):
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logging.error("Missing columns: %s", missing)
        raise ValueError(f"Input dataframe missing required columns: {sorted(missing)}")


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = typical_price(df)
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    denom = 0.015 * mad
    denom = denom.replace(0, np.nan)
    return (tp - sma) / denom


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def compute_indicators(df: pd.DataFrame, params: BotParams) -> pd.DataFrame:
    _validate_df(df)
    out = df.copy()
    out["ema_fast"] = ema(out["close"], params.ema_fast)
    out["ema_slow"] = ema(out["close"], params.ema_slow)
    out["ema_slow_slope"] = out["ema_slow"].diff()
    out["cci"] = cci(out, params.cci_period)
    out["cci_roll_min"] = out["cci"].rolling(params.cci_lookback, min_periods=1).min()
    out["cci_roll_max"] = out["cci"].rolling(params.cci_lookback, min_periods=1).max()
    out["atr"] = atr(out, params.atr_period)
    return out


def detect_cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)


def detect_cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


def generate_signals(df: pd.DataFrame, params: BotParams) -> pd.DataFrame:
    data = compute_indicators(df, params)
    sig = pd.Series("", index=data.index)
    cross_up = detect_cross_up(data["ema_fast"], data["ema_slow"])
    cross_down = detect_cross_down(data["ema_fast"], data["ema_slow"])
    bull = cross_up & (data["cci"] > params.cci_oversold) & (data["cci_roll_min"] <= params.cci_oversold) & (data["ema_slow_slope"].shift(1) < 0)
    bear = cross_down & (data["cci"] < params.cci_overbought) & (data["cci_roll_max"] >= params.cci_overbought) & (data["ema_slow_slope"].shift(1) > 0)
    if params.atr_multiplier and params.atr_multiplier > 0:
        atr_filter = data["atr"] >= (data["atr"].shift(1) * params.atr_multiplier)
        bull = bull & atr_filter
        bear = bear & atr_filter
    sig.loc[bull] = "BUY"
    sig.loc[bear] = "SELL"
    data["signal"] = sig
    return data


def backtest_signals(df: pd.DataFrame, params: BotParams, verbose: bool = False) -> pd.DataFrame:
    df_signals = generate_signals(df, params)
    mask = df_signals["signal"] != ""
    positions = np.flatnonzero(mask.to_numpy())
    trades = []
    n = len(df_signals)
    for pos in positions:
        entry_index = df_signals.index[pos]
        entry_price = df_signals.iloc[pos]["open"]
        exit_pos = pos + int(params.hold_bars)
        if exit_pos >= n:
            continue
        exit_index = df_signals.index[exit_pos]
        exit_price = df_signals.iloc[exit_pos]["close"]
        direction = df_signals.iloc[pos]["signal"]
        pnl = (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)
        trades.append({"entry_time": entry_index, "exit_time": exit_index, "direction": direction, "entry_price": entry_price, "exit_price": exit_price, "pnl": pnl})
        if verbose:
            logging.info("Trade: %s %s -> PnL=%s", entry_index, direction, pnl)
    if not trades:
        return pd.DataFrame(columns=["entry_time", "exit_time", "direction", "entry_price", "exit_price", "pnl"]) 
    out = pd.DataFrame(trades)
    out["cumulative_pnl"] = out["pnl"].cumsum()
    return out


class LiveAdapterPocketOption:
    def __init__(self, ssid: str, demo: bool = True):
        self.ssid = ssid
        self.demo = demo
        self._last_ts = None

    def run(self, on_new_candle):
        if PocketOption is None:
            raise ModuleNotFoundError("pocketoptionapi is required for live adapter")
        account = PocketOption(self.ssid, demo=self.demo)
        ok, msg = account.connect()
        if not ok:
            raise RuntimeError(msg)
        asset = "EURUSD"
        timeframe = "3m"
        import time
        while True:
            candles = account.get_candles(asset, timeframe, limit=10)
            if not candles:
                time.sleep(1)
                continue
            for c in candles:
                ts = c.get("timestamp") or c.get("t") or c.get("time")
                if ts is None:
                    on_new_candle(c)
                    continue
                try:
                    ts_val = pd.to_datetime(ts, unit="s") if isinstance(ts, (int, float)) else pd.to_datetime(ts)
                except Exception:
                    on_new_candle(c)
                    continue
                if self._last_ts is not None and ts_val <= self._last_ts:
                    continue
                self._last_ts = ts_val
                on_new_candle({"timestamp": ts_val, "open": c.get("o") or c.get("open"), "high": c.get("h") or c.get("high"), "low": c.get("l") or c.get("low"), "close": c.get("c") or c.get("close"), "volume": c.get("v") or c.get("volume")})
            time.sleep(1)
