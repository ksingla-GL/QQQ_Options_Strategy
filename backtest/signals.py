"""
Entry signals from QQQ closes.

* 15-minute bars close at :00/:15/:30/:45 (first bar 09:45).
* 1-hour bars are anchored at the 09:30 open (10:30, 11:30, ... 15:30, then the
  session close). Early-close days end at their close.
* Indicators run on the continuous bar series across days:
  TSF = time-series moving average (linear-regression endpoint),
  MACD(12, 26, 9) histogram, RSI(14) with Wilder smoothing.

trend state: +1 up (TSF fast > TSF slow and close > TSF fast),
             -1 down (TSF fast < TSF slow and close < TSF fast), 0 neutral.
"""

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .market import minute_key


# ---------------------------------------------------------------------- #
# Indicators
# ---------------------------------------------------------------------- #
def time_series_ma(s: pd.Series, n: int) -> pd.Series:
    """Linear-regression endpoint over the last n values (the 'time series' moving average)."""
    y = s.values.astype("float64")
    out = np.full(len(y), np.nan)
    if len(y) >= n:
        x = np.arange(n, dtype="float64")
        sx, sxx = x.sum(), (x * x).sum()
        sy = np.convolve(y, np.ones(n), "valid")
        sxy = np.convolve(y, x[::-1], "valid")
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        intercept = (sy - slope * sx) / n
        out[n - 1:] = intercept + slope * (n - 1)
    return pd.Series(out, index=s.index)


def macd_hist(s: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    line = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return line - line.ewm(span=signal, adjust=False).mean()


def rsi(s: pd.Series, n: int) -> pd.Series:
    """Wilder RSI (the first bar counts as a zero move)."""
    d = s.diff()
    gain = d.where(d > 0, 0.0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    loss = (-d).where(d < 0, 0.0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def trend_state(close: pd.Series, fast: pd.Series, slow: pd.Series) -> pd.Series:
    up = (fast > slow) & (close > fast)
    dn = (fast < slow) & (close < fast)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=close.index)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """Bollinger(20, 2): (position in band 0..1, bandwidth / close)."""
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std()
    upper, lower = mid + k * sd, mid - k * sd
    return (close - lower) / (upper - lower), (upper - lower) / close


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 7, mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Supertrend (line value, direction +1 support / -1 resistance), with an SMA ATR."""
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    hl2 = (high + low) / 2
    bu, bl = (hl2 + mult * atr).values, (hl2 - mult * atr).values
    c = close.values
    n = len(c)
    fu, fl = np.full(n, np.nan), np.full(n, np.nan)
    d = np.zeros(n, dtype=int)
    for i in range(period, n):
        if np.isnan(bu[i]):
            continue
        if i == period or np.isnan(fu[i - 1]):
            fu[i], fl[i] = bu[i], bl[i]
            d[i] = 1 if c[i] > fu[i] else -1
            continue
        fu[i] = bu[i] if (bu[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = bl[i] if (bl[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < fl[i] else 1
        else:
            d[i] = 1 if c[i] > fu[i] else -1
    st = np.where(d == 1, fl, np.where(d == -1, fu, np.nan))
    return pd.Series(st, index=close.index), pd.Series(d, index=close.index)


# ---------------------------------------------------------------------- #
# Bars
# ---------------------------------------------------------------------- #
def _session_bars(minutes: pd.Series, sess: pd.DataFrame, step: int, anchor_open: bool) -> pd.Series:
    """Bar closes from the minute series: the price at each bar-end minute.

    step=15 -> clock quarter hours; step=60 with anchor_open -> 10:30, 11:30, ...
    The session close is always a bar end (it ends the last, possibly short, bar).
    """
    idx = minutes.index
    mins = np.asarray(idx.hour * 60 + idx.minute)
    since_open = mins - (9 * 60 + 30)
    on_grid = (since_open % step == 0) if anchor_open else (mins % step == 0)
    close_key = {d: minute_key(c) for d, c in sess["close"].items()}
    ck = pd.Series(idx.date).map(close_key).values
    valid = ~pd.isna(ck)
    ck = np.where(valid, ck, -1).astype("int64")
    ik = minute_key(idx)
    keep = valid & (since_open > 0) & (ik <= ck) & (on_grid | (ik == ck))
    return minutes[keep]


def _session_ohlc(minutes: pd.Series, sess: pd.DataFrame, step: int) -> pd.DataFrame:
    """Open-anchored OHLC bars from minute prices (bar = minutes in (prev end, end]).

    Highs and lows come from minute prices, so they slightly understate the true
    range; good enough for ATR-based features. Bar ends match _session_bars.
    """
    idx = minutes.index
    since_open = np.asarray(idx.hour * 60 + idx.minute) - (9 * 60 + 30)
    close_key = {d: minute_key(c) for d, c in sess["close"].items()}
    ck = pd.Series(idx.date).map(close_key).values
    valid = ~pd.isna(ck) & (since_open >= 0)
    ck = np.where(valid, ck, -1).astype("int64")
    ik = minute_key(idx)
    valid &= ik <= ck
    bar_n = np.maximum(1, np.ceil(since_open / step)).astype("int64")
    open_key = ik - since_open
    end_key = np.minimum(open_key + bar_n * step, ck)
    df = pd.DataFrame({"end": end_key[valid], "p": minutes.values[valid]})
    g = df.groupby("end")["p"]
    out = pd.DataFrame({"open": g.first(), "high": g.max(), "low": g.min(), "close": g.last()})
    out.index = pd.to_datetime(out.index.values * 60, unit="s", utc=True).tz_convert("America/New_York")
    return out


def daily_closes(minutes: pd.Series, sess: pd.DataFrame) -> pd.Series:
    """Price at each session close (index = session close timestamp)."""
    idx = minutes.index
    close_key = {d: minute_key(c) for d, c in sess["close"].items()}
    ck = pd.Series(idx.date).map(close_key).values
    valid = ~pd.isna(ck)
    ik = minute_key(idx)
    at_close = valid & (ik == np.where(valid, ck, -1).astype("int64"))
    return minutes[at_close]


def build_signals(minutes: pd.Series, sess: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Signal state at every 15-minute close (index = bar-close timestamp, ET)."""
    c15 = _session_bars(minutes, sess, 15, anchor_open=False)
    c1h = _session_bars(minutes, sess, 60, anchor_open=True)

    def frame(c: pd.Series) -> pd.DataFrame:
        f = time_series_ma(c, cfg.tsf_fast)
        s = time_series_ma(c, cfg.tsf_slow)
        h = macd_hist(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        return pd.DataFrame({
            "close": c, "tsf_fast": f, "tsf_slow": s, "macd_hist": h,
            "rsi": rsi(c, cfg.rsi_period), "state": trend_state(c, f, s),
            "macd": np.sign(h).fillna(0).astype(int),
        })

    s15, s1h = frame(c15), frame(c1h)
    # 1-hour context used by the entry veto's features
    s1h["bbpos"], s1h["bbbw"] = bollinger(c1h)
    o1h = _session_ohlc(minutes, sess, 60)
    st_val, st_dir = supertrend(o1h["high"], o1h["low"], o1h["close"])
    s1h["st_val"], s1h["st_dir"] = st_val.reindex(s1h.index), st_dir.reindex(s1h.index)
    s1h = s1h.add_suffix("_1h")
    sig = pd.merge_asof(s15.reset_index(names="t"), s1h.reset_index(names="t_1h"),
                        left_on="t", right_on="t_1h", direction="backward")

    # daily trend state, known only once the session has closed
    dc = daily_closes(minutes, sess)
    sday = pd.DataFrame({"state_d": trend_state(dc, time_series_ma(dc, cfg.tsf_fast),
                                                time_series_ma(dc, cfg.tsf_slow))})
    sig = pd.merge_asof(sig, sday.reset_index(names="t_d"), left_on="t", right_on="t_d",
                        direction="backward").set_index("t")

    sig["macd_agrees"] = (sig["state"] != 0) & (sig["macd"] == sig["state"])
    sig["tf_agree"] = (sig["state"] != 0) & (sig["state_1h"] == sig["state"])
    rsi_aligned = (((sig["state"] == 1) & (sig["rsi_1h"] > cfg.rsi_high))
                   | ((sig["state"] == -1) & (sig["rsi_1h"] < cfg.rsi_low)))
    sig["tier1"] = sig["tf_agree"] & (sig["macd_1h"] == sig["state"]) & rsi_aligned
    return sig
