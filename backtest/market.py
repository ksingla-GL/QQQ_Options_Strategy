"""
Market data used around the option chains:

* sessions()          NYSE trading days with open/close times (early closes included)
* daily_gates()       prior-day inputs for the entry gates: NDX 20-day realized vol, VIX, VIX3M
* underlying_minutes() QQQ price every minute, rebuilt from the chain files' underlying_price

All times are US/Eastern. Minute keys are integer minutes since the Unix epoch (UTC),
used for fast exact lookups across modules.
"""

import json
import logging

import numpy as np
import pandas as pd

from .config import CACHE_DIR, CHAIN_DIR

ET = "America/New_York"
log = logging.getLogger("bt.market")


def minute_key(ts) -> np.ndarray | int:
    """Minutes since epoch (UTC) for a Timestamp, DatetimeIndex or datetime Series."""
    if isinstance(ts, pd.Timestamp):
        return ts.value // 60_000_000_000
    idx = pd.DatetimeIndex(ts)
    return idx.as_unit("ns").asi8 // 60_000_000_000


# ---------------------------------------------------------------------- #
# Calendar
# ---------------------------------------------------------------------- #
def sessions(start: str, end: str) -> pd.DataFrame:
    """NYSE sessions between start and end: index = date, columns open/close (ET)."""
    import pandas_market_calendars as mcal
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    out = pd.DataFrame({
        "open": sched["market_open"].dt.tz_convert(ET),
        "close": sched["market_close"].dt.tz_convert(ET),
    })
    out.index = pd.DatetimeIndex(out.index).date
    out["early_close"] = out["close"].dt.hour < 16
    return out


# ---------------------------------------------------------------------- #
# Daily gate inputs
# ---------------------------------------------------------------------- #
def _download_daily(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    cols = {}
    for name, ticker in (("ndx", "^NDX"), ("vix", "^VIX"), ("vix3m", "^VIX3M")):
        df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                         progress=False, threads=False)
        if df is None or len(df) == 0:
            raise RuntimeError(f"Yahoo returned no data for {ticker}")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        cols[name] = close
    return pd.DataFrame(cols).dropna(subset=["ndx"])


def daily_gates(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Gate inputs known before each session opens (index = session date).

    rv20   : NDX realized vol over the 20 closes before the date, as the NDX
             trader computes it (std of 19 daily returns x sqrt(252) x 100)
    vix_prev, vix3m_prev : prior-day closes
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "daily_gates_raw.parquet"
    pad_start = (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")  # MA250 warm-up
    pad_end = (pd.Timestamp(end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    raw = None
    if path.exists() and not refresh:
        raw = pd.read_parquet(path)
        # the first trading day can fall a few days after pad_start (holidays, weekends)
        if (raw.index.min() > pd.Timestamp(pad_start) + pd.Timedelta(days=10)
                or raw.index.max() < pd.Timestamp(end) - pd.Timedelta(days=5)):
            raw = None
    if raw is None:
        log.info("Downloading NDX / VIX / VIX3M daily closes from Yahoo...")
        raw = _download_daily(pad_start, pad_end)
        raw.to_parquet(path)

    ndx = raw["ndx"]
    rv = ndx.pct_change().rolling(19).std() * np.sqrt(252) * 100   # returns from the last 20 closes
    # Regime as the NDX trader defines it: A strong up, B pullback in uptrend,
    # C strong down, D bounce in downtrend, from close vs MA20 / MA250.
    ma20, ma250 = ndx.rolling(20).mean(), ndx.rolling(250).mean()
    bull, above20 = ndx > ma250, ndx > ma20
    regime = pd.Series(np.where(bull, np.where(above20, "A", "B"), np.where(above20, "D", "C")),
                       index=ndx.index).where(ma250.notna())
    out = pd.DataFrame({
        "rv20": rv.shift(1),               # through the prior close
        "vix_prev": raw["vix"].shift(1),
        "vix3m_prev": raw["vix3m"].shift(1),
        "regime_prev": regime.shift(1),
    })
    out.index = pd.DatetimeIndex(out.index).date
    return out


# ---------------------------------------------------------------------- #
# QQQ minute series from the chain files
# ---------------------------------------------------------------------- #
def underlying_minutes(refresh: bool = False) -> pd.Series:
    """QQQ price per minute (index = ET timestamps), merged across all chain files.

    Every file carries the underlying price at each minute; files overlap and
    agree, so the union gives a continuous series. Cached and rebuilt only when
    the set of chain files changes.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache, manifest = CACHE_DIR / "qqq_1m.parquet", CACHE_DIR / "qqq_1m_files.json"
    files = sorted(p.name for p in CHAIN_DIR.glob("*.parquet"))
    if cache.exists() and manifest.exists() and not refresh:
        if json.loads(manifest.read_text()) == files:
            s = pd.read_parquet(cache)["price"]
            return s

    log.info(f"Rebuilding QQQ minute series from {len(files)} chain files...")
    parts = []
    for name in files:
        u = pd.read_parquet(CHAIN_DIR / name, columns=["timestamp", "underlying_price"])
        parts.append(u.drop_duplicates("timestamp"))
    u = pd.concat(parts).drop_duplicates("timestamp").sort_values("timestamp")
    s = pd.Series(u["underlying_price"].astype("float64").values,
                  index=pd.DatetimeIndex(u["timestamp"]).tz_convert(ET), name="price")
    s = s[s > 0]
    s.to_frame().to_parquet(cache)
    manifest.write_text(json.dumps(files))
    return s
