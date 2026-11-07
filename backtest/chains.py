"""
Option chain access for the backtest.

ExpiryGrid turns one expiration's Parquet file into dense numpy grids
[minute x strike] per right (bid, ask, delta, Greeks-usable flag), so every
quote lookup is an O(1) array read. ChainStore loads grids lazily and keeps
the most recently used ones in memory.

Only standard strikes (multiples of $0.50) are kept: early years include
adjusted legacy series (e.g. 99.63) that a live trader would not use.
"""

import bisect
from collections import OrderedDict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .market import minute_key

COLS = ["timestamp", "strike", "right", "bid", "ask", "delta", "implied_vol", "iv_error"]


class ExpiryGrid:
    def __init__(self, path: Path, expiry: date, max_iv_error: float = 1.0):
        self.expiry = expiry
        df = pd.read_parquet(path, columns=COLS)
        k2 = df["strike"].astype("float64") * 2
        df = df[(k2 - k2.round()).abs() < 1e-3]
        tkeys = minute_key(df["timestamp"])
        self.tkeys = np.unique(tkeys)
        strikes = np.round(df["strike"].astype("float64").values, 2)
        self.strikes = np.unique(strikes)
        ti = np.searchsorted(self.tkeys, tkeys)
        ki = np.searchsorted(self.strikes, strikes)
        shape = (len(self.tkeys), len(self.strikes))
        right = df["right"].astype(str).values
        ok = ((df["implied_vol"].values > 0)
              & (np.abs(df["iv_error"].values) <= max_iv_error))
        self.g = {}
        for r in ("P", "C"):
            m = right == r
            grids = {}
            for name, vals in (("bid", df["bid"].values), ("ask", df["ask"].values),
                               ("delta", df["delta"].values), ("iv", df["implied_vol"].values)):
                a = np.full(shape, np.nan, dtype=np.float32)
                a[ti[m], ki[m]] = vals[m]
                grids[name] = a
            okg = np.zeros(shape, dtype=bool)
            okg[ti[m], ki[m]] = ok[m]
            grids["ok"] = okg
            self.g[r] = grids

    def row(self, tkey: int) -> int:
        i = np.searchsorted(self.tkeys, tkey)
        return int(i) if i < len(self.tkeys) and self.tkeys[i] == tkey else -1

    def col(self, strike: float) -> int:
        i = np.searchsorted(self.strikes, round(strike, 2))
        return int(i) if i < len(self.strikes) and abs(self.strikes[i] - strike) < 1e-6 else -1

    def quote(self, right: str, strike: float, tkey: int):
        """(bid, ask) for one contract at one minute; NaNs when missing."""
        i, j = self.row(tkey), self.col(strike)
        if i < 0 or j < 0:
            return np.nan, np.nan
        g = self.g[right]
        return float(g["bid"][i, j]), float(g["ask"][i, j])

    def select_short(self, right: str, tkey: int, target_delta: float, spot: float,
                     max_w_abs: float = 0.10, max_w_rel: float = 1.0):
        """Out-of-the-money strike whose |delta| is nearest the target, with a trusted quote."""
        i = self.row(tkey)
        if i < 0:
            return None
        g = self.g[right]
        bid, ask, dlt, ok = g["bid"][i], g["ask"][i], g["delta"][i], g["ok"][i]
        otm = self.strikes < spot if right == "P" else self.strikes > spot
        tight = (ask - bid) <= np.maximum(max_w_abs, max_w_rel * bid)
        valid = ok & otm & (bid > 0) & (ask > 0) & np.isfinite(dlt) & tight
        if not valid.any():
            return None
        dist = np.where(valid, np.abs(np.abs(dlt) - target_delta), np.inf)
        j = int(np.argmin(dist))
        return {"strike": float(self.strikes[j]), "bid": float(bid[j]), "ask": float(ask[j]),
                "delta": float(dlt[j]), "iv": float(g["iv"][i, j])}

    def select_beyond(self, right: str, tkey: int, level: float, spot: float,
                      max_w_abs: float = 0.10, max_w_rel: float = 1.0):
        """Nearest strike on the far side of a support (puts) / resistance (calls) level, trusted quote."""
        i = self.row(tkey)
        if i < 0 or not np.isfinite(level):
            return None
        g = self.g[right]
        bid, ask, dlt, ok = g["bid"][i], g["ask"][i], g["delta"][i], g["ok"][i]
        side = (self.strikes <= level) & (self.strikes < spot) if right == "P" else \
               (self.strikes >= level) & (self.strikes > spot)
        tight = (ask - bid) <= np.maximum(max_w_abs, max_w_rel * bid)
        valid = side & ok & (bid > 0) & (ask > 0) & np.isfinite(dlt) & tight
        if not valid.any():
            return None
        j = int(np.max(np.where(valid)[0])) if right == "P" else int(np.min(np.where(valid)[0]))
        return {"strike": float(self.strikes[j]), "bid": float(bid[j]), "ask": float(ask[j]),
                "delta": float(dlt[j]), "iv": float(g["iv"][i, j])}

    def select_long(self, right: str, tkey: int, short_strike: float, width: float):
        """Strike further out of the money, nearest short -/+ width, with an ask to buy at."""
        i = self.row(tkey)
        if i < 0:
            return None
        g = self.g[right]
        ask, bid = g["ask"][i], g["bid"][i]
        further = self.strikes < short_strike if right == "P" else self.strikes > short_strike
        valid = further & np.isfinite(ask) & (ask > 0) & np.isfinite(bid)
        if not valid.any():
            return None
        target = short_strike - width if right == "P" else short_strike + width
        dist = np.where(valid, np.abs(self.strikes - target), np.inf)
        j = int(np.argmin(dist))
        return {"strike": float(self.strikes[j]), "bid": float(bid[j]), "ask": float(ask[j])}


class ChainStore:
    def __init__(self, chain_dir: Path, max_cached: int = 24, max_iv_error: float = 1.0):
        self.paths = {}
        for p in chain_dir.glob("*.parquet"):
            self.paths[pd.Timestamp(p.stem).date()] = p
        self.expiries = sorted(self.paths)
        self.max_cached = max_cached
        self.max_iv_error = max_iv_error
        self._cache: OrderedDict[date, ExpiryGrid] = OrderedDict()
        self.loads = 0

    def get(self, expiry: date) -> ExpiryGrid | None:
        if expiry in self._cache:
            self._cache.move_to_end(expiry)
            return self._cache[expiry]
        path = self.paths.get(expiry)
        if path is None:
            return None
        grid = ExpiryGrid(path, expiry, self.max_iv_error)
        self.loads += 1
        self._cache[expiry] = grid
        if len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return grid

    def pick_expiry(self, today: date, target: int, lo: int, hi: int) -> date | None:
        """Expiry nearest `target` calendar days out, within [lo, hi]; ties go to the nearer one."""
        best, best_key = None, None
        i = bisect.bisect_left(self.expiries, today + timedelta(days=lo))
        for e in self.expiries[i:]:
            dte = (e - today).days
            if dte > hi:
                break
            key = (abs(dte - target), dte)
            if best_key is None or key < best_key:
                best, best_key = e, key
        return best
