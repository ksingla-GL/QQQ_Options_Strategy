"""
Event-driven backtest of the QQQ weekly credit spread strategy (Spec v1).

Clock per session: exit checks every `check_every_min` from `first_check` to the
close, plus the expiry-day close; entries on 15-minute closes inside the entry
window. At each minute exits run before entries. Quotes at a minute are the last
NBBO at or before it, so decisions never see the future.

Accounting: cash moves on fills (credit in, buyback out, commissions out).
Mark-to-market equity = cash - cost to close open spreads at the mid.
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .chains import ChainStore
from .config import BacktestConfig, chain_path
from .market import ET, daily_gates, minute_key, sessions, underlying_minutes
from .signals import build_signals, daily_closes

log = logging.getLogger("bt.engine")

# Entry features for the walk-forward ML veto (veto.py), all known at the entry minute
# and scale-free because QQQ's price rose ~7x over the data. "_al" means signed so
# positive = in favour of the position (puts bullish, calls bearish).
FEATURES = [
    "side_put", "credit_pct", "credit_to_width", "width_pct", "otm_pct", "short_delta",
    "short_iv", "iv_minus_rv", "dte", "vix", "vix_ratio", "rv20", "mso", "weekday",
    "ret60m_al", "ret1d_al", "day_ret_al", "macd15_al", "macd1h_al", "rsi15", "rsi1h",
    "bbpos1h", "bbbw1h", "stdir1h_al", "conv_al", "tsf_gap1h_al", "dist_tsf15_al",
    "tfagree", "tier1", "open_same_side",
]


@dataclass
class Position:
    pid: int
    side: str                 # "put" or "call"
    right: str                # "P" or "C"
    policy: str
    expiry: date
    short_k: float
    long_k: float
    width: float
    contracts: int
    credit: float             # per share, after entry slippage
    mid_credit: float
    entry_time: pd.Timestamp
    entry_spot: float
    short_delta: float
    entry_comm: float
    tags: dict = field(default_factory=dict)
    feat: dict = field(default_factory=dict)
    exit_time: pd.Timestamp | None = None
    exit_spot: float = np.nan
    exit_cost: float = np.nan
    exit_reason: str = ""
    exit_comm: float = 0.0
    exit_mid: float = np.nan   # spread mid at the exit minute (intrinsic if no trusted quote)
    exit_trusted: bool = True  # short leg had a trusted quote at exit
    last_mark: float = np.nan  # last mid cost to close, per share

    @property
    def max_loss(self) -> float:
        return (self.width - self.credit) * 100 * self.contracts

    def intrinsic(self, spot: float) -> float:
        v = self.short_k - spot if self.right == "P" else spot - self.short_k
        return min(max(v, 0.0), self.width)

    def pnl(self) -> float:
        cost = self.exit_cost if self.exit_time is not None else self.last_mark
        if not np.isfinite(cost):
            cost = 0.0
        return (self.credit - cost) * 100 * self.contracts - self.entry_comm - self.exit_comm


def _day_time(d: date, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{d} {hhmm}", tz=ET)


class Market:
    """Config-independent data shared by every backtest over one period.

    Several Backtesters can share one Market and step through the same days
    together (see sweep.py), so each expiry file is loaded once per sweep.
    """

    def __init__(self, start: str, end: str, max_iv_error: float = 1.0, chain_dir: str = ""):
        chains = chain_path(chain_dir)
        self.store = ChainStore(chains, max_iv_error=max_iv_error)
        if not self.store.expiries:
            raise RuntimeError(f"No chain files in {chains}")
        self.start = pd.Timestamp(start).date()
        self.end = min(pd.Timestamp(end).date(), self._data_end())
        if self.end <= self.start:
            raise RuntimeError(f"No data in range {start}..{end} (data ends {self.end})")
        hist_start = (pd.Timestamp(self.start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        self.sess = sessions(hist_start, self.end.isoformat())
        self.days = [d for d in self.sess.index if self.start <= d <= self.end]
        self.minutes = underlying_minutes()
        self.u_keys = minute_key(self.minutes.index)
        self.u_vals = self.minutes.values.astype("float64")
        self.gates = daily_gates(self.start.isoformat(), self.end.isoformat())
        closes = {ts.date(): float(p) for ts, p in daily_closes(self.minutes, self.sess).items()}
        dates = list(self.sess.index)
        self.prev_close = {d: closes.get(dates[i - 1]) for i, d in enumerate(dates) if i > 0}
        self._signals = {}

    def _data_end(self) -> date:
        """Last expiry before the first gap of more than 21 days.

        Downloads run in date order, so a long gap marks the download frontier. A
        single missing weekly (a failed request, re-pulled later) is tolerated:
        entries targeting it are skipped as no_expiry.
        """
        ex = [e for e in self.store.expiries if e >= self.start - timedelta(days=14)]
        if not ex:
            return self.start
        gaps = [(b - a).days for a, b in zip(ex, ex[1:])]
        limit = max(21, 1.5 * float(np.median(gaps))) if gaps else 21   # monthly sets: ~45 days
        last = ex[0]
        for e in ex[1:]:
            if (e - last).days > limit:
                break
            last = e
        return last

    def signals_for(self, cfg: BacktestConfig) -> pd.DataFrame:
        key = (cfg.tsf_fast, cfg.tsf_slow, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal,
               cfg.rsi_period, cfg.rsi_high, cfg.rsi_low)
        if key not in self._signals:
            self._signals[key] = build_signals(self.minutes, self.sess, cfg)
        return self._signals[key]


class Backtester:
    def __init__(self, cfg: BacktestConfig, market: Market | None = None):
        self.cfg = cfg
        self.m = market or Market(cfg.start, cfg.end, cfg.max_iv_error, cfg.chain_dir)
        self.store, self.sess, self.gates = self.m.store, self.m.sess, self.m.gates
        self.start, self.end = self.m.start, self.m.end
        self.u_keys, self.u_vals = self.m.u_keys, self.m.u_vals
        self.signals = self.m.signals_for(cfg)
        self.sig_keys = minute_key(self.signals.index)
        self.entry_filter = None   # optional callable(features: dict, t) -> bool (keep)
        self._gates_today = None

        self.cash = cfg.base_equity
        self.peak = cfg.base_equity
        self.equity = cfg.base_equity
        self.halted_on: date | None = None
        self.open: list[Position] = []
        self.closed: list[Position] = []
        self.last_entry: dict[str, pd.Timestamp] = {}
        self.skips = Counter()
        self.daily = []
        self._pid = 0

    # ------------------------------------------------------------------ #
    def _spot(self, tkey: int) -> float:
        i = np.searchsorted(self.u_keys, tkey)
        return float(self.u_vals[i]) if i < len(self.u_keys) and self.u_keys[i] == tkey else np.nan

    def _signal(self, tkey: int):
        i = np.searchsorted(self.sig_keys, tkey)
        if i < len(self.sig_keys) and self.sig_keys[i] == tkey:
            return self.signals.iloc[i]
        return None

    def _commission(self, premiums: list[float], contracts: int, opening: bool = False) -> float:
        if not premiums:
            return 0.0
        cfg = self.cfg
        fees = cfg.fees_per_contract * contracts * len(premiums)
        if cfg.commission_model == "flat":
            return cfg.commission_flat * contracts * len(premiums) + fees
        if cfg.commission_model == "open_only":
            per_leg = min(cfg.commission_open * contracts, cfg.commission_leg_cap) if opening else 0.0
            return per_leg * len(premiums) + fees
        total, legs = 0.0, 0
        for p in premiums:
            p = abs(p) if np.isfinite(p) else 0.0
            rate = (cfg.commission_low if p < 0.05 else
                    cfg.commission_mid if p < 0.10 else cfg.commission_high)
            total += rate * contracts
            legs += 1
        return max(cfg.min_commission_per_order, total) + cfg.fees_per_contract * contracts * legs

    def _sizing_base(self) -> float:
        return self.equity if self.cfg.sizing_base == "equity" else self.cfg.base_equity

    # ------------------------------------------------------------------ #
    def run(self):
        days = self.m.days
        log.info(f"Backtest {days[0]} to {days[-1]}: {len(days)} sessions, "
                 f"{len(self.store.expiries)} expiries on disk")
        for i, d in enumerate(days):
            self.run_day(d)
            if (i + 1) % 250 == 0:
                log.info(f"  {d}: equity ${self.equity:,.0f}, trades {len(self.closed)}")
        return self

    def run_day(self, d: date):
        cfg = self.cfg
        close_t = self.sess.at[d, "close"]
        g = self.gates.loc[d] if d in self.gates.index else None
        self._gates_today = g
        if cfg.monthly_platform_fee and (self.daily == [] or self.daily[-1]["date"].month != d.month):
            self.cash -= cfg.monthly_platform_fee
        block = []
        if g is not None:
            if cfg.vol_gate > 0 and g["rv20"] > cfg.vol_gate:
                block.append("gate_vol")
            if cfg.term_gate and g["vix_prev"] > g["vix3m_prev"]:
                block.append("gate_term")

        checks = set(pd.date_range(_day_time(d, cfg.first_check), close_t,
                                   freq=f"{cfg.check_every_min}min"))
        last_entry_t = min(_day_time(d, cfg.entry_end), close_t - pd.Timedelta(minutes=15))
        entries = set(pd.date_range(_day_time(d, cfg.entry_start), last_entry_t, freq="15min"))
        forced = min(_day_time(d, cfg.expiry_close_time), close_t - pd.Timedelta(minutes=30))
        cutoff = _day_time(d, cfg.friday_cutoff) if (cfg.friday_cutoff and d.weekday() == 4) else None

        self._settle_missed(d)
        for t in sorted(checks | entries | {forced}):
            tk = minute_key(t)
            spot = self._spot(tk)
            if not np.isfinite(spot):
                continue
            if t in checks or t == forced:
                self._exits(d, t, tk, spot, forced)
                self._mark_equity(d)
            if t in entries:
                day_block = block + (["gate_friday"] if cutoff is not None and t >= cutoff else [])
                self._entries(d, t, tk, spot, day_block)

        self.daily.append({
            "date": d, "equity": self.equity, "cash": self.cash,
            "open": len(self.open), "open_risk": sum(p.max_loss for p in self.open),
            "drawdown_pct": (self.equity / self.peak - 1) * 100,
            "halted": self.halted_on is not None,
            "rv20": None if g is None else g["rv20"],
            "vix_prev": None if g is None else g["vix_prev"],
            "vix3m_prev": None if g is None else g["vix3m_prev"],
        })

    # ------------------------------------------------------------------ #
    # Exits
    # ------------------------------------------------------------------ #
    def _trusted(self, bid: float, ask: float) -> bool:
        cfg = self.cfg
        return (np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0
                and ask - bid <= max(cfg.quote_max_width_abs, cfg.quote_max_width_rel * bid))

    def _exits(self, d: date, t: pd.Timestamp, tk: int, spot: float, forced: pd.Timestamp):
        """Stop / profit-take / expiry-day close.

        The spread is marked from trusted quotes only. With no trusted quote on the
        short leg (typically no bid and a placeholder ask on a near-worthless
        option) it is valued at intrinsic, so a placeholder can neither trigger a
        stop nor set a fill price. The long leg counts at its mid when trusted,
        else at its bid (0 if none), which never understates the cost to close.
        """
        cfg = self.cfg
        for pos in list(self.open):
            grid = self.store.get(pos.expiry)
            sb, sa = grid.quote(pos.right, pos.short_k, tk) if grid else (np.nan, np.nan)
            lb, la = grid.quote(pos.right, pos.long_k, tk) if grid else (np.nan, np.nan)
            intrinsic = pos.intrinsic(spot)
            short_ok = self._trusted(sb, sa)
            lbid = lb if np.isfinite(lb) and lb > 0 else 0.0
            long_val = (lb + la) / 2 if self._trusted(lb, la) else lbid
            if short_ok:
                mid = (sb + sa) / 2 - long_val
                mark = max(mid, intrinsic)
                pos.last_mark = min(max(mid, 0.0), pos.width)
            else:
                mid = mark = intrinsic
                pos.last_mark = intrinsic

            reason = ""
            if pos.expiry <= d and t >= forced:
                reason = "expiry_close"
            elif (cfg.exit_at_dte > 0 and (pos.expiry - d).days <= cfg.exit_at_dte
                  and t.hour * 60 + t.minute >= 600):   # from 10:00, after the wide opening quotes
                reason = "time_exit"
            elif mark >= cfg.stop_mult * pos.credit:
                reason = "stop"
            elif (short_ok and cfg.profit_take > 0 and mark <= (1 - cfg.profit_take) * pos.credit
                  and t.hour * 60 + t.minute >= 600):   # patient exit: not into the wide opening quotes
                reason = "profit_take"
            if not reason:
                continue

            if short_ok:
                natural = sa - lbid
                if reason == "stop":
                    cost = natural if cfg.exit_fill == "natural" else mid
                else:
                    cost = mid + cfg.patient_exit_frac * max(natural - mid, 0.0)
                legs = [sa] + ([lbid] if lbid > 0 else [])   # a no-bid long leg is left to expire
            elif intrinsic > 0:
                cost, legs = intrinsic, [intrinsic]
            elif (reason == "expiry_close" and cfg.expire_worthless
                  and abs(spot - pos.short_k) / spot * 100 >= cfg.expire_min_otm_pct):
                cost, legs = 0.0, []    # no bid and well out of the money: let both legs expire
            else:
                # out of the money with no real bid: buy the short back for a nickel at most
                ask = sa if np.isfinite(sa) and sa > 0 else cfg.worthless_buyback_cap
                cost = min(ask, cfg.worthless_buyback_cap)
                legs = [cost]
            pos.exit_mid, pos.exit_trusted = min(max(mid, 0.0), pos.width), short_ok
            self._close(pos, t, spot, min(max(cost, 0.0), pos.width), reason, legs)

    def _close(self, pos: Position, t, spot: float, cost: float, reason: str, legs: list):
        pos.exit_time, pos.exit_spot, pos.exit_cost, pos.exit_reason = t, spot, cost, reason
        pos.exit_comm = self._commission(legs, pos.contracts)
        pos.last_mark = cost
        self.cash -= cost * 100 * pos.contracts + pos.exit_comm
        self.open.remove(pos)
        self.closed.append(pos)

    def _settle_missed(self, d: date):
        """Safety net: a position past its expiry without a close settles at intrinsic."""
        for pos in list(self.open):
            if pos.expiry < d:
                tk = minute_key(_day_time(pos.expiry, "16:00"))
                i = np.searchsorted(self.u_keys, tk, side="right") - 1
                spot = float(self.u_vals[i]) if i >= 0 else pos.entry_spot
                cost = pos.intrinsic(spot)
                pos.exit_mid, pos.exit_trusted = cost, False   # no quotes: settled at intrinsic
                self._close(pos, _day_time(pos.expiry, "16:00"), spot, cost, "expired_missed", [cost])

    def _mark_equity(self, d: date):
        cfg = self.cfg
        liab = sum((p.last_mark if np.isfinite(p.last_mark) else 0.0) * 100 * p.contracts
                   for p in self.open)
        self.equity = self.cash - liab
        self.peak = max(self.peak, self.equity)
        dd = (self.equity / self.peak - 1) * 100
        if self.halted_on is None and cfg.dd_halt_pct > 0 and dd <= -cfg.dd_halt_pct:
            self.halted_on = d
            log.warning(f"{d}: drawdown halt, equity ${self.equity:,.0f} is {dd:.1f}% below peak")

    # ------------------------------------------------------------------ #
    # Entries
    # ------------------------------------------------------------------ #
    def _desired_sides(self, d: date, t: pd.Timestamp, tk: int) -> list[tuple[str, dict]]:
        cfg = self.cfg
        if cfg.entry_policy == "trend":
            s = self._signal(tk)
            if s is None or not bool(s["macd_agrees"]):
                return []
            tags = {"state_1h": int(s["state_1h"]) if np.isfinite(s["state_1h"]) else 0,
                    "tf_agree": bool(s["tf_agree"]), "tier1": bool(s["tier1"])}
            return [("put", tags)] if s["state"] == 1 else [("call", tags)]
        if d.weekday() != cfg.weekly_entry_dow or t.strftime("%H:%M") != cfg.weekly_entry_time:
            return []
        tags = {"state_1h": 0, "tf_agree": False, "tier1": False}
        if cfg.entry_policy == "put_weekly":
            return [("put", tags)]
        if cfg.entry_policy == "condor_weekly":
            return [("put", tags), ("call", dict(tags))]
        raise ValueError(f"Unknown entry_policy {cfg.entry_policy}")

    def _features(self, d: date, t: pd.Timestamp, tk: int, spot: float, side: str,
                  short: dict, credit: float, width: float, expiry: date) -> dict:
        """Entry features at the decision minute (see FEATURES)."""
        sg = 1.0 if side == "put" else -1.0
        nan = np.nan
        s = self._signal(tk)
        g = self._gates_today
        i = np.searchsorted(self.u_keys, tk)

        def back(n):  # price n trading minutes earlier
            return self.u_vals[i - n] if i - n >= 0 else nan

        def val(key):
            if s is None:
                return nan
            v = s[key]
            return float(v) if v is not None and np.isfinite(v) else nan

        rv = float(g["rv20"]) if g is not None else nan
        vix = float(g["vix_prev"]) if g is not None else nan
        v3m = float(g["vix3m_prev"]) if g is not None else nan
        iv = short.get("iv", nan) * 100
        prev = self.m.prev_close.get(d)
        c15, c1h = val("close"), val("close_1h")
        return {
            "side_put": 1.0 if side == "put" else 0.0,
            "credit_pct": credit / spot * 100,
            "credit_to_width": credit / width if width else nan,
            "width_pct": width / spot * 100,
            "otm_pct": abs(spot - short["strike"]) / spot * 100,
            "short_delta": abs(short["delta"]),
            "short_iv": iv,
            "iv_minus_rv": iv - rv,
            "dte": float((expiry - d).days),
            "vix": vix,
            "vix_ratio": vix / v3m if v3m else nan,
            "rv20": rv,
            "mso": (t - _day_time(d, "09:30")).total_seconds() / 60,
            "weekday": float(d.weekday()),
            "ret60m_al": (spot / back(60) - 1) * 100 * sg,
            "ret1d_al": (spot / back(390) - 1) * 100 * sg,
            "day_ret_al": (spot / prev - 1) * 100 * sg if prev else nan,
            "macd15_al": val("macd_hist") / c15 * 1000 * sg,
            "macd1h_al": val("macd_hist_1h") / c1h * 1000 * sg,
            "rsi15": val("rsi"),
            "rsi1h": val("rsi_1h"),
            "bbpos1h": val("bbpos_1h"),
            "bbbw1h": val("bbbw_1h") * 100,
            "stdir1h_al": val("st_dir_1h") * sg,
            "conv_al": (val("state") + val("state_1h") + val("state_d")) * sg,
            "tsf_gap1h_al": (val("tsf_fast_1h") - val("tsf_slow_1h")) / c1h * 100 * sg,
            "dist_tsf15_al": (c15 - val("tsf_slow")) / c15 * 100 * sg,
            "tfagree": float(bool(s["tf_agree"])) if s is not None else 0.0,
            "tier1": float(bool(s["tier1"])) if s is not None else 0.0,
            "open_same_side": float(sum(p.side == side for p in self.open)),
        }

    def _halted(self, d: date) -> bool:
        if self.halted_on is None:
            return False
        reset = self.cfg.dd_halt_reset_days
        if reset > 0 and (d - self.halted_on).days >= reset:
            self.halted_on, self.peak = None, self.equity
            return False
        return True

    def _entries(self, d: date, t: pd.Timestamp, tk: int, spot: float, block: list[str]):
        cfg = self.cfg
        wanted = self._desired_sides(d, t, tk)
        if not wanted:
            return
        if self._halted(d):
            self.skips["dd_halt"] += len(wanted)
            return
        if block:
            for reason in block:
                self.skips[reason] += len(wanted)
            return

        if cfg.regime_tilt != "off" and self._gates_today is not None:
            reg = self._gates_today.get("regime_prev")
            bull = reg in ("A", "B")
            if cfg.regime_tilt == "with_trend":
                allowed = {"put"} if bull else ({"call"} if reg in ("C", "D") else {"put", "call"})
            elif cfg.regime_tilt == "bull_puts_only":
                allowed = {"put"} if bull else {"put", "call"}
            else:
                raise ValueError(f"Unknown regime_tilt {cfg.regime_tilt}")
            self.skips["regime"] += sum(side not in allowed for side, _ in wanted)
            wanted = [w for w in wanted if w[0] in allowed]
            if not wanted:
                return

        dd = (self.equity / self.peak - 1) * 100
        side_cap = cfg.max_open_per_side
        if cfg.dd_reduce_pct > 0 and dd <= -cfg.dd_reduce_pct:
            side_cap = max(1, side_cap // 2)
        base = self._sizing_base()

        for side, tags in wanted:
            if sum(p.side == side for p in self.open) >= side_cap:
                self.skips["cap_side"] += 1
                continue
            if len(self.open) >= cfg.max_open_total:
                self.skips["cap_total"] += 1
                continue
            last = self.last_entry.get(side)
            if last is not None and t - last < pd.Timedelta(minutes=cfg.side_cooldown_min):
                self.skips["cooldown"] += 1
                continue
            expiry = self.store.pick_expiry(d, cfg.target_dte, cfg.min_dte, cfg.max_dte)
            grid = self.store.get(expiry) if expiry else None
            if grid is None:
                self.skips["no_expiry"] += 1
                continue

            right = "P" if side == "put" else "C"
            short = grid.select_short(right, tk, cfg.short_delta, spot,
                                      cfg.quote_max_width_abs, cfg.quote_max_width_rel)
            if short is not None and cfg.sr_filter != "off":
                s = self._signal(tk)
                st_dir = float(s["st_dir_1h"]) if s is not None and np.isfinite(s["st_dir_1h"]) else 0.0
                st_val = float(s["st_val_1h"]) if s is not None and np.isfinite(s["st_val_1h"]) else np.nan
                level_here = (side == "put" and st_dir == 1) or (side == "call" and st_dir == -1)
                inside = level_here and ((side == "put" and short["strike"] > st_val)
                                         or (side == "call" and short["strike"] < st_val))
                if inside:
                    if cfg.sr_filter == "skip":
                        self.skips["sr_inside_level"] += 1
                        continue
                    short = grid.select_beyond(right, tk, st_val, spot,
                                               cfg.quote_max_width_abs, cfg.quote_max_width_rel)
                    self.skips["sr_moved_strike"] += 1   # counted as an adjustment, not a skip
            width_target = spot * cfg.width_pct / 100 if cfg.width_mode == "pct" else cfg.width_dollar
            long = grid.select_long(right, tk, short["strike"], width_target) if short else None
            if short is None or long is None:
                self.skips["no_strikes"] += 1
                continue

            short_mid, long_mid = (short["bid"] + short["ask"]) / 2, (long["bid"] + long["ask"]) / 2
            mid_credit = short_mid - long_mid
            credit = mid_credit - cfg.entry_slippage
            width = abs(short["strike"] - long["strike"])
            if cfg.credit_mode == "pct":
                min_credit = spot * cfg.min_credit_pct / 100
            elif cfg.credit_mode == "width":
                min_credit = width * cfg.min_credit_width_pct / 100
            else:
                min_credit = cfg.min_credit_dollar
            if credit < min_credit:
                self.skips["low_credit"] += 1
                continue
            loss_per = (width - credit) * 100
            if loss_per <= 0:
                self.skips["bad_quote"] += 1
                continue

            n = math.floor(cfg.risk_per_trade_pct / 100 * base / loss_per)
            if n == 0 and loss_per <= cfg.max_risk_per_trade_pct / 100 * base:
                n = 1
            room = cfg.max_open_risk_pct / 100 * base - sum(p.max_loss for p in self.open)
            n = min(n, math.floor(room / loss_per)) if room > 0 else 0
            if n <= 0:
                self.skips["risk_limit"] += 1
                continue

            feat = self._features(d, t, tk, spot, side, short, credit, width, expiry)
            if self.entry_filter is not None and not self.entry_filter(feat, t):
                self.skips["veto"] += 1
                continue

            self._pid += 1
            comm = self._commission([short_mid, long_mid], n, opening=True)
            pos = Position(
                pid=self._pid, side=side, right=right, policy=cfg.entry_policy, expiry=expiry,
                short_k=short["strike"], long_k=long["strike"], width=width, contracts=n,
                credit=credit, mid_credit=mid_credit, entry_time=t, entry_spot=spot,
                short_delta=short["delta"], entry_comm=comm, tags=tags, feat=feat,
                last_mark=mid_credit)
            self.cash += credit * 100 * n - comm
            self.open.append(pos)
            self.last_entry[side] = t

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    def trades(self) -> pd.DataFrame:
        rows = []
        for p in self.closed + self.open:
            rows.append({
                "pid": p.pid, "status": "closed" if p.exit_time is not None else "open",
                "policy": p.policy, "side": p.side, "expiry": p.expiry,
                "entry_time": p.entry_time, "dte": (p.expiry - p.entry_time.date()).days,
                "entry_spot": p.entry_spot, "short_k": p.short_k, "long_k": p.long_k,
                "width": p.width, "short_delta": p.short_delta, "contracts": p.contracts,
                "mid_credit": round(p.mid_credit, 4), "credit": round(p.credit, 4),
                "max_loss": round(p.max_loss, 2),
                "state_1h": p.tags.get("state_1h"), "tf_agree": p.tags.get("tf_agree"),
                "tier1": p.tags.get("tier1"),
                "exit_time": p.exit_time, "exit_reason": p.exit_reason or "open",
                "exit_spot": p.exit_spot, "exit_cost": p.exit_cost,
                "exit_mid": p.exit_mid, "exit_trusted": p.exit_trusted,
                "entry_comm": round(p.entry_comm, 2), "exit_comm": round(p.exit_comm, 2),
                "commissions": round(p.entry_comm + p.exit_comm, 2),
                "pnl": round(p.pnl(), 2),
                **{f"f_{k}": v for k, v in p.feat.items()},
            })
        return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True) if rows else pd.DataFrame()

    def daily_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.daily).set_index("date")
