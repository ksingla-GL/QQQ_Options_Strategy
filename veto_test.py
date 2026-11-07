"""
Try the walk-forward ML entry veto on top of a base config.

Usage
-----
  python veto_test.py --name veto_sl3 --set stop_mult=3
  python veto_test.py --name veto_sl3 --set stop_mult=3 --modes reg:0,reg:0.02,clf:0.8,clf:0.7

Pass 1 runs the base strategy; its closed trades (with entry features) are the
training labels. Pass 2 runs every veto variant over the same days, each trained
walk-forward on those base trades. The veto only switches on once 150 base
trades have closed, so results are also shown for that active window alone.
"""

import argparse
import logging
import time

import pandas as pd

from backtest.config import RESULTS_DIR, BacktestConfig
from backtest.engine import Backtester, Market
from backtest.report import summarize, write_outputs
from backtest.veto import WalkForwardVeto


def window_stats(bt: Backtester, start) -> dict:
    """Results for trades entered on/after `start` (the veto-active window)."""
    daily = bt.daily_frame()["equity"]
    daily.index = pd.to_datetime(daily.index)
    start = pd.Timestamp(start.date())
    before = daily[daily.index < start]
    eq0 = before.iloc[-1] if len(before) else bt.cfg.base_equity
    w = daily[daily.index >= start]
    tr = bt.trades()
    tr = tr[(tr["status"] == "closed")
            & (pd.to_datetime(tr["entry_time"], utc=True).dt.tz_convert("America/New_York").dt.date >= start.date())]
    return {"win_trades": len(tr), "win_win%": round(float((tr["pnl"] > 0).mean() * 100), 1) if len(tr) else None,
            "win_net_$": round(float(w.iloc[-1] - eq0), 2) if len(w) else 0.0,
            "win_maxDD_$": round(float((w - w.cummax()).min()), 2) if len(w) else 0.0}


def main():
    ap = argparse.ArgumentParser(description="Walk-forward ML entry veto test")
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--modes", default="reg:0,reg:0.02,clf:0.8,clf:0.7",
                    help="reg:<min predicted return on risk> or clf:<keep fraction>")
    ap.add_argument("--retrain", type=int, default=40, help="sessions between refits")
    ap.add_argument("--min-train", type=int, default=150)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    common = dict(kv.split("=", 1) for kv in args.set)
    if args.start:
        common["start"] = args.start
    if args.end:
        common["end"] = args.end
    cfg = BacktestConfig().update(common)

    t0 = time.time()
    market = Market(cfg.start, cfg.end, cfg.max_iv_error, cfg.chain_dir)
    logging.info(f"Pass 1: base strategy {market.days[0]} to {market.days[-1]}")
    base = Backtester(cfg, market)
    for d in market.days:
        base.run_day(d)
    base_trades = base.trades()

    variants = []
    for spec in args.modes.split(","):
        mode, val = spec.split(":")
        kw = {"ev_min": float(val)} if mode == "reg" else {"keep_frac": float(val)}
        veto = WalkForwardVeto(base_trades, market.days, mode=mode, retrain_sessions=args.retrain,
                               min_train=args.min_train, **kw)
        bt = Backtester(cfg, market)
        bt.entry_filter = veto
        variants.append((spec, bt, veto))
    logging.info(f"Pass 2: {len(variants)} veto variants")
    for d in market.days:
        for _, bt, _ in variants:
            bt.run_day(d)

    starts = [v.active_from for _, _, v in variants if v.active_from is not None]
    if not starts:
        print(f"\nThe veto never switched on: only {int((base_trades['status'] == 'closed').sum())} "
              f"base trades closed (needs {args.min_train}).")
        return
    active = min(starts)

    rows = []
    for name, bt, veto in [("base", base, None)] + variants:
        s = summarize(bt)
        write_outputs(bt, s, f"{args.name}/{name.replace(':', '_')}")
        row = {"variant": name, "trades": s["trades"], "win%": s["win_pct"], "net_$": s["net_pnl"],
               "maxDD%": s["max_drawdown_pct_of_base"], "PF": s["profit_factor"],
               "worst_$": s["worst_trade"], "vetoed": veto.skipped if veto else 0,
               "veto_seen": veto.seen if veto else 0}
        row.update(window_stats(bt, active))
        for y, yv in s["by_year"].items():
            row[y] = yv["pnl"]
        rows.append(row)

    table = pd.DataFrame(rows)
    out = RESULTS_DIR / args.name
    table.to_csv(out / "comparison.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(f"\nVeto active from {active.date()} (first segment with >= {args.min_train} closed base trades)")
        print("win_* columns = trades entered from that date on\n")
        print(table.to_string(index=False))
    reg = next((v for n, _, v in variants if n.startswith("reg")), None)
    if reg:
        print("\nTop features (latest regression model):", reg.importances())
    print(f"\nSaved to {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
