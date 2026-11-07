"""
Compare variants of one setting over the same period, in one pass.

Usage
-----
  python sweep.py --name stops --param stop_mult --values 1.5,2,2.5,3,4,5,100
  python sweep.py --name deltas --param short_delta --values 0.05,0.1,0.15,0.2 --set stop_mult=3
  python sweep.py --name mixed --set stop_mult=3 \
      --variants "trend:;puts:entry_policy=put_weekly;condor:entry_policy=condor_weekly"

All variants share one Market and step through the same days together, so each
expiry file is loaded once. Each variant is saved to
QQQ_OPTIONS_HOME/results/<name>/<param>=<value>/ (param is "variant" with
--variants) and a comparison table is printed (and saved as comparison.csv).
"""

import argparse
import logging
import time

import pandas as pd

from backtest.config import RESULTS_DIR, BacktestConfig
from backtest.engine import Backtester, Market
from backtest.report import summarize, write_outputs


def stop_stats(trades: pd.DataFrame) -> dict:
    """How stops actually filled: count, gap-opens, and fill cost vs the trigger level."""
    closed = trades[trades["status"] == "closed"]
    stops = closed[closed["exit_reason"] == "stop"]
    if not len(stops):
        return {"stops": 0, "gap_stops": 0, "stop_avg_$": None, "fill/credit_med": None,
                "fill/credit_max": None}
    exit_t = pd.to_datetime(stops["exit_time"], utc=True).dt.tz_convert("America/New_York")
    at_open = exit_t.dt.strftime("%H:%M") == "09:35"
    ratio = stops["exit_cost"] / stops["credit"]
    return {"stops": len(stops), "gap_stops": int(at_open.sum()),
            "stop_avg_$": round(float(stops["pnl"].mean()), 2),
            "fill/credit_med": round(float(ratio.median()), 2),
            "fill/credit_max": round(float(ratio.max()), 2)}


def cost_share(trades: pd.DataFrame) -> float | None:
    """All frictions as % of the credit collected at mid: entry slippage, commissions,
    exit slippage vs mid, and buybacks of worthless shorts."""
    c = trades[trades["status"] == "closed"]
    if not len(c):
        return None
    n = c["contracts"] * 100
    credit = (c["mid_credit"] * n).sum()
    costs = (((c["mid_credit"] - c["credit"]) * n).sum() + c["commissions"].sum()
             + ((c["exit_cost"] - c["exit_mid"]).clip(lower=0) * n).sum())
    return round(float(costs / credit * 100), 1) if credit > 0 else None


def main():
    ap = argparse.ArgumentParser(description="Sweep one config setting")
    ap.add_argument("--name", required=True)
    ap.add_argument("--param", help="one setting to sweep")
    ap.add_argument("--values", help="comma-separated values for --param")
    ap.add_argument("--variants", help="instead of --param: 'name:f=v,f=v;name2:f=v' (several settings per variant)")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="settings shared by every variant")
    ap.add_argument("--start")
    ap.add_argument("--end")
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
    if args.variants:
        values, cfgs = [], []
        for spec in args.variants.split(";"):
            label, _, body = spec.partition(":")
            sets = dict(kv.split("=", 1) for kv in body.split(",") if kv)
            values.append(label.strip())
            cfgs.append(BacktestConfig().update(common).update(sets))
        args.param = "variant"
    else:
        values = [v.strip() for v in args.values.split(",")]
        cfgs = [BacktestConfig().update(common).update({args.param: v}) for v in values]

    t0 = time.time()
    market = Market(cfgs[0].start, cfgs[0].end, cfgs[0].max_iv_error, cfgs[0].chain_dir)
    bts = [Backtester(c, market) for c in cfgs]
    logging.info(f"Sweep {args.param} over {values}: {market.days[0]} to {market.days[-1]}, "
                 f"{len(market.days)} sessions")
    for i, d in enumerate(market.days):
        for bt in bts:
            bt.run_day(d)
        if (i + 1) % 250 == 0:
            logging.info(f"  {d}")

    rows = []
    for v, bt in zip(values, bts):
        s = summarize(bt)
        write_outputs(bt, s, f"{args.name}/{args.param}={v}")
        row = {args.param: v, "trades": s["trades"], "win%": s["win_pct"],
               "net_$": s["net_pnl"], "ann%": s["annual_return_on_base_pct"],
               "maxDD%": s["max_drawdown_pct_of_base"], "worst_wk_$": s["worst_week_usd"],
               "PF": s["profit_factor"], "worst_$": s["worst_trade"],
               "costs%credit": cost_share(bt.trades())}
        row.update(stop_stats(bt.trades()))
        for y, yv in s["by_year"].items():
            row[y] = yv["pnl"]
        rows.append(row)

    table = pd.DataFrame(rows)
    out = RESULTS_DIR / args.name
    table.to_csv(out / "comparison.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print()
        print(table.to_string(index=False))
    print(f"\nSaved to {out}  ({time.time() - t0:.0f}s, {market.store.loads} expiry files loaded)")


if __name__ == "__main__":
    main()
