"""
Run a QQQ weekly credit spread backtest (Spec v1 rules by default).

Usage
-----
  python run_backtest.py --name base
  python run_backtest.py --name put_weekly --set entry_policy=put_weekly
  python run_backtest.py --name d15 --set short_delta=0.15 --start 2022-01-01

Any field of backtest/config.py can be overridden with --set FIELD=VALUE.
Outputs go to QQQ_OPTIONS_HOME/results/<name>/ (default D:/QQQ_Options): trades,
daily equity, summary and the full config of the run.
The run ends at the last date with contiguous chain data on disk.
"""

import argparse
import logging
import time

from backtest.config import BacktestConfig
from backtest.engine import Backtester
from backtest.report import format_summary, summarize, write_outputs


def main():
    ap = argparse.ArgumentParser(description="QQQ credit spread backtest")
    ap.add_argument("--name", default="base", help="results folder name")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    overrides = dict(kv.split("=", 1) for kv in args.set)
    if args.start:
        overrides["start"] = args.start
    if args.end:
        overrides["end"] = args.end
    cfg = BacktestConfig().update(overrides)

    t0 = time.time()
    bt = Backtester(cfg).run()
    summary = summarize(bt)
    out = write_outputs(bt, summary, args.name)
    print()
    print(format_summary(summary))
    print(f"\nSaved to {out}  ({time.time() - t0:.0f}s, {bt.store.loads} expiry files loaded)")


if __name__ == "__main__":
    main()
