"""
Pull 5-minute QQQ option chains for MONTHLY expirations, for the 30-45 DTE test.

Monthly = the standard third-Friday expiry (the Thursday before when that Friday
is a holiday). Each expiry is pulled for the WINDOW_DAYS calendar days before it,
in two requests because ThetaData caps multi-day requests at one month. Strikes
are trimmed to +/-BAND of the daily spot (wide, since 16-delta strikes at 45 DTE
sit ~7% out and positions are held for weeks).

Usage
-----
  python pull_monthly.py --test      # latest monthly expiry only
  python pull_monthly.py             # all monthly expiries, 2016 onward

Resumable; output QQQ_OPTIONS_HOME/data/qqq_greeks_5m_monthly/<expiry>.parquet.
Auth: same API key file as pull_chains.py (never logged).
"""

import argparse
import calendar
import csv
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from pull_chains import FLOAT32, KEEP, ROOT, client, fetch_by_day, list_expirations

OUT_DIR = ROOT / "data" / "qqq_greeks_5m_monthly"
LOG_DIR = ROOT / "logs"
MANIFEST = LOG_DIR / "manifest_monthly.csv"

SYMBOL = "QQQ"
DATA_START = date(2016, 1, 1)
WINDOW_DAYS = 52        # covers entries up to 50 DTE
STRIKE_RANGE = 100      # strikes requested each side of spot (covers > +-20% of spot)
BAND = 0.30             # keep strikes within +/-30% of the daily spot
INTERVAL = "5m"

log = logging.getLogger("pull_monthly")
_lock = threading.Lock()


def third_friday(y: int, m: int) -> date:
    fridays = [d for d in calendar.Calendar().itermonthdates(y, m)
               if d.month == m and d.weekday() == 4]
    return fridays[2]


def monthly_expiries(exps: list[date]) -> list[date]:
    have, out = set(exps), []
    months = sorted({(e.year, e.month) for e in exps})
    for y, m in months:
        tf = third_friday(y, m)
        if tf in have:
            out.append(tf)
        elif tf - timedelta(days=1) in have:
            out.append(tf - timedelta(days=1))
    return out


def fetch(exp: date) -> pd.DataFrame:
    start = max(exp - timedelta(days=WINDOW_DAYS), DATA_START)
    mid = exp - timedelta(days=WINDOW_DAYS // 2)
    parts = []
    for lo, hi in ((start, min(mid, exp)), (mid + timedelta(days=1), exp)):
        if lo > hi:
            continue
        df = client().option_history_greeks_first_order(
            symbol=SYMBOL, expiration=exp, strike="*", right="both",
            start_date=lo, end_date=hi, interval=INTERVAL, strike_range=STRIKE_RANGE)
        if len(df):
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def trim(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[[c for c in KEEP if c in raw.columns]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    spot = df["underlying_price"].where(df["underlying_price"] > 0)
    day_spot = spot.groupby(df["timestamp"].dt.normalize()).transform("median")
    df = df[((df["strike"] / day_spot - 1).abs() <= BAND).fillna(False)]
    for c in FLOAT32:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    df["right"] = df["right"].astype(str).str.upper().str[0].astype("category")
    df["expiration"] = pd.to_datetime(df["expiration"].astype(str)).dt.date
    return (df.drop_duplicates(["timestamp", "right", "strike"])
              .sort_values(["timestamp", "right", "strike"]).reset_index(drop=True))


def out_path(exp: date) -> Path:
    return OUT_DIR / f"{exp:%Y%m%d}.parquet"


def record(row: dict):
    with _lock:
        new = not MANIFEST.exists()
        with open(MANIFEST, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


def pull_one(exp: date, retries: int = 3) -> dict:
    if out_path(exp).exists():
        return {"expiration": exp, "status": "skip"}
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            raw = fetch(exp)
            df = trim(raw) if len(raw) else raw
            tmp = out_path(exp).with_suffix(".tmp")
            df.to_parquet(tmp, compression="zstd", index=False)
            os.replace(tmp, out_path(exp))
            row = {"expiration": exp, "status": "ok" if len(df) else "empty",
                   "rows_raw": len(raw), "rows_kept": len(df),
                   "mb": round(out_path(exp).stat().st_size / 1e6, 2),
                   "seconds": round(time.time() - t0, 1),
                   "pulled_at": datetime.now().isoformat(timespec="seconds")}
            record(row)
            return row
        except Exception as e:
            log.warning(f"{exp} attempt {attempt}/{retries}: {type(e).__name__}: {str(e)[:300]}")
            time.sleep(10 * attempt)
    # Persistent failure: usually one corrupt day on the server. Fetch day by day.
    t0 = time.time()
    try:
        start = max(exp - timedelta(days=WINDOW_DAYS), DATA_START)
        raw, skipped = fetch_by_day(exp, start, exp, INTERVAL, STRIKE_RANGE)
        if len(raw):
            df = trim(raw)
            tmp = out_path(exp).with_suffix(".tmp")
            df.to_parquet(tmp, compression="zstd", index=False)
            os.replace(tmp, out_path(exp))
            row = {"expiration": exp, "status": "partial", "rows_raw": len(raw),
                   "rows_kept": len(df), "mb": round(out_path(exp).stat().st_size / 1e6, 2),
                   "seconds": round(time.time() - t0, 1),
                   "pulled_at": datetime.now().isoformat(timespec="seconds")}
            record(row)
            log.info(f"{exp}: saved day by day, skipped {[str(d) for d in skipped]}")
            return row
    except Exception as e:
        log.warning(f"{exp} day-by-day fallback failed: {type(e).__name__}: {str(e)[:200]}")
    row = {"expiration": exp, "status": "fail", "rows_raw": 0, "rows_kept": 0, "mb": 0,
           "seconds": 0, "pulled_at": datetime.now().isoformat(timespec="seconds")}
    record(row)
    return row


def main():
    ap = argparse.ArgumentParser(description="Pull 5-minute QQQ monthly option chains")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(LOG_DIR / f"pull_monthly_{date.today():%Y%m%d}.log"),
              logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)

    exps = list_expirations(DATA_START + timedelta(days=30), date.today() - timedelta(days=1))
    months = monthly_expiries(exps)
    log.info(f"{len(months)} monthly expiries from {months[0]} to {months[-1]}")
    if args.test:
        months = months[-1:]
    todo = [e for e in months if not out_path(e).exists()]
    log.info(f"{len(todo)} to pull")

    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for f in as_completed({pool.submit(pull_one, e): e for e in todo}):
            r = f.result()
            done += 1
            log.info(f"[{done}/{len(todo)}] {r['expiration']} {r['status']}  "
                     f"kept {r.get('rows_kept', 0):,} of {r.get('rows_raw', 0):,} rows  "
                     f"{r.get('mb', 0)} MB  {r.get('seconds', 0)}s  "
                     f"(elapsed {(time.time() - t0) / 60:.1f} min)")
    log.info(f"Done: {done} expiries in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
