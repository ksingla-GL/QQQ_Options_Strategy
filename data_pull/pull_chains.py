"""
Pull 1-minute QQQ option chains from ThetaData (Options Standard tier).

One request per expiration to the first-order Greeks endpoint, covering the
last WINDOW_DAYS calendar days before expiry. Each row is one contract at one
minute: bid, ask, delta, theta, vega, IV and the underlying price. Strikes
are trimmed to +/-BAND of the spot on that day and saved as one Parquet file
per expiration.

Usage
-----
  python pull_chains.py --test                 # latest expiration only, prints stats
  python pull_chains.py                        # full pull, 2016-01-01 to yesterday
  python pull_chains.py --start 2024-01-01 --workers 4

Resumable: an expiration whose Parquet file already exists is skipped.
Files go under QQQ_OPTIONS_HOME (default D:/QQQ_Options): data/, logs/.
Auth: API key read from QQQ_OPTIONS_KEY_FILE (default QQQ_OPTIONS_HOME/API_key.txt),
never logged and never stored in the repo.
Raw data stays on this machine (ThetaData terms forbid passing it on).
"""

import argparse
import csv
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from thetadata import ThetaClient

ROOT = Path(os.environ.get("QQQ_OPTIONS_HOME", "D:/QQQ_Options"))
KEY_FILE = Path(os.environ.get("QQQ_OPTIONS_KEY_FILE", ROOT / "API_key.txt"))
OUT_DIR = ROOT / "data" / "qqq_greeks_1m"
LOG_DIR = ROOT / "logs"
MANIFEST = LOG_DIR / "manifest.csv"

SYMBOL = "QQQ"
DATA_START = date(2016, 1, 1)   # first date on the Standard tier
WINDOW_DAYS = 14                # calendar days before expiry (covers 10-DTE entries)
STRIKE_RANGE = 100              # strikes requested each side of spot
BAND = 0.15                     # keep strikes within +/-15% of the daily spot
INTERVAL = "1m"

KEEP = ["timestamp", "expiration", "strike", "right", "bid", "ask", "delta",
        "theta", "vega", "implied_vol", "iv_error", "underlying_price"]
FLOAT32 = ["strike", "bid", "ask", "delta", "theta", "vega", "implied_vol",
           "iv_error", "underlying_price"]

log = logging.getLogger("pull")
_local = threading.local()
_manifest_lock = threading.Lock()
_root_lock = threading.Lock()
_root = None


# ---------------------------------------------------------------------- #
# ThetaData access (one login, one client per worker thread)
# ---------------------------------------------------------------------- #
def _api_key() -> str:
    if not KEY_FILE.exists():
        raise FileNotFoundError(f"No API key file at {KEY_FILE} (set QQQ_OPTIONS_KEY_FILE)")
    text = KEY_FILE.read_text().strip()
    if "=" in text:  # tolerate THETADATA_API_KEY="..." style files
        text = text.split("=", 1)[1]
    return text.strip().strip('"').strip("'")


def client() -> ThetaClient:
    global _root
    c = getattr(_local, "client", None)
    if c is None:
        with _root_lock:
            if _root is None:
                _root = ThetaClient(api_key=_api_key(), dataframe_type="pandas")
        c = ThetaClient(existing_authorized_client=_root, dataframe_type="pandas")
        _local.client = c
    return c


def list_expirations(start: date, end: date) -> list[date]:
    df = client().option_list_expirations(symbol=SYMBOL)
    col = "expiration" if "expiration" in df.columns else df.columns[0]
    exps = pd.to_datetime(df[col].astype(str)).dt.date
    return sorted(e for e in set(exps) if start <= e <= end)


def fetch(exp: date) -> pd.DataFrame:
    start = max(exp - timedelta(days=WINDOW_DAYS), DATA_START)
    return client().option_history_greeks_first_order(
        symbol=SYMBOL, expiration=exp, strike="*", right="both",
        start_date=start, end_date=exp, interval=INTERVAL,
        strike_range=STRIKE_RANGE)


def fetch_by_day(exp: date, start: date, end: date, interval: str,
                 strike_range: int) -> tuple[pd.DataFrame, list]:
    """Fallback when a multi-day request keeps failing on corrupt server-side data
    ("Corrupt FIT data", "Wrong number of data fields"): one request per weekday,
    skipping only the days that error. Returns (data, days skipped)."""
    parts, skipped = [], []
    for day in pd.bdate_range(start, end):
        d = day.date()
        try:
            df = client().option_history_greeks_first_order(
                symbol=SYMBOL, expiration=exp, strike="*", right="both", date=d,
                interval=interval, strike_range=strike_range)
            if len(df):
                parts.append(df)
        except Exception as e:  # holidays also land here; only real trading days matter
            skipped.append(d)
            log.warning(f"{exp} day {d} skipped: {type(e).__name__}: {str(e)[:120]}")
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), skipped


# ---------------------------------------------------------------------- #
# Trim + save
# ---------------------------------------------------------------------- #
def trim(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[[c for c in KEEP if c in raw.columns]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    spot = df["underlying_price"].where(df["underlying_price"] > 0)
    day_spot = spot.groupby(df["timestamp"].dt.normalize()).transform("median")
    in_band = ((df["strike"] / day_spot - 1).abs() <= BAND).fillna(False)
    df = df[in_band]
    for c in FLOAT32:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    df["right"] = df["right"].astype(str).str.upper().str[0].astype("category")
    df["expiration"] = pd.to_datetime(df["expiration"].astype(str)).dt.date
    return df.sort_values(["timestamp", "right", "strike"]).reset_index(drop=True)


def out_path(exp: date) -> Path:
    return OUT_DIR / f"{exp:%Y%m%d}.parquet"


def save(df: pd.DataFrame, exp: date) -> int:
    path = out_path(exp)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)
    return path.stat().st_size


def record(row: dict):
    with _manifest_lock:
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
            size = save(df, exp)
            row = {"expiration": exp, "status": "ok" if len(df) else "empty",
                   "rows_raw": len(raw), "rows_kept": len(df),
                   "strikes": df["strike"].nunique() if len(df) else 0,
                   "mb": round(size / 1e6, 2),
                   "seconds": round(time.time() - t0, 1),
                   "pulled_at": datetime.now().isoformat(timespec="seconds")}
            record(row)
            return row
        except Exception as e:  # network / server hiccups: back off and retry
            log.warning(f"{exp} attempt {attempt}/{retries}: "
                        f"{type(e).__name__}: {str(e)[:300]}")
            time.sleep(10 * attempt)
    # Persistent failure: usually one corrupt day on the server. Fetch day by day.
    t0 = time.time()
    try:
        start = max(exp - timedelta(days=WINDOW_DAYS), DATA_START)
        raw, skipped = fetch_by_day(exp, start, exp, INTERVAL, STRIKE_RANGE)
        if len(raw):
            df = trim(raw)
            size = save(df, exp)
            row = {"expiration": exp, "status": "partial", "rows_raw": len(raw),
                   "rows_kept": len(df), "strikes": df["strike"].nunique(),
                   "mb": round(size / 1e6, 2), "seconds": round(time.time() - t0, 1),
                   "pulled_at": datetime.now().isoformat(timespec="seconds")}
            record(row)
            log.info(f"{exp}: saved day by day, skipped {[str(d) for d in skipped]}")
            return row
    except Exception as e:
        log.warning(f"{exp} day-by-day fallback failed: {type(e).__name__}: {str(e)[:200]}")
    row = {"expiration": exp, "status": "fail", "rows_raw": 0, "rows_kept": 0,
           "strikes": 0, "mb": 0, "seconds": 0,
           "pulled_at": datetime.now().isoformat(timespec="seconds")}
    record(row)
    return row


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / f"pull_{date.today():%Y%m%d}.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)


def describe(exp: date):
    df = pd.read_parquet(out_path(exp))
    days = df["timestamp"].dt.date
    log.info(f"  rows {len(df):,}  contracts {df.groupby(['strike', 'right']).ngroups}  "
             f"strikes {df['strike'].min():.0f}-{df['strike'].max():.0f}")
    log.info(f"  {days.min()} to {days.max()}  ({days.nunique()} days, "
             f"{df.groupby(days)['timestamp'].nunique().median():.0f} minutes/day)")
    log.info(f"  spot {df['underlying_price'].min():.2f}-{df['underlying_price'].max():.2f}  "
             f"zero bid/ask rows {((df['bid'] <= 0) & (df['ask'] <= 0)).mean():.1%}")
    log.info("  sample:\n" + df.sample(min(5, len(df)), random_state=1).to_string())


def main():
    ap = argparse.ArgumentParser(description="Pull 1-minute QQQ option chains")
    ap.add_argument("--start", default=str(DATA_START))
    ap.add_argument("--end", default=str(date.today() - timedelta(days=1)))
    ap.add_argument("--workers", type=int, default=4, help="Standard tier allows 4")
    ap.add_argument("--limit", type=int, default=0, help="pull at most N expirations")
    ap.add_argument("--test", action="store_true", help="latest expiration only")
    args = ap.parse_args()

    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    exps = list_expirations(start, end)
    log.info(f"{SYMBOL}: {len(exps)} expirations from {start} to {end}")
    if args.test:
        exps = exps[-1:]
    elif args.limit:
        exps = exps[:args.limit]
    todo = [e for e in exps if not out_path(e).exists()]
    log.info(f"{len(todo)} to pull, {len(exps) - len(todo)} already saved")

    t0, done, rows = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(pull_one, e): e for e in todo}
        for f in as_completed(futures):
            r = f.result()
            done += 1
            rows += r.get("rows_kept", 0)
            log.info(f"[{done}/{len(todo)}] {r['expiration']} {r['status']}  "
                     f"kept {r.get('rows_kept', 0):,} of {r.get('rows_raw', 0):,} rows  "
                     f"{r.get('mb', 0)} MB  {r.get('seconds', 0)}s  "
                     f"(elapsed {(time.time() - t0) / 60:.1f} min)")

    if args.test and todo:
        describe(todo[0])
    log.info(f"Done: {done} expirations, {rows:,} rows kept, "
             f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
