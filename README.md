# QQQ Credit Spread Backtest

A minute-level backtest of a short credit spread strategy on QQQ options. The repo also holds the code that downloads the option data and the exact settings behind every variant I tested.

The short answer: no version made money after realistic costs over March 2016 to April 2025. The [results](#results) section has the numbers.

## The strategy

The defaults in `backtest/config.py` are the rules as specified:

- **Entry** - at every 15-minute close from 09:45 to 15:45, sell a put spread when the 15-minute trend state is up or a call spread when it is down. An entry needs the 15-minute MACD(12, 26, 9) histogram to agree. Trend state compares two time-series moving averages, TSF(14) and TSF(26), and the price against TSF(14).
- **Strikes** - the expiry is the one nearest 7 calendar days out (5 to 9). The short leg is the out-of-the-money strike nearest 10 delta. The long leg is about 0.7% of the QQQ price further out, and the minimum credit is 0.04% of the QQQ price.
- **Exits** - stop out when the spread's value reaches 2x the credit, checked every 5 minutes from 09:35. Otherwise close at 15:30 on expiry day.
- **Sizing** - each entry risks about 1% of a $45,000 base, never more than 2%. Open risk is capped at 20%, with at most 5 spreads per side, 10 in total, and 30 minutes between entries on the same side.
- **Drawdown overlay** - below a 10% drawdown the per-side cap is halved, and below 15% new entries stop until reset by hand. The research runs switch the halt off so that every year contributes trades.
- **Gates** - no new entries when NDX 20-day realized volatility is above 35%, or when VIX closed above VIX3M the day before.

## How the backtest works

- **Data** - one Parquet file per expiration holds every contract's bid, ask, delta and implied volatility, minute by minute. Weekly expirations are 1-minute data from January 2016 to October 2026, and monthly expirations are 5-minute data for the 45-day test. QQQ's own minute price is rebuilt from the underlying price stored in the chain files.
- **No look-ahead** - a decision at a given minute sees only the last quote at or before that minute. Exits run before entries, and the daily gates use the prior close.
- **Fills** - entries fill $0.01 below the mid-market credit. Stops pay the full bid/ask spread: the short leg at the ask, the long leg at the bid. A stop triggered by a gap fills at the real post-gap price, not at the stop level.
- **Quote filter** - a quote counts only if it has a bid and the ask is within max($0.10, 1x the bid). This keeps placeholder asks on near-worthless options from triggering stops or setting fills.
- **Costs** - commissions are $0.65, $0.50 or $0.25 per contract per leg depending on the premium, with a $1 minimum per order, plus $0.03 in fees. Flat and open-only commission models are also available.
- **Sizing base** - fixed at $45,000 (profits are not compounded), so every year is weighted equally.

## Layout

```
backtest/            the engine (a Python package)
  config.py          every setting, with the spec defaults, and the data locations
  engine.py          Market (shared data) and Backtester (exits, entries, sizing, accounting)
  chains.py          option chains as dense minute x strike arrays
  signals.py         trend state, MACD, RSI and the other indicators on 15-minute and 1-hour bars
  market.py          trading calendar, daily gate inputs, QQQ minute series
  veto.py            walk-forward machine-learning entry filter
  report.py          summary metrics and output files
data_pull/
  pull_chains.py     weekly expirations, 1-minute
  pull_monthly.py    monthly expirations, 5-minute (45-day test)
run_backtest.py      one backtest
sweep.py             several variants over the same days in one pass
veto_test.py         a base run, then the ML filter on top, walk-forward
EXPERIMENTS.md       the exact command behind every result below
docs/
  QQQ_Weekly_Credit_Spread_Spec_v1.pdf       the strategy as specified
  QQQ_Credit_Spread_Backtest_Report_v1.pdf   the full backtest report
```

## Setup

```bash
pip install -r requirements.txt
```

- **Python** - tested with 3.12.
- **Data location** - option data, caches and results live outside the repo, under `QQQ_OPTIONS_HOME` (default `D:/QQQ_Options`):

  ```
  data/qqq_greeks_1m/            weekly chains
  data/qqq_greeks_5m_monthly/    monthly chains
  cache/                         rebuilt automatically
  results/<run name>/            trades.csv, daily.csv, summary.txt/.json, config.json
  ```

- **Option data** - the pull scripts need a paid options data subscription (they use the `thetadata` client). Put the API key in `QQQ_OPTIONS_HOME/API_key.txt`, or point `QQQ_OPTIONS_KEY_FILE` at it. The raw data is licensed, so it is not in this repo.
- **Daily data** - NDX, VIX and VIX3M closes come from Yahoo Finance and are cached on first use.

## Running

```bash
python data_pull/pull_chains.py --test
```
```bash
python data_pull/pull_chains.py
```
```bash
python run_backtest.py --name spec
```
```bash
python run_backtest.py --name d15 --set short_delta=0.15 --start 2022-01-01
```
```bash
python sweep.py --name stops --param stop_mult --values 2,3,5,100
```
```bash
python veto_test.py --name veto --set stop_mult=3
```

- **Data pulls** - `--test` pulls one expiration and prints a sample. Without it the script pulls everything and can be resumed.
- **Overrides** - any setting in `backtest/config.py` can be changed with `--set FIELD=VALUE`.
- **Outputs** - each run saves its full config next to its results, so any result can be traced back to its settings.

## Results

March 2016 to April 2025, on a $45,000 account with profits not compounded. Drawdown is the largest fall from a prior peak, as a percentage of the base. A profit factor below 1.0 means a loss.

| Version | Trades | Win rate | Net P&L | Max drawdown | Profit factor |
|---|---:|---:|---:|---:|---:|
| As specified (2x stop) | 1,489 | 53% | -$15,391 | -34.5% | 0.49 |
| 3x stop | 1,317 | 69% | -$12,410 | -28.0% | 0.57 |
| 5x stop | 1,221 | 80% | -$11,778 | -26.7% | 0.60 |
| No stop | 1,087 | 87% | -$14,958 | -33.8% | 0.53 |
| 3x stop, let worthless spreads expire | 1,370 | 69% | -$9,726 | -22.5% | 0.68 |
| No signal: weekly put spread, 3x stop | 142 | 73% | -$1,322 | -3.3% | 0.59 |
| No signal: weekly iron condor, 3x stop | 415 | 69% | -$4,327 | -9.6% | 0.56 |
| 5x stop, sell only with the NDX regime | 692 | 81% | -$6,092 | -13.8% | 0.62 |
| 5x stop, 5 delta with a wider spread | 1,841 | 82% | -$6,739 | -15.3% | 0.53 |
| 3x stop, ML entry veto | 745 | 73% | -$5,102 | -12.2% | 0.67 |
| Regime + 5 delta + 5x stop + low-cost execution | 1,367 | 85% | -$552 | -5.0% | 1.05 |
| 45-day expiry, regime, no stop, full-spread exits | 702 | 77% | -$7,712 | -18.5% | 0.63 |
| 45-day expiry, same, exits at mid-market | 904 | 82% | +$1,617 | -6.8% | 1.17 |

- **Costs swamp the edge** - the 3x version earned $1,411 before costs and paid $13,821 in costs (commissions, bid/ask paid on entry and exit, and buybacks of near-worthless shorts).
- **The signal adds nothing per trade** - trend-timed entries lost about as much per trade as entering every week with no signal.
- **The best combination is breakeven at most** - it was tuned on the same data it was scored on.
- **The 45-day version** - it is positive only if every exit fills exactly at mid-market.

## Data notes

- **Strikes** - only standard strikes (multiples of $0.50) are used; early years include adjusted legacy series. The weekly files keep strikes within 15% of spot and the monthly files within 30%.
- **Partial files** - two expirations were rebuilt day by day around a corrupt vendor day, so each is missing one session: the 22 Jul 2016 weekly has no 15 Jul 2016, and the 20 May 2016 monthly has no 22 Apr 2016.
