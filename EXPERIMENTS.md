# Experiments

Every result in the README comes from one of the commands below.

- **Outputs** - each run writes its trades, daily equity, summary and full `config.json` to `QQQ_OPTIONS_HOME/results/<name>/`.
- **Defaults** - settings that aren't listed are the defaults in `backtest/config.py`.
- **Period** - all tuning used March 2016 to 30 April 2025. The final year, to October 2026, was held back. No version earned a test on it.
- **Drawdown halt** - `dd_halt_pct=0` switches the halt off so that every year contributes trades.

## 1. Core strategy and stop levels (7-day expiry)

```bash
python sweep.py --name card_core --end 2025-04-30 --set dd_halt_pct=0 \
  --variants "spec_2x:;stop_3x:stop_mult=3;no_stop:stop_mult=100;3x_expire:stop_mult=3,expire_worthless=true;put_weekly_3x:entry_policy=put_weekly,stop_mult=3;condor_weekly_3x:entry_policy=condor_weekly,stop_mult=3"
```

- **spec_2x** - the rules as specified.
- **stop_3x, no_stop** - wider stops. A `stop_mult` of 100 never triggers.
- **3x_expire** - a short leg with no bid that is at least 1% out of the money is left to expire instead of being bought back.
- **put_weekly_3x, condor_weekly_3x** - no-signal baselines, entered every Friday at 10:00.

## 2. Structure: regime filter, strike distance, low-cost execution

```bash
python sweep.py --name card_structure --end 2025-04-30 --set stop_mult=5 --set dd_halt_pct=0 \
  --variants "base_5x:;regime_with_trend:regime_tilt=with_trend;regime_bull_puts:regime_tilt=bull_puts_only;delta5_wide:short_delta=0.05,width_pct=1.4,min_credit_pct=0.02;delta7_wide:short_delta=0.07,width_pct=1.0,min_credit_pct=0.03;trend_delta5:regime_tilt=with_trend,short_delta=0.05,width_pct=1.4,min_credit_pct=0.02;trend_delta5_costfix:regime_tilt=with_trend,short_delta=0.05,width_pct=1.4,min_credit_pct=0.02,expire_worthless=true,commission_model=flat,monthly_platform_fee=10,fees_per_contract=0.1;bullputs_delta5_costfix:regime_tilt=bull_puts_only,short_delta=0.05,width_pct=1.4,min_credit_pct=0.02,expire_worthless=true,commission_model=flat,monthly_platform_fee=10,fees_per_contract=0.1"
```

- **regime_with_trend** - put spreads only while NDX is above its 250-day average (regimes A and B), and call spreads only while it is below (C and D).
- **regime_bull_puts** - no call spreads above the 250-day average, and both sides below it.
- **delta5_wide, delta7_wide** - the short leg sits further out, with a wider spread and a lower minimum credit.
- **\*_costfix** - low-cost execution: no commission, $0.10 per contract per leg in fees, $10 a month, and worthless shorts left to expire. `trend_delta5_costfix` is the -$552 combination.

## 3. Machine-learning entry veto (walk-forward)

`veto_test.py` first runs the base settings. It then reruns them with the veto, refit every 40 sessions on base trades that had already closed.

- **reg:x** - keeps an entry when the predicted return on risk is at least x.
- **clf:p** - keeps the entries least likely to lose; p is the share kept.

```bash
python veto_test.py --name card_veto --end 2025-04-30 --set stop_mult=3 --set dd_halt_pct=0
```

```bash
python veto_test.py --name card_opt_4x --end 2025-04-30 --set stop_mult=4 --set dd_halt_pct=0 --modes reg:0,reg:0.01,reg:0.02
```

```bash
python veto_test.py --name card_opt_5x --end 2025-04-30 --set stop_mult=5 --set dd_halt_pct=0 --modes reg:0,reg:0.01,reg:0.02
```

The veto on top of the two low-cost combinations from section 2:

```bash
python veto_test.py --name veto_trend_d5 --end 2025-04-30 --set regime_tilt=with_trend \
  --set short_delta=0.05 --set width_pct=1.4 --set min_credit_pct=0.02 --set expire_worthless=true \
  --set commission_model=flat --set monthly_platform_fee=10 --set fees_per_contract=0.1 \
  --set stop_mult=5 --set dd_halt_pct=0 --modes reg:0,reg:0.01,reg:0.02
```

```bash
python veto_test.py --name veto_bullputs_d5 --end 2025-04-30 --set regime_tilt=bull_puts_only \
  --set short_delta=0.05 --set width_pct=1.4 --set min_credit_pct=0.02 --set expire_worthless=true \
  --set commission_model=flat --set monthly_platform_fee=10 --set fees_per_contract=0.1 \
  --set stop_mult=5 --set dd_halt_pct=0 --modes reg:0,reg:0.01,reg:0.02
```

## 4. Commission models (March 2016 to September 2020)

```bash
python sweep.py --name commissions --end 2020-09-04 --set stop_mult=3 --set dd_halt_pct=0 \
  --variants "tiered:fees_per_contract=0.04;flat:commission_model=flat,monthly_platform_fee=10,fees_per_contract=0.1;open_only:commission_model=open_only,fees_per_contract=0.1"
```

- **tiered** - the default model: $0.65, $0.50 or $0.25 per contract by premium, with a $1 minimum per order.
- **flat** - no commission, $0.10 per contract in fees, $10 a month.
- **open_only** - $1 per contract per leg to open, capped at $10 a leg, and free to close.

## 5. 45-day expiry (monthly chains)

Settings shared by both sweeps:

- **Chains** - the 5-minute monthly set, with expiries 30 to 50 days out (target 45).
- **Strikes** - a 16-delta short leg, a spread about 1.2% of the QQQ price wide, and a minimum credit of 10% of the width.
- **Exits** - a 50% profit target and a time exit at 21 days to expiry.
- **Costs** - the flat commission model.
- **Baselines** - the no-signal versions enter on Mondays.

Exit fills: `_nat` pays the full bid/ask spread on non-stop exits, `_half` pays half of it, and `_mid` exits at mid-market.

```bash
python sweep.py --name dte45_fills --end 2025-04-30 \
  --set chain_dir=qqq_greeks_5m_monthly --set target_dte=45 --set min_dte=30 --set max_dte=50 \
  --set short_delta=0.16 --set width_pct=1.2 --set credit_mode=width --set commission_model=flat \
  --set monthly_platform_fee=10 --set fees_per_contract=0.1 --set profit_take=0.5 --set exit_at_dte=21 \
  --set dd_halt_pct=0 \
  --variants "putwk_sl2_nat:entry_policy=put_weekly,weekly_entry_dow=0;putwk_sl2_half:entry_policy=put_weekly,weekly_entry_dow=0,patient_exit_frac=0.5;putwk_nostop_nat:entry_policy=put_weekly,weekly_entry_dow=0,stop_mult=100;putwk_nostop_half:entry_policy=put_weekly,weekly_entry_dow=0,stop_mult=100,patient_exit_frac=0.5;putwk_nostop_mid:entry_policy=put_weekly,weekly_entry_dow=0,stop_mult=100,patient_exit_frac=0;trend_nostop_nat:regime_tilt=with_trend,stop_mult=100;trend_nostop_half:regime_tilt=with_trend,stop_mult=100,patient_exit_frac=0.5;trend_nostop_mid:regime_tilt=with_trend,stop_mult=100,patient_exit_frac=0;condorwk_nostop_half:entry_policy=condor_weekly,weekly_entry_dow=0,stop_mult=100,patient_exit_frac=0.5"
```

Stops, regime filter, strike distance and commissions at 45 days:

```bash
python sweep.py --name dte45_core --end 2025-04-30 \
  --set chain_dir=qqq_greeks_5m_monthly --set target_dte=45 --set min_dte=30 --set max_dte=50 \
  --set short_delta=0.16 --set width_pct=1.2 --set credit_mode=width --set commission_model=flat \
  --set monthly_platform_fee=10 --set fees_per_contract=0.1 --set profit_take=0.5 --set exit_at_dte=21 \
  --set dd_halt_pct=0 \
  --variants "trend_sl2:regime_tilt=with_trend;trend_sl3:regime_tilt=with_trend,stop_mult=3;trend_nostop:regime_tilt=with_trend,stop_mult=100;trend_sl2_noregime:;trend_sl2_d10:regime_tilt=with_trend,short_delta=0.10,min_credit_width_pct=6;trend_sl2_hold:regime_tilt=with_trend,profit_take=0,exit_at_dte=0;trend_sl2_tiered:regime_tilt=with_trend,commission_model=ibkr_fixed,monthly_platform_fee=0,fees_per_contract=0.04;put_weekly_sl2:entry_policy=put_weekly,weekly_entry_dow=0;condor_weekly_sl2:entry_policy=condor_weekly,weekly_entry_dow=0"
```

## Reproducing the README numbers

- **45-day runs** - rerunning `dte45_fills` reproduces its saved results exactly, including the README's -$7,712 and +$1,617 rows. `dte45_core` was first run before the profit-target and time-exit handling changed. Only its variant with neither exit (`trend_sl2_hold`) reproduces exactly. The others come out up to $1,435 lower, and its `trend_nostop` row then equals `dte45_fills`' `trend_nostop_nat`. The README does not use these numbers.
- **7-day runs (sections 1 to 4)** - the saved results were produced before the 22 Jul 2016 weekly expiry was recovered from a corrupt vendor day. A rerun today also trades that expiry. Section 1, rerun in full:
  - **Trend versions** - each gains 2 to 4 trades, and its net result moves by less than $100. For example, the 3x stop comes out at -$12,316 instead of -$12,410.
  - **No-signal baselines** - unchanged.
  - **Profit factor and drawdown** - profit factors move by at most 0.01, and maximum drawdowns by at most 0.2 percentage points.
  - **Sections 2 to 4** - these also predate the recovery and were not rerun, so expect differences of a similar size.
