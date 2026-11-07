"""
Backtest configuration for the QQQ weekly credit spread strategy.

Defaults follow "QQQ Weekly Credit Spread Strategy: Spec v1". Dollar settings
in the spec ($5 width, $0.30 minimum credit) were written at QQQ ~$740, but
QQQ was ~$100 in 2016, so width and minimum credit are expressed as % of spot
by default (the same levels at today's price). Set width_mode / credit_mode
to "dollar" to use fixed dollar values instead.

Data, caches and results live outside the repo, under QQQ_OPTIONS_HOME
(default D:/QQQ_Options): data/qqq_greeks_1m/ (weekly chains, 1-minute),
data/qqq_greeks_5m_monthly/ (monthly chains, 5-minute), cache/, results/.
"""

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DATA_ROOT = Path(os.environ.get("QQQ_OPTIONS_HOME", "D:/QQQ_Options"))
CHAIN_DIR = DATA_ROOT / "data" / "qqq_greeks_1m"
CACHE_DIR = DATA_ROOT / "cache"
RESULTS_DIR = DATA_ROOT / "results"


def chain_path(chain_dir: str = "") -> Path:
    """Folder of chain files: "" = the weekly 1-minute set, a bare folder name = a
    folder under DATA_ROOT/data (e.g. qqq_greeks_5m_monthly), otherwise a full path."""
    if not chain_dir:
        return CHAIN_DIR
    path = Path(chain_dir)
    return path if path.is_absolute() else DATA_ROOT / "data" / path


@dataclass
class BacktestConfig:
    # --- Period (first ~30 sessions of data are indicator warm-up) ---
    start: str = "2016-03-01"
    end: str = "2026-10-05"
    # Option chains to trade: "" = the weekly 1-minute set (CHAIN_DIR); another set by folder
    # name or full path (see chain_path), e.g. qqq_greeks_5m_monthly for the 30-45 DTE test.
    # QQQ's price for the signals always comes from the 1-minute weekly set.
    chain_dir: str = ""

    # --- Account and sizing ---
    base_equity: float = 45_000.0
    sizing_base: str = "fixed"          # "fixed": size off base_equity (profits swept); "equity": compound
    risk_per_trade_pct: float = 1.0     # target max loss per entry, % of sizing base
    max_risk_per_trade_pct: float = 2.0 # never open an entry whose max loss exceeds this
    max_open_risk_pct: float = 20.0     # sum of max losses on open spreads
    max_open_per_side: int = 5
    max_open_total: int = 10
    side_cooldown_min: int = 30

    # --- Entry policy ---
    entry_policy: str = "trend"         # "trend" | "put_weekly" | "condor_weekly" (Phase 2 baselines)
    weekly_entry_dow: int = 4           # baselines: weekday to enter (0 = Monday, 4 = Friday)
    weekly_entry_time: str = "10:00"
    entry_start: str = "09:45"          # first 15-minute close that can enter
    entry_end: str = "15:45"            # last 15-minute close that can enter

    # --- Signal (see signals.py) ---
    tsf_fast: int = 14
    tsf_slow: int = 26
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    rsi_high: float = 80.0
    rsi_low: float = 20.0

    # --- NDX regime tilt (spec Phase 2) ---
    # "off"; "with_trend": regime A/B (NDX above MA250) sells only put spreads, C/D only
    # call spreads; "bull_puts_only": no call spreads in A/B, both sides in C/D.
    regime_tilt: str = "off"

    # --- Support/resistance strike check (never sell a strike inside a confirmed level) ---
    # Level = 1-hour Supertrend(7, 3): support below price for puts, resistance above for calls.
    # "off"; "adjust": move a short strike that sits inside the level to the first strike
    # beyond it; "skip": skip the entry instead. Applied only when the level exists on that side.
    sr_filter: str = "off"

    # --- Expiry choice ---
    target_dte: int = 7                 # calendar days
    min_dte: int = 5
    max_dte: int = 9

    # --- Strikes ---
    short_delta: float = 0.10           # |delta| of the short leg
    width_mode: str = "pct"             # "pct" of spot or "dollar"
    width_pct: float = 0.70             # ~$5 at QQQ $715
    width_dollar: float = 5.0
    credit_mode: str = "pct"            # minimum net credit: "pct" of spot, "dollar", or "width" (% of width)
    min_credit_pct: float = 0.04        # ~$0.30 at QQQ $750
    min_credit_dollar: float = 0.30
    min_credit_width_pct: float = 10.0  # credit_mode "width": credit >= this % of the spread width
    max_iv_error: float = 1.0           # quotes with |iv_error| above this have unusable Greeks
    # A quote is trusted only if two-sided and not absurdly wide: bid > 0 and
    # ask - bid <= max(abs, rel x bid). Market makers post placeholder asks on
    # near-worthless options (e.g. bid 0 / ask 0.75 on an expiring call 3% OTM);
    # those must not trigger stops, fill exits, or inflate entry credits.
    quote_max_width_abs: float = 0.10
    quote_max_width_rel: float = 1.0
    worthless_buyback_cap: float = 0.05  # buy back a no-bid OTM short for at most this
    expire_worthless: bool = False      # spec: close everything at 15:30. True: let a short with
    expire_min_otm_pct: float = 1.0     #   no bid that is at least this % OTM expire (no buyback/commission)

    # --- Fills and costs ---
    entry_slippage: float = 0.01        # $ per share given up from the mid credit
    exit_fill: str = "natural"          # stops: "natural" (pay the spread) or "mid"
    # Non-urgent exits (profit-take, time exit, expiry-day close) can be worked as a
    # patient limit order: pay this fraction of the mid -> natural gap (1 = natural, 0 = mid).
    patient_exit_frac: float = 1.0
    # Commission model: "ibkr_fixed" (premium tiers below + per-order minimum),
    # "flat" (commission_flat per contract per leg, open and close, e.g. Tradier Pro = 0),
    # "open_only" (commission_open per contract per leg to open, capped per leg; free to close,
    # e.g. tastytrade $1 capped at $10). fees_per_contract applies to every model.
    commission_model: str = "ibkr_fixed"
    commission_flat: float = 0.0
    commission_open: float = 1.0
    commission_leg_cap: float = 10.0
    monthly_platform_fee: float = 0.0   # e.g. Tradier Pro $10
    commission_low: float = 0.25        # IBKR fixed, per contract: premium < $0.05
    commission_mid: float = 0.50        # premium $0.05 to $0.10
    commission_high: float = 0.65       # premium >= $0.10
    fees_per_contract: float = 0.03     # exchange and regulatory fees
    min_commission_per_order: float = 1.00

    # --- Exits ---
    stop_mult: float = 2.0              # close when the spread mark reaches this x credit
    profit_take: float = 0.0            # 0 = off; 0.5 = close once 50% of the credit is captured
    exit_at_dte: int = 0                # 0 = off; e.g. 21 = close at the first check once DTE <= 21
    expiry_close_time: str = "15:30"    # on expiry day (30 min before an early close)
    check_every_min: int = 5
    first_check: str = "09:35"          # 09:30 quotes are stale and wide

    # --- Gates (block new entries only) ---
    vol_gate: float = 35.0              # 20-day realized vol of NDX, %; 0 = off
    term_gate: bool = True              # block when VIX closed above VIX3M the prior day
    friday_cutoff: str = ""             # e.g. "13:00" blocks Friday entries from then on

    # --- Drawdown overlay ---
    dd_reduce_pct: float = 10.0         # halve the per-side cap below this drawdown
    dd_halt_pct: float = 15.0           # halt new entries below this drawdown (until reset)
    dd_halt_reset_days: int = 0         # 0 = never auto-reset (spec: reset by hand)

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, overrides: dict) -> "BacktestConfig":
        """Apply {name: value-string} overrides, casting to each field's type."""
        types = {f.name: f.type for f in fields(self)}
        for key, raw in overrides.items():
            if key not in types:
                raise KeyError(f"Unknown config field: {key}")
            current = getattr(self, key)
            if isinstance(current, bool):
                value = str(raw).lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = str(raw)
            setattr(self, key, value)
        return self
