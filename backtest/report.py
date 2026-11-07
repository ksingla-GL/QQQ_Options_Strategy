"""Summary metrics and output files for a backtest run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS_DIR


def _group(df: pd.DataFrame, key) -> dict:
    out = {}
    for k, g in df.groupby(key):
        out[str(k)] = {"trades": len(g), "pnl": round(float(g["pnl"].sum()), 2),
                       "win_pct": round(float((g["pnl"] > 0).mean() * 100), 1),
                       "avg_pnl": round(float(g["pnl"].mean()), 2)}
    return out


def summarize(bt) -> dict:
    cfg = bt.cfg
    trades, daily = bt.trades(), bt.daily_frame()
    base = cfg.base_equity
    eq = daily["equity"].astype(float)
    eq.index = pd.DatetimeIndex(pd.to_datetime(daily.index))
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 365.25)
    net = float(eq.iloc[-1] - base)
    chg = eq.diff().fillna(eq.iloc[0] - base)
    weekly = eq.resample("W-FRI").last().dropna().diff().dropna()
    dd_usd = eq - eq.cummax()
    dd_peak = eq / eq.cummax() - 1
    ann = net / base / years * 100
    max_dd_base = float(dd_usd.min() / base * 100)

    closed = trades[trades["status"] == "closed"] if len(trades) else trades
    pnl = closed["pnl"] if len(closed) else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]

    by_year = {}
    year_end = eq.groupby(eq.index.year).last()
    prev = base
    for y, last in year_end.items():
        e = eq[eq.index.year == y]
        yt = closed[pd.to_datetime(closed["exit_time"]).dt.year == y] if len(closed) else closed
        by_year[str(y)] = {"pnl": round(float(last - prev), 2),
                           "return_on_base_pct": round(float((last - prev) / base * 100), 2),
                           "max_dd_usd": round(float((e - e.cummax()).min()), 2),
                           "trades": len(yt),
                           "win_pct": round(float((yt["pnl"] > 0).mean() * 100), 1) if len(yt) else None}
        prev = last

    return {
        "period": f"{eq.index[0].date()} to {eq.index[-1].date()}",
        "years": round(years, 2),
        "base_equity": base,
        "end_equity": round(float(eq.iloc[-1]), 2),
        "net_pnl": round(net, 2),
        "annual_return_on_base_pct": round(ann, 2),
        "max_drawdown_usd": round(float(dd_usd.min()), 2),
        "max_drawdown_pct_of_base": round(max_dd_base, 2),
        "max_drawdown_pct_of_peak": round(float(dd_peak.min() * 100), 2),
        "calmar": round(ann / abs(max_dd_base), 2) if max_dd_base < 0 else None,
        "sharpe_daily": round(float(chg.mean() / chg.std() * np.sqrt(252)), 2) if chg.std() > 0 else None,
        "worst_day_usd": round(float(chg.min()), 2),
        "worst_week_usd": round(float(weekly.min()), 2) if len(weekly) else None,
        "trades": int(len(closed)),
        "open_at_end": int((trades["status"] == "open").sum()) if len(trades) else 0,
        "win_pct": round(float((pnl > 0).mean() * 100), 1) if len(pnl) else None,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else None,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        "worst_trade": round(float(pnl.min()), 2) if len(pnl) else None,
        "avg_credit": round(float(closed["credit"].mean()), 4) if len(closed) else None,
        "avg_width": round(float(closed["width"].mean()), 2) if len(closed) else None,
        "avg_contracts": round(float(closed["contracts"].mean()), 2) if len(closed) else None,
        "avg_max_loss_usd": round(float(closed["max_loss"].mean()), 2) if len(closed) else None,
        "commissions_usd": round(float(trades["commissions"].sum()), 2) if len(trades) else 0.0,
        "halted_on": str(bt.halted_on) if bt.halted_on else None,
        "by_side": _group(closed, "side") if len(closed) else {},
        "by_exit": _group(closed, "exit_reason") if len(closed) else {},
        "by_tier1": _group(closed, "tier1") if len(closed) else {},
        "by_year": by_year,
        "skipped_entries": dict(sorted(bt.skips.items(), key=lambda kv: -kv[1])),
    }


def format_summary(s: dict) -> str:
    lines = [
        f"Period            {s['period']} ({s['years']} years)",
        f"Net P&L           ${s['net_pnl']:,.0f} on ${s['base_equity']:,.0f} base "
        f"({s['annual_return_on_base_pct']:.1f}% a year)",
        f"Max drawdown      ${s['max_drawdown_usd']:,.0f} ({s['max_drawdown_pct_of_base']:.1f}% of base, "
        f"{s['max_drawdown_pct_of_peak']:.1f}% of peak)",
        f"Calmar / Sharpe   {s['calmar']} / {s['sharpe_daily']}",
        f"Worst day / week  ${s['worst_day_usd']:,.0f} / ${s['worst_week_usd'] or 0:,.0f}",
        f"Trades            {s['trades']} closed, {s['open_at_end']} open, win {s['win_pct']}%",
        f"Avg win / loss    ${s['avg_win']} / ${s['avg_loss']}  (profit factor {s['profit_factor']}, "
        f"worst ${s['worst_trade']})",
        f"Avg credit/width  ${s['avg_credit']} on ${s['avg_width']}, {s['avg_contracts']} contracts, "
        f"max loss ${s['avg_max_loss_usd']}",
        f"Commissions       ${s['commissions_usd']:,.0f}",
        f"Halted            {s['halted_on'] or 'no'}",
        "",
        "By exit:  " + "  ".join(f"{k} {v['trades']} (${v['pnl']:,.0f})" for k, v in s["by_exit"].items()),
        "By side:  " + "  ".join(f"{k} {v['trades']} (${v['pnl']:,.0f}, win {v['win_pct']}%)"
                                 for k, v in s["by_side"].items()),
        "By tier1: " + "  ".join(f"{k} {v['trades']} (${v['pnl']:,.0f})" for k, v in s["by_tier1"].items()),
        "",
        "Year   P&L        Return  MaxDD      Trades  Win%",
    ]
    for y, v in s["by_year"].items():
        lines.append(f"{y}  ${v['pnl']:>9,.0f}  {v['return_on_base_pct']:>5.1f}%  ${v['max_dd_usd']:>9,.0f}  "
                     f"{v['trades']:>6}  {v['win_pct'] if v['win_pct'] is not None else '-':>5}")
    lines += ["", "Skipped entries: " + ", ".join(f"{k} {v}" for k, v in s["skipped_entries"].items())]
    return "\n".join(lines)


def write_outputs(bt, summary: dict, name: str) -> Path:
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    bt.trades().to_csv(out / "trades.csv", index=False)
    bt.daily_frame().to_csv(out / "daily.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "summary.txt").write_text(format_summary(summary))
    (out / "config.json").write_text(json.dumps(bt.cfg.to_dict(), indent=2))
    return out
