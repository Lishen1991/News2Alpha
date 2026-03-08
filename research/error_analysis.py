"""
Error analysis for News2Alpha — identifies the weakest parts of the system
and produces actionable recommendations for what to improve next.

Covers:
  - Worst-performing assets, rules, and event types
  - Skip analysis (why signals were not traded)
  - Context flag combinations correlated with losses
  - Upside capture: were the best market moves in signals we filtered out?
  - Actionable recommendations ranked by estimated impact

Imports the shared data layer from diagnostics.py.

Usage:
    # Full error report
    py research/error_analysis.py

    # Include suppressed signals to see filter quality
    py research/error_analysis.py --all-signals

    # Show top N worst performers per category
    py research/error_analysis.py --top 5

    # Save recommendations to text file
    py research/error_analysis.py --output errors.csv

Example output:
    WORST ASSETS (avg net return, min 3 trades)──────────────────
      Asset                      Trades   Win%   Avg ret   Total PnL
      airlines_basket                12    33%    -2.14%     -25.7%

    RECOMMENDATIONS
      [HIGH]   Rule 'geo_oil_supply_shock' short leg (airlines, XLI) has -2.1% avg.
               Consider removing short leg or adding a momentum filter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.diagnostics import (
    _DB_PATH,
    _SEP,
    _SEP2,
    _pct,
    BacktestConfig,
    build_backtest_df,
    load_signal_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _worst_by(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    top_n: int = 5,
    min_trades: int = 3,
) -> pd.DataFrame:
    """Print and return the worst-performing groups by avg net return."""
    traded = df[df["traded"] == True]
    if traded.empty:
        print(f"\n{title:{len(_SEP) - 1}─}")
        print("  No traded signals.")
        return pd.DataFrame()

    grp = (
        traded.groupby(group_col)
        .agg(
            n_trades   =("idea_id",    "count"),
            win_rate   =("win",        "mean"),
            avg_return =("net_return", "mean"),
            med_return =("net_return", "median"),
            total_pnl  =("net_return", "sum"),
        )
        .query(f"n_trades >= {min_trades}")
        .sort_values("avg_return", ascending=True)
        .head(top_n)
    )

    print(f"\n{title:{len(_SEP) - 1}─}")
    print(
        f"  {'Group':<28}  {'Trades':>6}  {'Win%':>5}  "
        f"{'Avg ret':>8}  {'Median':>8}  {'Total PnL':>10}"
    )
    print(f"  {'─'*28}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}")
    for grp_val, row in grp.iterrows():
        print(
            f"  {str(grp_val):<28}  {int(row['n_trades']):>6}  "
            f"{row['win_rate']:>5.0%}  "
            f"{_pct(row['avg_return']):>8}  "
            f"{_pct(row['med_return']):>8}  "
            f"{_pct(row['total_pnl']):>10}"
        )
    return grp


def _skip_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse why signals were skipped by the portfolio engine.

    Helps identify structural issues:
      - max_positions_reached → consider increasing max_positions or reducing
        holding period so slots free up faster
      - asset_already_open    → consider --multi-asset or shorter hold
      - no_price_file         → price CSV is missing for this asset
      - no_price_data_entry   → signal date too recent; price data not yet available
    """
    skipped = df[df["traded"] == False].copy()
    skipped = skipped[skipped["skip_reason"].notna()]
    skipped = skipped[skipped["skip_reason"] != "not_loaded"]

    print(f"\n{'SKIP ANALYSIS':{len(_SEP) - 1}─}")
    if skipped.empty:
        print("  No skipped signals.")
        return pd.DataFrame()

    summary = (
        skipped.groupby("skip_reason")
        .agg(
            count        =("idea_id",    "count"),
            unique_assets=("asset",      "nunique"),
        )
        .sort_values("count", ascending=False)
    )

    _explanations = {
        "max_positions_reached":       "Portfolio full — signal arrived when 10 slots occupied.",
        "asset_already_open":          "Asset already in portfolio — one_pos_per_asset=True.",
        "no_price_file":               "Price CSV missing for this asset.",
        "no_price_data_entry":         "Signal date too recent; no T+1 price available.",
        "not_surfaced":                "Filtered by should_surface=False (market context).",
        "insufficient_history_for_exit": "Not enough price history to reach exit date.",
        "not_loaded":                  "Not loaded by signal loader (e.g. surface_only filter).",
        f"rank_score_below_min(0.0)":  "Score below min_rank_score threshold.",
    }

    print(f"  {'Reason':<40}  {'Count':>6}  {'Assets':>7}  Explanation")
    print(f"  {'─'*40}  {'─'*6}  {'─'*7}  {'─'*35}")
    for reason, row in summary.iterrows():
        explanation = _explanations.get(str(reason), "")
        # Handle dynamic rank_score reasons
        if not explanation and "rank_score_below_min" in str(reason):
            explanation = "Score below min_rank_score threshold."
        print(
            f"  {str(reason):<40}  {int(row['count']):>6}  "
            f"{int(row['unique_assets']):>7}  {explanation}"
        )
    return summary


def _flag_loss_analysis(df: pd.DataFrame) -> None:
    """
    Identify which context flag combinations are most associated with losses.

    Useful for tuning the market_context filter — if a combination of flags
    reliably predicts poor performance, add it to the suppression logic.
    """
    traded = df[df["traded"] == True].copy()
    if traded.empty:
        return

    flag_cols = [c for c in traded.columns if c.startswith("flag_")]
    if not flag_cols:
        return

    print(f"\n{'FLAG COMBINATIONS → LOSS CORRELATION':{len(_SEP) - 1}─}")

    rows = []
    for col in flag_cols:
        flagged = traded[traded[col] == True]
        if len(flagged) < 2:
            continue
        rows.append({
            "flag":       col.replace("flag_", ""),
            "n_flagged":  len(flagged),
            "win_rate":   flagged["win"].mean(),
            "avg_return": flagged["net_return"].mean(),
            "loss_count": (flagged["net_return"] < 0).sum(),
        })

    if not rows:
        print("  No flag data available (no market_context records joined).")
        return

    flag_df = pd.DataFrame(rows).sort_values("avg_return")
    print(f"  {'Flag':<28}  {'N':>5}  {'Win%':>5}  {'Avg ret':>8}  {'Losses':>7}")
    print(f"  {'─'*28}  {'─'*5}  {'─'*5}  {'─'*8}  {'─'*7}")
    for _, row in flag_df.iterrows():
        print(
            f"  {row['flag']:<28}  {int(row['n_flagged']):>5}  "
            f"{row['win_rate']:>5.0%}  {_pct(row['avg_return']):>8}  "
            f"{int(row['loss_count']):>7}"
        )
    print("  ↳ Flags with low win% and negative avg_return should suppress the signal.")


def _tail_loss_trades(df: pd.DataFrame, top_n: int = 10) -> None:
    """List the single worst-performing trades for manual review."""
    traded = df[df["traded"] == True].copy()
    if traded.empty:
        return

    worst = traded.nsmallest(top_n, "net_return")[
        ["idea_id", "asset", "direction", "primary_rule", "priority_bucket",
         "net_return", "entry_date", "exit_date"]
    ]

    print(f"\n{'WORST INDIVIDUAL TRADES (for manual review)':{len(_SEP) - 1}─}")
    print(
        f"  {'ID':>4}  {'Asset':<26}  {'Dir':<5}  {'Rule':<28}  "
        f"{'Bucket':<8}  {'Net ret':>8}"
    )
    print(f"  {'─'*4}  {'─'*26}  {'─'*5}  {'─'*28}  {'─'*8}  {'─'*8}")
    for _, row in worst.iterrows():
        print(
            f"  {int(row['idea_id']):>4}  {str(row['asset']):<26}  "
            f"{str(row['direction']):<5}  {str(row['primary_rule']):<28}  "
            f"{str(row['priority_bucket']):<8}  {_pct(row['net_return']):>8}"
        )
    print("  ↳ Investigate these to find common patterns — wrong direction, bad timing, thin market.")


def _generate_recommendations(df: pd.DataFrame) -> list[str]:
    """
    Generate actionable recommendations based on the diagnostic results.

    Returns a list of (priority, message) strings.
    Each recommendation references a specific component to change.
    """
    recs: list[tuple[str, str]] = []
    traded = df[df["traded"] == True]

    if traded.empty:
        return ["[CRITICAL] No trades executed — check price data coverage and signal pipeline."]

    # --- Rule-level ---
    rule_perf = traded.groupby("primary_rule")["net_return"].mean()
    for rule, avg_ret in rule_perf.items():
        n = (traded["primary_rule"] == rule).sum()
        if avg_ret < -0.005 and n >= 3:
            recs.append((
                "HIGH",
                f"Rule '{rule}' has avg net return {avg_ret:+.2%} over {n} trades. "
                "Review asset list and direction assignments in macro_rules.yaml.",
            ))

    # --- Asset-level ---
    asset_perf = traded.groupby(["asset", "direction"])["net_return"].agg(["mean", "count"])
    for (asset, direction), row in asset_perf.iterrows():
        if row["mean"] < -0.01 and row["count"] >= 3:
            recs.append((
                "MEDIUM",
                f"Asset '{asset}' {direction} has avg {row['mean']:+.2%} over {int(row['count'])} trades. "
                "Consider removing from rule signals or flipping direction.",
            ))

    # --- Bucket calibration ---
    if "priority_bucket" in traded.columns:
        bucket_perf = traded.groupby("priority_bucket")["net_return"].mean()
        high = bucket_perf.get("high", None)
        low  = bucket_perf.get("low",  None)
        if high is not None and low is not None and high <= low:
            recs.append((
                "HIGH",
                f"High-bucket avg ({high:+.2%}) ≤ low-bucket avg ({low:+.2%}). "
                "Ranking thresholds or score weights in ranking_service.py need recalibration.",
            ))

    # --- Rank effectiveness ---
    rank1_avg  = traded[traded["cluster_rank"] == 1]["net_return"].mean()
    rank2p_avg = traded[traded["cluster_rank"] > 1]["net_return"].mean()
    if not pd.isna(rank1_avg) and not pd.isna(rank2p_avg):
        gap = rank1_avg - rank2p_avg
        if gap < 0.002:
            recs.append((
                "MEDIUM",
                f"Rank-1 avg ({rank1_avg:+.2%}) ≈ rank-2+ avg ({rank2p_avg:+.2%}). "
                "The intra-cluster ranking adds little value — review assign_ranks() logic.",
            ))

    # --- Skip rate ---
    skip_rate = (df["traded"] == False).mean()
    if skip_rate > 0.5:
        recs.append((
            "LOW",
            f"{skip_rate:.0%} of signals are skipped. Consider raising max_positions "
            "or reducing holding_period to improve signal utilisation.",
        ))

    # --- Win rate ---
    win_rate = traded["win"].mean()
    if win_rate < 0.45:
        recs.append((
            "HIGH",
            f"Overall win rate is only {win_rate:.0%}. "
            "Consider adding a minimum rank_score filter (--min-score 0.55) "
            "to trade only high-conviction signals.",
        ))

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recs.sort(key=lambda r: priority_order.get(r[0], 99))

    return [f"[{p}]  {msg}" for p, msg in recs]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="News2Alpha error analysis.")
    parser.add_argument("--hold",        type=int,   default=5)
    parser.add_argument("--max-pos",     type=int,   default=10)
    parser.add_argument("--cost-bps",    type=float, default=10.0)
    parser.add_argument("--min-score",   type=float, default=0.0)
    parser.add_argument("--all-signals", action="store_true")
    parser.add_argument("--top",         type=int,   default=5,
                        help="Number of worst performers to show per category.")
    parser.add_argument("--output",      type=str,   default=None,
                        help="Save enriched signal table to CSV.")
    args = parser.parse_args()

    cfg = BacktestConfig(
        holding_period=args.hold,
        max_positions=args.max_pos,
        transaction_cost_bps=args.cost_bps,
        one_position_per_asset=True,
        surface_only=not args.all_signals,
        min_rank_score=args.min_score,
    )

    if not _DB_PATH.exists():
        print(f"Database not found: {_DB_PATH}")
        sys.exit(1)

    print("Loading signal metadata...")
    meta = load_signal_metadata(_DB_PATH)
    print(f"  {len(meta)} signals loaded.")

    print("Running backtest...")
    df = build_backtest_df(meta, cfg, _DB_PATH)
    traded_n = (df["traded"] == True).sum()
    print(f"  {traded_n} trades executed.")

    print(f"\n{_SEP2}")
    print("  NEWS2ALPHA — ERROR ANALYSIS")
    print(_SEP2)

    _worst_by(df, "asset",       f"WORST ASSETS (top {args.top}, min 3 trades)",         args.top)
    _worst_by(df, "primary_rule",f"WORST RULES (top {args.top}, min 3 trades)",           args.top)
    _worst_by(df, "event_type",  f"WORST EVENT TYPES (top {args.top}, min 3 trades)",     args.top)
    _worst_by(df, "direction",   f"WORST DIRECTIONS (top {args.top}, min 1 trade)", args.top, 1)

    _tail_loss_trades(df, args.top)
    _skip_analysis(df)
    _flag_loss_analysis(df)

    recs = _generate_recommendations(df)
    print(f"\n{'RECOMMENDATIONS':{len(_SEP) - 1}─}")
    if recs:
        for rec in recs:
            # Word-wrap at 72 chars
            words = rec.split()
            lines: list[str] = []
            line = "  "
            for word in words:
                if len(line) + len(word) + 1 > 72:
                    lines.append(line)
                    line = "    " + word
                else:
                    line += (" " if line.strip() else "") + word
            lines.append(line)
            print("\n".join(lines))
            print()
    else:
        print("  No significant issues detected.")

    if args.output:
        out = df.copy()
        out["matched_rule_ids"] = out["matched_rule_ids"].apply(str)
        out["context_flags"]    = out["context_flags"].apply(str)
        out["score_components"] = out["score_components"].apply(str)
        out["ranking_reasons"]  = out["ranking_reasons"].apply(str)
        out.to_csv(args.output, index=False)
        print(f"Full signal table saved to: {args.output}")

    print(f"\n{_SEP2}\n")


if __name__ == "__main__":
    main()
