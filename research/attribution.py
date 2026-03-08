"""
Attribution analysis for News2Alpha signal and trade performance.

Answers questions like:
  - Which rules are generating alpha, which are noise?
  - Does rank 1 in a cluster outperform rank 2+?
  - Do high-priority signals beat medium/low?
  - Which context flags help or hurt performance?
  - Which score components (confidence, severity, novelty) predict returns?

Imports the shared data layer from diagnostics.py.

Usage:
    # Full attribution report
    py research/attribution.py

    # With custom backtest settings
    py research/attribution.py --hold 3 --all-signals

    # Save each section as CSV
    py research/attribution.py --output-dir research/reports/

Example output:
    RANK EFFECTIVENESS───────────────────────────────────────────
      Group          Trades   Win%   Avg ret   Median   Total PnL
      rank_1            42    64%    +1.12%   +0.84%     +47.0%
      rank_2+           96    51%    +0.28%   +0.11%     +26.9%

    What this tells you:
      Top-ranked ideas within a cluster outperform lower-ranked ones.
      If the gap is small, the ranking model adds little value.
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

_BUCKET_ORDER = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _trade_stats(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Aggregate traded signals by label_col.

    Returns a DataFrame with columns:
        n_trades, win_rate, avg_return, med_return, total_pnl
    sorted by avg_return descending.
    """
    traded = df[df["traded"] == True]
    if traded.empty:
        return pd.DataFrame()

    return (
        traded.groupby(label_col)
        .agg(
            n_trades   =("idea_id",    "count"),
            win_rate   =("win",        "mean"),
            avg_return =("net_return", "mean"),
            med_return =("net_return", "median"),
            total_pnl  =("net_return", "sum"),
        )
        .sort_values("avg_return", ascending=False)
    )


def _print_stats(stats: pd.DataFrame, title: str, explanation: str) -> None:
    """Print a formatted stats table with an explanation footer."""
    print(f"\n{title:{len(_SEP) - 1}─}")
    if stats.empty:
        print("  No data.")
        return

    print(
        f"  {'Group':<28}  {'Trades':>6}  {'Win%':>5}  "
        f"{'Avg ret':>8}  {'Median':>8}  {'Total PnL':>10}"
    )
    print(f"  {'─'*28}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}")
    for grp_val, row in stats.iterrows():
        print(
            f"  {str(grp_val):<28}  {int(row['n_trades']):>6}  "
            f"{row['win_rate']:>5.0%}  "
            f"{_pct(row['avg_return']):>8}  "
            f"{_pct(row['med_return']):>8}  "
            f"{_pct(row['total_pnl']):>10}"
        )
    print(f"  ↳ {explanation}")


# ---------------------------------------------------------------------------
# Attribution sections
# ---------------------------------------------------------------------------

def rule_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performance broken down by which rule fired.

    Tells you which trading rules are generating alpha vs noise.
    If a rule consistently underperforms, review its asset/direction mappings
    or tighten its conditions.
    """
    stats = _trade_stats(df, "primary_rule")
    _print_stats(
        stats,
        "RULE ATTRIBUTION",
        "Which rules generate alpha? Weak rules should be tightened or asset lists revised.",
    )
    return stats


def rank_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare rank-1 ideas within each cluster vs rank-2+ ideas.

    If rank-1 consistently outperforms, the ranking model is adding value.
    A small gap suggests the ranking weights need tuning.
    """
    df = df.copy()
    df["rank_group"] = df["cluster_rank"].apply(
        lambda r: "rank_1" if r == 1 else "rank_2+"
    )
    stats = _trade_stats(df, "rank_group")
    _print_stats(
        stats,
        "RANK EFFECTIVENESS (rank 1 vs 2+)",
        "Large gap → ranking adds value. Small gap → review ranking weights.",
    )
    return stats


def bucket_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performance by priority bucket: high / medium / low.

    The scoring model assigns ideas to buckets based on rank_score thresholds.
    This shows whether the bucketing correlates with actual trade returns.
    """
    df = df.copy()
    df["_bucket_order"] = df["priority_bucket"].map(_BUCKET_ORDER)
    traded = df[df["traded"] == True].copy()

    if traded.empty:
        print(f"\n{'PRIORITY BUCKET ATTRIBUTION':{len(_SEP) - 1}─}")
        print("  No data.")
        return pd.DataFrame()

    stats = (
        traded.groupby("priority_bucket")
        .agg(
            n_trades   =("idea_id",    "count"),
            win_rate   =("win",        "mean"),
            avg_return =("net_return", "mean"),
            med_return =("net_return", "median"),
            total_pnl  =("net_return", "sum"),
            avg_score  =("rank_score", "mean"),
        )
        .sort_values("avg_score", ascending=False)
    )

    print(f"\n{'PRIORITY BUCKET ATTRIBUTION':{len(_SEP) - 1}─}")
    print(
        f"  {'Bucket':<12}  {'Trades':>6}  {'Win%':>5}  "
        f"{'Avg ret':>8}  {'Median':>8}  {'Total PnL':>10}  {'Avg score':>10}"
    )
    print(f"  {'─'*12}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
    for bucket, row in stats.iterrows():
        print(
            f"  {str(bucket):<12}  {int(row['n_trades']):>6}  "
            f"{row['win_rate']:>5.0%}  "
            f"{_pct(row['avg_return']):>8}  "
            f"{_pct(row['med_return']):>8}  "
            f"{_pct(row['total_pnl']):>10}  "
            f"{row['avg_score']:>10.3f}"
        )
    print("  ↳ If high-bucket returns ≤ low-bucket, the scoring thresholds need recalibration.")
    return stats


def flag_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare traded performance for signals with vs without each context flag.

    Shows whether market-context filters are correctly identifying bad signals.
    For example: if already_moved signals still get traded and underperform,
    the filter threshold is too permissive.

    Note: flags only appear if should_surface filtering is off (--all-signals),
    since the market context filter suppresses flagged signals by default.
    """
    flags = [
        ("flag_already_moved",           "already_moved"),
        ("flag_stale_signal",            "stale_signal"),
        ("flag_high_volatility",         "high_volatility"),
        ("flag_insufficient_price_data", "insufficient_price_data"),
    ]

    rows = []
    for col, label in flags:
        if col not in df.columns:
            continue
        for flag_val in [True, False]:
            subset = df[(df[col] == flag_val) & (df["traded"] == True)]
            if subset.empty:
                continue
            rows.append({
                "flag":        label,
                "flagged":     flag_val,
                "n_trades":    len(subset),
                "win_rate":    subset["win"].mean(),
                "avg_return":  subset["net_return"].mean(),
                "med_return":  subset["net_return"].median(),
                "total_pnl":   subset["net_return"].sum(),
            })

    if not rows:
        print(f"\n{'CONTEXT FLAG ATTRIBUTION':{len(_SEP) - 1}─}")
        print("  No flag data (run with --all-signals to include suppressed signals).")
        return pd.DataFrame()

    stats = pd.DataFrame(rows)
    print(f"\n{'CONTEXT FLAG ATTRIBUTION':{len(_SEP) - 1}─}")
    print(
        f"  {'Flag':<28}  {'Flagged':>7}  {'Trades':>6}  {'Win%':>5}  "
        f"{'Avg ret':>8}  {'Median':>8}"
    )
    print(f"  {'─'*28}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*8}")
    for _, row in stats.iterrows():
        print(
            f"  {row['flag']:<28}  {'YES' if row['flagged'] else 'NO':>7}  "
            f"{int(row['n_trades']):>6}  "
            f"{row['win_rate']:>5.0%}  "
            f"{_pct(row['avg_return']):>8}  "
            f"{_pct(row['med_return']):>8}"
        )
    print("  ↳ Flagged signals should underperform unflagged ones. If not, tighten the filter.")
    return stats


def score_component_correlation(df: pd.DataFrame) -> None:
    """
    Show Pearson correlation between each score component and net_return.

    Tells you which components of the ranking formula actually predict
    trade outcomes. Weak correlators are candidates for weight reduction.
    """
    traded = df[df["traded"] == True].copy()
    if traded.empty or traded["score_components"].apply(len).sum() == 0:
        print(f"\n{'SCORE COMPONENT CORRELATIONS':{len(_SEP) - 1}─}")
        print("  No score component data.")
        return

    components = [
        "adjusted_confidence",
        "event_severity",
        "event_novelty",
        "rule_support",
        "context_score",
    ]

    # Expand score_components dict into columns
    for comp in components:
        traded[comp] = traded["score_components"].apply(
            lambda d, k=comp: d.get(k) if isinstance(d, dict) else None
        )

    print(f"\n{'SCORE COMPONENT CORRELATIONS WITH NET RETURN':{len(_SEP) - 1}─}")
    print(f"  {'Component':<28}  {'Corr (Pearson)':>15}  {'Note'}")
    print(f"  {'─'*28}  {'─'*15}  {'─'*30}")

    for comp in components:
        col = traded[comp].dropna()
        if len(col) < 5:
            corr_str = "insufficient data"
            note = ""
        else:
            corr = traded[["net_return", comp]].dropna().corr().iloc[0, 1]
            corr_str = f"{corr:+.3f}"
            if abs(corr) >= 0.15:
                note = "↑ meaningful predictor"
            elif abs(corr) >= 0.05:
                note = "weak predictor"
            else:
                note = "near-zero — consider reducing weight"
        print(f"  {comp:<28}  {corr_str:>15}  {note}")

    print("  ↳ Components near zero should have their weights reduced in ranking_service.py.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="News2Alpha attribution analysis.")
    parser.add_argument("--hold",        type=int,   default=5)
    parser.add_argument("--max-pos",     type=int,   default=10)
    parser.add_argument("--cost-bps",    type=float, default=10.0)
    parser.add_argument("--min-score",   type=float, default=0.0)
    parser.add_argument("--all-signals", action="store_true")
    parser.add_argument("--output-dir",  type=str,   default=None,
                        help="Directory to save per-section CSVs.")
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
    print("  NEWS2ALPHA — ATTRIBUTION ANALYSIS")
    print(_SEP2)

    rule_stats   = rule_attribution(df)
    rank_stats   = rank_attribution(df)
    bucket_stats = bucket_attribution(df)
    flag_stats   = flag_attribution(df)
    score_component_correlation(df)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if not rule_stats.empty:
            rule_stats.to_csv(out / "attribution_by_rule.csv")
        if not rank_stats.empty:
            rank_stats.to_csv(out / "attribution_by_rank.csv")
        if not bucket_stats.empty:
            bucket_stats.to_csv(out / "attribution_by_bucket.csv")
        if not flag_stats.empty:
            flag_stats.to_csv(out / "attribution_by_flag.csv")
        print(f"\nCSVs saved to: {out}/")

    print(f"\n{_SEP2}\n")


if __name__ == "__main__":
    main()
