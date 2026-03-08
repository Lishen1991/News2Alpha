"""
Performance diagnostics for News2Alpha backtest results.

Loads all signal metadata from SQLite, runs the portfolio backtest,
joins trade outcomes to signal metadata, and prints attribution tables
broken down by event type, rule, asset, direction, and priority bucket.

This module is also the shared data layer for attribution.py and
error_analysis.py — both import load_signal_metadata() and
build_backtest_df() from here.

Usage:
    # Full report with default backtest settings
    py research/diagnostics.py

    # Shorter holding period
    py research/diagnostics.py --hold 3

    # Include suppressed signals
    py research/diagnostics.py --all-signals

    # Save full enriched table to CSV for further analysis
    py research/diagnostics.py --output diagnostics.csv

Example output:
    BY EVENT TYPE───────────────────────────────────────────
      Group                           N   Win%   Avg ret   Median  Total PnL
      geopolitical_conflict          85    58%    +0.42%   +0.31%    +35.4%
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.execution_rules import BacktestConfig
from research.portfolio_engine import RankedSignal, run_backtest
from research.replay_backtest import load_signals

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

_DB_PATH = Path(__file__).parent.parent / "news2alpha.db"
_SEP  = "─" * 72
_SEP2 = "═" * 72

# Context flags stored in market_context.context_flags as a JSON list of strings.
_CONTEXT_FLAGS = [
    "already_moved",
    "stale_signal",
    "high_volatility",
    "insufficient_price_data",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_signal_metadata(db_path: Path) -> pd.DataFrame:
    """
    Load all signal metadata from SQLite, joining:
        signal_rankings → cluster_trade_ideas → cluster_events → market_context

    Returns a DataFrame with one row per signal including event metadata,
    ranking attributes, and market-context flags.

    Signals without a market_context row still appear (LEFT JOIN); their
    context columns will be NaN.
    """
    query = """
        SELECT
            sr.cluster_trade_idea_id  AS idea_id,
            sr.asset,
            sr.direction,
            sr.rank_score,
            sr.rank                   AS cluster_rank,
            sr.priority_bucket,
            sr.should_surface,
            sr.score_components,
            sr.ranking_reasons,
            cti.created_at            AS signal_created_at,
            ce.event_type,
            ce.severity,
            ce.novelty,
            ce.matched_rule_ids,
            mc.adjusted_confidence,
            mc.original_confidence,
            mc.context_flags,
            mc.ret_1d_pre_signal,
            mc.ret_5d_pre_signal,
            mc.realized_vol_5d,
            mc.staleness_days,
            mc.should_surface         AS mc_should_surface
        FROM signal_rankings sr
        JOIN cluster_trade_ideas cti ON cti.id = sr.cluster_trade_idea_id
        JOIN cluster_events ce       ON ce.id  = cti.cluster_event_id
        LEFT JOIN market_context mc  ON mc.cluster_trade_idea_id = sr.cluster_trade_idea_id
        ORDER BY cti.created_at ASC, sr.rank_score DESC
    """
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(query, con)

    # --- Parse JSON columns ---
    def _parse_json(val, default):
        if val is None:
            return default
        try:
            return json.loads(val)
        except (TypeError, json.JSONDecodeError):
            return default

    df["matched_rule_ids"]  = df["matched_rule_ids"].apply(lambda v: _parse_json(v, []))
    df["context_flags"]     = df["context_flags"].apply(lambda v: _parse_json(v, []))
    df["score_components"]  = df["score_components"].apply(lambda v: _parse_json(v, {}))
    df["ranking_reasons"]   = df["ranking_reasons"].apply(lambda v: _parse_json(v, []))

    # --- Derived columns ---
    # Primary rule (first in matched list; "no_rule" if empty)
    df["primary_rule"] = df["matched_rule_ids"].apply(
        lambda r: r[0] if r else "no_rule"
    )

    # Expand context flags into individual boolean columns for easy groupby
    for flag in _CONTEXT_FLAGS:
        df[f"flag_{flag}"] = df["context_flags"].apply(lambda f, k=flag: k in f)

    # Whether this signal is rank 1 in its cluster (top-ranked idea)
    df["is_top_ranked"] = df["cluster_rank"] == 1

    return df


# ---------------------------------------------------------------------------
# Backtest runner + join
# ---------------------------------------------------------------------------

def build_backtest_df(
    meta_df: pd.DataFrame,
    cfg: BacktestConfig,
    db_path: Path,
) -> pd.DataFrame:
    """
    Run the portfolio backtest and join trade outcomes to signal metadata.

    Returns a copy of meta_df with additional columns:
        traded        bool   — True if a trade was entered
        entry_date    date | NaT
        exit_date     date | NaT
        gross_return  float | NaN
        net_return    float | NaN
        win           bool | NaN
        skip_reason   str | NaN  — set for skipped signals

    Signals not considered by the backtest (filtered before run_backtest)
    will have traded=False and skip_reason="not_loaded".
    """
    signals = load_signals(db_path, surface_only=cfg.surface_only)
    if not signals:
        df = meta_df.copy()
        df["traded"] = False
        df["skip_reason"] = "not_loaded"
        for col in ("entry_date", "exit_date", "gross_return", "net_return", "win"):
            df[col] = float("nan")
        return df

    # Apply min_rank_score filter that the engine would apply
    if cfg.min_rank_score > 0:
        signals = [s for s in signals if s.rank_score >= cfg.min_rank_score]

    trades, skipped = run_backtest(signals, cfg)

    trade_rows = [
        {
            "idea_id":      t.idea_id,
            "traded":       True,
            "skip_reason":  None,
            "entry_date":   t.entry_date,
            "exit_date":    t.exit_date,
            "gross_return": t.gross_return,
            "net_return":   t.net_return,
            "win":          t.net_return > 0,
        }
        for t in trades
    ]
    skip_rows = [
        {
            "idea_id":      s.idea_id,
            "traded":       False,
            "skip_reason":  s.reason,
            "entry_date":   None,
            "exit_date":    None,
            "gross_return": float("nan"),
            "net_return":   float("nan"),
            "win":          float("nan"),
        }
        for s in skipped
    ]

    outcome_df = pd.DataFrame(trade_rows + skip_rows)
    merged = meta_df.merge(outcome_df, on="idea_id", how="left")

    # Fill signals not considered by the loader (e.g. should_surface=False)
    merged["traded"] = merged["traded"].fillna(False)
    merged["skip_reason"] = merged["skip_reason"].fillna("not_loaded")

    return merged


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _pct(v) -> str:
    if pd.isna(v):
        return "   n/a"
    return f"{v:+.2%}"


def print_grouped_report(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    min_trades: int = 1,
) -> None:
    """
    Print a summary performance table grouped by a single column.

    Only rows where traded=True are included in trade-level metrics.
    Groups with fewer than min_trades executed trades are excluded.
    """
    traded = df[df["traded"] == True].copy()
    if traded.empty:
        print(f"\n{title:{len(_SEP) - 1}─}")
        print("  No traded signals.")
        return

    grp = (
        traded.groupby(group_col)
        .agg(
            n_trades    =("idea_id",     "count"),
            win_rate    =("win",         "mean"),
            avg_return  =("net_return",  "mean"),
            med_return  =("net_return",  "median"),
            total_pnl   =("net_return",  "sum"),
        )
        .query(f"n_trades >= {min_trades}")
        .sort_values("avg_return", ascending=False)
    )

    print(f"\n{title:{len(_SEP) - 1}─}")
    print(
        f"  {'Group':<30}  {'Trades':>6}  {'Win%':>5}  "
        f"{'Avg ret':>8}  {'Median':>8}  {'Total PnL':>10}"
    )
    print(f"  {'─'*30}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}")
    for grp_val, row in grp.iterrows():
        print(
            f"  {str(grp_val):<30}  {int(row['n_trades']):>6}  "
            f"{row['win_rate']:>5.0%}  "
            f"{_pct(row['avg_return']):>8}  "
            f"{_pct(row['med_return']):>8}  "
            f"{_pct(row['total_pnl']):>10}"
        )


def print_overview(df: pd.DataFrame, cfg: BacktestConfig) -> None:
    """Print a top-level summary of the diagnostic run."""
    traded = df[df["traded"] == True]
    skipped = df[df["traded"] == False]

    print(f"\n{_SEP2}")
    print("  NEWS2ALPHA — PERFORMANCE DIAGNOSTICS")
    print(_SEP2)
    print(f"  holding_period   : {cfg.holding_period}d")
    print(f"  max_positions    : {cfg.max_positions}")
    print(f"  cost_bps         : {cfg.transaction_cost_bps:.1f}")
    print(f"  surface_only     : {cfg.surface_only}")
    print(f"  min_rank_score   : {cfg.min_rank_score}")
    print(_SEP2)
    print(f"  Total signals    : {len(df)}")
    print(f"  Trades entered   : {len(traded)}")
    print(f"  Signals skipped  : {len(skipped)}")

    if not traded.empty:
        nr = traded["net_return"]
        print(f"  Win rate         : {traded['win'].mean():.1%}")
        print(f"  Avg net return   : {nr.mean():+.2%}")
        print(f"  Total PnL        : {nr.sum():+.2%}")
    print(_SEP2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="News2Alpha performance diagnostics.")
    parser.add_argument("--hold",        type=int,   default=5)
    parser.add_argument("--max-pos",     type=int,   default=10)
    parser.add_argument("--cost-bps",    type=float, default=10.0)
    parser.add_argument("--min-score",   type=float, default=0.0)
    parser.add_argument("--all-signals", action="store_true",
                        help="Include suppressed signals.")
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

    print_overview(df, cfg)
    print_grouped_report(df, "event_type",      "BY EVENT TYPE")
    print_grouped_report(df, "primary_rule",    "BY RULE")
    print_grouped_report(df, "asset",           "BY ASSET")
    print_grouped_report(df, "direction",       "BY DIRECTION")
    print_grouped_report(df, "priority_bucket", "BY PRIORITY BUCKET")

    if args.output:
        # Flatten list columns to strings for CSV compatibility
        out = df.copy()
        out["matched_rule_ids"] = out["matched_rule_ids"].apply(str)
        out["context_flags"]    = out["context_flags"].apply(str)
        out["score_components"] = out["score_components"].apply(str)
        out["ranking_reasons"]  = out["ranking_reasons"].apply(str)
        out.to_csv(args.output, index=False)
        print(f"\nEnriched signal table saved to: {args.output}")

    print(f"\n{_SEP2}\n")


if __name__ == "__main__":
    main()
