"""
Portfolio backtest / event replay script.

Loads ranked signals from SQLite, replays them chronologically through the
portfolio engine, and prints a full performance report.

Usage:
    # Run with defaults (5-day hold, 10 max positions, 10bps cost)
    py research/replay_backtest.py

    # 3-day holding period
    py research/replay_backtest.py --hold 3

    # Increase max positions and lower cost
    py research/replay_backtest.py --max-pos 20 --cost-bps 5

    # Include suppressed signals (ignore should_surface filter)
    py research/replay_backtest.py --all-signals

    # Only trade high-priority signals (rank_score >= 0.65)
    py research/replay_backtest.py --min-score 0.65

    # Allow multiple positions in the same asset
    py research/replay_backtest.py --multi-asset

    # Save trade log to CSV
    py research/replay_backtest.py --output trades.csv
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.execution_rules import BacktestConfig, SkippedSignal, TradeResult
from research.portfolio_engine import RankedSignal, run_backtest

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "news2alpha.db"
_SEP  = "─" * 72
_SEP2 = "═" * 72


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------
def load_signals(db_path: Path, surface_only: bool) -> list[RankedSignal]:
    """
    Load ranked signals from SQLite by joining signal_rankings →
    cluster_trade_ideas (for created_at) and applying the surface filter.

    Returns signals sorted by (signal_date, rank_score desc).
    """
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run the API and call /analyze-article, then run:")
        print("  py research/analyze_clusters.py")
        print("  py research/build_market_context.py")
        print("  py research/build_signal_rankings.py")
        sys.exit(1)

    query = """
        SELECT
            sr.cluster_trade_idea_id   AS idea_id,
            sr.asset,
            sr.direction,
            sr.rank_score,
            sr.should_surface,
            cti.created_at             AS signal_created_at
        FROM signal_rankings sr
        JOIN cluster_trade_ideas cti ON cti.id = sr.cluster_trade_idea_id
        ORDER BY signal_created_at ASC, sr.rank_score DESC
    """

    with sqlite3.connect(db_path) as con:
        rows = con.execute(query).fetchall()

    signals: list[RankedSignal] = []
    for row in rows:
        idea_id, asset, direction, rank_score, should_surface, created_at_str = row

        if surface_only and not should_surface:
            continue

        try:
            dt = datetime.fromisoformat(created_at_str)
            signal_date = dt.date()
        except Exception:
            logger.warning("Could not parse created_at for idea_id=%s, skipping.", idea_id)
            continue

        signals.append(RankedSignal.from_row(
            idea_id=int(idea_id),
            asset=asset,
            direction=direction,
            signal_date=signal_date,
            rank_score=float(rank_score),
            should_surface=bool(should_surface),
        ))

    return signals


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _compute_equity_curve(trades: list[TradeResult]) -> pd.Series:
    """
    Build a daily equity index starting at 1.0, updated on trade exit dates.

    Each trade contributes its net_return as an equal-weight increment.
    Multiple trades closing on the same day are averaged.
    """
    if not trades:
        return pd.Series(dtype=float)

    exits = pd.DataFrame([
        {"date": t.exit_date, "net_return": t.net_return}
        for t in trades
    ])
    daily = exits.groupby("date")["net_return"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    equity = (1 + daily).cumprod()
    equity.iloc[0] = 1 + daily.iloc[0]  # already set by cumprod — for clarity
    return equity


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the equity curve."""
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    return float(drawdowns.min())


def _sharpe(returns: list[float], risk_free: float = 0.0) -> float | None:
    """
    Annualised Sharpe ratio (sqrt(252) scaling, assumes daily frequency).
    Returns None if fewer than 2 returns.
    """
    if len(returns) < 2:
        return None
    s = pd.Series(returns)
    excess = s - risk_free
    if excess.std() == 0:
        return None
    return round(float(excess.mean() / excess.std() * (252 ** 0.5)), 3)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _pct(v: float) -> str:
    return f"{v:+.2%}"


def print_report(
    trades: list[TradeResult],
    skipped: list[SkippedSignal],
    cfg: BacktestConfig,
    output_path: Path | None,
) -> None:
    total_signals = len(trades) + len(skipped)
    print(f"\n{_SEP2}")
    print("  NEWS2ALPHA — PORTFOLIO BACKTEST REPORT")
    print(_SEP2)
    print(f"  holding_period      : {cfg.holding_period}d")
    print(f"  max_positions       : {cfg.max_positions}")
    print(f"  transaction_cost    : {cfg.transaction_cost_bps:.1f} bps")
    print(f"  one_pos_per_asset   : {cfg.one_position_per_asset}")
    print(f"  surface_only        : {cfg.surface_only}")
    print(f"  min_rank_score      : {cfg.min_rank_score}")
    print(_SEP2)

    if not trades:
        print("\n  No trades were executed.")
        _print_skip_summary(skipped)
        return

    net_returns = [t.net_return for t in trades]
    wins = [r for r in net_returns if r > 0]

    total_return    = sum(net_returns)
    avg_return      = total_return / len(net_returns)
    median_return   = sorted(net_returns)[len(net_returns) // 2]
    win_rate        = len(wins) / len(net_returns)
    sharpe          = _sharpe(net_returns)
    equity          = _compute_equity_curve(trades)
    mdd             = _max_drawdown(equity)

    print(f"\n{'OVERALL PERFORMANCE':─<{_SEP.__len__() - 1}}")
    print(f"  Signals considered  : {total_signals}")
    print(f"  Trades entered      : {len(trades)}")
    print(f"  Trades skipped      : {len(skipped)}")
    print(f"  Win rate            : {win_rate:.1%}")
    print(f"  Avg net return      : {_pct(avg_return)}")
    print(f"  Median net return   : {_pct(median_return)}")
    print(f"  Total return        : {_pct(total_return)}")
    print(f"  Max drawdown        : {_pct(mdd)}")
    if sharpe is not None:
        print(f"  Sharpe (ann.)       : {sharpe:.3f}")
    else:
        print(f"  Sharpe (ann.)       : n/a (< 2 trades)")

    # --- Per-asset breakdown ---
    print(f"\n{'BY ASSET':─<{_SEP.__len__() - 1}}")
    df = pd.DataFrame([{
        "asset":      t.asset,
        "direction":  t.direction,
        "net_return": t.net_return,
        "win":        t.net_return > 0,
    } for t in trades])

    by_asset = (
        df.groupby("asset")
        .agg(
            trades=("net_return", "count"),
            avg_ret=("net_return", "mean"),
            win_rate=("win", "mean"),
        )
        .sort_values("avg_ret", ascending=False)
    )
    print(f"  {'Asset':<28}  {'Trades':>6}  {'Avg ret':>9}  {'Win rate':>9}")
    print(f"  {'─'*28}  {'─'*6}  {'─'*9}  {'─'*9}")
    for asset, row in by_asset.iterrows():
        print(
            f"  {str(asset):<28}  {int(row['trades']):>6}  "
            f"{_pct(row['avg_ret']):>9}  {row['win_rate']:.0%}  "
        )

    # --- Per-direction breakdown ---
    print(f"\n{'BY DIRECTION':─<{_SEP.__len__() - 1}}")
    by_dir = (
        df.groupby("direction")
        .agg(trades=("net_return", "count"), avg_ret=("net_return", "mean"), win_rate=("win", "mean"))
    )
    for direction, row in by_dir.iterrows():
        print(
            f"  {str(direction).upper():<6}  trades={int(row['trades'])}  "
            f"avg_ret={_pct(row['avg_ret'])}  win_rate={row['win_rate']:.0%}"
        )

    # --- Equity curve (text sparkline) ---
    if not equity.empty:
        print(f"\n{'EQUITY CURVE (exit-date resolution)':─<{_SEP.__len__() - 1}}")
        for dt, val in equity.items():
            bar = "█" * max(1, int((val - equity.min()) / max(equity.max() - equity.min(), 0.001) * 20))
            print(f"  {str(dt.date()):<12}  {val:6.3f}  {bar}")

    _print_skip_summary(skipped)

    # --- Trade log ---
    print(f"\n{'TRADE LOG':─<{_SEP.__len__() - 1}}")
    print(
        f"  {'ID':>4}  {'Asset':<26}  {'Dir':<5}  {'Entry':>10}  "
        f"{'Exit':>10}  {'Gross':>8}  {'Net':>8}"
    )
    for t in sorted(trades, key=lambda x: x.entry_date):
        print(
            f"  {t.idea_id:>4}  {t.asset:<26}  {t.direction:<5}  "
            f"{str(t.entry_date):>10}  {str(t.exit_date):>10}  "
            f"{_pct(t.gross_return):>8}  {_pct(t.net_return):>8}"
        )

    print(f"\n{_SEP2}\n")

    # --- Optional CSV export ---
    if output_path:
        rows = [{
            "idea_id":      t.idea_id,
            "asset":        t.asset,
            "direction":    t.direction,
            "signal_date":  t.signal_date,
            "entry_date":   t.entry_date,
            "exit_date":    t.exit_date,
            "entry_price":  t.entry_price,
            "exit_price":   t.exit_price,
            "gross_return": t.gross_return,
            "net_return":   t.net_return,
        } for t in trades]
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Trade log saved to: {output_path}")


def _print_skip_summary(skipped: list[SkippedSignal]) -> None:
    if not skipped:
        return
    from collections import Counter
    reasons = Counter(s.reason for s in skipped)
    print(f"\n{'SKIPPED SIGNALS':─<{_SEP.__len__() - 1}}")
    print(f"  Total skipped: {len(skipped)}")
    for reason, count in reasons.most_common():
        print(f"    {reason:<35} {count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ranked signals as a portfolio backtest.")
    parser.add_argument("--hold",       type=int,   default=5,    help="Holding period in trading days (default: 5).")
    parser.add_argument("--max-pos",    type=int,   default=10,   help="Max concurrent positions (default: 10).")
    parser.add_argument("--cost-bps",   type=float, default=10.0, help="Round-trip transaction cost in bps (default: 10).")
    parser.add_argument("--min-score",  type=float, default=0.0,  help="Min rank_score to trade (default: 0).")
    parser.add_argument("--all-signals",action="store_true",      help="Include suppressed signals (ignore should_surface).")
    parser.add_argument("--multi-asset",action="store_true",      help="Allow multiple positions per asset.")
    parser.add_argument("--output",     type=str,   default=None, help="Save trade log to CSV file.")
    args = parser.parse_args()

    cfg = BacktestConfig(
        holding_period=args.hold,
        max_positions=args.max_pos,
        transaction_cost_bps=args.cost_bps,
        one_position_per_asset=not args.multi_asset,
        surface_only=not args.all_signals,
        min_rank_score=args.min_score,
    )

    signals = load_signals(_DB_PATH, surface_only=cfg.surface_only)
    if not signals:
        print("No ranked signals found. Run the full pipeline first:")
        print("  py research/analyze_clusters.py")
        print("  py research/build_market_context.py")
        print("  py research/build_signal_rankings.py")
        return

    print(f"Loaded {len(signals)} signal(s). Running backtest...")
    trades, skipped = run_backtest(signals, cfg)

    output_path = Path(args.output) if args.output else None
    print_report(trades, skipped, cfg, output_path)


if __name__ == "__main__":
    main()
