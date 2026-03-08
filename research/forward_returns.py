"""
Forward return analysis script.

For each trade idea stored in the database, this script:
    1. Looks up the price series for the idea's asset.
    2. Finds the price on the signal date (idea.created_at).
    3. Computes forward returns at 1, 3, and 5 trading days.
    4. Flips the sign for short ideas.
    5. Prints a summary table per horizon.

Usage:
    cd D:\\News2Alpha
    py research/forward_returns.py

    # Filter to a specific asset:
    py research/forward_returns.py --asset xle

    # Verbose output (show every signal row):
    py research/forward_returns.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Allow running the script from the project root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.price_loader import load_all_prices  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DB_PATH = Path(__file__).parent.parent / "news2alpha.db"
_HORIZONS = [1, 3, 5]  # trading days


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    idea_id: int
    asset: str
    direction: str        # 'long' or 'short'
    confidence: float
    signal_date: date


@dataclass
class ForwardReturn:
    idea_id: int
    asset: str
    direction: str
    confidence: float
    signal_date: date
    horizon: int          # days
    raw_return: float     # price-based percentage return
    signed_return: float  # sign-adjusted for direction


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def load_signals(db_path: Path, asset_filter: str | None = None) -> list[Signal]:
    """
    Load trade ideas from the SQLite database.

    Args:
        db_path: Path to news2alpha.db.
        asset_filter: If provided, only return ideas for this asset.

    Returns:
        List of Signal dataclass instances.
    """
    if not db_path.exists():
        logger.error("Database not found at %s. Run the API and call /analyze-article first.", db_path)
        sys.exit(1)

    query = """
        SELECT
            ti.id,
            ti.asset,
            ti.direction,
            ti.confidence,
            ti.created_at
        FROM trade_ideas ti
    """
    params: list = []
    if asset_filter:
        query += " WHERE LOWER(ti.asset) = LOWER(?)"
        params.append(asset_filter)

    query += " ORDER BY ti.created_at ASC"

    with sqlite3.connect(db_path) as con:
        rows = con.execute(query, params).fetchall()

    signals: list[Signal] = []
    for row in rows:
        idea_id, asset, direction, confidence, created_at_str = row
        try:
            # created_at is stored as ISO string e.g. "2026-03-07 10:23:00+00:00"
            signal_dt = datetime.fromisoformat(created_at_str)
            signals.append(Signal(
                idea_id=idea_id,
                asset=asset,
                direction=direction,
                confidence=confidence,
                signal_date=signal_dt.date(),
            ))
        except Exception as exc:
            logger.warning("Skipping idea id=%s — could not parse date '%s': %s", idea_id, created_at_str, exc)

    logger.info("Loaded %d signal(s) from database.", len(signals))
    return signals


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------
def compute_forward_return(
    prices: pd.Series,
    signal_date: date,
    horizon: int,
) -> float | None:
    """
    Compute the percentage price change from signal_date to signal_date + horizon.

    Uses the next available trading day if signal_date is not in the index
    (e.g. weekend or holiday).

    Args:
        prices: Series of daily prices indexed by datetime.
        signal_date: The date the signal was generated.
        horizon: Number of trading days forward.

    Returns:
        Percentage return as a decimal (e.g. 0.034 = 3.4%), or None if
        there are insufficient price observations.
    """
    # Align signal_date to the price index
    signal_ts = pd.Timestamp(signal_date)
    future_prices = prices[prices.index >= signal_ts]

    if len(future_prices) < horizon + 1:
        return None

    entry_price = future_prices.iloc[0]
    exit_price = future_prices.iloc[horizon]

    if entry_price == 0:
        return None

    return (exit_price - entry_price) / entry_price


def compute_all_returns(
    signals: list[Signal],
    price_map: dict[str, pd.Series],
) -> list[ForwardReturn]:
    """
    Compute forward returns for every signal × horizon combination.

    Args:
        signals: List of Signal objects from the database.
        price_map: Dict of asset → price Series.

    Returns:
        List of ForwardReturn objects (one per signal × horizon).
    """
    results: list[ForwardReturn] = []

    for signal in signals:
        prices = price_map.get(signal.asset.lower())
        if prices is None:
            logger.debug("No price data for asset '%s' — skipping.", signal.asset)
            continue

        for horizon in _HORIZONS:
            raw = compute_forward_return(prices, signal.signal_date, horizon)
            if raw is None:
                logger.debug(
                    "Insufficient price history for %s on %s at horizon %dd.",
                    signal.asset, signal.signal_date, horizon,
                )
                continue

            # Flip sign for short ideas: profit when price falls
            signed = raw if signal.direction == "long" else -raw

            results.append(ForwardReturn(
                idea_id=signal.idea_id,
                asset=signal.asset,
                direction=signal.direction,
                confidence=signal.confidence,
                signal_date=signal.signal_date,
                horizon=horizon,
                raw_return=raw,
                signed_return=signed,
            ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _separator(width: int = 72) -> str:
    return "─" * width


def print_summary(returns: list[ForwardReturn], verbose: bool = False) -> None:
    """
    Print a summary table of forward return statistics per horizon.

    Args:
        returns: List of ForwardReturn objects.
        verbose: If True, also print every individual signal row.
    """
    if not returns:
        print("\nNo forward returns to display.")
        print("Check that price CSV files exist in research/prices/ and cover the signal dates.")
        return

    df = pd.DataFrame([
        {
            "idea_id":      r.idea_id,
            "asset":        r.asset,
            "direction":    r.direction,
            "confidence":   r.confidence,
            "signal_date":  r.signal_date,
            "horizon_days": r.horizon,
            "raw_ret":      r.raw_return,
            "signed_ret":   r.signed_return,
        }
        for r in returns
    ])

    if verbose:
        print("\n" + _separator())
        print("INDIVIDUAL SIGNALS")
        print(_separator())
        print(df.to_string(index=False, float_format="{:.4f}".format))

    print("\n" + _separator())
    print("FORWARD RETURN SUMMARY (sign-adjusted for direction)")
    print(_separator())
    print(f"{'Horizon':>10}  {'Signals':>8}  {'Avg Ret':>10}  {'Med Ret':>10}  {'Hit Rate':>10}")
    print(_separator())

    for horizon in _HORIZONS:
        sub = df[df["horizon_days"] == horizon]["signed_ret"]
        if sub.empty:
            print(f"{f'{horizon}d':>10}  {'0':>8}  {'n/a':>10}  {'n/a':>10}  {'n/a':>10}")
            continue
        n         = len(sub)
        avg_ret   = sub.mean()
        med_ret   = sub.median()
        hit_rate  = (sub > 0).mean()
        print(
            f"{f'{horizon}d':>10}  {n:>8}  {avg_ret:>+10.2%}  {med_ret:>+10.2%}  {hit_rate:>10.1%}"
        )

    print(_separator())
    print(f"\nTotal signal × horizon rows evaluated: {len(df)}")

    # Per-asset breakdown
    print("\n" + _separator())
    print("PER-ASSET BREAKDOWN (all horizons combined)")
    print(_separator())
    asset_grp = (
        df.groupby("asset")["signed_ret"]
        .agg(signals="count", avg_ret="mean", hit_rate=lambda x: (x > 0).mean())
        .reset_index()
    )
    print(asset_grp.to_string(index=False, float_format="{:.4f}".format))
    print(_separator())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Forward return analysis for News2Alpha trade ideas.")
    parser.add_argument("--asset", type=str, default=None, help="Filter to a single asset (e.g. xle)")
    parser.add_argument("--verbose", action="store_true", help="Print individual signal rows")
    args = parser.parse_args()

    # 1. Load signals from DB
    signals = load_signals(_DB_PATH, asset_filter=args.asset)
    if not signals:
        print("No signals found in the database. Call /analyze-article first.")
        return

    # 2. Determine which assets we need prices for
    needed_assets = list({s.asset.lower() for s in signals})
    price_map = load_all_prices(needed_assets)

    if not price_map:
        print("\nNo price files found. See instructions below.\n")
        print("Expected location:  research/prices/<asset>.csv")
        print("Expected columns:   date,price")
        print("Example:            2026-01-02,75.32")
        return

    # 3. Compute forward returns
    forward_returns = compute_all_returns(signals, price_map)

    # 4. Print results
    print_summary(forward_returns, verbose=args.verbose)


if __name__ == "__main__":
    main()
