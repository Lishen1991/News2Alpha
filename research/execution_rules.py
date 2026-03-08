"""
Execution rules — deterministic entry/exit logic for the backtest.

All functions are pure (no DB access, no side effects) and operate on
pandas Series of daily prices. This makes them easy to unit-test and
replace independently of the portfolio engine.

Key design decisions
────────────────────
- Entry: next available trading day after the signal date (T+1).
  This avoids look-ahead bias on same-day prices.
- Exit: exactly `holding_period` trading days after entry.
- Transaction cost: applied as a flat round-trip deduction in basis
  points (bps), so cost = (entry_price + exit_price) × (bps / 10000 / 2)
  expressed as a fraction of entry price.
- Direction: short positions have their gross return sign flipped before
  cost is deducted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """
    All tunable parameters for the portfolio backtest.

    Attributes:
        holding_period:      Number of trading days to hold each position.
        max_positions:       Maximum number of concurrent open positions.
        transaction_cost_bps: Round-trip transaction cost in basis points.
        one_position_per_asset: If True, skip a signal if the asset already
                              has an open position.
        surface_only:        If True, only trade ideas with should_surface=True.
        min_rank_score:      Minimum rank_score to consider a signal.
    """
    holding_period: int          = 5
    max_positions: int           = 10
    transaction_cost_bps: float  = 10.0    # 10 bps round-trip ≈ typical liquid futures
    one_position_per_asset: bool = True
    surface_only: bool           = True
    min_rank_score: float        = 0.0     # set > 0 to filter weak signals


DEFAULT_CONFIG = BacktestConfig()


# ---------------------------------------------------------------------------
# Price lookup helpers
# ---------------------------------------------------------------------------
def _next_available_date(prices: pd.Series, after: date) -> date | None:
    """
    Return the first trading date strictly after `after` in the price series.
    Returns None if no such date exists.
    """
    ts = pd.Timestamp(after)
    future = prices[prices.index > ts]
    return future.index[0].date() if not future.empty else None


def _price_on_or_after(prices: pd.Series, target: date) -> tuple[date, float] | None:
    """
    Return (actual_date, price) for the first available date >= target.
    Returns None if the price series ends before target.
    """
    ts = pd.Timestamp(target)
    subset = prices[prices.index >= ts]
    if subset.empty:
        return None
    return subset.index[0].date(), float(subset.iloc[0])


def _price_n_days_later(
    prices: pd.Series,
    entry_date: date,
    n: int,
) -> tuple[date, float] | None:
    """
    Return (exit_date, exit_price) exactly n trading days after entry_date.
    Returns None if fewer than n trading days remain in the price series.
    """
    ts = pd.Timestamp(entry_date)
    future = prices[prices.index > ts]
    if len(future) < n:
        return None
    return future.index[n - 1].date(), float(future.iloc[n - 1])


# ---------------------------------------------------------------------------
# Trade result
# ---------------------------------------------------------------------------
@dataclass
class TradeResult:
    """Outcome of one executed trade."""
    idea_id: int
    asset: str
    direction: str
    signal_date: date
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    gross_return: float    # direction-adjusted, before costs
    net_return: float      # after transaction costs
    cost_fraction: float   # transaction cost as fraction of entry price


@dataclass
class SkippedSignal:
    """A signal that was considered but not traded, with a reason."""
    idea_id: int
    asset: str
    direction: str
    signal_date: date
    reason: str            # e.g. "max_positions", "asset_already_open", "no_price_data"


# ---------------------------------------------------------------------------
# Core execution logic
# ---------------------------------------------------------------------------
def try_execute(
    idea_id: int,
    asset: str,
    direction: str,
    signal_date: date,
    prices: pd.Series,
    cfg: BacktestConfig,
) -> tuple[TradeResult, None] | tuple[None, str]:
    """
    Attempt to construct a TradeResult for one signal.

    Does NOT check portfolio-level constraints (open positions, max_positions).
    Those are enforced by the portfolio engine. This function only handles
    price data availability and arithmetic.

    Returns:
        (TradeResult, None) on success.
        (None, reason_string) if execution is not possible.
    """
    # Entry: next trading day after signal
    entry_date = _next_available_date(prices, signal_date)
    if entry_date is None:
        return None, "no_price_data_entry"

    entry_pair = _price_on_or_after(prices, entry_date)
    if entry_pair is None:
        return None, "no_price_data_entry"
    entry_date, entry_price = entry_pair

    # Exit: holding_period trading days after entry
    exit_pair = _price_n_days_later(prices, entry_date, cfg.holding_period)
    if exit_pair is None:
        return None, "insufficient_history_for_exit"

    exit_date, exit_price = exit_pair

    # Gross return (direction-adjusted)
    raw_return = (exit_price - entry_price) / entry_price
    gross_return = raw_return if direction == "long" else -raw_return

    # Transaction cost (round-trip, symmetric)
    cost_fraction = cfg.transaction_cost_bps / 10_000
    net_return = gross_return - cost_fraction

    return TradeResult(
        idea_id=idea_id,
        asset=asset,
        direction=direction,
        signal_date=signal_date,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_return=round(gross_return, 6),
        net_return=round(net_return, 6),
        cost_fraction=round(cost_fraction, 6),
    ), None
