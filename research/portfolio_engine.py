"""
Portfolio engine — processes ranked signals chronologically and manages
open positions according to BacktestConfig constraints.

Design
──────
The engine maintains a simple state machine:
    open_positions: dict[asset, exit_date]
        Tracks which assets are currently held and when they expire.

Signals are processed in signal_date order. On each signal date the engine:
    1. Closes any positions whose exit_date <= today.
    2. Evaluates the new signal against portfolio constraints.
    3. If accepted: calls try_execute() and records a TradeResult.
    4. If rejected: records a SkippedSignal with a reason.

Equal-weight sizing is assumed throughout. Capital tracking is optional
but daily equity is reconstructed from trade-level returns for drawdown
and Sharpe computation.

No look-ahead bias
──────────────────
The signal_date is the date the idea was generated (idea.created_at.date()).
try_execute() enters on the *next* available trading day, never on the
signal date itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator

import pandas as pd

from research.execution_rules import (
    BacktestConfig,
    SkippedSignal,
    TradeResult,
    try_execute,
)
from research.price_loader import load_prices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal input dataclass
# ---------------------------------------------------------------------------
@dataclass(order=True)
class RankedSignal:
    """
    A single ranked trade signal ready for the portfolio engine.
    Sortable by (signal_date, rank_score desc) via the sort_index field.
    """
    sort_index: tuple = field(compare=True, repr=False)   # (signal_date, -rank_score)
    idea_id: int      = field(compare=False)
    asset: str        = field(compare=False)
    direction: str    = field(compare=False)
    signal_date: date = field(compare=False)
    rank_score: float = field(compare=False)
    should_surface: bool = field(compare=False)

    @classmethod
    def from_row(
        cls,
        idea_id: int,
        asset: str,
        direction: str,
        signal_date: date,
        rank_score: float,
        should_surface: bool,
    ) -> "RankedSignal":
        return cls(
            sort_index=(signal_date, -rank_score),
            idea_id=idea_id,
            asset=asset,
            direction=direction,
            signal_date=signal_date,
            rank_score=rank_score,
            should_surface=should_surface,
        )


# ---------------------------------------------------------------------------
# Portfolio engine
# ---------------------------------------------------------------------------
@dataclass
class PortfolioState:
    """Mutable state tracked during the replay loop."""
    open_positions: dict[str, date] = field(default_factory=dict)   # asset → exit_date
    trades: list[TradeResult]       = field(default_factory=list)
    skipped: list[SkippedSignal]    = field(default_factory=list)


def _close_expired_positions(state: PortfolioState, today: date) -> None:
    """Remove positions whose exit_date is on or before today."""
    expired = [asset for asset, exit_dt in state.open_positions.items() if exit_dt <= today]
    for asset in expired:
        del state.open_positions[asset]


def _check_portfolio_constraints(
    signal: RankedSignal,
    state: PortfolioState,
    cfg: BacktestConfig,
) -> str | None:
    """
    Return a skip reason string if the signal violates a portfolio constraint,
    else None (signal is acceptable).
    """
    if cfg.surface_only and not signal.should_surface:
        return "not_surfaced"

    if signal.rank_score < cfg.min_rank_score:
        return f"rank_score_below_min({cfg.min_rank_score})"

    if len(state.open_positions) >= cfg.max_positions:
        return "max_positions_reached"

    if cfg.one_position_per_asset and signal.asset in state.open_positions:
        return "asset_already_open"

    return None


def run_backtest(
    signals: list[RankedSignal],
    cfg: BacktestConfig = BacktestConfig(),
) -> tuple[list[TradeResult], list[SkippedSignal]]:
    """
    Replay signals chronologically and return all trade and skip records.

    Args:
        signals: All ranked signals to consider, in any order.
                 The engine sorts them internally by (signal_date, rank_score desc).
        cfg:     Backtest configuration.

    Returns:
        (trades, skipped) — lists of TradeResult and SkippedSignal.
    """
    # Sort chronologically; within same date, higher rank_score goes first
    sorted_signals = sorted(signals)

    # Cache price series to avoid re-reading CSVs on every signal
    price_cache: dict[str, pd.Series | None] = {}

    state = PortfolioState()

    for signal in sorted_signals:
        today = signal.signal_date
        _close_expired_positions(state, today)

        # --- Portfolio constraint checks ---
        skip_reason = _check_portfolio_constraints(signal, state, cfg)
        if skip_reason:
            state.skipped.append(SkippedSignal(
                idea_id=signal.idea_id,
                asset=signal.asset,
                direction=signal.direction,
                signal_date=today,
                reason=skip_reason,
            ))
            logger.debug("Skipped idea_id=%d asset=%s reason=%s", signal.idea_id, signal.asset, skip_reason)
            continue

        # --- Load prices (cached) ---
        if signal.asset not in price_cache:
            price_cache[signal.asset] = load_prices(signal.asset)

        prices = price_cache[signal.asset]
        if prices is None:
            state.skipped.append(SkippedSignal(
                idea_id=signal.idea_id,
                asset=signal.asset,
                direction=signal.direction,
                signal_date=today,
                reason="no_price_file",
            ))
            continue

        # --- Execution ---
        trade, skip_reason = try_execute(
            idea_id=signal.idea_id,
            asset=signal.asset,
            direction=signal.direction,
            signal_date=today,
            prices=prices,
            cfg=cfg,
        )

        if trade is None:
            state.skipped.append(SkippedSignal(
                idea_id=signal.idea_id,
                asset=signal.asset,
                direction=signal.direction,
                signal_date=today,
                reason=skip_reason or "execution_failed",
            ))
        else:
            state.open_positions[signal.asset] = trade.exit_date
            state.trades.append(trade)
            logger.debug(
                "Trade entered: idea_id=%d %s %s entry=%s exit=%s net_ret=%.4f",
                trade.idea_id, trade.direction, trade.asset,
                trade.entry_date, trade.exit_date, trade.net_return,
            )

    return state.trades, state.skipped
