"""
Market context service.

For each ClusterTradeIdea, computes pre-signal price features, derives
context flags, adjusts confidence, and decides whether to surface the idea.

Configuration
─────────────
All thresholds live in the Thresholds dataclass at the top of this module.
Edit them here or override per-call via the thresholds= argument.

    already_moved_3d    float   abs(ret_3d_pre_signal) > this → already_moved
    already_moved_1d    float   abs(ret_1d_pre_signal) > this → already_moved (secondary)
    stale_signal_days   float   staleness_days > this → stale_signal
    high_vol_threshold  float   realized_vol_5d (annualised) > this → high_volatility
    max_flags_to_surface int    ideas with >= this many flags are suppressed

Confidence penalty
──────────────────
Each flag reduces confidence by a fixed penalty (PENALTY_PER_FLAG).
Multiple flags stack. Result is clamped to [0, 1].

    adjusted = max(0.0, original - n_flags * PENALTY_PER_FLAG)

Staleness computation
─────────────────────
staleness_days = (idea.created_at − cluster.first_seen_at).total_seconds() / 86400

This is in calendar days (not trading days) so the calculation is
timezone-aware and needs no market calendar.

Surface decision logic
──────────────────────
should_surface = False  if any of:
  - "insufficient_price_data" flag is present
  - number of flags >= max_flags_to_surface

Otherwise should_surface = True.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Allow importing research.price_loader when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from research.price_loader import load_prices  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Thresholds:
    """
    All configurable thresholds for the market context filter.
    Override per-call: Thresholds(already_moved_3d=0.05)
    """
    already_moved_3d: float    = 0.04   # 4% 3-day move → flag
    already_moved_1d: float    = 0.025  # 2.5% 1-day move → flag (secondary)
    stale_signal_days: float   = 3.0    # > 3 calendar days → stale
    high_vol_threshold: float  = 0.40   # annualised vol > 40% → high_volatility
    max_flags_to_surface: int  = 2      # >= 2 flags → suppress

PENALTY_PER_FLAG: float = 0.10         # confidence reduction per flag

DEFAULT_THRESHOLDS = Thresholds()


# ---------------------------------------------------------------------------
# Price feature computation
# ---------------------------------------------------------------------------
def _get_price_at_or_before(prices: pd.Series, target_date: date) -> float | None:
    """Return the most recent price on or before target_date."""
    ts = pd.Timestamp(target_date)
    subset = prices[prices.index <= ts]
    return float(subset.iloc[-1]) if not subset.empty else None


def _pre_signal_return(
    prices: pd.Series,
    signal_date: date,
    lookback: int,
) -> float | None:
    """
    Compute the percentage return over the `lookback` trading days
    ending on signal_date (inclusive).

    Returns None if there are insufficient price observations.
    """
    ts = pd.Timestamp(signal_date)
    history = prices[prices.index <= ts]
    if len(history) < lookback + 1:
        return None
    entry = float(history.iloc[-(lookback + 1)])
    exit_ = float(history.iloc[-1])
    if entry == 0:
        return None
    return (exit_ - entry) / entry


def _realized_vol_5d(prices: pd.Series, signal_date: date) -> float | None:
    """
    5-day realised volatility (annualised) ending on signal_date.

    Uses log returns of the 6 most recent closes ending on signal_date,
    giving 5 daily log-return observations, then annualises by sqrt(252).

    Returns None if fewer than 6 price observations are available.
    """
    ts = pd.Timestamp(signal_date)
    history = prices[prices.index <= ts]
    if len(history) < 6:
        return None
    last_six = history.iloc[-6:]
    log_returns = np.log(last_six / last_six.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))


# ---------------------------------------------------------------------------
# Flag derivation
# ---------------------------------------------------------------------------
def _derive_flags(
    ret_1d: float | None,
    ret_3d: float | None,
    ret_5d: float | None,
    vol_5d: float | None,
    staleness_days: float | None,
    direction: str,
    t: Thresholds,
) -> list[str]:
    """
    Return a list of string flag labels based on computed features.

    The direction of the idea matters for already_moved:
      - For a LONG idea, a large positive pre-signal move means the market
        has run ahead of us (bearish for entry).
      - For a SHORT idea, a large negative pre-signal move means the same.
    We therefore flag based on the absolute move regardless of direction,
    since either way the opportunity may be diminished.
    """
    flags: list[str] = []

    if ret_1d is None and ret_3d is None and ret_5d is None and vol_5d is None:
        flags.append("insufficient_price_data")
        return flags   # No further checks possible

    # already_moved: primary check on 3d, secondary on 1d
    if ret_3d is not None and abs(ret_3d) > t.already_moved_3d:
        flags.append("already_moved")
    elif ret_1d is not None and abs(ret_1d) > t.already_moved_1d:
        flags.append("already_moved")

    if vol_5d is not None and vol_5d > t.high_vol_threshold:
        flags.append("high_volatility")

    if staleness_days is not None and staleness_days > t.stale_signal_days:
        flags.append("stale_signal")

    return flags


# ---------------------------------------------------------------------------
# Confidence adjustment
# ---------------------------------------------------------------------------
def _adjust_confidence(original: float, flags: list[str]) -> float:
    """
    Reduce confidence by PENALTY_PER_FLAG for each flag that is not
    insufficient_price_data (which triggers should_surface=False directly).
    """
    penalised_flags = [f for f in flags if f != "insufficient_price_data"]
    adjusted = original - len(penalised_flags) * PENALTY_PER_FLAG
    return round(max(0.0, min(1.0, adjusted)), 4)


# ---------------------------------------------------------------------------
# Surface decision
# ---------------------------------------------------------------------------
def _should_surface(flags: list[str], t: Thresholds) -> bool:
    if "insufficient_price_data" in flags:
        return False
    return len(flags) < t.max_flags_to_surface


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class MarketContextResult:
    """All computed values for one trade idea. Passed directly to persistence."""
    asset: str
    direction: str
    signal_date: date
    ret_1d_pre_signal: float | None
    ret_3d_pre_signal: float | None
    ret_5d_pre_signal: float | None
    realized_vol_5d: float | None
    staleness_days: float | None
    context_flags: list[str]
    original_confidence: float
    adjusted_confidence: float
    should_surface: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_market_context(
    asset: str,
    direction: str,
    original_confidence: float,
    signal_date: date,
    cluster_first_seen_at: datetime,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> MarketContextResult:
    """
    Compute the full market context evaluation for one trade idea.

    Args:
        asset:                 Asset identifier (e.g. 'brent_crude_front').
        direction:             'long' or 'short'.
        original_confidence:   Confidence score from the idea generator [0,1].
        signal_date:           Date the idea was generated (idea.created_at.date()).
        cluster_first_seen_at: UTC datetime of the cluster's first article.
        thresholds:            Override default thresholds if needed.

    Returns:
        A fully populated MarketContextResult.
    """
    prices = load_prices(asset)

    # Staleness: always computable regardless of price data availability
    idea_dt = datetime.combine(signal_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    first_seen = cluster_first_seen_at
    if first_seen.tzinfo is None:
        first_seen = first_seen.replace(tzinfo=timezone.utc)
    staleness_days = (idea_dt - first_seen).total_seconds() / 86400

    if prices is None:
        flags = ["insufficient_price_data"]
        return MarketContextResult(
            asset=asset,
            direction=direction,
            signal_date=signal_date,
            ret_1d_pre_signal=None,
            ret_3d_pre_signal=None,
            ret_5d_pre_signal=None,
            realized_vol_5d=None,
            staleness_days=round(staleness_days, 2),
            context_flags=flags,
            original_confidence=original_confidence,
            adjusted_confidence=_adjust_confidence(original_confidence, flags),
            should_surface=False,
        )

    ret_1d = _pre_signal_return(prices, signal_date, 1)
    ret_3d = _pre_signal_return(prices, signal_date, 3)
    ret_5d = _pre_signal_return(prices, signal_date, 5)
    vol_5d = _realized_vol_5d(prices, signal_date)

    flags = _derive_flags(ret_1d, ret_3d, ret_5d, vol_5d, staleness_days, direction, thresholds)
    adjusted = _adjust_confidence(original_confidence, flags)
    surface = _should_surface(flags, thresholds)

    logger.debug(
        "MarketContext: asset=%s dir=%s date=%s flags=%s conf %.2f→%.2f surface=%s",
        asset, direction, signal_date, flags, original_confidence, adjusted, surface,
    )

    return MarketContextResult(
        asset=asset,
        direction=direction,
        signal_date=signal_date,
        ret_1d_pre_signal=round(ret_1d, 6) if ret_1d is not None else None,
        ret_3d_pre_signal=round(ret_3d, 6) if ret_3d is not None else None,
        ret_5d_pre_signal=round(ret_5d, 6) if ret_5d is not None else None,
        realized_vol_5d=round(vol_5d, 6) if vol_5d is not None else None,
        staleness_days=round(staleness_days, 2),
        context_flags=flags,
        original_confidence=original_confidence,
        adjusted_confidence=adjusted,
        should_surface=surface,
    )
