"""
Idea generation service – converts matched rules + an ExtractedEvent into
a ranked list of TradeIdea objects.

Design notes
------------
- One TradeIdea is created per (asset, direction) pair across all matched rules.
- Confidence is derived deterministically from event.severity and the number
  of matched rules, capped to [0, 1].
- To integrate a scoring model or LLM reranker later, replace _score_idea()
  without changing the public surface (generate_ideas).
"""

from __future__ import annotations

import logging
from typing import Literal

from app.models.schemas import ExtractedEvent, TradeIdea
from app.services.rules_service import MatchedRule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(
    event: ExtractedEvent,
    matched_rules: list[MatchedRule],
    asset: str,
) -> float:
    """
    Deterministic confidence score in [0, 1].

    Formula:
        base  = event.severity
        boost = 0.05 per additional matched rule beyond the first (max 0.15)
        total = min(base + boost, 1.0)

    This keeps confidence grounded in severity while rewarding corroboration
    from multiple independent rules.
    """
    extra_rules = max(0, len(matched_rules) - 1)
    boost = min(extra_rules * 0.05, 0.15)
    return round(min(event.severity + boost, 1.0), 4)


# ---------------------------------------------------------------------------
# Thesis and supporting factor construction
# ---------------------------------------------------------------------------

def _build_thesis(
    asset: str,
    direction: Literal["long", "short"],
    event: ExtractedEvent,
    rule: MatchedRule,
) -> str:
    """Compose a readable one-sentence investment thesis."""
    mechanism_str = " and ".join(rule.mechanism) if rule.mechanism else "macro transmission"
    dir_label = "upside" if direction == "long" else "downside"
    return (
        f"{asset.upper()} offers {dir_label} exposure via {mechanism_str}. "
        f"Thesis: {event.summary}"
    )


def _build_supporting_factors(
    event: ExtractedEvent,
    rule: MatchedRule,
) -> list[str]:
    """Assemble supporting factors from the event and the matched rule."""
    factors: list[str] = [
        f"Matched rule: {rule.rule_id} — {rule.description}",
    ]
    if event.channels:
        factors.append(f"Active channels: {', '.join(event.channels)}")
    if event.affected_sectors:
        factors.append(f"Affected sectors: {', '.join(event.affected_sectors)}")
    factors.append(f"Event summary: {event.summary}")
    return factors


# ---------------------------------------------------------------------------
# Core idea factory
# ---------------------------------------------------------------------------

def _make_idea(
    asset: str,
    direction: Literal["long", "short"],
    event: ExtractedEvent,
    rule: MatchedRule,
    all_matched_rules: list[MatchedRule],
) -> TradeIdea:
    """Construct a single TradeIdea for one asset/direction pair."""
    return TradeIdea(
        asset=asset,
        direction=direction,
        confidence=_compute_confidence(event, all_matched_rules, asset),
        horizon=event.time_horizon,
        thesis=_build_thesis(asset, direction, event, rule),
        supporting_factors=_build_supporting_factors(event, rule),
        risk_flags=list(rule.risk_flags),
        invalidation_conditions=list(rule.invalidation_conditions),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ideas(
    event: ExtractedEvent,
    matched_rules: list[MatchedRule],
) -> list[TradeIdea]:
    """
    Generate a list of TradeIdea objects from an event and its matched rules.

    For each matched rule:
        - Every bullish asset  → TradeIdea with direction="long"
        - Every bearish asset  → TradeIdea with direction="short"

    Duplicate (asset, direction) pairs across rules are deduplicated, keeping
    the idea from the first matching rule (highest priority).

    Args:
        event: The validated ExtractedEvent driving the ideas.
        matched_rules: Output of rules_service.match_rules().

    Returns:
        List of TradeIdea instances. Empty list if no rules matched.
    """
    if not matched_rules:
        logger.info("No matched rules — returning empty idea list.")
        return []

    ideas: list[TradeIdea] = []
    seen: set[tuple[str, str]] = set()  # (asset, direction) dedup key

    for rule in matched_rules:
        for asset in rule.bullish_assets:
            key = (asset, "long")
            if key not in seen:
                ideas.append(_make_idea(asset, "long", event, rule, matched_rules))
                seen.add(key)

        for asset in rule.bearish_assets:
            key = (asset, "short")
            if key not in seen:
                ideas.append(_make_idea(asset, "short", event, rule, matched_rules))
                seen.add(key)

    logger.info("Generated %d trade idea(s) from %d matched rule(s).", len(ideas), len(matched_rules))
    return ideas
