"""
Signal ranking service.

Produces a deterministic, weighted composite score for each ClusterTradeIdea
using inputs from the event, the idea itself, and the market context filter.

Scoring formula
───────────────
rank_score = Σ (weight_i × normalised_component_i)   capped to [0, 1]

Components and default weights (must sum to 1.0):

    Component               Weight  Source
    ─────────────────────── ──────  ──────────────────────────────
    adjusted_confidence     0.35    market_context.adjusted_confidence
    event_severity          0.25    cluster_events.severity
    event_novelty           0.15    cluster_events.novelty
    rule_support            0.15    len(cluster_events.matched_rule_ids) / MAX_RULES
    context_penalty         0.10    1 - (n_flags / MAX_FLAGS)   [inverted — fewer=better]

All inputs are already in [0, 1] or are normalised before weighting.

Priority buckets
────────────────
    rank_score >= HIGH_THRESHOLD  → "high"
    rank_score >= MED_THRESHOLD   → "medium"
    otherwise                     → "low"

Surface decision
────────────────
    should_surface = False  if:
        - market_context.should_surface is False   (inherited from context filter)
        - priority_bucket == "low"
    otherwise True

Configuration
─────────────
All weights, thresholds, and caps live in the RankingConfig dataclass.
Override per-call: RankingConfig(w_confidence=0.40)

Extending
─────────
Add a new component by:
  1. Computing a normalised float in _compute_components().
  2. Adding a matching weight in RankingConfig.
  3. Keeping weights summed to 1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_RULES: int = 5    # normalisation ceiling for matched_rule_ids count
MAX_FLAGS: int = 3    # normalisation ceiling for context flag count


@dataclass
class RankingConfig:
    """
    Ranking weights and thresholds.

    Weights must sum to 1.0. Buckets are applied to rank_score in [0, 1].
    """
    # --- Component weights ---
    w_confidence: float = 0.35   # adjusted_confidence from market context
    w_severity:   float = 0.25   # event severity
    w_novelty:    float = 0.15   # event novelty
    w_rule_support: float = 0.15 # breadth of rule corroboration
    w_context:    float = 0.10   # market context cleanliness (inverted flags)

    # --- Priority bucket thresholds ---
    high_threshold: float = 0.65
    med_threshold:  float = 0.40

    # --- Surface suppression ---
    suppress_low_priority: bool = True   # suppress bucket="low" ideas


DEFAULT_CONFIG = RankingConfig()


# ---------------------------------------------------------------------------
# Internal computation
# ---------------------------------------------------------------------------
@dataclass
class _Components:
    adjusted_confidence: float
    event_severity: float
    event_novelty: float
    rule_support: float      # normalised to [0, 1]
    context_score: float     # 1 - normalised flag penalty


def _compute_components(
    adjusted_confidence: float,
    event_severity: float,
    event_novelty: float,
    matched_rule_count: int,
    context_flags: list[str],
) -> _Components:
    """Normalise all inputs to [0, 1] before weighting."""
    rule_support = min(matched_rule_count / MAX_RULES, 1.0)

    # Exclude insufficient_price_data from penalty count — it's already
    # handled by market_context.should_surface.
    penalised_flags = [f for f in context_flags if f != "insufficient_price_data"]
    context_score = max(0.0, 1.0 - len(penalised_flags) / MAX_FLAGS)

    return _Components(
        adjusted_confidence=max(0.0, min(1.0, adjusted_confidence)),
        event_severity=max(0.0, min(1.0, event_severity)),
        event_novelty=max(0.0, min(1.0, event_novelty)),
        rule_support=rule_support,
        context_score=context_score,
    )


def _weighted_score(c: _Components, cfg: RankingConfig) -> float:
    score = (
        cfg.w_confidence   * c.adjusted_confidence
        + cfg.w_severity   * c.event_severity
        + cfg.w_novelty    * c.event_novelty
        + cfg.w_rule_support * c.rule_support
        + cfg.w_context    * c.context_score
    )
    return round(max(0.0, min(1.0, score)), 6)


def _priority_bucket(score: float, cfg: RankingConfig) -> str:
    if score >= cfg.high_threshold:
        return "high"
    if score >= cfg.med_threshold:
        return "medium"
    return "low"


def _build_reasons(
    c: _Components,
    cfg: RankingConfig,
    score: float,
    bucket: str,
    context_flags: list[str],
    matched_rule_count: int,
) -> list[str]:
    """Produce human-readable explanations for the score."""
    reasons: list[str] = [f"rank_score={score:.3f} → priority={bucket}"]

    if c.adjusted_confidence >= 0.70:
        reasons.append(f"high adjusted confidence ({c.adjusted_confidence:.2f})")
    elif c.adjusted_confidence < 0.40:
        reasons.append(f"low adjusted confidence ({c.adjusted_confidence:.2f})")

    if c.event_severity >= 0.70:
        reasons.append(f"high event severity ({c.event_severity:.2f})")

    if c.event_novelty >= 0.60:
        reasons.append(f"notable event novelty ({c.event_novelty:.2f})")

    if matched_rule_count >= 2:
        reasons.append(f"corroborated by {matched_rule_count} rules")

    for flag in context_flags:
        if flag != "insufficient_price_data":
            reasons.append(f"market flag: {flag}")

    if bucket == "low":
        reasons.append("suppressed: priority too low to surface")

    return reasons


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RankingResult:
    """Ranking output for one ClusterTradeIdea."""
    cluster_trade_idea_id: int
    asset: str
    direction: str
    cluster_event_id: int
    rank_score: float
    rank: int                    # set after sorting within cluster
    priority_bucket: str
    should_surface: bool
    score_components: dict
    ranking_reasons: list[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def rank_idea(
    cluster_trade_idea_id: int,
    asset: str,
    direction: str,
    cluster_event_id: int,
    adjusted_confidence: float,
    event_severity: float,
    event_novelty: float,
    matched_rule_count: int,
    context_flags: list[str],
    context_should_surface: bool,
    config: RankingConfig = DEFAULT_CONFIG,
) -> RankingResult:
    """
    Compute the ranking score for a single trade idea.

    rank is set to 0 here and assigned by rank_ideas_for_cluster() after
    all ideas in the cluster are scored and sorted.

    Args:
        cluster_trade_idea_id: PK of the ClusterTradeIdea row.
        asset:                 Asset identifier.
        direction:             'long' or 'short'.
        cluster_event_id:      FK to the ClusterEvent.
        adjusted_confidence:   From MarketContext.adjusted_confidence.
        event_severity:        From ClusterEvent.severity.
        event_novelty:         From ClusterEvent.novelty.
        matched_rule_count:    len(ClusterEvent.matched_rule_ids).
        context_flags:         From MarketContext.context_flags.
        context_should_surface: From MarketContext.should_surface.
        config:                Override default weights/thresholds.

    Returns:
        RankingResult with rank=0 (placeholder until cluster sort).
    """
    components = _compute_components(
        adjusted_confidence=adjusted_confidence,
        event_severity=event_severity,
        event_novelty=event_novelty,
        matched_rule_count=matched_rule_count,
        context_flags=context_flags,
    )
    score = _weighted_score(components, config)
    bucket = _priority_bucket(score, config)

    # Inherit suppression from market context OR low priority bucket
    surface = (
        context_should_surface
        and not (config.suppress_low_priority and bucket == "low")
    )

    reasons = _build_reasons(components, config, score, bucket, context_flags, matched_rule_count)

    logger.debug(
        "Ranked idea_id=%d asset=%s score=%.3f bucket=%s surface=%s",
        cluster_trade_idea_id, asset, score, bucket, surface,
    )

    return RankingResult(
        cluster_trade_idea_id=cluster_trade_idea_id,
        asset=asset,
        direction=direction,
        cluster_event_id=cluster_event_id,
        rank_score=score,
        rank=0,
        priority_bucket=bucket,
        should_surface=surface,
        score_components={
            "adjusted_confidence": round(components.adjusted_confidence, 4),
            "event_severity":      round(components.event_severity, 4),
            "event_novelty":       round(components.event_novelty, 4),
            "rule_support":        round(components.rule_support, 4),
            "context_score":       round(components.context_score, 4),
        },
        ranking_reasons=reasons,
    )


def assign_ranks(results: list[RankingResult]) -> list[RankingResult]:
    """
    Sort a list of RankingResults by rank_score descending and assign
    rank positions (1 = best).

    Args:
        results: All RankingResult objects for one cluster (or any group).

    Returns:
        Same list, mutated in-place with rank set, sorted descending.
    """
    results.sort(key=lambda r: r.rank_score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1
    return results
