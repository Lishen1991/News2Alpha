"""
Cluster analysis service.

Takes an EventCluster and produces one ExtractedEvent + one set of
TradeIdeas representing the cluster as a whole, then persists them.

Pipeline
────────
1. Build a synthetic ArticleInput from all non-duplicate articles in the
   cluster  (cluster text builder).
2. Run the existing extract_event() on that synthetic input.
3. Run match_rules() + generate_ideas() exactly as the article pipeline does.
4. Persist ClusterEvent + ClusterTradeIdea records.

Idempotency
───────────
A UNIQUE constraint on cluster_events.cluster_id prevents duplicate rows.
analyse_cluster() checks for an existing ClusterEvent and returns it
immediately unless force=True is passed, in which case the existing record
(and its cascaded trade ideas) is deleted first.

Cluster text builder
────────────────────
The synthetic text is constructed as:

    CLUSTER SUMMARY
    Articles: {n}  Sources: {s}  First seen: {date}

    TITLES
    • <title 1>
    • <title 2>
    ...

    BODIES
    <body 1>

    ---

    <body 2>

    ---
    ...

The combined text is truncated to MAX_CLUSTER_TEXT_CHARS to stay within
any future LLM context window limits.

Replacing the mock extractor
────────────────────────────
No changes needed here. The cluster text is passed to extract_event() as
an ArticleInput.body. When extraction_service is upgraded to use a real
LLM, cluster-level analysis automatically benefits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.cluster_analysis_models import ClusterEvent, ClusterTradeIdea
from app.db.cluster_models import ArticleClusterMap, EventCluster
from app.db.database import get_session
from app.db.models import ArticleRecord
from app.models.schemas import ArticleInput, ExtractedEvent, TradeIdea
from app.services.extraction_service import extract_event
from app.services.idea_service import generate_ideas
from app.services.rules_service import match_rules

logger = logging.getLogger(__name__)

# Maximum character length of the combined cluster text sent to the extractor.
# Keeps token usage bounded when a real LLM is introduced.
MAX_CLUSTER_TEXT_CHARS: int = 8_000


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ClusterAnalysisResult:
    cluster_id: int
    cluster_event_id: int
    event_type: str
    matched_rule_ids: list[str]
    idea_count: int
    was_cached: bool  # True if the result already existed in the DB


# ---------------------------------------------------------------------------
# Cluster text builder
# ---------------------------------------------------------------------------
def _load_cluster_articles(
    cluster: EventCluster,
    session: Session,
) -> list[ArticleRecord]:
    """
    Return all non-duplicate articles for a cluster, ordered by created_at.
    Duplicates are excluded so their redundant text doesn't dilute the signal.
    """
    maps: list[ArticleClusterMap] = (
        session.query(ArticleClusterMap)
        .filter_by(cluster_id=cluster.id, is_duplicate=0)
        .order_by(ArticleClusterMap.created_at.asc())
        .all()
    )
    articles: list[ArticleRecord] = []
    for m in maps:
        article = session.get(ArticleRecord, m.article_id)
        if article:
            articles.append(article)
    return articles


def build_cluster_text(articles: list[ArticleRecord], cluster: EventCluster) -> str:
    """
    Construct a single text payload representing the entire cluster.

    Args:
        articles: Non-duplicate articles in the cluster (chronological order).
        cluster:  The EventCluster metadata record.

    Returns:
        A structured plain-text string safe to pass as ArticleInput.body.
    """
    sources = {a.source for a in articles if a.source}
    first_seen = cluster.first_seen_at.strftime("%Y-%m-%d")

    header = (
        f"CLUSTER SUMMARY\n"
        f"Articles: {len(articles)}  "
        f"Sources: {len(sources)}  "
        f"First seen: {first_seen}\n\n"
    )

    titles_block = "TITLES\n" + "\n".join(f"• {a.title}" for a in articles) + "\n\n"

    bodies_block = "BODIES\n" + "\n\n---\n\n".join(a.body for a in articles)

    combined = header + titles_block + bodies_block

    if len(combined) > MAX_CLUSTER_TEXT_CHARS:
        combined = combined[:MAX_CLUSTER_TEXT_CHARS] + "\n...[truncated]"
        logger.debug(
            "Cluster %d text truncated to %d chars.", cluster.id, MAX_CLUSTER_TEXT_CHARS
        )

    return combined


def _build_article_input(
    articles: list[ArticleRecord],
    cluster: EventCluster,
) -> ArticleInput:
    """
    Wrap the cluster text in an ArticleInput so the existing extractor
    can consume it without any changes.

    title  → cluster_title (the seed article's headline)
    source → comma-joined unique sources (truncated to 256 chars)
    body   → the full structured cluster text
    """
    sources = sorted({a.source for a in articles if a.source})
    source_str = ", ".join(sources)[:256]

    return ArticleInput(
        title=cluster.cluster_title,
        source=source_str or "multiple",
        body=build_cluster_text(articles, cluster),
        published_at=cluster.first_seen_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _existing_cluster_event(
    cluster_id: int, session: Session
) -> ClusterEvent | None:
    return session.query(ClusterEvent).filter_by(cluster_id=cluster_id).first()


def _delete_cluster_event(cluster_event: ClusterEvent, session: Session) -> None:
    """Delete an existing ClusterEvent (cascade removes trade ideas)."""
    session.delete(cluster_event)
    session.flush()
    logger.info("Deleted existing ClusterEvent id=%d for re-analysis.", cluster_event.id)


def _persist_cluster_analysis(
    cluster: EventCluster,
    event: ExtractedEvent,
    ideas: list[TradeIdea],
    matched_rule_ids: list[str],
    article_count: int,
    source_count: int,
    session: Session,
) -> ClusterEvent:
    """Write ClusterEvent + ClusterTradeIdea rows within the current session."""
    cluster_event = ClusterEvent(
        cluster_id=cluster.id,
        event_type=event.event_type,
        summary=event.summary,
        severity=event.severity,
        novelty=event.novelty,
        channels=event.channels,
        time_horizon=event.time_horizon,
        affected_sectors=event.affected_sectors,
        entities=event.entities,
        region=event.region,
        article_count=article_count,
        source_count=source_count,
        matched_rule_ids=matched_rule_ids,
    )
    session.add(cluster_event)
    session.flush()

    for idea in ideas:
        session.add(ClusterTradeIdea(
            cluster_event_id=cluster_event.id,
            asset=idea.asset,
            direction=idea.direction,
            confidence=idea.confidence,
            horizon=idea.horizon,
            thesis=idea.thesis,
            supporting_factors=idea.supporting_factors,
            risk_flags=idea.risk_flags,
            invalidation_conditions=idea.invalidation_conditions,
        ))

    logger.info(
        "Persisted ClusterEvent id=%d cluster_id=%d event_type=%r ideas=%d",
        cluster_event.id,
        cluster.id,
        event.event_type,
        len(ideas),
    )
    return cluster_event


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyse_cluster(
    cluster_id: int,
    force: bool = False,
) -> ClusterAnalysisResult | None:
    """
    Run the full extraction → rules → ideas pipeline for one cluster.

    Args:
        cluster_id: Primary key of an EventCluster record.
        force:      If True, delete any existing analysis and rerun.
                    If False (default), return the cached result immediately.

    Returns:
        ClusterAnalysisResult, or None if the cluster does not exist or
        contains no non-duplicate articles.
    """
    try:
        with get_session() as session:
            cluster = session.get(EventCluster, cluster_id)
            if cluster is None:
                logger.warning("analyse_cluster: cluster_id=%d not found.", cluster_id)
                return None

            # --- Idempotency check ---
            existing = _existing_cluster_event(cluster_id, session)
            if existing and not force:
                logger.debug(
                    "Cluster %d already analysed (ClusterEvent id=%d). "
                    "Pass force=True to rerun.",
                    cluster_id,
                    existing.id,
                )
                return ClusterAnalysisResult(
                    cluster_id=cluster_id,
                    cluster_event_id=existing.id,
                    event_type=existing.event_type,
                    matched_rule_ids=existing.matched_rule_ids,
                    idea_count=len(existing.trade_ideas),
                    was_cached=True,
                )

            if existing and force:
                _delete_cluster_event(existing, session)

            # --- Load articles ---
            articles = _load_cluster_articles(cluster, session)
            if not articles:
                logger.warning(
                    "Cluster %d has no non-duplicate articles. Skipping.", cluster_id
                )
                return None

            # Expunge so objects survive session close
            for a in articles:
                session.expunge(a)
            session.expunge(cluster)

        # --- Run pipeline (outside the DB session to keep sessions short) ---
        article_input = _build_article_input(articles, cluster)

        event: ExtractedEvent = extract_event(article_input)
        matched_rules = match_rules(event)
        ideas: list[TradeIdea] = generate_ideas(event, matched_rules)
        matched_rule_ids = [r.rule_id for r in matched_rules]

        # --- Persist in a fresh session ---
        with get_session() as session:
            # Re-fetch cluster inside new session
            cluster = session.get(EventCluster, cluster_id)
            cluster_event = _persist_cluster_analysis(
                cluster=cluster,
                event=event,
                ideas=ideas,
                matched_rule_ids=matched_rule_ids,
                article_count=len(articles),
                source_count=len({a.source for a in articles}),
                session=session,
            )
            cluster_event_id = cluster_event.id
            idea_count = len(ideas)

        return ClusterAnalysisResult(
            cluster_id=cluster_id,
            cluster_event_id=cluster_event_id,
            event_type=event.event_type,
            matched_rule_ids=matched_rule_ids,
            idea_count=idea_count,
            was_cached=False,
        )

    except Exception:
        logger.exception("Failed to analyse cluster_id=%d", cluster_id)
        return None


def analyse_all_clusters(force: bool = False) -> list[ClusterAnalysisResult]:
    """
    Run analyse_cluster() for every cluster in the database.

    Args:
        force: Passed through to analyse_cluster(). Set True to rerun all.

    Returns:
        List of ClusterAnalysisResult (one per cluster, skipping failures).
    """
    with get_session() as session:
        cluster_ids: list[int] = [
            row[0] for row in session.query(EventCluster.id).order_by(EventCluster.id).all()
        ]

    results: list[ClusterAnalysisResult] = []
    for cid in cluster_ids:
        result = analyse_cluster(cid, force=force)
        if result:
            results.append(result)

    return results
