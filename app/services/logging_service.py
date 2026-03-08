"""
Persistence (logging) service — writes analysis results to the database.

Public surface:
    log_analysis(article, event, ideas) -> None

All database errors are caught and logged so that a persistence failure
never crashes the API response. The caller always gets its result back.
"""

from __future__ import annotations

import logging

from app.db.database import get_session
from app.db.models import ArticleRecord, EventRecord, TradeIdeaRecord
from app.models.schemas import ArticleInput, ExtractedEvent, TradeIdea
from app.services import clustering_service

logger = logging.getLogger(__name__)


def log_analysis(
    article: ArticleInput,
    event: ExtractedEvent,
    ideas: list[TradeIdea],
) -> None:
    """
    Persist one full analysis run (article + event + trade ideas) to SQLite.

    The three records are written in a single transaction so they are
    either all committed or all rolled back together.

    Errors are swallowed after logging — persistence failures must not
    affect the HTTP response returned to the caller.

    Args:
        article: The original ArticleInput received by the endpoint.
        event:   The ExtractedEvent produced by the extraction service.
        ideas:   The list of TradeIdea objects produced by the idea service.
    """
    try:
        with get_session() as session:
            # --- 1. Article ---
            article_record = ArticleRecord(
                title=article.title,
                source=article.source,
                body=article.body,
                published_at=article.published_at,
                url=article.url,
            )
            session.add(article_record)
            session.flush()  # Populate article_record.id before referencing it

            # --- 2. Event ---
            event_record = EventRecord(
                article_id=article_record.id,
                event_type=event.event_type,
                summary=event.summary,
                severity=event.severity,
                novelty=event.novelty,
                channels=event.channels,           # stored as JSON
                time_horizon=event.time_horizon,
            )
            session.add(event_record)
            session.flush()  # Populate event_record.id before referencing it

            # --- 3. Trade ideas ---
            for idea in ideas:
                session.add(
                    TradeIdeaRecord(
                        event_id=event_record.id,
                        asset=idea.asset,
                        direction=idea.direction,
                        confidence=idea.confidence,
                        horizon=idea.horizon,
                        thesis=idea.thesis,
                    )
                )

            # Session context manager commits here on clean exit
            logger.info(
                "Persisted analysis: article_id=%s event_id=%s ideas=%d",
                article_record.id,
                event_record.id,
                len(ideas),
            )

        # --- 4. Cluster assignment (separate session so a clustering failure
        #        never rolls back the article/event/ideas write above) ---
        article_id = article_record.id
        result = clustering_service.assign_cluster(article_id)
        if result:
            logger.info(
                "Cluster assignment: article_id=%d cluster_id=%d new=%s dup=%s score=%s",
                article_id,
                result.cluster_id,
                result.is_new_cluster,
                result.is_duplicate,
                f"{result.similarity_score:.3f}" if result.similarity_score is not None else "n/a",
            )

    except Exception:
        # Log the full traceback but do not re-raise — the API must still respond
        logger.exception(
            "Failed to persist analysis for article '%s'. "
            "API response will still be returned.",
            article.title,
        )
