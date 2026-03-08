"""
Article processing service — handles the clustering stage for a single article.

Responsibilities
────────────────
- Call assign_cluster() for one article.
- Update article.processing_status in the DB so the orchestrator and
  any future UI can track where each article is in the pipeline.
- Return a typed result the orchestrator uses to build its cluster work-list.

This module intentionally does one thing. All multi-cluster orchestration,
market context, and ranking logic lives in pipeline_orchestrator.py.

Processing statuses written to ArticleRecord.processing_status:
    'clustered'  — assign_cluster() succeeded (new or existing cluster).
    'duplicate'  — article was marked as a duplicate by the clustering service.
    'failed'     — an exception was raised during processing.
    None / null  — article has not been through this service yet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.db.database import get_session
from app.db.models import ArticleRecord
from app.services.clustering_service import assign_cluster

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ArticleProcessingResult:
    """
    Outcome of processing one article through the clustering stage.

    Attributes:
        article_id:     DB primary key of the article.
        status:         'clustered' | 'duplicate' | 'failed'.
        cluster_id:     Cluster the article was assigned to (None on failure).
        is_new_cluster: True if a brand-new cluster was created.
        error:          Exception message if status='failed', else None.
    """
    article_id:     int
    status:         str
    cluster_id:     int | None
    is_new_cluster: bool
    error:          str | None = None


# ---------------------------------------------------------------------------
# Status persistence
# ---------------------------------------------------------------------------

def _set_processing_status(article_id: int, status: str) -> None:
    """Update ArticleRecord.processing_status in the DB."""
    try:
        with get_session() as session:
            article = session.get(ArticleRecord, article_id)
            if article is not None:
                article.processing_status = status
    except Exception:
        # Status update is best-effort; don't mask the real error.
        logger.warning(
            "Could not update processing_status for article_id=%d to '%s'.",
            article_id, status,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_article(article_id: int) -> ArticleProcessingResult:
    """
    Run the clustering stage for one article.

    Calls assign_cluster() and writes the resulting status back to the
    articles table. Idempotent — if the article is already mapped to a
    cluster, assign_cluster() returns the existing mapping without
    creating a duplicate.

    Args:
        article_id: Primary key of an ArticleRecord already stored in the DB.

    Returns:
        ArticleProcessingResult describing the outcome.
    """
    try:
        result = assign_cluster(article_id)
    except Exception as exc:
        logger.exception("Clustering failed for article_id=%d.", article_id)
        _set_processing_status(article_id, "failed")
        return ArticleProcessingResult(
            article_id=article_id,
            status="failed",
            cluster_id=None,
            is_new_cluster=False,
            error=str(exc),
        )

    if result is None:
        # article_id not found in DB — shouldn't happen if caller is correct
        _set_processing_status(article_id, "failed")
        return ArticleProcessingResult(
            article_id=article_id,
            status="failed",
            cluster_id=None,
            is_new_cluster=False,
            error="assign_cluster returned None (article not found?)",
        )

    if result.is_duplicate:
        _set_processing_status(article_id, "duplicate")
        return ArticleProcessingResult(
            article_id=article_id,
            status="duplicate",
            cluster_id=result.cluster_id,
            is_new_cluster=False,
        )

    _set_processing_status(article_id, "clustered")
    logger.info(
        "Article %d → cluster %d (%s).",
        article_id,
        result.cluster_id,
        "new cluster" if result.is_new_cluster else "existing cluster",
    )
    return ArticleProcessingResult(
        article_id=article_id,
        status="clustered",
        cluster_id=result.cluster_id,
        is_new_cluster=result.is_new_cluster,
    )
