"""
Collector ingestion service — deduplicates and stores normalized articles
from the RSS collector into SQLite.

Design
──────
Deduplication is deterministic and two-stage:
  1. URL match:   if the article has a URL, check for an exact match in
                  the articles table. Fast index lookup.
  2. Title hash:  if URL is absent or not matched, check normalized_title_hash.
                  This catches reposts and articles with different URLs but
                  the same headline.

An article is inserted only if both checks pass (i.e. not a duplicate).

The service does NOT trigger downstream analysis (clustering, extraction,
ranking) by default. That keeps collection decoupled from analysis and
lets you run the pipeline in batch on your own schedule.

Optional downstream hook
────────────────────────
If analyze=True is passed to ingest_articles(), each newly stored article
is handed off to clustering_service.assign_cluster() after insertion.
This is the only coupling point and is explicitly opt-in.

Usage (batch store only):
    from app.services.collector_ingestion_service import ingest_articles
    result = ingest_articles(normalized_articles)

Usage (store + cluster):
    result = ingest_articles(normalized_articles, analyze=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.collectors.article_normalizer import NormalizedArticle
from app.db.database import get_session
from app.db.models import ArticleRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """
    Summary of one ingestion batch.

    Attributes:
        total:      Total NormalizedArticle objects considered.
        inserted:   New articles stored in the DB.
        skipped:    Duplicates detected and skipped.
        failed:     Articles that could not be stored due to an error.
        new_ids:    DB primary keys of newly inserted ArticleRecord rows.
    """
    total:    int = 0
    inserted: int = 0
    skipped:  int = 0
    failed:   int = 0
    new_ids:  list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Duplicate check helpers
# ---------------------------------------------------------------------------

def _url_exists(url: str, session) -> bool:
    """Return True if any article with this exact URL is already stored."""
    return (
        session.query(ArticleRecord.id)
        .filter(ArticleRecord.url == url)
        .first()
    ) is not None


def _hash_exists(norm_hash: str, session) -> bool:
    """Return True if any article with this normalized_title_hash is already stored."""
    return (
        session.query(ArticleRecord.id)
        .filter(ArticleRecord.normalized_title_hash == norm_hash)
        .first()
    ) is not None


def _is_duplicate(article: NormalizedArticle, session) -> bool:
    """
    Check whether a NormalizedArticle already exists in the database.

    Deduplication policy (in order):
      1. If the article has a URL → check exact URL match.
      2. Always check normalized_title_hash (catches same story, different URL).
    """
    if article.url and _url_exists(article.url, session):
        logger.debug("Duplicate (URL): %s", article.url)
        return True

    if _hash_exists(article.normalized_title_hash, session):
        logger.debug("Duplicate (title hash): %s", article.normalized_title)
        return True

    return False


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _to_record(article: NormalizedArticle) -> ArticleRecord:
    """Map a NormalizedArticle to an ArticleRecord ORM instance."""
    return ArticleRecord(
        title=article.title,
        source=article.source,
        body=article.body,
        url=article.url,
        published_at=article.published_at,
        collected_at=article.collected_at,
        category=article.category,
        normalized_title_hash=article.normalized_title_hash,
        ingestion_source=article.ingestion_source,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_articles(
    articles: list[NormalizedArticle],
    analyze: bool = False,
) -> IngestionResult:
    """
    Deduplicate and store a batch of normalized articles.

    Args:
        articles: List of NormalizedArticle objects from the collector.
        analyze:  If True, call assign_cluster() for each newly inserted
                  article after storage. Default False (deferred pipeline).

    Returns:
        IngestionResult with counts of inserted / skipped / failed.
    """
    result = IngestionResult(total=len(articles))

    if not articles:
        return result

    for article in articles:
        try:
            with get_session() as session:
                if _is_duplicate(article, session):
                    result.skipped += 1
                    continue

                record = _to_record(article)
                session.add(record)
                session.flush()
                new_id = record.id

            result.inserted += 1
            result.new_ids.append(new_id)
            logger.info(
                "Inserted article id=%d source=%r title=%r",
                new_id, article.source, article.title[:60],
            )

        except Exception:
            result.failed += 1
            logger.exception(
                "Failed to store article title=%r source=%r",
                article.title[:60], article.source,
            )

    # --- Optional downstream analysis hook ---
    if analyze and result.new_ids:
        _trigger_clustering(result.new_ids)

    return result


def _trigger_clustering(article_ids: list[int]) -> None:
    """
    Assign newly stored articles to event clusters.

    Called only when analyze=True. Errors are logged but do not affect
    the ingestion result — clustering is best-effort at collection time.
    """
    try:
        from app.services.clustering_service import assign_cluster
    except ImportError:
        logger.warning("clustering_service not available; skipping auto-clustering.")
        return

    clustered = failed = 0
    for article_id in article_ids:
        try:
            result = assign_cluster(article_id)
            if result is not None:
                clustered += 1
        except Exception:
            failed += 1
            logger.exception("Clustering failed for article_id=%d", article_id)

    logger.info(
        "Auto-clustering: %d clustered, %d failed (of %d new articles).",
        clustered, failed, len(article_ids),
    )
