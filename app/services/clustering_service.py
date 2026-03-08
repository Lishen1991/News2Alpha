"""
Article deduplication and event clustering service.

Pipeline for each incoming article
───────────────────────────────────
1. Dedup check
   a. Exact URL match (fastest)
   b. Normalised title match (catches reposts)
   c. Body text hash match (catches byte-for-byte copies)

2. Cluster assignment (only for non-duplicates)
   a. Fetch recent clusters within the configured time window.
   b. Build a TF-IDF vector for the new article's title + body.
   c. Compute cosine similarity against cluster seed texts.
   d. If max similarity >= SIMILARITY_THRESHOLD → join that cluster.
   e. Otherwise → create a new cluster.

3. Persist the assignment and update cluster metadata.

Configuration
─────────────
Two module-level constants control behaviour. Edit them here or override
them from a .env / config file in a later iteration.

    SIMILARITY_THRESHOLD  float in (0, 1)
        Minimum cosine similarity to merge into an existing cluster.
        Lower → more aggressive merging.  Default: 0.40

    CLUSTER_WINDOW_HOURS  int
        How far back (in hours) to look for candidate clusters.
        Prevents old clusters from absorbing new articles about the same
        topic months later.  Default: 24

Replacing TF-IDF with embeddings
─────────────────────────────────
Swap out _vectorise() to return an embedding vector instead of a TF-IDF
vector.  The rest of the pipeline (cosine similarity, threshold, DB writes)
stays identical.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from app.db.cluster_models import ArticleClusterMap, EventCluster
from app.db.database import get_session
from app.db.models import ArticleRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD: float = 0.15   # raise to merge less aggressively
CLUSTER_WINDOW_HOURS: int = 24       # look-back window for candidate clusters


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------
@dataclass
class ClusterResult:
    cluster_id: int
    cluster_key: str
    is_duplicate: bool
    is_new_cluster: bool
    similarity_score: float | None   # None for duplicates and seed articles


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
def _normalise_title(title: str) -> str:
    """
    Lowercase, strip accents, collapse whitespace, remove punctuation.
    Used for fast exact-match deduplication before similarity is computed.
    """
    title = unicodedata.normalize("NFKD", title.lower())
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _body_hash(body: str) -> str:
    """SHA-256 of the raw body text. Used for byte-level dedup."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _make_cluster_key(title: str, created_at: datetime) -> str:
    """
    Generate a human-readable, unique cluster key.
    Format: <normalised-title-slug>_<YYYYMMDD>
    """
    slug = re.sub(r"\s+", "_", _normalise_title(title))[:60]
    date_str = created_at.strftime("%Y%m%d")
    return f"{slug}_{date_str}"


def _article_text(article: ArticleRecord) -> str:
    """Concatenate title and body for vectorisation."""
    return f"{article.title} {article.body}"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def _find_duplicate(
    article: ArticleRecord,
    session: Session,
) -> ArticleClusterMap | None:
    """
    Return an existing ArticleClusterMap if the article is a duplicate,
    else None.

    Checks (in order of cost):
    1. URL match — if both articles have a URL and they match exactly.
    2. Normalised title match — compares cleaned titles.
    3. Body hash match — compares SHA-256 of raw body text.
    """
    existing_articles = session.query(ArticleRecord).filter(
        ArticleRecord.id != article.id
    ).all()

    norm_title = _normalise_title(article.title)
    body_hash  = _body_hash(article.body)

    for existing in existing_articles:
        # 1. URL match
        if article.url and existing.url and article.url.strip() == existing.url.strip():
            mapping = session.query(ArticleClusterMap).filter_by(article_id=existing.id).first()
            if mapping:
                logger.info("Duplicate detected (URL match): article_id=%d", article.id)
                return mapping

        # 2. Normalised title match
        if _normalise_title(existing.title) == norm_title:
            mapping = session.query(ArticleClusterMap).filter_by(article_id=existing.id).first()
            if mapping:
                logger.info("Duplicate detected (title match): article_id=%d", article.id)
                return mapping

        # 3. Body hash match
        if _body_hash(existing.body) == body_hash:
            mapping = session.query(ArticleClusterMap).filter_by(article_id=existing.id).first()
            if mapping:
                logger.info("Duplicate detected (body hash match): article_id=%d", article.id)
                return mapping

    return None


# ---------------------------------------------------------------------------
# TF-IDF vectorisation and similarity
# ---------------------------------------------------------------------------
def _vectorise(texts: list[str]) -> np.ndarray:
    """
    Return a TF-IDF matrix (n_texts × n_features).

    Uses character n-grams (2–4) in addition to word n-grams so that
    partial word matches (e.g. "Hormuz" vs "Hormuz strait") are captured.
    """
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=10_000,
        sublinear_tf=True,
    )
    return vectorizer.fit_transform(texts).toarray()


def _find_best_cluster(
    article: ArticleRecord,
    session: Session,
    now: datetime,
) -> tuple[EventCluster, float] | None:
    """
    Find the most similar recent cluster for the given article.

    Returns (cluster, similarity_score) if a match above the threshold is
    found, else None.
    """
    cutoff = now - timedelta(hours=CLUSTER_WINDOW_HOURS)

    recent_clusters: list[EventCluster] = (
        session.query(EventCluster)
        .filter(EventCluster.last_seen_at >= cutoff)
        .all()
    )

    if not recent_clusters:
        return None

    # Gather seed texts for each cluster (first article in cluster)
    cluster_texts: list[str] = []
    valid_clusters: list[EventCluster] = []

    for cluster in recent_clusters:
        seed_map = (
            session.query(ArticleClusterMap)
            .filter_by(cluster_id=cluster.id, is_duplicate=0)
            .order_by(ArticleClusterMap.created_at.asc())
            .first()
        )
        if seed_map:
            seed_article = session.get(ArticleRecord, seed_map.article_id)
            if seed_article:
                cluster_texts.append(_article_text(seed_article))
                valid_clusters.append(cluster)

    if not cluster_texts:
        return None

    # Vectorise all texts together so the vocabulary is consistent
    new_text = _article_text(article)
    all_texts = cluster_texts + [new_text]

    try:
        matrix = _vectorise(all_texts)
    except ValueError:
        # Happens when vocabulary is empty (e.g. all stop words)
        return None

    new_vec = matrix[-1].reshape(1, -1)
    cluster_matrix = matrix[:-1]

    similarities = cosine_similarity(new_vec, cluster_matrix)[0]
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score >= SIMILARITY_THRESHOLD:
        logger.debug(
            "Best cluster match: cluster_id=%d score=%.3f",
            valid_clusters[best_idx].id,
            best_score,
        )
        return valid_clusters[best_idx], best_score

    return None


# ---------------------------------------------------------------------------
# Cluster creation
# ---------------------------------------------------------------------------
def _create_cluster(article: ArticleRecord, session: Session, now: datetime) -> EventCluster:
    """Create a new cluster seeded by the given article."""
    key = _make_cluster_key(article.title, now)

    # Ensure uniqueness — append article_id if key already exists
    existing = session.query(EventCluster).filter_by(cluster_key=key).first()
    if existing:
        key = f"{key}_{article.id}"

    cluster = EventCluster(
        cluster_key=key,
        cluster_title=article.title[:512],
        first_seen_at=now,
        last_seen_at=now,
        article_count=1,
    )
    session.add(cluster)
    session.flush()  # populate cluster.id
    logger.info("Created new cluster: key=%r article_id=%d", key, article.id)
    return cluster


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def assign_cluster(article_id: int) -> ClusterResult | None:
    """
    Assign an article to a cluster (or mark it as a duplicate).

    This is the single public entry point for the clustering pipeline.
    It is safe to call multiple times for the same article_id — if a mapping
    already exists it is returned immediately.

    Args:
        article_id: Primary key of an ArticleRecord already in the database.

    Returns:
        ClusterResult, or None if the article_id does not exist.
    """
    try:
        with get_session() as session:
            article = session.get(ArticleRecord, article_id)
            if article is None:
                logger.warning("assign_cluster: article_id=%d not found.", article_id)
                return None

            now = datetime.now(timezone.utc)

            # --- Return early if already assigned ---
            existing_map = session.query(ArticleClusterMap).filter_by(article_id=article_id).first()
            if existing_map:
                logger.debug("Article %d already has a cluster assignment.", article_id)
                return ClusterResult(
                    cluster_id=existing_map.cluster_id,
                    cluster_key=existing_map.cluster.cluster_key,
                    is_duplicate=bool(existing_map.is_duplicate),
                    is_new_cluster=False,
                    similarity_score=existing_map.similarity_score,
                )

            # --- Step 1: Deduplication ---
            dup_map = _find_duplicate(article, session)
            if dup_map:
                cluster = dup_map.cluster
                mapping = ArticleClusterMap(
                    article_id=article_id,
                    cluster_id=cluster.id,
                    similarity_score=1.0,
                    is_duplicate=1,
                )
                session.add(mapping)
                cluster.last_seen_at = now
                cluster.article_count += 1
                return ClusterResult(
                    cluster_id=cluster.id,
                    cluster_key=cluster.cluster_key,
                    is_duplicate=True,
                    is_new_cluster=False,
                    similarity_score=1.0,
                )

            # --- Step 2: Cluster assignment ---
            is_new_cluster = False
            similarity_score: float | None = None

            match = _find_best_cluster(article, session, now)
            if match:
                cluster, similarity_score = match
                cluster.last_seen_at = now
                cluster.article_count += 1
                logger.info(
                    "Article %d joined cluster %d (score=%.3f)",
                    article_id, cluster.id, similarity_score,
                )
            else:
                cluster = _create_cluster(article, session, now)
                is_new_cluster = True

            mapping = ArticleClusterMap(
                article_id=article_id,
                cluster_id=cluster.id,
                similarity_score=similarity_score,
                is_duplicate=0,
            )
            session.add(mapping)

            return ClusterResult(
                cluster_id=cluster.id,
                cluster_key=cluster.cluster_key,
                is_duplicate=False,
                is_new_cluster=is_new_cluster,
                similarity_score=similarity_score,
            )

    except Exception:
        logger.exception("Clustering failed for article_id=%d", article_id)
        return None
