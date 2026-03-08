"""
Cluster backfill script.

Loads all articles from the database in chronological order and runs the
clustering pipeline on each one. Useful after:
  - Changing SIMILARITY_THRESHOLD or CLUSTER_WINDOW_HOURS
  - Importing a batch of historical articles
  - Resetting the cluster tables

Usage:
    # Backfill all articles (skips already-assigned ones by default)
    py research/cluster_backfill.py

    # Force re-cluster everything (wipes existing assignments first)
    py research/cluster_backfill.py --reset

    # Dry run — show what would happen without writing to DB
    py research/cluster_backfill.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import engine, get_session, init_db
from app.db.cluster_models import ArticleClusterMap, EventCluster
from app.db.models import ArticleRecord
from app.services.clustering_service import (
    CLUSTER_WINDOW_HOURS,
    SIMILARITY_THRESHOLD,
    assign_cluster,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def _reset_clusters() -> None:
    """Delete all cluster assignments and cluster records."""
    with get_session() as session:
        deleted_maps = session.query(ArticleClusterMap).delete()
        deleted_clusters = session.query(EventCluster).delete()
        logger.info("Reset: deleted %d mappings and %d clusters.", deleted_maps, deleted_clusters)


def _load_articles_chronological() -> list[ArticleRecord]:
    """Return all articles ordered by created_at ascending."""
    with get_session() as session:
        articles = (
            session.query(ArticleRecord)
            .order_by(ArticleRecord.created_at.asc())
            .all()
        )
        # Expunge so objects are usable outside the session
        for a in articles:
            session.expunge(a)
        return articles


def _already_assigned(article_id: int) -> bool:
    with get_session() as session:
        return session.query(ArticleClusterMap).filter_by(article_id=article_id).first() is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill cluster assignments for all articles.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing cluster data before backfilling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing to the database.",
    )
    args = parser.parse_args()

    init_db()

    print(f"\nCluster backfill")
    print(f"  similarity_threshold : {SIMILARITY_THRESHOLD}")
    print(f"  cluster_window_hours : {CLUSTER_WINDOW_HOURS}")
    print(f"  reset                : {args.reset}")
    print(f"  dry_run              : {args.dry_run}\n")

    if args.reset and not args.dry_run:
        _reset_clusters()

    articles = _load_articles_chronological()
    if not articles:
        print("No articles found in the database. Call /analyze-article first.")
        return

    print(f"Processing {len(articles)} article(s) in chronological order...\n")

    new_clusters = 0
    joined_clusters = 0
    duplicates = 0
    skipped = 0

    for article in articles:
        if args.dry_run:
            print(f"  [DRY RUN] Would cluster article_id={article.id}: {article.title[:60]!r}")
            continue

        if not args.reset and _already_assigned(article.id):
            logger.debug("Article %d already assigned — skipping.", article.id)
            skipped += 1
            continue

        result = assign_cluster(article.id)
        if result is None:
            logger.warning("Clustering returned None for article_id=%d", article.id)
            continue

        if result.is_duplicate:
            duplicates += 1
            status = "DUPLICATE"
        elif result.is_new_cluster:
            new_clusters += 1
            status = "NEW CLUSTER"
        else:
            joined_clusters += 1
            score_str = f"{result.similarity_score:.3f}" if result.similarity_score else "n/a"
            status = f"JOINED cluster {result.cluster_id} (score={score_str})"

        print(f"  article_id={article.id:>4}  {status:<40}  {article.title[:50]!r}")

    if not args.dry_run:
        print(f"\n{'─'*60}")
        print(f"  New clusters   : {new_clusters}")
        print(f"  Joined         : {joined_clusters}")
        print(f"  Duplicates     : {duplicates}")
        print(f"  Already done   : {skipped}")
        print(f"  Total articles : {len(articles)}")
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
