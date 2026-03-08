"""
Cluster analysis research script.

Loads all clusters from the database, runs extraction + rules + idea
generation on each one, and stores the results.

Usage:
    # Analyse all clusters that haven't been analysed yet
    py research/analyze_clusters.py

    # Force rerun all clusters (re-extracts even if already done)
    py research/analyze_clusters.py --force

    # Analyse a single cluster by ID
    py research/analyze_clusters.py --cluster-id 1

    # Show what's in the DB without running analysis
    py research/analyze_clusters.py --status
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import init_db, get_session
from app.db.cluster_models import EventCluster
from app.db.cluster_analysis_models import ClusterEvent, ClusterTradeIdea
from app.services.cluster_analysis_service import analyse_cluster, analyse_all_clusters

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------
def _print_status() -> None:
    """Print a summary of all clusters and their analysis state."""
    with get_session() as session:
        clusters = session.query(EventCluster).order_by(EventCluster.id).all()

        if not clusters:
            print("No clusters found. Run the backfill first:")
            print("  py research/cluster_backfill.py")
            return

        print(f"\n{'─'*72}")
        print(f"{'ID':>4}  {'Articles':>8}  {'Analysed':>9}  {'Event type':<25}  Cluster title")
        print(f"{'─'*72}")

        for cluster in clusters:
            ce = session.query(ClusterEvent).filter_by(cluster_id=cluster.id).first()
            analysed = "YES" if ce else "NO"
            event_type = ce.event_type if ce else "-"
            title = cluster.cluster_title[:35]
            print(
                f"{cluster.id:>4}  {cluster.article_count:>8}  {analysed:>9}  "
                f"{event_type:<25}  {title!r}"
            )

        print(f"{'─'*72}")
        total = len(clusters)
        done = sum(
            1 for c in clusters
            if session.query(ClusterEvent).filter_by(cluster_id=c.id).first()
        )
        print(f"  Total clusters: {total}   Analysed: {done}   Pending: {total - done}\n")


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------
def _print_result(result) -> None:
    print(f"\n  cluster_id      : {result.cluster_id}")
    print(f"  cluster_event_id: {result.cluster_event_id}")
    print(f"  event_type      : {result.event_type}")
    print(f"  matched_rules   : {', '.join(result.matched_rule_ids) or 'none'}")
    print(f"  ideas generated : {result.idea_count}")
    print(f"  cached          : {result.was_cached}")

    if not result.was_cached:
        with get_session() as session:
            ce = session.query(ClusterEvent).get(result.cluster_event_id)
            if ce:
                print(f"  summary         : {ce.summary[:120]}")
                print(f"  severity        : {ce.severity}")
                print(f"  channels        : {ce.channels}")

                ideas = (
                    session.query(ClusterTradeIdea)
                    .filter_by(cluster_event_id=ce.id)
                    .all()
                )
                if ideas:
                    print(f"\n  Trade ideas:")
                    for idea in ideas:
                        print(
                            f"    {idea.direction.upper():<6} {idea.asset:<30} "
                            f"confidence={idea.confidence:.2f}  horizon={idea.horizon}"
                        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run cluster-level event and idea analysis.")
    parser.add_argument("--force", action="store_true", help="Rerun analysis even if already done.")
    parser.add_argument("--cluster-id", type=int, default=None, help="Analyse a single cluster by ID.")
    parser.add_argument("--status", action="store_true", help="Show cluster status without running analysis.")
    args = parser.parse_args()

    init_db()

    if args.status:
        _print_status()
        return

    print(f"\nCluster analysis")
    print(f"  force       : {args.force}")
    print(f"  cluster_id  : {args.cluster_id or 'all'}\n")

    if args.cluster_id:
        result = analyse_cluster(args.cluster_id, force=args.force)
        if result is None:
            print(f"Cluster {args.cluster_id} not found or has no articles.")
        else:
            status = "CACHED" if result.was_cached else "ANALYSED"
            print(f"  [{status}] cluster_id={result.cluster_id}")
            _print_result(result)
    else:
        results = analyse_all_clusters(force=args.force)
        if not results:
            print("No clusters to analyse.")
            return

        print(f"{'─'*68}")
        print(f"{'ID':>4}  {'Status':<10}  {'Event type':<25}  {'Ideas':>5}  {'Rules'}")
        print(f"{'─'*68}")

        for r in results:
            status = "CACHED" if r.was_cached else "ANALYSED"
            rules = ", ".join(r.matched_rule_ids[:2])
            if len(r.matched_rule_ids) > 2:
                rules += f" +{len(r.matched_rule_ids)-2}"
            print(
                f"{r.cluster_id:>4}  {status:<10}  {r.event_type:<25}  "
                f"{r.idea_count:>5}  {rules}"
            )

        print(f"{'─'*68}")
        analysed = sum(1 for r in results if not r.was_cached)
        cached   = sum(1 for r in results if r.was_cached)
        total_ideas = sum(r.idea_count for r in results if not r.was_cached)
        print(f"\n  Newly analysed : {analysed}")
        print(f"  Cached         : {cached}")
        print(f"  New trade ideas: {total_ideas}\n")


if __name__ == "__main__":
    main()
