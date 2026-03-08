"""
Signal ranking research script.

For every ClusterTradeIdea that has a MarketContext record:
  1. Loads event metadata (severity, novelty, matched rules).
  2. Loads market context (adjusted confidence, flags, surface decision).
  3. Calls ranking_service.rank_idea() for each idea.
  4. Assigns intra-cluster ranks after all ideas in a cluster are scored.
  5. Persists SignalRanking rows.

Ideas without a MarketContext row are skipped with a warning — run
build_market_context.py first.

Usage:
    # Rank all ideas not yet ranked
    py research/build_signal_rankings.py

    # Force rerun (overwrite existing rankings)
    py research/build_signal_rankings.py --force

    # Rank a single cluster
    py research/build_signal_rankings.py --cluster-id 1

    # Show current ranking table without rerunning
    py research/build_signal_rankings.py --status

    # Show full score breakdown for every idea
    py research/build_signal_rankings.py --status --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.cluster_analysis_models import ClusterEvent, ClusterTradeIdea
from app.db.database import get_session, init_db
from app.db.market_context_models import MarketContext
from app.db.ranking_models import SignalRanking
from app.services.ranking_service import RankingResult, assign_ranks, rank_idea

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

_SEP = "─" * 88


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_rankable_ideas(
    cluster_id: int | None = None,
) -> list[tuple[ClusterTradeIdea, ClusterEvent, MarketContext]]:
    """
    Return (idea, cluster_event, market_context) triples for all ideas
    that have a MarketContext record.

    If cluster_id is provided, filters to that cluster only.
    """
    with get_session() as session:
        q = (
            session.query(ClusterTradeIdea, ClusterEvent, MarketContext)
            .join(ClusterEvent, ClusterTradeIdea.cluster_event_id == ClusterEvent.id)
            .join(MarketContext, MarketContext.cluster_trade_idea_id == ClusterTradeIdea.id)
        )
        if cluster_id is not None:
            q = q.filter(ClusterEvent.cluster_id == cluster_id)

        rows = q.order_by(ClusterEvent.id, ClusterTradeIdea.id).all()

        results = []
        expunged_events: set[int] = set()
        for idea, event, mc in rows:
            session.expunge(idea)
            session.expunge(mc)
            if event.id not in expunged_events:
                session.expunge(event)
                expunged_events.add(event.id)
            results.append((idea, event, mc))
        return results


def _already_ranked(idea_id: int) -> bool:
    with get_session() as session:
        return (
            session.query(SignalRanking)
            .filter_by(cluster_trade_idea_id=idea_id)
            .first()
        ) is not None


def _delete_ranking(idea_id: int) -> None:
    with get_session() as session:
        session.query(SignalRanking).filter_by(cluster_trade_idea_id=idea_id).delete()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _persist_rankings(rankings: list[RankingResult]) -> None:
    with get_session() as session:
        for r in rankings:
            session.add(SignalRanking(
                cluster_trade_idea_id=r.cluster_trade_idea_id,
                asset=r.asset,
                direction=r.direction,
                cluster_event_id=r.cluster_event_id,
                rank_score=r.rank_score,
                rank=r.rank,
                priority_bucket=r.priority_bucket,
                should_surface=r.should_surface,
                score_components=r.score_components,
                ranking_reasons=r.ranking_reasons,
            ))


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------
def _print_status(verbose: bool = False) -> None:
    with get_session() as session:
        rows = (
            session.query(SignalRanking)
            .order_by(SignalRanking.cluster_event_id, SignalRanking.rank)
            .all()
        )

        if not rows:
            print("No rankings found. Run:  py research/build_signal_rankings.py")
            return

        current_event: int | None = None
        print(f"\n{_SEP}")
        print(
            f"{'Rank':>4}  {'Asset':<28}  {'Dir':<6}  {'Score':>6}  "
            f"{'Bucket':<8}  {'Surface':<8}  Reasons"
        )
        print(_SEP)

        for r in rows:
            if r.cluster_event_id != current_event:
                current_event = r.cluster_event_id
                print(f"\n  ── Cluster event {r.cluster_event_id} ──")

            reasons_short = r.ranking_reasons[0] if r.ranking_reasons else ""
            print(
                f"  {r.rank:>2}  {r.asset:<28}  {r.direction:<6}  "
                f"{r.rank_score:>6.3f}  {r.priority_bucket:<8}  "
                f"{'YES' if r.should_surface else 'NO':<8}  {reasons_short}"
            )
            if verbose and r.score_components:
                comp = r.score_components
                print(
                    f"        adj_conf={comp.get('adjusted_confidence', '?'):.3f}  "
                    f"severity={comp.get('event_severity', '?'):.3f}  "
                    f"novelty={comp.get('event_novelty', '?'):.3f}  "
                    f"rule_support={comp.get('rule_support', '?'):.3f}  "
                    f"ctx={comp.get('context_score', '?'):.3f}"
                )

        print(f"\n{_SEP}")
        total = len(rows)
        surfaced = sum(1 for r in rows if r.should_surface)
        high = sum(1 for r in rows if r.priority_bucket == "high")
        med  = sum(1 for r in rows if r.priority_bucket == "medium")
        low  = sum(1 for r in rows if r.priority_bucket == "low")
        print(f"  Total ranked: {total}  |  high: {high}  medium: {med}  low: {low}  |  surfaced: {surfaced}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Rank cluster-level trade ideas.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing rankings.")
    parser.add_argument("--cluster-id", type=int, default=None, help="Rank one cluster only.")
    parser.add_argument("--status", action="store_true", help="Show ranking table without rerunning.")
    parser.add_argument("--verbose", action="store_true", help="Show score components in status.")
    args = parser.parse_args()

    init_db()

    if args.status:
        _print_status(verbose=args.verbose)
        return

    triples = _load_rankable_ideas(cluster_id=args.cluster_id)
    if not triples:
        print("No rankable ideas found. Run build_market_context.py first.")
        return

    print(f"\nSignal ranking")
    print(f"  force      : {args.force}")
    print(f"  cluster_id : {args.cluster_id or 'all'}")
    print(f"  ideas found: {len(triples)}\n")

    # Group ideas by cluster_event_id so we can assign intra-cluster ranks
    groups: dict[int, list[tuple[ClusterTradeIdea, ClusterEvent, MarketContext]]] = defaultdict(list)
    for idea, event, mc in triples:
        groups[event.id].append((idea, event, mc))

    total_processed = total_cached = total_errors = 0

    for event_id, group in groups.items():
        # Determine which ideas need (re)ranking
        to_rank: list[tuple[ClusterTradeIdea, ClusterEvent, MarketContext]] = []
        for idea, event, mc in group:
            if force := args.force:
                _delete_ranking(idea.id)
                to_rank.append((idea, event, mc))
            elif not _already_ranked(idea.id):
                to_rank.append((idea, event, mc))
            else:
                total_cached += 1

        if not to_rank:
            continue

        # Score all new ideas in this cluster
        results: list[RankingResult] = []
        for idea, event, mc in to_rank:
            try:
                result = rank_idea(
                    cluster_trade_idea_id=idea.id,
                    asset=idea.asset,
                    direction=idea.direction,
                    cluster_event_id=event.id,
                    adjusted_confidence=mc.adjusted_confidence,
                    event_severity=event.severity,
                    event_novelty=event.novelty,
                    matched_rule_count=len(event.matched_rule_ids),
                    context_flags=mc.context_flags,
                    context_should_surface=mc.should_surface,
                )
                results.append(result)
                total_processed += 1
            except Exception:
                logger.exception("Failed to rank idea_id=%d", idea.id)
                total_errors += 1

        # Assign intra-cluster ranks (1 = best) and persist
        assign_ranks(results)
        _persist_rankings(results)

        # Print cluster block
        print(f"  Cluster event {event_id}  ({len(results)} idea(s) ranked)")
        print(f"  {'Rank':>4}  {'Asset':<28}  {'Dir':<6}  {'Score':>6}  {'Bucket':<8}  Surface")
        for r in results:
            print(
                f"  {r.rank:>4}  {r.asset:<28}  {r.direction:<6}  "
                f"{r.rank_score:>6.3f}  {r.priority_bucket:<8}  "
                f"{'YES' if r.should_surface else 'NO'}"
            )
        print()

    print(_SEP)
    print(f"  Newly ranked : {total_processed}")
    print(f"  Cached       : {total_cached}")
    print(f"  Errors       : {total_errors}")
    print(_SEP + "\n")


if __name__ == "__main__":
    main()
