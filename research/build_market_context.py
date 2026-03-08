"""
Market context research script.

For every ClusterTradeIdea in the database:
  1. Loads the cluster's first_seen_at (staleness anchor).
  2. Computes pre-signal price features from local CSV files.
  3. Derives context flags and adjusted confidence.
  4. Decides whether the idea should be surfaced.
  5. Persists a MarketContext row linked to the ClusterTradeIdea.

Usage:
    # Process all ideas not yet evaluated
    py research/build_market_context.py

    # Force rerun all ideas (overwrites existing context rows)
    py research/build_market_context.py --force

    # Process a single cluster trade idea by ID
    py research/build_market_context.py --idea-id 3

    # Print a summary of existing context records without running
    py research/build_market_context.py --status
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.cluster_analysis_models import ClusterEvent, ClusterTradeIdea
from app.db.cluster_models import EventCluster
from app.db.database import get_session, init_db
from app.db.market_context_models import MarketContext
from app.services.market_context_service import compute_market_context

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

_SEP = "─" * 76


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _load_ideas_with_cluster(
    idea_id: int | None = None,
) -> list[tuple[ClusterTradeIdea, EventCluster]]:
    """
    Return (ClusterTradeIdea, EventCluster) pairs ordered by idea.created_at.
    If idea_id is provided, return only that idea.
    """
    with get_session() as session:
        q = (
            session.query(ClusterTradeIdea, EventCluster)
            .join(ClusterEvent, ClusterTradeIdea.cluster_event_id == ClusterEvent.id)
            .join(EventCluster, ClusterEvent.cluster_id == EventCluster.id)
        )
        if idea_id is not None:
            q = q.filter(ClusterTradeIdea.id == idea_id)
        rows = q.order_by(ClusterTradeIdea.created_at.asc()).all()

        # Expunge so objects survive session close.
        # The same EventCluster instance can appear multiple times in the join
        # result (one per idea), so track which cluster IDs are already expelled.
        results = []
        expunged_cluster_ids: set[int] = set()
        for idea, cluster in rows:
            session.expunge(idea)
            if cluster.id not in expunged_cluster_ids:
                session.expunge(cluster)
                expunged_cluster_ids.add(cluster.id)
            results.append((idea, cluster))
        return results


def _already_processed(idea_id: int) -> bool:
    with get_session() as session:
        return (
            session.query(MarketContext)
            .filter_by(cluster_trade_idea_id=idea_id)
            .first()
        ) is not None


def _delete_existing_context(idea_id: int) -> None:
    with get_session() as session:
        session.query(MarketContext).filter_by(cluster_trade_idea_id=idea_id).delete()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _persist_context(idea_id: int, result) -> None:
    with get_session() as session:
        session.add(MarketContext(
            cluster_trade_idea_id=idea_id,
            asset=result.asset,
            direction=result.direction,
            signal_date=result.signal_date.isoformat(),
            ret_1d_pre_signal=result.ret_1d_pre_signal,
            ret_3d_pre_signal=result.ret_3d_pre_signal,
            ret_5d_pre_signal=result.ret_5d_pre_signal,
            realized_vol_5d=result.realized_vol_5d,
            staleness_days=result.staleness_days,
            context_flags=result.context_flags,
            original_confidence=result.original_confidence,
            adjusted_confidence=result.adjusted_confidence,
            should_surface=result.should_surface,
        ))


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------
def _print_status() -> None:
    with get_session() as session:
        ideas = session.query(ClusterTradeIdea).all()
        contexts = {
            mc.cluster_trade_idea_id: mc
            for mc in session.query(MarketContext).all()
        }

        if not ideas:
            print("No cluster trade ideas found. Run analyze_clusters.py first.")
            return

        print(f"\n{_SEP}")
        print(
            f"{'ID':>4}  {'Asset':<28}  {'Dir':<6}  {'Conf':>6}  "
            f"{'AdjConf':>8}  {'Flags':<30}  {'Surface'}"
        )
        print(_SEP)
        for idea in ideas:
            mc = contexts.get(idea.id)
            if mc:
                flags_str = ", ".join(mc.context_flags) if mc.context_flags else "none"
                print(
                    f"{idea.id:>4}  {idea.asset:<28}  {idea.direction:<6}  "
                    f"{idea.confidence:>6.2f}  {mc.adjusted_confidence:>8.2f}  "
                    f"{flags_str:<30}  {'YES' if mc.should_surface else 'NO'}"
                )
            else:
                print(
                    f"{idea.id:>4}  {idea.asset:<28}  {idea.direction:<6}  "
                    f"{idea.confidence:>6.2f}  {'(not run)':>8}  "
                    f"{'':30}  -"
                )
        print(_SEP)
        evaluated = len(contexts)
        surfaced = sum(1 for mc in contexts.values() if mc.should_surface)
        print(f"  Total ideas : {len(ideas)}")
        print(f"  Evaluated   : {evaluated}")
        print(f"  Surfaced    : {surfaced}\n")


# ---------------------------------------------------------------------------
# Per-idea processing
# ---------------------------------------------------------------------------
def _process_idea(
    idea: ClusterTradeIdea,
    cluster: EventCluster,
    force: bool,
) -> str:
    """
    Compute and persist market context for one idea.
    Returns a status string: 'PROCESSED', 'CACHED', or 'ERROR'.
    """
    if not force and _already_processed(idea.id):
        return "CACHED"

    if force:
        _delete_existing_context(idea.id)

    signal_date: date = idea.created_at.date()

    first_seen = cluster.first_seen_at
    if isinstance(first_seen, str):
        first_seen = datetime.fromisoformat(first_seen)
    if first_seen.tzinfo is None:
        first_seen = first_seen.replace(tzinfo=timezone.utc)

    try:
        result = compute_market_context(
            asset=idea.asset,
            direction=idea.direction,
            original_confidence=idea.confidence,
            signal_date=signal_date,
            cluster_first_seen_at=first_seen,
        )
        _persist_context(idea.id, result)
        return "PROCESSED"
    except Exception:
        logger.exception("Failed to process idea_id=%d", idea.id)
        return "ERROR"


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------
def _fmt_pct(v: float | None) -> str:
    return f"{v:+.2%}" if v is not None else "  n/a  "


def _print_idea_detail(idea: ClusterTradeIdea) -> None:
    with get_session() as session:
        mc = session.query(MarketContext).filter_by(cluster_trade_idea_id=idea.id).first()
        if not mc:
            return
        print(f"\n  asset              : {mc.asset}")
        print(f"  direction          : {mc.direction}")
        print(f"  signal_date        : {mc.signal_date}")
        print(f"  staleness_days     : {mc.staleness_days}")
        print(f"  ret_1d_pre_signal  : {_fmt_pct(mc.ret_1d_pre_signal)}")
        print(f"  ret_3d_pre_signal  : {_fmt_pct(mc.ret_3d_pre_signal)}")
        print(f"  ret_5d_pre_signal  : {_fmt_pct(mc.ret_5d_pre_signal)}")
        print(f"  realized_vol_5d    : {_fmt_pct(mc.realized_vol_5d)}")
        print(f"  context_flags      : {mc.context_flags or 'none'}")
        print(f"  original_conf      : {mc.original_confidence:.4f}")
        print(f"  adjusted_conf      : {mc.adjusted_confidence:.4f}")
        print(f"  should_surface     : {'YES' if mc.should_surface else 'NO'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute market context for cluster-level trade ideas."
    )
    parser.add_argument("--force", action="store_true", help="Rerun even if already processed.")
    parser.add_argument("--idea-id", type=int, default=None, help="Process a single idea by ID.")
    parser.add_argument("--status", action="store_true", help="Show context status table.")
    args = parser.parse_args()

    init_db()

    if args.status:
        _print_status()
        return

    pairs = _load_ideas_with_cluster(idea_id=args.idea_id)
    if not pairs:
        msg = f"Idea {args.idea_id} not found." if args.idea_id else "No cluster trade ideas found."
        print(msg)
        return

    print(f"\nMarket context filter")
    print(f"  force   : {args.force}")
    print(f"  idea_id : {args.idea_id or 'all'}")
    print(f"\n{_SEP}")
    print(
        f"{'ID':>4}  {'Status':<10}  {'Asset':<28}  {'Dir':<6}  "
        f"{'3d ret':>8}  {'Vol':>7}  {'Flags':<25}  {'Conf→Adj':>10}  Surface"
    )
    print(_SEP)

    processed = cached = errors = surfaced = suppressed = 0

    for idea, cluster in pairs:
        status = _process_idea(idea, cluster, args.force)

        if status == "PROCESSED":
            processed += 1
        elif status == "CACHED":
            cached += 1
        else:
            errors += 1

        # Read back for display
        with get_session() as session:
            mc = session.query(MarketContext).filter_by(cluster_trade_idea_id=idea.id).first()
            if mc:
                flags_str = ",".join(mc.context_flags) if mc.context_flags else "none"
                conf_str = f"{mc.original_confidence:.2f}→{mc.adjusted_confidence:.2f}"
                surface_str = "YES" if mc.should_surface else "NO"
                ret3_str = _fmt_pct(mc.ret_3d_pre_signal)
                vol_str  = _fmt_pct(mc.realized_vol_5d)
                if mc.should_surface:
                    surfaced += 1
                else:
                    suppressed += 1
            else:
                flags_str = conf_str = ret3_str = vol_str = surface_str = "-"

        print(
            f"{idea.id:>4}  {status:<10}  {idea.asset:<28}  {idea.direction:<6}  "
            f"{ret3_str:>8}  {vol_str:>7}  {flags_str:<25}  {conf_str:>10}  {surface_str}"
        )

        if args.idea_id:
            _print_idea_detail(idea)

    print(_SEP)
    print(f"\n  Processed  : {processed}")
    print(f"  Cached     : {cached}")
    print(f"  Errors     : {errors}")
    print(f"  Surfaced   : {surfaced}")
    print(f"  Suppressed : {suppressed}\n")


if __name__ == "__main__":
    main()
