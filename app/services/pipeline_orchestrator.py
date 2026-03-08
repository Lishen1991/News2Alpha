"""
Pipeline orchestrator — connects collected articles to the full analysis pipeline.

Flow (four stages)
──────────────────
1. CLUSTERING     For each article_id → assign_cluster() via article_processing_service.
                  Idempotent: already-mapped articles return their existing cluster.

2. ANALYSIS       For each affected cluster, decide whether to run analyse_cluster():
                  - No ClusterEvent exists            → first-time analysis (force=False)
                  - ClusterEvent exists but article_count grew → re-analysis (force=True)
                  - ClusterEvent exists, count unchanged      → skip (already up to date)

3. MARKET CONTEXT For each trade idea in newly analyzed clusters that has no
                  MarketContext row yet: compute and persist MarketContext.

4. RANKINGS       For each newly analyzed cluster: (re)compute SignalRankings for
                  all its trade ideas together so intra-cluster ranks are consistent.
                  Existing rankings for the cluster are deleted and rewritten.

Idempotency
───────────
Each stage checks for existing records before writing. Running the orchestrator
twice on the same article_ids is safe — the second run is effectively a no-op
(all stages find existing records and skip).

Error handling
──────────────
A failure in any single article or cluster is logged and skipped; processing
continues for all remaining items. The PipelineResult.failed_* counters record
what went wrong without crashing the run.

Usage
─────
    from app.services.pipeline_orchestrator import run_pipeline

    result = run_pipeline(article_ids=[101, 102, 103])
    print(f"Analyzed {result.clusters_analyzed} clusters, "
          f"generated {result.ideas_generated} trade ideas.")

    # Force re-analysis of all affected clusters even if already analyzed:
    result = run_pipeline(article_ids=[101], force_analysis=True)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from app.db.cluster_analysis_models import ClusterEvent, ClusterTradeIdea
from app.db.cluster_models import EventCluster
from app.db.database import get_session
from app.db.market_context_models import MarketContext
from app.db.ranking_models import SignalRanking
from app.services.article_processing_service import ArticleProcessingResult, process_article
from app.services.cluster_analysis_service import analyse_cluster
from app.services.market_context_service import compute_market_context
from app.services.ranking_service import RankingResult, assign_ranks, rank_idea

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Aggregate outcome of one orchestration run.

    Attributes:
        articles_processed:  Total article_ids attempted.
        articles_clustered:  Articles successfully clustered.
        articles_duplicate:  Articles that were duplicates (skipped).
        articles_failed:     Articles where clustering raised an exception.
        clusters_new:        Brand-new clusters created this run.
        clusters_updated:    Existing clusters that received new articles.
        clusters_analyzed:   Clusters where analysis was run (new or re-run).
        clusters_skipped:    Clusters already up to date (analysis skipped).
        ideas_generated:     Total ClusterTradeIdea rows from analysis.
        contexts_computed:   MarketContext rows written this run.
        rankings_computed:   SignalRanking rows written this run.
        failed_clusters:     Cluster IDs where analysis or downstream failed.
    """
    articles_processed:  int = 0
    articles_clustered:  int = 0
    articles_duplicate:  int = 0
    articles_failed:     int = 0
    clusters_new:        int = 0
    clusters_updated:    int = 0
    clusters_analyzed:   int = 0
    clusters_skipped:    int = 0
    ideas_generated:     int = 0
    contexts_computed:   int = 0
    rankings_computed:   int = 0
    failed_clusters:     list[int] = field(default_factory=list)

    def log_summary(self) -> None:
        logger.info(
            "Pipeline complete: "
            "%d articles processed (%d clustered, %d dup, %d failed) | "
            "%d clusters analyzed (%d new, %d updated, %d skipped) | "
            "%d ideas | %d contexts | %d rankings",
            self.articles_processed,
            self.articles_clustered, self.articles_duplicate, self.articles_failed,
            self.clusters_analyzed, self.clusters_new, self.clusters_updated,
            self.clusters_skipped,
            self.ideas_generated,
            self.contexts_computed,
            self.rankings_computed,
        )


# ---------------------------------------------------------------------------
# Stage 2 helpers — analysis decision
# ---------------------------------------------------------------------------

def _get_cluster(cluster_id: int) -> EventCluster | None:
    """Load an EventCluster by ID (detached from session)."""
    with get_session() as session:
        cluster = session.get(EventCluster, cluster_id)
        if cluster:
            session.expunge(cluster)
        return cluster


def _get_cluster_event(cluster_id: int) -> ClusterEvent | None:
    """Return the existing ClusterEvent for a cluster, or None."""
    with get_session() as session:
        ce = session.query(ClusterEvent).filter_by(cluster_id=cluster_id).first()
        if ce:
            session.expunge(ce)
        return ce


def _needs_analysis(cluster_id: int) -> tuple[bool, bool]:
    """
    Decide whether a cluster needs (re-)analysis.

    Returns:
        (needs_analysis, is_reanalysis)
        - (True,  False) → new cluster, first-time analysis
        - (True,  True)  → existing cluster, article_count grew → re-analyze
        - (False, False) → already analyzed and unchanged → skip
    """
    cluster = _get_cluster(cluster_id)
    if cluster is None:
        return False, False

    existing = _get_cluster_event(cluster_id)
    if existing is None:
        return True, False  # no analysis yet

    if cluster.article_count > existing.article_count:
        return True, True  # cluster grew since last analysis

    return False, False  # up to date


# ---------------------------------------------------------------------------
# Stage 3 helpers — market context
# ---------------------------------------------------------------------------

def _idea_has_context(idea_id: int) -> bool:
    with get_session() as session:
        return (
            session.query(MarketContext.id)
            .filter_by(cluster_trade_idea_id=idea_id)
            .first()
        ) is not None


def _load_ideas_for_cluster(cluster_event_id: int) -> list[tuple[ClusterTradeIdea, EventCluster]]:
    """Return (idea, cluster) pairs for a ClusterEvent, detached from session."""
    with get_session() as session:
        rows = (
            session.query(ClusterTradeIdea, EventCluster)
            .join(ClusterEvent, ClusterTradeIdea.cluster_event_id == ClusterEvent.id)
            .join(EventCluster, ClusterEvent.cluster_id == EventCluster.id)
            .filter(ClusterTradeIdea.cluster_event_id == cluster_event_id)
            .all()
        )
        result = []
        expunged_cluster_ids: set[int] = set()
        for idea, cluster in rows:
            session.expunge(idea)
            if cluster.id not in expunged_cluster_ids:
                session.expunge(cluster)
                expunged_cluster_ids.add(cluster.id)
            result.append((idea, cluster))
        return result


def _persist_context(idea_id: int, ctx) -> None:
    """Write a MarketContext row. Skips if one already exists (UNIQUE constraint)."""
    with get_session() as session:
        # Guard against race / double-call
        existing = session.query(MarketContext).filter_by(cluster_trade_idea_id=idea_id).first()
        if existing:
            return
        session.add(MarketContext(
            cluster_trade_idea_id=idea_id,
            asset=ctx.asset,
            direction=ctx.direction,
            signal_date=str(ctx.signal_date),
            ret_1d_pre_signal=ctx.ret_1d_pre_signal,
            ret_3d_pre_signal=ctx.ret_3d_pre_signal,
            ret_5d_pre_signal=ctx.ret_5d_pre_signal,
            realized_vol_5d=ctx.realized_vol_5d,
            staleness_days=ctx.staleness_days,
            context_flags=ctx.context_flags,
            original_confidence=ctx.original_confidence,
            adjusted_confidence=ctx.adjusted_confidence,
            should_surface=ctx.should_surface,
        ))


def _run_market_context_for_cluster(
    cluster_event_id: int,
) -> int:
    """
    Compute and persist MarketContext for all trade ideas in a cluster
    that don't already have one.

    Returns the number of new MarketContext rows written.
    """
    pairs = _load_ideas_for_cluster(cluster_event_id)
    written = 0

    for idea, cluster in pairs:
        if _idea_has_context(idea.id):
            continue

        signal_date: date = idea.created_at.date()

        try:
            ctx = compute_market_context(
                asset=idea.asset,
                direction=idea.direction,
                original_confidence=idea.confidence,
                signal_date=signal_date,
                cluster_first_seen_at=cluster.first_seen_at,
            )
            _persist_context(idea.id, ctx)
            written += 1
        except Exception:
            logger.exception(
                "Market context failed for idea_id=%d asset=%s.", idea.id, idea.asset
            )

    return written


# ---------------------------------------------------------------------------
# Stage 4 helpers — rankings
# ---------------------------------------------------------------------------

def _load_ideas_with_context(
    cluster_event_id: int,
) -> list[tuple[ClusterTradeIdea, ClusterEvent, MarketContext]]:
    """Return (idea, event, context) triples for a ClusterEvent, detached."""
    with get_session() as session:
        rows = (
            session.query(ClusterTradeIdea, ClusterEvent, MarketContext)
            .join(ClusterEvent, ClusterTradeIdea.cluster_event_id == ClusterEvent.id)
            .join(MarketContext, MarketContext.cluster_trade_idea_id == ClusterTradeIdea.id)
            .filter(ClusterTradeIdea.cluster_event_id == cluster_event_id)
            .all()
        )
        result = []
        expunged_event_ids: set[int] = set()
        for idea, event, mc in rows:
            session.expunge(idea)
            session.expunge(mc)
            if event.id not in expunged_event_ids:
                session.expunge(event)
                expunged_event_ids.add(event.id)
            result.append((idea, event, mc))
        return result


def _delete_cluster_rankings(cluster_event_id: int) -> None:
    """Remove all SignalRanking rows for a cluster so they can be rewritten."""
    with get_session() as session:
        idea_ids = [
            row[0] for row in
            session.query(ClusterTradeIdea.id)
            .filter_by(cluster_event_id=cluster_event_id)
            .all()
        ]
        if idea_ids:
            session.query(SignalRanking).filter(
                SignalRanking.cluster_trade_idea_id.in_(idea_ids)
            ).delete(synchronize_session=False)


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


def _run_rankings_for_cluster(cluster_event_id: int) -> int:
    """
    (Re)compute SignalRankings for all trade ideas in a cluster.

    Deletes existing rankings first so intra-cluster ranks are always
    consistent (no stale rank-1 from a previous smaller idea set).

    Returns the number of SignalRanking rows written.
    """
    triples = _load_ideas_with_context(cluster_event_id)
    if not triples:
        return 0

    # Delete stale rankings before rewriting
    _delete_cluster_rankings(cluster_event_id)

    results: list[RankingResult] = []
    for idea, event, mc in triples:
        try:
            r = rank_idea(
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
            results.append(r)
        except Exception:
            logger.exception("Ranking failed for idea_id=%d.", idea.id)

    assign_ranks(results)
    _persist_rankings(results)
    return len(results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    article_ids: list[int],
    force_analysis: bool = False,
) -> PipelineResult:
    """
    Run the full analysis pipeline for a list of newly stored article IDs.

    Stages:
        1. Clustering     — assign each article to a cluster.
        2. Analysis       — extract events and trade ideas for affected clusters.
        3. Market context — compute price features and flags for new trade ideas.
        4. Rankings       — score and rank trade ideas within each cluster.

    Args:
        article_ids:     List of ArticleRecord PKs to process.
        force_analysis:  If True, force re-analysis of all affected clusters
                         even if they were already analyzed and unchanged.
                         Default False (idempotent / skip-if-current).

    Returns:
        PipelineResult with per-stage counts.
    """
    result = PipelineResult(articles_processed=len(article_ids))

    if not article_ids:
        return result

    # ── Stage 1: Clustering ───────────────────────────────────────────────
    logger.info("Pipeline stage 1: clustering %d article(s).", len(article_ids))

    # cluster_id → set of article results (to detect updated vs new clusters)
    cluster_articles: dict[int, list[ArticleProcessingResult]] = defaultdict(list)

    for article_id in article_ids:
        art_result = process_article(article_id)

        if art_result.status == "clustered":
            result.articles_clustered += 1
            if art_result.is_new_cluster:
                result.clusters_new += 1
            else:
                result.clusters_updated += 1
            if art_result.cluster_id is not None:
                cluster_articles[art_result.cluster_id].append(art_result)

        elif art_result.status == "duplicate":
            result.articles_duplicate += 1

        else:  # 'failed'
            result.articles_failed += 1
            logger.warning(
                "Article %d failed clustering: %s", article_id, art_result.error
            )

    affected_cluster_ids = list(cluster_articles.keys())
    logger.info(
        "Stage 1 complete: %d article(s) → %d cluster(s) affected.",
        result.articles_clustered, len(affected_cluster_ids),
    )

    # ── Stage 2: Cluster analysis ─────────────────────────────────────────
    logger.info("Pipeline stage 2: analyzing %d cluster(s).", len(affected_cluster_ids))

    analyzed_cluster_event_ids: list[int] = []

    for cluster_id in affected_cluster_ids:
        try:
            needs, is_reanalysis = _needs_analysis(cluster_id)

            if force_analysis:
                needs = True

            if not needs:
                result.clusters_skipped += 1
                logger.debug("Cluster %d is up to date — skipping analysis.", cluster_id)
                continue

            analysis = analyse_cluster(cluster_id, force=is_reanalysis or force_analysis)
            if analysis is None:
                logger.warning("analyse_cluster returned None for cluster_id=%d.", cluster_id)
                result.failed_clusters.append(cluster_id)
                continue

            result.clusters_analyzed += 1
            result.ideas_generated += analysis.idea_count
            analyzed_cluster_event_ids.append(analysis.cluster_event_id)

            logger.info(
                "Cluster %d analyzed: event_type=%s ideas=%d %s.",
                cluster_id, analysis.event_type, analysis.idea_count,
                "(re-analysis)" if is_reanalysis or force_analysis else "(first-time)",
            )

        except Exception:
            logger.exception("Analysis failed for cluster_id=%d.", cluster_id)
            result.failed_clusters.append(cluster_id)

    logger.info(
        "Stage 2 complete: %d analyzed, %d skipped, %d failed.",
        result.clusters_analyzed, result.clusters_skipped, len(result.failed_clusters),
    )

    if not analyzed_cluster_event_ids:
        result.log_summary()
        return result

    # ── Stage 3: Market context ───────────────────────────────────────────
    logger.info(
        "Pipeline stage 3: market context for %d cluster event(s).",
        len(analyzed_cluster_event_ids),
    )

    for cluster_event_id in analyzed_cluster_event_ids:
        try:
            written = _run_market_context_for_cluster(cluster_event_id)
            result.contexts_computed += written
        except Exception:
            logger.exception(
                "Market context stage failed for cluster_event_id=%d.", cluster_event_id
            )

    logger.info("Stage 3 complete: %d market context record(s) written.", result.contexts_computed)

    # ── Stage 4: Rankings ─────────────────────────────────────────────────
    logger.info(
        "Pipeline stage 4: rankings for %d cluster event(s).",
        len(analyzed_cluster_event_ids),
    )

    for cluster_event_id in analyzed_cluster_event_ids:
        try:
            written = _run_rankings_for_cluster(cluster_event_id)
            result.rankings_computed += written
        except Exception:
            logger.exception(
                "Rankings stage failed for cluster_event_id=%d.", cluster_event_id
            )

    logger.info("Stage 4 complete: %d ranking record(s) written.", result.rankings_computed)

    result.log_summary()
    return result
