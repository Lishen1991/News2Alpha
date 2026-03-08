"""
SQLAlchemy ORM model for signal ranking results.

One SignalRanking row per ClusterTradeIdea. Stores the composite score,
rank position within the cluster's idea set, priority bucket, surface
decision, and human-readable reasoning.

Kept separate from cluster_trade_ideas so rankings can be recomputed
independently (e.g. after changing weights) without touching idea records.
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class SignalRanking(Base):
    """
    Ranking result for one ClusterTradeIdea.

    rank is relative to other ideas within the same cluster_event.
    rank_score is an absolute composite score in [0, 1] comparable
    across clusters.
    """

    __tablename__ = "signal_rankings"
    __table_args__ = (
        UniqueConstraint("cluster_trade_idea_id", name="uq_signal_ranking_idea"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_trade_idea_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("cluster_trade_ideas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Denormalised for easy querying without joins
    asset: Mapped[str] = mapped_column(String(128), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    cluster_event_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Ranking outputs
    rank_score: Mapped[float] = mapped_column(Float, nullable=False)  # [0, 1]
    rank: Mapped[int] = mapped_column(Integer, nullable=False)         # 1 = best in cluster
    priority_bucket: Mapped[str] = mapped_column(String(16), nullable=False)  # high/medium/low
    should_surface: Mapped[bool] = mapped_column(nullable=False, default=True)

    # Component scores stored for inspection / debugging
    score_components: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    ranking_reasons: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<SignalRanking id={self.id} asset={self.asset!r} "
            f"rank={self.rank} score={self.rank_score:.3f} bucket={self.priority_bucket!r}>"
        )
