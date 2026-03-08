"""
SQLAlchemy ORM models for cluster-level analysis results.

Kept separate from article-level models (app/db/models.py) so the two
levels of analysis remain independently queryable and neither pollutes
the other with nullable foreign keys.

Tables:
    cluster_events      – one extracted event per cluster
    cluster_trade_ideas – one row per (cluster_event, asset, direction)
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class ClusterEvent(Base):
    """
    A single extracted investment event for an entire cluster.

    The unique constraint on cluster_id enforces one analysis per cluster.
    To rerun, delete the existing ClusterEvent (cascade removes ideas too).
    """

    __tablename__ = "cluster_events"
    __table_args__ = (UniqueConstraint("cluster_id", name="uq_cluster_event"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("event_clusters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Mirrors ExtractedEvent fields
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[float] = mapped_column(Float, nullable=False)
    novelty: Mapped[float] = mapped_column(Float, nullable=False)
    channels: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    time_horizon: Mapped[str] = mapped_column(String(32), nullable=False)
    affected_sectors: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    entities: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    region: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Cluster-level metadata
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    matched_rule_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    trade_ideas: Mapped[list[ClusterTradeIdea]] = relationship(
        "ClusterTradeIdea", back_populates="cluster_event", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<ClusterEvent id={self.id} cluster_id={self.cluster_id} "
            f"event_type={self.event_type!r}>"
        )


class ClusterTradeIdea(Base):
    """One candidate trade idea produced from a cluster-level analysis."""

    __tablename__ = "cluster_trade_ideas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("cluster_events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Mirrors TradeIdea fields
    asset: Mapped[str] = mapped_column(String(128), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    horizon: Mapped[str] = mapped_column(String(32), nullable=False)
    thesis: Mapped[str] = mapped_column(Text, nullable=False)
    supporting_factors: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    risk_flags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    invalidation_conditions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    cluster_event: Mapped[ClusterEvent] = relationship(
        "ClusterEvent", back_populates="trade_ideas"
    )

    def __repr__(self) -> str:
        return (
            f"<ClusterTradeIdea id={self.id} asset={self.asset!r} "
            f"direction={self.direction!r}>"
        )
