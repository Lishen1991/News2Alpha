"""
SQLAlchemy ORM model for market context results.

One MarketContext row is written per ClusterTradeIdea. It stores the
pre-signal price features, derived flags, adjusted confidence, and the
final surface decision.

Kept in its own table (not columns on cluster_trade_ideas) so:
  - market context can be rerun independently without touching idea records
  - the ideas table stays clean and schema-stable
  - context records are easy to query, inspect, and join
"""

from __future__ import annotations

import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class MarketContext(Base):
    """
    Market context evaluation for one ClusterTradeIdea.

    The UNIQUE constraint on cluster_trade_idea_id enforces one context
    record per idea. To rerun, delete the existing row first (or use the
    --force flag in the research script).
    """

    __tablename__ = "market_context"
    __table_args__ = (
        UniqueConstraint("cluster_trade_idea_id", name="uq_market_context_idea"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_trade_idea_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("cluster_trade_ideas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Signal metadata (denormalised for easy querying)
    asset: Mapped[str] = mapped_column(String(128), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    signal_date: Mapped[str] = mapped_column(String(32), nullable=False)  # ISO date string

    # Pre-signal price features (None = insufficient data)
    ret_1d_pre_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret_3d_pre_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret_5d_pre_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_vol_5d: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Staleness (calendar days from cluster first_seen_at to idea created_at)
    staleness_days: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Derived flags — stored as a JSON list of string labels
    context_flags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Confidence adjustment
    original_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Surface decision
    should_surface: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<MarketContext id={self.id} asset={self.asset!r} "
            f"direction={self.direction!r} surface={self.should_surface}>"
        )
