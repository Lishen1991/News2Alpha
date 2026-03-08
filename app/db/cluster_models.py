"""
SQLAlchemy ORM models for article clustering.

Tables:
    event_clusters      – one row per identified event cluster
    article_cluster_map – many-to-one mapping of articles to clusters
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class EventCluster(Base):
    """
    A group of articles that describe the same underlying real-world event.

    cluster_key is a stable slug derived from event_type + a normalised title
    fragment so it is human-readable when inspecting the DB.
    """

    __tablename__ = "event_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_key: Mapped[str] = mapped_column(String(256), nullable=False, unique=True, index=True)
    cluster_title: Mapped[str] = mapped_column(String(512), nullable=False)
    first_seen_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    members: Mapped[list[ArticleClusterMap]] = relationship(
        "ArticleClusterMap", back_populates="cluster", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<EventCluster id={self.id} key={self.cluster_key!r} n={self.article_count}>"


class ArticleClusterMap(Base):
    """
    Maps an article to its assigned cluster.

    Each article belongs to at most one cluster (enforced by unique constraint).
    is_duplicate=1 means the article was detected as a near-duplicate of the
    cluster's seed article and should be excluded from backtests.
    """

    __tablename__ = "article_cluster_map"
    __table_args__ = (UniqueConstraint("article_id", name="uq_article_cluster"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cluster_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("event_clusters.id", ondelete="CASCADE"), nullable=False, index=True
    )
    similarity_score: Mapped[float | None] = mapped_column(nullable=True)
    is_duplicate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 0=no, 1=yes
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    cluster: Mapped[EventCluster] = relationship("EventCluster", back_populates="members")

    def __repr__(self) -> str:
        return (
            f"<ArticleClusterMap article_id={self.article_id} "
            f"cluster_id={self.cluster_id} dup={self.is_duplicate}>"
        )
