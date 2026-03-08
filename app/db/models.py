"""
SQLAlchemy ORM models.

Table layout:
    articles    – raw article inputs
    events      – one extracted event per article
    trade_ideas – one row per (event, asset, direction) pair

JSON list fields (channels, entities, etc.) are stored as JSON strings
using SQLAlchemy's JSON type, which maps to TEXT in SQLite and is
automatically serialised / deserialised by the ORM.
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class ArticleRecord(Base):
    """Persisted representation of an ArticleInput payload."""

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[str] = mapped_column(String(256), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    # --- Collector fields (added by RSS collector; nullable for backwards compat) ---
    collected_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    normalized_title_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    ingestion_source: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # One article → one event (cascade delete keeps DB consistent)
    event: Mapped[EventRecord | None] = relationship(
        "EventRecord", back_populates="article", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ArticleRecord id={self.id} title={self.title!r}>"


class EventRecord(Base):
    """Persisted representation of an ExtractedEvent."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[float] = mapped_column(Float, nullable=False)
    novelty: Mapped[float] = mapped_column(Float, nullable=False)
    # List fields stored as JSON arrays
    channels: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    time_horizon: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    article: Mapped[ArticleRecord] = relationship("ArticleRecord", back_populates="event")
    trade_ideas: Mapped[list[TradeIdeaRecord]] = relationship(
        "TradeIdeaRecord", back_populates="event", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<EventRecord id={self.id} event_type={self.event_type!r}>"


class TradeIdeaRecord(Base):
    """Persisted representation of a TradeIdea."""

    __tablename__ = "trade_ideas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asset: Mapped[str] = mapped_column(String(128), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    horizon: Mapped[str] = mapped_column(String(32), nullable=False)
    thesis: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    event: Mapped[EventRecord] = relationship("EventRecord", back_populates="trade_ideas")

    def __repr__(self) -> str:
        return f"<TradeIdeaRecord id={self.id} asset={self.asset!r} direction={self.direction!r}>"
