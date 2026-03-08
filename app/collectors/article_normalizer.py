"""
Article normalizer — converts raw feedparser entries into a clean,
deterministic NormalizedArticle ready for storage and deduplication.

Design
──────
- All normalization is pure (no DB access, no network calls).
- HTML is stripped using only stdlib html.parser — no extra dependencies.
- normalized_title and normalized_title_hash are computed deterministically
  so the same article title always produces the same hash regardless of
  how it arrived.
- Fields missing from a feed entry are filled with sensible defaults
  rather than raising an exception, so one bad entry doesn't abort a run.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser

from app.collectors.feed_registry import Feed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NormalizedArticle
# ---------------------------------------------------------------------------

@dataclass
class NormalizedArticle:
    """
    A single article normalized from a feed entry, ready for deduplication
    and database storage.

    Fields mirror the ArticleRecord columns so ingestion is a direct mapping.
    """
    title:                  str
    source:                 str        # feed.name
    url:                    str | None
    body:                   str        # full content or summary, HTML-stripped
    published_at:           str | None # ISO-8601 UTC string, e.g. "2025-04-15T08:00:00+00:00"
    collected_at:           str        # ISO-8601 UTC string of when we fetched it
    category:               str | None # from feed config
    normalized_title:       str        # lowercase, stripped, whitespace-collapsed
    normalized_title_hash:  str        # SHA-256 of normalized_title (hex, 64 chars)
    ingestion_source:       str = "rss"


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Minimal HTML stripper using stdlib only."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(raw: str) -> str:
    """
    Remove HTML tags and decode HTML entities from a string.

    Returns plain text with collapsed whitespace.
    Does not raise on malformed HTML.
    """
    try:
        stripper = _HTMLStripper()
        stripper.feed(html.unescape(raw or ""))
        text = stripper.get_text()
    except Exception:
        # Fallback: crude tag removal
        text = re.sub(r"<[^>]+>", " ", raw or "")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Title normalization
# ---------------------------------------------------------------------------

def normalize_title(title: str) -> str:
    """
    Produce a canonical, lowercase, punctuation-stripped title for hashing.

    Steps:
      1. Unicode NFKD normalization (handles accented chars)
      2. Lowercase
      3. Remove all non-alphanumeric, non-space characters
      4. Collapse whitespace and strip leading/trailing space
    """
    title = unicodedata.normalize("NFKD", title)
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_hash(normalized: str) -> str:
    """SHA-256 hex digest of a normalized title string (64 chars)."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def _parse_time_struct(struct: time.struct_time | None) -> str | None:
    """Convert a feedparser time_struct (UTC assumed) to ISO-8601 string."""
    if struct is None:
        return None
    try:
        dt = datetime(*struct[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Body extraction
# ---------------------------------------------------------------------------

def _extract_body(entry: object) -> str:
    """
    Extract the best available body text from a feedparser entry.

    Priority:
      1. entry.content[0].value  — full article text (some feeds)
      2. entry.summary           — abstract / excerpt (most feeds)
      3. entry.title             — last resort (avoids empty body)
    """
    # Try full content first
    content = getattr(entry, "content", None)
    if content and isinstance(content, list) and content[0].get("value"):
        return strip_html(content[0]["value"])

    summary = getattr(entry, "summary", None) or getattr(entry, "description", None)
    if summary:
        return strip_html(summary)

    # Fall back to title — better than empty string
    return strip_html(getattr(entry, "title", "") or "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_entry(entry: object, feed: Feed, collected_at: str) -> NormalizedArticle | None:
    """
    Normalize a single feedparser entry into a NormalizedArticle.

    Args:
        entry:        A feedparser entry object.
        feed:         The Feed that produced this entry.
        collected_at: ISO timestamp of when the collection run started.

    Returns:
        NormalizedArticle, or None if the entry is too malformed to use
        (e.g. missing both title and URL).
    """
    raw_title = getattr(entry, "title", None) or ""
    raw_title = strip_html(raw_title).strip()

    url: str | None = (
        getattr(entry, "link", None)
        or getattr(entry, "id", None)
        or None
    )
    if url:
        url = url.strip() or None

    if not raw_title and not url:
        logger.debug("Skipping entry with no title and no URL from feed '%s'.", feed.name)
        return None

    # Fill missing title from URL fragment
    if not raw_title and url:
        raw_title = url.split("/")[-1].replace("-", " ").replace("_", " ")[:120]

    body = _extract_body(entry)
    if not body:
        body = raw_title  # body must not be empty (DB constraint)

    published_struct = (
        getattr(entry, "published_parsed", None)
        or getattr(entry, "updated_parsed", None)
    )
    published_at = _parse_time_struct(published_struct)

    norm_title = normalize_title(raw_title)
    norm_hash  = title_hash(norm_title)

    return NormalizedArticle(
        title=raw_title[:512],
        source=feed.name[:256],
        url=url[:2048] if url else None,
        body=body,
        published_at=published_at,
        collected_at=collected_at,
        category=feed.category or None,
        normalized_title=norm_title,
        normalized_title_hash=norm_hash,
        ingestion_source="rss",
    )


def normalize_feed_entries(
    entries: list[object],
    feed: Feed,
    collected_at: str | None = None,
) -> list[NormalizedArticle]:
    """
    Normalize all entries from a single feed.

    Skips malformed entries and logs a warning for each one skipped.

    Args:
        entries:      List of feedparser entry objects.
        feed:         The source Feed.
        collected_at: ISO timestamp; defaults to current UTC time.

    Returns:
        List of NormalizedArticle (only successfully normalized entries).
    """
    ts = collected_at or _utcnow_iso()
    results: list[NormalizedArticle] = []

    for entry in entries:
        try:
            article = normalize_entry(entry, feed, ts)
            if article is not None:
                results.append(article)
        except Exception:
            logger.warning(
                "Failed to normalize entry from feed '%s': %s",
                feed.name,
                getattr(entry, "title", "<no title>"),
                exc_info=True,
            )

    return results
