"""
RSS collector — fetches entries from a list of configured feeds using feedparser.

Design
──────
- One function per concern: fetch_feed() handles one feed, fetch_all_feeds()
  orchestrates across all enabled feeds.
- Network and parsing errors are caught per-feed so one bad feed never
  aborts the whole run.
- Returns raw feedparser entry objects; normalization is handled by
  article_normalizer.py, keeping this module focused on I/O only.
- feedparser is the only new dependency added by the collector.

feedparser returns entries with these commonly useful fields:
    entry.title           str
    entry.link            str (canonical URL)
    entry.id              str (often same as link)
    entry.published       str (raw date string from feed)
    entry.published_parsed time.struct_time (UTC, parsed by feedparser)
    entry.updated_parsed  time.struct_time (fallback if published missing)
    entry.summary         str (HTML, excerpt or abstract)
    entry.content         list[dict] (full text, only some feeds)
    entry.tags            list[dict] (with 'term' key)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import feedparser

from app.collectors.feed_registry import Feed

logger = logging.getLogger(__name__)

# Maximum entries to accept per feed per run.
# Prevents a firehose feed from dominating the collector.
MAX_ENTRIES_PER_FEED: int = 50


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FeedFetchResult:
    """
    Outcome of fetching one feed.

    Attributes:
        feed:       The Feed that was fetched.
        entries:    Raw feedparser entry objects (empty list on error).
        ok:         True if the feed was fetched without a fatal error.
        error:      Human-readable error description if ok=False.
        http_status: HTTP status code returned by the server (0 if unknown).
    """
    feed:        Feed
    entries:     list[object] = field(default_factory=list)
    ok:          bool         = True
    error:       str          = ""
    http_status: int          = 0


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------

def fetch_feed(feed: Feed, timeout: int = 15) -> FeedFetchResult:
    """
    Fetch and parse one RSS/Atom feed.

    Args:
        feed:    The Feed to fetch (must have enabled=True before calling).
        timeout: Request timeout in seconds.

    Returns:
        FeedFetchResult with entries populated on success, error set on failure.
    """
    logger.info("Fetching feed: %s → %s", feed.name, feed.url)

    try:
        parsed = feedparser.parse(feed.url, request_headers={"User-Agent": "News2Alpha/1.0"})
    except Exception as exc:
        logger.warning("Network error fetching '%s': %s", feed.name, exc)
        return FeedFetchResult(feed=feed, ok=False, error=str(exc))

    # feedparser doesn't raise on HTTP errors — check bozo and status
    http_status: int = getattr(parsed, "status", 0)

    if getattr(parsed, "bozo", False) and not parsed.entries:
        bozo_exc = getattr(parsed, "bozo_exception", None)
        error_msg = str(bozo_exc) if bozo_exc else "malformed feed"
        logger.warning("Malformed feed '%s' (status=%d): %s", feed.name, http_status, error_msg)
        return FeedFetchResult(
            feed=feed,
            ok=False,
            error=error_msg,
            http_status=http_status,
        )

    if http_status and http_status >= 400:
        error_msg = f"HTTP {http_status}"
        logger.warning("Feed '%s' returned %s", feed.name, error_msg)
        return FeedFetchResult(feed=feed, ok=False, error=error_msg, http_status=http_status)

    entries = list(parsed.entries)[:MAX_ENTRIES_PER_FEED]
    logger.info(
        "Fetched %d entries from '%s' (HTTP %d).",
        len(entries), feed.name, http_status or 200,
    )
    return FeedFetchResult(feed=feed, entries=entries, ok=True, http_status=http_status)


# ---------------------------------------------------------------------------
# Multi-feed orchestration
# ---------------------------------------------------------------------------

@dataclass
class CollectionSummary:
    """Aggregate result of one collection cycle across all feeds."""
    feeds_attempted:  int = 0
    feeds_ok:         int = 0
    feeds_failed:     int = 0
    total_entries:    int = 0
    results:          list[FeedFetchResult] = field(default_factory=list)


def fetch_all_feeds(feeds: list[Feed]) -> CollectionSummary:
    """
    Fetch all enabled feeds and return their raw entries.

    Disabled feeds are silently skipped.
    Failed feeds are recorded in the summary but do not abort the run.

    Args:
        feeds: Full list of Feed objects (enabled and disabled).

    Returns:
        CollectionSummary with per-feed results and aggregate counts.
    """
    summary = CollectionSummary()

    enabled = [f for f in feeds if f.enabled]
    if not enabled:
        logger.warning("No enabled feeds found. Check config/news_sources.yaml.")
        return summary

    for feed in enabled:
        summary.feeds_attempted += 1
        result = fetch_feed(feed)
        summary.results.append(result)

        if result.ok:
            summary.feeds_ok += 1
            summary.total_entries += len(result.entries)
        else:
            summary.feeds_failed += 1

    logger.info(
        "Collection cycle complete: %d/%d feeds ok, %d total entries.",
        summary.feeds_ok,
        summary.feeds_attempted,
        summary.total_entries,
    )
    return summary
