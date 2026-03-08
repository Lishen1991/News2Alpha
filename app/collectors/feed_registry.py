"""
Feed registry — loads and validates RSS feed configurations from YAML.

The registry is the single source of truth for which feeds are active.
All other collector components receive Feed objects from here rather than
reading YAML directly, keeping configuration parsing in one place.

Usage:
    from app.collectors.feed_registry import load_feeds, Feed

    feeds = load_feeds()          # loads from default config path
    feeds = load_feeds(path)      # loads from explicit path
    enabled = [f for f in feeds if f.enabled]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config path relative to project root
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "news_sources.yaml"


@dataclass(frozen=True)
class Feed:
    """
    A single configured RSS/Atom feed.

    Attributes:
        name:     Human-readable label, used as article.source.
        url:      Full feed URL.
        category: Optional grouping tag (e.g. 'energy', 'macro', 'markets').
        enabled:  If False the feed is skipped during collection.
    """
    name:     str
    url:      str
    category: str  = "general"
    enabled:  bool = True

    def __str__(self) -> str:
        status = "ON " if self.enabled else "OFF"
        return f"[{status}] {self.name} ({self.category}) → {self.url}"


def load_feeds(config_path: Path | None = None) -> list[Feed]:
    """
    Load feed configurations from a YAML file.

    Args:
        config_path: Path to news_sources.yaml. Defaults to
                     config/news_sources.yaml at the project root.

    Returns:
        List of Feed objects, including disabled ones.
        Call [f for f in feeds if f.enabled] to get only active feeds.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML structure is invalid.
    """
    path = config_path or _DEFAULT_CONFIG

    if not path.exists():
        raise FileNotFoundError(
            f"Feed config not found: {path}\n"
            "Expected: config/news_sources.yaml"
        )

    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    raw_feeds = raw.get("feeds", [])
    if not isinstance(raw_feeds, list):
        raise ValueError(f"'feeds' in {path} must be a list, got {type(raw_feeds)}")

    feeds: list[Feed] = []
    for i, entry in enumerate(raw_feeds):
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict feed entry at index %d", i)
            continue

        name = entry.get("name", "").strip()
        url  = entry.get("url",  "").strip()

        if not name:
            logger.warning("Feed at index %d has no 'name', skipping.", i)
            continue
        if not url:
            logger.warning("Feed '%s' has no 'url', skipping.", name)
            continue

        feeds.append(Feed(
            name=name,
            url=url,
            category=str(entry.get("category", "general")).strip(),
            enabled=bool(entry.get("enabled", True)),
        ))

    enabled_count = sum(1 for f in feeds if f.enabled)
    logger.info(
        "Loaded %d feed(s) from %s (%d enabled).",
        len(feeds), path.name, enabled_count,
    )
    return feeds
