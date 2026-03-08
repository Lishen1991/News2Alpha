"""
RSS collector runner — fetches configured feeds, stores new articles in SQLite.

Runs one complete collection cycle and exits. Designed to be scheduled
via cron, Task Scheduler, or called manually during development.

Usage:
    # One collection cycle (default settings)
    py scripts/run_collector.py

    # Use a custom feed config
    py scripts/run_collector.py --config config/news_sources.yaml

    # Store articles AND auto-assign to clusters (optional pipeline hook)
    py scripts/run_collector.py --analyze

    # Dry run: fetch and normalize but do not write to DB
    py scripts/run_collector.py --dry-run

    # Verbose logging
    py scripts/run_collector.py --verbose

Example output:
    ════════════════════════════════════════════════════════════════
      NEWS2ALPHA — RSS COLLECTOR
    ════════════════════════════════════════════════════════════════
      Config      : config/news_sources.yaml
      Feeds       : 7 enabled
      Dry run     : False
      Auto-analyze: False
    ════════════════════════════════════════════════════════════════

    Fetching feeds...
      [1/7] Reuters Commodities       → 25 entries
      [2/7] Reuters Business          → 30 entries
      [3/7] BBC Business              → 15 entries  FAILED: HTTP 403
      [4/7] CNBC Economy              → 20 entries
      [5/7] MarketWatch Top Stories   → 18 entries
      [6/7] Yahoo Finance News        → 22 entries
      [7/7] Al Jazeera Economy        → 12 entries

    ────────────────────────────────────────────────────────────────
      Feeds ok      : 6 / 7
      Entries raw   : 127
      Normalized    : 124
      Inserted      : 41
      Skipped (dup) : 83
      Failed        : 0
    ════════════════════════════════════════════════════════════════

To run the full analysis pipeline on newly collected articles:
    py research/cluster_backfill.py
    py research/analyze_clusters.py
    py research/build_market_context.py
    py research/build_signal_rankings.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path before any app imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.collectors.article_normalizer import normalize_feed_entries
from app.collectors.feed_registry import load_feeds
from app.collectors.rss_collector import fetch_all_feeds
from app.db.database import init_db, migrate_collector_columns
from app.services.collector_ingestion_service import ingest_articles

_SEP  = "─" * 64
_SEP2 = "═" * 64

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "news_sources.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)-8s %(name)s: %(message)s")


def _print_header(config_path: Path, n_enabled: int, dry_run: bool, analyze: bool) -> None:
    print(f"\n{_SEP2}")
    print("  NEWS2ALPHA — RSS COLLECTOR")
    print(_SEP2)
    print(f"  Config      : {config_path.name}")
    print(f"  Feeds       : {n_enabled} enabled")
    print(f"  Dry run     : {dry_run}")
    print(f"  Auto-analyze: {analyze}")
    print(_SEP2)


# ---------------------------------------------------------------------------
# Main collection cycle
# ---------------------------------------------------------------------------

def run_collection_cycle(
    config_path: Path,
    dry_run: bool = False,
    analyze: bool = False,
) -> None:
    """
    Execute one complete collection cycle:
      1. Load feed configuration
      2. Fetch all enabled feeds
      3. Normalize entries
      4. Ingest into SQLite (unless dry_run)
      5. Optionally trigger clustering (if analyze=True and not dry_run)

    Args:
        config_path: Path to news_sources.yaml.
        dry_run:     If True, skip DB writes and only report what would happen.
        analyze:     If True, auto-assign new articles to clusters after storage.
    """
    # --- Load feeds ---
    try:
        feeds = load_feeds(config_path)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    enabled = [f for f in feeds if f.enabled]
    _print_header(config_path, len(enabled), dry_run, analyze)

    if not enabled:
        print("\n  No enabled feeds. Edit config/news_sources.yaml to enable feeds.")
        return

    # --- Fetch ---
    print("\nFetching feeds...")
    fetch_summary = fetch_all_feeds(enabled)

    for i, result in enumerate(fetch_summary.results, 1):
        status = f"{len(result.entries):>3} entries"
        error  = f"  FAILED: {result.error}" if not result.ok else ""
        print(f"  [{i:>{len(str(len(fetch_summary.results)))}}/{len(fetch_summary.results)}] "
              f"{result.feed.name:<28}  → {status}{error}")

    # --- Normalize ---
    all_normalized = []
    for result in fetch_summary.results:
        if result.ok and result.entries:
            normalized = normalize_feed_entries(result.entries, result.feed)
            all_normalized.extend(normalized)

    # --- Ingest (or dry-run report) ---
    print(f"\n{_SEP}")
    print(f"  Feeds ok      : {fetch_summary.feeds_ok} / {fetch_summary.feeds_attempted}")
    print(f"  Entries raw   : {fetch_summary.total_entries}")
    print(f"  Normalized    : {len(all_normalized)}")

    if dry_run:
        print(f"  [DRY RUN] Would insert up to {len(all_normalized)} article(s).")
        print(f"  [DRY RUN] No changes made to the database.")
    else:
        ingest_result = ingest_articles(all_normalized, analyze=analyze)
        print(f"  Inserted      : {ingest_result.inserted}")
        print(f"  Skipped (dup) : {ingest_result.skipped}")
        print(f"  Failed        : {ingest_result.failed}")

        if analyze and ingest_result.inserted > 0:
            print(f"  Auto-clustered: {ingest_result.inserted} article(s) passed to clustering.")

    print(_SEP2)

    if not dry_run and not analyze:
        print("\nTo run the full analysis pipeline on collected articles:")
        print("  py research/cluster_backfill.py")
        print("  py research/analyze_clusters.py")
        print("  py research/build_market_context.py")
        print("  py research/build_signal_rankings.py\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch RSS feeds and store new articles in SQLite."
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="Path to news_sources.yaml (default: config/news_sources.yaml)."
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Auto-assign new articles to clusters after storage."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and normalize but do not write to the database."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging."
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    # Ensure DB tables exist and collector columns are migrated
    if not args.dry_run:
        init_db()
        migrate_collector_columns()

    run_collection_cycle(
        config_path=args.config,
        dry_run=args.dry_run,
        analyze=args.analyze,
    )


if __name__ == "__main__":
    main()
