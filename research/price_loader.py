"""
Price loader — reads per-asset CSV files from research/prices/ and returns
a tidy DataFrame indexed by date.

CSV format (one file per asset):
    date,price
    2026-01-01,75.32
    2026-01-02,76.10
    ...

File naming convention:
    research/prices/{asset_name}.csv

    e.g.  research/prices/xle.csv
          research/prices/brent_crude_front.csv

Asset names are normalised to lowercase before file lookup so the caller
does not need to worry about casing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Resolved once at import time — works regardless of the working directory
_PRICES_DIR = Path(__file__).parent / "prices"

# Map asset identifiers used in the database to their CSV filenames.
# Extend this dict when new assets are added to the rules engine.
ASSET_FILE_MAP: dict[str, str] = {
    "brent_crude_front": "brent_crude_front.csv",
    "wti_crude_front":   "wti_crude_front.csv",
    "xle":               "xle.csv",
    "xli":               "xli.csv",
    "gold_spot":         "gold_spot.csv",
    "usd_proxy":         "usd_proxy.csv",
    "tlt":               "tlt.csv",
    "airlines_basket":   "airlines_basket.csv",
    "dry_bulk_etf":      "dry_bulk_etf.csv",
    "container_shipping_basket": "container_shipping_basket.csv",
    "growth_equities_basket":    "growth_equities_basket.csv",
    "global_retail_basket":      "global_retail_basket.csv",
    "xlb":               "xlb.csv",
    "xly":               "xly.csv",
}


def load_prices(asset: str) -> pd.Series | None:
    """
    Load daily closing prices for a single asset from a CSV file.

    Args:
        asset: Asset identifier as stored in the trade_ideas table
               (e.g. 'xle', 'brent_crude_front').

    Returns:
        A pandas Series indexed by date (datetime64) with float prices,
        sorted ascending. Returns None if the file does not exist or
        cannot be parsed.
    """
    key = asset.lower().strip()
    filename = ASSET_FILE_MAP.get(key)

    if filename is None:
        logger.warning("No CSV mapping for asset '%s'. Skipping.", asset)
        return None

    path = _PRICES_DIR / filename
    if not path.exists():
        logger.warning("Price file not found: %s. Skipping.", path)
        return None

    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        return None

    if "date" not in df.columns or "price" not in df.columns:
        logger.error("CSV %s must have 'date' and 'price' columns.", path)
        return None

    series = (
        df.set_index("date")["price"]
        .sort_index()
        .astype(float)
    )
    logger.debug("Loaded %d price rows for '%s'.", len(series), asset)
    return series


def load_all_prices(assets: list[str]) -> dict[str, pd.Series]:
    """
    Load prices for a list of assets. Assets with missing files are omitted.

    Args:
        assets: List of asset identifiers.

    Returns:
        Dict mapping asset name → price Series. Only includes assets for
        which a price file was successfully loaded.
    """
    result: dict[str, pd.Series] = {}
    for asset in assets:
        series = load_prices(asset)
        if series is not None:
            result[asset] = series
    return result
