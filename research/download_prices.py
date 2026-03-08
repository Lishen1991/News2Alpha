"""
Price downloader — fetches real historical daily closing prices from
Yahoo Finance using yfinance and saves them as CSV files in research/prices/.

Each CSV overwrites the sample placeholder file so forward_returns.py
automatically picks up real data on the next run.

Usage:
    # Download 1 year of history (default)
    py research/download_prices.py

    # Download 2 years of history
    py research/download_prices.py --period 2y

    # Download a specific date range
    py research/download_prices.py --start 2024-01-01 --end 2025-01-01

Supported periods (yfinance syntax):
    1mo  3mo  6mo  1y  2y  5y  10y  ytd  max
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

_PRICES_DIR = Path(__file__).parent / "prices"

# ---------------------------------------------------------------------------
# Asset → Yahoo Finance ticker mapping
#
# Notes:
#   BZ=F  Brent Crude front-month futures
#   CL=F  WTI Crude front-month futures
#   GC=F  Gold front-month futures
#   UUP   Invesco DB USD Index Bullish Fund (USD proxy ETF)
#   XLE   Energy Select Sector SPDR
#   XLI   Industrials Select Sector SPDR
#   TLT   iShares 20+ Year Treasury Bond ETF
#   JETS  US Global Jets ETF (airlines basket)
#   BDRY  Breakwave Dry Bulk Shipping ETF
#   BOAT  SonicShares Global Shipping ETF (container)
#   XLY   Consumer Discretionary SPDR (growth/retail proxy)
#   XLB   Materials Select Sector SPDR
# ---------------------------------------------------------------------------
ASSET_TICKER_MAP: dict[str, str] = {
    "brent_crude_front":         "BZ=F",
    "wti_crude_front":           "CL=F",
    "xle":                       "XLE",
    "xli":                       "XLI",
    "gold_spot":                 "GC=F",
    "usd_proxy":                 "UUP",
    "tlt":                       "TLT",
    "airlines_basket":           "JETS",
    "dry_bulk_etf":              "BDRY",
    "container_shipping_basket": "BOAT",
    "growth_equities_basket":    "QQQ",
    "global_retail_basket":      "XLY",
    "xlb":                       "XLB",
    "xly":                       "XLY",
}


def download_asset(
    asset: str,
    ticker: str,
    period: str | None,
    start: str | None,
    end: str | None,
) -> bool:
    """
    Download closing prices for one asset and save to research/prices/<asset>.csv.

    Args:
        asset:  Internal asset name (used as filename).
        ticker: Yahoo Finance ticker symbol.
        period: yfinance period string (e.g. '1y'). Used when start/end are None.
        start:  Start date string 'YYYY-MM-DD'.
        end:    End date string 'YYYY-MM-DD'.

    Returns:
        True if the file was written successfully, False otherwise.
    """
    logger.info("Downloading %-30s  (%s)", asset, ticker)

    try:
        if start and end:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        else:
            raw = yf.download(ticker, period=period or "1y", auto_adjust=True, progress=False)
    except Exception as exc:
        logger.error("  Failed to download %s: %s", ticker, exc)
        return False

    if raw.empty:
        logger.warning("  No data returned for %s. Skipping.", ticker)
        return False

    # yfinance returns a MultiIndex when downloading a single ticker with
    # auto_adjust — flatten it and grab the Close column.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if "Close" not in raw.columns:
        logger.error("  'Close' column not found for %s. Got: %s", ticker, list(raw.columns))
        return False

    prices = (
        raw[["Close"]]
        .rename(columns={"Close": "price"})
        .reset_index()
        .rename(columns={"Date": "date", "Datetime": "date"})
    )
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.dropna(subset=["price"]).sort_values("date")

    out_path = _PRICES_DIR / f"{asset}.csv"
    prices.to_csv(out_path, index=False)
    logger.info("  Saved %d rows → %s", len(prices), out_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real price data from Yahoo Finance.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--period",
        type=str,
        default="1y",
        help="yfinance period (default: 1y). Options: 1mo 3mo 6mo 1y 2y 5y 10y ytd max",
    )
    group.add_argument(
        "--start",
        type=str,
        help="Start date YYYY-MM-DD (use with --end)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (defaults to today)",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Download a single asset only (e.g. --asset xle)",
    )
    args = parser.parse_args()

    # Validate --start/--end usage
    if args.start and not args.end:
        from datetime import date
        args.end = date.today().isoformat()

    _PRICES_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to a single asset if requested
    assets_to_download = (
        {args.asset.lower(): ASSET_TICKER_MAP[args.asset.lower()]}
        if args.asset and args.asset.lower() in ASSET_TICKER_MAP
        else ASSET_TICKER_MAP
    )

    if args.asset and args.asset.lower() not in ASSET_TICKER_MAP:
        logger.error("Unknown asset '%s'. Known assets: %s", args.asset, list(ASSET_TICKER_MAP))
        sys.exit(1)

    successes = 0
    for asset, ticker in assets_to_download.items():
        ok = download_asset(
            asset=asset,
            ticker=ticker,
            period=args.period if not args.start else None,
            start=args.start,
            end=args.end,
        )
        if ok:
            successes += 1

    print(f"\nDone. {successes}/{len(assets_to_download)} assets downloaded.")
    print("Run  py research/forward_returns.py  to analyse with real prices.")


if __name__ == "__main__":
    main()
