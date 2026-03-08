"""
News2Alpha – Event Extraction Service
Entry point for the FastAPI application.

Run with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI

from app.api.routes_analyze import router as analyze_router
from app.api.routes_extract import router as extract_router
from app.db.database import init_db

# ---------------------------------------------------------------------------
# Logging – write structured logs to stdout so they are easy to capture in
# any container or log-aggregation pipeline later.
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="News2Alpha – Event Extraction & Analysis API",
    description=(
        "Accepts raw news articles and returns validated structured events, "
        "matched investment rules, and candidate trade ideas. "
        "Designed for incremental LLM integration."
    ),
    version="0.2.0",
)

app.include_router(extract_router)
app.include_router(analyze_router)

# Create tables on startup — safe to call on every restart
init_db()

logger.info("News2Alpha extraction service started.")
