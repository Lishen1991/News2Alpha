"""
Router for event-extraction endpoints.

Registered in main.py with prefix="" so the paths are exactly:
    POST /extract-event
    GET  /health
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import ArticleInput, ExtractedEvent
from app.services.extraction_service import extract_event

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    response_model=dict,
    status_code=status.HTTP_200_OK,
)
def health() -> dict:
    """Returns a simple liveness payload. No external dependencies checked."""
    return {"status": "ok", "service": "news2alpha-extractor"}


@router.post(
    "/extract-event",
    summary="Extract structured investment event from a news article",
    response_model=ExtractedEvent,
    status_code=status.HTTP_200_OK,
)
def extract_event_route(article: ArticleInput) -> ExtractedEvent:
    """
    Accept a news article and return a validated structured event.

    - **article**: Full article payload (title, source, body are required).
    - **returns**: ExtractedEvent with investment-relevant structured fields.
    """
    try:
        event = extract_event(article)
    except ValueError as exc:
        # Catches Pydantic ValidationError (subclass of ValueError) from the service layer
        logger.error(
            "Validation error while extracting event for article '%s': %s",
            article.title,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Event extraction produced invalid data: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during event extraction for article '%s'", article.title)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal extraction error. Check server logs for details.",
        ) from exc

    return event
