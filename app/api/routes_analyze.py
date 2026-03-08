"""
Router for the /analyze-article endpoint.

This is the top-level pipeline route: it chains
    ArticleInput → extraction → rule matching → idea generation → response

Registered in main.py alongside the extraction router.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import AnalyzeArticleResponse, ArticleInput
from app.services.extraction_service import extract_event
from app.services.idea_service import generate_ideas
from app.services.logging_service import log_analysis
from app.services.rules_service import match_rules

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/analyze-article",
    summary="Extract event, match rules, and generate trade ideas from a news article",
    response_model=AnalyzeArticleResponse,
    status_code=status.HTTP_200_OK,
)
def analyze_article(article: ArticleInput) -> AnalyzeArticleResponse:
    """
    Full analysis pipeline for a news article.

    Steps:
    1. Extract a structured event via the extraction service.
    2. Match the event against the YAML rules engine.
    3. Generate candidate trade ideas from matched rules.
    4. Return the bundled AnalyzeArticleResponse.

    - **article**: Full article payload (title, source, body are required).
    - **returns**: Event, matched rule IDs, and a list of trade ideas.
    """
    # --- Step 1: Event extraction ---
    try:
        event = extract_event(article)
    except ValueError as exc:
        logger.error("Extraction validation error for '%s': %s", article.title, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Event extraction produced invalid data: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected extraction error for '%s'", article.title)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Event extraction failed. Check server logs.",
        ) from exc

    # --- Step 2: Rule matching ---
    try:
        matched_rules = match_rules(event)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Rules engine error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rules engine error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected rules engine error for event_type='%s'", event.event_type)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Rules matching failed. Check server logs.",
        ) from exc

    # --- Step 3: Idea generation ---
    try:
        ideas = generate_ideas(event, matched_rules)
    except Exception as exc:
        logger.exception("Idea generation failed for event_type='%s'", event.event_type)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Idea generation failed. Check server logs.",
        ) from exc

    # --- Step 4: Persist (fire-and-forget — errors are logged, never raised) ---
    log_analysis(article, event, ideas)

    return AnalyzeArticleResponse(
        event=event,
        ideas=ideas,
        matched_rule_ids=[r.rule_id for r in matched_rules],
    )
