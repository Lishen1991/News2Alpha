"""
Extraction service – the core business logic layer.

Currently uses a deterministic mock extractor. The public surface
(extract_event) is designed so that swapping in a real LLM call only
requires changing this one module.

Replacement pattern for real LLM integration:
    1. Replace _mock_extract() with an async call to your LLM client.
    2. Parse the LLM JSON response into ExtractedEvent using model_validate().
    3. Retain the same function signature: extract_event(article) -> ExtractedEvent.
"""

from __future__ import annotations

import logging

from app.models.schemas import ArticleInput, ExtractedEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword sets used by the mock extractor
# ---------------------------------------------------------------------------

_GEOPOLITICAL_OIL_KEYWORDS = frozenset(
    ["iran", "hormuz", "oil", "opec", "crude", "petroleum", "shipping", "tanker", "sanctions"]
)

_MACRO_KEYWORDS = frozenset(
    ["fed", "federal reserve", "interest rate", "inflation", "cpi", "gdp", "recession"]
)

_CORPORATE_KEYWORDS = frozenset(
    ["earnings", "revenue", "profit", "acquisition", "merger", "ipo", "layoff", "ceo"]
)


def _normalise(text: str) -> str:
    """Lowercase the combined article text for keyword matching."""
    return text.lower()


def _contains_any(text: str, keywords: frozenset[str]) -> bool:
    return any(kw in text for kw in keywords)


# ---------------------------------------------------------------------------
# Scenario templates
# Structured as plain dicts so they are trivially serialisable and easy to
# extend without touching the service logic.
# ---------------------------------------------------------------------------

def _geopolitical_oil_event(article: ArticleInput) -> dict:
    return {
        "event_type": "geopolitical_conflict",
        "entities": ["Iran", "OPEC", "Strait of Hormuz"],
        "region": ["Middle East", "Global"],
        "summary": (
            "Geopolitical tensions in the Middle East are disrupting oil shipping routes, "
            "raising the risk of a near-term crude supply shock."
        ),
        "severity": 0.82,
        "novelty": 0.65,
        "channels": ["energy_prices", "supply_chain", "risk_sentiment"],
        "time_horizon": "short",
        "affected_sectors": ["Energy", "Transportation", "Industrials"],
        "evidence_statements": [
            f"Article headline: '{article.title}'",
            "Keywords detected: Iran / Hormuz / oil / shipping",
        ],
        "uncertainty_notes": [
            "Geopolitical situations can de-escalate quickly.",
            "OPEC spare capacity could offset supply disruption.",
        ],
    }


def _macro_event(article: ArticleInput) -> dict:
    return {
        "event_type": "monetary_policy_shift",
        "entities": ["Federal Reserve"],
        "region": ["United States", "Global"],
        "summary": (
            "A shift in central-bank policy signals changing credit conditions "
            "that may reprice risk assets across multiple sectors."
        ),
        "severity": 0.60,
        "novelty": 0.50,
        "channels": ["interest_rates", "credit_spreads", "currency"],
        "time_horizon": "medium",
        "affected_sectors": ["Financials", "Real Estate", "Utilities"],
        "evidence_statements": [
            f"Article headline: '{article.title}'",
            "Keywords detected: Fed / inflation / interest rate",
        ],
        "uncertainty_notes": [
            "Forward guidance can be revised at subsequent meetings.",
        ],
    }


def _corporate_event(article: ArticleInput) -> dict:
    return {
        "event_type": "corporate_action",
        "entities": [article.source],
        "region": [],
        "summary": (
            "A significant corporate development has been reported that may affect "
            "the company's valuation or competitive positioning."
        ),
        "severity": 0.40,
        "novelty": 0.45,
        "channels": ["equity_prices", "sector_sentiment"],
        "time_horizon": "short",
        "affected_sectors": [],
        "evidence_statements": [
            f"Article headline: '{article.title}'",
        ],
        "uncertainty_notes": [
            "Impact depends on management execution and market reaction.",
        ],
    }


def _generic_fallback_event(article: ArticleInput) -> dict:
    return {
        "event_type": "general_news",
        "entities": [],
        "region": [],
        "summary": (
            "A news article was received but no specific investment-relevant "
            "event pattern was matched by the extractor."
        ),
        "severity": 0.10,
        "novelty": 0.10,
        "channels": [],
        "time_horizon": "short",
        "affected_sectors": [],
        "evidence_statements": [],
        "uncertainty_notes": ["No structured pattern detected; manual review recommended."],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _mock_extract(article: ArticleInput) -> dict:
    """
    Deterministic keyword-based event extractor.

    Priority order: geopolitical/oil > macro > corporate > fallback.
    Replace this function body with an LLM call to upgrade the service.
    """
    text = _normalise(f"{article.title} {article.body}")

    if _contains_any(text, _GEOPOLITICAL_OIL_KEYWORDS):
        logger.debug("Matched geopolitical/oil scenario for article: '%s'", article.title)
        return _geopolitical_oil_event(article)

    if _contains_any(text, _MACRO_KEYWORDS):
        logger.debug("Matched macro scenario for article: '%s'", article.title)
        return _macro_event(article)

    if _contains_any(text, _CORPORATE_KEYWORDS):
        logger.debug("Matched corporate scenario for article: '%s'", article.title)
        return _corporate_event(article)

    logger.debug("No pattern matched for article: '%s'; returning fallback", article.title)
    return _generic_fallback_event(article)


def extract_event(article: ArticleInput) -> ExtractedEvent:
    """
    Extract a structured investment event from a news article.

    Args:
        article: Validated ArticleInput instance.

    Returns:
        A fully validated ExtractedEvent instance.

    Raises:
        ValueError: If the extracted data fails schema validation.
    """
    raw = _mock_extract(article)
    # model_validate() will raise ValidationError (a subclass of ValueError)
    # if the mock returns malformed data — caught upstream in the route handler.
    return ExtractedEvent.model_validate(raw)
