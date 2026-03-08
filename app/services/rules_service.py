"""
Rules engine – loads YAML rules from disk and evaluates them against
an ExtractedEvent.

Design notes
------------
- Rules are loaded once at import time and cached in module state.
  Call reload_rules() in tests or if you need hot-reloading.
- Condition evaluation is intentionally kept simple:
    * event_type  – exact string match
    * channels_any – set intersection (any overlap triggers the rule)
- Adding new condition types: implement a new branch in _evaluate_conditions()
  and document the key in macro_rules.yaml.
- Adding a real scoring model: replace or augment _evaluate_conditions()
  without touching the public surface (match_rules).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.models.schemas import ExtractedEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path to the rules file — resolved relative to this module so it works
# regardless of the working directory uvicorn is launched from.
# ---------------------------------------------------------------------------
_RULES_PATH = Path(__file__).parent.parent / "rules" / "macro_rules.yaml"


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class MatchedRule:
    """A rule that fired against an ExtractedEvent."""

    rule_id: str
    description: str
    bullish_assets: list[str]
    bearish_assets: list[str]
    mechanism: list[str]
    risk_flags: list[str]
    invalidation_conditions: list[str]


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_rules_cache: list[dict[str, Any]] | None = None


def _load_rules(path: Path = _RULES_PATH) -> list[dict[str, Any]]:
    """
    Load and validate the rules YAML file.

    Raises:
        FileNotFoundError: If the rules file does not exist.
        ValueError: If the file is not valid YAML or is missing required keys.
    """
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    if not path.exists():
        raise FileNotFoundError(f"Rules file not found at: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse rules YAML: {exc}") from exc

    if not isinstance(data, dict) or "rules" not in data:
        raise ValueError("Rules YAML must be a mapping with a top-level 'rules' key.")

    rules = data["rules"]
    if not isinstance(rules, list):
        raise ValueError("'rules' must be a list of rule objects.")

    for i, rule in enumerate(rules):
        _validate_rule_schema(rule, index=i)

    logger.info("Loaded %d rules from %s", len(rules), path)
    _rules_cache = rules
    return _rules_cache


def _validate_rule_schema(rule: dict[str, Any], index: int) -> None:
    """Raise ValueError if a rule is missing required keys."""
    required = {"id", "description", "conditions", "signals", "mechanism"}
    missing = required - rule.keys()
    if missing:
        raise ValueError(f"Rule at index {index} is missing required keys: {missing}")

    signals = rule.get("signals", {})
    if not isinstance(signals.get("bullish", []), list) or \
       not isinstance(signals.get("bearish", []), list):
        raise ValueError(f"Rule '{rule.get('id')}': signals.bullish and signals.bearish must be lists.")


def reload_rules() -> None:
    """Force reload of rules from disk (useful in tests or after file edits)."""
    global _rules_cache
    _rules_cache = None
    _load_rules()


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def _evaluate_conditions(conditions: dict[str, Any], event: ExtractedEvent) -> bool:
    """
    Return True if ALL conditions in the rule match the event.

    Supported condition keys:
        event_type   – exact match against event.event_type
        channels_any – fires if event.channels shares at least one element
                       with the listed channels
    """
    if not conditions:
        # A rule with no conditions always fires — useful for catch-all rules.
        return True

    # --- event_type: exact match ---
    if "event_type" in conditions:
        if event.event_type != conditions["event_type"]:
            return False

    # --- channels_any: set intersection ---
    if "channels_any" in conditions:
        required_channels = set(conditions["channels_any"])
        event_channels = set(event.channels)
        if not required_channels.intersection(event_channels):
            return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_rules(event: ExtractedEvent) -> list[MatchedRule]:
    """
    Evaluate all loaded rules against an ExtractedEvent.

    Args:
        event: A validated ExtractedEvent instance.

    Returns:
        List of MatchedRule objects for every rule whose conditions fired.
        Returns an empty list if no rules match.

    Raises:
        FileNotFoundError: If the rules file is missing.
        ValueError: If the rules file is malformed.
    """
    rules = _load_rules()
    matched: list[MatchedRule] = []

    for rule in rules:
        conditions = rule.get("conditions", {})
        if _evaluate_conditions(conditions, event):
            signals = rule.get("signals", {})
            matched.append(
                MatchedRule(
                    rule_id=rule["id"],
                    description=rule["description"],
                    bullish_assets=signals.get("bullish", []),
                    bearish_assets=signals.get("bearish", []),
                    mechanism=rule.get("mechanism", []),
                    risk_flags=rule.get("risk_flags", []),
                    invalidation_conditions=rule.get("invalidation_conditions", []),
                )
            )
            logger.debug("Rule '%s' matched event_type='%s'", rule["id"], event.event_type)

    logger.info(
        "%d rule(s) matched for event_type='%s' channels=%s",
        len(matched),
        event.event_type,
        event.channels,
    )
    return matched
