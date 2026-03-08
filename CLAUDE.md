# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
py -m pip install -r requirements.txt

# Run the dev server (auto-reload)
uvicorn app.main:app --reload
```

## Architecture

News2Alpha is a FastAPI microservice that turns news articles into structured investment research. The pipeline has two layers:

```
app/
  main.py                         # App factory, logging config, router registration
  api/
    routes_extract.py             # POST /extract-event, GET /health
    routes_analyze.py             # POST /analyze-article (full pipeline)
  models/
    schemas.py                    # All Pydantic schemas: ArticleInput, ExtractedEvent,
                                  #   TradeIdea, AnalyzeArticleResponse
  services/
    extraction_service.py         # Event extraction (currently mocked, LLM-ready)
    rules_service.py              # YAML rules engine: loads + evaluates macro_rules.yaml
    idea_service.py               # Trade idea generator: rules → TradeIdea list
  rules/
    macro_rules.yaml              # Declarative investment rules (conditions + signals)
```

## Pipeline flow

```
POST /analyze-article
  → extraction_service.extract_event()      returns ExtractedEvent
  → rules_service.match_rules()             returns list[MatchedRule]
  → idea_service.generate_ideas()           returns list[TradeIdea]
  → AnalyzeArticleResponse
```

`POST /extract-event` runs only the first step (extraction only).

## Key design decisions

**Rules engine (`rules_service.py`):** Rules are loaded from `app/rules/macro_rules.yaml` and cached in module state. Each rule defines `conditions` (supports `event_type` exact match and `channels_any` set intersection) and `signals` (bullish/bearish asset lists). Add new condition types by implementing a new branch in `_evaluate_conditions()`.

**Idea generation (`idea_service.py`):** One `TradeIdea` per `(asset, direction)` pair. Confidence is derived from `event.severity` plus a small boost per additional matched rule, capped at 1.0. Deduplication ensures the same asset doesn't appear twice from overlapping rules.

**Schema validation:** `severity`, `novelty`, `confidence` are validated to `[0, 1]`. `time_horizon` and `direction` are enum-constrained. `ValidationError` from Pydantic is caught in route handlers and returned as `422`.

**LLM integration path:** Replace `_mock_extract()` in `extraction_service.py` only. The rules and idea layers are decoupled from the extraction implementation.

## Database

SQLite file is created at the project root as `news2alpha.db`. Tables are auto-created on every server startup via `init_db()` in `main.py`.

```
app/db/
  database.py   # engine, SessionLocal, Base, get_session() context manager, init_db()
  models.py     # ArticleRecord, EventRecord, TradeIdeaRecord (SQLAlchemy ORM)

app/services/
  logging_service.py  # log_analysis() — writes one full pipeline run to DB
```

**Session pattern:** all writes use `get_session()` as a context manager — commits on clean exit, rolls back on exception, always closes. Persistence errors are caught and logged but never re-raised, so a DB failure never breaks the API response.

**JSON fields:** list columns (`channels`) are stored as native JSON (TEXT in SQLite), handled automatically by SQLAlchemy's `JSON` type.

**Inspect the database:**
```powershell
# Quick check — count rows in each table
py -c "
import sqlite3, pathlib
con = sqlite3.connect('news2alpha.db')
for t in ['articles','events','trade_ideas']:
    print(t, con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0])
"
```

## Extending

- **New rules:** add entries to `app/rules/macro_rules.yaml` — no code changes needed
- **New condition types:** add a branch in `rules_service._evaluate_conditions()`
- **Real LLM extraction:** replace `_mock_extract()` in `extraction_service.py`
- **New endpoints:** create a router in `app/api/` and register it in `main.py`
