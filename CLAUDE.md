# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
pip install .                # install package
pip install -e ".[dev]"      # editable install with dev tools (pytest, black, isort, mypy)
```

## Development Commands

```bash
pytest                              # run all tests
pytest tests/test_foo.py::test_bar  # run a single test
black .                             # format
isort .                             # sort imports
mypy lit_review/                    # type check
```

## Environment

Set provider API keys in `.env` at the project root (loaded via python-dotenv). At minimum `OPENAI_API_KEY` for embeddings. Any LiteLLM-supported provider works for completions.

## Architecture

Single agent that extracts species distribution modeling (SDM) requirements from research PDFs into a validated Pydantic model. The extracted data is designed to drive virtual species experiments.

### Pipeline flow (`run_pipeline`)

`run_pipeline` is implemented as a **LangGraph `StateGraph`** (`prepare → extract → validate → [retry loop] → evaluate → quality`); the retry loop is a conditional edge that re-runs while critical validation errors remain and retries are left. The steps it performs:

1. Extract PDF text via PyMuPDF
2. Parse into sections (Abstract, Methods, Results, etc.) via regex heuristics
3. Optionally retrieve context from reference SDM papers via vector memory
4. Extract `SDMRequirements` using targeted paper sections (not full truncated text)
5. Score evidence confidence per section (rule-based, no LLM call)
6. Validate constraints (metric ranges, species format, occurrence consistency)
7. Retry critical field errors with targeted re-extraction from relevant sections
8. Cross-reference extraction against the paper using a separate eval model
9. Compute overall quality score (pass/marginal/fail) combining validation, eval, and confidence

### Modules

- **`agent.py`** — `SDMExtractionAgent` with two entry points: `extract_from_pdf()` for simple extraction, `run_pipeline()` for the full flow returning `PipelineResult`. `run_pipeline()` is orchestrated by a LangGraph `StateGraph` (nodes `_prepare_node`/`_extract_node`/`_validate_node`/`_retry_node`/`_evaluate_node`/`_quality_node`, threaded through the `_PipelineState` TypedDict). Also contains `score_confidence()` and `compute_quality()`.
- **`models.py`** — Pydantic v2 models: `AgentConfig` (with separate `model` and `eval_model`), `PaperSections`, nested `SDMRequirements` with typed fields, `ExtractionEval` (counts are `@computed_field`), `FieldConfidence`/`ConfidenceReport`, `QualityScore`/`PipelineResult`, `ValidationReport`, benchmark models.
- **`sections.py`** — `parse_sections()` splits PDF text by detected headings. `SECTION_MAP` maps extraction fields to relevant paper sections. `get_text_for_field()` retrieves targeted text.
- **`prompts.py`** — All prompt text for extraction, evaluation, and retry, separated from pipeline logic.
- **`validators.py`** — Constraint validation via `validate()`. Helpers `get_critical_errors()` and `violations_by_section()` support targeted retry.
- **`memory.py`** — `VectorMemory` wraps FAISS for in-memory vector search. Used to provide context from reference SDM papers during extraction. Embeds via `litellm.aembedding()`, splits text via `langchain-text-splitters`.
- **`pdf.py`** — PDF text extraction via PyMuPDF.
- **`benchmark.py`** — Gold-standard annotation management and precision/recall scoring.

Provider-agnostic: pass any LiteLLM model string (e.g. `"gpt-4"`, `"anthropic/claude-sonnet-4-6"`, `"gemini/gemini-pro"`) in `AgentConfig.model`. Use `AgentConfig.eval_model` to run verification with a different model.
