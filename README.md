# lit-review

Structured SDM requirements extraction from research PDFs. Provider-agnostic via [LiteLLM](https://github.com/BerriAI/litellm).

Extracts species distribution modeling methodology from papers into a structured format suitable for driving virtual species experiments.

## Install

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
pytest
black .
isort .
mypy lit_review/
```

## Setup

Add your API key to `.env` at the project root. It is loaded automatically via `python-dotenv`.

```
OPENAI_API_KEY=sk-...
```

Any LiteLLM-supported provider works — just set the relevant key (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc).

## Usage

### Full pipeline (recommended)

`run_pipeline()` handles extraction, validation, retry, evaluation, and quality scoring in one call:

```python
import asyncio
from lit_review import SDMExtractionAgent, AgentConfig

async def main():
    config = AgentConfig(
        model="gpt-4",
        eval_model="anthropic/claude-sonnet-4-6",  # use a different model for verification
    )
    agent = SDMExtractionAgent(config)
    result = await agent.run_pipeline("paper.pdf")

    # Extraction
    print(result.requirements.study.title)
    print(result.requirements.study.species)        # ['Bufo marinus']
    print(result.requirements.predictors.variables)  # ['BIO1', 'BIO12', ...]

    # Quality gate
    print(result.quality.grade)    # 'pass', 'marginal', or 'fail'
    print(result.quality.score)    # 0.0 - 1.0
    print(result.quality.reasons)  # ['2 field(s) with low confidence', ...]

    # Confidence per section
    for fc in result.confidence.field_scores:
        print(f"{fc.field_path}: {fc.confidence} — {fc.reason}")

    # Validation
    if result.validation and not result.validation.is_valid:
        for v in result.validation.violations:
            print(f"{v.field_path}: {v.rule} ({v.severity})")

    # Evaluation (cross-reference check)
    if result.evaluation:
        print(f"Verified: {result.evaluation.num_verified}")
        print(f"Inaccurate: {result.evaluation.num_inaccurate}")

asyncio.run(main())
```

The pipeline parses the PDF into sections (Abstract, Methods, Results, etc.) and uses targeted sections for extraction and verification instead of truncating the full text. Critical validation errors (e.g., AUC out of range) trigger an automatic retry with the relevant paper section.

### Simple extraction

For quick extraction without the full pipeline:

```python
agent = SDMExtractionAgent()
requirements = await agent.extract_from_pdf("paper.pdf")
print(requirements.study.species)
```

### Configuration

Switch providers by changing the model string:

```python
config = AgentConfig(model="anthropic/claude-sonnet-4-6")
agent = SDMExtractionAgent(config)
```

Pass reference documents for context-aware extraction:

```python
result = await agent.run_pipeline(
    "paper.pdf",
    references=["Smith et al. 2023 used MaxEnt with spatial block CV..."],
)
```

Control pipeline steps:

```python
result = await agent.run_pipeline(
    "paper.pdf",
    run_validation=True,     # constraint checks (default: True)
    run_evaluation=True,     # LLM cross-reference (default: True)
    retry_on_errors=True,    # retry critical fields (default: True)
    max_retries=1,           # retry attempts (default: 1)
)
```

## Output

### PipelineResult

Returned by `run_pipeline()`, bundles everything:

- `requirements` — `SDMRequirements` (the extraction)
- `confidence` — `ConfidenceReport` with per-section evidence quality scores
- `validation` — `ValidationReport` with constraint violations
- `evaluation` — `ExtractionEval` with per-field verification against the paper
- `quality` — `QualityScore` with overall `score` (0-1), `grade` (pass/marginal/fail), and `reasons`
- `sections_used` — which paper sections were identified
- `retries_performed` — how many retry rounds ran

### SDMRequirements

Grouped into methodology sections with machine-readable typed fields:

- `study` — `title: str`, `species: list[str]`, `geographic_extent: str`
- `occurrence` — `occurrence_type: str`, `total_presences: int`, `total_absences: int`, `data_source: str`
- `predictors` — `variables: list[str]`, `data_source: str`, `spatial_resolution: str`
- `models` — `list[SDMModelSpec]`, one per algorithm/variant tested. Each has `algorithm: str`, `software: str`, `hyperparameters: str`, `performance: list[PerformanceMetric]`, `is_best: bool`
- `evaluation` — `cv_strategy: str`, `metrics_used: list[str]`, `threshold_method: str`
- `results` — `key_predictors: list[str]`, `projected_scenarios: list[ProjectedScenario]`

Performance metrics are numeric: `PerformanceMetric(metric="AUC", value=0.92, std=0.03)`. Each section has an `evidence` field for provenance. Missing details default to `None` or `[]`.
