import asyncio
import json
from typing import Any, Literal, TypedDict

import instructor
import litellm
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .memory import VectorMemory
from .models import (
    AgentConfig,
    ConfidenceReport,
    EnvironmentalPredictors,
    EvaluationProtocol,
    ExtractionEval,
    FieldConfidence,
    OccurrenceData,
    PaperSections,
    PipelineResult,
    QualityScore,
    SDMModelSpec,
    SDMRequirements,
    SDMResults,
    ValidationReport,
    Violation,
)
from .pdf import extract_text
from .prompts import (
    EVAL_EXTRACTION_PREFIX,
    EVAL_PAPER_PREFIX,
    EVAL_SYSTEM,
    EXTRACTION_CONTEXT_PREFIX,
    EXTRACTION_PAPER_PREFIX,
    EXTRACTION_SYSTEM,
    RETRY_EXTRACTION_PREFIX,
    RETRY_PAPER_PREFIX,
    RETRY_SYSTEM,
    RETRY_VIOLATIONS_PREFIX,
)
from .sections import SECTION_MAP, get_text_for_field, parse_sections
from .tracing import trace_async
from .validators import get_critical_errors, validate, violations_by_section

load_dotenv()

QUERY_CHARS = 500
MIN_PARAGRAPH_CHARS = 80

EVIDENCE_LOW_THRESHOLD = 20
EVIDENCE_HIGH_THRESHOLD = 80

ConfidenceLevel = Literal["high", "medium", "low"]
Grade = Literal["pass", "marginal", "fail"]


class _PipelineState(TypedDict, total=False):
    """Mutable state threaded through the run_pipeline LangGraph."""

    # Inputs / configuration
    pdf_path: str
    references: list[str] | None
    run_validation: bool
    run_evaluation: bool
    retries_remaining: int
    # Derived during the run
    text: str
    sections: PaperSections
    context: str
    requirements: SDMRequirements
    confidence: ConfidenceReport
    validation: ValidationReport | None
    evaluation: ExtractionEval | None
    quality: QualityScore
    retries_performed: int


class _ModelsRetry(BaseModel):
    models: list[SDMModelSpec] = Field(
        description="Re-extracted list of SDM model specs with corrected values."
    )


SECTION_TO_MODEL: dict[str, type[BaseModel]] = {
    "occurrence": OccurrenceData,
    "predictors": EnvironmentalPredictors,
    "evaluation": EvaluationProtocol,
    "results": SDMResults,
    "models": _ModelsRetry,
}


def _retrieval_query(text: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]
    for paragraph in paragraphs:
        if len(paragraph) >= MIN_PARAGRAPH_CHARS:
            return paragraph[:QUERY_CHARS]
    return text[:QUERY_CHARS]


def _build_extraction_content(text: str, context: str, max_chars: int) -> str:
    if not context:
        return f"{EXTRACTION_PAPER_PREFIX}{text}"[:max_chars]

    prefix = f"{EXTRACTION_CONTEXT_PREFIX}{context}\n\n{EXTRACTION_PAPER_PREFIX}"
    available = max_chars - len(prefix)
    if available <= 0:
        context_budget = (
            max_chars - len(EXTRACTION_CONTEXT_PREFIX) - len(f"\n\n{EXTRACTION_PAPER_PREFIX}")
        )
        context = context[: max(0, context_budget)]
        return f"{EXTRACTION_CONTEXT_PREFIX}{context}\n\n{EXTRACTION_PAPER_PREFIX}"[:max_chars]

    return f"{prefix}{text[:available]}"


def _build_eval_content(requirements_json: str, paper_text: str, max_chars: int) -> str:
    prefix = f"{EVAL_EXTRACTION_PREFIX}{requirements_json}{EVAL_PAPER_PREFIX}"
    available = max_chars - len(prefix)
    if available <= 0:
        return prefix[:max_chars]
    return f"{prefix}{paper_text[:available]}"


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            c = _drop_empty(item)
            if c is None or c == [] or c == {}:
                continue
            cleaned[key] = c
        return cleaned or None
    if isinstance(value, list):
        result = [c for item in value if (c := _drop_empty(item)) not in (None, [], {})]
        return result or None
    return value


def _build_retry_content(
    section_json: str,
    violations_text: str,
    paper_section: str,
    max_chars: int,
) -> str:
    prefix = (
        f"{RETRY_EXTRACTION_PREFIX}{section_json}"
        f"{RETRY_VIOLATIONS_PREFIX}{violations_text}"
        f"{RETRY_PAPER_PREFIX}"
    )
    available = max_chars - len(prefix)
    if available <= 0:
        return prefix[:max_chars]
    return f"{prefix}{paper_section[:available]}"


def _score_evidence(evidence: str | None) -> tuple[ConfidenceLevel, str]:
    if evidence is None or len(evidence) < EVIDENCE_LOW_THRESHOLD:
        return "low", "Missing or very short evidence"
    if len(evidence) < EVIDENCE_HIGH_THRESHOLD:
        return "medium", "Evidence present but brief"
    return "high", "Substantive evidence provided"


def _evidence_confidence(
    field_path: str, evidence: str | None, *, missing: bool = False, missing_reason: str = ""
) -> FieldConfidence:
    """Score a field from its evidence, but drop straight to 'low' when a required value is absent."""
    conf: ConfidenceLevel
    reason: str
    if missing:
        conf, reason = "low", missing_reason
    else:
        conf, reason = _score_evidence(evidence)
    return FieldConfidence(field_path=field_path, confidence=conf, reason=reason)


def _models_confidence(models: list[SDMModelSpec]) -> FieldConfidence:
    if not models:
        return FieldConfidence(field_path="models", confidence="low", reason="No models extracted")
    if any(m.performance for m in models):
        return FieldConfidence(
            field_path="models", confidence="high", reason="Models with performance metrics"
        )
    return FieldConfidence(
        field_path="models",
        confidence="medium",
        reason="Models extracted but no performance metrics",
    )


def score_confidence(requirements: SDMRequirements) -> ConfidenceReport:
    r = requirements
    scores = [
        _evidence_confidence(
            "study",
            r.study.evidence,
            missing=not r.study.species,
            missing_reason="No species extracted",
        ),
        _evidence_confidence(
            "occurrence",
            r.occurrence.evidence,
            missing=r.occurrence.total_presences is None,
            missing_reason="No presence count extracted",
        ),
        _evidence_confidence(
            "predictors",
            r.predictors.evidence,
            missing=not r.predictors.variables,
            missing_reason="No predictor variables extracted",
        ),
        _models_confidence(r.models),
        _evidence_confidence("evaluation", r.evaluation.evidence),
        _evidence_confidence(
            "results",
            r.results.evidence,
            missing=not r.results.key_predictors,
            missing_reason="No key predictors identified",
        ),
    ]

    return ConfidenceReport(field_scores=scores)


def compute_quality(
    validation: ValidationReport | None,
    evaluation: ExtractionEval | None,
    confidence: ConfidenceReport | None,
) -> QualityScore:
    reasons: list[str] = []

    if validation is not None:
        val_score = max(0.0, 1.0 - (validation.num_errors * 0.15 + validation.num_warnings * 0.05))
        if validation.num_errors:
            reasons.append(f"{validation.num_errors} validation error(s)")
        if validation.num_warnings:
            reasons.append(f"{validation.num_warnings} validation warning(s)")
    else:
        val_score = 0.5

    if evaluation is not None:
        total = len(evaluation.field_verifications)
        if total > 0:
            eval_score = evaluation.num_verified / total
            if evaluation.num_inaccurate:
                reasons.append(f"{evaluation.num_inaccurate} field(s) marked inaccurate")
            if evaluation.num_unverifiable:
                reasons.append(f"{evaluation.num_unverifiable} field(s) unverifiable")
        else:
            eval_score = 0.5
    else:
        eval_score = 0.5

    if confidence is not None:
        total_fields = len(confidence.field_scores)
        if total_fields > 0:
            conf_score = confidence.num_high / total_fields
            if confidence.num_low:
                reasons.append(f"{confidence.num_low} field(s) with low confidence")
        else:
            conf_score = 0.5
    else:
        conf_score = 0.5

    score = round(0.3 * val_score + 0.4 * eval_score + 0.3 * conf_score, 3)
    score = max(0.0, min(1.0, score))

    grade: Grade
    if score >= 0.7:
        grade = "pass"
    elif score >= 0.4:
        grade = "marginal"
    else:
        grade = "fail"

    return QualityScore(score=score, grade=grade, reasons=reasons)


class SDMExtractionAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.client = instructor.from_litellm(litellm.acompletion)
        self._pipeline = self._build_pipeline_graph()

    async def _retrieve_context(self, text: str, references: list[str] | None) -> str:
        """Embed reference papers and return the chunks most relevant to ``text``."""
        if not references:
            return ""
        memory = VectorMemory(
            model=self.config.embedding_model,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        await memory.add(references[: self.config.max_reference_docs])
        context_chunks = await memory.query(_retrieval_query(text))
        return "\n\n".join(context_chunks)

    async def extract_from_pdf(
        self, pdf_path: str, references: list[str] | None = None
    ) -> SDMRequirements:
        text = extract_text(pdf_path).strip()
        if not text:
            raise ValueError("PDF contains no extractable text")

        context = await self._retrieve_context(text, references)

        user_content = _build_extraction_content(text, context, self.config.max_input_chars)

        return await self.client.create(
            model=self.config.model,
            response_model=SDMRequirements,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
            max_retries=self.config.request_retries,
        )

    async def evaluate(
        self,
        requirements: SDMRequirements,
        pdf_path: str | None = None,
        sections: PaperSections | None = None,
    ) -> ExtractionEval:
        if sections is not None:
            paper_text = sections.get_sections("abstract", "methods", "results")
        elif pdf_path is not None:
            paper_text = extract_text(pdf_path).strip()
            if not paper_text:
                raise ValueError("PDF contains no extractable text")
        else:
            raise ValueError("Either pdf_path or sections must be provided")

        cleaned = _drop_empty(requirements.model_dump())
        requirements_json = json.dumps(cleaned, indent=2)
        user_content = _build_eval_content(
            requirements_json, paper_text, self.config.max_input_chars
        )

        return await self.client.create(
            model=self.config.eval_model or self.config.model,
            response_model=ExtractionEval,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
            max_retries=self.config.request_retries,
        )

    async def _retry_section(
        self,
        requirements: SDMRequirements,
        section_name: str,
        violations: list[Violation],
        sections: PaperSections,
    ) -> tuple[str, Any] | None:
        """Re-extract one errored section. Returns the (attribute, value) update to apply, or
        None if the section is not retryable, so callers can merge several retries at once."""
        if section_name not in SECTION_TO_MODEL:
            return None

        model_cls = SECTION_TO_MODEL[section_name]
        section_data = getattr(requirements, section_name)
        if isinstance(section_data, list):
            raw = [_drop_empty(item.model_dump()) for item in section_data]
        else:
            raw = _drop_empty(section_data.model_dump())
        section_json = json.dumps(raw, indent=2)
        violations_text = "\n".join(
            f"- {v.field_path}: {v.rule} (got: {v.actual_value})" for v in violations
        )
        paper_text = get_text_for_field(sections, section_name)

        user_content = _build_retry_content(
            section_json, violations_text, paper_text, self.config.max_input_chars
        )

        corrected = await self.client.create(
            model=self.config.model,
            response_model=model_cls,
            messages=[
                {"role": "system", "content": RETRY_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
            max_retries=self.config.request_retries,
        )

        if isinstance(corrected, _ModelsRetry):
            return "models", corrected.models
        return section_name, corrected

    # ------------------------------------------------------------------
    # Pipeline graph (LangGraph StateGraph)
    # ------------------------------------------------------------------

    def _build_pipeline_graph(self) -> Any:
        """Compile the run_pipeline flow as a stateful graph with a retry loop.

        prepare → extract → validate → (critical errors & retries left? → retry ↺)
                                     → evaluate → quality → END
        """
        graph = StateGraph(_PipelineState)
        graph.add_node("prepare", self._prepare_node)
        graph.add_node("extract", self._extract_node)
        graph.add_node("validate", self._validate_node)
        graph.add_node("retry", self._retry_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("quality", self._quality_node)

        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "extract")
        graph.add_edge("extract", "validate")
        graph.add_conditional_edges(
            "validate", self._should_retry, {"retry": "retry", "evaluate": "evaluate"}
        )
        graph.add_conditional_edges(
            "retry", self._should_retry, {"retry": "retry", "evaluate": "evaluate"}
        )
        graph.add_edge("evaluate", "quality")
        graph.add_edge("quality", END)
        return graph.compile()

    async def _prepare_node(self, state: _PipelineState) -> dict[str, Any]:
        text = extract_text(state["pdf_path"]).strip()
        if not text:
            raise ValueError("PDF contains no extractable text")
        sections = parse_sections(text)
        context = await self._retrieve_context(text, state.get("references"))
        return {"text": text, "sections": sections, "context": context}

    async def _extract_node(self, state: _PipelineState) -> dict[str, Any]:
        sections = state["sections"]
        has_key_sections = "methods" in sections.sections or "results" in sections.sections
        if has_key_sections:
            extraction_text = sections.get_sections("abstract", "methods", "results")
        else:
            extraction_text = sections.raw_text
        user_content = _build_extraction_content(
            extraction_text, state["context"], self.config.max_input_chars
        )

        requirements = await self.client.create(
            model=self.config.model,
            response_model=SDMRequirements,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
            max_retries=self.config.request_retries,
        )
        return {"requirements": requirements, "confidence": score_confidence(requirements)}

    def _validate_node(self, state: _PipelineState) -> dict[str, Any]:
        if not state.get("run_validation", True):
            return {"validation": None}
        return {"validation": validate(state["requirements"])}

    def _should_retry(self, state: _PipelineState) -> str:
        validation = state.get("validation")
        if validation is not None and state.get("retries_remaining", 0) > 0:
            if get_critical_errors(validation):
                return "retry"
        return "evaluate"

    async def _retry_node(self, state: _PipelineState) -> dict[str, Any]:
        requirements = state["requirements"]
        sections = state["sections"]
        validation = state["validation"]
        assert validation is not None  # guaranteed by _should_retry

        critical = get_critical_errors(validation)
        by_section = violations_by_section(
            ValidationReport(violations=critical, num_errors=len(critical), num_warnings=0)
        )
        # Errored sections are disjoint, so re-extract them concurrently and merge in one pass.
        updates = await asyncio.gather(
            *(
                self._retry_section(requirements, sec_name, sec_violations, sections)
                for sec_name, sec_violations in by_section.items()
            )
        )
        merged = {attr: value for u in updates if u is not None for attr, value in [u]}
        if merged:
            requirements = requirements.model_copy(update=merged)

        return {
            "requirements": requirements,
            "validation": validate(requirements),
            "confidence": score_confidence(requirements),
            "retries_performed": state.get("retries_performed", 0) + 1,
            "retries_remaining": state.get("retries_remaining", 0) - 1,
        }

    async def _evaluate_node(self, state: _PipelineState) -> dict[str, Any]:
        if not state.get("run_evaluation", True):
            return {"evaluation": None}
        evaluation = await self.evaluate(state["requirements"], sections=state["sections"])
        return {"evaluation": evaluation}

    def _quality_node(self, state: _PipelineState) -> dict[str, Any]:
        quality = compute_quality(
            state.get("validation"), state.get("evaluation"), state.get("confidence")
        )
        return {"quality": quality}

    @trace_async("SDMExtractionAgent.run_pipeline")
    async def run_pipeline(
        self,
        pdf_path: str,
        references: list[str] | None = None,
        *,
        run_validation: bool = True,
        run_evaluation: bool = True,
        retry_on_errors: bool = True,
        max_retries: int = 1,
    ) -> PipelineResult:
        initial: _PipelineState = {
            "pdf_path": pdf_path,
            "references": references,
            "run_validation": run_validation,
            "run_evaluation": run_evaluation,
            "retries_remaining": max_retries if (run_validation and retry_on_errors) else 0,
            "retries_performed": 0,
        }
        final = await self._pipeline.ainvoke(initial)
        sections: PaperSections = final["sections"]
        return PipelineResult(
            requirements=final["requirements"],
            sections_used=list(sections.sections.keys()),
            confidence=final.get("confidence"),
            validation=final.get("validation"),
            evaluation=final.get("evaluation"),
            quality=final.get("quality"),
            retries_performed=final.get("retries_performed", 0),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass
