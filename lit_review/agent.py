import json
from typing import Any, Literal

import instructor
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

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
    SDMRequirements,
    SDMResults,
    ValidationReport,
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
from .validators import get_critical_errors, validate, violations_by_section

load_dotenv()

QUERY_CHARS = 500
MIN_PARAGRAPH_CHARS = 80

EVIDENCE_LOW_THRESHOLD = 20
EVIDENCE_HIGH_THRESHOLD = 80

ConfidenceLevel = Literal["high", "medium", "low"]
Grade = Literal["pass", "marginal", "fail"]

SECTION_TO_MODEL: dict[str, type[BaseModel]] = {
    "occurrence": OccurrenceData,
    "predictors": EnvironmentalPredictors,
    "evaluation": EvaluationProtocol,
    "results": SDMResults,
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


def score_confidence(requirements: SDMRequirements) -> ConfidenceReport:
    scores: list[FieldConfidence] = []
    conf: ConfidenceLevel
    reason: str

    conf, reason = _score_evidence(requirements.study.evidence)
    if not requirements.study.species:
        conf, reason = "low", "No species extracted"
    scores.append(FieldConfidence(field_path="study", confidence=conf, reason=reason))

    conf, reason = _score_evidence(requirements.occurrence.evidence)
    if requirements.occurrence.total_presences is None:
        conf, reason = "low", "No presence count extracted"
    scores.append(FieldConfidence(field_path="occurrence", confidence=conf, reason=reason))

    conf, reason = _score_evidence(requirements.predictors.evidence)
    if not requirements.predictors.variables:
        conf, reason = "low", "No predictor variables extracted"
    scores.append(FieldConfidence(field_path="predictors", confidence=conf, reason=reason))

    if not requirements.models:
        scores.append(
            FieldConfidence(field_path="models", confidence="low", reason="No models extracted")
        )
    else:
        has_perf = any(m.performance for m in requirements.models)
        if has_perf:
            scores.append(
                FieldConfidence(
                    field_path="models", confidence="high", reason="Models with performance metrics"
                )
            )
        else:
            scores.append(
                FieldConfidence(
                    field_path="models",
                    confidence="medium",
                    reason="Models extracted but no performance metrics",
                )
            )

    conf, reason = _score_evidence(requirements.evaluation.evidence)
    scores.append(FieldConfidence(field_path="evaluation", confidence=conf, reason=reason))

    conf, reason = _score_evidence(requirements.results.evidence)
    if not requirements.results.key_predictors:
        conf, reason = "low", "No key predictors identified"
    scores.append(FieldConfidence(field_path="results", confidence=conf, reason=reason))

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

    async def extract_from_pdf(
        self, pdf_path: str, references: list[str] | None = None
    ) -> SDMRequirements:
        text = extract_text(pdf_path).strip()
        if not text:
            raise ValueError("PDF contains no extractable text")

        context = ""
        if references:
            memory = VectorMemory(
                model=self.config.embedding_model,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            await memory.add(references[: self.config.max_reference_docs])
            context_chunks = await memory.query(_retrieval_query(text))
            context = "\n\n".join(context_chunks)

        user_content = _build_extraction_content(text, context, self.config.max_input_chars)

        return await self.client.create(
            model=self.config.model,
            response_model=SDMRequirements,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
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
        )

    async def _retry_section(
        self,
        requirements: SDMRequirements,
        section_name: str,
        violations: list,
        sections: PaperSections,
    ) -> SDMRequirements:
        if section_name not in SECTION_TO_MODEL:
            return requirements

        model_cls = SECTION_TO_MODEL[section_name]
        section_data = getattr(requirements, section_name)
        section_json = json.dumps(
            _drop_empty(section_data.model_dump()),
            indent=2,
        )
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
        )

        updated = requirements.model_copy(update={section_name: corrected})
        return updated

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
        text = extract_text(pdf_path).strip()
        if not text:
            raise ValueError("PDF contains no extractable text")

        sections = parse_sections(text)

        context = ""
        if references:
            memory = VectorMemory(
                model=self.config.embedding_model,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            await memory.add(references[: self.config.max_reference_docs])
            context_chunks = await memory.query(_retrieval_query(text))
            context = "\n\n".join(context_chunks)

        extraction_text = sections.get_sections("abstract", "methods", "results")
        user_content = _build_extraction_content(
            extraction_text, context, self.config.max_input_chars
        )

        requirements = await self.client.create(
            model=self.config.model,
            response_model=SDMRequirements,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
        )

        confidence = score_confidence(requirements)

        validation: ValidationReport | None = None
        retries_performed = 0

        if run_validation:
            validation = validate(requirements)

            if retry_on_errors and max_retries > 0:
                for _ in range(max_retries):
                    critical = get_critical_errors(validation)
                    if not critical:
                        break
                    by_section = violations_by_section(
                        ValidationReport(
                            violations=critical,
                            num_errors=len(critical),
                            num_warnings=0,
                        )
                    )
                    for sec_name, sec_violations in by_section.items():
                        requirements = await self._retry_section(
                            requirements, sec_name, sec_violations, sections
                        )
                    retries_performed += 1
                    validation = validate(requirements)
                    confidence = score_confidence(requirements)

        evaluation: ExtractionEval | None = None
        if run_evaluation:
            evaluation = await self.evaluate(requirements, sections=sections)

        quality = compute_quality(validation, evaluation, confidence)

        return PipelineResult(
            requirements=requirements,
            sections_used=list(sections.sections.keys()),
            confidence=confidence,
            validation=validation,
            evaluation=evaluation,
            quality=quality,
            retries_performed=retries_performed,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass
