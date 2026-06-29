from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .benchmark import _compare_fields
from .models import (
    ErrorAnalysisRow,
    ErrorAnalysisSummary,
    FieldScore,
    FieldVerification,
    PipelineResult,
    SDMRequirements,
    ValidationReport,
    Violation,
)

FIELD_SECTIONS: dict[str, str] = {
    "study": "abstract/introduction",
    "occurrence": "methods",
    "predictors": "methods/results",
    "models": "methods/results",
    "evaluation": "methods/results",
    "results": "results/discussion",
}


def _serialize(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _top_level(path: str) -> str:
    return path.split(".", 1)[0].split("[", 1)[0]


def _is_child_path(child: str, parent: str) -> bool:
    return child.startswith(f"{parent}.") or child.startswith(f"{parent}[")


def _paths_related(left: str, right: str) -> bool:
    return left == right or _is_child_path(left, right) or _is_child_path(right, left)


def _violation_matches_field(violation_path: str, field_path: str) -> bool:
    return violation_path == field_path or _is_child_path(violation_path, field_path)


def _field_evidence(requirements: SDMRequirements, field_path: str) -> str | None:
    top = _top_level(field_path)
    if top == "models":
        return None
    section = getattr(requirements, top, None)
    return getattr(section, "evidence", None)


def _flatten(value: Any, path: str) -> list[tuple[str, Any]]:
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=False)

    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            if key == "evidence":
                continue
            rows.extend(_flatten(item, f"{path}.{key}" if path else key))
        return rows

    if isinstance(value, list):
        if not value:
            return [(path, value)]
        if all(not isinstance(item, (dict, list, BaseModel)) for item in value):
            return [(path, value)]
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten(item, f"{path}[{index}]"))
        return rows

    return [(path, value)]


def _validation_for_field(
    report: ValidationReport | None, field_path: str
) -> tuple[Literal["ok", "warning", "error", "not_run"], str | None]:
    if report is None:
        return "not_run", None

    related = [v for v in report.violations if _violation_matches_field(v.field_path, field_path)]
    if not related:
        return "ok", None

    error = next((v for v in related if v.severity == "error"), None)
    chosen = error or related[0]
    return chosen.severity, chosen.rule


def _eval_for_field(
    verifications: list[FieldVerification] | None, field_path: str
) -> tuple[Literal["verified", "inaccurate", "unverifiable", "not_run"], str | None]:
    if verifications is None:
        return "not_run", None

    exact = next((v for v in verifications if v.field_path == field_path), None)
    related = exact or next(
        (v for v in verifications if _paths_related(v.field_path, field_path)), None
    )
    if related is None:
        return "not_run", None
    return related.status, related.evidence


def _gold_scores(gold: SDMRequirements | None, extracted: SDMRequirements) -> dict[str, FieldScore]:
    if gold is None:
        return {}
    return {score.field_path: score for score in _compare_fields(gold, extracted)}


def _gold_for_field(
    scores: dict[str, FieldScore], field_path: str
) -> tuple[str | None, bool | None]:
    exact = scores.get(field_path)
    if exact is not None:
        return exact.expected, exact.match

    related = next(
        (score for path, score in scores.items() if _paths_related(path, field_path)), None
    )
    if related is None:
        return None, None
    return related.expected, related.match


def _failure_type(
    *,
    extracted_value: Any,
    evidence: str | None,
    validation_status: str,
    validation_message: str | None,
    eval_status: str,
    match: bool | None,
) -> tuple[str | None, str | None]:
    if _is_empty(extracted_value):
        return "missing_extraction", "No value was extracted for this field"
    if validation_status == "error":
        message = validation_message or "Deterministic validation found an impossible value"
        return "validation_error", message
    if validation_status == "warning":
        message = validation_message or "Deterministic validation found an unexpected value"
        return "validation_warning", message
    if eval_status == "inaccurate":
        return "model_hallucination", "Verifier says the value conflicts with the paper"
    if eval_status == "unverifiable":
        return "missing_evidence", "Verifier could not find enough support in the paper"
    if match is False:
        return "benchmark_mismatch", "Value does not match the gold annotation"
    if evidence is not None and len(evidence.strip()) < 20:
        return "weak_evidence", "Field has only very short supporting evidence"
    return None, None


def analyze_pipeline_result(
    result: PipelineResult,
    *,
    gold: SDMRequirements | None = None,
    paper_id: str | None = None,
) -> list[ErrorAnalysisRow]:
    """Build one review row per extracted field in a pipeline result."""

    gold_by_field = _gold_scores(gold, result.requirements)
    verifications = result.evaluation.field_verifications if result.evaluation is not None else None

    rows: list[ErrorAnalysisRow] = []
    for field_path, value in _flatten(result.requirements, ""):
        top = _top_level(field_path)
        validation_status, validation_message = _validation_for_field(result.validation, field_path)
        eval_status, eval_evidence = _eval_for_field(verifications, field_path)
        gold_value, match = _gold_for_field(gold_by_field, field_path)
        evidence = _field_evidence(result.requirements, field_path)
        if evidence is None and _top_level(field_path) == "models":
            evidence = eval_evidence
        failure_type, notes = _failure_type(
            extracted_value=value,
            evidence=evidence,
            validation_status=validation_status,
            validation_message=validation_message,
            eval_status=eval_status,
            match=match,
        )

        rows.append(
            ErrorAnalysisRow(
                paper_id=paper_id,
                field_path=field_path,
                extracted_value=_serialize(value),
                evidence=evidence,
                section_used=FIELD_SECTIONS.get(top),
                validation_status=validation_status,
                validation_message=validation_message,
                eval_status=eval_status,
                eval_evidence=eval_evidence,
                gold_value=gold_value,
                match=match,
                failure_type=failure_type,
                notes=notes,
            )
        )

    return rows


def summarize_error_analysis(rows: list[ErrorAnalysisRow]) -> ErrorAnalysisSummary:
    failures = [row.failure_type for row in rows if row.failure_type is not None]
    return ErrorAnalysisSummary(
        total_fields=len(rows),
        num_failures=len(failures),
        by_failure_type=dict(Counter(failures)),
        by_eval_status=dict(Counter(row.eval_status for row in rows)),
        by_validation_status=dict(Counter(row.validation_status for row in rows)),
    )


def export_error_analysis_csv(
    rows: list[ErrorAnalysisRow], path: str | Path, *, create_dirs: bool = False
) -> None:
    output_path = Path(path)
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(ErrorAnalysisRow.model_fields.keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())
