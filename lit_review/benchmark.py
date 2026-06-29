from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

from .models import BenchmarkResult, BenchmarkSummary, FieldScore, PipelineResult, SDMRequirements

if TYPE_CHECKING:
    from .agent import SDMExtractionAgent

FLOAT_TOLERANCE = 0.01

_MISSING = "<missing>"


def _str_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _compare_strings(expected: str, actual: str) -> bool:
    if expected.lower() == actual.lower():
        return True
    return _str_similarity(expected, actual) > 0.85


def _compare_lists(expected: list[str], actual: list[str]) -> tuple[int, int, int]:
    expected_lower = {v.lower() for v in expected}
    actual_lower = {v.lower() for v in actual}
    intersection = expected_lower & actual_lower
    return len(intersection), len(expected), len(actual)


def _compare_numbers(expected: int | float, actual: int | float) -> bool:
    if isinstance(expected, int) and isinstance(actual, int):
        return expected == actual
    return abs(float(expected) - float(actual)) <= FLOAT_TOLERANCE


def _scalar_score(field_path: str, expected: str, actual: str | None, match: bool) -> FieldScore:
    """One gold item; n_actual is 0 when nothing was extracted so it only affects recall."""
    extracted = actual if actual not in (None, "", _MISSING) else None
    return FieldScore(
        field_path=field_path,
        match=match,
        expected=expected,
        actual=actual if actual is not None else "",
        n_expected=1,
        n_actual=1 if extracted is not None else 0,
        n_correct=1 if match else 0,
    )


def _list_score(field_path: str, expected: list[str], actual: list[str]) -> FieldScore:
    matched, n_exp, n_act = _compare_lists(expected, actual)
    return FieldScore(
        field_path=field_path,
        match=matched == n_exp == n_act,
        expected=str(expected),
        actual=str(actual),
        n_expected=n_exp,
        n_actual=n_act,
        n_correct=matched,
    )


def _compare_fields(gold: SDMRequirements, extracted: SDMRequirements) -> list[FieldScore]:
    scores: list[FieldScore] = []

    if gold.study.title:
        scores.append(
            _scalar_score(
                "study.title",
                gold.study.title,
                extracted.study.title,
                _compare_strings(gold.study.title, extracted.study.title),
            )
        )

    if gold.study.species:
        scores.append(_list_score("study.species", gold.study.species, extracted.study.species))

    if gold.study.geographic_extent:
        scores.append(
            _scalar_score(
                "study.geographic_extent",
                gold.study.geographic_extent,
                extracted.study.geographic_extent,
                _compare_strings(
                    gold.study.geographic_extent, extracted.study.geographic_extent or ""
                ),
            )
        )

    if gold.occurrence.occurrence_type:
        scores.append(
            _scalar_score(
                "occurrence.occurrence_type",
                gold.occurrence.occurrence_type,
                extracted.occurrence.occurrence_type,
                gold.occurrence.occurrence_type == extracted.occurrence.occurrence_type,
            )
        )

    if gold.occurrence.total_presences is not None:
        scores.append(
            _scalar_score(
                "occurrence.total_presences",
                str(gold.occurrence.total_presences),
                (
                    None
                    if extracted.occurrence.total_presences is None
                    else str(extracted.occurrence.total_presences)
                ),
                _compare_numbers(
                    gold.occurrence.total_presences, extracted.occurrence.total_presences or 0
                ),
            )
        )

    if gold.occurrence.total_absences is not None:
        scores.append(
            _scalar_score(
                "occurrence.total_absences",
                str(gold.occurrence.total_absences),
                (
                    None
                    if extracted.occurrence.total_absences is None
                    else str(extracted.occurrence.total_absences)
                ),
                _compare_numbers(
                    gold.occurrence.total_absences, extracted.occurrence.total_absences or 0
                ),
            )
        )

    if gold.predictors.variables:
        scores.append(
            _list_score(
                "predictors.variables", gold.predictors.variables, extracted.predictors.variables
            )
        )

    if gold.predictors.spatial_resolution:
        scores.append(
            _scalar_score(
                "predictors.spatial_resolution",
                gold.predictors.spatial_resolution,
                extracted.predictors.spatial_resolution,
                _compare_strings(
                    gold.predictors.spatial_resolution,
                    extracted.predictors.spatial_resolution or "",
                ),
            )
        )

    gold_models_by_algo = {m.algorithm.lower(): m for m in gold.models}
    extracted_models_by_algo = {m.algorithm.lower(): m for m in extracted.models}

    for idx, (algo_lower, gold_model) in enumerate(gold_models_by_algo.items()):
        ext_model = extracted_models_by_algo.get(algo_lower)

        if ext_model is None:
            scores.append(
                _scalar_score(
                    f"models[{idx}].algorithm", gold_model.algorithm, _MISSING, match=False
                )
            )
            continue

        scores.append(
            _scalar_score(
                f"models[{idx}].algorithm", gold_model.algorithm, ext_model.algorithm, match=True
            )
        )

        if gold_model.software:
            scores.append(
                _scalar_score(
                    f"models[{idx}].software",
                    gold_model.software,
                    ext_model.software,
                    _compare_strings(gold_model.software, ext_model.software or ""),
                )
            )

        gold_perf_by_metric = {p.metric.upper(): p for p in gold_model.performance}
        ext_perf_by_metric = {p.metric.upper(): p for p in ext_model.performance}

        for metric_upper, gold_pm in gold_perf_by_metric.items():
            ext_pm = ext_perf_by_metric.get(metric_upper)
            scores.append(
                _scalar_score(
                    f"models[{idx}].performance.{gold_pm.metric}",
                    str(gold_pm.value),
                    _MISSING if ext_pm is None else str(ext_pm.value),
                    match=ext_pm is not None and _compare_numbers(gold_pm.value, ext_pm.value),
                )
            )

    if gold.evaluation.cv_strategy:
        scores.append(
            _scalar_score(
                "evaluation.cv_strategy",
                gold.evaluation.cv_strategy,
                extracted.evaluation.cv_strategy,
                _compare_strings(
                    gold.evaluation.cv_strategy, extracted.evaluation.cv_strategy or ""
                ),
            )
        )

    if gold.evaluation.metrics_used:
        scores.append(
            _list_score(
                "evaluation.metrics_used",
                gold.evaluation.metrics_used,
                extracted.evaluation.metrics_used,
            )
        )

    if gold.results.key_predictors:
        scores.append(
            _list_score(
                "results.key_predictors",
                gold.results.key_predictors,
                extracted.results.key_predictors,
            )
        )

    return scores


def _compute_precision_recall(
    scores: list[FieldScore],
) -> tuple[float, float]:
    total_correct = sum(s.n_correct for s in scores)
    total_actual = sum(s.n_actual for s in scores)
    total_expected = sum(s.n_expected for s in scores)
    precision = total_correct / total_actual if total_actual else 0.0
    recall = total_correct / total_expected if total_expected else 0.0
    return precision, recall


class Benchmark:
    def __init__(self, path: Path = Path("benchmarks")) -> None:
        self.path = path
        self.annotations_dir = path / "annotations"
        self.papers_dir = path / "papers"
        self.manifest_path = path / "manifest.json"

    def _ensure_dirs(self) -> None:
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.papers_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> list[dict]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return []

    def _save_manifest(self, manifest: list[dict]) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2))

    def add_annotation(self, pdf_path: str, requirements: SDMRequirements) -> str:
        self._ensure_dirs()
        paper_id = str(uuid.uuid4())[:8]

        annotation_path = self.annotations_dir / f"{paper_id}.json"
        annotation_path.write_text(json.dumps(requirements.model_dump(exclude_none=True), indent=2))

        dest_pdf = self.papers_dir / f"{paper_id}.pdf"
        shutil.copy2(pdf_path, dest_pdf)

        manifest = self._load_manifest()
        manifest.append(
            {
                "id": paper_id,
                "pdf_path": str(dest_pdf),
                "annotation_path": str(annotation_path),
                "title": requirements.study.title,
                "species": requirements.study.species,
            }
        )
        self._save_manifest(manifest)
        return paper_id

    def list_annotations(self) -> list[dict]:
        return self._load_manifest()

    async def run_single(self, agent: SDMExtractionAgent, paper_id: str) -> BenchmarkResult:
        manifest = self._load_manifest()
        entry = next((e for e in manifest if e["id"] == paper_id), None)
        if entry is None:
            raise ValueError(f"Paper {paper_id} not found in manifest")

        gold = SDMRequirements.model_validate_json(Path(entry["annotation_path"]).read_text())

        # Benchmark the shipped pipeline (section-aware extraction + validation/retry), not the
        # raw single-shot path. Skip evaluation since it costs an extra call without changing
        # the requirements that get scored.
        result: PipelineResult = await agent.run_pipeline(entry["pdf_path"], run_evaluation=False)
        extracted = result.requirements

        scores = _compare_fields(gold, extracted)
        precision, recall = _compute_precision_recall(scores)

        return BenchmarkResult(
            paper_id=paper_id,
            scores=scores,
            precision=precision,
            recall=recall,
        )

    async def run(self, agent: SDMExtractionAgent) -> BenchmarkSummary:
        manifest = self._load_manifest()
        if not manifest:
            return BenchmarkSummary()

        # Score papers concurrently; a failure on one paper must not abort the whole run.
        outcomes = await asyncio.gather(
            *(self.run_single(agent, entry["id"]) for entry in manifest),
            return_exceptions=True,
        )
        results = [r for r in outcomes if isinstance(r, BenchmarkResult)]
        if not results:
            return BenchmarkSummary()

        all_scores = [s for r in results for s in r.scores]
        overall_precision, overall_recall = _compute_precision_recall(all_scores)

        field_hits: dict[str, list[bool]] = {}
        for r in results:
            for s in r.scores:
                base_path = s.field_path.split("[")[0] if "[" in s.field_path else s.field_path
                field_hits.setdefault(base_path, []).append(s.match)

        per_field_accuracy = {path: sum(hits) / len(hits) for path, hits in field_hits.items()}

        return BenchmarkSummary(
            results=results,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            per_field_accuracy=per_field_accuracy,
        )
