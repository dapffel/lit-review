from __future__ import annotations

import json
import shutil
import uuid
from difflib import SequenceMatcher
from pathlib import Path

from .models import BenchmarkResult, BenchmarkSummary, FieldScore, SDMRequirements

FLOAT_TOLERANCE = 0.01


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


def _compare_fields(gold: SDMRequirements, extracted: SDMRequirements) -> list[FieldScore]:
    scores: list[FieldScore] = []

    if gold.study.title:
        scores.append(
            FieldScore(
                field_path="study.title",
                match=_compare_strings(gold.study.title, extracted.study.title),
                expected=gold.study.title,
                actual=extracted.study.title,
            )
        )

    if gold.study.species:
        matched, total_exp, total_act = _compare_lists(gold.study.species, extracted.study.species)
        scores.append(
            FieldScore(
                field_path="study.species",
                match=matched == total_exp == total_act,
                expected=str(gold.study.species),
                actual=str(extracted.study.species),
            )
        )

    if gold.study.geographic_extent:
        scores.append(
            FieldScore(
                field_path="study.geographic_extent",
                match=_compare_strings(
                    gold.study.geographic_extent,
                    extracted.study.geographic_extent or "",
                ),
                expected=gold.study.geographic_extent,
                actual=extracted.study.geographic_extent or "",
            )
        )

    if gold.occurrence.occurrence_type:
        scores.append(
            FieldScore(
                field_path="occurrence.occurrence_type",
                match=(gold.occurrence.occurrence_type == extracted.occurrence.occurrence_type),
                expected=gold.occurrence.occurrence_type,
                actual=extracted.occurrence.occurrence_type or "",
            )
        )

    if gold.occurrence.total_presences is not None:
        scores.append(
            FieldScore(
                field_path="occurrence.total_presences",
                match=_compare_numbers(
                    gold.occurrence.total_presences,
                    extracted.occurrence.total_presences or 0,
                ),
                expected=str(gold.occurrence.total_presences),
                actual=str(extracted.occurrence.total_presences),
            )
        )

    if gold.occurrence.total_absences is not None:
        scores.append(
            FieldScore(
                field_path="occurrence.total_absences",
                match=_compare_numbers(
                    gold.occurrence.total_absences,
                    extracted.occurrence.total_absences or 0,
                ),
                expected=str(gold.occurrence.total_absences),
                actual=str(extracted.occurrence.total_absences),
            )
        )

    if gold.predictors.variables:
        matched, total_exp, total_act = _compare_lists(
            gold.predictors.variables, extracted.predictors.variables
        )
        scores.append(
            FieldScore(
                field_path="predictors.variables",
                match=matched == total_exp == total_act,
                expected=str(gold.predictors.variables),
                actual=str(extracted.predictors.variables),
            )
        )

    if gold.predictors.spatial_resolution:
        scores.append(
            FieldScore(
                field_path="predictors.spatial_resolution",
                match=_compare_strings(
                    gold.predictors.spatial_resolution,
                    extracted.predictors.spatial_resolution or "",
                ),
                expected=gold.predictors.spatial_resolution,
                actual=extracted.predictors.spatial_resolution or "",
            )
        )

    gold_models_by_algo = {m.algorithm.lower(): m for m in gold.models}
    extracted_models_by_algo = {m.algorithm.lower(): m for m in extracted.models}

    for algo_lower, gold_model in gold_models_by_algo.items():
        ext_model = extracted_models_by_algo.get(algo_lower)
        idx = list(gold_models_by_algo.keys()).index(algo_lower)

        if ext_model is None:
            scores.append(
                FieldScore(
                    field_path=f"models[{idx}].algorithm",
                    match=False,
                    expected=gold_model.algorithm,
                    actual="<missing>",
                )
            )
            continue

        scores.append(
            FieldScore(
                field_path=f"models[{idx}].algorithm",
                match=True,
                expected=gold_model.algorithm,
                actual=ext_model.algorithm,
            )
        )

        if gold_model.software:
            scores.append(
                FieldScore(
                    field_path=f"models[{idx}].software",
                    match=_compare_strings(gold_model.software, ext_model.software or ""),
                    expected=gold_model.software,
                    actual=ext_model.software or "",
                )
            )

        gold_perf_by_metric = {p.metric.upper(): p for p in gold_model.performance}
        ext_perf_by_metric = {p.metric.upper(): p for p in ext_model.performance}

        for metric_upper, gold_pm in gold_perf_by_metric.items():
            ext_pm = ext_perf_by_metric.get(metric_upper)
            if ext_pm is None:
                scores.append(
                    FieldScore(
                        field_path=f"models[{idx}].performance.{gold_pm.metric}",
                        match=False,
                        expected=str(gold_pm.value),
                        actual="<missing>",
                    )
                )
            else:
                scores.append(
                    FieldScore(
                        field_path=f"models[{idx}].performance.{gold_pm.metric}",
                        match=_compare_numbers(gold_pm.value, ext_pm.value),
                        expected=str(gold_pm.value),
                        actual=str(ext_pm.value),
                    )
                )

    if gold.evaluation.cv_strategy:
        scores.append(
            FieldScore(
                field_path="evaluation.cv_strategy",
                match=_compare_strings(
                    gold.evaluation.cv_strategy,
                    extracted.evaluation.cv_strategy or "",
                ),
                expected=gold.evaluation.cv_strategy,
                actual=extracted.evaluation.cv_strategy or "",
            )
        )

    if gold.evaluation.metrics_used:
        matched, total_exp, total_act = _compare_lists(
            gold.evaluation.metrics_used, extracted.evaluation.metrics_used
        )
        scores.append(
            FieldScore(
                field_path="evaluation.metrics_used",
                match=matched == total_exp == total_act,
                expected=str(gold.evaluation.metrics_used),
                actual=str(extracted.evaluation.metrics_used),
            )
        )

    if gold.results.key_predictors:
        matched, total_exp, total_act = _compare_lists(
            gold.results.key_predictors, extracted.results.key_predictors
        )
        scores.append(
            FieldScore(
                field_path="results.key_predictors",
                match=matched == total_exp == total_act,
                expected=str(gold.results.key_predictors),
                actual=str(extracted.results.key_predictors),
            )
        )

    return scores


def _compute_precision_recall(
    scores: list[FieldScore],
) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    matched = sum(1 for s in scores if s.match)
    precision = matched / len(scores) if scores else 0.0
    recall = matched / len(scores) if scores else 0.0
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

    async def run_single(self, agent: object, paper_id: str) -> BenchmarkResult:
        manifest = self._load_manifest()
        entry = next((e for e in manifest if e["id"] == paper_id), None)
        if entry is None:
            raise ValueError(f"Paper {paper_id} not found in manifest")

        gold = SDMRequirements.model_validate_json(Path(entry["annotation_path"]).read_text())

        extracted = await agent.extract_from_pdf(entry["pdf_path"])  # type: ignore[attr-defined]

        scores = _compare_fields(gold, extracted)
        precision, recall = _compute_precision_recall(scores)

        return BenchmarkResult(
            paper_id=paper_id,
            scores=scores,
            precision=precision,
            recall=recall,
        )

    async def run(self, agent: object) -> BenchmarkSummary:
        manifest = self._load_manifest()
        if not manifest:
            return BenchmarkSummary()

        results: list[BenchmarkResult] = []
        for entry in manifest:
            result = await self.run_single(agent, entry["id"])
            results.append(result)

        total_matched = sum(sum(1 for s in r.scores if s.match) for r in results)
        total_fields = sum(len(r.scores) for r in results)

        overall_precision = total_matched / total_fields if total_fields else 0.0
        overall_recall = total_matched / total_fields if total_fields else 0.0

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
