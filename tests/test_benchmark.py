import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lit_review import (
    EnvironmentalPredictors,
    EvaluationProtocol,
    OccurrenceData,
    PerformanceMetric,
    SDMModelSpec,
    SDMRequirements,
    SDMResults,
    StudyMetadata,
)
from lit_review.benchmark import (
    Benchmark,
    _compare_fields,
    _compare_lists,
    _compare_numbers,
    _compare_strings,
    _compute_precision_recall,
    _scalar_score,
)
from lit_review.models import FieldScore, PipelineResult

GOLD = SDMRequirements(
    study=StudyMetadata(
        title="The art of modelling range-shifting species",
        species=["Bufo marinus"],
        geographic_extent="Australia",
    ),
    occurrence=OccurrenceData(
        occurrence_type="presence-absence",
        total_presences=1183,
        total_absences=451,
    ),
    predictors=EnvironmentalPredictors(
        variables=["BIO1", "BIO12", "elevation"],
        spatial_resolution="0.05 deg (~5 km)",
    ),
    models=[
        SDMModelSpec(
            algorithm="MaxEnt",
            software="MaxEnt v3.3.1",
            performance=[
                PerformanceMetric(metric="AUC", value=0.79),
                PerformanceMetric(metric="COR", value=0.82),
            ],
        ),
    ],
    evaluation=EvaluationProtocol(
        cv_strategy="10-fold cross-validation",
        metrics_used=["AUC", "COR"],
    ),
    results=SDMResults(key_predictors=["BIO1", "elevation"]),
)


# --- String comparison ---


def test_compare_strings_exact():
    assert _compare_strings("MaxEnt", "MaxEnt")


def test_compare_strings_case_insensitive():
    assert _compare_strings("MaxEnt", "maxent")


def test_compare_strings_fuzzy_match():
    assert _compare_strings("10-fold cross-validation", "10-fold cross validation")


def test_compare_strings_no_match():
    assert not _compare_strings("MaxEnt", "Random Forest")


# --- List comparison ---


def test_compare_lists_exact():
    matched, exp, act = _compare_lists(["AUC", "TSS"], ["AUC", "TSS"])
    assert matched == 2 and exp == 2 and act == 2


def test_compare_lists_case_insensitive():
    matched, exp, act = _compare_lists(["AUC"], ["auc"])
    assert matched == 1


def test_compare_lists_partial():
    matched, exp, act = _compare_lists(["AUC", "TSS", "COR"], ["AUC", "TSS"])
    assert matched == 2 and exp == 3 and act == 2


def test_compare_lists_order_independent():
    matched, _, _ = _compare_lists(["TSS", "AUC"], ["AUC", "TSS"])
    assert matched == 2


# --- Number comparison ---


def test_compare_numbers_int_exact():
    assert _compare_numbers(1183, 1183)


def test_compare_numbers_int_mismatch():
    assert not _compare_numbers(1183, 1184)


def test_compare_numbers_float_tolerance():
    assert _compare_numbers(0.79, 0.791)


def test_compare_numbers_float_outside_tolerance():
    assert not _compare_numbers(0.79, 0.81)


# --- Field comparison ---


def test_compare_fields_perfect_match():
    scores = _compare_fields(GOLD, GOLD)
    assert all(s.match for s in scores)
    assert len(scores) > 0


def test_compare_fields_wrong_species():
    extracted = GOLD.model_copy(
        update={"study": StudyMetadata(title=GOLD.study.title, species=["Rana temporaria"])}
    )
    scores = _compare_fields(GOLD, extracted)
    species_score = next(s for s in scores if s.field_path == "study.species")
    assert not species_score.match


def test_compare_fields_wrong_presences():
    extracted = GOLD.model_copy(
        update={"occurrence": OccurrenceData(total_presences=999, total_absences=451)}
    )
    scores = _compare_fields(GOLD, extracted)
    pres_score = next(s for s in scores if s.field_path == "occurrence.total_presences")
    assert not pres_score.match


def test_compare_fields_missing_model():
    extracted = GOLD.model_copy(update={"models": []})
    scores = _compare_fields(GOLD, extracted)
    algo_score = next(s for s in scores if "algorithm" in s.field_path)
    assert not algo_score.match
    assert algo_score.actual == "<missing>"


def test_compare_fields_wrong_performance_value():
    extracted = GOLD.model_copy(
        update={
            "models": [
                SDMModelSpec(
                    algorithm="MaxEnt",
                    software="MaxEnt v3.3.1",
                    performance=[
                        PerformanceMetric(metric="AUC", value=0.95),
                        PerformanceMetric(metric="COR", value=0.82),
                    ],
                )
            ]
        }
    )
    scores = _compare_fields(GOLD, extracted)
    auc_score = next(s for s in scores if "AUC" in s.field_path)
    assert not auc_score.match
    cor_score = next(s for s in scores if "COR" in s.field_path)
    assert cor_score.match


# --- Precision / Recall ---


def test_precision_recall_perfect():
    scores = [FieldScore(field_path="a", match=True, expected="x", actual="x")]
    p, r = _compute_precision_recall(scores)
    assert p == 1.0 and r == 1.0


def test_precision_recall_half():
    scores = [
        FieldScore(field_path="a", match=True, expected="x", actual="x"),
        FieldScore(field_path="b", match=False, expected="y", actual="z"),
    ]
    p, r = _compute_precision_recall(scores)
    assert p == 0.5 and r == 0.5


def test_precision_recall_empty():
    p, r = _compute_precision_recall([])
    assert p == 0.0 and r == 0.0


def test_precision_recall_differ_on_partial_list():
    # Gold has 4 variables, extraction returned 2 (both correct): high precision, low recall.
    gold = GOLD.model_copy(
        update={
            "predictors": EnvironmentalPredictors(variables=["BIO1", "BIO12", "elevation", "NDVI"])
        }
    )
    extracted = GOLD.model_copy(
        update={"predictors": EnvironmentalPredictors(variables=["BIO1", "BIO12"])}
    )
    scores = _compare_fields(gold, extracted)
    var_score = next(s for s in scores if s.field_path == "predictors.variables")
    assert (var_score.n_correct, var_score.n_expected, var_score.n_actual) == (2, 4, 2)

    p, r = _compute_precision_recall([var_score])
    assert p == 1.0  # 2 correct / 2 extracted
    assert r == 0.5  # 2 correct / 4 gold
    assert p != r


def test_missing_extraction_only_hurts_recall():
    # Nothing extracted for a scalar gold field: counts against recall, not precision.
    score = _scalar_score("study.title", "A title", None, match=False)
    assert score.n_actual == 0 and score.n_expected == 1 and score.n_correct == 0
    p, r = _compute_precision_recall([score])
    assert p == 0.0 and r == 0.0


# --- Benchmark class ---


def test_add_annotation(tmp_path):
    bench = Benchmark(path=tmp_path / "bench")
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("fake pdf")

    paper_id = bench.add_annotation(str(pdf), GOLD)
    assert paper_id

    manifest = bench.list_annotations()
    assert len(manifest) == 1
    assert manifest[0]["id"] == paper_id
    assert manifest[0]["title"] == GOLD.study.title

    annotation = json.loads(Path(manifest[0]["annotation_path"]).read_text())
    assert annotation["study"]["title"] == GOLD.study.title

    assert Path(manifest[0]["pdf_path"]).exists()


async def test_run_single(tmp_path):
    bench = Benchmark(path=tmp_path / "bench")
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("fake pdf")
    paper_id = bench.add_annotation(str(pdf), GOLD)

    mock_agent = AsyncMock()
    mock_agent.run_pipeline.return_value = PipelineResult(requirements=GOLD)

    result = await bench.run_single(mock_agent, paper_id)
    assert result.paper_id == paper_id
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert all(s.match for s in result.scores)
    mock_agent.run_pipeline.assert_awaited_once()


async def test_run_full(tmp_path):
    bench = Benchmark(path=tmp_path / "bench")
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("fake pdf")
    bench.add_annotation(str(pdf), GOLD)

    mock_agent = AsyncMock()
    mock_agent.run_pipeline.return_value = PipelineResult(requirements=GOLD)

    summary = await bench.run(mock_agent)
    assert len(summary.results) == 1
    assert summary.overall_precision == 1.0
    assert summary.overall_recall == 1.0
    assert len(summary.per_field_accuracy) > 0


async def test_run_survives_failing_paper(tmp_path):
    bench = Benchmark(path=tmp_path / "bench")
    for _ in range(2):
        pdf = tmp_path / f"{uuid.uuid4().hex}.pdf"
        pdf.write_text("fake pdf")
        bench.add_annotation(str(pdf), GOLD)

    mock_agent = AsyncMock()
    mock_agent.run_pipeline.side_effect = [
        RuntimeError("API timeout"),
        PipelineResult(requirements=GOLD),
    ]

    summary = await bench.run(mock_agent)
    # One paper failed; the run still returns the successful result instead of crashing.
    assert len(summary.results) == 1
    assert summary.overall_precision == 1.0


async def test_run_empty_manifest(tmp_path):
    bench = Benchmark(path=tmp_path / "bench")
    bench._ensure_dirs()
    bench._save_manifest([])

    mock_agent = AsyncMock()
    summary = await bench.run(mock_agent)
    assert summary.results == []
    assert summary.overall_precision == 0.0


async def test_run_single_not_found(tmp_path):
    bench = Benchmark(path=tmp_path / "bench")
    bench._ensure_dirs()
    bench._save_manifest([])

    mock_agent = AsyncMock()
    with pytest.raises(ValueError, match="not found"):
        await bench.run_single(mock_agent, "nonexistent")
