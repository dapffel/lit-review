from lit_review import (
    EnvironmentalPredictors,
    EvaluationProtocol,
    OccurrenceData,
    PerformanceMetric,
    PipelineResult,
    QualityScore,
    RunStore,
    SDMModelSpec,
    SDMRequirements,
    SDMResults,
    StudyMetadata,
    ValidationReport,
    Violation,
    learn_from_runs,
)


def _requirements() -> SDMRequirements:
    return SDMRequirements(
        study=StudyMetadata(
            title="A useful SDM paper",
            species=["Bufo marinus"],
            geographic_extent="Australia",
            evidence="The study models the distribution of the cane toad in Australia.",
        ),
        occurrence=OccurrenceData(
            occurrence_type="presence-only",
            total_presences=0,
            evidence="The paper reports occurrence records from museum and field sources.",
        ),
        predictors=EnvironmentalPredictors(
            variables=["BIO1", "BIO12"],
            evidence="Environmental predictors included BIO1 and BIO12 climate variables.",
        ),
        models=[
            SDMModelSpec(
                algorithm="MaxEnt", performance=[PerformanceMetric(metric="AUC", value=0.9)]
            )
        ],
        evaluation=EvaluationProtocol(metrics_used=["AUC"], evidence="Evaluated using AUC."),
        results=SDMResults(key_predictors=["BIO1"], evidence="BIO1 was the strongest predictor."),
    )


def _result_with_presence_error() -> PipelineResult:
    return PipelineResult(
        requirements=_requirements(),
        validation=ValidationReport(
            violations=[
                Violation(
                    field_path="occurrence.total_presences",
                    rule="Must be >= 1 when set",
                    actual_value="0",
                    severity="error",
                )
            ],
            num_errors=1,
            num_warnings=0,
        ),
        quality=QualityScore(score=0.55, grade="marginal", reasons=["1 validation error(s)"]),
    )


def test_record_appends_jsonl_and_reloads(tmp_path):
    store = RunStore(tmp_path / "runs.jsonl")
    result = _result_with_presence_error()

    record = store.record(result, paper_id="paper-1", model="gpt-4o")

    assert record.paper_id == "paper-1"
    assert record.model == "gpt-4o"
    assert record.quality_grade == "marginal"
    assert record.has_gold is False
    assert store.path.exists()

    # A second run appends rather than overwriting.
    store.record(result, paper_id="paper-2")
    reloaded = store.load()
    assert len(reloaded) == 2
    assert [r.paper_id for r in reloaded] == ["paper-1", "paper-2"]
    assert reloaded[0].rows  # per-field rows survive the round-trip


def test_load_missing_file_returns_empty(tmp_path):
    assert RunStore(tmp_path / "nope.jsonl").load() == []


def test_learn_from_runs_ranks_weak_fields(tmp_path):
    store = RunStore(tmp_path / "runs.jsonl")
    # The same validation error recurs across two papers.
    store.record(_result_with_presence_error(), paper_id="paper-1")
    store.record(_result_with_presence_error(), paper_id="paper-2")

    report = learn_from_runs(store.load())

    assert report.num_runs == 2
    assert report.num_papers == 2
    assert report.summary.by_failure_type.get("validation_error") == 2

    presence = next(w for w in report.weak_fields if w.field_path == "occurrence.total_presences")
    assert presence.total == 2
    assert presence.num_failures == 2
    assert presence.failure_rate == 1.0
    assert presence.by_failure_type == {"validation_error": 2}
    # Only fields with failures are reported, worst first.
    assert all(w.num_failures > 0 for w in report.weak_fields)
    assert report.weak_fields[0].failure_rate >= report.weak_fields[-1].failure_rate


def test_learn_from_runs_normalizes_list_indices(tmp_path):
    store = RunStore(tmp_path / "runs.jsonl")
    result = PipelineResult(
        requirements=_requirements(),
        validation=ValidationReport(
            violations=[
                Violation(
                    field_path="models[0].performance[0].value",
                    rule="AUC must be between 0 and 1",
                    actual_value="1.2",
                    severity="error",
                )
            ],
            num_errors=1,
            num_warnings=0,
        ),
    )
    store.record(result, paper_id="paper-1")

    report = learn_from_runs(store.load())
    paths = {w.field_path for w in report.weak_fields}
    # Indices are collapsed so weaknesses aggregate across model/metric positions.
    assert "models.performance.value" in paths
    assert not any("[" in p for p in paths)
