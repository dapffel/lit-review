from lit_review import (
    EnvironmentalPredictors,
    EvaluationProtocol,
    ExtractionEval,
    FieldVerification,
    OccurrenceData,
    PerformanceMetric,
    PipelineResult,
    SDMModelSpec,
    SDMRequirements,
    SDMResults,
    StudyMetadata,
    ValidationReport,
    Violation,
    analyze_pipeline_result,
    export_error_analysis_csv,
    summarize_error_analysis,
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
            total_absences=10,
            evidence="The paper reports occurrence records from museum and field sources.",
        ),
        predictors=EnvironmentalPredictors(
            variables=["BIO1", "BIO12"],
            evidence="Environmental predictors included BIO1 and BIO12 climate variables.",
        ),
        models=[
            SDMModelSpec(
                algorithm="MaxEnt",
                performance=[PerformanceMetric(metric="AUC", value=1.2)],
            )
        ],
        evaluation=EvaluationProtocol(
            metrics_used=["AUC"],
            evidence="Models were evaluated using AUC.",
        ),
        results=SDMResults(
            key_predictors=["BIO1"],
            evidence="BIO1 was the strongest predictor in the fitted models.",
        ),
    )


def test_analyze_pipeline_result_adds_validation_and_eval_statuses():
    req = _requirements()
    result = PipelineResult(
        requirements=req,
        validation=ValidationReport(
            violations=[
                Violation(
                    field_path="occurrence.total_presences",
                    rule="Must be >= 1 when set",
                    actual_value="0",
                    severity="error",
                ),
                Violation(
                    field_path="models[0].performance[0].value",
                    rule="AUC must be between 0 and 1",
                    actual_value="1.2",
                    severity="error",
                ),
            ],
            num_errors=2,
            num_warnings=0,
        ),
        evaluation=ExtractionEval(
            field_verifications=[
                FieldVerification(
                    field_path="study.species",
                    extracted_value="['Bufo marinus']",
                    status="verified",
                    evidence="The paper names Bufo marinus.",
                ),
                FieldVerification(
                    field_path="occurrence.total_absences",
                    extracted_value="10",
                    status="unverifiable",
                    evidence="No absence count is reported.",
                ),
                FieldVerification(
                    field_path="models[0].performance[0].value",
                    extracted_value="1.2",
                    status="inaccurate",
                    evidence="The paper reports AUC values below 1.",
                ),
            ],
            overall_assessment="Mixed quality.",
        ),
    )

    rows = analyze_pipeline_result(result, paper_id="paper-1")
    assert len(rows) == 29
    assert {row.field_path.split(".", 1)[0].split("[", 1)[0] for row in rows} == {
        "study",
        "occurrence",
        "predictors",
        "models",
        "evaluation",
        "results",
    }

    presence_row = next(row for row in rows if row.field_path == "occurrence.total_presences")
    assert presence_row.paper_id == "paper-1"
    assert presence_row.validation_status == "error"
    assert presence_row.failure_type == "validation_error"
    assert presence_row.notes == "Must be >= 1 when set"

    absence_row = next(row for row in rows if row.field_path == "occurrence.total_absences")
    assert absence_row.validation_status == "ok"
    assert absence_row.eval_status == "unverifiable"
    assert absence_row.failure_type == "missing_evidence"

    species_row = next(row for row in rows if row.field_path == "study.species")
    assert species_row.eval_status == "verified"
    assert species_row.failure_type is None

    auc_row = next(row for row in rows if row.field_path == "models[0].performance[0].value")
    assert auc_row.evidence == "The paper reports AUC values below 1."
    assert auc_row.validation_status == "error"
    assert auc_row.notes == "AUC must be between 0 and 1"


def test_analyze_pipeline_result_uses_gold_matches():
    req = _requirements()
    gold = req.model_copy(
        update={
            "study": StudyMetadata(title="A useful SDM paper", species=["Rana temporaria"]),
        }
    )
    result = PipelineResult(requirements=req)

    rows = analyze_pipeline_result(result, gold=gold)

    species_row = next(row for row in rows if row.field_path == "study.species")
    assert species_row.match is False
    assert species_row.failure_type == "benchmark_mismatch"
    assert "Rana temporaria" in (species_row.gold_value or "")


def test_summarize_and_export_error_analysis(tmp_path):
    result = PipelineResult(requirements=_requirements(), validation=ValidationReport())
    rows = analyze_pipeline_result(result)
    summary = summarize_error_analysis(rows)

    assert summary.total_fields == len(rows)
    assert summary.by_validation_status["ok"] == len(rows)

    output_path = tmp_path / "errors.csv"
    export_error_analysis_csv(rows, output_path)
    text = output_path.read_text()

    assert "field_path" in text
    assert "study.species" in text
