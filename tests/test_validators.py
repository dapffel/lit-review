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
from lit_review.validators import (
    BINOMIAL_PATTERN,
    KNOWN_ALGORITHMS,
    KNOWN_METRICS,
    NORMALIZED_METRICS,
    OCCURRENCE_TYPES,
    validate,
)

VALID_REQUIREMENTS = SDMRequirements(
    study=StudyMetadata(
        title="Test paper",
        species=["Bufo marinus"],
        geographic_extent="Australia",
    ),
    occurrence=OccurrenceData(
        occurrence_type="presence-absence",
        total_presences=100,
        total_absences=50,
    ),
    predictors=EnvironmentalPredictors(
        variables=["BIO1", "BIO12", "elevation"],
    ),
    models=[
        SDMModelSpec(
            algorithm="MaxEnt",
            performance=[
                PerformanceMetric(metric="AUC", value=0.85),
                PerformanceMetric(metric="TSS", value=0.70),
            ],
        ),
    ],
    evaluation=EvaluationProtocol(metrics_used=["AUC", "TSS"]),
    results=SDMResults(key_predictors=["BIO1", "elevation"]),
)


def test_valid_requirements_no_violations():
    report = validate(VALID_REQUIREMENTS)
    assert report.is_valid
    assert report.num_errors == 0
    assert report.num_warnings == 0
    assert report.violations == []


# --- Species binomial checks ---


def test_species_binomial_valid():
    report = validate(VALID_REQUIREMENTS)
    assert not any(v.field_path.startswith("study.species") for v in report.violations)


def test_species_binomial_warning_lowercase_genus():
    req = VALID_REQUIREMENTS.model_copy(
        update={"study": StudyMetadata(title="Test", species=["bufo marinus"])}
    )
    report = validate(req)
    assert any(
        v.field_path == "study.species[0]" and v.severity == "warning" for v in report.violations
    )


def test_species_binomial_warning_single_word():
    req = VALID_REQUIREMENTS.model_copy(
        update={"study": StudyMetadata(title="Test", species=["Amphibians"])}
    )
    report = validate(req)
    assert any(v.field_path == "study.species[0]" for v in report.violations)


def test_species_subspecies_valid():
    req = VALID_REQUIREMENTS.model_copy(
        update={"study": StudyMetadata(title="Test", species=["Bufo marinus subsp. africanus"])}
    )
    report = validate(req)
    assert not any(v.field_path.startswith("study.species") for v in report.violations)


# --- Occurrence checks ---


def test_occurrence_type_invalid():
    req = VALID_REQUIREMENTS.model_copy(
        update={"occurrence": OccurrenceData(occurrence_type="something-else")}
    )
    report = validate(req)
    assert any(
        v.field_path == "occurrence.occurrence_type" and v.severity == "error"
        for v in report.violations
    )


def test_occurrence_presences_zero():
    req = VALID_REQUIREMENTS.model_copy(update={"occurrence": OccurrenceData(total_presences=0)})
    report = validate(req)
    assert any(
        v.field_path == "occurrence.total_presences" and v.severity == "error"
        for v in report.violations
    )


def test_occurrence_absences_negative():
    req = VALID_REQUIREMENTS.model_copy(update={"occurrence": OccurrenceData(total_absences=-5)})
    report = validate(req)
    assert any(
        v.field_path == "occurrence.total_absences" and v.severity == "error"
        for v in report.violations
    )


def test_occurrence_presence_only_with_absences():
    req = VALID_REQUIREMENTS.model_copy(
        update={"occurrence": OccurrenceData(occurrence_type="presence-only", total_absences=100)}
    )
    report = validate(req)
    assert any(
        v.field_path == "occurrence.total_absences" and "presence-only" in v.rule
        for v in report.violations
    )


def test_occurrence_none_values_no_violations():
    req = VALID_REQUIREMENTS.model_copy(update={"occurrence": OccurrenceData()})
    report = validate(req)
    assert not any(v.field_path.startswith("occurrence") for v in report.violations)


# --- Performance metric checks ---


def test_auc_out_of_range():
    req = VALID_REQUIREMENTS.model_copy(
        update={
            "models": [
                SDMModelSpec(
                    algorithm="MaxEnt",
                    performance=[PerformanceMetric(metric="AUC", value=1.5)],
                )
            ]
        }
    )
    report = validate(req)
    assert any(
        v.field_path == "models[0].performance[0].value" and v.severity == "error"
        for v in report.violations
    )


def test_auc_negative():
    req = VALID_REQUIREMENTS.model_copy(
        update={
            "models": [
                SDMModelSpec(
                    algorithm="MaxEnt",
                    performance=[PerformanceMetric(metric="AUC", value=-0.1)],
                )
            ]
        }
    )
    report = validate(req)
    assert any(
        v.field_path == "models[0].performance[0].value" and v.severity == "error"
        for v in report.violations
    )


def test_rmse_negative():
    req = VALID_REQUIREMENTS.model_copy(
        update={
            "models": [
                SDMModelSpec(
                    algorithm="MaxEnt",
                    performance=[PerformanceMetric(metric="RMSE", value=-0.5)],
                )
            ]
        }
    )
    report = validate(req)
    assert any(
        v.field_path == "models[0].performance[0].value" and v.severity == "error"
        for v in report.violations
    )


def test_rmse_positive_valid():
    req = VALID_REQUIREMENTS.model_copy(
        update={
            "models": [
                SDMModelSpec(
                    algorithm="MaxEnt",
                    performance=[PerformanceMetric(metric="RMSE", value=2.5)],
                )
            ]
        }
    )
    report = validate(req)
    assert not any(
        v.field_path.startswith("models[0].performance") and v.severity == "error"
        for v in report.violations
    )


def test_auc_boundary_values():
    for val in [0.0, 0.5, 1.0]:
        req = VALID_REQUIREMENTS.model_copy(
            update={
                "models": [
                    SDMModelSpec(
                        algorithm="MaxEnt",
                        performance=[PerformanceMetric(metric="AUC", value=val)],
                    )
                ]
            }
        )
        report = validate(req)
        assert not any(
            v.field_path == "models[0].performance[0].value" and v.severity == "error"
            for v in report.violations
        ), f"AUC={val} should be valid"


# --- Algorithm checks ---


def test_unknown_algorithm_warning():
    req = VALID_REQUIREMENTS.model_copy(
        update={"models": [SDMModelSpec(algorithm="DeepSpeciesNet", performance=[])]}
    )
    report = validate(req)
    assert any(
        v.field_path == "models[0].algorithm" and v.severity == "warning" for v in report.violations
    )


def test_known_algorithm_no_warning():
    for algo in ["MaxEnt", "GLM", "BRT", "RF"]:
        req = VALID_REQUIREMENTS.model_copy(
            update={"models": [SDMModelSpec(algorithm=algo, performance=[])]}
        )
        report = validate(req)
        assert not any(
            v.field_path == "models[0].algorithm" for v in report.violations
        ), f"{algo} should not trigger a warning"


# --- Evaluation metrics checks ---


def test_unknown_evaluation_metric_warning():
    req = VALID_REQUIREMENTS.model_copy(
        update={"evaluation": EvaluationProtocol(metrics_used=["AUC", "FooMetric"])}
    )
    report = validate(req)
    assert any(
        v.field_path == "evaluation.metrics_used[1]" and v.severity == "warning"
        for v in report.violations
    )


def test_known_evaluation_metrics_no_warning():
    req = VALID_REQUIREMENTS.model_copy(
        update={"evaluation": EvaluationProtocol(metrics_used=["AUC", "TSS", "Boyce"])}
    )
    report = validate(req)
    assert not any(v.field_path.startswith("evaluation.metrics_used") for v in report.violations)


# --- Key predictors consistency ---


def test_key_predictor_not_in_variables():
    req = VALID_REQUIREMENTS.model_copy(
        update={"results": SDMResults(key_predictors=["BIO1", "NDVI"])}
    )
    report = validate(req)
    assert any(
        v.field_path == "results.key_predictors[1]" and v.severity == "error"
        for v in report.violations
    )


def test_key_predictors_case_insensitive():
    req = VALID_REQUIREMENTS.model_copy(
        update={"results": SDMResults(key_predictors=["bio1", "Elevation"])}
    )
    report = validate(req)
    assert not any(v.field_path.startswith("results.key_predictors") for v in report.violations)


def test_key_predictors_empty_skips_check():
    req = VALID_REQUIREMENTS.model_copy(update={"results": SDMResults(key_predictors=[])})
    report = validate(req)
    assert not any(v.field_path.startswith("results.key_predictors") for v in report.violations)


def test_variables_empty_skips_check():
    req = VALID_REQUIREMENTS.model_copy(
        update={
            "predictors": EnvironmentalPredictors(variables=[]),
            "results": SDMResults(key_predictors=["BIO1"]),
        }
    )
    report = validate(req)
    assert not any(v.field_path.startswith("results.key_predictors") for v in report.violations)


# --- Multiple violations ---


def test_multiple_violations_counted():
    req = VALID_REQUIREMENTS.model_copy(
        update={
            "study": StudyMetadata(title="Test", species=["invalid"]),
            "occurrence": OccurrenceData(occurrence_type="wrong", total_presences=0),
            "models": [
                SDMModelSpec(
                    algorithm="MaxEnt",
                    performance=[PerformanceMetric(metric="AUC", value=1.5)],
                )
            ],
        }
    )
    report = validate(req)
    assert report.num_errors >= 2
    assert report.num_warnings >= 1
    assert not report.is_valid


# ---------------------------------------------------------------------------
# Validator helper tests
# ---------------------------------------------------------------------------


def test_get_critical_errors():
    from lit_review.validators import get_critical_errors

    req = VALID_REQUIREMENTS.model_copy(
        update={
            "occurrence": OccurrenceData(
                occurrence_type="invalid-type",
                total_presences=0,
            ),
        }
    )
    report = validate(req)
    critical = get_critical_errors(report)
    assert len(critical) >= 1
    assert all(v.severity == "error" for v in critical)


def test_get_critical_errors_ignores_warnings():
    from lit_review.validators import get_critical_errors

    req = VALID_REQUIREMENTS.model_copy(
        update={
            "study": StudyMetadata(title="Test", species=["cane toad"]),
        }
    )
    report = validate(req)
    assert report.num_warnings >= 1
    critical = get_critical_errors(report)
    assert len(critical) == 0


def test_violations_by_section():
    from lit_review.validators import violations_by_section

    req = VALID_REQUIREMENTS.model_copy(
        update={
            "occurrence": OccurrenceData(occurrence_type="invalid"),
            "study": StudyMetadata(title="Test", species=["cane toad"]),
        }
    )
    report = validate(req)
    groups = violations_by_section(report)
    assert "occurrence" in groups
    assert "study" in groups
