from __future__ import annotations

import re

from .models import SDMRequirements, ValidationReport, Violation

CRITICAL_FIELD_PREFIXES = {
    "occurrence.total_presences",
    "occurrence.total_absences",
    "occurrence.occurrence_type",
    "models",
}

# ---------------------------------------------------------------------------
# Known vocabularies
# ---------------------------------------------------------------------------

NORMALIZED_METRICS = {"AUC", "TSS", "COR", "Boyce", "kappa", "sensitivity", "specificity"}
ERROR_METRICS = {"RMSE", "MAE", "MSE"}
KNOWN_METRICS = NORMALIZED_METRICS | ERROR_METRICS | {"KUL", "AUCmech", "deviance", "R2"}

KNOWN_ALGORITHMS = {
    "MaxEnt",
    "GLM",
    "GAM",
    "BRT",
    "RF",
    "Random Forest",
    "SVM",
    "ANN",
    "BIOCLIM",
    "DOMAIN",
    "ensemble",
    "GBM",
    "MARS",
    "FDA",
}

OCCURRENCE_TYPES = {"presence-only", "presence-absence", "abundance"}

BINOMIAL_PATTERN = re.compile(r"^[A-Z][a-z]+ [a-z]+(?:\s+(?:subsp\.|var\.)\s+[a-z]+)?$")


def validate(requirements: SDMRequirements) -> ValidationReport:
    violations: list[Violation] = []

    _check_species(requirements, violations)
    _check_occurrence(requirements, violations)
    _check_models(requirements, violations)
    _check_evaluation_metrics(requirements, violations)
    _check_key_predictors(requirements, violations)

    num_errors = sum(1 for v in violations if v.severity == "error")
    num_warnings = len(violations) - num_errors
    return ValidationReport(violations=violations, num_errors=num_errors, num_warnings=num_warnings)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_species(req: SDMRequirements, violations: list[Violation]) -> None:
    for i, name in enumerate(req.study.species):
        if not BINOMIAL_PATTERN.match(name):
            violations.append(
                Violation(
                    field_path=f"study.species[{i}]",
                    rule="Species name should follow binomial nomenclature (Genus species)",
                    actual_value=name,
                    severity="warning",
                )
            )


def _check_occurrence(req: SDMRequirements, violations: list[Violation]) -> None:
    occ = req.occurrence

    if occ.occurrence_type is not None and occ.occurrence_type not in OCCURRENCE_TYPES:
        violations.append(
            Violation(
                field_path="occurrence.occurrence_type",
                rule=f"Must be one of: {', '.join(sorted(OCCURRENCE_TYPES))}",
                actual_value=occ.occurrence_type,
                severity="error",
            )
        )

    if occ.total_presences is not None and occ.total_presences < 1:
        violations.append(
            Violation(
                field_path="occurrence.total_presences",
                rule="Must be >= 1 when set",
                actual_value=str(occ.total_presences),
                severity="error",
            )
        )

    if occ.total_absences is not None and occ.total_absences < 0:
        violations.append(
            Violation(
                field_path="occurrence.total_absences",
                rule="Must be >= 0 when set",
                actual_value=str(occ.total_absences),
                severity="error",
            )
        )

    if occ.occurrence_type == "presence-only" and occ.total_absences is not None:
        violations.append(
            Violation(
                field_path="occurrence.total_absences",
                rule="Must be None for presence-only studies",
                actual_value=str(occ.total_absences),
                severity="error",
            )
        )


def _check_models(req: SDMRequirements, violations: list[Violation]) -> None:
    for i, model in enumerate(req.models):
        if model.algorithm not in KNOWN_ALGORITHMS:
            violations.append(
                Violation(
                    field_path=f"models[{i}].algorithm",
                    rule=f"Unknown algorithm (known: {', '.join(sorted(KNOWN_ALGORITHMS))})",
                    actual_value=model.algorithm,
                    severity="warning",
                )
            )

        for j, pm in enumerate(model.performance):
            metric_upper = pm.metric.upper()

            if any(metric_upper == m.upper() for m in NORMALIZED_METRICS):
                if not (0 <= pm.value <= 1):
                    violations.append(
                        Violation(
                            field_path=f"models[{i}].performance[{j}].value",
                            rule=f"{pm.metric} must be between 0 and 1",
                            actual_value=str(pm.value),
                            severity="error",
                        )
                    )
            elif any(metric_upper == m.upper() for m in ERROR_METRICS):
                if pm.value < 0:
                    violations.append(
                        Violation(
                            field_path=f"models[{i}].performance[{j}].value",
                            rule=f"{pm.metric} must be >= 0",
                            actual_value=str(pm.value),
                            severity="error",
                        )
                    )


def _check_evaluation_metrics(req: SDMRequirements, violations: list[Violation]) -> None:
    for i, metric in enumerate(req.evaluation.metrics_used):
        if not any(metric.upper() == m.upper() for m in KNOWN_METRICS):
            violations.append(
                Violation(
                    field_path=f"evaluation.metrics_used[{i}]",
                    rule=f"Unknown metric (known: {', '.join(sorted(KNOWN_METRICS))})",
                    actual_value=metric,
                    severity="warning",
                )
            )


def _check_key_predictors(req: SDMRequirements, violations: list[Violation]) -> None:
    variables = req.predictors.variables
    key_preds = req.results.key_predictors
    if not variables or not key_preds:
        return

    variables_lower = {v.lower() for v in variables}
    for i, pred in enumerate(key_preds):
        if pred.lower() not in variables_lower:
            violations.append(
                Violation(
                    field_path=f"results.key_predictors[{i}]",
                    rule="Must be a subset of predictors.variables",
                    actual_value=pred,
                    severity="error",
                )
            )


def get_critical_errors(report: ValidationReport) -> list[Violation]:
    return [
        v
        for v in report.violations
        if v.severity == "error"
        and any(v.field_path.startswith(p) for p in CRITICAL_FIELD_PREFIXES)
    ]


def violations_by_section(report: ValidationReport) -> dict[str, list[Violation]]:
    groups: dict[str, list[Violation]] = {}
    for v in report.violations:
        section = v.field_path.split(".")[0].split("[")[0]
        groups.setdefault(section, []).append(v)
    return groups
