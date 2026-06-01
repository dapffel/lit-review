from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class AgentConfig(BaseModel):
    model: str = "gpt-4"
    eval_model: str | None = Field(
        default=None,
        description=(
            "Model used for the evaluation/verification step. "
            "Defaults to the extraction model when not set."
        ),
    )
    embedding_model: str = "text-embedding-ada-002"
    temperature: float = Field(default=0.2, ge=0, le=1)
    max_reference_docs: int = Field(default=10, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_input_chars: int = Field(default=100_000, gt=0)


# ---------------------------------------------------------------------------
# Extraction models — machine-readable, typed fields
# ---------------------------------------------------------------------------


class StudyMetadata(BaseModel):
    title: str = Field(description="Title of the paper")
    species: list[str] = Field(
        default_factory=list,
        description=(
            "List of focal species or taxonomic groups, using scientific names. "
            "Example: ['Bufo marinus', 'Quercus robur']"
        ),
    )
    geographic_extent: str | None = Field(
        default=None,
        description=(
            "Concise geographic scope. "
            "Example: 'Australia', 'Iberian Peninsula (35-44N, 10W-4E)'"
        ),
    )
    evidence: str | None = Field(
        default=None,
        description="Key quotes from the paper supporting study metadata extraction.",
    )


class OccurrenceData(BaseModel):
    occurrence_type: str | None = Field(
        default=None,
        description="One of: 'presence-only', 'presence-absence', or 'abundance'.",
    )
    total_presences: int | None = Field(
        default=None,
        description="Number of presence records used.",
    )
    total_absences: int | None = Field(
        default=None,
        description=(
            "Number of absence or pseudo-absence records. " "Null for presence-only studies."
        ),
    )
    sample_size_details: str | None = Field(
        default=None,
        description=(
            "Additional context on sampling: thinning, train/test split sizes, "
            "pseudo-absence generation method. Keep concise."
        ),
    )
    data_source: str | None = Field(
        default=None,
        description="Data source(s). Example: 'GBIF, field surveys 2010-2020'",
    )
    evidence: str | None = Field(
        default=None,
        description="Key quotes supporting occurrence data extraction.",
    )


class EnvironmentalPredictors(BaseModel):
    variables: list[str] = Field(
        default_factory=list,
        description=(
            "List of environmental variable names or codes retained in the final model. "
            "Use codes when available. "
            "Example: ['BIO1', 'BIO12', 'elevation', 'NDVI']"
        ),
    )
    data_source: str | None = Field(
        default=None,
        description="Source dataset(s) with versions. Example: 'WorldClim v2.1, SoilGrids 250m'",
    )
    spatial_resolution: str | None = Field(
        default=None,
        description="Grid cell size. Example: '30 arc-seconds (~1 km)'",
    )
    temporal_range: str | None = Field(
        default=None,
        description="Time period of environmental data. Example: '1970-2000'",
    )
    evidence: str | None = Field(
        default=None,
        description="Key quotes supporting predictor data extraction.",
    )


class PerformanceMetric(BaseModel):
    metric: str = Field(description="Metric name. Example: 'AUC', 'TSS', 'COR', 'Boyce', 'RMSE'")
    value: float = Field(description="Numeric value of the metric.")
    std: float | None = Field(
        default=None,
        description="Standard deviation or uncertainty, if reported.",
    )
    context: str | None = Field(
        default=None,
        description=(
            "Context for this metric value. "
            "Example: 'cross-validated', 'test set', 'current climate', '2070 SSP5-8.5'"
        ),
    )


class SDMModelSpec(BaseModel):
    algorithm: str = Field(
        description="Algorithm name. Example: 'MaxEnt', 'GLM', 'BRT', 'Random Forest'"
    )
    variant: str | None = Field(
        default=None,
        description=(
            "Model variant if applicable. "
            "Example: 'smooth', 'default', 'hinge-only', 'with interaction terms'"
        ),
    )
    software: str | None = Field(
        default=None,
        description="Software and version. Example: 'R dismo v1.3-9', 'MaxEnt v3.4.4'",
    )
    hyperparameters: str | None = Field(
        default=None,
        description=(
            "Key settings for this specific model. "
            "Example: 'regularization multiplier = 1.5, feature classes = LQH'"
        ),
    )
    performance: list[PerformanceMetric] = Field(
        default_factory=list,
        description="Performance metrics reported for this model.",
    )
    is_best: bool = Field(
        default=False,
        description="True if this was the best-performing or recommended model.",
    )


class EvaluationProtocol(BaseModel):
    cv_strategy: str | None = Field(
        default=None,
        description=(
            "Cross-validation or splitting strategy. "
            "Example: '10-fold CV', 'spatial block CV', '70/30 random split'"
        ),
    )
    metrics_used: list[str] = Field(
        default_factory=list,
        description="List of metric names used for evaluation. Example: ['AUC', 'TSS']",
    )
    threshold_method: str | None = Field(
        default=None,
        description=(
            "Method for converting continuous suitability to binary predictions. "
            "Example: 'maximum sensitivity + specificity'. Null if not reported."
        ),
    )
    evidence: str | None = Field(
        default=None,
        description="Key quotes supporting evaluation protocol extraction.",
    )


class ProjectedScenario(BaseModel):
    scenario_name: str = Field(
        description="Scenario identifier. Example: 'SSP5-8.5 2070', 'RCP4.5 2050', '20xx extreme'"
    )
    description: str | None = Field(
        default=None,
        description="Brief description of the scenario conditions.",
    )


class SDMResults(BaseModel):
    key_predictors: list[str] = Field(
        default_factory=list,
        description=(
            "Most important predictor variables, ordered by importance. "
            "Use the same names as in predictors.variables. "
            "Example: ['BIO1', 'BIO12', 'elevation']"
        ),
    )
    variable_importance: str | None = Field(
        default=None,
        description=(
            "Numeric importance values or contribution percentages if reported. "
            "Example: 'BIO1: 45%, BIO12: 27%, elevation: 15%'"
        ),
    )
    current_distribution: str | None = Field(
        default=None,
        description="Brief description of predicted current distribution.",
    )
    projected_scenarios: list[ProjectedScenario] = Field(
        default_factory=list,
        description="Future climate scenarios evaluated, if any.",
    )
    projected_distribution: str | None = Field(
        default=None,
        description="Brief description of projected distribution changes.",
    )
    evidence: str | None = Field(
        default=None,
        description="Key quotes supporting results extraction.",
    )


class PaperSections(BaseModel):
    raw_text: str = Field(description="Full original text for fallback")
    sections: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of normalized section heading to section text. "
            "Keys: 'abstract', 'introduction', 'methods', 'results', "
            "'discussion', 'references', 'supplementary'."
        ),
    )

    def get_sections(self, *names: str) -> str:
        parts = [self.sections[n] for n in names if n in self.sections]
        return "\n\n".join(parts) if parts else self.raw_text


class SDMRequirements(BaseModel):
    study: StudyMetadata = Field(description="Study metadata and focal species")
    occurrence: OccurrenceData = Field(
        default_factory=OccurrenceData,
        description="Species occurrence data used in the SDM",
    )
    predictors: EnvironmentalPredictors = Field(
        default_factory=EnvironmentalPredictors,
        description="Environmental predictor variables and data sources",
    )
    models: list[SDMModelSpec] = Field(
        default_factory=list,
        description=(
            "One entry per modeling algorithm or variant tested. "
            "For single-model studies, this list has one entry. "
            "For comparison or ensemble studies, one entry per algorithm/variant."
        ),
    )
    evaluation: EvaluationProtocol = Field(
        default_factory=EvaluationProtocol,
        description="Evaluation strategy: cross-validation, metrics, thresholding",
    )
    results: SDMResults = Field(
        default_factory=SDMResults,
        description="Key findings: predictor importance and distribution predictions",
    )


# ---------------------------------------------------------------------------
# Evaluation models — cross-reference check
# ---------------------------------------------------------------------------


class FieldVerification(BaseModel):
    field_path: str = Field(
        description=(
            "Dot-separated path to the field being verified. "
            "Use indexing for list items. "
            "Example: 'models[0].algorithm', 'predictors.variables', 'study.species'"
        )
    )
    extracted_value: str = Field(description="The value that was extracted for this field")
    status: str = Field(
        description=(
            "'verified' if the extraction accurately reflects the paper, "
            "'inaccurate' if it contradicts or misrepresents the paper, "
            "'unverifiable' if the paper does not clearly state this information"
        ),
    )
    evidence: str | None = Field(
        default=None,
        description=(
            "Brief quote or paraphrase from the paper "
            "supporting or contradicting the extracted value"
        ),
    )


class ExtractionEval(BaseModel):
    field_verifications: list[FieldVerification] = Field(
        description="Verification result for each substantive extracted field"
    )
    overall_assessment: str = Field(
        description="Brief overall assessment of extraction quality and any key issues found"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_verified(self) -> int:
        return sum(1 for fv in self.field_verifications if fv.status == "verified")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_inaccurate(self) -> int:
        return sum(1 for fv in self.field_verifications if fv.status == "inaccurate")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_unverifiable(self) -> int:
        return sum(1 for fv in self.field_verifications if fv.status == "unverifiable")


# ---------------------------------------------------------------------------
# Constraint validation models
# ---------------------------------------------------------------------------


class Violation(BaseModel):
    field_path: str = Field(description="Dot-separated path, e.g. 'models[0].performance[0].value'")
    rule: str = Field(description="Human-readable rule that was violated")
    actual_value: str = Field(description="The problematic value as a string")
    severity: Literal["error", "warning"] = Field(
        description="'error' = impossible value, 'warning' = unexpected but possible"
    )


class ValidationReport(BaseModel):
    violations: list[Violation] = Field(default_factory=list)
    num_errors: int = Field(default=0)
    num_warnings: int = Field(default=0)

    @property
    def is_valid(self) -> bool:
        return self.num_errors == 0


# ---------------------------------------------------------------------------
# Confidence models
# ---------------------------------------------------------------------------


class FieldConfidence(BaseModel):
    field_path: str = Field(description="Dot-separated path, e.g. 'study.species'")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level based on evidence quality"
    )
    reason: str = Field(description="Why this confidence level was assigned")


class ConfidenceReport(BaseModel):
    field_scores: list[FieldConfidence] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_high(self) -> int:
        return sum(1 for f in self.field_scores if f.confidence == "high")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_low(self) -> int:
        return sum(1 for f in self.field_scores if f.confidence == "low")


# ---------------------------------------------------------------------------
# Quality models
# ---------------------------------------------------------------------------


class QualityScore(BaseModel):
    score: float = Field(ge=0, le=1, description="Overall quality 0-1")
    grade: Literal["pass", "marginal", "fail"] = Field(
        description="pass >= 0.7, marginal >= 0.4, fail < 0.4"
    )
    reasons: list[str] = Field(default_factory=list, description="Key factors affecting the score")


class PipelineResult(BaseModel):
    requirements: SDMRequirements
    sections_used: list[str] = Field(
        default_factory=list, description="Which paper sections were identified"
    )
    confidence: ConfidenceReport | None = None
    validation: ValidationReport | None = None
    evaluation: ExtractionEval | None = None
    quality: QualityScore | None = None
    retries_performed: int = Field(default=0)


# ---------------------------------------------------------------------------
# Benchmark models
# ---------------------------------------------------------------------------


class FieldScore(BaseModel):
    field_path: str = Field(description="Dot-separated path, e.g. 'study.species'")
    match: bool = Field(description="Whether extracted value matches gold standard")
    expected: str = Field(description="Gold-standard value as string")
    actual: str = Field(description="Extracted value as string")


class BenchmarkResult(BaseModel):
    paper_id: str = Field(description="Identifier for the benchmarked paper")
    scores: list[FieldScore] = Field(default_factory=list)
    precision: float = Field(default=0.0, description="Correct extractions / total extracted")
    recall: float = Field(default=0.0, description="Correct extractions / total gold fields")


class BenchmarkSummary(BaseModel):
    results: list[BenchmarkResult] = Field(default_factory=list)
    overall_precision: float = Field(default=0.0)
    overall_recall: float = Field(default=0.0)
    per_field_accuracy: dict[str, float] = Field(default_factory=dict)
