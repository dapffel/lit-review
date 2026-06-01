from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lit_review import (
    AgentConfig,
    EnvironmentalPredictors,
    EvaluationProtocol,
    ExtractionEval,
    FieldVerification,
    OccurrenceData,
    PerformanceMetric,
    ProjectedScenario,
    SDMExtractionAgent,
    SDMModelSpec,
    SDMRequirements,
    SDMResults,
    StudyMetadata,
)
from lit_review.agent import _build_eval_content, _retrieval_query
from lit_review.prompts import EVAL_EXTRACTION_PREFIX, EVAL_PAPER_PREFIX

FAKE_REQUIREMENTS = SDMRequirements(
    study=StudyMetadata(
        title="The art of modelling range-shifting species",
        species=["Bufo marinus"],
        geographic_extent="Australia (invaded range)",
    ),
    occurrence=OccurrenceData(
        occurrence_type="presence-absence",
        total_presences=1183,
        total_absences=451,
        data_source="Urban et al. (2007), plus 270 additional records from 2006",
    ),
    predictors=EnvironmentalPredictors(
        variables=["clim1", "clim3", "clim4", "clim5", "clim8", "clim12", "clim18", "humidity"],
        data_source="Anuclim (ANU 2009)",
        spatial_resolution="0.05 deg (~5 km)",
    ),
    models=[
        SDMModelSpec(
            algorithm="MaxEnt",
            variant="hinge-only, smooth",
            software="MaxEnt v3.3.1",
            hyperparameters="regularization multiplier = 2.5, hinge features only",
            performance=[
                PerformanceMetric(metric="AUC", value=0.79, context="cross-validated"),
                PerformanceMetric(metric="COR", value=0.82, context="current climate"),
            ],
            is_best=True,
        ),
        SDMModelSpec(
            algorithm="BRT",
            software="R gbm v1.6.3",
            hyperparameters="tree complexity = 5, learning rate for >= 1000 trees",
            performance=[
                PerformanceMetric(metric="AUC", value=0.77, context="cross-validated"),
            ],
        ),
    ],
    evaluation=EvaluationProtocol(
        cv_strategy="10-fold cross-validation",
        metrics_used=["AUC", "COR", "KUL", "AUCmech"],
        threshold_method="3 months suitable for breeding",
    ),
    results=SDMResults(
        key_predictors=["humidity", "clim1", "clim4"],
        variable_importance="humidity consistently most important across models",
        current_distribution="Suitable habitat across eastern and northern Australia",
        projected_scenarios=[
            ProjectedScenario(
                scenario_name="20xx extreme",
                description="+2.8-5.4 deg C annual mean temperature",
            ),
        ],
        projected_distribution="Smoother models predicted northern Australia as suitable",
    ),
)

FAKE_EVAL = ExtractionEval(
    field_verifications=[
        FieldVerification(
            field_path="study.species",
            extracted_value="['Bufo marinus']",
            status="verified",
            evidence="the cane toad (Bufo marinus)",
        ),
        FieldVerification(
            field_path="models[0].algorithm",
            extracted_value="MaxEnt",
            status="verified",
            evidence="Models were fit using MaxEnt",
        ),
    ],
    overall_assessment="Extraction accurately reflects the paper.",
)

FAKE_TEXT = "Fake paper content about cane toad species distribution modeling."


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


@patch("lit_review.agent.extract_text", return_value=FAKE_TEXT)
@patch("lit_review.agent.instructor")
async def test_extract_from_pdf_with_references(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_REQUIREMENTS)
    mock_instructor.from_litellm.return_value.create = mock_create

    agent = SDMExtractionAgent()

    with patch("lit_review.agent.VectorMemory") as mock_memory_cls:
        mock_memory = AsyncMock()
        mock_memory.query.return_value = ["relevant context"]
        mock_memory_cls.return_value = mock_memory

        result = await agent.extract_from_pdf("fake.pdf", references=["ref1"])

    assert result.study.species == ["Bufo marinus"]
    assert result.occurrence.total_presences == 1183
    assert len(result.models) == 2
    assert result.models[0].is_best is True
    mock_extract.assert_called_once_with("fake.pdf")
    mock_memory.add.assert_called_once()
    mock_memory.query.assert_called_once()
    mock_create.assert_called_once()


@patch("lit_review.agent.extract_text", return_value=FAKE_TEXT)
@patch("lit_review.agent.instructor")
async def test_extract_from_pdf_no_references(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_REQUIREMENTS)
    mock_instructor.from_litellm.return_value.create = mock_create

    agent = SDMExtractionAgent()

    with patch("lit_review.agent.VectorMemory") as mock_memory_cls:
        result = await agent.extract_from_pdf("fake.pdf")

    assert result.study.title == "The art of modelling range-shifting species"
    mock_memory_cls.assert_not_called()
    mock_create.assert_called_once()


@patch("lit_review.agent.extract_text", return_value="")
@patch("lit_review.agent.instructor")
async def test_empty_pdf_raises(mock_instructor, mock_extract):
    agent = SDMExtractionAgent()
    with pytest.raises(ValueError, match="no extractable text"):
        await agent.extract_from_pdf("empty.pdf")


@patch("lit_review.agent.extract_text")
@patch("lit_review.agent.instructor")
async def test_long_text_truncated(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_REQUIREMENTS)
    mock_instructor.from_litellm.return_value.create = mock_create
    mock_extract.return_value = "x" * 5000

    agent = SDMExtractionAgent(AgentConfig(max_input_chars=100))
    await agent.extract_from_pdf("big.pdf")

    call_args = mock_create.call_args
    user_msg = call_args.kwargs["messages"][1]["content"]
    assert len(user_msg) == 100


@patch("lit_review.agent.extract_text")
@patch("lit_review.agent.instructor")
async def test_prompt_with_context_respects_max_input_chars(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_REQUIREMENTS)
    mock_instructor.from_litellm.return_value.create = mock_create
    mock_extract.return_value = "x" * 5000

    agent = SDMExtractionAgent(AgentConfig(max_input_chars=200))

    with patch("lit_review.agent.VectorMemory") as mock_memory_cls:
        mock_memory = AsyncMock()
        mock_memory.query.return_value = ["context " * 20]
        mock_memory_cls.return_value = mock_memory

        await agent.extract_from_pdf("big.pdf", references=["ref1"])

    user_msg = mock_create.call_args.kwargs["messages"][1]["content"]
    assert len(user_msg) == 200


def test_retrieval_query_uses_first_substantial_paragraph():
    text = "Title\n\nShort.\n\n" + ("This paragraph has enough ecological detail. " * 4)
    query = _retrieval_query(text)
    assert query.startswith("This paragraph")


@patch("lit_review.pdf.fitz")
def test_pdf_document_closed(mock_fitz):
    mock_doc = MagicMock()
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)
    mock_doc.__iter__ = MagicMock(return_value=iter([]))
    mock_fitz.open.return_value = mock_doc

    from lit_review.pdf import extract_text

    extract_text("test.pdf")
    mock_doc.__exit__.assert_called_once()


def test_config_validation():
    with pytest.raises(ValueError):
        AgentConfig(temperature=2.0)
    with pytest.raises(ValueError):
        AgentConfig(chunk_size=-1)
    with pytest.raises(ValueError):
        AgentConfig(max_input_chars=0)

    config = AgentConfig(model="anthropic/claude-sonnet-4-6", temperature=0.1)
    assert config.model == "anthropic/claude-sonnet-4-6"
    assert config.temperature == 0.1


# ---------------------------------------------------------------------------
# Model structure tests
# ---------------------------------------------------------------------------


def test_sdm_requirements_sections():
    data = FAKE_REQUIREMENTS.model_dump()
    assert set(data.keys()) == {
        "study",
        "occurrence",
        "predictors",
        "models",
        "evaluation",
        "results",
    }


def test_sdm_requirements_minimal():
    minimal = SDMRequirements(study=StudyMetadata(title="A paper"))
    assert minimal.study.title == "A paper"
    assert minimal.study.species == []
    assert minimal.occurrence.total_presences is None
    assert minimal.predictors.variables == []
    assert minimal.models == []
    assert minimal.evaluation.metrics_used == []
    assert minimal.results.key_predictors == []


def test_species_is_list():
    study = StudyMetadata(title="Test", species=["Quercus robur", "Fagus sylvatica"])
    assert len(study.species) == 2
    assert study.species[0] == "Quercus robur"


def test_variables_is_list():
    preds = EnvironmentalPredictors(variables=["BIO1", "BIO12", "elevation"])
    assert len(preds.variables) == 3


def test_multi_model_specs():
    assert len(FAKE_REQUIREMENTS.models) == 2
    assert FAKE_REQUIREMENTS.models[0].algorithm == "MaxEnt"
    assert FAKE_REQUIREMENTS.models[0].is_best is True
    assert FAKE_REQUIREMENTS.models[1].algorithm == "BRT"
    assert FAKE_REQUIREMENTS.models[1].is_best is False


def test_performance_metrics_are_numeric():
    metrics = FAKE_REQUIREMENTS.models[0].performance
    assert len(metrics) == 2
    assert metrics[0].metric == "AUC"
    assert metrics[0].value == 0.79
    assert isinstance(metrics[0].value, float)


def test_occurrence_counts_are_int():
    assert FAKE_REQUIREMENTS.occurrence.total_presences == 1183
    assert FAKE_REQUIREMENTS.occurrence.total_absences == 451
    assert isinstance(FAKE_REQUIREMENTS.occurrence.total_presences, int)


def test_projected_scenarios():
    scenarios = FAKE_REQUIREMENTS.results.projected_scenarios
    assert len(scenarios) == 1
    assert scenarios[0].scenario_name == "20xx extreme"


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


@patch("lit_review.agent.extract_text", return_value=FAKE_TEXT)
@patch("lit_review.agent.instructor")
async def test_evaluate_calls_llm_with_eval_prompt(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_EVAL)
    mock_instructor.from_litellm.return_value.create = mock_create

    agent = SDMExtractionAgent()
    result = await agent.evaluate(FAKE_REQUIREMENTS, "fake.pdf")

    assert result.num_verified == 2
    assert result.num_inaccurate == 0
    assert len(result.field_verifications) == 2
    assert result.field_verifications[0].status == "verified"

    call_args = mock_create.call_args
    assert call_args.kwargs["response_model"] is ExtractionEval
    user_msg = call_args.kwargs["messages"][1]["content"]
    assert call_args.kwargs["model"] == "gpt-4"
    assert EVAL_EXTRACTION_PREFIX in user_msg
    assert FAKE_TEXT in user_msg


@patch("lit_review.agent.extract_text", return_value=FAKE_TEXT)
@patch("lit_review.agent.instructor")
async def test_evaluate_uses_eval_model(mock_instructor, mock_extract):
    mock_create = AsyncMock(return_value=FAKE_EVAL)
    mock_instructor.from_litellm.return_value.create = mock_create

    config = AgentConfig(model="gpt-4", eval_model="anthropic/claude-sonnet-4-6")
    agent = SDMExtractionAgent(config)
    await agent.evaluate(FAKE_REQUIREMENTS, "fake.pdf")

    call_args = mock_create.call_args
    assert call_args.kwargs["model"] == "anthropic/claude-sonnet-4-6"


@patch("lit_review.agent.extract_text", return_value="")
@patch("lit_review.agent.instructor")
async def test_evaluate_empty_pdf_raises(mock_instructor, mock_extract):
    agent = SDMExtractionAgent()
    with pytest.raises(ValueError, match="no extractable text"):
        await agent.evaluate(FAKE_REQUIREMENTS, "empty.pdf")


def test_eval_content_respects_max_chars():
    req_json = '{"study": {"title": "Test"}}'
    paper = "x" * 5000
    content = _build_eval_content(req_json, paper, max_chars=200)
    assert len(content) == 200
    assert content.startswith(EVAL_EXTRACTION_PREFIX)


def test_drop_empty_preserves_evidence():
    from lit_review.agent import _drop_empty

    data = {
        "study": {
            "title": "Test",
            "species": ["Bufo marinus"],
            "evidence": "the cane toad (Bufo marinus)",
            "geographic_extent": None,
        }
    }
    cleaned = _drop_empty(data)
    assert cleaned["study"]["evidence"] == "the cane toad (Bufo marinus)"
    assert "geographic_extent" not in cleaned["study"]


def test_eval_content_includes_both_sections():
    req_json = '{"models": [{"algorithm": "MaxEnt"}]}'
    paper = "This paper describes a MaxEnt species distribution model."
    content = _build_eval_content(req_json, paper, max_chars=10_000)
    assert EVAL_EXTRACTION_PREFIX in content
    assert req_json in content
    assert paper in content


def test_field_verification_model():
    fv = FieldVerification(
        field_path="models[0].algorithm",
        extracted_value="MaxEnt",
        status="verified",
        evidence="The authors used MaxEnt v3.4.4",
    )
    assert fv.status == "verified"

    fv_no_evidence = FieldVerification(
        field_path="predictors.variables",
        extracted_value="['BIO1', 'BIO12']",
        status="unverifiable",
    )
    assert fv_no_evidence.evidence is None


def test_extraction_eval_model():
    eval_result = ExtractionEval(
        field_verifications=[
            FieldVerification(
                field_path="study.species",
                extracted_value="['Quercus robur']",
                status="verified",
                evidence="Study species was Q. robur",
            ),
            FieldVerification(
                field_path="occurrence.total_presences",
                extracted_value="500",
                status="inaccurate",
                evidence="Paper states 350 presence records",
            ),
        ],
        overall_assessment="One field inaccurate: presence count mismatch.",
    )
    assert len(eval_result.field_verifications) == 2
    assert eval_result.num_verified == 1
    assert eval_result.num_inaccurate == 1


def test_computed_eval_counts_in_model_dump():
    eval_result = ExtractionEval(
        field_verifications=[
            FieldVerification(
                field_path="study.species",
                extracted_value="['Bufo marinus']",
                status="verified",
            ),
            FieldVerification(
                field_path="occurrence.total_presences",
                extracted_value="500",
                status="inaccurate",
            ),
            FieldVerification(
                field_path="predictors.variables",
                extracted_value="['BIO1']",
                status="unverifiable",
            ),
        ],
        overall_assessment="Mixed results.",
    )
    data = eval_result.model_dump()
    assert data["num_verified"] == 1
    assert data["num_inaccurate"] == 1
    assert data["num_unverifiable"] == 1
