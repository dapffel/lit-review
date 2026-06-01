from lit_review.models import PaperSections
from lit_review.sections import SECTION_MAP, get_text_for_field, parse_sections

SAMPLE_PAPER = """This is the abstract of the paper about species distribution modeling.

1. Introduction
Species distribution models are widely used in ecology.

2. Materials and Methods
We used MaxEnt to model the distribution of Bufo marinus.
Occurrence data was obtained from GBIF (n=1183 presences).

3. Results
The model achieved an AUC of 0.79. BIO1 was the most important predictor.

4. Discussion
Our results suggest that temperature is the primary driver.

5. Conclusions
We conclude that SDMs are effective for predicting distributions.

References
Smith et al. 2020. Ecology Letters.
"""

CAPS_PAPER = """Abstract of a paper.

INTRODUCTION
Some background text.

METHODS
Modeling details here.

RESULTS
Performance metrics here.

DISCUSSION
Interpretation here.

REFERENCES
Jones 2021.
"""

NO_HEADINGS_PAPER = """This paper describes a MaxEnt model for cane toads in Australia.
We used 1183 presence records and achieved an AUC of 0.79.
Temperature was the most important predictor variable.
"""


def test_parse_numbered_headings():
    result = parse_sections(SAMPLE_PAPER)
    assert "abstract" in result.sections
    assert "introduction" in result.sections
    assert "methods" in result.sections
    assert "results" in result.sections
    assert "discussion" in result.sections
    assert "references" in result.sections


def test_parse_caps_headings():
    result = parse_sections(CAPS_PAPER)
    assert "introduction" in result.sections
    assert "methods" in result.sections
    assert "results" in result.sections
    assert "discussion" in result.sections


def test_no_headings_fallback():
    result = parse_sections(NO_HEADINGS_PAPER)
    assert result.sections == {} or len(result.sections) == 0
    assert result.raw_text == NO_HEADINGS_PAPER


def test_get_sections_returns_content():
    result = parse_sections(SAMPLE_PAPER)
    methods = result.get_sections("methods")
    assert "MaxEnt" in methods
    assert "GBIF" in methods


def test_get_sections_multiple():
    result = parse_sections(SAMPLE_PAPER)
    combined = result.get_sections("methods", "results")
    assert "MaxEnt" in combined
    assert "AUC" in combined


def test_get_sections_fallback_to_raw():
    result = parse_sections(SAMPLE_PAPER)
    text = result.get_sections("nonexistent")
    assert text == result.raw_text


def test_heading_normalization():
    result = parse_sections(SAMPLE_PAPER)
    assert "materials and methods" not in result.sections
    assert "methods" in result.sections
    assert "conclusions" not in result.sections
    assert "discussion" in result.sections


def test_abstract_from_preamble():
    result = parse_sections(SAMPLE_PAPER)
    assert "abstract" in result.sections
    assert "species distribution modeling" in result.sections["abstract"]


def test_raw_text_always_set():
    result = parse_sections(SAMPLE_PAPER)
    assert result.raw_text == SAMPLE_PAPER


def test_section_map_keys():
    expected_fields = {"study", "occurrence", "predictors", "models", "evaluation", "results"}
    assert set(SECTION_MAP.keys()) == expected_fields


def test_get_text_for_field_uses_primary():
    result = parse_sections(SAMPLE_PAPER)
    text = get_text_for_field(result, "models")
    assert "MaxEnt" in text


def test_get_text_for_field_unknown():
    result = parse_sections(SAMPLE_PAPER)
    text = get_text_for_field(result, "unknown_field")
    assert text == result.raw_text


def test_get_text_for_field_fallback():
    sections = PaperSections(
        raw_text="full text",
        sections={"results": "AUC was 0.79"},
    )
    text = get_text_for_field(sections, "results")
    assert "AUC" in text


def test_paper_sections_model():
    ps = PaperSections(raw_text="hello", sections={"abstract": "abs"})
    assert ps.raw_text == "hello"
    assert ps.get_sections("abstract") == "abs"
