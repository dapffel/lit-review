from __future__ import annotations

import re

from .models import PaperSections

HEADING_RE = re.compile(
    r"^(?:"
    r"\d+\.?\s+[A-Z][A-Za-z &]+"  # numbered: "1. Introduction", "2. Materials and Methods"
    r"|[A-Z][A-Z ]{2,}"  # ALL CAPS: "METHODS", "RESULTS"
    r"|(?:Abstract|Introduction|Methods|Materials and Methods|Study Area|"
    r"Data|Results|Discussion|Conclusions?|Acknowledgements?|References|"
    r"Supplementary|Supporting Information|Appendix)"
    r")\s*$",
    re.MULTILINE,
)

HEADING_NORMALIZE: dict[str, str] = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "methods": "methods",
    "methodology": "methods",
    "materials and methods": "methods",
    "materials & methods": "methods",
    "study area": "methods",
    "study site": "methods",
    "data": "methods",
    "data collection": "methods",
    "model": "methods",
    "modeling approach": "methods",
    "modelling approach": "methods",
    "statistical analysis": "methods",
    "species distribution modelling": "methods",
    "species distribution modeling": "methods",
    "results": "results",
    "model evaluation": "results",
    "model performance": "results",
    "discussion": "discussion",
    "conclusion": "discussion",
    "conclusions": "discussion",
    "acknowledgements": "references",
    "acknowledgments": "references",
    "references": "references",
    "literature cited": "references",
    "supplementary": "supplementary",
    "supporting information": "supplementary",
    "appendix": "supplementary",
}

SECTION_MAP: dict[str, tuple[list[str], list[str]]] = {
    "study": (["abstract", "introduction"], ["abstract"]),
    "occurrence": (["methods"], ["abstract", "methods"]),
    "predictors": (["methods"], ["methods", "results"]),
    "models": (["methods"], ["methods", "results"]),
    "evaluation": (["methods"], ["methods", "results"]),
    "results": (["results"], ["results", "discussion"]),
}


def _normalize_heading(raw: str) -> str:
    cleaned = re.sub(r"^\d+\.?\s*", "", raw).strip().rstrip(".")
    key = cleaned.lower()
    return HEADING_NORMALIZE.get(key, key)


def parse_sections(text: str) -> PaperSections:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return PaperSections(raw_text=text)

    sections: dict[str, str] = {}

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections["abstract"] = preamble

    for i, match in enumerate(matches):
        heading_text = match.group().strip()
        normalized = _normalize_heading(heading_text)

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        if not body:
            continue

        if normalized in sections:
            sections[normalized] += "\n\n" + body
        else:
            sections[normalized] = body

    return PaperSections(raw_text=text, sections=sections)


def get_text_for_field(sections: PaperSections, field: str) -> str:
    if field not in SECTION_MAP:
        return sections.raw_text
    primary, fallback = SECTION_MAP[field]
    if any(name in sections.sections for name in primary):
        return sections.get_sections(*primary)
    if fallback != primary and any(name in sections.sections for name in fallback):
        return sections.get_sections(*fallback)
    return sections.raw_text
