import json
from typing import Any

import instructor
import litellm
from dotenv import load_dotenv

from .memory import VectorMemory
from .models import AgentConfig, ExtractionEval, SDMRequirements
from .pdf import extract_text
from .prompts import (
    EVAL_EXTRACTION_PREFIX,
    EVAL_PAPER_PREFIX,
    EVAL_SYSTEM,
    EXTRACTION_CONTEXT_PREFIX,
    EXTRACTION_PAPER_PREFIX,
    EXTRACTION_SYSTEM,
)

load_dotenv()

QUERY_CHARS = 500
MIN_PARAGRAPH_CHARS = 80


def _retrieval_query(text: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]
    for paragraph in paragraphs:
        if len(paragraph) >= MIN_PARAGRAPH_CHARS:
            return paragraph[:QUERY_CHARS]
    return text[:QUERY_CHARS]


def _build_extraction_content(text: str, context: str, max_chars: int) -> str:
    if not context:
        return f"{EXTRACTION_PAPER_PREFIX}{text}"[:max_chars]

    prefix = f"{EXTRACTION_CONTEXT_PREFIX}{context}\n\n{EXTRACTION_PAPER_PREFIX}"
    available = max_chars - len(prefix)
    if available <= 0:
        context_budget = (
            max_chars - len(EXTRACTION_CONTEXT_PREFIX) - len(f"\n\n{EXTRACTION_PAPER_PREFIX}")
        )
        context = context[: max(0, context_budget)]
        return f"{EXTRACTION_CONTEXT_PREFIX}{context}\n\n{EXTRACTION_PAPER_PREFIX}"[:max_chars]

    return f"{prefix}{text[:available]}"


def _build_eval_content(requirements_json: str, paper_text: str, max_chars: int) -> str:
    prefix = f"{EVAL_EXTRACTION_PREFIX}{requirements_json}{EVAL_PAPER_PREFIX}"
    available = max_chars - len(prefix)
    if available <= 0:
        return prefix[:max_chars]
    return f"{prefix}{paper_text[:available]}"


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            c = _drop_empty(item)
            if c is None or c == [] or c == {}:
                continue
            cleaned[key] = c
        return cleaned or None
    if isinstance(value, list):
        result = [c for item in value if (c := _drop_empty(item)) not in (None, [], {})]
        return result or None
    return value


class SDMExtractionAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.client = instructor.from_litellm(litellm.acompletion)

    async def extract_from_pdf(
        self, pdf_path: str, references: list[str] | None = None
    ) -> SDMRequirements:
        text = extract_text(pdf_path).strip()
        if not text:
            raise ValueError("PDF contains no extractable text")

        context = ""
        if references:
            memory = VectorMemory(
                model=self.config.embedding_model,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            await memory.add(references[: self.config.max_reference_docs])
            context_chunks = await memory.query(_retrieval_query(text))
            context = "\n\n".join(context_chunks)

        user_content = _build_extraction_content(text, context, self.config.max_input_chars)

        return await self.client.create(
            model=self.config.model,
            response_model=SDMRequirements,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
        )

    async def evaluate(self, requirements: SDMRequirements, pdf_path: str) -> ExtractionEval:
        text = extract_text(pdf_path).strip()
        if not text:
            raise ValueError("PDF contains no extractable text")

        cleaned = _drop_empty(requirements.model_dump())
        requirements_json = json.dumps(cleaned, indent=2)
        user_content = _build_eval_content(requirements_json, text, self.config.max_input_chars)

        return await self.client.create(
            model=self.config.eval_model or self.config.model,
            response_model=ExtractionEval,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass
