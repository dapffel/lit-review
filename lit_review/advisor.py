"""Turn the failures of similar past runs into advice for the current run.

This is the synthesis half of the run-history RAG loop. Retrieval (embedding a
paper and finding its nearest neighbors in the run log) lives in ``RunAdvisor``;
everything here is a pure function of a set of neighbor ``RunRecord``s, so the
logic that decides *which fields to warn about* is testable without any
embedding calls.

Only *confirmed* failures feed advice — a field that merely had weak evidence,
or that the extractor guessed at without any gold/verifier signal, is noise. See
``_is_confirmed``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import faiss
import litellm
import numpy as np

from .models import ErrorAnalysisRow, FieldHint, PaperAdvice, RunRecord
from .run_store import RunStore, _normalize_path

Embedder = Callable[[list[str]], Awaitable["np.ndarray"]]

# Failure types we trust as real signal: gold mismatches, verifier-flagged
# contradictions/unsupported values, and deterministic validation errors.
CONFIRMED_FAILURE_TYPES = frozenset(
    {"benchmark_mismatch", "model_hallucination", "missing_evidence", "validation_error"}
)

# How each failure type should be phrased as a caution to the next extraction.
_FAILURE_ADVICE: dict[str, str] = {
    "benchmark_mismatch": "prior extractions here didn't match the gold annotation — double-check it",
    "model_hallucination": "prior extractions here contradicted the paper — verify against the text",
    "missing_evidence": "hard to support from the paper — only extract if the paper clearly states it",
    "validation_error": "prior extractions here produced impossible values — sanity-check ranges and counts",
    "missing_extraction": "often missed on similar papers — look carefully before leaving it empty",
}

_MAX_HINTS = 6
_EVIDENCE_CHARS = 200


def _is_confirmed(row: ErrorAnalysisRow) -> bool:
    """A failure worth advising on: gold/verifier/validation-backed, not a bare guess."""
    if row.failure_type in CONFIRMED_FAILURE_TYPES:
        return True
    # A missing field only counts as confirmed when a gold annotation says it should be there.
    return row.failure_type == "missing_extraction" and row.match is False


@dataclass
class _FieldAgg:
    papers: set[str] = field(default_factory=set)
    breakdown: Counter[str] = field(default_factory=Counter)
    evidence: str | None = None


def _message(agg: _FieldAgg) -> str:
    n = len(agg.papers)
    dominant = agg.breakdown.most_common(1)[0][0]
    phrase = _FAILURE_ADVICE.get(dominant, "was unreliable on similar papers — double-check it")
    papers_word = "paper" if n == 1 else "papers"
    return f"flagged on {n} similar {papers_word}; {phrase}"


def synthesize_advice(neighbors: list[RunRecord], *, max_hints: int = _MAX_HINTS) -> PaperAdvice:
    """Distill confirmed failures across neighbor runs into per-field cautions.

    Support is counted per *paper*, not per row, so one paper that fumbles a field
    across several list items can't inflate a hint on its own.
    """
    per_field: dict[str, _FieldAgg] = {}
    neighbor_ids: list[str] = []

    for index, record in enumerate(neighbors):
        neighbor_id = record.paper_id or f"run-{index}"
        neighbor_ids.append(neighbor_id)
        for row in record.rows:
            if not _is_confirmed(row):
                continue
            path = _normalize_path(row.field_path)
            agg = per_field.setdefault(path, _FieldAgg())
            agg.papers.add(neighbor_id)
            assert row.failure_type is not None  # guaranteed by _is_confirmed
            agg.breakdown[row.failure_type] += 1
            if agg.evidence is None and row.evidence:
                agg.evidence = row.evidence[:_EVIDENCE_CHARS]

    hints = [
        FieldHint(
            field_path=path,
            message=_message(agg),
            based_on_n=len(agg.papers),
            failure_breakdown=dict(agg.breakdown),
            example_evidence=agg.evidence,
        )
        for path, agg in per_field.items()
    ]
    # Most-corroborated first: by number of papers, then total confirmed failures.
    hints.sort(key=lambda h: (h.based_on_n, sum(h.failure_breakdown.values())), reverse=True)

    return PaperAdvice(hints=hints[:max_hints], neighbor_ids=neighbor_ids)


class RunAdvisor:
    """Retrieval half of the loop: find the past runs most similar to the current
    paper and hand them to :func:`synthesize_advice`.

    Embeds every stored run's signature and the current signature, then takes the
    nearest neighbors by L2 distance. Embedding all of history each call is O(N)
    embedding rows; fine for hundreds of papers via one batched call, and the
    ``embedder`` seam lets tests skip the API entirely.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
        embedder: Embedder | None = None,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self._embedder = embedder

    async def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embedder is not None:
            return await self._embedder(texts)
        response = await litellm.aembedding(model=self.embedding_model, input=texts)
        return np.array([d["embedding"] for d in response.data], dtype=np.float32)

    async def advise(
        self,
        signature: str,
        *,
        exclude_paper_id: str | None = None,
        prompt_version: str | None = None,
    ) -> PaperAdvice:
        """Advise the current paper from its nearest neighbors in the run log.

        ``exclude_paper_id`` drops the paper's own past runs so a re-run doesn't
        advise itself. ``prompt_version``, when given, restricts history to runs
        made under the same prompt, so stale advice can't survive a prompt change.
        """
        if not signature:
            return PaperAdvice()

        records = [
            r
            for r in self.store.load()
            if r.signature
            and r.paper_id != exclude_paper_id
            and (prompt_version is None or r.prompt_version == prompt_version)
        ]
        if not records:
            return PaperAdvice()

        doc_vectors = await self._embed([r.signature for r in records])
        query_vector = await self._embed([signature])

        index = faiss.IndexFlatL2(doc_vectors.shape[1])
        index.add(doc_vectors)
        k = min(self.top_k, len(records))
        _, indices = index.search(query_vector, k)
        neighbors = [records[i] for i in indices[0] if 0 <= i < len(records)]

        return synthesize_advice(neighbors)
