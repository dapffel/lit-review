import numpy as np

from lit_review import ErrorAnalysisRow, RunAdvisor, RunRecord, RunStore, synthesize_advice


def _row(field_path: str, failure_type: str | None, **kwargs) -> ErrorAnalysisRow:
    return ErrorAnalysisRow(
        field_path=field_path,
        extracted_value=kwargs.get("extracted_value", "x"),
        failure_type=failure_type,
        evidence=kwargs.get("evidence"),
        match=kwargs.get("match"),
    )


def _record(paper_id: str, rows: list[ErrorAnalysisRow]) -> RunRecord:
    return RunRecord(paper_id=paper_id, timestamp="2026-07-11T00:00:00+00:00", rows=rows)


def test_no_neighbors_yields_empty_advice():
    advice = synthesize_advice([])
    assert advice.hints == []
    assert advice.neighbor_ids == []
    assert advice.as_prompt_block() == ""


def test_only_confirmed_failures_become_hints():
    record = _record(
        "paper-1",
        [
            _row(
                "occurrence.total_absences", "model_hallucination", evidence="no absences reported"
            ),
            _row("predictors.variables", "weak_evidence"),  # not confirmed -> ignored
            _row("study.species", None),  # clean field -> ignored
        ],
    )

    advice = synthesize_advice([record])

    paths = {h.field_path for h in advice.hints}
    assert paths == {"occurrence.total_absences"}
    assert advice.neighbor_ids == ["paper-1"]


def test_missing_extraction_only_counts_when_gold_confirms():
    confirmed = _record("p1", [_row("evaluation.cv_strategy", "missing_extraction", match=False)])
    unconfirmed = _record("p2", [_row("evaluation.cv_strategy", "missing_extraction", match=None)])

    assert {h.field_path for h in synthesize_advice([confirmed]).hints} == {
        "evaluation.cv_strategy"
    }
    assert synthesize_advice([unconfirmed]).hints == []


def test_support_is_counted_per_paper_and_ranked():
    # Field A: flagged by two papers. Field B: flagged many times but by one paper.
    records = [
        _record("p1", [_row("occurrence.total_absences", "model_hallucination")]),
        _record("p2", [_row("occurrence.total_absences", "missing_evidence")]),
        _record(
            "p3",
            [
                _row("models[0].performance[0].value", "validation_error"),
                _row("models[1].performance[0].value", "validation_error"),
                _row("models[2].performance[0].value", "validation_error"),
            ],
        ),
    ]

    advice = synthesize_advice(records)

    top = advice.hints[0]
    assert top.field_path == "occurrence.total_absences"
    assert top.based_on_n == 2  # two distinct papers, not two rows
    assert top.failure_breakdown == {"model_hallucination": 1, "missing_evidence": 1}

    # List indices are normalized so the three model rows collapse to one field, one paper.
    perf = next(h for h in advice.hints if h.field_path == "models.performance.value")
    assert perf.based_on_n == 1
    assert perf.failure_breakdown == {"validation_error": 3}


def test_prompt_block_and_evidence_rendering():
    record = _record(
        "paper-1",
        [
            _row(
                "study.species", "benchmark_mismatch", evidence="The paper studies Rana temporaria."
            )
        ],
    )

    advice = synthesize_advice([record])
    hint = advice.hints[0]

    assert hint.example_evidence == "The paper studies Rana temporaria."
    block = advice.as_prompt_block()
    assert "study.species" in block
    assert "1 similar paper" in block  # singular


def test_max_hints_caps_output():
    rows = [_row(f"predictors.variable_{i}", "benchmark_mismatch") for i in range(10)]
    advice = synthesize_advice([_record("p1", rows)], max_hints=3)
    assert len(advice.hints) == 3


# ---------------------------------------------------------------------------
# RunAdvisor retrieval (embedding seam is faked; no API calls)
# ---------------------------------------------------------------------------


def _fake_embedder(vectors: dict[str, list[float]]):
    """Map exact signature strings to fixed vectors so nearest-neighbor is deterministic."""

    async def embed(texts: list[str]) -> np.ndarray:
        return np.array([vectors[t] for t in texts], dtype=np.float32)

    return embed


def _record_with_sig(paper_id, signature, rows, **kwargs) -> RunRecord:
    return RunRecord(
        paper_id=paper_id,
        timestamp="2026-07-11T00:00:00+00:00",
        signature=signature,
        rows=rows,
        **kwargs,
    )


async def test_advise_empty_signature_or_history(tmp_path):
    store = RunStore(tmp_path / "runs.jsonl")
    advisor = RunAdvisor(store, embedder=_fake_embedder({}))

    # No signature -> nothing to match on.
    assert (await advisor.advise("")).hints == []
    # Signature but empty store -> cold start.
    assert (await advisor.advise("query")).hints == []


async def test_advise_picks_nearest_neighbor(tmp_path):
    store = RunStore(tmp_path / "runs.jsonl")
    store.append(
        _record_with_sig(
            "near", "near-sig", [_row("occurrence.total_absences", "model_hallucination")]
        )
    )
    store.append(_record_with_sig("far", "far-sig", [_row("study.species", "benchmark_mismatch")]))

    vectors = {"query": [0.0, 0.0], "near-sig": [0.1, 0.0], "far-sig": [9.0, 9.0]}
    advisor = RunAdvisor(store, top_k=1, embedder=_fake_embedder(vectors))

    advice = await advisor.advise("query")
    assert advice.neighbor_ids == ["near"]
    assert {h.field_path for h in advice.hints} == {"occurrence.total_absences"}


async def test_advise_excludes_own_history_and_off_version(tmp_path):
    store = RunStore(tmp_path / "runs.jsonl")
    store.append(
        _record_with_sig(
            "self", "self-sig", [_row("study.species", "benchmark_mismatch")], prompt_version="v1"
        )
    )
    store.append(
        _record_with_sig(
            "old",
            "old-sig",
            [_row("predictors.variables", "model_hallucination")],
            prompt_version="v0",
        )
    )
    vectors = {"query": [0.0, 0.0], "self-sig": [0.1, 0.0], "old-sig": [0.2, 0.0]}
    advisor = RunAdvisor(store, embedder=_fake_embedder(vectors))

    # Excluding the paper's own run and off-version history leaves no neighbors.
    advice = await advisor.advise("query", exclude_paper_id="self", prompt_version="v1")
    assert advice.hints == []
