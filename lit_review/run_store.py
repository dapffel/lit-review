"""Persist pipeline runs and learn from their accumulated history.

Every ``run_pipeline`` call is stateless: it emits a rich ``PipelineResult`` but
nothing survives the process. ``RunStore`` closes that loop by appending one
``RunRecord`` per run to a JSONL file, and ``learn_from_runs`` aggregates those
records into a ``LearningReport`` that highlights the fields extraction gets
wrong most often. That report is the input to offline improvement: tightening
prompts, adding validators, or re-mapping sections for chronically weak fields.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .error_analysis import analyze_pipeline_result, summarize_error_analysis
from .models import (
    ErrorAnalysisRow,
    FieldWeakness,
    LearningReport,
    PipelineResult,
    RunRecord,
    SDMRequirements,
)

_INDEX_RE = re.compile(r"\[\d+\]")


def _normalize_path(field_path: str) -> str:
    """Collapse list indices so ``models[0].algorithm`` and ``models[1].algorithm``
    aggregate under a single ``models.algorithm`` weakness."""
    return _INDEX_RE.sub("", field_path)


class RunStore:
    """Append-only JSONL log of pipeline runs, one ``RunRecord`` per line."""

    def __init__(self, path: str | Path = Path("runs/runs.jsonl")) -> None:
        self.path = Path(path)

    def record(
        self,
        result: PipelineResult,
        *,
        gold: SDMRequirements | None = None,
        paper_id: str | None = None,
        model: str | None = None,
        signature: str = "",
        prompt_version: str | None = None,
    ) -> RunRecord:
        """Turn a pipeline result into a ``RunRecord`` and append it to the log."""
        rows = analyze_pipeline_result(result, gold=gold, paper_id=paper_id)
        record = RunRecord(
            paper_id=paper_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            quality_score=result.quality.score if result.quality is not None else None,
            quality_grade=result.quality.grade if result.quality is not None else None,
            has_gold=gold is not None,
            prompt_version=prompt_version,
            signature=signature,
            rows=rows,
        )
        self.append(record)
        return record

    def append(self, record: RunRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(record.model_dump_json() + "\n")

    def load(self) -> list[RunRecord]:
        if not self.path.exists():
            return []
        records: list[RunRecord] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(RunRecord.model_validate_json(line))
        return records


def _field_weaknesses(rows: list[ErrorAnalysisRow]) -> list[FieldWeakness]:
    totals: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        path = _normalize_path(row.field_path)
        totals[path] += 1
        if row.failure_type is not None:
            failures[path] += 1
            by_type[path][row.failure_type] += 1

    weaknesses = [
        FieldWeakness(
            field_path=path,
            total=total,
            num_failures=failures[path],
            failure_rate=failures[path] / total if total else 0.0,
            by_failure_type=dict(by_type[path]),
        )
        for path, total in totals.items()
    ]
    # Worst first: highest failure rate, breaking ties by absolute failure volume.
    weaknesses.sort(key=lambda w: (w.failure_rate, w.num_failures), reverse=True)
    return weaknesses


def learn_from_runs(records: list[RunRecord]) -> LearningReport:
    """Aggregate stored runs into a report of where extraction is weakest."""
    all_rows = [row for record in records for row in record.rows]
    paper_ids = {record.paper_id for record in records if record.paper_id is not None}

    weak_fields = [w for w in _field_weaknesses(all_rows) if w.num_failures > 0]

    return LearningReport(
        num_runs=len(records),
        num_papers=len(paper_ids),
        summary=summarize_error_analysis(all_rows),
        weak_fields=weak_fields,
    )
