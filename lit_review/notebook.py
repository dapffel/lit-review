from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import litellm
from dotenv import load_dotenv

from .agent import SDMExtractionAgent
from .models import AgentConfig, PipelineResult


@dataclass
class NotebookDemo:
    config: AgentConfig
    agent: SDMExtractionAgent
    run_pipeline: Callable[[str], Awaitable[PipelineResult]]
    langsmith_enabled: bool
    langsmith_project: str | None
    flush_traces: Callable[[], None]


def _choose_model() -> str:
    configured = os.getenv("LIT_REVIEW_MODEL")
    if configured:
        return configured
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-6"
    return "gpt-4o"


def _configure_langsmith(project: str) -> tuple[bool, Callable[[], None]]:
    if not os.getenv("LANGSMITH_API_KEY"):
        return False, lambda: None

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    os.environ.setdefault("LANGSMITH_DEFAULT_RUN_NAME", "SDM LLM Call")

    if "langsmith" not in litellm.success_callback:
        litellm.success_callback.append("langsmith")
    if "langsmith" not in litellm.failure_callback:
        litellm.failure_callback.append("langsmith")

    def flush() -> None:
        from langsmith import Client as LangSmithClient

        LangSmithClient().flush()

    return True, flush


def configure_pipeline_demo(
    *,
    dotenv_path: str | Path = "../.env",
    langsmith_project: str = "ecology-summarizer-demo",
    temperature: float = 0.2,
) -> NotebookDemo:
    load_dotenv(dotenv_path)

    model = _choose_model()
    eval_model = os.getenv("LIT_REVIEW_EVAL_MODEL") or model
    langsmith_enabled, flush_traces = _configure_langsmith(langsmith_project)

    config = AgentConfig(model=model, eval_model=eval_model, temperature=temperature)
    agent = SDMExtractionAgent(config)

    async def run_pipeline(pdf_path: str) -> PipelineResult:
        return await agent.run_pipeline(pdf_path)

    if langsmith_enabled:
        from langsmith import traceable

        run_pipeline = traceable(name="SDMExtractionAgent.run_pipeline", run_type="chain")(
            run_pipeline
        )

    return NotebookDemo(
        config=config,
        agent=agent,
        run_pipeline=run_pipeline,
        langsmith_enabled=langsmith_enabled,
        langsmith_project=os.getenv("LANGSMITH_PROJECT") if langsmith_enabled else None,
        flush_traces=flush_traces,
    )
