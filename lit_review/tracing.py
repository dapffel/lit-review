"""Optional LangSmith tracing for the core pipeline.

Tracing is opt-in and gated on the environment: when ``LANGSMITH_API_KEY`` is
unset (or ``LANGSMITH_TRACING`` is explicitly disabled), the decorator is a
no-op and ``langsmith`` need not be installed. Install the optional extra with
``pip install "lit-review[tracing]"`` to enable it.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Awaitable, Callable
from typing import Literal, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


def tracing_enabled() -> bool:
    """True when a LangSmith key is present and tracing is not explicitly disabled."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return False
    return os.getenv("LANGSMITH_TRACING", "true").lower() not in ("false", "0", "")


def trace_async(
    name: str, run_type: RunType = "chain"
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Wrap an async function with LangSmith ``@traceable`` when tracing is enabled.

    The enabled check happens at call time (not import time), so tracing picks up
    environment configured after import, e.g. via ``load_dotenv()``.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not tracing_enabled():
                return await func(*args, **kwargs)
            try:
                from langsmith import traceable
            except ImportError:
                return await func(*args, **kwargs)
            traced = traceable(name=name, run_type=run_type)(func)
            # langsmith's wrapper adds a langsmith_extra kwarg to its signature, which
            # collides with our forwarded **kwargs at the type level only.
            return await traced(*args, **kwargs)  # type: ignore[arg-type]

        return wrapper

    return decorator
