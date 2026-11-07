"""A tool run must be closed when the call is cancelled.

``ToolRun.arun`` catches ``BaseException`` and awaits ``on_tool_error``, but a bare
``except BaseException`` is not enough when the cancel arrives from an anyio cancel
scope (what Starlette/FastAPI use when a client disconnects): every await inside a
cancelled scope re-raises, so the callback is cut off unless the manager shields it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.tools import BaseTool


class _Recorder(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.errors: list[BaseException] = []

    async def on_tool_start(self, *_args: Any, **_kwargs: Any) -> None:
        self.events.append("start")

    async def on_tool_end(self, *_args: Any, **_kwargs: Any) -> None:
        self.events.append("end")

    async def on_tool_error(
        self, error: BaseException, *_args: Any, **_kwargs: Any
    ) -> None:
        self.events.append("error")
        self.errors.append(error)


class _SlowTool(BaseTool):
    name: str = "slow_tool"
    description: str = "a tool that takes its time"

    def _run(self, query: str) -> str:
        return query

    async def _arun(self, query: str) -> str:
        await asyncio.sleep(10)
        return query


def test_cancel_from_anyio_scope_closes_the_tool_run() -> None:
    anyio = pytest.importorskip("anyio")

    recorder = _Recorder()

    async def scenario() -> None:
        with anyio.move_on_after(0.1):
            await _SlowTool().ainvoke({"query": "q"}, config={"callbacks": [recorder]})
        for _ in range(50):
            if len(recorder.events) == 2:
                break
            await asyncio.sleep(0.02)

    anyio.run(scenario)
    assert recorder.events == ["start", "error"]
    assert isinstance(recorder.errors[0], asyncio.CancelledError)
