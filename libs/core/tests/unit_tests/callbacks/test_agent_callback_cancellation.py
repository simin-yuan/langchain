"""CI probe: agent callbacks must survive a cancelled anyio scope."""

import asyncio
import contextlib
import uuid
from typing import Any

import pytest
from typing_extensions import override

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import (
    AsyncCallbackHandler,
    AsyncCallbackManagerForChainRun,
)


class _Recorder(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def _record(self, name: str) -> None:
        await asyncio.sleep(0)
        self.calls.append(name)

    @override
    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        await self._record("on_agent_action")

    @override
    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        await self._record("on_agent_finish")


@pytest.mark.parametrize(
    ("method", "arg"),
    [
        ("on_agent_action", AgentAction(tool="t", tool_input="i", log="")),
        ("on_agent_finish", AgentFinish(return_values={"o": "x"}, log="")),
    ],
    ids=["on_agent_action", "on_agent_finish"],
)
async def test_agent_callbacks_survive_cancelled_anyio_scope(
    method: str, arg: Any
) -> None:
    anyio = pytest.importorskip("anyio")
    recorder = _Recorder()
    manager = AsyncCallbackManagerForChainRun(
        run_id=uuid.uuid4(), handlers=[recorder], inheritable_handlers=[recorder]
    )

    with anyio.CancelScope() as scope:
        scope.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await getattr(manager, method)(arg)

    for _ in range(100):
        if recorder.calls:
            break
        await asyncio.sleep(0)

    assert recorder.calls == [method]
