"""Cancellation that lands while a start callback is still being dispatched.

The start dispatch happens before the ``try`` block that closes the run, so a
cancellation that arrives from inside it leaves the run opened on the tracing
side but never closed: no end, no error. These tests pin that such a run is
closed with an error callback carrying the same ``run_id`` it was opened with.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from typing_extensions import override

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableSequence,
)
from langchain_core.tools import tool as tool_decorator

if TYPE_CHECKING:
    from uuid import UUID

TIMEOUT = 5


class _BlockingStartHandler(AsyncCallbackHandler):
    """Blocks inside the start dispatch of one component until cancelled."""

    def __init__(self, start_event: str, error_event: str) -> None:
        self.start_event = start_event
        self.error_event = error_event
        self.reached = asyncio.Event()
        self.closed = asyncio.Event()
        self.events: list[tuple[str, UUID | None]] = []

    async def _record(self, name: str, run_id: UUID | None) -> None:
        self.events.append((name, run_id))
        if name == self.start_event:
            self.reached.set()
            await asyncio.Event().wait()
        elif name == self.error_event:
            self.closed.set()

    @override
    async def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        await self._record("on_chain_start", kwargs.get("run_id"))

    @override
    async def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        await self._record("on_tool_start", kwargs.get("run_id"))

    @override
    async def on_retriever_start(self, *args: Any, **kwargs: Any) -> None:
        await self._record("on_retriever_start", kwargs.get("run_id"))

    @override
    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        await self._record("on_llm_start", kwargs.get("run_id"))

    @override
    async def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        await self._record("on_chat_model_start", kwargs.get("run_id"))

    @override
    async def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        await self._record("on_llm_error", kwargs.get("run_id"))

    @override
    async def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        await self._record("on_chain_error", kwargs.get("run_id"))

    @override
    async def on_tool_error(self, *args: Any, **kwargs: Any) -> None:
        await self._record("on_tool_error", kwargs.get("run_id"))

    @override
    async def on_retriever_error(self, *args: Any, **kwargs: Any) -> None:
        await self._record("on_retriever_error", kwargs.get("run_id"))


class _StaticRetriever(BaseRetriever):
    """Retriever that never runs: the start dispatch is what is under test."""

    def _get_relevant_documents(self, query: str, **_kwargs: Any) -> list[Document]:
        return [Document(page_content=query)]

    async def _aget_relevant_documents(
        self, query: str, **_kwargs: Any
    ) -> list[Document]:
        return [Document(page_content=query)]


@tool_decorator
async def _echo(text: str) -> str:
    """Echo the text back."""
    return text


async def _invoke(component: str, config: RunnableConfig) -> None:
    if component == "chain":
        await RunnableLambda(lambda x: x).ainvoke("hi", config=config)
    elif component == "tool":
        await _echo.ainvoke({"text": "hi"}, config=config)
    elif component == "retriever":
        await _StaticRetriever().ainvoke("hi", config=config)
    elif component == "llm":
        await FakeListLLM(responses=["ok"]).ainvoke("hi", config=config)
    elif component == "chat_model":
        await FakeListChatModel(responses=["ok"]).ainvoke(
            [("user", "hi")], config=config
        )
    else:  # pragma: no cover
        msg = f"unknown component {component}"
        raise AssertionError(msg)


async def _cancel_during_start(
    handler: _BlockingStartHandler, component: str
) -> list[tuple[str, UUID | None]]:
    config = RunnableConfig(callbacks=[handler])
    task = asyncio.create_task(_invoke(component, config))
    await asyncio.wait_for(handler.reached.wait(), TIMEOUT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The close is dispatched from a detached task, so it lands shortly after.
    await asyncio.wait_for(handler.closed.wait(), TIMEOUT)
    return handler.events


@pytest.mark.parametrize(
    ("component", "start_event", "error_event"),
    [
        ("chain", "on_chain_start", "on_chain_error"),
        ("tool", "on_tool_start", "on_tool_error"),
        ("retriever", "on_retriever_start", "on_retriever_error"),
        ("llm", "on_llm_start", "on_llm_error"),
        ("chat_model", "on_chat_model_start", "on_llm_error"),
    ],
)
async def test_cancelled_start_dispatch_still_closes_the_run(
    component: str, start_event: str, error_event: str
) -> None:
    handler = _BlockingStartHandler(start_event, error_event)

    events = await _cancel_during_start(handler, component)

    opened = [run_id for name, run_id in events if name == start_event]
    closed = [run_id for name, run_id in events if name == error_event]
    assert len(opened) == 1, events
    assert closed == opened, events


async def test_cancelled_start_of_a_nested_root_run_is_closed() -> None:
    """A composite runnable's root run is closed too, not just leaf components."""
    handler = _BlockingStartHandler("on_chain_start", "on_chain_error")
    chain: Runnable[Any, Any] = RunnableSequence(
        RunnableLambda(lambda x: x), RunnableLambda(lambda x: x)
    )

    config = RunnableConfig(callbacks=[handler])
    task = asyncio.create_task(chain.ainvoke("hi", config=config))
    await asyncio.wait_for(handler.reached.wait(), TIMEOUT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(handler.closed.wait(), TIMEOUT)

    # Only the root run was opened, and it is the one that gets closed.
    assert [name for name, _ in handler.events] == [
        "on_chain_start",
        "on_chain_error",
    ], handler.events
    assert handler.events[0][1] == handler.events[1][1]


async def test_cancelled_from_a_cancel_scope_closes_the_run() -> None:
    """A cancel scope cancels every await it covers, not just the first one.

    Starlette and FastAPI cancel through anyio cancel scopes, so this is the
    path a request cancelled mid-flight takes.
    """
    handler = _BlockingStartHandler("on_chain_start", "on_chain_error")
    config = RunnableConfig(callbacks=[handler])

    async def invoke() -> None:
        await RunnableLambda(lambda x: x).ainvoke("hi", config=config)

    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as tg:
            tg.start_soon(invoke)
            await asyncio.wait_for(handler.reached.wait(), TIMEOUT)
            scope.cancel()

    await asyncio.wait_for(handler.closed.wait(), TIMEOUT)
    opened = [run_id for name, run_id in handler.events if name == "on_chain_start"]
    closed = [run_id for name, run_id in handler.events if name == "on_chain_error"]
    assert len(opened) == 1, handler.events
    assert closed == opened, handler.events
