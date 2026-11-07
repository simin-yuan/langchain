"""Interrupted start callbacks leave the run opened but never closed.

The start dispatch happens before the ``try`` block that closes the run, so an
interrupt raised from inside it (``KeyboardInterrupt`` is the one users hit)
leaves the run opened on the tracing side but never closed: no end, no error.
These tests pin that such a run is closed with an error callback carrying the
same ``run_id`` it was opened with, and that the interrupt still propagates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from typing_extensions import override

from langchain_core.callbacks import BaseCallbackHandler
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


class _Interrupt(BaseException):
    """Stands in for ``KeyboardInterrupt``: not an ``Exception``, not swallowed."""


class _InterruptingStartHandler(BaseCallbackHandler):
    """Raises from inside the start dispatch of one component."""

    def __init__(self, start_event: str, error_event: str) -> None:
        self.start_event = start_event
        self.error_event = error_event
        self.events: list[tuple[str, UUID | None]] = []

    def _record(self, name: str, run_id: UUID | None) -> None:
        self.events.append((name, run_id))
        if name == self.start_event:
            raise _Interrupt

    @override
    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_chain_start", kwargs.get("run_id"))

    @override
    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_tool_start", kwargs.get("run_id"))

    @override
    def on_retriever_start(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_retriever_start", kwargs.get("run_id"))

    @override
    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_llm_start", kwargs.get("run_id"))

    @override
    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_chat_model_start", kwargs.get("run_id"))

    @override
    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_chain_error", kwargs.get("run_id"))

    @override
    def on_tool_error(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_tool_error", kwargs.get("run_id"))

    @override
    def on_retriever_error(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_retriever_error", kwargs.get("run_id"))

    @override
    def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_llm_error", kwargs.get("run_id"))


class _StaticRetriever(BaseRetriever):
    """Retriever that never runs: the start dispatch is what is under test."""

    def _get_relevant_documents(self, query: str, **_kwargs: Any) -> list[Document]:
        return [Document(page_content=query)]


@tool_decorator
def _echo(text: str) -> str:
    """Echo the text back."""
    return text


def _invoke(component: str, config: RunnableConfig) -> None:
    if component == "chain":
        RunnableLambda(lambda x: x).invoke("hi", config=config)
    elif component == "tool":
        _echo.invoke({"text": "hi"}, config=config)
    elif component == "retriever":
        _StaticRetriever().invoke("hi", config=config)
    elif component == "llm":
        FakeListLLM(responses=["ok"]).invoke("hi", config=config)
    elif component == "chat_model":
        FakeListChatModel(responses=["ok"]).invoke([("user", "hi")], config=config)
    else:  # pragma: no cover
        msg = f"unknown component {component}"
        raise AssertionError(msg)


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
def test_interrupted_start_dispatch_still_closes_the_run(
    component: str, start_event: str, error_event: str
) -> None:
    handler = _InterruptingStartHandler(start_event, error_event)

    with pytest.raises(_Interrupt):
        _invoke(component, RunnableConfig(callbacks=[handler]))

    opened = [run_id for name, run_id in handler.events if name == start_event]
    closed = [run_id for name, run_id in handler.events if name == error_event]
    assert len(opened) == 1, handler.events
    assert closed == opened, handler.events


def test_interrupted_start_of_a_nested_root_run_is_closed() -> None:
    """A composite runnable's root run is closed too, not just leaf components."""
    handler = _InterruptingStartHandler("on_chain_start", "on_chain_error")
    chain: Runnable[Any, Any] = RunnableSequence(
        RunnableLambda(lambda x: x), RunnableLambda(lambda x: x)
    )

    with pytest.raises(_Interrupt):
        chain.invoke("hi", config=RunnableConfig(callbacks=[handler]))

    # Only the root run was opened, and it is the one that gets closed.
    assert [name for name, _ in handler.events] == [
        "on_chain_start",
        "on_chain_error",
    ], handler.events
    assert handler.events[0][1] == handler.events[1][1]
