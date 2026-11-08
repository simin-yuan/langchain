"""Tests for the retriever run lifecycle, including cancellation.

A retriever run must be closed (``on_retriever_error`` or ``on_retriever_end``)
for every call, including calls that end in ``asyncio.CancelledError`` or
``KeyboardInterrupt``. Both are ``BaseException`` subclasses, so a bare
``except Exception`` leaves the run open and the trace pending.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class _Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.errors: list[BaseException] = []

    def on_retriever_start(self, *_args: Any, **_kwargs: Any) -> None:
        self.events.append("start")

    def on_retriever_end(self, *_args: Any, **_kwargs: Any) -> None:
        self.events.append("end")

    def on_retriever_error(
        self, error: BaseException, *_args: Any, **_kwargs: Any
    ) -> None:
        self.events.append("error")
        self.errors.append(error)


class _AsyncRecorder(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.errors: list[BaseException] = []

    async def on_retriever_start(self, *_args: Any, **_kwargs: Any) -> None:
        self.events.append("start")

    async def on_retriever_end(self, *_args: Any, **_kwargs: Any) -> None:
        self.events.append("end")

    async def on_retriever_error(
        self, error: BaseException, *_args: Any, **_kwargs: Any
    ) -> None:
        self.events.append("error")
        self.errors.append(error)


class _SlowRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, **_kwargs: Any) -> list[Document]:
        return [Document(page_content=query)]

    async def _aget_relevant_documents(
        self, query: str, **_kwargs: Any
    ) -> list[Document]:
        await asyncio.sleep(10)
        return [Document(page_content=query)]


class _RaisingRetriever(BaseRetriever):
    error: Any = None

    def _get_relevant_documents(self, query: str, **_kwargs: Any) -> list[Document]:
        del query  # unused by design: this stub only raises
        raise self.error

    async def _aget_relevant_documents(
        self, query: str, **_kwargs: Any
    ) -> list[Document]:
        del query  # unused by design: this stub only raises
        raise self.error


async def test_cancelled_call_closes_the_run() -> None:
    started = asyncio.Event()

    class _BlockingRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, **_kwargs: Any) -> list[Document]:
            return [Document(page_content=query)]

        async def _aget_relevant_documents(
            self, query: str, **_kwargs: Any
        ) -> list[Document]:
            started.set()
            await asyncio.sleep(10)
            return [Document(page_content=query)]

    recorder = _AsyncRecorder()
    task = asyncio.create_task(
        _BlockingRetriever().ainvoke("q", config={"callbacks": [recorder]})
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert recorder.events == ["start", "error"]
    assert isinstance(recorder.errors[0], asyncio.CancelledError)


def test_keyboard_interrupt_closes_the_run() -> None:
    recorder = _Recorder()
    with pytest.raises(KeyboardInterrupt):
        _RaisingRetriever(error=KeyboardInterrupt()).invoke(
            "q", config={"callbacks": [recorder]}
        )
    assert recorder.events == ["start", "error"]
    assert isinstance(recorder.errors[0], KeyboardInterrupt)


def test_exception_is_still_reported() -> None:
    recorder = _Recorder()
    with pytest.raises(ValueError, match="boom"):
        _RaisingRetriever(error=ValueError("boom")).invoke(
            "q", config={"callbacks": [recorder]}
        )
    assert recorder.events == ["start", "error"]
    assert isinstance(recorder.errors[0], ValueError)


def test_successful_call_reports_end() -> None:
    recorder = _Recorder()
    docs = _SlowRetriever().invoke("q", config={"callbacks": [recorder]})
    assert len(docs) == 1
    assert recorder.events == ["start", "end"]


class _SuspendingAsyncRecorder(AsyncCallbackHandler):
    """Records end/error only after yielding once.

    Inside a cancelled anyio scope an unshielded callback dies at its first
    suspension point, so recording after a suspension makes a missing shield
    fail every time instead of depending on timing.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self._recorded = asyncio.Event()

    async def on_retriever_end(self, documents: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        self.events.append(("end", documents))
        self._recorded.set()

    async def on_retriever_error(self, error: BaseException, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        self.events.append(("error", error))
        self._recorded.set()

    async def settle(self) -> None:
        """Give a shielded callback, which outlives the cancelled scope, time to run."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._recorded.wait(), timeout=1)


async def test_anyio_scope_cancellation_calls_on_retriever_error() -> None:
    """A retrieval cancelled by an anyio cancel scope still closes its run.

    Starlette and FastAPI cancel request handlers this way on client disconnect.
    """
    anyio = pytest.importorskip("anyio")
    recorder = _SuspendingAsyncRecorder()

    class _CancellingRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, **_kwargs: Any
        ) -> list[Document]:
            return [Document(page_content=query)]

        async def _aget_relevant_documents(
            self, query: str, **_kwargs: Any
        ) -> list[Document]:
            scope.cancel()
            await asyncio.Event().wait()
            return [Document(page_content=query)]

    with anyio.CancelScope() as scope:
        await _CancellingRetriever().ainvoke("q", config={"callbacks": [recorder]})

    await recorder.settle()
    assert scope.cancelled_caught
    assert [name for name, _ in recorder.events] == ["error"]
    assert isinstance(recorder.events[0][1], asyncio.CancelledError)
