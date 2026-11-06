"""Tests for `BaseRetriever` run lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class _Recorder(AsyncCallbackHandler):
    """Records the terminal callbacks of a retriever run."""

    def __init__(self) -> None:
        self.events: list[str] = []

    async def on_retriever_start(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("start")

    async def on_retriever_end(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("end")

    async def on_retriever_error(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("error")


class _SlowRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> list[Document]:
        return [Document(page_content=query)]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> list[Document]:
        await asyncio.sleep(30)
        return [Document(page_content=query)]


async def test_cancelled_retriever_call_emits_the_error_callback() -> None:
    """A cancelled retriever call must not leave its run open.

    `asyncio.CancelledError` derives from `BaseException`, so the run was never
    closed and stayed pending in tracing.
    """
    recorder = _Recorder()
    task = asyncio.create_task(
        _SlowRetriever().ainvoke("query", config={"callbacks": [recorder]})
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert recorder.events == ["start", "error"]
