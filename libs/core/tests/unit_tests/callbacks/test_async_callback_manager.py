"""Unit tests for verifying event dispatching.

Much of this code is indirectly tested already through many end-to-end tests
that generate traces based on the callbacks. The traces are all verified
via snapshot testing (e.g., see unit tests for runnables).
"""

import asyncio
import contextlib
import contextvars
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import pytest
from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackHandler,
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForLLMRun,
    AsyncCallbackManagerForRetrieverRun,
    AsyncCallbackManagerForToolRun,
    BaseCallbackHandler,
)
from langchain_core.outputs import LLMResult


async def test_inline_handlers_share_parent_context() -> None:
    """Verify that handlers that are configured to run_inline can update parent context.

    This test was created because some of the inline handlers were getting
    their own context as the handling logic was kicked off using asyncio.gather
    which does not automatically propagate the parent context (by design).

    This issue was affecting only a few specific handlers:

    * on_llm_start
    * on_chat_model_start

    which in some cases were triggered with multiple prompts and as a result
    triggering multiple tasks that were launched in parallel.
    """
    some_var: contextvars.ContextVar[str] = contextvars.ContextVar("some_var")

    class CustomHandler(AsyncCallbackHandler):
        """A handler that sets the context variable.

        The handler sets the context variable to the name of the callback that was
        called.
        """

        def __init__(self, *, run_inline: bool) -> None:
            """Initialize the handler."""
            self.run_inline = run_inline

        @override
        async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
            """Update the callstack with the name of the callback."""
            some_var.set("on_llm_start")

    # The manager serves as a callback dispatcher.
    # It's responsible for dispatching callbacks to all registered handlers.
    manager = AsyncCallbackManager(handlers=[CustomHandler(run_inline=True)])

    # Check on_llm_start
    some_var.set("unset")
    await manager.on_llm_start({}, ["prompt 1"])
    assert some_var.get() == "on_llm_start"

    # Check what happens when run_inline is False
    # We don't expect the context to be updated
    manager2 = AsyncCallbackManager(
        handlers=[
            CustomHandler(run_inline=False),
        ]
    )

    some_var.set("unset")
    await manager2.on_llm_start({}, ["prompt 1"])
    # Will not be updated because the handler is not inline
    assert some_var.get() == "unset"


async def test_inline_handlers_share_parent_context_multiple() -> None:
    """A slightly more complex variation of the test unit test above.

    This unit test verifies that things work correctly when there are multiple prompts,
    and multiple handlers that are configured to run inline.
    """
    counter_var = contextvars.ContextVar("counter", default=0)

    shared_stack = []

    @asynccontextmanager
    async def set_counter_var() -> Any:
        token = counter_var.set(0)
        try:
            yield
        finally:
            counter_var.reset(token)

    class StatefulAsyncCallbackHandler(AsyncCallbackHandler):
        def __init__(self, name: str, *, run_inline: bool = True):
            self.name = name
            self.run_inline = run_inline

        async def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            if self.name == "StateModifier":
                current_counter = counter_var.get()
                counter_var.set(current_counter + 1)
                state = counter_var.get()
            elif self.name == "StateReader":
                state = counter_var.get()
            else:
                state = None

            shared_stack.append(state)

            await super().on_llm_start(
                serialized,
                prompts,
                run_id=run_id,
                parent_run_id=parent_run_id,
                **kwargs,
            )

    handlers: list[BaseCallbackHandler] = [
        StatefulAsyncCallbackHandler("StateModifier", run_inline=True),
        StatefulAsyncCallbackHandler("StateReader", run_inline=True),
        StatefulAsyncCallbackHandler("NonInlineHandler", run_inline=False),
    ]

    prompts = ["Prompt1", "Prompt2", "Prompt3"]

    async with set_counter_var():
        shared_stack.clear()
        manager = AsyncCallbackManager(handlers=handlers)
        await manager.on_llm_start({}, prompts)

        # Assert the order of states
        states = [entry for entry in shared_stack if entry is not None]
        assert states == [
            1,
            1,
            2,
            2,
            3,
            3,
        ]


async def test_shielded_callback_context_preservation() -> None:
    """Verify that shielded callbacks preserve context variables.

    This test specifically addresses the issue where async callbacks decorated
    with @shielded do not properly preserve context variables, breaking
    instrumentation and other context-dependent functionality.

    The issue manifests in callbacks that use the @shielded decorator:
    * on_llm_end
    * on_llm_error
    * on_chain_end
    * on_chain_error
    * And other shielded callback methods
    """
    context_var: contextvars.ContextVar[str] = contextvars.ContextVar("test_context")

    class ContextTestHandler(AsyncCallbackHandler):
        """Handler that reads context variables in shielded callbacks."""

        def __init__(self) -> None:
            self.run_inline = False
            self.context_values: list[str] = []

        @override
        async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            """This method is decorated with @shielded in the run manager."""
            # This should preserve the context variable value
            self.context_values.append(context_var.get("not_found"))

        @override
        async def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
            """This method is decorated with @shielded in the run manager."""
            # This should preserve the context variable value
            self.context_values.append(context_var.get("not_found"))

    # Set up the test context
    context_var.set("test_value")
    handler = ContextTestHandler()
    manager = AsyncCallbackManager(handlers=[handler])

    # Create run managers that have the shielded methods
    llm_managers = await manager.on_llm_start({}, ["test prompt"])
    llm_run_manager = llm_managers[0]

    chain_run_manager = await manager.on_chain_start({}, {"test": "input"})

    # Test LLM end callback (which is shielded)
    await llm_run_manager.on_llm_end({"response": "test"})  # type: ignore[arg-type]

    # Test Chain end callback (which is shielded)
    await chain_run_manager.on_chain_end({"output": "test"})

    # The context should be preserved in shielded callbacks
    # This was the main issue - shielded decorators were not preserving context
    assert handler.context_values == ["test_value", "test_value"], (
        f"Expected context values ['test_value', 'test_value'], "
        f"but got {handler.context_values}. "
        f"This indicates the shielded decorator is not preserving context variables."
    )


class _SuspendingEndErrorRecorder(AsyncCallbackHandler):
    """Records every end/error callback, but only after yielding once.

    Inside a cancelled anyio scope an unshielded callback dies at its first
    suspension point, so yielding before recording makes a missing `@shielded`
    fail deterministically.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def _record(self, name: str) -> None:
        await asyncio.sleep(0)
        self.calls.append(name)

    @override
    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        await self._record("on_llm_end")

    @override
    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        await self._record("on_llm_error")

    @override
    async def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        await self._record("on_chain_end")

    @override
    async def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        await self._record("on_chain_error")

    @override
    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        await self._record("on_tool_end")

    @override
    async def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        await self._record("on_tool_error")

    @override
    async def on_retriever_end(self, documents: Any, **kwargs: Any) -> None:
        await self._record("on_retriever_end")

    @override
    async def on_retriever_error(self, error: BaseException, **kwargs: Any) -> None:
        await self._record("on_retriever_error")


_END_ERROR_CALLBACKS = [
    (AsyncCallbackManagerForLLMRun, "on_llm_end", LLMResult(generations=[])),
    (AsyncCallbackManagerForLLMRun, "on_llm_error", ValueError("boom")),
    (AsyncCallbackManagerForChainRun, "on_chain_end", {"output": "x"}),
    (AsyncCallbackManagerForChainRun, "on_chain_error", ValueError("boom")),
    (AsyncCallbackManagerForToolRun, "on_tool_end", "x"),
    (AsyncCallbackManagerForToolRun, "on_tool_error", ValueError("boom")),
    (AsyncCallbackManagerForRetrieverRun, "on_retriever_end", []),
    (AsyncCallbackManagerForRetrieverRun, "on_retriever_error", ValueError("boom")),
]


@pytest.mark.parametrize(
    ("manager_cls", "method", "arg"),
    _END_ERROR_CALLBACKS,
    ids=[method for _, method, _ in _END_ERROR_CALLBACKS],
)
async def test_end_and_error_callbacks_survive_cancelled_anyio_scope(
    manager_cls: type[Any], method: str, arg: Any
) -> None:
    """Every async run manager must close its run even inside a cancelled scope.

    Starlette and FastAPI cancel request handlers through an anyio cancel scope
    when the client disconnects. If an end/error callback is not shielded, the run
    it should close is left open with no end or error event.
    """
    anyio = pytest.importorskip("anyio")
    recorder = _SuspendingEndErrorRecorder()
    manager = manager_cls(
        run_id=uuid4(), handlers=[recorder], inheritable_handlers=[recorder]
    )

    with anyio.CancelScope() as scope:
        scope.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await getattr(manager, method)(arg)

    # A shielded callback finishes in its own task after the scope has exited.
    for _ in range(100):
        if recorder.calls:
            break
        await asyncio.sleep(0)

    assert recorder.calls == [method]
