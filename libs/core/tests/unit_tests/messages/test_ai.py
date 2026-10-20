from collections.abc import Iterator
from typing import Any, cast

from typing_extensions import override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages import content as types
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
    add_ai_message_chunks,
    add_usage,
    subtract_usage,
)
from langchain_core.messages.tool import invalid_tool_call as create_invalid_tool_call
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.messages.utils import message_chunk_to_message
from langchain_core.outputs import ChatGenerationChunk, ChatResult


def test_serdes_message() -> None:
    msg = AIMessage(
        content=[{"text": "blah", "type": "text"}],
        tool_calls=[create_tool_call(name="foo", args={"bar": 1}, id="baz")],
        invalid_tool_calls=[
            create_invalid_tool_call(name="foobad", args="blah", id="booz", error="bad")
        ],
    )
    expected = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "type": "ai",
            "content": [{"text": "blah", "type": "text"}],
            "tool_calls": [
                {"name": "foo", "args": {"bar": 1}, "id": "baz", "type": "tool_call"}
            ],
            "invalid_tool_calls": [
                {
                    "name": "foobad",
                    "args": "blah",
                    "id": "booz",
                    "error": "bad",
                    "type": "invalid_tool_call",
                }
            ],
        },
    }
    actual = dumpd(msg)
    assert actual == expected
    assert load(actual, allowed_objects=[AIMessage]) == msg


def test_serdes_message_chunk() -> None:
    chunk = AIMessageChunk(
        content=[{"text": "blah", "type": "text"}],
        tool_call_chunks=[
            create_tool_call_chunk(name="foo", args='{"bar": 1}', id="baz", index=0),
            create_tool_call_chunk(
                name="foobad",
                args="blah",
                id="booz",
                index=1,
            ),
        ],
    )
    expected = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessageChunk"],
        "kwargs": {
            "type": "AIMessageChunk",
            "content": [{"text": "blah", "type": "text"}],
            "tool_calls": [
                {"name": "foo", "args": {"bar": 1}, "id": "baz", "type": "tool_call"}
            ],
            "invalid_tool_calls": [
                {
                    "name": "foobad",
                    "args": "blah",
                    "id": "booz",
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            "tool_call_chunks": [
                {
                    "name": "foo",
                    "args": '{"bar": 1}',
                    "id": "baz",
                    "index": 0,
                    "type": "tool_call_chunk",
                },
                {
                    "name": "foobad",
                    "args": "blah",
                    "id": "booz",
                    "index": 1,
                    "type": "tool_call_chunk",
                },
            ],
        },
    }
    actual = dumpd(chunk)
    assert actual == expected
    assert load(actual, allowed_objects=[AIMessageChunk]) == chunk


def test_add_usage_both_none() -> None:
    result = add_usage(None, None)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_add_usage_one_none() -> None:
    usage = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    result = add_usage(usage, None)
    assert result == usage


def test_add_usage_both_present() -> None:
    usage1 = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
    result = add_usage(usage1, usage2)
    assert result == UsageMetadata(input_tokens=15, output_tokens=30, total_tokens=45)


def test_add_usage_with_details() -> None:
    usage1 = UsageMetadata(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_token_details=InputTokenDetails(audio=5),
        output_token_details=OutputTokenDetails(reasoning=10),
    )
    usage2 = UsageMetadata(
        input_tokens=5,
        output_tokens=10,
        total_tokens=15,
        input_token_details=InputTokenDetails(audio=3),
        output_token_details=OutputTokenDetails(reasoning=5),
    )
    result = add_usage(usage1, usage2)
    assert result["input_token_details"]["audio"] == 8
    assert result["output_token_details"]["reasoning"] == 15


def test_subtract_usage_both_none() -> None:
    result = subtract_usage(None, None)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_subtract_usage_one_none() -> None:
    usage = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    result = subtract_usage(usage, None)
    assert result == usage


def test_subtract_usage_both_present() -> None:
    usage1 = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
    result = subtract_usage(usage1, usage2)
    assert result == UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)


def test_subtract_usage_with_negative_result() -> None:
    usage1 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
    usage2 = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    result = subtract_usage(usage1, usage2)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_add_ai_message_chunks_usage() -> None:
    chunks = [
        AIMessageChunk(content="", usage_metadata=None),
        AIMessageChunk(
            content="",
            usage_metadata=UsageMetadata(
                input_tokens=2, output_tokens=3, total_tokens=5
            ),
        ),
        AIMessageChunk(
            content="",
            usage_metadata=UsageMetadata(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
                input_token_details=InputTokenDetails(audio=1, cache_read=1),
                output_token_details=OutputTokenDetails(audio=1, reasoning=2),
            ),
        ),
    ]
    combined = add_ai_message_chunks(*chunks)
    assert combined == AIMessageChunk(
        content="",
        usage_metadata=UsageMetadata(
            input_tokens=4,
            output_tokens=6,
            total_tokens=10,
            input_token_details=InputTokenDetails(audio=1, cache_read=1),
            output_token_details=OutputTokenDetails(audio=1, reasoning=2),
        ),
    )


def test_init_tool_calls() -> None:
    # Test we add "type" key on init
    msg = AIMessage("", tool_calls=[{"name": "foo", "args": {"a": "b"}, "id": "abc"}])
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["type"] == "tool_call"

    # Test we can assign without adding type key
    msg.tool_calls = [{"name": "bar", "args": {"c": "d"}, "id": "def"}]


def test_content_blocks() -> None:
    message = AIMessage(
        "",
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"}
        ],
    )
    assert len(message.content_blocks) == 1
    assert message.content_blocks[0]["type"] == "tool_call"
    assert message.content_blocks == [
        {"type": "tool_call", "id": "abc_123", "name": "foo", "args": {"a": "b"}}
    ]
    assert message.content == ""

    message = AIMessage(
        "foo",
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"}
        ],
    )
    assert len(message.content_blocks) == 2
    assert message.content_blocks[0]["type"] == "text"
    assert message.content_blocks[1]["type"] == "tool_call"
    assert message.content_blocks == [
        {"type": "text", "text": "foo"},
        {"type": "tool_call", "id": "abc_123", "name": "foo", "args": {"a": "b"}},
    ]
    assert message.content == "foo"

    # With standard blocks
    standard_content: list[types.ContentBlock] = [
        {"type": "reasoning", "reasoning": "foo"},
        {"type": "text", "text": "bar"},
        {
            "type": "text",
            "text": "baz",
            "annotations": [{"type": "citation", "url": "http://example.com"}],
        },
        {
            "type": "image",
            "url": "http://example.com/image.png",
            "extras": {"foo": "bar"},
        },
        {
            "type": "non_standard",
            "value": {"custom_key": "custom_value", "another_key": 123},
        },
        {
            "type": "tool_call",
            "name": "foo",
            "args": {"a": "b"},
            "id": "abc_123",
        },
    ]
    missing_tool_call: types.ToolCall = {
        "type": "tool_call",
        "name": "bar",
        "args": {"c": "d"},
        "id": "abc_234",
    }
    message = AIMessage(
        content_blocks=standard_content,
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
            missing_tool_call,
        ],
    )
    assert message.content_blocks == [*standard_content, missing_tool_call]

    # Check we auto-populate tool_calls
    standard_content = [
        {"type": "text", "text": "foo"},
        {
            "type": "tool_call",
            "name": "foo",
            "args": {"a": "b"},
            "id": "abc_123",
        },
        missing_tool_call,
    ]
    message = AIMessage(content_blocks=standard_content)
    assert message.tool_calls == [
        {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
        missing_tool_call,
    ]

    # Chunks
    message = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "foo",
                "args": "",
                "id": "abc_123",
                "index": 0,
            }
        ],
    )
    assert len(message.content_blocks) == 1
    assert message.content_blocks[0]["type"] == "tool_call_chunk"
    assert message.content_blocks == [
        {
            "type": "tool_call_chunk",
            "name": "foo",
            "args": "",
            "id": "abc_123",
            "index": 0,
        }
    ]
    assert message.content == ""

    # Test we parse tool call chunks into tool calls for v1 content
    chunk_1 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "foo",
                "args": '{"foo": "b',
                "id": "abc_123",
                "index": 0,
            }
        ],
    )

    chunk_2 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",
                "args": 'ar"}',
                "id": "abc_123",
                "index": 0,
            }
        ],
    )
    chunk_3 = AIMessageChunk(content="", chunk_position="last")
    chunk = chunk_1 + chunk_2 + chunk_3
    assert chunk.content == ""
    assert chunk.content_blocks == chunk.tool_calls

    # test v1 content
    chunk_1.content = cast("list[str | dict[str, Any]]", chunk_1.content_blocks)
    assert len(chunk_1.content) == 1
    chunk_1.content[0]["extras"] = {"baz": "qux"}  # type: ignore[index]
    chunk_1.response_metadata["output_version"] = "v1"
    chunk_2.content = cast("list[str | dict[str, Any]]", chunk_2.content_blocks)

    chunk = chunk_1 + chunk_2 + chunk_3
    assert chunk.content == [
        {
            "type": "tool_call",
            "name": "foo",
            "args": {"foo": "bar"},
            "id": "abc_123",
            "extras": {"baz": "qux"},
        }
    ]

    # Non-standard
    standard_content_1: list[types.ContentBlock] = [
        {"type": "non_standard", "index": 0, "value": {"foo": "bar "}}
    ]
    standard_content_2: list[types.ContentBlock] = [
        {"type": "non_standard", "index": 0, "value": {"foo": "baz"}}
    ]
    chunk_1 = AIMessageChunk(
        content=cast("list[str | dict[str, Any]]", standard_content_1)
    )
    chunk_2 = AIMessageChunk(
        content=cast("list[str | dict[str, Any]]", standard_content_2)
    )
    merged_chunk = chunk_1 + chunk_2
    assert merged_chunk.content == [
        {"type": "non_standard", "index": 0, "value": {"foo": "bar baz"}},
    ]

    # Test server_tool_call_chunks
    chunk_1 = AIMessageChunk(
        content=[
            {
                "type": "server_tool_call_chunk",
                "index": 0,
                "name": "foo",
            }
        ]
    )
    chunk_2 = AIMessageChunk(
        content=[{"type": "server_tool_call_chunk", "index": 0, "args": '{"a'}]
    )
    chunk_3 = AIMessageChunk(
        content=[{"type": "server_tool_call_chunk", "index": 0, "args": '": 1}'}]
    )
    merged_chunk = chunk_1 + chunk_2 + chunk_3
    assert merged_chunk.content == [
        {
            "type": "server_tool_call_chunk",
            "name": "foo",
            "index": 0,
            "args": '{"a": 1}',
        }
    ]

    full_chunk = merged_chunk + AIMessageChunk(
        content=[], chunk_position="last", response_metadata={"output_version": "v1"}
    )
    assert full_chunk.content == [
        {"type": "server_tool_call", "name": "foo", "index": 0, "args": {"a": 1}}
    ]

    # Server tool calls with no input stream no args
    empty_args_chunk = AIMessageChunk(
        content=[
            {"type": "server_tool_call_chunk", "index": 0, "name": "foo", "args": ""}
        ]
    ) + AIMessageChunk(
        content=[], chunk_position="last", response_metadata={"output_version": "v1"}
    )
    assert empty_args_chunk.content == [
        {"type": "server_tool_call", "name": "foo", "index": 0, "args": {}}
    ]

    # Test non-standard + non-standard
    chunk_1 = AIMessageChunk(
        content=[
            {
                "type": "non_standard",
                "index": 0,
                "value": {"type": "non_standard_tool", "foo": "bar"},
            }
        ]
    )
    chunk_2 = AIMessageChunk(
        content=[
            {
                "type": "non_standard",
                "index": 0,
                "value": {"type": "input_json_delta", "partial_json": "a"},
            }
        ]
    )
    chunk_3 = AIMessageChunk(
        content=[
            {
                "type": "non_standard",
                "index": 0,
                "value": {"type": "input_json_delta", "partial_json": "b"},
            }
        ]
    )
    merged_chunk = chunk_1 + chunk_2 + chunk_3
    assert merged_chunk.content == [
        {
            "type": "non_standard",
            "index": 0,
            "value": {"type": "non_standard_tool", "foo": "bar", "partial_json": "ab"},
        }
    ]

    # Test standard + non-standard with same index
    standard_content_1 = [
        {
            "type": "server_tool_call",
            "name": "web_search",
            "id": "ws_123",
            "args": {"query": "web query"},
            "index": 0,
        }
    ]
    standard_content_2 = [{"type": "non_standard", "value": {"foo": "bar"}, "index": 0}]
    chunk_1 = AIMessageChunk(
        content=cast("list[str | dict[str, Any]]", standard_content_1)
    )
    chunk_2 = AIMessageChunk(
        content=cast("list[str | dict[str, Any]]", standard_content_2)
    )
    merged_chunk = chunk_1 + chunk_2
    assert merged_chunk.content == [
        {
            "type": "server_tool_call",
            "name": "web_search",
            "id": "ws_123",
            "args": {"query": "web query"},
            "index": 0,
            "extras": {"foo": "bar"},
        }
    ]


def test_content_blocks_v1_string_content_falls_through() -> None:
    """Test that content_blocks falls through to translator when content is a string.

    When output_version="v1" is set but content is a string (as in Chat
    Completions streaming), content_blocks must not short-circuit. It should
    fall through to the model_provider translator so tool calls are included.
    Covers both `AIMessage` and `AIMessageChunk`.
    """
    # AIMessage with string content + tool_calls + v1 metadata
    msg = AIMessage(
        content="Hello",
        tool_calls=[
            create_tool_call(name="foo", args={"a": 1}, id="tc_1"),
        ],
        response_metadata={
            "output_version": "v1",
            "model_provider": "openai",
        },
    )
    blocks = msg.content_blocks
    assert isinstance(blocks, list)
    # Should contain a text block and a tool_call block, not the raw string
    types_found = {b["type"] for b in blocks}
    assert "text" in types_found
    assert "tool_call" in types_found

    # AIMessageChunk with string content + tool_call_chunks + v1 metadata
    chunk = AIMessageChunk(
        content="Hello",
        tool_call_chunks=[
            create_tool_call_chunk(name="foo", args='{"a": 1}', id="tc_1", index=0),
        ],
        response_metadata={
            "output_version": "v1",
            "model_provider": "openai",
        },
    )
    blocks = chunk.content_blocks
    assert isinstance(blocks, list)
    types_found = {b["type"] for b in blocks}
    assert "text" in types_found
    assert "tool_call_chunk" in types_found


def test_content_blocks_v1_list_content_short_circuits() -> None:
    """Test that content_blocks short-circuits when v1 content is already a list.

    The `isinstance(self.content, list)` guard must preserve the fast path:
    when content is already a list of `ContentBlock` dicts, `content_blocks`
    returns it verbatim (the same object) without routing through the
    translator. Covers both `AIMessage` and `AIMessageChunk`.
    """
    content: list[str | dict[str, Any]] = [
        {"type": "text", "text": "Hello"},
        {"type": "tool_call", "name": "foo", "args": {"a": 1}, "id": "tc_1"},
    ]
    for cls in (AIMessage, AIMessageChunk):
        msg = cls(
            content=content,
            response_metadata={"output_version": "v1", "model_provider": "openai"},
        )
        # Short-circuit returns the stored content object itself; the translator
        # would build and return a new list instead.
        assert msg.content_blocks is msg.content


def test_content_blocks_reasoning_extraction() -> None:
    """Test best-effort reasoning extraction from `additional_kwargs`."""
    message = AIMessage(
        content="The answer is 42.",
        additional_kwargs={"reasoning_content": "Let me think about this problem..."},
    )
    content_blocks = message.content_blocks
    assert len(content_blocks) == 2
    assert content_blocks[0]["type"] == "reasoning"
    assert content_blocks[0].get("reasoning") == "Let me think about this problem..."
    assert content_blocks[1]["type"] == "text"
    assert content_blocks[1]["text"] == "The answer is 42."

    # Test no reasoning extraction when no reasoning content
    message = AIMessage(
        content="The answer is 42.", additional_kwargs={"other_field": "some value"}
    )
    content_blocks = message.content_blocks
    assert len(content_blocks) == 1
    assert content_blocks[0]["type"] == "text"


_TRUNCATED_ARGS = '{"path": "report.txt", "content": "line 01\\nline 02\\nline 0'


def test_tool_call_chunks_parse_partial_args_mid_stream() -> None:
    # Mid-stream, partial arguments are still parsed for progressive display.
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            create_tool_call_chunk(name="write_file", args=_TRUNCATED_ARGS, id="1")
        ],
    )
    assert chunk.invalid_tool_calls == []
    assert chunk.tool_calls[0]["args"]["path"] == "report.txt"


def test_truncated_tool_call_args_are_invalid_after_stream_ends() -> None:
    # A stream that ends while arguments are still incomplete (output limit
    # reached or connection dropped) must not produce an executable tool call.
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[
            create_tool_call_chunk(
                name="write_file", args=_TRUNCATED_ARGS, id="1", index=0
            )
        ],
    )
    last = AIMessageChunk(content="", chunk_position="last")
    final = first + last
    assert final.tool_calls == []
    assert len(final.invalid_tool_calls) == 1
    invalid = final.invalid_tool_calls[0]
    assert invalid["name"] == "write_file"
    assert invalid["args"] == _TRUNCATED_ARGS
    assert invalid["error"] is not None
    assert "not valid JSON" in invalid["error"]

    message = message_chunk_to_message(final)
    assert isinstance(message, AIMessage)
    assert message.tool_calls == []
    assert message.invalid_tool_calls == final.invalid_tool_calls


def test_complete_tool_call_args_after_stream_ends() -> None:
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[
            create_tool_call_chunk(name="a", args='{"x": 1}', id="1", index=0),
            create_tool_call_chunk(name="b", args="", id="2", index=1),
        ],
    )
    final = first + AIMessageChunk(content="", chunk_position="last")
    assert final.invalid_tool_calls == []
    assert [(tc["name"], tc["args"]) for tc in final.tool_calls] == [
        ("a", {"x": 1}),
        ("b", {}),
    ]


def test_only_the_truncated_call_is_invalid_after_stream_ends() -> None:
    # Parallel calls where only one is cut off: the complete call must stay
    # executable, the cut one must not.
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[
            create_tool_call_chunk(name="a", args='{"x": 1}', id="1", index=0),
            create_tool_call_chunk(name="b", args='{"y": "line 0', id="2", index=1),
        ],
    )
    final = first + AIMessageChunk(content="", chunk_position="last")
    assert [tc["name"] for tc in final.tool_calls] == ["a"]
    assert len(final.invalid_tool_calls) == 1
    assert final.invalid_tool_calls[0]["name"] == "b"


def test_raw_control_characters_after_stream_ends() -> None:
    # Matches the non-streaming parser, which accepts raw control characters
    # inside JSON strings (json.loads(..., strict=False)).
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[
            # "\n" below is a raw newline character inside the JSON string.
            create_tool_call_chunk(
                name="a", args='{"text": "one\ntwo"}', id="1", index=0
            )
        ],
    )
    final = first + AIMessageChunk(content="", chunk_position="last")
    assert final.invalid_tool_calls == []
    assert final.tool_calls[0]["args"] == {"text": "one\ntwo"}


class _TruncatingChatModel(BaseChatModel):
    """Streams a tool call whose arguments are cut off before the stream ends."""

    streaming: bool = False

    @property
    @override
    def _llm_type(self) -> str:
        return "truncating"

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for i in range(0, len(_TRUNCATED_ARGS), 8):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        create_tool_call_chunk(
                            name="write_file" if i == 0 else None,
                            args=_TRUNCATED_ARGS[i : i + 8],
                            id="call_1" if i == 0 else None,
                            index=0,
                        )
                    ],
                )
            )


def test_streamed_invoke_does_not_return_truncated_tool_call() -> None:
    message = _TruncatingChatModel(streaming=True).invoke("write the report")
    assert isinstance(message, AIMessage)
    assert message.tool_calls == []
    assert len(message.invalid_tool_calls) == 1
    assert message.invalid_tool_calls[0]["args"] == _TRUNCATED_ARGS


async def test_streamed_ainvoke_does_not_return_truncated_tool_call() -> None:
    message = await _TruncatingChatModel(streaming=True).ainvoke("write the report")
    assert isinstance(message, AIMessage)
    assert message.tool_calls == []
    assert len(message.invalid_tool_calls) == 1
