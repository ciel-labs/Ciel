"""Tests for src/Ciel/llm.py — LLM client, models, and prompt builder.

All tests here are OFFLINE — no real API key or network is needed.
Tests that require a live API are marked @pytest.mark.live and are
excluded from normal CI runs:  pytest -m "not live"

What is tested
--------------
- TokenUsage and LLMResponse data models
- LLMError exception hierarchy
- _build_tool_schema(): correct OpenAI JSON schema output
- _parse_response(): text replies, tool calls, edge cases, malformed input
- build_system_prompt(): content and structure
- build_user_message(): correct dict shape
- MessageHistory: message accumulation in correct OpenAI format
- LLMClient: auth check, response size limit (via monkeypatched httpx)

What is NOT tested here
-----------------------
- Real network calls to the Qwen API (see @pytest.mark.live tests below)
- Policy engine integration (tested in test_policy.py)
- Full agent loop (tested in a later step)
"""

import json
import pytest

from Ciel.config import AppConfig
from Ciel.llm import (
    LLMAuthError,
    LLMClient,
    LLMError,
    LLMParseError,
    LLMRateLimitError,
    LLMResponse,
    LLMServerError,
    LLMTimeoutError,
    MessageHistory,
    TokenUsage,
    _build_tool_schema,
    _parse_response,
    build_system_prompt,
    build_user_message,
)
from Ciel.tools import ToolCall
from Ciel.tools.definitions import default_registry


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_text_response(
    content: str = "hello",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict:
    """Build a minimal valid OpenAI-format text response dict."""
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {
                "role": "assistant",
                "content": content,
            },
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_tool_call_response(
    tool_name: str = "hostname",
    arguments: dict | None = None,
    call_id: str = "call_abc123",
) -> dict:
    """Build a minimal valid OpenAI-format tool_call response dict."""
    args_str = json.dumps(arguments or {})
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": args_str,
                    },
                }],
            },
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenUsage:

    def test_fields_stored_correctly(self):
        u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 5
        assert u.total_tokens == 15

    def test_is_frozen(self):
        u = TokenUsage(10, 5, 15)
        with pytest.raises((AttributeError, TypeError)):
            u.prompt_tokens = 99  # type: ignore[misc]


class TestLLMResponse:

    def test_text_response(self):
        r = LLMResponse(
            reply_text="hello",
            tool_call=None,
            finish_reason="stop",
            usage=TokenUsage(10, 5, 15),
        )
        assert r.has_reply() is True
        assert r.has_tool_call() is False
        assert r.reply_text == "hello"

    def test_tool_call_response(self):
        tc = ToolCall(tool_name="hostname", arguments={})
        r = LLMResponse(
            reply_text=None,
            tool_call=tc,
            finish_reason="tool_calls",
            usage=TokenUsage(10, 5, 15),
        )
        assert r.has_tool_call() is True
        assert r.has_reply() is False
        assert r.tool_call.tool_name == "hostname"

    def test_is_frozen(self):
        r = LLMResponse(
            reply_text="hi",
            tool_call=None,
            finish_reason="stop",
            usage=TokenUsage(0, 0, 0),
        )
        with pytest.raises((AttributeError, TypeError)):
            r.reply_text = "changed"  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# Error hierarchy
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMErrorHierarchy:

    def test_all_errors_are_llm_error_subclasses(self):
        for cls in (LLMAuthError, LLMRateLimitError, LLMTimeoutError,
                    LLMServerError, LLMParseError):
            assert issubclass(cls, LLMError), f"{cls} must be a subclass of LLMError"

    def test_catching_llm_error_catches_auth_error(self):
        with pytest.raises(LLMError):
            raise LLMAuthError("bad key")

    def test_catching_llm_error_catches_parse_error(self):
        with pytest.raises(LLMError):
            raise LLMParseError("bad json")

    def test_specific_subclass_still_catchable_specifically(self):
        with pytest.raises(LLMAuthError):
            raise LLMAuthError("bad key")


# ══════════════════════════════════════════════════════════════════════════════
# _build_tool_schema
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildToolSchema:

    def setup_method(self):
        self.registry = default_registry()
        self.schema = _build_tool_schema(self.registry)

    def test_returns_a_list(self):
        assert isinstance(self.schema, list)

    def test_one_entry_per_tool(self):
        assert len(self.schema) == len(self.registry)

    def test_each_entry_has_type_function(self):
        for entry in self.schema:
            assert entry["type"] == "function"

    def test_each_entry_has_function_dict(self):
        for entry in self.schema:
            assert "function" in entry
            fn = entry["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_parameters_is_object_type(self):
        for entry in self.schema:
            params = entry["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_tool_names_match_registry(self):
        schema_names = {e["function"]["name"] for e in self.schema}
        registry_names = {t.name for t in self.registry.all_tools()}
        assert schema_names == registry_names

    def test_disk_usage_has_optional_path_property(self):
        """disk_usage has one optional 'path' parameter."""
        du_entry = next(
            e for e in self.schema if e["function"]["name"] == "disk_usage"
        )
        params = du_entry["function"]["parameters"]
        assert "path" in params["properties"]
        # optional means NOT in "required"
        assert "path" not in params.get("required", [])

    def test_no_required_params_for_no_arg_tools(self):
        """Tools with no parameters must not have a 'required' key."""
        no_arg_tools = {"hostname", "current_user", "free_memory",
                        "kernel_info", "system_uptime"}
        for entry in self.schema:
            if entry["function"]["name"] in no_arg_tools:
                assert "required" not in entry["function"]["parameters"]

    def test_property_values_have_type_and_description(self):
        for entry in self.schema:
            for prop in entry["function"]["parameters"]["properties"].values():
                assert "type" in prop
                assert "description" in prop


# ══════════════════════════════════════════════════════════════════════════════
# _parse_response
# ══════════════════════════════════════════════════════════════════════════════

class TestParseResponse:

    def test_text_reply_parsed_correctly(self):
        raw = _make_text_response("The disk is 50% full.")
        result = _parse_response(raw)
        assert result.reply_text == "The disk is 50% full."
        assert result.tool_call is None
        assert result.finish_reason == "stop"

    def test_tool_call_parsed_correctly(self):
        raw = _make_tool_call_response("disk_usage", {"path": "/home"})
        result = _parse_response(raw)
        assert result.tool_call is not None
        assert result.tool_call.tool_name == "disk_usage"
        assert result.tool_call.arguments == {"path": "/home"}
        assert result.reply_text is None
        assert result.finish_reason == "tool_calls"

    def test_tool_call_no_arguments(self):
        raw = _make_tool_call_response("hostname", {})
        result = _parse_response(raw)
        assert result.tool_call.arguments == {}

    def test_tool_call_integer_args_coerced_to_str(self):
        """Non-string argument values must be coerced to str."""
        raw = _make_tool_call_response("disk_usage", {"count": 5})
        result = _parse_response(raw)
        assert result.tool_call.arguments["count"] == "5"

    def test_usage_parsed(self):
        raw = _make_text_response(prompt_tokens=20, completion_tokens=8)
        result = _parse_response(raw)
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 8
        assert result.usage.total_tokens == 28

    def test_missing_usage_defaults_to_zero(self):
        raw = _make_text_response()
        del raw["usage"]
        result = _parse_response(raw)
        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0

    def test_finish_reason_preserved(self):
        raw = _make_text_response(finish_reason="length")
        result = _parse_response(raw)
        assert result.finish_reason == "length"

    def test_empty_choices_raises_parse_error(self):
        raw = {"choices": [], "usage": {}}
        with pytest.raises(LLMParseError, match="empty"):
            _parse_response(raw)

    def test_missing_choices_key_raises_parse_error(self):
        with pytest.raises(LLMParseError):
            _parse_response({"usage": {}})

    def test_invalid_tool_arguments_json_raises_parse_error(self):
        raw = _make_tool_call_response("hostname")
        # Corrupt the arguments string
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            "not valid json {"
        )
        with pytest.raises(LLMParseError, match="not valid JSON"):
            _parse_response(raw)

    def test_tool_arguments_not_object_raises_parse_error(self):
        raw = _make_tool_call_response("hostname")
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            '"just a string"'
        )
        with pytest.raises(LLMParseError, match="must be a JSON object"):
            _parse_response(raw)

    def test_empty_content_becomes_empty_string(self):
        raw = _make_text_response(content="")
        result = _parse_response(raw)
        assert result.reply_text == ""


# ══════════════════════════════════════════════════════════════════════════════
# build_system_prompt
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildSystemPrompt:

    def setup_method(self):
        self.registry = default_registry()
        self.prompt = build_system_prompt(self.registry)

    def test_prompt_is_a_string(self):
        assert isinstance(self.prompt, str)

    def test_prompt_is_non_empty(self):
        assert len(self.prompt) > 0

    def test_prompt_mentions_ciel(self):
        assert "Ciel" in self.prompt

    def test_prompt_contains_all_tool_names(self):
        for tool in self.registry.all_tools():
            assert tool.name in self.prompt, (
                f"Tool {tool.name!r} missing from system prompt"
            )

    def test_prompt_contains_all_tool_descriptions(self):
        for tool in self.registry.all_tools():
            # At least part of the description should appear
            assert tool.description[:30] in self.prompt

    def test_prompt_mentions_not_fabricating(self):
        """The prompt must tell the LLM not to guess system state."""
        prompt_lower = self.prompt.lower()
        assert "fabricate" in prompt_lower or "guess" in prompt_lower or "invent" in prompt_lower

    def test_prompt_has_disk_usage_path_parameter(self):
        """disk_usage's optional path parameter must appear in the prompt."""
        assert "path" in self.prompt


# ══════════════════════════════════════════════════════════════════════════════
# build_user_message
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildUserMessage:

    def test_returns_dict(self):
        msg = build_user_message("what is my disk usage?")
        assert isinstance(msg, dict)

    def test_role_is_user(self):
        msg = build_user_message("hello")
        assert msg["role"] == "user"

    def test_content_matches_task(self):
        task = "how much RAM do I have?"
        msg = build_user_message(task)
        assert msg["content"] == task


# ══════════════════════════════════════════════════════════════════════════════
# MessageHistory
# ══════════════════════════════════════════════════════════════════════════════

class TestMessageHistory:

    def test_starts_with_system_message(self):
        h = MessageHistory("you are an agent")
        messages = h.as_list()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "you are an agent"

    def test_add_user_appends_user_message(self):
        h = MessageHistory("sys")
        h.add_user("hello")
        msgs = h.as_list()
        assert len(msgs) == 2
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "hello"

    def test_add_assistant_text(self):
        h = MessageHistory("sys")
        h.add_assistant_text("I checked. Disk is 80% full.")
        msgs = h.as_list()
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-1]["content"] == "I checked. Disk is 80% full."

    def test_add_tool_call(self):
        h = MessageHistory("sys")
        h.add_tool_call("disk_usage", {"path": "/home"}, call_id="call_1")
        msgs = h.as_list()
        last = msgs[-1]
        assert last["role"] == "assistant"
        assert last["content"] is None
        assert last["tool_calls"][0]["id"] == "call_1"
        assert last["tool_calls"][0]["function"]["name"] == "disk_usage"

    def test_add_tool_result(self):
        h = MessageHistory("sys")
        h.add_tool_result("disk_usage", "Filesystem 80% used", call_id="call_1")
        msgs = h.as_list()
        last = msgs[-1]
        assert last["role"] == "tool"
        assert last["tool_call_id"] == "call_1"
        assert "80%" in last["content"]

    def test_full_conversation_sequence(self):
        """Simulate a complete one-round tool-use conversation."""
        h = MessageHistory("you are an agent")
        h.add_user("How full is my disk?")
        h.add_tool_call("disk_usage", {}, call_id="call_x")
        h.add_tool_result("disk_usage", "Filesystem 60% used", call_id="call_x")
        h.add_assistant_text("Your disk is 60% full.")

        msgs = h.as_list()
        assert len(msgs) == 5
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_as_list_returns_copy(self):
        """Mutating the returned list must not affect the history."""
        h = MessageHistory("sys")
        lst = h.as_list()
        lst.append({"role": "user", "content": "extra"})
        assert len(h.as_list()) == 1  # original unchanged

    def test_len_reflects_message_count(self):
        h = MessageHistory("sys")
        assert len(h) == 1
        h.add_user("hi")
        assert len(h) == 2


# ══════════════════════════════════════════════════════════════════════════════
# LLMClient (offline — no real API calls)
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMClientOffline:

    def test_raises_auth_error_when_no_key(self):
        """complete() must raise LLMAuthError immediately if no key is set."""
        config = AppConfig()  # no CIEL_API_KEY set
        with LLMClient(config) as client:
            with pytest.raises(LLMAuthError, match="CIEL_API_KEY"):
                client.complete([{"role": "user", "content": "hi"}])

    def test_context_manager_closes_cleanly(self):
        """The context manager must not raise on exit."""
        config = AppConfig()
        with LLMClient(config):
            pass  # just entering and exiting should not raise

    def test_response_size_limit_enforced(self, monkeypatch):
        """Responses over 1 MB must raise LLMParseError."""
        import httpx
        from Ciel.llm import _MAX_RESPONSE_BYTES

        monkeypatch.setenv("CIEL_API_KEY", "fake-key-for-test")
        config = AppConfig()

        # Build a fake response that is too large
        big_body = b"x" * (_MAX_RESPONSE_BYTES + 1)

        def fake_post(*args, **kwargs):
            resp = httpx.Response(200, content=big_body)
            return resp

        with LLMClient(config) as client:
            monkeypatch.setattr(client._client, "post", fake_post)
            with pytest.raises(LLMParseError, match="exceeds"):
                client.complete([{"role": "user", "content": "hi"}])

    def test_401_raises_auth_error(self, monkeypatch):
        """A 401 response must immediately raise LLMAuthError."""
        import httpx

        monkeypatch.setenv("CIEL_API_KEY", "fake-key")
        config = AppConfig()

        def fake_post(*args, **kwargs):
            return httpx.Response(401, content=b'{"error": "Unauthorized"}')

        with LLMClient(config) as client:
            monkeypatch.setattr(client._client, "post", fake_post)
            with pytest.raises(LLMAuthError, match="401"):
                client.complete([{"role": "user", "content": "hi"}])

    def test_successful_text_response(self, monkeypatch):
        """A 200 response with a text reply must return an LLMResponse."""
        import httpx

        monkeypatch.setenv("CIEL_API_KEY", "fake-key")
        config = AppConfig()

        raw = _make_text_response("The hostname is mybox.")
        body = json.dumps(raw).encode()

        def fake_post(*args, **kwargs):
            return httpx.Response(200, content=body)

        with LLMClient(config) as client:
            monkeypatch.setattr(client._client, "post", fake_post)
            result = client.complete([{"role": "user", "content": "hi"}])

        assert result.reply_text == "The hostname is mybox."
        assert result.finish_reason == "stop"

    def test_successful_tool_call_response(self, monkeypatch):
        """A 200 tool_calls response must return an LLMResponse with tool_call set."""
        import httpx

        monkeypatch.setenv("CIEL_API_KEY", "fake-key")
        config = AppConfig()

        raw = _make_tool_call_response("hostname", {})
        body = json.dumps(raw).encode()

        def fake_post(*args, **kwargs):
            return httpx.Response(200, content=body)

        with LLMClient(config) as client:
            monkeypatch.setattr(client._client, "post", fake_post)
            result = client.complete([{"role": "user", "content": "hi"}])

        assert result.has_tool_call()
        assert result.tool_call.tool_name == "hostname"


# ══════════════════════════════════════════════════════════════════════════════
# Live tests (require CIEL_API_KEY and network — skipped in normal CI)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.live
class TestLLMClientLive:
    """These tests make real API calls. Run with:  pytest -m live"""

    def test_simple_text_reply(self):
        config = AppConfig()
        if not config.has_api_key():
            pytest.skip("CIEL_API_KEY not set")

        with LLMClient(config) as client:
            result = client.complete([
                {"role": "user", "content": "Reply with exactly the word: PONG"}
            ])

        assert result.reply_text is not None
        assert "PONG" in result.reply_text.upper()
        assert result.usage.total_tokens > 0

    def test_tool_call_returned_for_tool_question(self):
        config = AppConfig()
        if not config.has_api_key():
            pytest.skip("CIEL_API_KEY not set")

        registry = default_registry()
        tools = _build_tool_schema(registry)
        prompt = build_system_prompt(registry)

        with LLMClient(config) as client:
            result = client.complete(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "What is the system hostname?"},
                ],
                tools=tools,
            )

        # The model should call the hostname tool
        assert result.has_tool_call() or result.has_reply()
        # We don't assert tool_call specifically because the model may choose
        # to answer directly — both are valid responses.
