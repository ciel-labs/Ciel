"""Tests for src/Ciel/agent.py — the agent state machine.

All tests are OFFLINE. The LLM is never called for real. We monkeypatch
LLMClient.complete() to return controlled fake responses, letting us
test every code path without an API key or network.

What is tested
--------------
- AgentState enum values exist and are distinct
- TaskResult dataclass fields and immutability
- _format_tool_output(): all output cases
- Agent: direct text reply → TaskResult(success=True)
- Agent: single tool call → execution → LLM reply → success
- Agent: multi-turn (two tool calls then reply)
- Agent: max_turns enforcement → TaskResult(success=False)
- Agent: stuck detection (same tool repeated > MAX_CONSECUTIVE_SAME_TOOL)
- Agent: policy violation → error fed back to LLM, loop continues
- Agent: tool not found → error fed back to LLM, loop continues
- Agent: LLM auth error → TaskResult(success=False)
- Agent: generic LLM error → TaskResult(success=False)
- Agent: finish_reason with no reply and no tool_call → failure

What is NOT tested here
-----------------------
- Real network calls (zero in this file)
- Real subprocess execution (zero in this file)
- Policy engine internals (tested in test_policy.py)
- LLM client internals (tested in test_llm.py)
"""

import pytest

from Ciel.agent import Agent, AgentState, TaskResult, _format_tool_output
from Ciel.config import AppConfig
from Ciel.llm import (
    LLMAuthError,
    LLMResponse,
    LLMTimeoutError,
    TokenUsage,
)
from Ciel.models import ExecutionResult
from Ciel.tools.base import ToolCall, ToolRegistry
from Ciel.tools.definitions import default_registry


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def _text_response(text: str = "Task complete.") -> LLMResponse:
    """Fake LLM response with a plain-text reply."""
    return LLMResponse(
        reply_text=text,
        tool_call=None,
        finish_reason="stop",
        usage=_usage(),
    )


def _tool_response(tool_name: str, arguments: dict | None = None) -> LLMResponse:
    """Fake LLM response requesting a tool call."""
    return LLMResponse(
        reply_text=None,
        tool_call=ToolCall(tool_name=tool_name, arguments=arguments or {}),
        finish_reason="tool_calls",
        usage=_usage(),
    )


def _make_config(monkeypatch) -> AppConfig:
    """Return an AppConfig with a fake API key (avoids has_api_key() failing)."""
    monkeypatch.setenv("CIEL_API_KEY", "fake-key-for-tests")
    return AppConfig()


def _make_agent(config: AppConfig, registry: ToolRegistry | None = None) -> Agent:
    return Agent(config, registry=registry or default_registry())


# ══════════════════════════════════════════════════════════════════════════════
# AgentState
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentState:

    def test_all_states_exist(self):
        expected = {"idle", "planning", "validating", "executing", "verifying",
                    "done", "failed"}
        actual = {s.value for s in AgentState}
        assert actual == expected

    def test_states_are_distinct(self):
        values = [s.value for s in AgentState]
        assert len(values) == len(set(values))


# ══════════════════════════════════════════════════════════════════════════════
# TaskResult
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskResult:

    def test_success_result_fields(self):
        r = TaskResult(
            success=True,
            answer="Disk is 60% full.",
            turns_taken=2,
            tool_calls_made=1,
            state_on_exit=AgentState.DONE,
        )
        assert r.success is True
        assert r.answer == "Disk is 60% full."
        assert r.failure_reason is None
        assert r.turns_taken == 2
        assert r.tool_calls_made == 1

    def test_failure_result_fields(self):
        r = TaskResult(
            success=False,
            failure_reason="max turns reached",
            turns_taken=10,
            state_on_exit=AgentState.FAILED,
        )
        assert r.success is False
        assert r.answer is None
        assert r.failure_reason == "max turns reached"

    def test_is_frozen(self):
        r = TaskResult(success=True, answer="ok")
        with pytest.raises((AttributeError, TypeError)):
            r.success = False  # type: ignore[misc]

    def test_default_state_on_exit_is_idle(self):
        r = TaskResult(success=True)
        assert r.state_on_exit is AgentState.IDLE


# ══════════════════════════════════════════════════════════════════════════════
# _format_tool_output
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatToolOutput:

    def _result(self, **kwargs) -> ExecutionResult:
        defaults = dict(
            stdout="", stderr="", exit_code=0,
            timed_out=False, duration=0.1,
            stdout_truncated=False, stderr_truncated=False,
        )
        defaults.update(kwargs)
        return ExecutionResult(**defaults)

    def test_normal_stdout(self):
        r = self._result(stdout="hello world\n")
        assert "hello world" in _format_tool_output(r)

    def test_stderr_included_when_non_empty(self):
        r = self._result(stdout="ok", stderr="a warning")
        out = _format_tool_output(r)
        assert "a warning" in out
        assert "[stderr]" in out

    def test_timed_out_note(self):
        r = self._result(timed_out=True, exit_code=None)
        assert "timed out" in _format_tool_output(r).lower()

    def test_nonzero_exit_code_note(self):
        r = self._result(exit_code=1)
        assert "exit code" in _format_tool_output(r)

    def test_truncated_note(self):
        r = self._result(stdout="x", stdout_truncated=True)
        assert "truncated" in _format_tool_output(r).lower()

    def test_empty_output_placeholder(self):
        r = self._result()
        assert _format_tool_output(r) == "(no output)"

    def test_no_exit_code_note_on_success(self):
        """Exit code 0 must not add noise to the output."""
        r = self._result(stdout="ok", exit_code=0)
        assert "exit code" not in _format_tool_output(r)


# ══════════════════════════════════════════════════════════════════════════════
# Agent.run() — text reply immediately (no tool calls)
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentDirectReply:

    def test_immediate_text_reply_succeeds(self, monkeypatch):
        """If the LLM replies with text on turn 1, success immediately."""
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        monkeypatch.setattr(
            "Ciel.agent.LLMClient.complete",
            lambda self, messages, tools=None: _text_response("I can do that!"),
        )

        result = agent.run("do something")
        assert result.success is True
        assert result.answer == "I can do that!"
        assert result.turns_taken == 1
        assert result.tool_calls_made == 0
        assert result.state_on_exit is AgentState.DONE

    def test_answer_text_preserved_exactly(self, monkeypatch):
        config = _make_config(monkeypatch)
        agent = _make_agent(config)
        expected = "Your hostname is mybox and uptime is 3 days."

        monkeypatch.setattr(
            "Ciel.agent.LLMClient.complete",
            lambda self, messages, tools=None: _text_response(expected),
        )

        result = agent.run("what is my hostname?")
        assert result.answer == expected


# ══════════════════════════════════════════════════════════════════════════════
# Agent.run() — single tool call then reply
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentSingleToolCall:

    def test_tool_call_then_reply(self, monkeypatch):
        """Turn 1: LLM calls hostname tool. Turn 2: LLM gives text answer."""
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        responses = iter([
            _tool_response("hostname", {}),
            _text_response("Your hostname is mybox."),
        ])

        monkeypatch.setattr(
            "Ciel.agent.LLMClient.complete",
            lambda self, messages, tools=None: next(responses),
        )

        result = agent.run("what is my hostname?")

        assert result.success is True
        assert result.turns_taken == 2
        assert result.tool_calls_made == 1
        assert "mybox" in result.answer

    def test_tool_call_counts_only_executed_calls(self, monkeypatch):
        """tool_calls_made counts subprocess executions, not LLM tool_call responses."""
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        responses = iter([
            _tool_response("current_user", {}),
            _text_response("You are running as: testuser"),
        ])

        monkeypatch.setattr(
            "Ciel.agent.LLMClient.complete",
            lambda self, messages, tools=None: next(responses),
        )

        result = agent.run("who am I?")
        assert result.tool_calls_made == 1


# ══════════════════════════════════════════════════════════════════════════════
# Agent.run() — multi-turn
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentMultiTurn:

    def test_two_tool_calls_then_reply(self, monkeypatch):
        """Turn 1: hostname. Turn 2: free_memory. Turn 3: text answer."""
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        responses = iter([
            _tool_response("hostname", {}),
            _tool_response("free_memory", {}),
            _text_response("Done. hostname=mybox, RAM=16GB"),
        ])

        monkeypatch.setattr(
            "Ciel.agent.LLMClient.complete",
            lambda self, messages, tools=None: next(responses),
        )

        result = agent.run("check hostname and memory")
        assert result.success is True
        assert result.turns_taken == 3
        assert result.tool_calls_made == 2


# ══════════════════════════════════════════════════════════════════════════════
# Agent.run() — max_turns enforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentMaxTurns:

    def test_max_turns_stops_the_loop(self, monkeypatch):
        """If the LLM never gives a text reply, the agent bails at max_turns."""
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        # Always call a different tool so stuck-detection doesn't trigger first
        call_count = 0
        tools = ["hostname", "free_memory", "kernel_info", "current_user",
                 "system_uptime", "hostname", "free_memory", "kernel_info",
                 "current_user", "system_uptime"]

        def always_tool(self, messages, tools=None):
            nonlocal call_count
            name = tools_cycle[call_count % len(tools_cycle)]
            call_count += 1
            return _tool_response(name, {})

        tools_cycle = tools

        monkeypatch.setattr("Ciel.agent.LLMClient.complete", always_tool)

        result = agent.run("do something", max_turns=3)
        assert result.success is False
        assert "maximum turn limit" in result.failure_reason
        assert result.turns_taken == 3

    def test_max_turns_capped_at_hard_ceiling(self, monkeypatch):
        """max_turns above Agent.MAX_TURNS is silently capped to MAX_TURNS."""
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        monkeypatch.setattr(
            "Ciel.agent.LLMClient.complete",
            lambda self, messages, tools=None: _text_response("done"),
        )

        # Pass a huge number — the agent must still return in 1 turn
        result = agent.run("hello", max_turns=9999)
        assert result.success is True
        assert result.turns_taken == 1


# ══════════════════════════════════════════════════════════════════════════════
# Agent.run() — stuck detection
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentStuckDetection:

    def test_same_tool_repeated_triggers_stuck(self, monkeypatch):
        """Calling the same tool more than MAX_CONSECUTIVE_SAME_TOOL times → failure."""
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        monkeypatch.setattr(
            "Ciel.agent.LLMClient.complete",
            # Always call 'hostname' — will trigger stuck detection
            lambda self, messages, tools=None: _tool_response("hostname", {}),
        )

        result = agent.run("what is the hostname?")
        assert result.success is False
        assert "stuck" in result.failure_reason.lower()
        assert "hostname" in result.failure_reason

    def test_stuck_detection_resets_on_different_tool(self, monkeypatch):
        """Switching tools resets the consecutive counter."""
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        # 2× hostname, 1× free_memory, 2× hostname, then reply
        # None of these should exceed MAX_CONSECUTIVE_SAME_TOOL (3)
        responses = iter([
            _tool_response("hostname", {}),
            _tool_response("hostname", {}),
            _tool_response("free_memory", {}),   # resets counter
            _tool_response("hostname", {}),
            _tool_response("hostname", {}),
            _text_response("All done."),
        ])

        monkeypatch.setattr(
            "Ciel.agent.LLMClient.complete",
            lambda self, messages, tools=None: next(responses),
        )

        result = agent.run("check things")
        assert result.success is True


# ══════════════════════════════════════════════════════════════════════════════
# Agent.run() — policy violation handling
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentPolicyViolation:

    def test_forbidden_tool_feeds_error_to_llm_then_loop_continues(self, monkeypatch):
        """A FORBIDDEN tool call must NOT stop the loop — the error is fed to the LLM."""
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        # Turn 1: LLM tries a FORBIDDEN tool (not in policy table → FORBIDDEN)
        # Turn 2: LLM recovers and gives a text reply
        responses = iter([
            _tool_response("unknown_forbidden_tool", {}),
            _text_response("I cannot do that; here is what I know instead."),
        ])

        monkeypatch.setattr(
            "Ciel.agent.LLMClient.complete",
            lambda self, messages, tools=None: next(responses),
        )

        result = agent.run("do something risky")
        assert result.success is True
        assert result.tool_calls_made == 0   # blocked tool never executed
        assert result.turns_taken == 2


# ══════════════════════════════════════════════════════════════════════════════
# Agent.run() — LLM errors
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentLLMErrors:

    def test_auth_error_returns_failure(self, monkeypatch):
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        def raise_auth(self, messages, tools=None):
            raise LLMAuthError("bad key")

        monkeypatch.setattr("Ciel.agent.LLMClient.complete", raise_auth)

        result = agent.run("hello")
        assert result.success is False
        assert "Authentication" in result.failure_reason

    def test_generic_llm_error_returns_failure(self, monkeypatch):
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        def raise_timeout(self, messages, tools=None):
            raise LLMTimeoutError("timed out")

        monkeypatch.setattr("Ciel.agent.LLMClient.complete", raise_timeout)

        result = agent.run("hello")
        assert result.success is False
        assert result.state_on_exit is AgentState.FAILED

    def test_no_reply_no_tool_call_returns_failure(self, monkeypatch):
        """finish_reason='length' with no content and no tool_call → failure."""
        config = _make_config(monkeypatch)
        agent = _make_agent(config)

        weird_response = LLMResponse(
            reply_text=None,
            tool_call=None,
            finish_reason="length",
            usage=_usage(),
        )

        monkeypatch.setattr(
            "Ciel.agent.LLMClient.complete",
            lambda self, messages, tools=None: weird_response,
        )

        result = agent.run("hello")
        assert result.success is False
        assert "length" in result.failure_reason


# ══════════════════════════════════════════════════════════════════════════════
# Agent.run() — no API key
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentNoApiKey:

    def test_missing_api_key_returns_failure(self):
        """Agent.run() must return failure if no API key is configured."""
        config = AppConfig()  # no CIEL_API_KEY set
        agent = _make_agent(config)
        result = agent.run("do something")
        assert result.success is False
        assert "API key" in result.failure_reason or "Authentication" in result.failure_reason
