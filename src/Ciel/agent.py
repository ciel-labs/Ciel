"""Agent state machine — the orchestration layer for Ciel.

This module wires every previously built layer into one coherent loop:

    User task (string)
         │
    ┌────▼────────────────────────────────────────┐
    │  MessageHistory  (builds conversation)      │
    └────┬────────────────────────────────────────┘
         │  loop (bounded by max_turns)
    ┌────▼────────────────────────────────────────┐
    │  LLMClient.complete()     [PLANNING state]  │
    └────┬────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────┐
    │  reply_text? → TaskResult(success=True)     │
    │  tool_call?  → continue                     │
    └────┬────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────┐
    │  apply_policy()           [VALIDATING]      │◄── blocks FORBIDDEN
    └────┬────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────┐
    │  ToolRegistry.build_request()               │◄── validates args
    └────┬────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────┐
    │  execute()                [EXECUTING]       │◄── safe subprocess
    └────┬────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────┐
    │  Add tool result to history   [VERIFYING]   │
    └────┬────────────────────────────────────────┘
         │  back to top of loop
         ▼

Security guarantees maintained
-------------------------------
- The LLM never calls the executor directly.
- Every tool call passes through apply_policy() before execution.
- Every tool call passes through ToolRegistry.build_request() before execution.
- The agent loop is bounded — it cannot run forever.
- Stuck detection prevents the LLM from looping on the same tool.
- All errors surface as TaskResult(success=False), never as exceptions
  that propagate to the caller (the agent is responsible for its own errors).

What this module does NOT do
-----------------------------
- It does not render output to the terminal (the CLI does that).
- It does not persist audit logs (Phase 9 / Step 10).
- It does not handle human confirmation prompts (all current tools are
  LOW risk; confirmation will be wired in at the CLI layer in a later phase).
- It does not retry failed LLM calls beyond what LLMClient already does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from Ciel.config import AppConfig
from Ciel.executor import execute
from Ciel.llm import (
    LLMAuthError,
    LLMClient,
    LLMError,
    _build_tool_schema,
    build_system_prompt,
    MessageHistory,
)
from Ciel.models import ExecutionResult
from Ciel.policy import PolicyViolation, apply_policy
from Ciel.tools.base import ToolArgumentError, ToolCall, ToolNotFoundError, ToolRegistry
from Ciel.tools.definitions import default_registry
from Ciel.validator import ValidationError


# ══════════════════════════════════════════════════════════════════════════════
# State types
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(Enum):
    """The lifecycle state of the agent within one task execution.

    Transitions
    -----------
    IDLE → PLANNING → VALIDATING → EXECUTING → PLANNING (loop)
                                                        ↓ (when LLM replies)
                                               DONE or FAILED
    """

    IDLE       = "idle"        # no task running
    PLANNING   = "planning"    # calling the LLM for a decision
    VALIDATING = "validating"  # running policy + registry checks
    EXECUTING  = "executing"   # running the approved tool subprocess
    VERIFYING  = "verifying"   # feeding tool output back to the LLM
    DONE       = "done"        # LLM gave a final text answer
    FAILED     = "failed"      # unrecoverable error or limit reached


@dataclass(frozen=True)
class TaskResult:
    """The outcome of one complete agent task execution.

    Always produced by Agent.run() — the caller never receives an
    unhandled exception from the agent itself.

    Attributes
    ----------
    success :
        True if the agent produced a final answer. False on any failure.
    answer :
        The LLM's final plain-text answer to the user. Set iff success=True.
    failure_reason :
        Human-readable explanation of what went wrong. Set iff success=False.
    turns_taken :
        Number of LLM calls made during this task.
    tool_calls_made :
        Number of tools actually executed (excluding rejected calls).
    state_on_exit :
        The AgentState when the task ended. Useful for diagnostics.
    """

    success: bool
    answer: str | None = None
    failure_reason: str | None = None
    turns_taken: int = 0
    tool_calls_made: int = 0
    state_on_exit: AgentState = AgentState.IDLE


# ══════════════════════════════════════════════════════════════════════════════
# Output formatter
# ══════════════════════════════════════════════════════════════════════════════

def _format_tool_output(result: ExecutionResult) -> str:
    """Convert an ExecutionResult into a string to feed back to the LLM.

    The LLM needs readable output, not a Python object. We include:
    - stdout (the main output)
    - stderr if non-empty (warnings, errors from the program)
    - A note if the process timed out
    - A note if the exit code was non-zero
    - A note if the output was truncated

    We do NOT include raw binary data or the duration (irrelevant to the LLM).
    """
    parts: list[str] = []

    if result.timed_out:
        parts.append("[Process timed out and was killed]")

    if result.stdout.strip():
        parts.append(result.stdout.rstrip())

    if result.stderr.strip():
        parts.append(f"[stderr]: {result.stderr.rstrip()}")

    if result.exit_code is not None and result.exit_code != 0:
        parts.append(f"[exit code: {result.exit_code}]")

    if result.stdout_truncated or result.stderr_truncated:
        parts.append("[Note: output was truncated to stay within limits]")

    return "\n".join(parts) if parts else "(no output)"


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════

class Agent:
    """Orchestrates the full task execution loop for Ciel.

    The agent is stateless between tasks — each call to run() starts
    a fresh conversation. It is NOT thread-safe; use one Agent per thread.

    Parameters
    ----------
    config :
        AppConfig instance. Must have an API key set for real tasks.
    registry :
        ToolRegistry to use. Defaults to default_registry() if None.
        Pass a custom registry in tests to control which tools are available.

    Usage
    -----
    config = AppConfig()          # reads CIEL_API_KEY from env
    agent = Agent(config)
    result = agent.run("How much disk space do I have?")
    if result.success:
        print(result.answer)
    else:
        print(f"Failed: {result.failure_reason}")
    """

    # Hard limits — not configurable at runtime (they are safety ceilings)
    MAX_TURNS: int = 10
    MAX_CONSECUTIVE_SAME_TOOL: int = 3

    def __init__(
        self,
        config: AppConfig,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry if registry is not None else default_registry()
        self._tool_schema = _build_tool_schema(self._registry)
        self._system_prompt = build_system_prompt(self._registry)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, task: str, *, max_turns: int = MAX_TURNS) -> TaskResult:
        """Execute a natural-language task and return the result.

        This method never raises. All errors are captured in TaskResult.

        Parameters
        ----------
        task :
            The user's plain-English instruction.
        max_turns :
            Maximum number of LLM calls before giving up. Capped at
            self.MAX_TURNS to prevent infinite loops.

        Returns
        -------
        TaskResult
            Always returned. Check .success to determine outcome.
        """
        # Enforce the hard ceiling even if the caller passes a larger value
        effective_max_turns = min(max_turns, self.MAX_TURNS)

        history = MessageHistory(self._system_prompt)
        history.add_user(task)

        with LLMClient(self._config) as client:
            return self._run_loop(task, history, client, effective_max_turns)

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run_loop(
        self,
        task: str,
        history: MessageHistory,
        client: LLMClient,
        max_turns: int,
    ) -> TaskResult:
        """The main state-machine loop. Separated from run() for testability."""

        state = AgentState.IDLE
        turns = 0
        tool_calls_made = 0
        last_tool_name: str | None = None
        consecutive_same_tool = 0

        while True:

            # ── Limit check ───────────────────────────────────────────────────
            if turns >= max_turns:
                return TaskResult(
                    success=False,
                    failure_reason=(
                        f"Reached the maximum turn limit ({max_turns}). "
                        "The task was not completed."
                    ),
                    turns_taken=turns,
                    tool_calls_made=tool_calls_made,
                    state_on_exit=AgentState.FAILED,
                )

            # ── LLM call (PLANNING) ───────────────────────────────────────────
            state = AgentState.PLANNING
            turns += 1

            try:
                response = client.complete(
                    history.as_list(),
                    tools=self._tool_schema,
                )
            except LLMAuthError as exc:
                return TaskResult(
                    success=False,
                    failure_reason=f"Authentication failed: {exc}",
                    turns_taken=turns,
                    tool_calls_made=tool_calls_made,
                    state_on_exit=AgentState.FAILED,
                )
            except LLMError as exc:
                return TaskResult(
                    success=False,
                    failure_reason=f"LLM error after retries: {exc}",
                    turns_taken=turns,
                    tool_calls_made=tool_calls_made,
                    state_on_exit=AgentState.FAILED,
                )

            # ── Text reply → task complete (DONE) ─────────────────────────────
            if response.has_reply():
                state = AgentState.DONE
                history.add_assistant_text(response.reply_text or "")
                return TaskResult(
                    success=True,
                    answer=response.reply_text,
                    turns_taken=turns,
                    tool_calls_made=tool_calls_made,
                    state_on_exit=AgentState.DONE,
                )

            # ── Unexpected: neither reply nor tool_call ────────────────────────
            if not response.has_tool_call():
                return TaskResult(
                    success=False,
                    failure_reason=(
                        f"LLM returned finish_reason={response.finish_reason!r} "
                        "but produced neither a text reply nor a tool call."
                    ),
                    turns_taken=turns,
                    tool_calls_made=tool_calls_made,
                    state_on_exit=AgentState.FAILED,
                )

            # ── Tool call path (VALIDATING) ───────────────────────────────────
            state = AgentState.VALIDATING
            tool_call: ToolCall = response.tool_call  # type: ignore[assignment]
            call_id = f"call_{turns}"

            # Stuck detection: same tool called too many times in a row
            if tool_call.tool_name == last_tool_name:
                consecutive_same_tool += 1
            else:
                consecutive_same_tool = 1
                last_tool_name = tool_call.tool_name

            if consecutive_same_tool > self.MAX_CONSECUTIVE_SAME_TOOL:
                return TaskResult(
                    success=False,
                    failure_reason=(
                        f"Agent appears stuck: {tool_call.tool_name!r} was called "
                        f"{consecutive_same_tool} times consecutively without "
                        "producing a final answer."
                    ),
                    turns_taken=turns,
                    tool_calls_made=tool_calls_made,
                    state_on_exit=AgentState.FAILED,
                )

            # Policy check
            # auto_confirm=True: the agent loop is non-interactive.
            # All current tools are LOW risk so this never triggers a prompt.
            # In a future phase, the CLI will confirm MEDIUM/HIGH before
            # calling agent.run(), not inside the loop.
            try:
                apply_policy(
                    tool_call.tool_name,
                    tool_call.arguments,
                    auto_confirm=True,
                )
            except PolicyViolation as exc:
                # Feed the rejection back to the LLM so it can try differently
                history.add_tool_call(
                    tool_call.tool_name, tool_call.arguments, call_id
                )
                history.add_tool_result(
                    tool_call.tool_name,
                    f"POLICY VIOLATION — this tool call was blocked: {exc}",
                    call_id,
                )
                continue

            # Registry validation: check the tool is registered and args are valid
            try:
                request = self._registry.build_request(tool_call)
            except ToolNotFoundError as exc:
                history.add_tool_call(
                    tool_call.tool_name, tool_call.arguments, call_id
                )
                history.add_tool_result(
                    tool_call.tool_name,
                    f"TOOL NOT FOUND: {exc}",
                    call_id,
                )
                continue
            except ToolArgumentError as exc:
                history.add_tool_call(
                    tool_call.tool_name, tool_call.arguments, call_id
                )
                history.add_tool_result(
                    tool_call.tool_name,
                    f"INVALID ARGUMENTS: {exc}",
                    call_id,
                )
                continue

            # ── Execute (EXECUTING) ───────────────────────────────────────────
            state = AgentState.EXECUTING
            tool_calls_made += 1

            try:
                result = execute(request)
            except ValidationError as exc:
                # The validator caught something the tool schema missed
                history.add_tool_call(
                    tool_call.tool_name, tool_call.arguments, call_id
                )
                history.add_tool_result(
                    tool_call.tool_name,
                    f"EXECUTION REFUSED (validation): {exc}",
                    call_id,
                )
                continue

            # ── Feed result back to LLM (VERIFYING) ──────────────────────────
            state = AgentState.VERIFYING
            output = _format_tool_output(result)
            history.add_tool_call(tool_call.tool_name, tool_call.arguments, call_id)
            history.add_tool_result(tool_call.tool_name, output, call_id)
            # Loop back → LLM will process the result in the next turn
