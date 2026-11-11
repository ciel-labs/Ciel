"""LLM client for Ciel — Qwen API via OpenAI-compatible interface.

Architecture position
---------------------
The LLM is an *input source*, not a trusted authority.

                    User task (plain text)
                           │
              ┌────────────▼────────────┐
              │  LLMClient.complete()   │
              │  POST /chat/completions │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  _parse_response()      │
              │  → LLMResponse          │
              │    .reply_text  or      │
              │    .tool_call           │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Policy engine          │◄─ blocks forbidden tools
              │  apply_policy()         │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  ToolRegistry           │◄─ validates tool + args
              │  build_request()        │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Executor               │
              │  execute()              │
              └─────────────────────────┘

The LLM never touches the executor directly. Everything it produces
passes through the policy engine and tool registry first.

Security rules
--------------
- api_key is extracted from SecretStr only at request time.
- Request body size is bounded; response body is capped at 1 MB.
- All JSON parsing is wrapped; malformed responses raise LLMParseError.
- Retries are bounded (max_retries); no infinite loops.
- Tool call arguments from the LLM are always validated by the registry
  before they reach the executor.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from Ciel.config import AppConfig
from Ciel.tools.base import ToolCall, ToolRegistry


# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_RESPONSE_BYTES: int = 1 * 1024 * 1024   # 1 MB response body cap
_RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the API for one request/response pair.

    Used for cost tracking and to detect runaway prompts.

    Attributes
    ----------
    prompt_tokens :
        Tokens in the input (system prompt + conversation history).
    completion_tokens :
        Tokens in the model's response.
    total_tokens :
        Sum of prompt and completion tokens (may differ from their sum
        if the API applies internal rounding).
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    """The parsed result of one call to LLMClient.complete().

    Exactly one of reply_text or tool_call will be non-None.
    The agent loop uses finish_reason to decide what to do next:

      finish_reason == "stop"        → reply_text is the final answer
      finish_reason == "tool_calls"  → tool_call must be executed
      finish_reason == "length"      → response was truncated (handle carefully)

    Attributes
    ----------
    reply_text :
        Plain-text reply from the model. Present when the model chose
        to respond in prose rather than invoke a tool.
    tool_call :
        A ToolCall the model wants to invoke. Present when
        finish_reason is "tool_calls". Must pass through the policy
        engine and tool registry before execution.
    finish_reason :
        The reason the model stopped generating. See above.
    usage :
        Token usage for this round-trip.
    """

    reply_text: str | None
    tool_call: ToolCall | None
    finish_reason: str
    usage: TokenUsage

    def has_tool_call(self) -> bool:
        """True if the model responded with a tool invocation request."""
        return self.tool_call is not None

    def has_reply(self) -> bool:
        """True if the model responded with plain text."""
        return self.reply_text is not None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Error types
# ══════════════════════════════════════════════════════════════════════════════

class LLMError(Exception):
    """Base class for all LLM client errors.

    Catching LLMError catches every LLM-related failure. Catching a
    specific subclass lets callers handle individual failure modes.
    """


class LLMAuthError(LLMError):
    """API key is missing, invalid, or revoked (HTTP 401).

    The agent must stop and ask the user to check CIEL_API_KEY.
    Retrying will not help.
    """


class LLMRateLimitError(LLMError):
    """Rate limit hit and all retry attempts exhausted (HTTP 429).

    The agent should wait before attempting another task.
    """


class LLMTimeoutError(LLMError):
    """The API did not respond within request_timeout seconds.

    May indicate a network issue or an overloaded API endpoint.
    """


class LLMServerError(LLMError):
    """The API returned a 5xx error after all retries were exhausted.

    Indicates a temporary problem on the API provider's side.
    """


class LLMParseError(LLMError):
    """The API response could not be parsed.

    Possible causes:
    - Response body is not valid JSON.
    - Required fields are missing from the JSON structure.
    - Tool call arguments are not valid JSON.
    - Response body exceeded _MAX_RESPONSE_BYTES.

    The raw response (or its size) is included in the message to aid
    debugging without exposing the full payload in logs.
    """


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Tool schema builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_tool_schema(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Convert registered tools into OpenAI function-calling schema format.

    The returned list is passed directly as the `tools` field in the
    API request body. Each entry tells the LLM:
    - What the tool is called (function.name)
    - What it does (function.description)
    - What arguments it accepts (function.parameters, JSON Schema format)

    Only string-typed parameters are supported in Phase 4. Richer types
    will be added when needed.

    Example output for a tool with one optional string parameter:
    [
      {
        "type": "function",
        "function": {
          "name": "disk_usage",
          "description": "Show disk space usage...",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {
                "type": "string",
                "description": "Absolute path to check..."
              }
            }
          }
        }
      }
    ]
    """
    schemas = []

    for tool in registry.all_tools():
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in tool.parameters:
            properties[param.name] = {
                "type": "string",
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        schemas.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        })

    return schemas


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Response parser
# ══════════════════════════════════════════════════════════════════════════════

def _parse_response(raw: dict[str, Any]) -> LLMResponse:
    """Parse a raw API response dict into a typed LLMResponse.

    Raises LLMParseError for any structural problem. This function
    must never raise KeyError, IndexError, or json.JSONDecodeError
    — all such errors are caught and re-raised as LLMParseError.

    Tool call arguments from the LLM arrive as a JSON *string* inside
    the response (OpenAI convention). We parse that string here and
    convert all values to str to satisfy ToolCall's dict[str, str] type.

    Parameters
    ----------
    raw :
        The parsed JSON response body from the API.

    Returns
    -------
    LLMResponse
        A fully typed, frozen response object.
    """
    try:
        choices = raw["choices"]
        if not choices:
            raise LLMParseError("API response contained an empty 'choices' list")

        choice = choices[0]
        finish_reason: str = choice.get("finish_reason") or "stop"
        message: dict[str, Any] = choice["message"]

        # Token usage (optional — some endpoints omit it)
        usage_raw = raw.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
            completion_tokens=int(usage_raw.get("completion_tokens", 0)),
            total_tokens=int(usage_raw.get("total_tokens", 0)),
        )

        # Tool call path
        tool_calls_raw = message.get("tool_calls")
        if tool_calls_raw:
            tc_raw = tool_calls_raw[0]
            fn = tc_raw["function"]
            tool_name: str = fn["name"]
            args_str: str = fn.get("arguments", "{}")

            try:
                args_parsed = json.loads(args_str)
            except json.JSONDecodeError as exc:
                raise LLMParseError(
                    f"Tool call arguments for {tool_name!r} are not valid JSON: {exc}"
                ) from exc

            if not isinstance(args_parsed, dict):
                raise LLMParseError(
                    f"Tool call arguments for {tool_name!r} must be a JSON object, "
                    f"got {type(args_parsed).__name__}"
                )

            # Coerce all argument values to str (ToolCall expects dict[str, str])
            arguments: dict[str, str] = {
                k: str(v) for k, v in args_parsed.items()
            }

            return LLMResponse(
                reply_text=None,
                tool_call=ToolCall(tool_name=tool_name, arguments=arguments),
                finish_reason=finish_reason,
                usage=usage,
            )

        # Text reply path
        reply_text: str = message.get("content") or ""
        return LLMResponse(
            reply_text=reply_text,
            tool_call=None,
            finish_reason=finish_reason,
            usage=usage,
        )

    except LLMParseError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMParseError(
            f"Unexpected API response structure: {exc}. "
            f"Top-level keys present: {sorted(raw.keys()) if isinstance(raw, dict) else 'N/A'}"
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LLM Client
# ══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """HTTP client for the Qwen OpenAI-compatible API.

    Typical usage
    -------------
    config = AppConfig()
    with LLMClient(config) as client:
        response = client.complete(messages, tools=tool_schema)
        if response.has_tool_call():
            # pass to policy engine + registry + executor
            ...
        else:
            print(response.reply_text)

    Thread safety
    -------------
    httpx.Client is not thread-safe. Create one LLMClient per thread
    if you need concurrent requests (not relevant for the agent loop).

    Error handling
    --------------
    complete() raises typed LLMError subclasses — never raw httpx errors.
    Callers should catch LLMError and handle or propagate it.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._endpoint = f"{config.api_base_url.rstrip('/')}/chat/completions"
        self._client = httpx.Client(timeout=config.request_timeout)

    # ── Context manager support ──────────────────────────────────────────

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()

    # ── Core request method ──────────────────────────────────────────────

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and return a typed response.

        Parameters
        ----------
        messages :
            Conversation history in OpenAI format. Must contain at least
            one message. Use MessageHistory.as_list() to build this.
        tools :
            Optional list of tool schemas from _build_tool_schema().
            When provided, the model may respond with a tool_call.

        Returns
        -------
        LLMResponse
            Parsed response with either reply_text or tool_call set.

        Raises
        ------
        LLMAuthError
            API key is missing or invalid.
        LLMRateLimitError
            Rate limit exceeded and retries exhausted.
        LLMTimeoutError
            Request timed out.
        LLMServerError
            5xx error after retries exhausted.
        LLMParseError
            Response could not be parsed.
        """
        if not self._config.has_api_key():
            raise LLMAuthError(
                "No API key configured. Set the CIEL_API_KEY environment variable."
            )

        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self._config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        last_error: LLMError | None = None

        for attempt in range(self._config.max_retries + 1):
            if attempt > 0:
                # Exponential backoff: 1s, 2s, 4s, ...
                wait = 2 ** (attempt - 1)
                time.sleep(wait)

            try:
                response = self._client.post(
                    self._endpoint,
                    json=body,
                    headers=headers,
                )
            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(
                    f"Request timed out after {self._config.request_timeout}s"
                )
                continue  # retry on timeout
            except httpx.RequestError as exc:
                last_error = LLMError(f"HTTP request failed: {exc}")
                continue

            # Auth error — do not retry
            if response.status_code == 401:
                raise LLMAuthError(
                    "API returned 401 Unauthorized. Check your CIEL_API_KEY."
                )

            # Rate limit — retry with backoff
            if response.status_code == 429:
                last_error = LLMRateLimitError(
                    f"Rate limit hit (attempt {attempt + 1}/{self._config.max_retries + 1})"
                )
                continue

            # Server errors — retry
            if response.status_code in _RETRY_STATUS_CODES:
                last_error = LLMServerError(
                    f"API returned {response.status_code} "
                    f"(attempt {attempt + 1}/{self._config.max_retries + 1})"
                )
                continue

            # Any other non-2xx — do not retry
            if not response.is_success:
                raise LLMError(
                    f"API returned unexpected status {response.status_code}: "
                    f"{response.text[:200]}"
                )

            # Cap response body size before parsing
            if len(response.content) > _MAX_RESPONSE_BYTES:
                raise LLMParseError(
                    f"API response body exceeds {_MAX_RESPONSE_BYTES} bytes "
                    f"({len(response.content)} received). Refusing to parse."
                )

            try:
                raw = response.json()
            except Exception as exc:
                raise LLMParseError(
                    f"API response is not valid JSON: {exc}"
                ) from exc

            return _parse_response(raw)

        # All attempts exhausted — raise the last recorded error
        assert last_error is not None
        raise last_error


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Prompt builder and message history
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(registry: ToolRegistry) -> str:
    """Build the system prompt that tells the LLM what Ciel is and what it can do.

    The system prompt is the first message in every conversation. It:
    1. Defines Ciel's role and operating principles.
    2. Lists every registered tool with its name, description, and parameters.
    3. Gives the LLM clear instructions on when to call tools vs. reply in prose.

    The tool list is derived from the live registry — it always reflects
    exactly what tools are registered, with no manual synchronisation needed.

    Parameters
    ----------
    registry :
        The ToolRegistry populated with available tools. Usually the
        result of default_registry().

    Returns
    -------
    str
        A multi-line system prompt string.
    """
    lines: list[str] = [
        "You are Ciel, a security-first autonomous Linux computer-use agent.",
        "You help users accomplish Linux tasks by using the tools available to you.",
        "",
        "Operating principles:",
        "- Always use a tool to gather real system information before answering.",
        "- Never guess, invent, or fabricate system state.",
        "- If a task requires information you do not have, call the appropriate tool first.",
        "- When you have gathered enough information, respond with a clear, concise answer.",
        "- Be honest about what you can and cannot do.",
        "",
        "Available tools:",
    ]

    for tool in registry.all_tools():
        lines.append(f"")
        lines.append(f"  {tool.name}")
        lines.append(f"    Description: {tool.description}")
        if tool.parameters:
            lines.append(f"    Parameters:")
            for param in tool.parameters:
                req_label = "required" if param.required else "optional"
                lines.append(f"      - {param.name} ({req_label}): {param.description}")
        else:
            lines.append(f"    Parameters: none")

    lines.append("")
    lines.append(
        "When you have finished answering the user's task, respond with plain text. "
        "Do not call another tool unless you need more information."
    )

    return "\n".join(lines)


def build_user_message(task: str) -> dict[str, str]:
    """Wrap a user task string into an OpenAI-format user message dict.

    Parameters
    ----------
    task :
        The user's plain-English task description.

    Returns
    -------
    dict
        {"role": "user", "content": task}
    """
    return {"role": "user", "content": task}


class MessageHistory:
    """Maintains conversation context in OpenAI message-list format.

    The agent loop uses this to build the `messages` argument for
    LLMClient.complete() across multiple tool-call rounds.

    Message flow in one agent turn
    -------------------------------
    1. add_user(task)                     ← user's task
    2. complete() → tool_call             ← model wants to call a tool
    3. add_tool_call(name, args, call_id) ← record the model's decision
    4. execute the tool, get output
    5. add_tool_result(name, out, call_id)← feed result back to model
    6. complete() → reply_text            ← model gives final answer
    7. add_assistant_text(reply)          ← record for context

    Parameters
    ----------
    system_prompt :
        The system message string. Becomes the first message in the list.
    """

    def __init__(self, system_prompt: str) -> None:
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    def add_user(self, text: str) -> None:
        """Append a user message."""
        self._messages.append({"role": "user", "content": text})

    def add_assistant_text(self, text: str) -> None:
        """Append a plain-text assistant reply."""
        self._messages.append({"role": "assistant", "content": text})

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, str],
        call_id: str,
    ) -> None:
        """Append the assistant's tool-call decision.

        This records what tool the model chose to invoke, which is
        required by the OpenAI protocol before adding a tool result.

        Parameters
        ----------
        tool_name :
            The tool being called (e.g. "disk_usage").
        arguments :
            The arguments dict the model supplied.
        call_id :
            Unique ID for this tool call. Must match the call_id used
            in the subsequent add_tool_result() call.
        """
        self._messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }],
        })

    def add_tool_result(
        self,
        tool_name: str,
        output: str,
        call_id: str,
    ) -> None:
        """Append the result of a tool execution.

        Parameters
        ----------
        tool_name :
            The tool that was executed.
        output :
            The stdout (or error message) from the tool execution.
        call_id :
            Must match the call_id from the preceding add_tool_call().
        """
        self._messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": output,
        })

    def as_list(self) -> list[dict[str, Any]]:
        """Return a copy of the message list for passing to LLMClient.complete()."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
