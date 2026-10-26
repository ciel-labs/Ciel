"""Tool abstraction layer: the containment boundary between the LLM and the executor.

Architecture
------------
                      ┌─────────────────────────────────────────────┐
                      │  The LLM produces a ToolCall                │
                      │  (tool name + argument values)              │
                      └──────────────────┬──────────────────────────┘
                                         │
                      ┌──────────────────▼──────────────────────────┐
                      │  ToolRegistry.build_request(call)           │
                      │  1. Is the tool registered?                 │
                      │  2. Are the arguments valid?                │
                      │  3. Build a typed ExecutionRequest          │
                      └──────────────────┬──────────────────────────┘
                                         │
                      ┌──────────────────▼──────────────────────────┐
                      │  Executor.execute(request)                  │
                      │  Runs the process safely                    │
                      └─────────────────────────────────────────────┘

The LLM never constructs an ExecutionRequest directly. It can only invoke
tools that are registered. Each tool has a fixed program. The LLM only
fills in the argument slots that the tool explicitly permits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from Ciel.models import ExecutionRequest


# ── Error types ────────────────────────────────────────────────────────────────

class ToolError(Exception):
    """Base class for all tool-related errors."""


class ToolNotFoundError(ToolError):
    """Raised when a ToolCall names a tool that is not in the registry.

    This is a security boundary: if a tool is not registered, it cannot
    be called, regardless of what the LLM requested.
    """


class ToolArgumentError(ToolError):
    """Raised when a ToolCall supplies invalid, missing, or unsafe arguments.

    Specific cases:
    - A required parameter is missing.
    - An unknown parameter was supplied (possible injection attempt).
    - A parameter value contains a null byte.
    """


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolParameter:
    """Describes one argument slot that a tool accepts.

    The LLM reads the `description` to understand what value to supply.
    The registry uses `required` to decide whether to reject a ToolCall
    that omits this parameter.
    """

    name: str
    description: str
    required: bool = True


@dataclass
class ToolCall:
    """A request from the LLM to invoke a specific tool.

    This is the raw, unvalidated request produced by the LLM response
    parser. The ToolRegistry validates it before any process is started.

    Why not frozen=True?
    --------------------
    dict[str, str] cannot be part of a frozen dataclass in a meaningful
    way — the dict itself is still mutable. We leave ToolCall mutable
    because it is a short-lived data-transfer object, not a value type.
    """

    tool_name: str
    arguments: dict[str, str] = field(default_factory=dict)


# ── Abstract base tool ─────────────────────────────────────────────────────────

class Tool(ABC):
    """Abstract base class for all Ciel tools.

    Each concrete tool represents one specific capability that Ciel is
    allowed to use. A tool has a fixed program. The LLM can only supply
    the values listed in `parameters` — nothing else ever reaches the
    executor.

    Implementing a new tool
    -----------------------
    1. Subclass Tool.
    2. Implement the four abstract properties/methods.
    3. In build_request(), construct and return an ExecutionRequest using
       only the validated arguments and your fixed program path.
    4. Register an instance with ToolRegistry.register().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used in ToolCall.tool_name.

        Use lowercase_with_underscores. Example: 'disk_usage'.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """Plain-English description for the LLM and for humans.

        Explain what the tool does and when to use it.
        """

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Ordered list of argument slots this tool accepts.

        Return an empty list for tools that take no arguments.
        """

    @abstractmethod
    def build_request(self, arguments: dict[str, str]) -> ExecutionRequest:
        """Translate validated arguments into an ExecutionRequest.

        Called by ToolRegistry.build_request() after validate_arguments()
        has already confirmed the arguments are structurally valid.

        Implementations must:
        - Use a hard-coded, absolute or PATH-resolved program name.
        - Only use values from the `arguments` dict — never from env vars,
          config files, or any other external source.
        - Not perform I/O, network calls, or any side effects.
        - Return a fully constructed ExecutionRequest.
        """

    def validate_arguments(self, arguments: dict[str, str]) -> None:
        """Check that arguments are complete, known, and safe.

        Called automatically by ToolRegistry.build_request() before
        build_request() is ever called. Tools should not call this
        themselves.

        Checks performed
        ----------------
        1. No unknown argument names (prevents injection via extra keys).
        2. All required parameters are present.
        3. No argument value contains a null byte.
        """
        param_names = {p.name for p in self.parameters}
        required_names = {p.name for p in self.parameters if p.required}

        # Unknown keys are not merely extra data — they may be an attempt
        # to supply arguments that the tool does not expect, potentially
        # confusing the build_request() implementation.
        unknown = set(arguments) - param_names
        if unknown:
            raise ToolArgumentError(
                f"Tool {self.name!r} received unknown arguments: "
                f"{sorted(unknown)}. Allowed: {sorted(param_names)}"
            )

        missing = required_names - set(arguments)
        if missing:
            raise ToolArgumentError(
                f"Tool {self.name!r} is missing required arguments: "
                f"{sorted(missing)}"
            )

        for key, value in arguments.items():
            if "\x00" in value:
                raise ToolArgumentError(
                    f"Tool {self.name!r} argument {key!r} contains a null byte"
                )


# ── Registry ───────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Maps tool names to Tool definitions.

    The registry is the authoritative allowlist of what Ciel is permitted
    to do. A tool that is not registered cannot be invoked, regardless of
    what the LLM requests.

    Typical usage
    -------------
    registry = default_registry()          # get pre-populated registry
    call = ToolCall("disk_usage", {"path": "/home"})
    request = registry.build_request(call) # validated ExecutionRequest
    result = execute(request)              # runs df -h /home safely
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry.

        Raises ValueError if a tool with the same name is already registered.
        Silent overwrites are not permitted because they would allow one
        part of the codebase to shadow another tool's definition without
        an obvious failure.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named {tool.name!r} is already registered. "
                "Overwriting a registered tool is not permitted."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the Tool with the given name.

        Raises ToolNotFoundError if no such tool is registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                f"No tool named {name!r} is registered. "
                f"Registered tools: {sorted(self._tools)}"
            )
        return self._tools[name]

    def build_request(self, call: ToolCall) -> ExecutionRequest:
        """Validate a ToolCall and translate it into an ExecutionRequest.

        Steps
        -----
        1. Look up the tool by name (raises ToolNotFoundError if absent).
        2. Validate the arguments (raises ToolArgumentError if invalid).
        3. Delegate to tool.build_request() for the ExecutionRequest.

        The returned ExecutionRequest is then passed to execute(), which
        runs validate_request() again as a second independent check.
        Two layers of validation is intentional: the registry validates
        tool-level semantics; the executor validates OS-level safety.
        """
        tool = self.get(call.tool_name)
        tool.validate_arguments(call.arguments)
        return tool.build_request(call.arguments)

    def all_tools(self) -> list[Tool]:
        """Return all registered tools sorted by name."""
        return [self._tools[name] for name in sorted(self._tools)]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        names = sorted(self._tools)
        return f"ToolRegistry({names})"
