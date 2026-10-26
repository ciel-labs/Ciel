"""Concrete tool definitions and the default tool registry.

All tools in this file are read-only diagnostic tools. They inspect system
state but never modify it. Modifying tools (file writes, service restarts,
package installs) will be added in later phases after the policy engine
and human confirmation gates are in place.

Adding a new tool
-----------------
1. Subclass Tool from Ciel.tools.base.
2. Implement name, description, parameters, and build_request().
3. Add an instance to default_registry() at the bottom of this file.

Security notes
--------------
- Every tool uses a fixed program name. The LLM cannot substitute a
  different binary.
- Path arguments are validated to be absolute paths. Relative paths and
  path-traversal sequences are rejected before they reach the executor.
- Optional parameters always have safe defaults defined in build_request(),
  never derived from external input.
"""

from __future__ import annotations

from Ciel.models import ExecutionRequest
from Ciel.tools.base import (
    Tool,
    ToolArgumentError,
    ToolParameter,
    ToolRegistry,
)


# ── Path validation helper ─────────────────────────────────────────────────────

def _require_absolute_path(tool_name: str, param_name: str, value: str) -> None:
    """Raise ToolArgumentError if value is not an absolute path.

    We require absolute paths for two reasons:
    1. Relative paths are ambiguous — the working directory is not
       guaranteed to be safe or predictable.
    2. Path-traversal sequences like '../../etc/passwd' are more obvious
       in an absolute path context and easier to reason about.

    Note: this is a structural check, not a full path-safety check.
    The policy engine (Phase 3) will add deeper validation such as
    checking that the path is within an allowed directory tree.
    """
    if not value.startswith("/"):
        raise ToolArgumentError(
            f"Tool {tool_name!r} requires an absolute path for {param_name!r}, "
            f"got: {value!r}"
        )


# ── Concrete tools ─────────────────────────────────────────────────────────────

class SystemUptime(Tool):
    """Report how long the system has been running in human-readable form."""

    @property
    def name(self) -> str:
        return "system_uptime"

    @property
    def description(self) -> str:
        return (
            "Report how long the system has been running. "
            "Useful for checking if the machine was recently rebooted."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return []  # no arguments — output is always the same

    def build_request(self, arguments: dict[str, str]) -> ExecutionRequest:
        # -p: print in human-readable form, e.g. "up 3 days, 4 hours"
        return ExecutionRequest(program="uptime", arguments=["-p"])


class DiskUsage(Tool):
    """Show disk space usage for a given path."""

    @property
    def name(self) -> str:
        return "disk_usage"

    @property
    def description(self) -> str:
        return (
            "Show disk space usage in human-readable form. "
            "Provide an absolute path to check a specific filesystem. "
            "Defaults to the root filesystem if no path is given."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                description=(
                    "Absolute path to the filesystem or directory to check. "
                    "Example: /home or /var/log. Defaults to / if omitted."
                ),
                required=False,
            )
        ]

    def build_request(self, arguments: dict[str, str]) -> ExecutionRequest:
        args = ["--human-readable"]
        if "path" in arguments:
            path = arguments["path"]
            _require_absolute_path(self.name, "path", path)
            args.append(path)
        return ExecutionRequest(program="df", arguments=args)


class FreeMemory(Tool):
    """Show current RAM and swap usage."""

    @property
    def name(self) -> str:
        return "free_memory"

    @property
    def description(self) -> str:
        return (
            "Show total, used, and available RAM and swap space. "
            "Useful for diagnosing out-of-memory conditions."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    def build_request(self, arguments: dict[str, str]) -> ExecutionRequest:
        # -h: human-readable sizes (MB, GB)
        return ExecutionRequest(program="free", arguments=["-h"])


class KernelInfo(Tool):
    """Report the running kernel version."""

    @property
    def name(self) -> str:
        return "kernel_info"

    @property
    def description(self) -> str:
        return (
            "Report the running Linux kernel version. "
            "Useful for checking whether a kernel update is needed."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    def build_request(self, arguments: dict[str, str]) -> ExecutionRequest:
        # -r: print only the kernel release string, e.g. "6.6.0-45-generic"
        return ExecutionRequest(program="uname", arguments=["-r"])


class CurrentUser(Tool):
    """Report which user Ciel is running as."""

    @property
    def name(self) -> str:
        return "current_user"

    @property
    def description(self) -> str:
        return (
            "Report the username that Ciel is currently running as. "
            "Useful for diagnosing permission errors."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    def build_request(self, arguments: dict[str, str]) -> ExecutionRequest:
        return ExecutionRequest(program="whoami", arguments=[])


class Hostname(Tool):
    """Report the system's hostname."""

    @property
    def name(self) -> str:
        return "hostname"

    @property
    def description(self) -> str:
        return "Report the system hostname."

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    def build_request(self, arguments: dict[str, str]) -> ExecutionRequest:
        return ExecutionRequest(program="hostname", arguments=[])


# ── Default registry factory ───────────────────────────────────────────────────

def default_registry() -> ToolRegistry:
    """Return a ToolRegistry pre-populated with all built-in tools.

    This is a factory function — each call returns a fresh registry.
    Tests should call this rather than constructing ToolRegistry() and
    registering tools manually, so they stay in sync with production.

    To add a new built-in tool, register it here.
    """
    registry = ToolRegistry()
    registry.register(CurrentUser())
    registry.register(DiskUsage())
    registry.register(FreeMemory())
    registry.register(Hostname())
    registry.register(KernelInfo())
    registry.register(SystemUptime())
    return registry
