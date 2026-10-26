"""Tests for the tool registry and concrete tool definitions.

These tests cover:
- ToolRegistry registration and lookup
- Argument validation (missing, unknown, null bytes)
- Each concrete tool's build_request() output
- Path validation for tools with path arguments
- Full integration: registry → execute()

Tests that spawn real processes are marked @pytest.mark.slow.
All other tests are pure in-memory and run instantly.
"""

import pytest

from Ciel.executor import execute
from Ciel.models import ExecutionRequest
from Ciel.tools import (
    ToolArgumentError,
    ToolCall,
    ToolNotFoundError,
    ToolRegistry,
    default_registry,
    CurrentUser,
    DiskUsage,
    FreeMemory,
    Hostname,
    KernelInfo,
    SystemUptime,
)
from Ciel.tools.base import Tool, ToolParameter


# ── Registry: registration ─────────────────────────────────────────────────────

class TestRegistryRegistration:

    def test_register_and_retrieve_tool(self):
        """A registered tool must be retrievable by name."""
        registry = ToolRegistry()
        registry.register(Hostname())
        tool = registry.get("hostname")
        assert tool.name == "hostname"

    def test_duplicate_registration_raises(self):
        """Registering the same tool name twice must raise ValueError.

        Silent overwrites would let one module shadow another's tool
        definition without any visible failure.
        """
        registry = ToolRegistry()
        registry.register(Hostname())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(Hostname())

    def test_get_unknown_tool_raises_tool_not_found(self):
        """Requesting an unregistered tool must raise ToolNotFoundError."""
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("rm")

    def test_contains_returns_true_for_registered_tool(self):
        registry = ToolRegistry()
        registry.register(Hostname())
        assert "hostname" in registry

    def test_contains_returns_false_for_unknown_tool(self):
        registry = ToolRegistry()
        assert "rm" not in registry

    def test_len_reflects_number_of_registered_tools(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(Hostname())
        assert len(registry) == 1
        registry.register(CurrentUser())
        assert len(registry) == 2

    def test_all_tools_sorted_by_name(self):
        registry = default_registry()
        names = [t.name for t in registry.all_tools()]
        assert names == sorted(names)

    def test_default_registry_has_six_tools(self):
        registry = default_registry()
        assert len(registry) == 6

    def test_default_registry_contains_expected_tools(self):
        registry = default_registry()
        expected = {
            "current_user", "disk_usage", "free_memory",
            "hostname", "kernel_info", "system_uptime",
        }
        actual = {t.name for t in registry.all_tools()}
        assert actual == expected


# ── Registry: argument validation ─────────────────────────────────────────────

class TestRegistryArgumentValidation:

    def test_unknown_argument_raises_tool_argument_error(self):
        """An argument the tool does not declare must be rejected.

        Allowing unknown arguments could let an attacker supply extra
        data that confuses the build_request() implementation.
        """
        registry = default_registry()
        call = ToolCall(tool_name="hostname", arguments={"unexpected_key": "value"})
        with pytest.raises(ToolArgumentError, match="unknown arguments"):
            registry.build_request(call)

    def test_missing_required_argument_raises_tool_argument_error(self):
        """Omitting a required argument must raise ToolArgumentError."""
        # We need a tool with a required parameter. Let's use a minimal
        # custom tool since all built-in tools either have no required
        # params or have optional ones.
        class RequiredParamTool(Tool):
            @property
            def name(self): return "test_required"
            @property
            def description(self): return "test"
            @property
            def parameters(self):
                return [ToolParameter("target", "A required target", required=True)]
            def build_request(self, arguments):
                return ExecutionRequest(program="echo", arguments=[arguments["target"]])

        registry = ToolRegistry()
        registry.register(RequiredParamTool())
        call = ToolCall(tool_name="test_required", arguments={})
        with pytest.raises(ToolArgumentError, match="missing required"):
            registry.build_request(call)

    def test_null_byte_in_argument_raises_tool_argument_error(self):
        """A null byte in any argument value must be rejected."""
        registry = default_registry()
        call = ToolCall(
            tool_name="disk_usage",
            arguments={"path": "/home\x00injected"},
        )
        with pytest.raises(ToolArgumentError, match="null byte"):
            registry.build_request(call)

    def test_unregistered_tool_name_raises_tool_not_found(self):
        """A ToolCall naming an unregistered tool must raise ToolNotFoundError."""
        registry = default_registry()
        call = ToolCall(tool_name="rm", arguments={})
        with pytest.raises(ToolNotFoundError):
            registry.build_request(call)


# ── Concrete tools: build_request() output ─────────────────────────────────────

class TestConcreteToolRequests:
    """Verify that each tool builds the correct ExecutionRequest.

    These tests do not execute any process. They only check that
    build_request() produces the right program and arguments.
    """

    def test_system_uptime_uses_uptime_p(self):
        req = SystemUptime().build_request({})
        assert req.program == "uptime"
        assert req.arguments == ["-p"]

    def test_hostname_uses_hostname(self):
        req = Hostname().build_request({})
        assert req.program == "hostname"
        assert req.arguments == []

    def test_current_user_uses_whoami(self):
        req = CurrentUser().build_request({})
        assert req.program == "whoami"
        assert req.arguments == []

    def test_kernel_info_uses_uname_r(self):
        req = KernelInfo().build_request({})
        assert req.program == "uname"
        assert req.arguments == ["-r"]

    def test_free_memory_uses_free_h(self):
        req = FreeMemory().build_request({})
        assert req.program == "free"
        assert req.arguments == ["-h"]

    def test_disk_usage_without_path_uses_root(self):
        """When path is omitted, df must run without a path (defaults to all mounts)."""
        req = DiskUsage().build_request({})
        assert req.program == "df"
        assert "--human-readable" in req.arguments
        assert len(req.arguments) == 1  # only the flag, no path

    def test_disk_usage_with_absolute_path(self):
        req = DiskUsage().build_request({"path": "/home"})
        assert req.program == "df"
        assert "/home" in req.arguments

    def test_disk_usage_with_relative_path_raises(self):
        """A relative path must be rejected before it reaches the executor."""
        with pytest.raises(ToolArgumentError, match="absolute path"):
            DiskUsage().build_request({"path": "relative/path"})

    def test_disk_usage_with_traversal_path_raises(self):
        """A path-traversal attempt must be rejected."""
        with pytest.raises(ToolArgumentError, match="absolute path"):
            DiskUsage().build_request({"path": "../etc/passwd"})


# ── Full integration: registry → executor ─────────────────────────────────────

class TestRegistryToExecutorIntegration:
    """End-to-end tests that run real processes via the registry."""

    @pytest.mark.slow
    def test_hostname_tool_runs_and_returns_output(self):
        """hostname must return a non-empty string."""
        registry = default_registry()
        call = ToolCall(tool_name="hostname", arguments={})
        request = registry.build_request(call)
        result = execute(request)
        assert result.exit_code == 0
        assert result.stdout.strip() != ""
        assert result.timed_out is False

    @pytest.mark.slow
    def test_current_user_tool_runs_and_returns_output(self):
        """whoami must return a non-empty username."""
        registry = default_registry()
        call = ToolCall(tool_name="current_user", arguments={})
        request = registry.build_request(call)
        result = execute(request)
        assert result.exit_code == 0
        assert result.stdout.strip() != ""

    @pytest.mark.slow
    def test_kernel_info_tool_runs_and_returns_version(self):
        """uname -r must return something that looks like a kernel version."""
        registry = default_registry()
        call = ToolCall(tool_name="kernel_info", arguments={})
        request = registry.build_request(call)
        result = execute(request)
        assert result.exit_code == 0
        # Kernel versions always contain dots, e.g. "6.6.0-45-generic"
        assert "." in result.stdout

    @pytest.mark.slow
    def test_disk_usage_tool_runs_with_root_path(self):
        """df --human-readable / must return output containing 'Filesystem'."""
        registry = default_registry()
        call = ToolCall(tool_name="disk_usage", arguments={"path": "/"})
        request = registry.build_request(call)
        result = execute(request)
        assert result.exit_code == 0
        assert "Filesystem" in result.stdout or "filesystem" in result.stdout.lower()
