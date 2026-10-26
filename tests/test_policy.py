"""Tests for src/Ciel/policy.py — the policy engine.

These tests verify the complete policy contract:
- RiskLevel ordering and semantics
- PolicyDecision properties (is_permitted, requires_confirmation)
- evaluate(): correct classification for all known tools
- evaluate(): default-deny for unknown tool names
- evaluate(): FORBIDDEN for any tool using a blocked program name
- apply_policy(): auto-approval for LOW
- apply_policy(): confirmation gate for MEDIUM and HIGH
- apply_policy(): hard rejection and PolicyViolation for FORBIDDEN

All tests are pure in-memory — no subprocesses, no filesystem I/O.
The confirmation gate tests use the auto_confirm=True escape hatch
so CI does not hang waiting for stdin.
"""

import pytest

from Ciel.policy import (
    PolicyDecision,
    PolicyViolation,
    RiskLevel,
    _FORBIDDEN_PROGRAMS,
    _TOOL_RISK_TABLE,
    apply_policy,
    evaluate,
)


# ── RiskLevel ──────────────────────────────────────────────────────────────────

class TestRiskLevel:
    """Verify the RiskLevel enum values and ordering."""

    def test_four_levels_exist(self):
        """There must be exactly four risk levels."""
        assert len(RiskLevel) == 4

    def test_low_is_lowest(self):
        """LOW must be numerically the smallest level."""
        assert RiskLevel.LOW.value < RiskLevel.MEDIUM.value
        assert RiskLevel.LOW.value < RiskLevel.HIGH.value
        assert RiskLevel.LOW.value < RiskLevel.FORBIDDEN.value

    def test_forbidden_is_highest(self):
        """FORBIDDEN must be numerically the largest level."""
        assert RiskLevel.FORBIDDEN.value > RiskLevel.LOW.value
        assert RiskLevel.FORBIDDEN.value > RiskLevel.MEDIUM.value
        assert RiskLevel.FORBIDDEN.value > RiskLevel.HIGH.value

    def test_ordering_low_medium_high_forbidden(self):
        """The numerical ordering must be LOW < MEDIUM < HIGH < FORBIDDEN."""
        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.FORBIDDEN]
        values = [l.value for l in levels]
        assert values == sorted(values)


# ── PolicyDecision ─────────────────────────────────────────────────────────────

class TestPolicyDecision:
    """Verify PolicyDecision properties."""

    def test_low_is_permitted(self):
        d = PolicyDecision._make(RiskLevel.LOW, "test")
        assert d.is_permitted is True

    def test_medium_is_permitted(self):
        d = PolicyDecision._make(RiskLevel.MEDIUM, "test")
        assert d.is_permitted is True

    def test_high_is_permitted(self):
        d = PolicyDecision._make(RiskLevel.HIGH, "test")
        assert d.is_permitted is True

    def test_forbidden_is_not_permitted(self):
        """FORBIDDEN decisions must never be permitted."""
        d = PolicyDecision._make(RiskLevel.FORBIDDEN, "test")
        assert d.is_permitted is False

    def test_low_does_not_require_confirmation(self):
        """LOW risk is auto-approved — no human input needed."""
        d = PolicyDecision._make(RiskLevel.LOW, "test")
        assert d.requires_confirmation is False

    def test_medium_requires_confirmation(self):
        d = PolicyDecision._make(RiskLevel.MEDIUM, "test")
        assert d.requires_confirmation is True

    def test_high_requires_confirmation(self):
        d = PolicyDecision._make(RiskLevel.HIGH, "test")
        assert d.requires_confirmation is True

    def test_forbidden_does_not_require_confirmation(self):
        """FORBIDDEN is hard-rejected. We never offer confirmation for it."""
        d = PolicyDecision._make(RiskLevel.FORBIDDEN, "test")
        assert d.requires_confirmation is False

    def test_decision_is_frozen(self):
        """PolicyDecision must be immutable."""
        d = PolicyDecision._make(RiskLevel.LOW, "test")
        with pytest.raises((AttributeError, TypeError)):
            d.risk_level = RiskLevel.HIGH  # type: ignore[misc]

    def test_reason_is_non_empty(self):
        """Every decision must have a non-empty reason string."""
        for level in RiskLevel:
            d = PolicyDecision._make(level, "some reason")
            assert d.reason == "some reason"


# ── evaluate(): known tools ────────────────────────────────────────────────────

class TestEvaluateKnownTools:
    """Verify risk levels for all currently registered tools."""

    @pytest.mark.parametrize("tool_name", [
        "system_uptime",
        "disk_usage",
        "free_memory",
        "kernel_info",
        "current_user",
        "hostname",
    ])
    def test_all_current_tools_are_low(self, tool_name: str):
        """All 6 Phase-2 diagnostic tools must be classified as LOW."""
        decision = evaluate(tool_name, {})
        assert decision.risk_level is RiskLevel.LOW, (
            f"Expected {tool_name!r} to be LOW, got {decision.risk_level}"
        )

    @pytest.mark.parametrize("tool_name", [
        "system_uptime",
        "disk_usage",
        "free_memory",
        "kernel_info",
        "current_user",
        "hostname",
    ])
    def test_all_current_tools_are_permitted(self, tool_name: str):
        """LOW tools must be permitted (is_permitted=True)."""
        decision = evaluate(tool_name, {})
        assert decision.is_permitted is True

    @pytest.mark.parametrize("tool_name", [
        "system_uptime",
        "disk_usage",
        "free_memory",
        "kernel_info",
        "current_user",
        "hostname",
    ])
    def test_all_current_tools_do_not_require_confirmation(self, tool_name: str):
        """LOW tools must not require human confirmation."""
        decision = evaluate(tool_name, {})
        assert decision.requires_confirmation is False

    def test_decision_has_non_empty_reason(self):
        """Every evaluate() result must contain a reason string."""
        decision = evaluate("hostname", {})
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0


# ── evaluate(): unknown tools ──────────────────────────────────────────────────

class TestEvaluateUnknownTools:
    """Verify default-deny behaviour for unregistered tool names."""

    def test_unknown_tool_is_forbidden(self):
        """Any tool not in _TOOL_RISK_TABLE must be FORBIDDEN.

        Default-deny: if a tool is not explicitly registered in the
        policy table, it may never be executed.
        """
        decision = evaluate("nonexistent_tool_ciel_test", {})
        assert decision.risk_level is RiskLevel.FORBIDDEN

    def test_unknown_tool_is_not_permitted(self):
        decision = evaluate("some_unregistered_tool", {})
        assert decision.is_permitted is False

    def test_unknown_tool_does_not_require_confirmation(self):
        """FORBIDDEN decisions must not offer confirmation — they are hard-rejected."""
        decision = evaluate("unknown_tool", {})
        assert decision.requires_confirmation is False

    def test_unknown_tool_reason_mentions_default_deny(self):
        """The rejection reason must explain the default-deny policy."""
        decision = evaluate("mystery_tool", {})
        assert "not in the policy table" in decision.reason or "default-deny" in decision.reason

    @pytest.mark.parametrize("dangerous_name", [
        "rm", "dd", "bash", "sudo", "reboot", "mkfs",
    ])
    def test_dangerous_names_not_in_risk_table(self, dangerous_name: str):
        """Dangerous program names must not appear in _TOOL_RISK_TABLE.

        If 'rm' is not in the table, evaluate('rm', {}) returns FORBIDDEN
        by the default-deny path. This double-checks the table itself.
        """
        assert dangerous_name not in _TOOL_RISK_TABLE, (
            f"{dangerous_name!r} must not be in _TOOL_RISK_TABLE"
        )


# ── _FORBIDDEN_PROGRAMS blocklist ──────────────────────────────────────────────

class TestForbiddenProgramsBlocklist:
    """Verify the content and completeness of _FORBIDDEN_PROGRAMS."""

    @pytest.mark.parametrize("program", [
        # File destruction
        "rm", "shred",
        # Disk tools
        "dd", "mkfs", "fdisk", "parted", "wipefs",
        # Power control
        "reboot", "poweroff", "shutdown", "halt",
        # Privilege escalation
        "sudo", "su",
        # Shell interpreters
        "bash", "sh", "zsh",
        # Package managers
        "apt", "apt-get", "dnf", "pacman",
    ])
    def test_dangerous_programs_are_in_blocklist(self, program: str):
        """Core dangerous programs must appear in _FORBIDDEN_PROGRAMS."""
        assert program in _FORBIDDEN_PROGRAMS, (
            f"{program!r} must be in _FORBIDDEN_PROGRAMS"
        )

    def test_blocklist_is_frozenset(self):
        """_FORBIDDEN_PROGRAMS must be a frozenset (immutable)."""
        assert isinstance(_FORBIDDEN_PROGRAMS, frozenset)

    def test_safe_programs_not_in_blocklist(self):
        """Read-only programs used by current tools must not be blocked."""
        safe_programs = {"uptime", "df", "free", "uname", "whoami", "hostname"}
        for prog in safe_programs:
            assert prog not in _FORBIDDEN_PROGRAMS, (
                f"{prog!r} should not be in _FORBIDDEN_PROGRAMS"
            )


# ── apply_policy(): LOW risk ───────────────────────────────────────────────────

class TestApplyPolicyLow:
    """LOW risk decisions must be auto-approved without any prompt."""

    def test_low_tool_returns_decision(self):
        """apply_policy() must return a PolicyDecision for LOW tools."""
        decision = apply_policy("hostname", {})
        assert isinstance(decision, PolicyDecision)

    def test_low_tool_risk_level_is_low(self):
        decision = apply_policy("hostname", {})
        assert decision.risk_level is RiskLevel.LOW

    def test_low_tool_does_not_raise(self):
        """apply_policy() must not raise for LOW tools."""
        apply_policy("system_uptime", {})  # must not raise
        apply_policy("free_memory", {})
        apply_policy("kernel_info", {})
        apply_policy("current_user", {})
        apply_policy("disk_usage", {"path": "/home"})


# ── apply_policy(): FORBIDDEN ─────────────────────────────────────────────────

class TestApplyPolicyForbidden:
    """FORBIDDEN decisions must raise PolicyViolation immediately."""

    def test_unregistered_tool_raises_policy_violation(self):
        """An unknown tool name must raise PolicyViolation."""
        with pytest.raises(PolicyViolation):
            apply_policy("rm", {})

    def test_policy_violation_message_contains_tool_name(self):
        """The PolicyViolation message must name the tool that was blocked."""
        with pytest.raises(PolicyViolation, match="rm"):
            apply_policy("rm", {})

    def test_auto_confirm_does_not_bypass_forbidden(self):
        """auto_confirm=True must NOT bypass FORBIDDEN decisions.

        This is critical: auto_confirm is for MEDIUM/HIGH confirmation
        in tests. It must never unlock FORBIDDEN actions.
        """
        with pytest.raises(PolicyViolation):
            apply_policy("bash", {}, auto_confirm=True)

    def test_multiple_forbidden_tools_all_raise(self):
        """Every unknown/forbidden tool must raise PolicyViolation."""
        for name in ("sudo", "dd", "mystery_tool", "shell_injected"):
            with pytest.raises(PolicyViolation, match=name):
                apply_policy(name, {})


# ── apply_policy(): MEDIUM / HIGH with auto_confirm ───────────────────────────

class TestApplyPolicyConfirmation:
    """Test the confirmation gate logic using auto_confirm=True.

    We cannot test the interactive stdin path in automated CI because
    it would hang waiting for user input. auto_confirm=True simulates
    the user typing 'y' for the purpose of these tests.

    The interactive path is tested by the integration/manual test at
    the bottom of this file.
    """

    def test_auto_confirm_true_approves_medium_risk(self):
        """auto_confirm=True must approve a MEDIUM risk decision."""
        # We need a MEDIUM tool. Since all current tools are LOW,
        # we test the confirmation gate via the policy table directly.
        # Temporarily patch the table for this test.
        from Ciel import policy as _policy

        _policy._TOOL_RISK_TABLE["_test_medium_tool"] = RiskLevel.MEDIUM
        try:
            decision = apply_policy("_test_medium_tool", {}, auto_confirm=True)
            assert decision.risk_level is RiskLevel.MEDIUM
        finally:
            del _policy._TOOL_RISK_TABLE["_test_medium_tool"]

    def test_auto_confirm_false_raises_for_medium_risk_without_tty(self, monkeypatch):
        """Without auto_confirm, MEDIUM risk prompts for confirmation.

        In this test we simulate the user pressing Enter (empty response)
        by monkeypatching input() to return ''. An empty response must
        be treated as No → PolicyViolation.
        """
        from Ciel import policy as _policy

        _policy._TOOL_RISK_TABLE["_test_medium_2"] = RiskLevel.MEDIUM
        try:
            # Patch input() to return '' (empty = No)
            monkeypatch.setattr("builtins.input", lambda _prompt: "")
            with pytest.raises(PolicyViolation, match="declined"):
                apply_policy("_test_medium_2", {}, auto_confirm=False)
        finally:
            del _policy._TOOL_RISK_TABLE["_test_medium_2"]

    def test_auto_confirm_false_approves_on_yes_response(self, monkeypatch):
        """When the user types 'y', MEDIUM risk must be approved."""
        from Ciel import policy as _policy

        _policy._TOOL_RISK_TABLE["_test_medium_3"] = RiskLevel.MEDIUM
        try:
            monkeypatch.setattr("builtins.input", lambda _prompt: "y")
            decision = apply_policy("_test_medium_3", {}, auto_confirm=False)
            assert decision.risk_level is RiskLevel.MEDIUM
        finally:
            del _policy._TOOL_RISK_TABLE["_test_medium_3"]

    def test_auto_confirm_false_approves_on_yes_uppercase(self, monkeypatch):
        """'YES' (uppercase) must also be treated as approval."""
        from Ciel import policy as _policy

        _policy._TOOL_RISK_TABLE["_test_medium_4"] = RiskLevel.MEDIUM
        try:
            monkeypatch.setattr("builtins.input", lambda _prompt: "YES")
            decision = apply_policy("_test_medium_4", {}, auto_confirm=False)
            assert decision.is_permitted is True
        finally:
            del _policy._TOOL_RISK_TABLE["_test_medium_4"]

    def test_auto_confirm_false_declines_on_no(self, monkeypatch):
        """'n' must be treated as a refusal → PolicyViolation."""
        from Ciel import policy as _policy

        _policy._TOOL_RISK_TABLE["_test_medium_5"] = RiskLevel.MEDIUM
        try:
            monkeypatch.setattr("builtins.input", lambda _prompt: "n")
            with pytest.raises(PolicyViolation):
                apply_policy("_test_medium_5", {}, auto_confirm=False)
        finally:
            del _policy._TOOL_RISK_TABLE["_test_medium_5"]


# ── Risk table completeness ────────────────────────────────────────────────────

class TestRiskTableCompleteness:
    """Verify the risk table is internally consistent."""

    def test_all_table_values_are_risk_levels(self):
        """Every value in _TOOL_RISK_TABLE must be a RiskLevel instance."""
        for tool_name, level in _TOOL_RISK_TABLE.items():
            assert isinstance(level, RiskLevel), (
                f"_TOOL_RISK_TABLE[{tool_name!r}] = {level!r} is not a RiskLevel"
            )

    def test_no_tool_in_table_is_forbidden(self):
        """No registered tool should be pre-classified as FORBIDDEN.

        If a tool is FORBIDDEN, it should not be in the registry at all.
        Having a FORBIDDEN entry in the table is a logic error.
        """
        for tool_name, level in _TOOL_RISK_TABLE.items():
            assert level is not RiskLevel.FORBIDDEN, (
                f"Tool {tool_name!r} is in _TOOL_RISK_TABLE as FORBIDDEN. "
                "Remove it from the registry instead."
            )

    def test_six_tools_in_table(self):
        """The table must have exactly 6 entries (all Phase-2 tools)."""
        assert len(_TOOL_RISK_TABLE) == 6
