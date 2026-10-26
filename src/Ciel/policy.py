"""Policy engine: the independent authority that decides what Ciel may do.

Architecture position
---------------------
The LLM proposes a ToolCall.
The ToolRegistry validates its structure.
THIS MODULE decides whether execution is permitted.
The Executor carries out permitted actions.

The policy engine is the only component that answers:
  "Is this action permitted for Ciel to run right now?"

It operates completely independently of the LLM. The LLM cannot
influence policy decisions by constructing clever inputs, because the
policy reads from a hard-coded table — not from the LLM's output.

Separation of concerns
----------------------
ToolRegistry  →  Is this a known, structurally valid tool call?
Validator     →  Is this OS invocation safe to attempt?
PolicyEngine  →  Is this action permitted? (this module)
Executor      →  Carry out the approved, validated request.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum


# ── Risk levels ────────────────────────────────────────────────────────────────

class RiskLevel(Enum):
    """The four risk tiers used by the policy engine.

    Ordered from least to most severe. The integer values allow
    direct comparison: LOW < MEDIUM < HIGH < FORBIDDEN.

    LOW
        Read-only operations with no side effects on system state.
        Examples: uptime, df, free, uname, whoami, hostname.
        Gate: auto-approved, no human interaction required.

    MEDIUM
        Operations that read sensitive data or make non-critical
        state changes. Examples: reading /etc/hosts, listing processes,
        checking open ports, modifying user config files.
        Gate: human confirmation required (Y/N prompt).

    HIGH
        Operations that modify system state or touch critical paths.
        Examples: writing system config files, restarting services,
        installing packages, changing file permissions.
        Gate: human confirmation required with an explicit warning.

    FORBIDDEN
        Destructive or irreversible operations that Ciel must never
        perform in V1, regardless of what the user or LLM requests.
        Examples: rm, dd, mkfs, reboot, poweroff, shred.
        Gate: hard reject. No confirmation is offered. Ever.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    FORBIDDEN = 4


# ── Policy decision ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyDecision:
    """The outcome of evaluating one tool call against the policy.

    Produced by evaluate(). Consumed by apply_policy() which decides
    whether to auto-approve, prompt the user, or raise PolicyViolation.

    Attributes
    ----------
    risk_level :
        The risk tier assigned to this tool call.
    reason :
        A human-readable explanation of why this risk level was assigned.
        Used in confirmation prompts and audit logs.
    requires_confirmation :
        True when the user must explicitly approve before execution.
        Set automatically based on risk_level:
        LOW → False, MEDIUM → True, HIGH → True, FORBIDDEN → False
        (FORBIDDEN is False because it is never presented for confirmation —
        it is rejected outright before the user is asked.)
    """

    risk_level: RiskLevel
    reason: str
    requires_confirmation: bool

    @property
    def is_permitted(self) -> bool:
        """True if execution is not categorically blocked.

        FORBIDDEN decisions are never permitted, regardless of user input.
        All other decisions may proceed (subject to confirmation).
        """
        return self.risk_level is not RiskLevel.FORBIDDEN

    @classmethod
    def _make(cls, risk_level: RiskLevel, reason: str) -> PolicyDecision:
        """Internal factory that sets requires_confirmation automatically."""
        requires = risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
        return cls(
            risk_level=risk_level,
            reason=reason,
            requires_confirmation=requires,
        )


# ── Exceptions ─────────────────────────────────────────────────────────────────

class PolicyViolation(Exception):
    """Raised when a FORBIDDEN action is attempted.

    This exception is raised by apply_policy() and must be caught by
    the agent loop. It must never be silently swallowed. Every occurrence
    should be logged in the audit record.

    The message always describes the exact tool and reason so the audit
    log is useful without additional context.
    """


# ── Risk table ─────────────────────────────────────────────────────────────────

# Maps every registered tool name to its risk level.
#
# Design rule: this table must be exhaustive. Any tool not listed here
# will be classified as FORBIDDEN by evaluate(). This is intentional —
# default-deny is the only safe default for a security-critical system.
#
# When a new tool is added to definitions.py, it MUST also be added here
# with a deliberate, justified risk level. There is no automatic inheritance.
_TOOL_RISK_TABLE: dict[str, RiskLevel] = {
    # ── Read-only diagnostic tools (Phase 2, all LOW) ──────────────────────
    # These tools run standard POSIX utilities in read-only mode.
    # They cannot modify system state in any way.
    "system_uptime": RiskLevel.LOW,   # uptime -p: reads /proc/uptime
    "disk_usage":    RiskLevel.LOW,   # df --human-readable: reads mount info
    "free_memory":   RiskLevel.LOW,   # free -h: reads /proc/meminfo
    "kernel_info":   RiskLevel.LOW,   # uname -r: reads kernel version string
    "current_user":  RiskLevel.LOW,   # whoami: reads /etc/passwd / getuid()
    "hostname":      RiskLevel.LOW,   # hostname: reads kernel hostname
    #
    # Future tools will be added here in later phases, each with an
    # explicit, justified risk classification.
}


# ── Forbidden program blocklist ────────────────────────────────────────────────

# Defense-in-depth: even if a destructive binary somehow got registered
# as a tool (e.g., a bug in definitions.py or a future mistake), the
# policy engine will still block it here.
#
# This list uses bare program names (not absolute paths) because shutil.which()
# is used for resolution in the validator. We block the name, not a specific
# path, to catch both /bin/rm and /usr/bin/rm.
_FORBIDDEN_PROGRAMS: frozenset[str] = frozenset({
    # File destruction
    "rm", "rmdir", "shred", "wipe", "srm",
    # Disk and partition tools
    "dd", "mkfs", "mke2fs", "mkswap", "fdisk", "parted",
    "gdisk", "wipefs", "blkdiscard",
    # System shutdown / power control
    "reboot", "poweroff", "shutdown", "halt", "init",
    # Process termination (broad kill)
    "kill", "pkill", "killall",
    # Privilege escalation
    "sudo", "su", "doas",
    # Package management (modifies system state — reserved for HIGH in later phases)
    "apt", "apt-get", "dnf", "yum", "pacman", "zypper", "pip",
    # Shell interpreters (never execute raw shell strings)
    "bash", "sh", "zsh", "fish", "dash", "ksh",
})


# ── Core evaluation function ───────────────────────────────────────────────────

def evaluate(tool_name: str, arguments: dict[str, str]) -> PolicyDecision:
    """Evaluate a tool call and return a PolicyDecision.

    This is the main entry point for policy checks. Call this before
    building an ExecutionRequest. Never skip this step.

    Decision logic (in priority order)
    -----------------------------------
    1. If tool_name is not in _TOOL_RISK_TABLE → FORBIDDEN.
       Unknown tools are never permitted. Default-deny.

    2. Look up the risk level from the table.

    3. [Future] Argument-level checks (e.g., sensitive path detection)
       will be added here in Phase 3D (deeper path policy). For now,
       risk is classified entirely at the tool level.

    Parameters
    ----------
    tool_name :
        The name of the tool being requested (from ToolCall.tool_name).
    arguments :
        The validated arguments dict (from ToolCall.arguments).

    Returns
    -------
    PolicyDecision
        Always returns a decision — never raises. Callers must check
        decision.is_permitted and decision.requires_confirmation.
    """
    # 1. Default-deny: unknown tool names are always FORBIDDEN.
    if tool_name not in _TOOL_RISK_TABLE:
        return PolicyDecision._make(
            RiskLevel.FORBIDDEN,
            reason=(
                f"Tool {tool_name!r} is not in the policy table. "
                "Only explicitly registered and approved tools may be executed. "
                "This is a default-deny policy."
            ),
        )

    risk_level = _TOOL_RISK_TABLE[tool_name]

    # 2. Return the classified decision with a descriptive reason.
    reasons: dict[RiskLevel, str] = {
        RiskLevel.LOW: (
            f"Tool {tool_name!r} is a read-only diagnostic — "
            "it cannot modify system state."
        ),
        RiskLevel.MEDIUM: (
            f"Tool {tool_name!r} reads sensitive data or makes "
            "non-critical state changes. Human confirmation required."
        ),
        RiskLevel.HIGH: (
            f"Tool {tool_name!r} modifies system state. "
            "Explicit human confirmation required."
        ),
        RiskLevel.FORBIDDEN: (
            f"Tool {tool_name!r} is classified as FORBIDDEN and "
            "may never be executed by Ciel."
        ),
    }

    return PolicyDecision._make(risk_level, reason=reasons[risk_level])


# ── Human confirmation gate ────────────────────────────────────────────────────

def request_confirmation(
    decision: PolicyDecision,
    tool_name: str,
    arguments: dict[str, str],
) -> bool:
    """Display a risk summary and prompt the user for explicit approval.

    Called by apply_policy() for MEDIUM and HIGH risk decisions.
    Never called for LOW (auto-approved) or FORBIDDEN (hard rejected).

    Safe-default design
    -------------------
    Any response that is not an explicit affirmative ('y' or 'yes',
    case-insensitive) is treated as a refusal. This includes:
    - Empty input (just pressing Enter)
    - EOF (Ctrl-D)
    - Anything other than 'y' / 'yes'

    This means the user must actively choose to proceed. Inaction,
    confusion, or accidental Enter presses all default to No.

    Parameters
    ----------
    decision :
        The PolicyDecision that triggered this prompt.
    tool_name :
        Name of the tool being confirmed (used in the display).
    arguments :
        The arguments being passed to the tool (displayed for review).

    Returns
    -------
    bool
        True if and only if the user explicitly typed 'y' or 'yes'.
    """
    risk_label = decision.risk_level.name  # e.g., "MEDIUM"

    # Write to stderr so the confirmation prompt is not captured if
    # stdout is being redirected or piped.
    print(file=sys.stderr)
    print("─" * 60, file=sys.stderr)
    print(f"  ⚠  ACTION REQUIRES CONFIRMATION  [{risk_label}]", file=sys.stderr)
    print("─" * 60, file=sys.stderr)
    print(f"  Tool     : {tool_name}", file=sys.stderr)
    if arguments:
        for key, value in sorted(arguments.items()):
            print(f"  {key:<8} : {value}", file=sys.stderr)
    print(f"  Reason   : {decision.reason}", file=sys.stderr)
    print("─" * 60, file=sys.stderr)

    try:
        response = input("  Proceed? [y/N]: ").strip().lower()
    except EOFError:
        # Non-interactive environment (e.g., piped input). Default to No.
        print("  (no input — defaulting to No)", file=sys.stderr)
        return False

    approved = response in ("y", "yes")
    status = "Approved." if approved else "Declined."
    print(f"  {status}", file=sys.stderr)
    print("─" * 60, file=sys.stderr)
    print(file=sys.stderr)
    return approved


# ── apply_policy: the single function the agent loop calls ────────────────────

def apply_policy(
    tool_name: str,
    arguments: dict[str, str],
    *,
    auto_confirm: bool = False,
) -> PolicyDecision:
    """Evaluate policy and handle the confirmation gate in one call.

    This is the function the agent loop uses. It combines evaluate() and
    request_confirmation() into a single, linear decision sequence:

      1. Evaluate the tool call → PolicyDecision
      2. If FORBIDDEN → raise PolicyViolation (never ask for confirmation)
      3. If LOW → return decision (auto-approved, no prompt)
      4. If MEDIUM or HIGH → prompt user → if declined, raise PolicyViolation

    Parameters
    ----------
    tool_name :
        Tool name from the LLM's ToolCall.
    arguments :
        Validated arguments dict from the ToolCall.
    auto_confirm : bool, keyword-only
        If True, MEDIUM and HIGH decisions are approved without prompting.
        Use ONLY in non-interactive automated tests. Never set this True
        in production code.

    Returns
    -------
    PolicyDecision
        The approved decision. Only returned if execution is permitted.

    Raises
    ------
    PolicyViolation
        If the tool is FORBIDDEN, or if the user declines confirmation.
    """
    decision = evaluate(tool_name, arguments)

    if not decision.is_permitted:
        raise PolicyViolation(
            f"Execution of {tool_name!r} is FORBIDDEN. "
            f"Reason: {decision.reason}"
        )

    if decision.requires_confirmation and not auto_confirm:
        approved = request_confirmation(decision, tool_name, arguments)
        if not approved:
            raise PolicyViolation(
                f"Execution of {tool_name!r} was declined by the user. "
                f"Risk level was {decision.risk_level.name}."
            )

    return decision
