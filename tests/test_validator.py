"""Tests for src/Ciel/validator.py — structural input validation.

These tests verify the *validator* contract only. They do not spawn any
processes. Every test is fast, deterministic, and side-effect-free.

Contract under test
-------------------
validate_request(request) → None          if the request is structurally valid
validate_request(request) → ValidationError  otherwise

The validator answers one question: "Is this request coherent enough to
attempt?" It is NOT responsible for deciding whether an action is
*permitted* (that is the policy engine's job, built in Step 4).
"""

import pytest

from Ciel.models import ExecutionRequest
from Ciel.validator import ValidationError, validate_request


# ── Helpers ───────────────────────────────────────────────────────────────────

def assert_valid(request: ExecutionRequest) -> None:
    """Assert that validate_request raises nothing."""
    validate_request(request)  # must not raise


def assert_invalid(request: ExecutionRequest, fragment: str = "") -> None:
    """Assert that validate_request raises ValidationError.

    If `fragment` is given, the error message must contain it (case-sensitive).
    This lets tests assert on the *reason* without coupling to exact wording.
    """
    with pytest.raises(ValidationError) as exc_info:
        validate_request(request)
    if fragment:
        assert fragment in str(exc_info.value), (
            f"Expected fragment {fragment!r} in error message, "
            f"got: {str(exc_info.value)!r}"
        )


# ── Valid requests ─────────────────────────────────────────────────────────────

class TestValidRequests:
    """Requests that must pass validation without raising."""

    def test_bare_name_resolved_via_path(self):
        """'echo' is on PATH and executable — must be accepted."""
        assert_valid(ExecutionRequest(program="echo", arguments=["hi"]))

    def test_absolute_path(self):
        """An absolute path to a real executable must be accepted."""
        assert_valid(ExecutionRequest(program="/bin/ls", arguments=[]))

    def test_arguments_with_spaces(self):
        """Spaces inside a single argument string are fine.

        This is safe because we never pass arguments through a shell.
        The space is literal data passed to the subprocess, not a
        word-splitting delimiter.
        """
        assert_valid(ExecutionRequest(program="echo", arguments=["hello world", "foo"]))

    def test_empty_arguments_list(self):
        """An empty argument list is valid — many tools take no arguments."""
        assert_valid(ExecutionRequest(program="true", arguments=[]))

    def test_default_timeout_is_accepted(self):
        """The default timeout (10.0s) must be within bounds."""
        assert_valid(ExecutionRequest(program="echo", arguments=[]))

    def test_explicit_timeout_within_ceiling(self):
        """A timeout of 60s is within the 300s ceiling."""
        assert_valid(ExecutionRequest(program="echo", arguments=[], timeout=60.0))

    def test_max_output_bytes_at_ceiling_minus_one(self):
        """max_output_bytes one byte below the ceiling must be accepted."""
        assert_valid(ExecutionRequest(
            program="echo",
            arguments=[],
            max_output_bytes=100 * 1024 * 1024 - 1,
        ))


# ── Invalid: program ───────────────────────────────────────────────────────────

class TestInvalidProgram:
    """Requests that must be rejected because the program field is invalid."""

    def test_empty_program_string(self):
        """An empty program name is meaningless — must be rejected."""
        assert_invalid(
            ExecutionRequest(program="", arguments=[]),
            fragment="empty",
        )

    def test_null_byte_in_program(self):
        """A null byte in the program string is a classic injection vector.

        The kernel's execve() terminates C strings at \\x00. A program name
        like 'safe\\x00rm -rf /' would be silently truncated to 'safe' at the
        kernel level, but could mislead logging and auditing at the Python
        level. We reject unconditionally.
        """
        assert_invalid(
            ExecutionRequest(program="ls\x00injected", arguments=[]),
            fragment="null byte",
        )

    def test_nonexistent_binary(self):
        """A binary that does not exist on PATH must be rejected before exec."""
        assert_invalid(
            ExecutionRequest(
                program="this_binary_absolutely_does_not_exist_ciel_test",
                arguments=[],
            ),
            fragment="not found",
        )


# ── Invalid: arguments ─────────────────────────────────────────────────────────

class TestInvalidArguments:
    """Requests that must be rejected because an argument is invalid."""

    def test_null_byte_in_argument(self):
        """A null byte inside an argument is a kernel-level truncation risk.

        The same reasoning as for the program name applies here.
        """
        assert_invalid(
            ExecutionRequest(program="echo", arguments=["safe\x00injected"]),
            fragment="null byte",
        )

    def test_null_byte_in_second_argument(self):
        """The check must scan every argument, not just the first."""
        assert_invalid(
            ExecutionRequest(program="echo", arguments=["clean", "bad\x00arg"]),
            fragment="null byte",
        )


# ── Invalid: timeout bounds ────────────────────────────────────────────────────

class TestInvalidTimeout:
    """Requests with out-of-range timeout values must be rejected."""

    def test_timeout_zero(self):
        """A zero timeout would expire instantly — meaningless."""
        assert_invalid(
            ExecutionRequest(program="echo", arguments=[], timeout=0),
            fragment="timeout",
        )

    def test_timeout_negative(self):
        """A negative timeout is logically impossible."""
        assert_invalid(
            ExecutionRequest(program="echo", arguments=[], timeout=-1.0),
            fragment="timeout",
        )

    def test_timeout_exactly_at_ceiling(self):
        """A timeout of exactly 300s must be accepted (inclusive ceiling)."""
        assert_valid(ExecutionRequest(program="echo", arguments=[], timeout=300.0))

    def test_timeout_just_above_ceiling(self):
        """A timeout of 300.001s exceeds the ceiling and must be rejected."""
        assert_invalid(
            ExecutionRequest(program="echo", arguments=[], timeout=300.001),
            fragment="exceeds",
        )

    def test_timeout_far_above_ceiling(self):
        """A very large timeout (e.g., 1 hour) must be rejected."""
        assert_invalid(
            ExecutionRequest(program="echo", arguments=[], timeout=3600.0),
            fragment="exceeds",
        )


# ── Invalid: max_output_bytes bounds ──────────────────────────────────────────

class TestInvalidMaxOutputBytes:
    """Requests with out-of-range max_output_bytes must be rejected."""

    def test_max_output_bytes_zero(self):
        """Zero bytes of output allowed is meaningless."""
        assert_invalid(
            ExecutionRequest(program="echo", arguments=[], max_output_bytes=0),
            fragment="max_output_bytes",
        )

    def test_max_output_bytes_negative(self):
        """A negative byte limit is logically impossible."""
        assert_invalid(
            ExecutionRequest(program="echo", arguments=[], max_output_bytes=-1),
            fragment="max_output_bytes",
        )

    def test_max_output_bytes_above_ceiling(self):
        """200 MB exceeds the 100 MB ceiling — must be rejected."""
        assert_invalid(
            ExecutionRequest(
                program="echo",
                arguments=[],
                max_output_bytes=200 * 1024 * 1024,
            ),
            fragment="exceeds",
        )

    def test_max_output_bytes_exactly_at_ceiling(self):
        """Exactly 100 MB is the ceiling and must be accepted."""
        assert_valid(ExecutionRequest(
            program="echo",
            arguments=[],
            max_output_bytes=100 * 1024 * 1024,
        ))
