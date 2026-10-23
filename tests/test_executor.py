"""Tests for src/Ciel/executor.py — controlled process execution.

These tests verify the *executor* contract end-to-end. They spawn real
processes and measure real timing. Each test is independent and cleans
up after itself.

Contract under test
-------------------
execute(request) → ExecutionResult   on normal completion (any exit code)
execute(request) → ExecutionResult   with timed_out=True on timeout
execute(request) → raises ValidationError for invalid requests

Key properties verified
-----------------------
- stdout / stderr are captured correctly
- exit_code reflects the real process exit code
- Timed-out processes return timed_out=True and exit_code=None
- Output beyond max_output_bytes is truncated; flags are set
- Killing a timed-out process also kills its grandchildren (group kill)
- Invalid requests raise ValidationError before any process is started

Why subprocess-based tests are acceptable here
-----------------------------------------------
These tests are integration tests — they test the real executor against
real system binaries. This is intentional: we *want* to verify that
signal delivery, pipe draining, and process-group teardown behave
correctly on the target OS. Unit tests with mocked subprocesses would
give us false confidence in the most security-critical layer.

Slow tests are marked @pytest.mark.slow so CI can skip them with -m "not slow".
"""

import subprocess
import time

import pytest

from Ciel.executor import execute
from Ciel.models import ExecutionRequest
from Ciel.validator import ValidationError


# ── Normal execution ───────────────────────────────────────────────────────────

class TestNormalExecution:
    """Tests for processes that complete successfully within the timeout."""

    def test_stdout_captured(self):
        """echo writes to stdout; we must receive it verbatim."""
        request = ExecutionRequest(program="echo", arguments=["hello world"])
        result = execute(request)
        assert result.stdout == "hello world\n"

    def test_stderr_captured(self):
        """Output written to stderr must appear in result.stderr, not stdout."""
        request = ExecutionRequest(
            program="python3",
            arguments=["-c", "import sys; sys.stderr.write('err output\\n')"],
        )
        result = execute(request)
        assert "err output" in result.stderr
        assert result.stdout == ""

    def test_exit_code_zero_on_success(self):
        """A process that exits normally must have exit_code == 0."""
        result = execute(ExecutionRequest(program="true", arguments=[]))
        assert result.exit_code == 0

    def test_exit_code_nonzero_on_failure(self):
        """A process that exits with a non-zero code must surface that code.

        'false' is a standard POSIX tool that always exits with code 1.
        We must NOT treat a non-zero exit code as a timeout or an error
        in the executor itself — exit code is information, not an exception.
        """
        result = execute(ExecutionRequest(program="false", arguments=[]))
        assert result.exit_code == 1
        assert result.timed_out is False

    def test_timed_out_false_on_success(self):
        """A process that finishes within the timeout must not be marked timed_out."""
        result = execute(ExecutionRequest(program="echo", arguments=["ok"]))
        assert result.timed_out is False

    def test_duration_is_positive(self):
        """Execution duration must be a positive float."""
        result = execute(ExecutionRequest(program="echo", arguments=["hi"]))
        assert result.duration > 0.0

    def test_truncation_flags_false_on_small_output(self):
        """No truncation flags must be set when output fits within the limit."""
        result = execute(ExecutionRequest(program="echo", arguments=["hi"]))
        assert result.stdout_truncated is False
        assert result.stderr_truncated is False

    def test_absolute_path_accepted(self):
        """The executor must work with an absolute path, not just a bare name."""
        result = execute(ExecutionRequest(program="/bin/echo", arguments=["abs"]))
        assert result.stdout.strip() == "abs"
        assert result.exit_code == 0


# ── Output truncation ──────────────────────────────────────────────────────────

class TestOutputTruncation:
    """Tests for the bounded-output (_truncate) behaviour."""

    def test_stdout_truncated_when_over_limit(self):
        """stdout longer than max_output_bytes must be cut to the limit."""
        request = ExecutionRequest(
            program="python3",
            arguments=["-c", "print('A' * 10_000)"],
            max_output_bytes=100,
        )
        result = execute(request)
        # The decoded string length must be exactly the byte limit
        # (all chars are ASCII, so 1 char == 1 byte).
        assert len(result.stdout) == 100
        assert result.stdout_truncated is True

    def test_stderr_truncated_when_over_limit(self):
        """stderr longer than max_output_bytes must also be truncated."""
        request = ExecutionRequest(
            program="python3",
            arguments=["-c", "import sys; sys.stderr.write('B' * 10_000)"],
            max_output_bytes=50,
        )
        result = execute(request)
        assert len(result.stderr) == 50
        assert result.stderr_truncated is True

    def test_stdout_truncated_false_at_exact_limit(self):
        """Output that fits exactly within the limit must NOT be flagged."""
        # 'echo hi' produces "hi\n" = 3 bytes.
        request = ExecutionRequest(
            program="echo",
            arguments=["hi"],
            max_output_bytes=3,
        )
        result = execute(request)
        assert result.stdout == "hi\n"
        assert result.stdout_truncated is False


# ── Timeout enforcement ────────────────────────────────────────────────────────

class TestTimeoutEnforcement:
    """Tests for timeout detection and process termination."""

    @pytest.mark.slow
    def test_timed_out_set_to_true(self):
        """A process that runs past its timeout must have timed_out=True."""
        request = ExecutionRequest(program="sleep", arguments=["60"], timeout=1.0)
        result = execute(request)
        assert result.timed_out is True

    @pytest.mark.slow
    def test_exit_code_none_on_timeout(self):
        """A timed-out process was killed, so there is no exit code.

        exit_code=None distinguishes "we killed it" from "it exited with 0"
        or "it exited with some non-zero code". Callers must handle None.
        """
        request = ExecutionRequest(program="sleep", arguments=["60"], timeout=1.0)
        result = execute(request)
        assert result.exit_code is None

    @pytest.mark.slow
    def test_duration_reflects_real_elapsed_time(self):
        """Duration must be at least as long as the timeout.

        We allow a generous upper bound of timeout + 5s to absorb CI
        scheduling latency without making the test flaky.
        """
        timeout = 1.0
        request = ExecutionRequest(program="sleep", arguments=["60"], timeout=timeout)
        result = execute(request)
        assert result.duration >= timeout
        assert result.duration < timeout + 5.0

    @pytest.mark.slow
    def test_partial_stdout_captured_after_timeout(self):
        """Output written before the timeout must still be captured.

        When we kill a process, it may have already written bytes to the
        pipe buffer. We must drain those bytes after the kill so they are
        not lost and so we do not leave an open pipe handle.
        """
        request = ExecutionRequest(
            program="python3",
            arguments=["-c", (
                "import sys, time; "
                "sys.stdout.write('before timeout\\n'); "
                "sys.stdout.flush(); "
                "time.sleep(60)"
            )],
            timeout=1.5,
        )
        result = execute(request)
        assert result.timed_out is True
        assert "before timeout" in result.stdout


# ── Process-group kill (grandchild termination) ────────────────────────────────

class TestProcessGroupKill:
    """Tests that the executor kills entire process groups, not just direct children.

    Security motivation: without group kill, a malicious or buggy child
    process could spawn long-lived grandchildren before being killed.
    Those grandchildren would become orphans — still consuming resources
    and potentially doing harmful work — even though the executor
    'stopped' the operation.

    We use start_new_session=True so every child's PGID equals its own
    PID. os.killpg(pgid, SIGKILL) then kills every member of that group.
    """

    @pytest.mark.slow
    def test_grandchild_processes_are_killed(self):
        """After a timeout, no grandchild 'sleep 999' processes must survive.

        The child Python process spawns 'sleep 999' and then sleeps itself.
        When our timeout fires, we kill the entire process group. If the
        implementation is correct, 'sleep 999' dies with its parent.
        """
        # Kill any pre-existing 'sleep 999' so our before/after diff is clean.
        subprocess.run(["pkill", "-9", "-f", "sleep 999"], capture_output=True)
        time.sleep(0.3)

        before = set(
            subprocess.run(
                ["pgrep", "-f", "sleep 999"], capture_output=True, text=True
            ).stdout.split()
        )

        execute(ExecutionRequest(
            program="python3",
            arguments=["-c", (
                "import subprocess, time; "
                "subprocess.Popen(['sleep', '999']); "
                "time.sleep(60)"
            )],
            timeout=1.5,
        ))

        # Give the kernel a moment to reap zombie entries.
        time.sleep(1.0)

        after = set(
            subprocess.run(
                ["pgrep", "-f", "sleep 999"], capture_output=True, text=True
            ).stdout.split()
        )

        new_orphans = after - before
        assert not new_orphans, (
            f"Orphan grandchild PIDs survived after group kill: {new_orphans}"
        )


# ── Validation integration ─────────────────────────────────────────────────────

class TestValidationIntegration:
    """Tests that the executor propagates ValidationError for bad requests.

    The executor calls validate_request() before touching subprocess.
    If validation fails, no process is ever started. These tests verify
    that the executor is not bypassing the validator.
    """

    def test_empty_program_raises_validation_error(self):
        """An empty program string must raise ValidationError, not crash."""
        with pytest.raises(ValidationError):
            execute(ExecutionRequest(program="", arguments=[]))

    def test_missing_binary_raises_validation_error(self):
        """A binary that does not exist must raise ValidationError."""
        with pytest.raises(ValidationError):
            execute(ExecutionRequest(
                program="binary_that_does_not_exist_ciel_pytest",
                arguments=[],
            ))

    def test_null_byte_in_program_raises_validation_error(self):
        """A null byte in the program name must raise ValidationError."""
        with pytest.raises(ValidationError):
            execute(ExecutionRequest(program="echo\x00injected", arguments=[]))

    def test_null_byte_in_argument_raises_validation_error(self):
        """A null byte in an argument must raise ValidationError."""
        with pytest.raises(ValidationError):
            execute(ExecutionRequest(program="echo", arguments=["bad\x00arg"]))

    def test_valid_request_executes_after_validation(self):
        """After validation passes, the process must actually run."""
        result = execute(ExecutionRequest(program="echo", arguments=["validation ok"]))
        assert result.stdout.strip() == "validation ok"
        assert result.exit_code == 0
