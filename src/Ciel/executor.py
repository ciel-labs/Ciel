"""Controlled Linux process execution."""

import os
import signal
import subprocess
import time

from Ciel.models import ExecutionRequest, ExecutionResult
from Ciel.validator import ValidationError, validate_request


def _truncate(raw: bytes, limit: int) -> tuple[str, bool]:
    """Decode raw bytes and enforce the byte limit.

    Returns the decoded string and True if the output was cut short.
    Bytes beyond `limit` are discarded. Invalid byte sequences are
    replaced with the Unicode replacement character (U+FFFD) rather
    than raising an exception.
    """
    truncated = len(raw) > limit
    return raw[:limit].decode("utf-8", errors="replace"), truncated


def _kill_group(process: subprocess.Popen) -> None:
    """Send SIGKILL to the entire process group of the given process.

    Because we launch every child with start_new_session=True, the
    child's process group ID (PGID) equals its own PID. Killing the
    group terminates the child and every grandchild it may have spawned.

    If the process has already exited by the time we reach here,
    os.killpg raises ProcessLookupError. We catch it silently because
    the group is already gone — that is the outcome we wanted.
    """
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        # The process group no longer exists. Nothing to do.
        pass


def execute(request: ExecutionRequest) -> ExecutionResult:
    """Execute an approved program without invoking a shell.

    Raises ValidationError before touching subprocess if the request
    fails structural validation. Callers must handle this exception.
    """

    # Validate first. If this raises, no process is ever started.
    validate_request(request)

    start_time = time.monotonic()

    # start_new_session=True places the child in its own session and
    # process group. Its PGID becomes equal to its own PID. This makes
    # it possible to kill the entire group later with a single call to
    # os.killpg(), including any grandchildren the child may spawn.
    process = subprocess.Popen(
        [request.program, *request.arguments],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    try:
        raw_stdout, raw_stderr = process.communicate(timeout=request.timeout)

    except subprocess.TimeoutExpired:
        # Kill the whole process group, not just the direct child.
        _kill_group(process)

        # Drain whatever output was already written before the kill.
        # Without this second communicate() the pipe buffers stay open
        # and Python emits ResourceWarning.
        raw_stdout, raw_stderr = process.communicate()

        duration = time.monotonic() - start_time
        stdout, stdout_truncated = _truncate(raw_stdout, request.max_output_bytes)
        stderr, stderr_truncated = _truncate(raw_stderr, request.max_output_bytes)

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=None,
            timed_out=True,
            duration=duration,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    duration = time.monotonic() - start_time
    stdout, stdout_truncated = _truncate(raw_stdout, request.max_output_bytes)
    stderr, stderr_truncated = _truncate(raw_stderr, request.max_output_bytes)

    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=process.returncode,
        timed_out=False,
        duration=duration,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )
