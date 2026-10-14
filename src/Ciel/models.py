"""Data models used by the Linux agent."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionRequest:
    """Describes an approved process execution request."""

    program: str
    arguments: list[str]
    timeout: float = 10.0
    max_output_bytes: int = 262_144  # 256 KB per stream


@dataclass(frozen=True)
class ExecutionResult:
    """Describes the result of a process execution."""

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    duration: float
    stdout_truncated: bool = False
    stderr_truncated: bool = False
