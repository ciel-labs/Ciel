"""Input validation for ExecutionRequest.

This module validates the structural and technical correctness of an
ExecutionRequest before it reaches the executor. It is NOT a policy
engine — it does not decide whether an action is permitted. It decides
whether the request is coherent enough to attempt at all.

Separation of concerns
----------------------
Validator  →  Is this request structurally valid?
Policy     →  Is this request permitted? (future step)
Executor   →  Carry out the approved, validated request.
"""

import os
import shutil

from Ciel.models import ExecutionRequest

# Hard upper bounds. These are not configurable at runtime because they
# are safety ceilings, not tuning parameters.
_MAX_TIMEOUT_SECONDS: float = 300.0       # 5 minutes
_MAX_OUTPUT_BYTES: int = 100 * 1024 * 1024  # 100 MB


class ValidationError(Exception):
    """Raised when an ExecutionRequest fails structural validation.

    The message always describes the exact reason so callers can log it
    clearly and users can understand why execution was refused.
    """


def _contains_null_byte(value: str) -> bool:
    """Return True if the string contains a null byte (\\x00).

    Null bytes terminate C-level strings early. A value like
    'safe\\x00rm -rf /' looks like one Python string but the kernel
    sees only 'safe'. We refuse any such value unconditionally.
    """
    return "\x00" in value


def validate_request(request: ExecutionRequest) -> None:
    """Validate an ExecutionRequest before execution.

    Raises ValidationError with a descriptive message if any check
    fails. Returns None silently if the request is valid.

    Checks performed
    ----------------
    1. program is not empty.
    2. program contains no null bytes.
    3. program resolves to an existing, executable file.
    4. No argument contains a null byte.
    5. timeout is positive and within the allowed ceiling.
    6. max_output_bytes is positive and within the allowed ceiling.
    """

    # --- program checks ---

    if not request.program:
        raise ValidationError("program must not be empty")

    if _contains_null_byte(request.program):
        raise ValidationError(
            "program contains a null byte, which is never valid"
        )

    # Resolve the program to an absolute path. shutil.which() searches
    # PATH for bare names and returns None if not found. os.path.isfile
    # and os.access cover absolute paths that exist but aren't executable.
    resolved = shutil.which(request.program)

    if resolved is None:
        raise ValidationError(
            f"program not found or not executable: {request.program!r}"
        )

    if not os.access(resolved, os.X_OK):
        raise ValidationError(
            f"program exists but is not executable: {resolved!r}"
        )

    # --- argument checks ---

    for index, arg in enumerate(request.arguments):
        if not isinstance(arg, str):
            raise ValidationError(
                f"argument at index {index} is not a string: {type(arg).__name__!r}"
            )
        if _contains_null_byte(arg):
            raise ValidationError(
                f"argument at index {index} contains a null byte"
            )

    # --- numeric bound checks ---

    if request.timeout <= 0:
        raise ValidationError(
            f"timeout must be positive, got {request.timeout}"
        )

    if request.timeout > _MAX_TIMEOUT_SECONDS:
        raise ValidationError(
            f"timeout {request.timeout}s exceeds the maximum of "
            f"{_MAX_TIMEOUT_SECONDS}s"
        )

    if request.max_output_bytes <= 0:
        raise ValidationError(
            f"max_output_bytes must be positive, got {request.max_output_bytes}"
        )

    if request.max_output_bytes > _MAX_OUTPUT_BYTES:
        raise ValidationError(
            f"max_output_bytes {request.max_output_bytes} exceeds the "
            f"maximum of {_MAX_OUTPUT_BYTES}"
        )
