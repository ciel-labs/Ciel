"""Step 2H verification tests — input validation.

Run from the project root:  python3 test_2h.py
"""

import sys

sys.path.insert(0, "src")

from Ciel.models import ExecutionRequest
from Ciel.executor import execute
from Ciel.validator import ValidationError, validate_request

passed = 0
failed = 0


def check(label: str, condition: bool) -> None:
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {label}")


def expect_validation_error(label: str, request: ExecutionRequest) -> None:
    """Assert that validate_request raises ValidationError."""
    try:
        validate_request(request)
        check(label, False)
        print(f"         (expected ValidationError but got none)")
    except ValidationError as exc:
        check(label, True)
        print(f"         reason: {exc}")


def expect_valid(label: str, request: ExecutionRequest) -> None:
    """Assert that validate_request passes without raising."""
    try:
        validate_request(request)
        check(label, True)
    except ValidationError as exc:
        check(label, False)
        print(f"         unexpected error: {exc}")


# ── Valid requests ─────────────────────────────────────────────────────────────

print("=== Valid requests (should pass) ===")
expect_valid("bare name resolved via PATH: 'echo'",
             ExecutionRequest(program="echo", arguments=["hi"]))

expect_valid("absolute path: '/bin/ls'",
             ExecutionRequest(program="/bin/ls", arguments=[]))

expect_valid("arguments with spaces are fine",
             ExecutionRequest(program="echo", arguments=["hello world", "foo"]))

# ── Invalid: program ───────────────────────────────────────────────────────────

print()
print("=== Invalid program (should fail) ===")
expect_validation_error("empty program string",
                         ExecutionRequest(program="", arguments=[]))

expect_validation_error("null byte in program",
                         ExecutionRequest(program="ls\x00injected", arguments=[]))

expect_validation_error("nonexistent binary",
                         ExecutionRequest(program="this_binary_does_not_exist_ciel", arguments=[]))

# ── Invalid: arguments ─────────────────────────────────────────────────────────

print()
print("=== Invalid arguments (should fail) ===")
expect_validation_error("null byte in argument",
                         ExecutionRequest(program="echo", arguments=["safe\x00injected"]))

# ── Invalid: numeric bounds ────────────────────────────────────────────────────

print()
print("=== Invalid numeric bounds (should fail) ===")
expect_validation_error("timeout = 0",
                         ExecutionRequest(program="echo", arguments=[], timeout=0))

expect_validation_error("timeout negative",
                         ExecutionRequest(program="echo", arguments=[], timeout=-1.0))

expect_validation_error("timeout exceeds ceiling (301s)",
                         ExecutionRequest(program="echo", arguments=[], timeout=301.0))

expect_validation_error("max_output_bytes = 0",
                         ExecutionRequest(program="echo", arguments=[], max_output_bytes=0))

expect_validation_error("max_output_bytes exceeds ceiling",
                         ExecutionRequest(program="echo", arguments=[], max_output_bytes=200 * 1024 * 1024))

# ── Integration: executor refuses invalid requests ─────────────────────────────

print()
print("=== Executor integration (ValidationError propagates) ===")
try:
    execute(ExecutionRequest(program="", arguments=[]))
    check("executor raises ValidationError for empty program", False)
except ValidationError:
    check("executor raises ValidationError for empty program", True)

try:
    execute(ExecutionRequest(program="nonexistent_ciel_binary", arguments=[]))
    check("executor raises ValidationError for missing binary", False)
except ValidationError:
    check("executor raises ValidationError for missing binary", True)

# ── Sanity: valid request still executes ──────────────────────────────────────

print()
print("=== Sanity: valid request still executes correctly ===")
r = execute(ExecutionRequest(program="echo", arguments=["step 2h works"]))
check("echo executes and returns correct output", r.stdout.strip() == "step 2h works")
check("exit_code is 0", r.exit_code == 0)

# ── Summary ───────────────────────────────────────────────────────────────────

print()
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
