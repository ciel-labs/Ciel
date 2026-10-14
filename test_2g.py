"""Step 2G verification tests — run from project root after pip install -e ."""

import os
import subprocess
import sys
import time

sys.path.insert(0, "src")

from Ciel.models import ExecutionRequest
from Ciel.executor import execute

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


print("=== Test 1: Normal execution ===")
r = execute(ExecutionRequest(program="echo", arguments=["hello world"]))
check("stdout is 'hello world\\n'", r.stdout == "hello world\n")
check("exit_code is 0", r.exit_code == 0)
check("timed_out is False", r.timed_out is False)
check("stdout_truncated is False", r.stdout_truncated is False)

print()
print("=== Test 2: Output truncation ===")
r2 = execute(ExecutionRequest(
    program="python3",
    arguments=["-c", "print('A' * 10000)"],
    max_output_bytes=100,
))
check("stdout length equals limit", len(r2.stdout) == 100)
check("stdout_truncated is True", r2.stdout_truncated is True)
check("timed_out is False", r2.timed_out is False)

print()
print("=== Test 3: Timeout ===")
r3 = execute(ExecutionRequest(program="sleep", arguments=["60"], timeout=1.0))
check("timed_out is True", r3.timed_out is True)
check("exit_code is None", r3.exit_code is None)

print()
print("=== Test 4: Grandchild termination (process-group kill) ===")
# Kill any pre-existing sleep 60 processes so our diff is clean
subprocess.run(["pkill", "-9", "-f", "sleep 60"], capture_output=True)
time.sleep(0.4)

before = set(subprocess.run(
    ["pgrep", "-f", "sleep 60"], capture_output=True, text=True
).stdout.split())

execute(ExecutionRequest(
    program="python3",
    arguments=["-c", "import subprocess,time; subprocess.Popen(['sleep','60']); time.sleep(60)"],
    timeout=1.5,
))

time.sleep(1.0)  # give the kernel time to reap

after = set(subprocess.run(
    ["pgrep", "-f", "sleep 60"], capture_output=True, text=True
).stdout.split())

new_orphans = after - before
check("no orphan grandchild processes survive", not new_orphans)
if new_orphans:
    print(f"    orphan PIDs: {new_orphans}")

print()
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
