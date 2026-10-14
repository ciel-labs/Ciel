""" Controlled Linux Process Execution """

import subprocess
import time

from Ciel.models import ExecutionRequest, ExecutionResult

def execute(request: ExecutionRequest) -> ExecutionResult:
    """ Execute an approved request without invoking the shell. """

    start_time = time.monotonic()

    try:
        completed = subprocess.run(
            [request.program, *request.arguments],
            shell = False,
            capture_output = True,
            text = True,
            timeout = request.timeout,
        )

        duration = time.monotonic() - start_time

        return ExecutionResult(
            stdout = completed.stdout,
            stderr = completed.stderr,
            exit_code = completed.returncode,
            timed_out = False,
            duration = duration,
        )
    
    except subprocess.TimeoutExpired as exc:

        duration = time.monotonic() - start_time

        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        return ExecutionResult(
            stdout = stdout, 
            stderr = stderr, 
            exit_code = None, 
            timed_out = True,
            duration = duration,
        )
