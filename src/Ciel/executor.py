""" Controlled Linux Process Execution """

import subprocess 

from Ciel.models import ExecutionRequest, ExecutionResult

def execute(request: ExecutionRequest) -> ExecutionResult:
    """ Execute an approved request without invoking the shell. """

    completed = subprocess.run(
        [request.program, *request.arguments],
        shell = False,
        capture_output = True,
        text = True,
    )

    return ExecutionResult(
        stdout = completed.stdout,
        stderr = completed.stderr,
        exit_code = completed.returncode,
        timed_out = False,
        duration = 0.0,

    )