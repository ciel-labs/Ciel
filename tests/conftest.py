"""Shared pytest fixtures for the Ciel test suite.

conftest.py is loaded automatically by pytest before any test file.
Fixtures defined here are available to every test in this directory
without any explicit import.

Why fixtures instead of module-level constants?
-----------------------------------------------
A fixture is a factory — pytest calls it fresh for each test that
requests it. This prevents test state from leaking between tests.
A frozen dataclass like ExecutionRequest is immutable, so sharing it
would be safe, but the factory pattern scales better when we later
add fixtures that open files, sockets, or temporary directories.
"""

import pytest

from Ciel.models import ExecutionRequest


@pytest.fixture()
def echo_request() -> ExecutionRequest:
    """Minimal valid request: echo a single word."""
    return ExecutionRequest(program="echo", arguments=["hello"])


@pytest.fixture()
def ls_request() -> ExecutionRequest:
    """Valid request using an absolute path."""
    return ExecutionRequest(program="/bin/ls", arguments=["/tmp"])


@pytest.fixture()
def short_timeout_request() -> ExecutionRequest:
    """A sleep that will always time out under the given timeout."""
    return ExecutionRequest(program="sleep", arguments=["60"], timeout=1.0)
