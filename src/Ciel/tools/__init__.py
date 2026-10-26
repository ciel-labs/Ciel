"""Public API for the Ciel tools package.

Import from here rather than from submodules directly.

    from Ciel.tools import ToolCall, ToolRegistry, default_registry
    from Ciel.tools import ToolNotFoundError, ToolArgumentError
"""

from Ciel.tools.base import (
    Tool,
    ToolArgumentError,
    ToolCall,
    ToolError,
    ToolNotFoundError,
    ToolParameter,
    ToolRegistry,
)
from Ciel.tools.definitions import (
    CurrentUser,
    DiskUsage,
    FreeMemory,
    Hostname,
    KernelInfo,
    SystemUptime,
    default_registry,
)

__all__ = [
    # Abstract base and data types
    "Tool",
    "ToolParameter",
    "ToolCall",
    # Error types
    "ToolError",
    "ToolNotFoundError",
    "ToolArgumentError",
    # Registry
    "ToolRegistry",
    "default_registry",
    # Concrete tools (exported so tests can instantiate them directly)
    "CurrentUser",
    "DiskUsage",
    "FreeMemory",
    "Hostname",
    "KernelInfo",
    "SystemUptime",
]
