"""
Application Configuration 

This Application intentionally contains only configuration that is safe 
for the initial project skeleton

"""

from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    """ Configuration for the linux agent """

    project_name: str =  "Ciel"
    version: str = "0.1.0"

