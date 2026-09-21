"""Command Line Interface for the Linux Agent."""

import argparse

from Ciel.config import AppConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="Ciel",
        description="Security-first autonomous Linux computer-use agent.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Accept a task for the agent.",
    )

    run_parser.add_argument(
        "task",
        help="Natural-language task for the agent.",
    )

    return parser


def run_task(task: str, config: AppConfig) -> None:
    """Accept a task without executing anything."""

    print(f"Project: {config.project_name}")
    print(f"Version: {config.version}")
    print()
    print("Task Received:")
    print(task)
    print()
    print("Status: Accepted")
    print("Execution: Disabled in Step 1")


def main() -> None:
    """CLI entry point."""

    config = AppConfig()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_task(args.task, config)


if __name__ == "__main__":
    main()