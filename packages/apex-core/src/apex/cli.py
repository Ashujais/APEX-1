from __future__ import annotations

import argparse
import json

from apex.hardware import detect_hardware


def main() -> None:
    parser = argparse.ArgumentParser(prog="apex", description="APEX-1 research CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("hardware", help="Inspect available local compute")
    arguments = parser.parse_args()
    if arguments.command == "hardware":
        print(json.dumps(detect_hardware().to_dict(), indent=2))


if __name__ == "__main__":
    main()
