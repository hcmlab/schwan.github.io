#!/usr/bin/env python3
"""Compatibility wrapper for prediction and reporting."""

import sys

from swan_ft.cli import main


def _translate(argv: list[str]) -> list[str]:
    if not argv:
        raise SystemExit("Usage: evaluate.py <predict|report> ...")
    command = argv[0]
    if command == "predict":
        return ["predict", "cv", *argv[1:]]
    if command == "report":
        return ["report", "cv", *argv[1:]]
    raise SystemExit(f"Unknown evaluate command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(_translate(sys.argv[1:])))
