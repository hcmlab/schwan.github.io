#!/usr/bin/env python3
"""Compatibility wrapper for dataset variant generation."""

from swan_ft.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["dataset", "build", *__import__("sys").argv[1:]]))
