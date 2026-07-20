#!/usr/bin/env python3
"""Compatibility wrapper for fold creation."""

from swan_ft.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["folds", "create", *__import__("sys").argv[1:]]))
