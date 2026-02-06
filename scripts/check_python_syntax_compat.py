#!/usr/bin/env python3
"""Check that Python files parse under multiple Python grammar versions."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def parse_version(value: str) -> tuple[int, int]:
    """Parse version string like '3.12' into a feature_version tuple."""
    major, minor = value.split(".")
    return int(major), int(minor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["3.12", "3.13", "3.14"],
        help="Python grammar versions to validate (default: 3.12 3.13 3.14)",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["plugins/llm/src", "plugins/llm/tests"],
        help="Paths to scan for .py files",
    )
    args = parser.parse_args()

    versions = [parse_version(v) for v in args.versions]
    files = sorted(
        path for root in args.paths for path in Path(root).rglob("*.py") if path.is_file()
    )

    if not files:
        print("No Python files found to validate.")
        return 0

    errors: list[str] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        for version in versions:
            try:
                ast.parse(source, filename=str(path), feature_version=version)
            except SyntaxError as exc:
                major, minor = version
                lineno = exc.lineno or 1
                offset = exc.offset or 1
                errors.append(f"{path}:{lineno}:{offset}: py{major}.{minor}: {exc.msg}")

    if errors:
        print("Syntax compatibility check failed:")
        for err in errors:
            print(err)
        return 1

    version_list = ", ".join(args.versions)
    print(f"Syntax compatibility check passed for: {version_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
