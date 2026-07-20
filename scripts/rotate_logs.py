#!/usr/bin/env python3
"""Rotate VibeBot's `messages.log` to keep working copies bounded.

VibeBot/Limnoria do not rotate `messages.log` on their own. This script
performs a simple size-based rotation: when the active log exceeds
``--max-mb``, it is renamed to ``messages.log.<timestamp>`` and a fresh
empty file is created. Limnoria's `BetterFileHandler` subclasses
`logging.handlers.WatchedFileHandler` (see `supybot/log.py`), which
detects the inode change on the next write and reopens the new file —
so no bot restart is required after rotation.

Pre-existing rotated files are kept up to ``--keep`` and the rest are
deleted (oldest first).

Usage::

    uv run python scripts/rotate_logs.py \\
        [--log logs/messages.log] [--max-mb 50] [--keep 7] [--dry-run]

Recommended: invoke from a systemd timer or cron once an hour. Combine
with ``make clean-logs`` for a full reset during development.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="logs/messages.log", type=Path, help="Log file path")
    parser.add_argument(
        "--max-mb", default=50, type=int, help="Rotate when size exceeds this many MB"
    )
    parser.add_argument("--keep", default=7, type=int, help="How many rotated files to keep")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show actions without performing them"
    )
    args = parser.parse_args()

    log_path: Path = args.log
    if not log_path.exists():
        print(f"No log file at {log_path}; nothing to do.")
        return 0

    size_mb = log_path.stat().st_size / (1024 * 1024)
    if size_mb < args.max_mb:
        print(f"{log_path} is {size_mb:.1f} MB; under {args.max_mb} MB threshold — skipping.")
        return 0

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    rotated = log_path.with_name(f"{log_path.name}.{timestamp}")
    print(f"Rotating {log_path} ({size_mb:.1f} MB) -> {rotated.name}")
    if not args.dry_run:
        try:
            log_path.rename(rotated)
            log_path.touch()
        except OSError as exc:
            print(f"Rotation failed: {exc}", file=sys.stderr)
            return 1

    siblings = sorted(path for path in log_path.parent.glob(f"{log_path.name}.*") if path.is_file())
    if len(siblings) > args.keep:
        # keep=0 must delete all: siblings[:-0] == siblings[:0] == [] (a silent
        # no-op), so slice only when keep>0 and otherwise take everything.
        to_delete = siblings[: -args.keep] if args.keep else siblings
        for old in to_delete:
            print(f"  removing old rotated file {old.name}")
            if not args.dry_run:
                try:
                    old.unlink()
                except OSError as exc:
                    print(f"  failed to remove {old}: {exc}", file=sys.stderr)
                    return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
