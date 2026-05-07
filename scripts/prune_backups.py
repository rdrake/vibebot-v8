#!/usr/bin/env python3
"""Prune Limnoria's bot.conf rolling backups.

Limnoria writes a fresh `bot.conf.backup.<timestamp>` on every config
change. Over time this can fill `backup/` with hundreds of identical-ish
files. This script keeps the N most recent and deletes the rest.

Usage::

    uv run python scripts/prune_backups.py [--keep 20] [--dir backup] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", default="backup", type=Path, help="Backup directory (default: backup)"
    )
    parser.add_argument(
        "--keep", default=20, type=int, help="How many recent backups to retain (default: 20)"
    )
    parser.add_argument(
        "--pattern", default="bot.conf.backup.*", help="Glob pattern (default: bot.conf.backup.*)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be deleted without deleting"
    )
    args = parser.parse_args()

    target_dir: Path = args.dir
    if not target_dir.is_dir():
        print(f"No backup directory at {target_dir}; nothing to prune.", file=sys.stderr)
        return 0

    matches = sorted(target_dir.glob(args.pattern))
    if len(matches) <= args.keep:
        print(f"Found {len(matches)} backups in {target_dir}; nothing over keep={args.keep}.")
        return 0

    to_delete = matches[: -args.keep]
    print(f"Found {len(matches)} backups; keeping {args.keep} newest, removing {len(to_delete)}.")
    for path in to_delete:
        if args.dry_run:
            print(f"  [dry-run] would remove {path}")
        else:
            try:
                path.unlink()
                print(f"  removed {path}")
            except OSError as exc:
                print(f"  failed to remove {path}: {exc}", file=sys.stderr)
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
