#!/usr/bin/env python3
"""
Rank top-100 candidates against a job description.
Requires artifacts from build_offline.py to be present in data/.

Usage:
    python rank.py path/to/jd.txt output.csv
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank candidates against a job description"
    )
    parser.add_argument(
        "jd",
        type=Path,
        help="Path to job description (.txt)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output CSV path (e.g. team_xxx.csv)",
    )
    args = parser.parse_args()

    if not args.jd.exists():
        print(f"Error: JD file not found: {args.jd}", file=sys.stderr)
        sys.exit(1)

    from config import ARTIFACT_PATHS

    missing = [name for name, path in ARTIFACT_PATHS.items() if not path.exists()]
    if missing:
        print(
            f"Error: missing artifacts: {missing}\n"
            "Run build_offline.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    jd_text = args.jd.read_text(encoding="utf-8")

    from ranker.pipeline import rank
    rank(jd_text, args.output)


if __name__ == "__main__":
    main()
