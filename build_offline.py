#!/usr/bin/env python3
"""
Precompute candidate artifacts from candidates.jsonl.
Run this once before ranking. Output goes to data/.

Usage:
    python build_offline.py path/to/candidates.jsonl
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute candidate embeddings and feature vectors"
    )
    parser.add_argument(
        "candidates",
        type=Path,
        help="Path to candidates.jsonl",
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"Error: file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    from offline.build_artifacts import build
    build(args.candidates)


if __name__ == "__main__":
    main()
