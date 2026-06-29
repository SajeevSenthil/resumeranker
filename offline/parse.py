import json
from pathlib import Path
from typing import Iterator


def iter_candidates(path: Path) -> Iterator[dict]:
    """
    Handles both JSONL (one object per line) and JSON array files.
    The format is detected from the first non-whitespace character.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = ""
        while not first_char:
            ch = f.read(1)
            if not ch:
                return
            first_char = ch.strip()
        f.seek(0)

        if first_char == "[":
            # JSON array format (e.g. sample_candidates.json)
            for item in json.load(f):
                yield item
        else:
            # JSONL format (e.g. candidates.jsonl)
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_candidates(path: Path) -> list[dict]:
    return list(iter_candidates(path))
