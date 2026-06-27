"""
scripts/_io.py -- Shared JSONL sample I/O for the data-factory scripts.

Extracted so generate_reference.py, verify_samples.py, mutate_code.py and
build_dataset.py all load Sample JSONL through ONE implementation, avoiding
drift between near-identical "open -> strip -> from_json_line" loops.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample  # noqa: E402


def load_samples_file(path: "str | Path") -> list[Sample]:
    """Load Sample objects from a single JSONL file (blank lines skipped)."""
    path = Path(path)
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(Sample.from_json_line(line))
    return samples


def load_samples(paths: "list[str | Path]") -> list[Sample]:
    """Load and concatenate samples from one or more JSONL files.

    Raises
    ------
    FileNotFoundError
        If any of *paths* does not exist.
    """
    samples: list[Sample] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")
        samples.extend(load_samples_file(p))
    return samples
