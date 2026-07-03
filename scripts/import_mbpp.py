"""
scripts/import_mbpp.py -- Import google-research-datasets/mbpp into Sample format.

Downloads the MBPP dataset from HuggingFace via ``datasets.load_dataset``,
converts each task to the project's canonical ``Sample`` schema
(task_type=code_generation), and writes:

    <output-dir>/normalized/<split>.jsonl
    <output-dir>/manifest.json

Only the normalized JSONL and manifest are written to the output directory;
raw dataset bytes remain inside the HuggingFace cache (already git-ignored
via .hf_cache/).  No large raw files are written into the project tree.

Usage
-----
    python scripts/import_mbpp.py [--output-dir DIR] [--split train]

Exit codes
----------
    0   success
    1   failure (datasets unavailable / network error / no samples)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Eager import of `datasets` -- a heavy/network library.  Mirrors the pattern
# in scripts/download_model.py: catch ImportError AND OSError (Windows DLL
# issues) so the module remains importable for tests even when the library is
# missing or broken.
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset as _load_dataset
    _DATASETS_AVAILABLE = True
    _DATASETS_ERROR: Optional[str] = None
except (ImportError, OSError) as _exc:
    _load_dataset = None  # type: ignore[assignment]
    _DATASETS_AVAILABLE = False
    _DATASETS_ERROR = f"{type(_exc).__name__}: {_exc}"

# ---------------------------------------------------------------------------
# Project-root import guard (so the script works from any cwd)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample, Verification  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SOURCE_REPO = "google-research-datasets/mbpp"
_LICENSE = "Apache-2.0"
_DATASET_VERSION = "mbpp-v1"
_GENERATOR = "mbpp-importer"

# MBPP samples are imported as verified code_generation references; we have NOT
# run ruff on them, so ruff_ok=False (matches the spec in the task brief).
_VERIFIED_VER = Verification(
    syntax_ok=True, pytest_ok=True, ruff_ok=False, timeout=False
)

# MBPP ships difficulty as a string ("Introductory" / "Interview" /
# "Competition").  Map to the project's 0..4 scale.
_DIFFICULTY_MAP: dict[str, int] = {
    "introductory": 1,
    "easy": 1,
    "interview": 2,
    "medium": 2,
    "competition": 3,
    "hard": 3,
}

# Canonical skill-keyword vocabulary used to derive skill_tags from the MBPP
# prompt text.  Order is significant only for output stability.
_SKILL_KEYWORDS: list[str] = [
    "list", "array", "string", "sort", "sorting", "search", "binary search",
    "recursion", "recursive", "loop", "math", "prime", "matrix", "dict",
    "dictionary", "set", "tree", "graph", "number", "digit", "palindrome",
    "fibonacci", "factorial", "sum", "count", "reverse", "replace",
    "substring", "regex", "class", "function", "tuple", "stack", "queue",
    "permutation", "combination",
]


# ---------------------------------------------------------------------------
# Pure conversion helpers (importable, fully testable -- no network)
# ---------------------------------------------------------------------------

def extract_skill_tags(text: str) -> list[str]:
    """Return skill tags extracted from *text* by keyword matching.

    Falls back to ``["general"]`` when no known keyword matches.
    """
    lower = (text or "").lower()
    matched: list[str] = []
    for kw in _SKILL_KEYWORDS:
        if kw in lower and kw not in matched:
            matched.append(kw)
    return matched or ["general"]


def infer_difficulty(difficulty: Any, code: str) -> int:
    """Map an MBPP difficulty value to the project's 0..4 scale.

    Accepts strings (e.g. "Introductory"), ints, or None.  When the value
    cannot be resolved, falls back to a bucket based on ``len(code)``.
    """
    if difficulty is not None:
        s = str(difficulty).strip().lower()
        if s in _DIFFICULTY_MAP:
            return _DIFFICULTY_MAP[s]
        try:
            n = int(s)
            if 0 <= n <= 4:
                return n
        except ValueError:
            pass

    n = len(code or "")
    if n < 200:
        return 0
    if n < 500:
        return 1
    if n < 1000:
        return 2
    if n < 2000:
        return 3
    return 4


def split_mbpp_tests(test_list: list[str]) -> tuple[str, str]:
    """Split an MBPP ``test_list`` into (public_tests, hidden_tests) strings.

    The first 1-2 (non-blank) asserts become public; the remainder become
    hidden.  Items are joined with a blank line separator.
    """
    tests = [t for t in (test_list or []) if t and t.strip()]
    n_public = min(2, len(tests)) if tests else 0
    public = "\n\n".join(tests[:n_public])
    hidden = "\n\n".join(tests[n_public:])
    return public, hidden


def mbpp_record_to_sample(
    record: dict,
    *,
    created_at: Optional[str] = None,
) -> Sample:
    """Convert a single MBPP dataset record to a ``Sample`` (code_generation).

    Expected record keys: ``task_id``, ``text`` (or ``prompt``), ``code``,
    ``test_list``, and optionally ``difficulty``.
    """
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    task_id = record.get("task_id")
    text = record.get("text") or record.get("prompt") or ""
    code = record.get("code") or ""
    test_list = record.get("test_list") or []
    difficulty_field = record.get("difficulty")

    public_tests, hidden_tests = split_mbpp_tests(test_list)

    return Sample(
        sample_id=f"mbpp_{task_id}",
        family_id=f"mbpp_fam_{task_id}",
        difficulty=infer_difficulty(difficulty_field, code),
        task_type="code_generation",
        language="python",
        skill_tags=extract_skill_tags(text),
        instruction=text,
        broken_code=None,
        execution_feedback=None,
        target_code=code,
        public_tests=public_tests,
        hidden_tests=hidden_tests,
        verified=True,
        verification=_VERIFIED_VER,
        generator=_GENERATOR,
        created_at=created_at,
        dataset_version=_DATASET_VERSION,
    )


# ---------------------------------------------------------------------------
# I/O helpers (importable, testable)
# ---------------------------------------------------------------------------

def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    source: str,
    split: str,
    sample_count: int,
    sha256: str,
    license: str,
    imported_at: str,
) -> dict:
    """Build the manifest dict for an MBPP import (does not write it)."""
    return {
        "source": source,
        "split": split,
        "sample_count": sample_count,
        "sha256": sha256,
        "license": license,
        "imported_at": imported_at,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    """Write *manifest* as indented JSON to *path* (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def write_normalized_jsonl(samples: list[Sample], path: Path) -> None:
    """Write *samples* as one JSONL line per Sample (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(s.to_json_line() + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Import google-research-datasets/mbpp into Sample format.",
    )
    p.add_argument(
        "--output-dir", default="data/external/mbpp",
        help="Output directory (default: data/external/mbpp).",
    )
    p.add_argument(
        "--split", default="train",
        help="Dataset split to load (default: train).",
    )
    return p


def main() -> int:
    """CLI entry point.  Returns 0 on success, 1 on failure."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    out_dir = Path(args.output_dir)
    split = args.split

    if not _DATASETS_AVAILABLE:
        print(f"ERROR: datasets library not available: {_DATASETS_ERROR}",
              file=sys.stderr)
        return 1

    print(f"Loading {_SOURCE_REPO} (split={split}) from HuggingFace ...")
    try:
        ds = _load_dataset(_SOURCE_REPO, split=split)
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: failed to download {_SOURCE_REPO} (split={split}): {exc}",
            file=sys.stderr,
        )
        print("  Check network connection / HuggingFace access and retry.",
              file=sys.stderr)
        return 1

    created_at = datetime.now(timezone.utc).isoformat()
    samples: list[Sample] = []
    for record in ds:
        try:
            samples.append(
                mbpp_record_to_sample(dict(record), created_at=created_at)
            )
        except Exception as exc:  # noqa: BLE001
            task_id = record.get("task_id", "?") if hasattr(record, "get") else "?"
            print(f"WARNING: skipping task_id={task_id}: {exc}", file=sys.stderr)

    if not samples:
        print(
            "ERROR: no samples converted (dataset empty or all conversions failed).",
            file=sys.stderr,
        )
        return 1

    jsonl_path = out_dir / "normalized" / f"{split}.jsonl"
    write_normalized_jsonl(samples, jsonl_path)

    sha = compute_sha256(jsonl_path)
    manifest = build_manifest(
        source=_SOURCE_REPO,
        split=split,
        sample_count=len(samples),
        sha256=sha,
        license=_LICENSE,
        imported_at=created_at,
    )
    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest, manifest_path)

    print(f"import_mbpp: wrote {len(samples)} samples -> {jsonl_path}")
    print(f"  sha256:    {sha[:16]}...")
    print(f"  manifest:  -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
