"""
scripts/import_mbpp.py -- Import google-research-datasets/mbpp into Sample format.

Downloads the MBPP dataset from HuggingFace via ``datasets.load_dataset``,
converts each task to the project's canonical ``Sample`` schema
(task_type=code_generation), and writes:

    <output-dir>/normalized/<split>.jsonl
    <output-dir>/manifest.<split>.json
    <output-dir>/manifest.index.json

The importer ONLY downloads + normalises.  It does NOT run pytest or
:func:`src.validators.verify_sample` -- that is the job of
:mod:`scripts.verify_imported_mbpp`.  Imported samples default to
``verified=False`` with an all-false :class:`Verification` so that no
unverified claim escapes the import step.

Per P3 global constraint #20 (import + verify split) and the task brief,
the per-split manifest is never overwritten by another split: each
``--split`` invocation writes only ``manifest.<split>.json`` and merges
its entry into the shared ``manifest.index.json``.

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

# HF revision pin (P3 global constraint #16).  ``"main"`` is always valid for
# HuggingFace datasets; it pins the branch (not a specific commit hash), which
# is the most discoverable stable identifier available without scraping the HF
# commit log.
_SOURCE_REVISION = "main"

# MBPP samples are imported as UNVERIFIED code_generation references.  Per the
# P3 task brief, the importer must NOT claim any verification -- no compile,
# no pytest, no ruff, no timeout.  All four flags default to False and stay
# False until scripts/verify_imported_mbpp.py runs the real checks.
_VERIFIED_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
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
    source_split: Optional[str] = None,
) -> Sample:
    """Convert a single MBPP dataset record to a ``Sample`` (code_generation).

    Expected record keys: ``task_id``, ``text`` (or ``prompt``), ``code``,
    ``test_list``, and optionally ``difficulty``.

    Parameters
    ----------
    record:
        Raw MBPP record dict.
    created_at:
        ISO-8601 timestamp; defaults to ``datetime.now(timezone.utc)``.
    source_split:
        Name of the dataset split this record came from (e.g. ``"train"``,
        ``"test"``, ``"validation"``).  Stored on the Sample as the new
        P3 ``source_split`` field for downstream traceability.
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
        verified=False,
        verification=_VERIFIED_VER,
        generator=_GENERATOR,
        created_at=created_at,
        dataset_version=_DATASET_VERSION,
        source_split=source_split,
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
    source_revision: str,
    dataset_fingerprint: Optional[str],
    split: str,
    sample_count: int,
    normalized_sha256: str,
    normalized_file: str,
    license: str,
    imported_at: str,
    benchmark_contaminated: bool,
    standard_mbpp_test_claims_disallowed: bool,
) -> dict:
    """Build the per-split manifest dict for an MBPP import (does not write it).

    The returned dict matches the per-split schema in the P3 task brief:
    ``manifest.<split>.json``.  Verifier-filled fields
    (``verified_sha256`` / ``verified_count`` / ``rejected_count`` /
    ``rejected_sha256`` / ``verified_at``) are emitted as ``None`` placeholders
    so the schema is stable across the import -> verify lifecycle.
    """
    return {
        "source": source,
        "source_revision": source_revision,
        "dataset_fingerprint": dataset_fingerprint,
        "split": split,
        "sample_count": sample_count,
        "normalized_sha256": normalized_sha256,
        "normalized_file": normalized_file,
        "license": license,
        "imported_at": imported_at,
        "benchmark_contaminated": benchmark_contaminated,
        "standard_mbpp_test_claims_disallowed": standard_mbpp_test_claims_disallowed,
        # Verifier-filled placeholders -- the importer does NOT fabricate these.
        "verified_sha256": None,
        "verified_count": None,
        "rejected_count": None,
        "rejected_sha256": None,
        "verified_at": None,
    }


def build_manifest_index(
    *,
    source: str,
    splits_detail: dict[str, dict],
    updated_at: str,
) -> dict:
    """Build the manifest.index.json dict (does not write it).

    Parameters
    ----------
    source:
        HF repo id (e.g. ``google-research-datasets/mbpp``).
    splits_detail:
        Mapping ``{split_name: {sample_count, normalized_sha256, manifest}}``.
        Typically produced incrementally by :func:`update_manifest_index`.
    updated_at:
        ISO-8601 timestamp of the most recent import.
    """
    return {
        "source": source,
        "splits": sorted(splits_detail.keys()),
        "updated_at": updated_at,
        "splits_detail": splits_detail,
    }


def update_manifest_index(
    *,
    index_path: Path,
    source: str,
    split: str,
    split_detail: dict,
    updated_at: str,
) -> dict:
    """Merge *split_detail* into the manifest.index.json at *index_path*.

    Reads the existing index (if any), inserts/replaces the entry for
    *split*, writes the merged result back, and returns the merged dict.

    Per the brief: only ``manifest.index.json`` is shared across splits and
    merged on each import; per-split ``manifest.<split>.json`` files are
    never overwritten by other splits.
    """
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as fh:
            index = json.load(fh)
    else:
        index = {"source": source, "splits": [], "updated_at": updated_at,
                 "splits_detail": {}}

    # Defensive: ensure the expected keys exist even if a stale/partial file
    # was left on disk by a previous run.
    index.setdefault("source", source)
    index.setdefault("splits", [])
    index.setdefault("splits_detail", {})
    index.setdefault("updated_at", updated_at)

    index["splits_detail"][split] = split_detail
    index["splits"] = sorted(index["splits_detail"].keys())
    index["updated_at"] = updated_at
    index["source"] = source

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)
    return index


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


def extract_dataset_fingerprint(ds: Any) -> Optional[str]:
    """Best-effort extraction of the HuggingFace dataset fingerprint.

    Returns ``None`` when no fingerprint attribute is discoverable.  Per the
    brief: do NOT fabricate a value -- ``null`` is the correct sentinel.
    """
    # ``datasets`` exposes fingerprint on the Dataset object via various
    # attributes across versions; check the most common ones defensively.
    for attr in ("_fingerprint", "fingerprint"):
        try:
            val = getattr(ds, attr, None)
        except Exception:  # noqa: BLE001 -- defensive; never propagate
            val = None
        if isinstance(val, str) and val:
            return val
    # Some versions stash it under `ds.info` or `ds._info`.
    info = getattr(ds, "info", None) or getattr(ds, "_info", None)
    if info is not None:
        for attr in ("fingerprint", "_fingerprint"):
            try:
                val = getattr(info, attr, None)
            except Exception:  # noqa: BLE001
                val = None
            if isinstance(val, str) and val:
                return val
    return None


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

    print(f"Loading {_SOURCE_REPO} (split={split}, revision={_SOURCE_REVISION}) "
          "from HuggingFace ...")
    try:
        ds = _load_dataset(_SOURCE_REPO, split=split, revision=_SOURCE_REVISION)
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: failed to download {_SOURCE_REPO} (split={split}): {exc}",
            file=sys.stderr,
        )
        print("  Check network connection / HuggingFace access and retry.",
              file=sys.stderr)
        return 1

    dataset_fingerprint = extract_dataset_fingerprint(ds)
    created_at = datetime.now(timezone.utc).isoformat()
    samples: list[Sample] = []
    for record in ds:
        try:
            samples.append(
                mbpp_record_to_sample(
                    dict(record),
                    created_at=created_at,
                    source_split=split,
                )
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

    # ----- normalised JSONL -----
    jsonl_path = out_dir / "normalized" / f"{split}.jsonl"
    write_normalized_jsonl(samples, jsonl_path)

    sha = compute_sha256(jsonl_path)

    # ----- per-split manifest (NEVER overwritten by other splits) -----
    # Per P3 global constraint #17: only the test split is benchmark-
    # contaminated (because P3 will reuse it for training); train/validation
    # are clean.
    is_test_split = split == "test"
    manifest = build_manifest(
        source=_SOURCE_REPO,
        source_revision=_SOURCE_REVISION,
        dataset_fingerprint=dataset_fingerprint,
        split=split,
        sample_count=len(samples),
        normalized_sha256=sha,
        normalized_file=f"normalized/{split}.jsonl",
        license=_LICENSE,
        imported_at=created_at,
        benchmark_contaminated=is_test_split,
        standard_mbpp_test_claims_disallowed=is_test_split,
    )
    manifest_path = out_dir / f"manifest.{split}.json"
    write_manifest(manifest, manifest_path)

    # ----- merged manifest.index.json (the one shared file) -----
    split_detail = {
        "sample_count": len(samples),
        "normalized_sha256": sha,
        "manifest": f"manifest.{split}.json",
    }
    update_manifest_index(
        index_path=out_dir / "manifest.index.json",
        source=_SOURCE_REPO,
        split=split,
        split_detail=split_detail,
        updated_at=created_at,
    )

    print(f"import_mbpp: wrote {len(samples)} samples -> {jsonl_path}")
    print(f"  sha256:    {sha[:16]}...")
    print(f"  manifest:  -> {manifest_path}")
    print(f"  index:      -> {out_dir / 'manifest.index.json'}")
    if dataset_fingerprint is None:
        print("  note: dataset_fingerprint unavailable; recorded as null "
              "(P3 constraint: do NOT fabricate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
