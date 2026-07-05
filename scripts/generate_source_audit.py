"""
scripts/generate_source_audit.py -- Generate reports/p3/mbpp-source-audit.json.

Reads the per-split MBPP manifests (``manifest.<split>.json``) plus the
normalised JSONL files (``normalized/<split>.jsonl``) and emits a structured
audit JSON with:

- Per-split sample counts, task_id ranges, missing/duplicate IDs, SHA digests,
  verified/rejected counts, benchmark_contaminated flag, dataset_fingerprint.
- Cross-split task_id overlap (train/test/validation).
- Total samples, total verified, total rejected.
- ``new_families_available`` = test verified + validation verified (new
  families not used in P2).
- ``conclusion``: LIKELY_FEASIBLE if test verified >= 240, else INFEASIBLE.

The audit reads ONLY on-disk artefacts -- it does NOT call the network and
does NOT re-run any verification.  When a split's manifest or JSONL is
missing, the script records the split as ``present=False`` with zero counts
rather than raising, so partial progress (e.g. only test imported) still
produces a useful audit.

Usage
-----
    python scripts/generate_source_audit.py \\
        --output-dir data/external/mbpp \\
        --report-dir reports/p3

Exit codes
----------
    0   report written successfully
    1   error (no manifests found at all / output dir not writable)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Project-root import guard (so the script works from any cwd)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical MBPP splits audited by this report.  Order is significant only
# for output stability; we always sort splits alphabetically in the report.
DEFAULT_SPLITS = ("train", "test", "validation")

# Conclusion thresholds per the task brief:
# - LIKELY_FEASIBLE: test verified_count >= 240 (180 train + 60 val minimum)
# - INFEASIBLE: test verified_count < 240
_TEST_VERIFIED_THRESHOLD = 240


# ---------------------------------------------------------------------------
# JSONL / manifest readers (pure, importable, testable)
# ---------------------------------------------------------------------------

def extract_task_id(sample_id: str) -> Optional[int]:
    """Parse an MBPP ``sample_id`` (format: ``mbpp_<int>``) into an int.

    Returns ``None`` when the prefix is missing or the suffix is not a
    valid integer.  This defensive parser never raises.
    """
    if not isinstance(sample_id, str) or not sample_id:
        return None
    # Accept ``mbpp_601`` and case variants; reject anything else.
    lower = sample_id.lower()
    prefix = "mbpp_"
    if not lower.startswith(prefix):
        return None
    tail = sample_id[len(prefix):]
    try:
        return int(tail)
    except (TypeError, ValueError):
        return None


def load_task_ids(jsonl_path: Path) -> list[int]:
    """Read a normalised JSONL file and return task_ids in file order.

    Lines whose ``sample_id`` does not parse to an int are silently
    skipped (they do not contribute to task_id ranges).  An empty list
    is returned when the file does not exist.
    """
    if not jsonl_path.exists():
        return []
    task_ids: list[int] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = extract_task_id(record.get("sample_id", ""))
            if tid is not None:
                task_ids.append(tid)
    return task_ids


def find_missing_task_ids(task_ids: list[int]) -> list[int]:
    """Return the sorted set of integers in [min, max] not present in *task_ids*.

    Returns ``[]`` when *task_ids* is empty.
    """
    if not task_ids:
        return []
    seen = set(task_ids)
    lo, hi = min(seen), max(seen)
    return [i for i in range(lo, hi + 1) if i not in seen]


def find_duplicate_task_ids(task_ids: list[int]) -> list[int]:
    """Return the sorted list of task_ids that appear more than once."""
    if not task_ids:
        return []
    counts: dict[int, int] = {}
    for tid in task_ids:
        counts[tid] = counts.get(tid, 0) + 1
    return sorted([tid for tid, n in counts.items() if n > 1])


def load_manifest(manifest_path: Path) -> Optional[dict]:
    """Load a per-split manifest JSON, returning None if file is missing."""
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Per-split audit
# ---------------------------------------------------------------------------

def audit_split(
    *,
    split: str,
    output_dir: Path,
) -> dict:
    """Build the audit dict for a single split.

    The returned dict always contains the same key set regardless of
    whether the split has been imported/verified yet; missing fields
    surface as ``null`` / zero / empty list so downstream consumers do
    not need to special-case absent keys.
    """
    manifest_path = output_dir / f"manifest.{split}.json"
    normalized_path = output_dir / "normalized" / f"{split}.jsonl"

    manifest = load_manifest(manifest_path)

    present = manifest is not None
    sample_count: Optional[int] = None
    normalized_sha256: Optional[str] = None
    verified_sha256: Optional[str] = None
    verified_count: Optional[int] = None
    rejected_count: Optional[int] = None
    benchmark_contaminated: Optional[bool] = None
    dataset_fingerprint: Optional[str] = None
    source_revision: Optional[str] = None
    task_id_min: Optional[int] = None
    task_id_max: Optional[int] = None
    missing_task_ids: list[int] = []
    duplicate_task_ids: list[int] = []

    if manifest is not None:
        sample_count = manifest.get("sample_count")
        normalized_sha256 = manifest.get("normalized_sha256")
        verified_sha256 = manifest.get("verified_sha256")
        verified_count = manifest.get("verified_count")
        rejected_count = manifest.get("rejected_count")
        benchmark_contaminated = manifest.get("benchmark_contaminated")
        dataset_fingerprint = manifest.get("dataset_fingerprint")
        source_revision = manifest.get("source_revision")

    task_ids = load_task_ids(normalized_path)
    if task_ids:
        task_id_min = min(task_ids)
        task_id_max = max(task_ids)
        missing_task_ids = find_missing_task_ids(task_ids)
        duplicate_task_ids = find_duplicate_task_ids(task_ids)

    return {
        "present": present,
        "sample_count": sample_count,
        "task_id_range": {
            "min": task_id_min,
            "max": task_id_max,
        },
        "missing_task_ids": missing_task_ids,
        "duplicate_task_ids": duplicate_task_ids,
        "normalized_sha256": normalized_sha256,
        "verified_sha256": verified_sha256,
        "verified_count": verified_count,
        "rejected_count": rejected_count,
        "benchmark_contaminated": benchmark_contaminated,
        "dataset_fingerprint": dataset_fingerprint,
        "source_revision": source_revision,
    }


# ---------------------------------------------------------------------------
# Cross-split overlap
# ---------------------------------------------------------------------------

def load_task_id_set(jsonl_path: Path) -> set[int]:
    """Return the set of task_ids in a normalised JSONL file."""
    return set(load_task_ids(jsonl_path))


def cross_split_overlap(
    *,
    output_dir: Path,
    splits: tuple[str, ...],
) -> dict[str, list[int]]:
    """Compute pairwise task_id overlap across splits.

    Returns a mapping ``"<split_a>_<split_b>" -> sorted_intersection``.
    Pairs are emitted in canonical (alphabetically-sorted) split order so
    the key is stable: e.g. ``"test_train"``, never ``"train_test"``.
    """
    sets: dict[str, set[int]] = {
        s: load_task_id_set(output_dir / "normalized" / f"{s}.jsonl")
        for s in splits
    }
    out: dict[str, list[int]] = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            key_parts = sorted([a, b])
            key = f"{key_parts[0]}_{key_parts[1]}"
            inter = sorted(sets[a] & sets[b])
            out[key] = inter
    return out


# ---------------------------------------------------------------------------
# Conclusion logic
# ---------------------------------------------------------------------------

def compute_conclusion(
    *,
    splits_audit: dict[str, dict],
) -> tuple[str, str, int]:
    """Compute the overall conclusion.

    Returns
    -------
    (conclusion, feasibility_notes, new_families_available)
    """
    test_audit = splits_audit.get("test", {})
    validation_audit = splits_audit.get("validation", {})

    test_verified = test_audit.get("verified_count") or 0
    val_verified = validation_audit.get("verified_count") or 0
    new_families_available = test_verified + val_verified

    if test_verified >= _TEST_VERIFIED_THRESHOLD:
        conclusion = "LIKELY_FEASIBLE"
        notes = (
            f"test verified_count={test_verified} >= "
            f"{_TEST_VERIFIED_THRESHOLD} (180 train + 60 val minimum). "
            f"new_families_available={new_families_available} "
            f"(test {test_verified} + validation {val_verified})."
        )
    else:
        conclusion = "INFEASIBLE"
        notes = (
            f"test verified_count={test_verified} < "
            f"{_TEST_VERIFIED_THRESHOLD} (180 train + 60 val minimum). "
            f"new_families_available={new_families_available}. "
            "Stop and escalate per P3 plan."
        )
    return conclusion, notes, new_families_available


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def generate_audit(
    *,
    output_dir: Path,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict:
    """Build the complete audit dict (does not write to disk).

    Reads manifests + normalised JSONL files for each split, computes the
    cross-split overlap, totals, conclusion, and returns the full audit
    dict ready to be JSON-serialised.
    """
    splits_audit: dict[str, dict] = {}
    for split in splits:
        splits_audit[split] = audit_split(split=split, output_dir=output_dir)

    # Pick a single source_revision (any non-null entry; prefer the first
    # split alphabetically that has one).  All splits should report the
    # same revision because they were imported from the same HF repo.
    source_revision: Optional[str] = None
    for split in sorted(splits):
        rev = splits_audit[split].get("source_revision")
        if rev:
            source_revision = rev
            break

    overlap = cross_split_overlap(output_dir=output_dir, splits=splits)

    total_samples = sum(
        (s.get("sample_count") or 0) for s in splits_audit.values()
    )
    total_verified = sum(
        (s.get("verified_count") or 0) for s in splits_audit.values()
    )
    total_rejected = sum(
        (s.get("rejected_count") or 0) for s in splits_audit.values()
    )

    conclusion, feasibility_notes, new_families_available = compute_conclusion(
        splits_audit=splits_audit,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "google-research-datasets/mbpp",
        "source_revision": source_revision,
        "splits": splits_audit,
        "cross_split_task_id_overlap": overlap,
        "total_samples": total_samples,
        "total_verified": total_verified,
        "total_rejected": total_rejected,
        "new_families_available": new_families_available,
        "conclusion": conclusion,
        "feasibility_notes": feasibility_notes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate reports/p3/mbpp-source-audit.json from MBPP import + "
            "verify manifests."
        ),
    )
    p.add_argument(
        "--output-dir", default="data/external/mbpp",
        help="Root import/output directory (default: data/external/mbpp).",
    )
    p.add_argument(
        "--report-dir", default="reports/p3",
        help="Directory to write the audit report (default: reports/p3).",
    )
    p.add_argument(
        "--report-name", default="mbpp-source-audit.json",
        help="Report filename (default: mbpp-source-audit.json).",
    )
    p.add_argument(
        "--splits", nargs="+", default=list(DEFAULT_SPLITS),
        help=(
            "Splits to audit (default: train test validation). "
            "Unknown splits are allowed but produce empty audit entries."
        ),
    )
    return p


def main() -> int:
    """CLI entry point.  Returns 0 on success, 1 on error."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    splits = tuple(args.splits)

    if not output_dir.exists():
        print(
            f"ERROR: output-dir not found: {output_dir}",
            file=sys.stderr,
        )
        return 1

    audit = generate_audit(output_dir=output_dir, splits=splits)

    # Sanity check: at least one split must be present, otherwise the audit
    # is meaningless (probably wrong output-dir).
    present = [s for s in splits if audit["splits"][s].get("present")]
    if not present:
        print(
            f"ERROR: no manifest.<split>.json files found in {output_dir}.",
            file=sys.stderr,
        )
        return 1

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / args.report_name
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    print(f"generate_source_audit: wrote {report_path}")
    print(f"  splits audited: {sorted(present)}")
    print(
        f"  total_samples={audit['total_samples']}  "
        f"total_verified={audit['total_verified']}  "
        f"total_rejected={audit['total_rejected']}"
    )
    print(f"  new_families_available={audit['new_families_available']}")
    print(f"  conclusion={audit['conclusion']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
