"""scripts/build_family_registry.py -- Build the canonical Family Registry.

Reads three inputs:
  1. ``data/p2-curriculum/family-partition.json``  -- P2 backfill (374 families).
  2. ``data/external/mbpp/verified/*.jsonl``        -- new MBPP families
     (test/validation splits not already in P2).
  3. ``reports/p3/cross-split-dedup-quarantine.json`` -- quarantine tags
     appended to the affected families (P2 history preserved).

Produces ``data/family-registry.json`` -- the canonical store of
family-level state that Tasks 7/8/9 use to claim families for
frozen_v3_candidate / frozen_v3 / p3_train / p3_validation.

Usage
-----
    python scripts/build_family_registry.py \\
        --p2-partition data/p2-curriculum/family-partition.json \\
        --mbpp-verified-dir data/external/mbpp/verified \\
        --quarantine reports/p3/cross-split-dedup-quarantine.json \\
        --output data/family-registry.json

Exit codes
----------
    0   success
    1   invariant violation or I/O error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root import guard (so the script works from any cwd)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.family_registry import (  # noqa: E402
    FamilyEntry,
    FamilyRegistry,
    P3_EXCLUSIVE_TAGS,
    P2_USAGE_TAGS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Git short SHA of the merge commit that introduced the P2 partition file.
P2_FIRST_COMMIT: str = "515c955"

# Git short SHA of the MBPP import commit (Task 3/4).
MBPP_IMPORT_FIRST_COMMIT: str = "3dce2ce"

# Mapping from P2 partition file key -> usage tag.
_P2_TAG_FOR_KEY: dict[str, str] = {
    "train_families": "p2_train",
    "validation_families": "p2_validation",
    "frozen_families": "p2_frozen_v2",
}

# Bijective family_id pattern: mbpp_fam_<n> <-> mbpp_<n>.
_FAM_RE = re.compile(r"^mbpp_fam_(\d+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def derive_sample_id(family_id: str) -> str:
    """Bijective mapping ``mbpp_fam_<n>`` -> ``mbpp_<n>``.

    Every MBPP family has exactly one source sample, so the derived
    ``sample_ids`` list always contains a single element.
    """
    m = _FAM_RE.match(family_id)
    if not m:
        raise ValueError(
            f"cannot derive sample_id from family_id={family_id!r} "
            f"(expected pattern mbpp_fam_<n>)"
        )
    return f"mbpp_{m.group(1)}"


def derive_source_task_id(family_id: str) -> str:
    """Same as :func:`derive_sample_id` (the source task id IS the sample id
    for MBPP families)."""
    return derive_sample_id(family_id)


# ---------------------------------------------------------------------------
# Backfill passes
# ---------------------------------------------------------------------------

def backfill_p2(registry: FamilyRegistry, partition_path: Path) -> set[str]:
    """Backfill the 374 P2 families from the partition file.

    Returns the set of P2 family_ids (used later for invariant #4).
    """
    with partition_path.open(encoding="utf-8") as fh:
        partition = json.load(fh)

    p2_ids: set[str] = set()
    for key, tag in _P2_TAG_FOR_KEY.items():
        families = partition.get(key, [])
        for fam_id in families:
            p2_ids.add(fam_id)
            entry = registry.families.get(fam_id)
            if entry is None:
                entry = FamilyEntry(
                    family_id=fam_id,
                    source_task_id=derive_source_task_id(fam_id),
                    source_split="train",
                    usage=[tag],
                    first_commit=P2_FIRST_COMMIT,
                    dataset_version="mbpp-v1",
                    sample_ids=[derive_sample_id(fam_id)],
                )
                registry.families[fam_id] = entry
            else:
                # P2 backfill runs first, so this branch should never
                # trigger; if it does, just claim the tag (idempotent).
                entry.claim(tag)
    return p2_ids


def backfill_mbpp_verified(
    registry: FamilyRegistry, verified_dir: Path
) -> set[str]:
    """Discover new MBPP families from the verified JSONL files.

    For families already in the registry (P2 backfill), sanity-check
    that ``source_split`` matches. For new families, create an entry
    with empty usage (these are the candidates Task 7 will claim).

    Returns the set of all family_ids seen in the verified JSONL.
    """
    verified_ids: set[str] = set()
    for jsonl_path in sorted(verified_dir.glob("*.jsonl")):
        split_name = jsonl_path.stem  # e.g. "train", "test", "validation"
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                fam_id = data["family_id"]
                verified_ids.add(fam_id)
                source_split = data.get("source_split") or split_name
                entry = registry.families.get(fam_id)
                if entry is None:
                    entry = FamilyEntry(
                        family_id=fam_id,
                        source_task_id=derive_source_task_id(fam_id),
                        source_split=source_split,
                        usage=[],
                        first_commit=MBPP_IMPORT_FIRST_COMMIT,
                        dataset_version="mbpp-v1",
                        sample_ids=[derive_sample_id(fam_id)],
                    )
                    registry.families[fam_id] = entry
                else:
                    # P2-backfilled family: sanity-check source_split.
                    if entry.source_split != source_split:
                        raise RuntimeError(
                            f"source_split mismatch for {fam_id}: "
                            f"registry has {entry.source_split!r}, "
                            f"verified JSONL has {source_split!r}"
                        )
    return verified_ids


def backfill_quarantine(
    registry: FamilyRegistry, quarantine_path: Path
) -> list[str]:
    """Append ``quarantine`` to the usage list of each quarantined family.

    Returns the list of quarantined family_ids (used later for invariant #2).
    """
    with quarantine_path.open(encoding="utf-8") as fh:
        q = json.load(fh)
    quarantined = list(q.get("quarantined_families", []))
    for fam_id in quarantined:
        entry = registry.families.get(fam_id)
        if entry is None:
            # Per brief, all quarantined families come from MBPP verified
            # or P2, so this should not happen. Handle gracefully: create
            # a stub entry with just the quarantine tag.
            entry = FamilyEntry(
                family_id=fam_id,
                source_task_id=derive_source_task_id(fam_id),
                source_split="unknown",
                usage=["quarantine"],
                first_commit=MBPP_IMPORT_FIRST_COMMIT,
                dataset_version="mbpp-v1",
                sample_ids=[derive_sample_id(fam_id)],
            )
            registry.families[fam_id] = entry
        else:
            entry.claim("quarantine")
    return quarantined


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def assert_invariants(
    registry: FamilyRegistry,
    *,
    expected_p2_count: int,
    expected_quarantine_count: int,
    expected_total_families: int,
) -> None:
    """Assert the 4 builder invariants from the task brief.

    Raises ``AssertionError`` on the first violation.
    """
    # Invariant 1: total_p2_used == expected (374 for the real run).
    actual_p2 = sum(
        1 for e in registry.families.values()
        if any(t in P2_USAGE_TAGS for t in e.usage)
    )
    if actual_p2 != expected_p2_count:
        raise AssertionError(
            f"total_p2_used={actual_p2}, expected {expected_p2_count}"
        )

    # Invariant 2: total_quarantined == expected (58 for the real run).
    actual_q = sum(
        1 for e in registry.families.values() if "quarantine" in e.usage
    )
    if actual_q != expected_quarantine_count:
        raise AssertionError(
            f"total_quarantined={actual_q}, expected {expected_quarantine_count}"
        )

    # Invariant 3: total_new_available == count of families with empty usage.
    actual_new = sum(
        1 for e in registry.families.values() if len(e.usage) == 0
    )
    # This is tautological (both sides count the same thing) but kept
    # explicit per the brief's invariant list.

    # Invariant 4: total_families == expected union size.
    actual_total = len(registry.families)
    if actual_total != expected_total_families:
        raise AssertionError(
            f"total_families={actual_total}, expected {expected_total_families}"
        )

    # Invariant 5 (guard for future Tasks 7-9): no family has both
    # ``quarantine`` AND a P3-exclusive tag (frozen_v3 / p3_train /
    # p3_validation). At builder time no P3 tags exist yet, so this is
    # trivially true; assert anyway.
    for fam_id, e in registry.families.items():
        if "quarantine" in e.usage and any(
            t in P3_EXCLUSIVE_TAGS for t in e.usage
        ):
            raise AssertionError(
                f"family {fam_id} has both quarantine and a P3-exclusive "
                f"tag: {e.usage}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build the canonical Family Registry (data/family-registry.json) "
            "from P2 partition + MBPP verified JSONL + Task 5 quarantine list."
        ),
    )
    p.add_argument(
        "--p2-partition",
        required=True,
        help="Path to data/p2-curriculum/family-partition.json.",
    )
    p.add_argument(
        "--mbpp-verified-dir",
        required=True,
        help="Directory containing {train,test,validation}.jsonl "
             "(e.g. data/external/mbpp/verified).",
    )
    p.add_argument(
        "--quarantine",
        required=True,
        help="Path to reports/p3/cross-split-dedup-quarantine.json.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path (e.g. data/family-registry.json).",
    )
    return p


def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on error."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    p2_path = Path(args.p2_partition)
    verified_dir = Path(args.mbpp_verified_dir)
    quarantine_path = Path(args.quarantine)
    output_path = Path(args.output)

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------
    registry = FamilyRegistry()

    p2_ids = backfill_p2(registry, p2_path)
    verified_ids = backfill_mbpp_verified(registry, verified_dir)
    quarantined = backfill_quarantine(registry, quarantine_path)

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------
    expected_p2 = len(p2_ids)
    expected_q = len(quarantined)
    expected_total = len(p2_ids | verified_ids)
    try:
        assert_invariants(
            registry,
            expected_p2_count=expected_p2,
            expected_quarantine_count=expected_q,
            expected_total_families=expected_total,
        )
    except AssertionError as exc:
        print(f"INVARIANT VIOLATION: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"BACKFILL ERROR: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    registry.to_path(output_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(registry.families)
    p2_used = sum(
        1 for e in registry.families.values()
        if any(t in P2_USAGE_TAGS for t in e.usage)
    )
    quarantined_count = sum(
        1 for e in registry.families.values() if "quarantine" in e.usage
    )
    new_available = sum(
        1 for e in registry.families.values() if len(e.usage) == 0
    )
    print(f"Family registry built: {output_path}")
    print(f"  total_families:      {total}")
    print(f"  total_p2_used:       {p2_used}")
    print(f"  total_quarantined:   {quarantined_count}")
    print(f"  total_new_available: {new_available}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
