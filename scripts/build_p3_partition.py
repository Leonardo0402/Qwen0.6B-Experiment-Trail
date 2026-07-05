"""scripts/build_p3_partition.py -- P3 Validation + Train partition orchestrator.

Builds the P3 family partition by claiming three new tags in
``data/family-registry.json`` and writing the canonical partition record to
``data/p3-curriculum/family-partition.json``.

Partition sets (per .superpowers/sdd/task-9-brief.md):
  - p3_validation       90 families  (61 from validation split + 29 sampled
                                      from available test split, seed=42)
  - p3_train            219 families (all remaining available test families)
  - p3_train_replay     N families   (all p2_train families EXCLUDING those
                                      also tagged quarantine; keeps p2_train)

Pairwise disjoint enforcement (Amendment A4):
  - frozen_v3, p3_validation, p3_train, p3_train_replay are pairwise disjoint
  - whitelist: (p3_train_replay, p2_train) -- the 198 replay families are
    intentionally reused from P2 train

Quarantine exclusion (Amendment A9): no P3 split may contain any family
tagged ``quarantine``.

Idempotent: re-running produces the same partition (seed=42, sorted-then-
sampled) and re-claiming is a no-op. Existing ``frozen_v3`` and ``p2_*`` tags
are NOT modified.

Usage
-----
    python scripts/build_p3_partition.py
    python scripts/build_p3_partition.py --registry data/family-registry.json \\
        --output-partition data/p3-curriculum/family-partition.json

Exit codes
----------
    0   success
    1   invariant violation, insufficient pool, or I/O error
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root import guard (so the script works from any cwd)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.family_registry import FamilyRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
GENERATOR_NAME: str = "build_p3_partition.py"
SEED: int = 42

VALIDATION_TARGET: int = 90
TRAIN_NEW_MIN: int = 180
TRAIN_TOTAL_MIN: int = 404

TAG_P3_VALIDATION: str = "p3_validation"
TAG_P3_TRAIN: str = "p3_train"
TAG_P3_TRAIN_REPLAY: str = "p3_train_replay"

TAG_FROZEN_V3: str = "frozen_v3"
TAG_P2_TRAIN: str = "p2_train"
TAG_P2_FROZEN_V2: str = "p2_frozen_v2"
TAG_QUARANTINE: str = "quarantine"

DEFAULT_REGISTRY = _ROOT / "data" / "family-registry.json"
DEFAULT_OUTPUT_PARTITION = _ROOT / "data" / "p3-curriculum" / "family-partition.json"


# ---------------------------------------------------------------------------
# Source pool builders
# ---------------------------------------------------------------------------

def reset_p3_tags(registry: FamilyRegistry) -> None:
    """Unclaim all P3 tags (p3_validation / p3_train / p3_train_replay)
    from every family.

    This makes the script idempotent: each run starts from a clean slate
    before re-deriving the partition. Per the brief, ``frozen_v3`` and
    ``p2_*`` tags are NOT touched (p3_train_replay is removed from p2_train
    families, but p2_train itself stays).
    """
    p3_tags = (TAG_P3_VALIDATION, TAG_P3_TRAIN, TAG_P3_TRAIN_REPLAY)
    for entry in registry.families.values():
        for tag in p3_tags:
            entry.unclaim(tag)


def available_validation_families(registry: FamilyRegistry) -> list[str]:
    """All families with ``source_split=='validation'`` and empty usage.

    These form the base of the P3 validation split (61 families per the
    brief's verified counts).
    """
    return sorted(
        fid for fid, e in registry.families.items()
        if e.source_split == "validation" and not e.is_used()
    )


def available_test_families(registry: FamilyRegistry) -> list[str]:
    """All families with ``source_split=='test'`` and empty usage.

    These are the supplement pool for validation (29 sampled) AND the
    source for the new P3 train split (remaining 219).
    """
    return sorted(
        fid for fid, e in registry.families.items()
        if e.source_split == "test" and not e.is_used()
    )


def p2_train_replay_families(registry: FamilyRegistry) -> tuple[list[str], list[str]]:
    """Return ``(replay_ids, excluded_quarantine_ids)``.

    ``replay_ids`` is the sorted list of p2_train families that are NOT
    also tagged ``quarantine``. ``excluded_quarantine_ids`` is the sorted
    list of p2_train families that ARE also tagged ``quarantine`` (these
    are excluded per Amendment A9).
    """
    replay: list[str] = []
    excluded: list[str] = []
    for fid in registry.families_with_usage(TAG_P2_TRAIN):
        entry = registry.get(fid)
        if entry is None:
            continue
        if TAG_QUARANTINE in entry.usage:
            excluded.append(fid)
        else:
            replay.append(fid)
    return sorted(replay), sorted(excluded)


def sample_validation_supplement(
    available_test_ids: list[str], count: int, seed: int = SEED,
) -> list[str]:
    """Sample ``count`` family_ids from ``available_test_ids`` using
    ``random.Random(seed).sample(sorted_ids, count)``.

    The input list MUST already be sorted ascending (caller's responsibility).
    Returns a sorted list of sampled ids.
    """
    if count > len(available_test_ids):
        raise ValueError(
            f"cannot sample {count} from {len(available_test_ids)} available "
            f"test families"
        )
    rng = random.Random(seed)
    sampled = rng.sample(available_test_ids, count)
    return sorted(sampled)


# ---------------------------------------------------------------------------
# Hard assertions
# ---------------------------------------------------------------------------

def assert_no_overlap_with(set_a: list[str], set_b: list[str], label: str) -> None:
    """Assert that ``set_a`` and ``set_b`` have no common ids."""
    overlap = set(set_a) & set(set_b)
    assert not overlap, (
        f"{label}: {len(overlap)} overlapping family_ids: {sorted(overlap)}"
    )


def assert_no_quarantine(family_ids: list[str], registry: FamilyRegistry, label: str) -> None:
    """Assert that no family in ``family_ids`` is tagged ``quarantine``."""
    bad = []
    for fid in family_ids:
        entry = registry.get(fid)
        if entry is not None and TAG_QUARANTINE in entry.usage:
            bad.append(fid)
    assert not bad, (
        f"{label}: {len(bad)} families are tagged quarantine: {sorted(bad)}"
    )


def assert_no_frozen_v3(family_ids: list[str], registry: FamilyRegistry, label: str) -> None:
    """Assert that no family in ``family_ids`` is tagged ``frozen_v3``."""
    bad = []
    for fid in family_ids:
        entry = registry.get(fid)
        if entry is not None and TAG_FROZEN_V3 in entry.usage:
            bad.append(fid)
    assert not bad, (
        f"{label}: {len(bad)} families are tagged frozen_v3: {sorted(bad)}"
    )


def assert_no_p2_frozen_v2(family_ids: list[str], registry: FamilyRegistry, label: str) -> None:
    """Assert that no family in ``family_ids`` is tagged ``p2_frozen_v2``."""
    bad = []
    for fid in family_ids:
        entry = registry.get(fid)
        if entry is not None and TAG_P2_FROZEN_V2 in entry.usage:
            bad.append(fid)
    assert not bad, (
        f"{label}: {len(bad)} families are tagged p2_frozen_v2: {sorted(bad)}"
    )


def assert_no_p2_train_or_validation(
    family_ids: list[str], registry: FamilyRegistry, label: str,
) -> None:
    """Assert that no family in ``family_ids`` is tagged ``p2_train`` or
    ``p2_validation`` (P3 splits must not overlap with P2 train/validation
    except for the whitelisted replay)."""
    bad = []
    for fid in family_ids:
        entry = registry.get(fid)
        if entry is None:
            continue
        if TAG_P2_TRAIN in entry.usage or "p2_validation" in entry.usage:
            bad.append((fid, list(entry.usage)))
    assert not bad, (
        f"{label}: {len(bad)} families have p2_train/p2_validation tags: "
        f"{bad}"
    )


def assert_all_in_registry(family_ids: list[str], registry: FamilyRegistry, label: str) -> None:
    """Assert every id in ``family_ids`` exists in the registry."""
    missing = [fid for fid in family_ids if fid not in registry.families]
    assert not missing, (
        f"{label}: {len(missing)} family_ids not in registry: {missing}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build the P3 family partition: claim p3_validation / p3_train / "
            "p3_train_replay tags and write family-partition.json."
        ),
    )
    p.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help="Path to data/family-registry.json (input, modified in-place).",
    )
    p.add_argument(
        "--output-partition",
        default=str(DEFAULT_OUTPUT_PARTITION),
        help="Output path for family-partition.json.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="RNG seed for validation supplement sampling (default: 42).",
    )
    return p


def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on error."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    registry_path = Path(args.registry)
    output_partition = Path(args.output_partition)
    seed = args.seed

    # ------------------------------------------------------------------
    # Load registry
    # ------------------------------------------------------------------
    try:
        registry = FamilyRegistry.from_path(registry_path)
    except Exception as exc:
        print(
            f"ERROR: cannot load registry from {registry_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Reset P3 tags (idempotency): clear any previously-claimed
    # p3_validation / p3_train / p3_train_replay so this run starts from a
    # clean slate. frozen_v3 and p2_* tags are NOT touched.
    # ------------------------------------------------------------------
    reset_p3_tags(registry)

    # ------------------------------------------------------------------
    # Build partition sets
    # ------------------------------------------------------------------
    available_validation = available_validation_families(registry)
    available_test = available_test_families(registry)
    replay_ids, replay_excluded_quarantine = p2_train_replay_families(registry)

    n_avail_val = len(available_validation)
    n_avail_test = len(available_test)
    n_p2_train_total = len(registry.families_with_usage(TAG_P2_TRAIN))
    n_replay = len(replay_ids)
    n_replay_excluded = len(replay_excluded_quarantine)

    if n_avail_val > VALIDATION_TARGET:
        print(
            f"ERROR: validation split has {n_avail_val} available families, "
            f"more than target {VALIDATION_TARGET} (no supplement needed).",
            file=sys.stderr,
        )
        return 1

    supplement_count = VALIDATION_TARGET - n_avail_val
    if supplement_count > n_avail_test:
        print(
            f"FIX_FIRST: validation needs {supplement_count} test-split "
            f"families but only {n_avail_test} are available.",
            file=sys.stderr,
        )
        return 1

    try:
        supplement_ids = sample_validation_supplement(
            available_test, supplement_count, seed=seed,
        )
    except ValueError as exc:
        print(f"ERROR: cannot sample validation supplement: {exc}", file=sys.stderr)
        return 1

    p3_validation_ids = sorted(set(available_validation) | set(supplement_ids))

    # P3 Train (new) = remaining available test families
    supplement_set = set(supplement_ids)
    p3_train_new_ids = sorted(
        fid for fid in available_test if fid not in supplement_set
    )

    # P3 Train (replay) = p2_train minus quarantine (already computed)
    p3_train_replay_ids = replay_ids

    n_validation = len(p3_validation_ids)
    n_train_new = len(p3_train_new_ids)
    n_train_replay = len(p3_train_replay_ids)
    n_train_total = n_train_new + n_train_replay

    # ------------------------------------------------------------------
    # Hard gates (pre-claim)
    # ------------------------------------------------------------------
    try:
        # 1. P3 Validation count >= 90
        assert n_validation >= VALIDATION_TARGET, (
            f"P3 Validation count {n_validation} < {VALIDATION_TARGET}"
        )
        # 2. P3 Train (new) count >= 180
        assert n_train_new >= TRAIN_NEW_MIN, (
            f"P3 Train (new) count {n_train_new} < {TRAIN_NEW_MIN}"
        )
        # 3. P3 Train (replay) count == expected
        expected_replay = n_p2_train_total - n_replay_excluded
        assert n_train_replay == expected_replay, (
            f"P3 Train (replay) count {n_train_replay} != expected "
            f"{expected_replay} ({n_p2_train_total} p2_train - "
            f"{n_replay_excluded} quarantine)"
        )
        # 4. P3 Train total >= 404
        assert n_train_total >= TRAIN_TOTAL_MIN, (
            f"P3 Train total {n_train_total} < {TRAIN_TOTAL_MIN}"
        )
        # 7. No family appears in more than one P3 partition set
        assert_no_overlap_with(p3_validation_ids, p3_train_new_ids,
                               "p3_validation ∩ p3_train")
        assert_no_overlap_with(p3_validation_ids, p3_train_replay_ids,
                               "p3_validation ∩ p3_train_replay")
        assert_no_overlap_with(p3_train_new_ids, p3_train_replay_ids,
                               "p3_train ∩ p3_train_replay")
        # All ids exist in registry
        assert_all_in_registry(p3_validation_ids, registry, "p3_validation")
        assert_all_in_registry(p3_train_new_ids, registry, "p3_train_new")
        assert_all_in_registry(p3_train_replay_ids, registry, "p3_train_replay")
    except AssertionError as exc:
        print(f"INVARIANT VIOLATION (pre-claim): {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Claim tags in registry (idempotent)
    # ------------------------------------------------------------------
    for fid in p3_validation_ids:
        registry.claim(fid, TAG_P3_VALIDATION)
    for fid in p3_train_new_ids:
        registry.claim(fid, TAG_P3_TRAIN)
    for fid in p3_train_replay_ids:
        registry.claim(fid, TAG_P3_TRAIN_REPLAY)

    # ------------------------------------------------------------------
    # Pairwise disjoint enforcement (post-claim)
    # ------------------------------------------------------------------
    pairwise_tags = [TAG_FROZEN_V3, TAG_P3_VALIDATION, TAG_P3_TRAIN, TAG_P3_TRAIN_REPLAY]
    pairwise_whitelist = [(TAG_P3_TRAIN_REPLAY, TAG_P2_TRAIN)]
    pairwise_result = "PASS"
    try:
        registry.assert_pairwise_disjoint(
            pairwise_tags, whitelist=pairwise_whitelist,
        )
    except AssertionError as exc:
        pairwise_result = "FAIL"
        print(f"INVARIANT VIOLATION (pairwise disjoint): {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Additional quarantine / p2_frozen_v2 / frozen_v3 disjoint checks
    # ------------------------------------------------------------------
    try:
        # frozen_v3 ∩ quarantine
        frozen_v3_ids = registry.families_with_usage(TAG_FROZEN_V3)
        quarantine_ids = registry.families_with_usage(TAG_QUARANTINE)
        p2_frozen_v2_ids = registry.families_with_usage(TAG_P2_FROZEN_V2)

        assert_no_overlap_with(frozen_v3_ids, quarantine_ids,
                              "frozen_v3 ∩ quarantine")
        assert_no_quarantine(p3_validation_ids, registry,
                             "p3_validation ∩ quarantine")
        assert_no_quarantine(p3_train_new_ids, registry,
                             "p3_train ∩ quarantine")
        assert_no_quarantine(p3_train_replay_ids, registry,
                             "p3_train_replay ∩ quarantine")
        assert_no_overlap_with(p3_validation_ids, p2_frozen_v2_ids,
                               "p3_validation ∩ p2_frozen_v2")
        assert_no_overlap_with(p3_train_new_ids, p2_frozen_v2_ids,
                               "p3_train ∩ p2_frozen_v2")
        assert_no_overlap_with(p3_train_replay_ids, p2_frozen_v2_ids,
                               "p3_train_replay ∩ p2_frozen_v2")
        assert_no_overlap_with(frozen_v3_ids, p2_frozen_v2_ids,
                               "frozen_v3 ∩ p2_frozen_v2")
        # P3 new splits must not contain frozen_v3
        assert_no_frozen_v3(p3_validation_ids, registry,
                            "p3_validation ∩ frozen_v3")
        assert_no_frozen_v3(p3_train_new_ids, registry,
                            "p3_train ∩ frozen_v3")
        assert_no_frozen_v3(p3_train_replay_ids, registry,
                            "p3_train_replay ∩ frozen_v3")
        # P3 validation/train_new must not overlap p2_train/p2_validation
        # (replay is whitelisted for p2_train overlap, but must not have
        # p2_validation either)
        assert_no_p2_train_or_validation(
            p3_validation_ids, registry,
            "p3_validation ∩ {p2_train, p2_validation}",
        )
        assert_no_p2_train_or_validation(
            p3_train_new_ids, registry,
            "p3_train ∩ {p2_train, p2_validation}",
        )
    except AssertionError as exc:
        print(f"INVARIANT VIOLATION (extra disjoint): {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Save registry
    # ------------------------------------------------------------------
    try:
        registry.to_path(registry_path)
    except Exception as exc:
        print(
            f"ERROR: cannot save registry to {registry_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Write family-partition.json
    # ------------------------------------------------------------------
    try:
        output_partition.parent.mkdir(parents=True, exist_ok=True)
        # Whitelist intersection count (the 198/206 replay families)
        replay_set = set(p3_train_replay_ids)
        p2_train_set = set(registry.families_with_usage(TAG_P2_TRAIN))
        whitelist_intersection_count = len(replay_set & p2_train_set)

        # Quarantine exclusion breakdown (by source_split of quarantine)
        quarantine_test = 0
        quarantine_validation = 0
        quarantine_train = 0
        for fid in quarantine_ids:
            entry = registry.get(fid)
            if entry is None:
                continue
            if entry.source_split == "test":
                quarantine_test += 1
            elif entry.source_split == "validation":
                quarantine_validation += 1
            elif entry.source_split == "train":
                quarantine_train += 1

        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": GENERATOR_NAME,
            "seed": seed,
            "p3_validation": {
                "count": n_validation,
                "from_validation_split": n_avail_val,
                "from_test_split_supplement": supplement_count,
                "family_ids": p3_validation_ids,
            },
            "p3_train_new": {
                "count": n_train_new,
                "source": "test split (remaining after validation supplement)",
                "family_ids": p3_train_new_ids,
            },
            "p3_train_replay": {
                "count": n_train_replay,
                "source": "p2_train minus quarantine",
                "excluded_quarantine_count": n_replay_excluded,
                "family_ids": p3_train_replay_ids,
            },
            "p3_train_total": n_train_total,
            "frozen_v3_count": len(frozen_v3_ids),
            "pairwise_disjoint": {
                "tags_checked": pairwise_tags,
                "pairs_checked": 6,
                "whitelist": [list(pair) for pair in pairwise_whitelist],
                "whitelist_intersection_count": whitelist_intersection_count,
                "result": pairwise_result,
            },
            "quarantine_exclusion": {
                "quarantine_total": len(quarantine_ids),
                "excluded_from_validation": quarantine_validation,
                "excluded_from_train_new": quarantine_test,
                "excluded_from_train_replay": n_replay_excluded,
                "excluded_from_frozen_v3": 0,
            },
        }
        with output_partition.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except Exception as exc:
        print(
            f"ERROR: cannot write partition to {output_partition}: {exc}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"P3 partition written: {output_partition}")
    print(f"  p3_validation:        {n_validation} "
          f"(validation_split={n_avail_val}, test_supplement={supplement_count})")
    print(f"  p3_train_new:         {n_train_new}")
    print(f"  p3_train_replay:      {n_train_replay} "
          f"(p2_train={n_p2_train_total}, excluded_quarantine={n_replay_excluded})")
    print(f"  p3_train_total:       {n_train_total}")
    print(f"  frozen_v3_count:      {len(frozen_v3_ids)}")
    print(f"  pairwise_disjoint:    {pairwise_result} "
          f"(whitelist intersection={whitelist_intersection_count})")
    print(f"  quarantine_exclusion: total={len(quarantine_ids)} "
          f"(val={quarantine_validation}, test={quarantine_test}, "
          f"replay={n_replay_excluded}, frozen_v3=0)")
    print(f"  registry updated:     {registry_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
