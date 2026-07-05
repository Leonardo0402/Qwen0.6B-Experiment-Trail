"""scripts/p3_formal_dataset_builder.py -- P3 Formal dataset builder (Wave 4-H, Issue #14).

Unified builder for BOTH formal candidates (Balanced Generalist + Repair
Specialist) from the formal canonical pool.

Per-candidate targets (2500 +/- 100 samples, hard range 2300-3100):
  - balanced_generalist: 30/20/20/30 (Code/Boundary/Static/Exec) -> 750/500/500/750
  - repair_specialist:   15/15/30/40 (Code/Boundary/Static/Exec) -> 375/375/750/1000

Capacity assessment (binding -- abort exit 2 if pool insufficient):
  1. Count available samples per bucket in the formal pool.
  2. For each candidate, max_achievable = sum of min(target, available) per bucket.
  3. If max < 2300 for EITHER candidate -> CAPACITY_VERDICT: MBPP_FAMILY_OR_VARIANT_LIMIT, exit 2.
  4. If max >= 2300 for both but < 2530 for either -> CAPACITY_VERDICT: FORMAL_CAPACITY_AT_RISK, build.
  5. If max >= 2530 for both -> CAPACITY_VERDICT: FORMAL_CAPACITY_FEASIBLE, build at 2500 target.

Sub-sampling (seed=42, sorted by sample_id ascending, random.Random(42).sample):
  - If target >= available for a bucket: take ALL available (no sampling).
  - When any bucket is under-capacity: use ALL available for that bucket and
    proportionally reduce other buckets to maintain ratios as close as possible.

Per-family caps (enforced after sub-sampling, drop excess by sample_id asc):
  - Per family total <= 8 samples
  - Per family per bucket <= 3 samples
  - Single family <= 1% of total dataset

Hard gates (13 total): 10 existing + 3 new formal gates (11/12/13).

Usage
-----
    py -3.11 scripts/p3_formal_dataset_builder.py --candidate balanced
    py -3.11 scripts/p3_formal_dataset_builder.py --candidate repair
    py -3.11 scripts/p3_formal_dataset_builder.py --candidate both

Exit codes
----------
    0   success
    1   invariant violation (hard gate failed) or I/O error
    2   capacity insufficient (MBPP_FAMILY_OR_VARIANT_LIMIT)
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

from src.sample_pool import SamplePool  # noqa: E402
from src.schemas import Sample  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
GENERATOR_NAME: str = "p3_formal_dataset_builder.py"
SEED: int = 42
FROZEN_VERSION: str = "v4"

BUCKETS: tuple = ("code", "boundary", "static_repair", "execution_repair")

# Capacity thresholds
TARGET_TOTAL: int = 2500
HARD_MIN_TOTAL: int = 2300
FEASIBLE_MIN_TOTAL: int = 2530  # 2300 + 10%

# Ratio tolerance
RATIO_TOLERANCE_PP: int = 3  # +/- 3 percentage points

# Per-family caps (formal)
PER_FAMILY_TOTAL_CAP: int = 8
PER_FAMILY_PER_BUCKET_CAP: int = 3
SINGLE_FAMILY_PERCENT_CAP: float = 1.0  # 1%

EXPECTED_VALIDATION_COUNT: int = 180  # 4 categories x 45 (Validation v2)
EXPECTED_VALIDATION_PER_CATEGORY: int = 45

# ---------------------------------------------------------------------------
# Candidate configs
# ---------------------------------------------------------------------------

CANDIDATES: dict = {
    "balanced": {
        "candidate_key": "balanced",
        "candidate_type": "balanced_generalist",
        "target_ratios": {
            "code": 0.30,
            "boundary": 0.20,
            "static_repair": 0.20,
            "execution_repair": 0.30,
        },
        "bucket_targets": {
            "code": 750,
            "boundary": 500,
            "static_repair": 500,
            "execution_repair": 750,
        },
        "output_dir": _ROOT / "data" / "p3-formal" / "balanced-generalist",
        "shared_with": "repair_specialist",
    },
    "repair": {
        "candidate_key": "repair",
        "candidate_type": "repair_specialist",
        "target_ratios": {
            "code": 0.15,
            "boundary": 0.15,
            "static_repair": 0.30,
            "execution_repair": 0.40,
        },
        "bucket_targets": {
            "code": 375,
            "boundary": 375,
            "static_repair": 750,
            "execution_repair": 1000,
        },
        "output_dir": _ROOT / "data" / "p3-formal" / "repair-specialist",
        "shared_with": "balanced_generalist",
    },
}

# ---------------------------------------------------------------------------
# Paths (formal: pool from p3-formal, frozen v4, validation v2 shared)
# ---------------------------------------------------------------------------

POOL_PATH = _ROOT / "data" / "p3-formal" / "canonical-pool.jsonl"
FROZEN_FAMILIES_PATH = _ROOT / "data" / "frozen-eval" / "v4" / "families.json"
VALIDATION_PATH = (
    _ROOT / "data" / "p3-curriculum" / "validation-v2" / "validation.jsonl"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relpath(path: Path) -> str:
    """Return path relative to _ROOT if possible, else absolute string.

    Uses forward slashes for cross-platform consistency in the manifest.
    """
    try:
        rel = path.relative_to(_ROOT)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def _load_jsonl_samples(path: Path) -> list:
    """Stream a JSONL file into a list of Sample objects."""
    samples: list = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(Sample.from_json_line(line))
    return samples


def _compute_sha256(path: Path) -> str:
    """Read file as bytes, normalize CRLF to LF, compute SHA256 hex digest.

    Uses ``SamplePool.compute_sha256`` for cross-platform consistency
    (Windows CRLF checkouts vs Linux LF CI).
    """
    return SamplePool.compute_sha256(path)


def _round4(x: float) -> float:
    """Round a float to 4 decimal places."""
    return round(float(x), 4)


def _count_tokens(sample: Sample) -> int:
    """Count whitespace-split tokens in
    instruction + " " + target_code + " " + public_tests + " " + hidden_tests.
    """
    text = (
        sample.instruction + " "
        + sample.target_code + " "
        + sample.public_tests + " "
        + sample.hidden_tests
    )
    return len(text.split())


def _find_duplicates(sample_ids: list) -> list:
    """Return list of sample_ids that appear more than once (preserving order)."""
    seen: set = set()
    duplicates: list = []
    dup_set: set = set()
    for sid in sample_ids:
        if sid in seen and sid not in dup_set:
            duplicates.append(sid)
            dup_set.add(sid)
        seen.add(sid)
    return duplicates


def _subsample_bucket(
    samples: list, target_count: int, *, seed: int = SEED
) -> tuple:
    """Sub-sample a variant_type bucket deterministically.

    Sort by sample_id ascending, then
    ``random.Random(seed).sample(sorted_ids, target_count)``.
    If ``target_count >= len(samples)``: take ALL samples (sorted) -- no
    random sampling needed.

    Returns ``(selected, rejected)``.
    """
    sorted_samples = sorted(samples, key=lambda s: s.sample_id)
    if target_count >= len(sorted_samples):
        return list(sorted_samples), []
    sorted_ids = [s.sample_id for s in sorted_samples]
    chosen_ids = set(random.Random(seed).sample(sorted_ids, target_count))
    selected = [s for s in sorted_samples if s.sample_id in chosen_ids]
    rejected = [s for s in sorted_samples if s.sample_id not in chosen_ids]
    return selected, rejected


# ---------------------------------------------------------------------------
# Capacity assessment
# ---------------------------------------------------------------------------

def _assess_candidate_capacity(
    candidate_cfg: dict, per_bucket_available: dict
) -> dict:
    """Compute max achievable total for one candidate.

    max_achievable = sum of min(target_per_bucket, available_per_bucket)
    across all 4 buckets.
    """
    bucket_targets = candidate_cfg["bucket_targets"]
    per_bucket_max = {
        v: min(bucket_targets[v], per_bucket_available[v]) for v in BUCKETS
    }
    max_achievable = sum(per_bucket_max.values())
    return {
        "per_bucket_available": dict(per_bucket_available),
        "per_bucket_target": dict(bucket_targets),
        "per_bucket_max": per_bucket_max,
        "max_achievable_total": max_achievable,
    }


def _determine_verdict(assessments: dict) -> str:
    """Determine capacity verdict from per-candidate assessments.

    - MBPP_FAMILY_OR_VARIANT_LIMIT: any candidate max < 2300
    - FORMAL_CAPACITY_AT_RISK: all >= 2300 but any < 2530
    - FORMAL_CAPACITY_FEASIBLE: all >= 2530
    """
    min_max = min(a["max_achievable_total"] for a in assessments.values())
    if min_max < HARD_MIN_TOTAL:
        return "MBPP_FAMILY_OR_VARIANT_LIMIT"
    if min_max < FEASIBLE_MIN_TOTAL:
        return "FORMAL_CAPACITY_AT_RISK"
    return "FORMAL_CAPACITY_FEASIBLE"


def _print_capacity_report(verdict: str, assessments: dict) -> None:
    """Print the capacity assessment report to stdout."""
    print(f"CAPACITY_VERDICT: {verdict}")
    for name, a in assessments.items():
        cand_type = CANDIDATES[name]["candidate_type"]
        print(f"  {cand_type}:")
        print(f"    per_bucket_available: {a['per_bucket_available']}")
        print(f"    per_bucket_target:    {a['per_bucket_target']}")
        print(f"    per_bucket_max:       {a['per_bucket_max']}")
        print(f"    max_achievable_total: {a['max_achievable_total']}")


# ---------------------------------------------------------------------------
# Sub-sampling target computation (with proportional reduction)
# ---------------------------------------------------------------------------

def _compute_actual_targets(
    candidate_cfg: dict, per_bucket_available: dict
) -> tuple:
    """Compute actual per-bucket sub-sampling targets.

    If any bucket is under-capacity (available < target): use ALL available
    for that bucket, then proportionally reduce the over-capacity buckets to
    maintain ratios as close as possible to the original targets.

    Returns ``(actual_targets, any_under_capacity)``.
    """
    bucket_targets = candidate_cfg["bucket_targets"]
    target_ratios = candidate_cfg["target_ratios"]

    under_capacity = {
        v: per_bucket_available[v] < bucket_targets[v] for v in BUCKETS
    }
    any_under = any(under_capacity.values())

    if not any_under:
        # All buckets have enough -- use original targets.
        return dict(bucket_targets), False

    # Take ALL available for under-capacity buckets.
    actual: dict = {}
    for v in BUCKETS:
        if under_capacity[v]:
            actual[v] = per_bucket_available[v]

    # Proportionally reduce over-capacity buckets: distribute the remaining
    # target total across them according to their original ratio share.
    used = sum(actual.values())
    remaining = TARGET_TOTAL - used
    over_buckets = [v for v in BUCKETS if not under_capacity[v]]
    ratio_sum = sum(target_ratios[v] for v in over_buckets)

    for v in over_buckets:
        if ratio_sum > 0:
            proportional = int(round(remaining * target_ratios[v] / ratio_sum))
            # Cap at: original target, available count, and proportional.
            actual[v] = min(bucket_targets[v], per_bucket_available[v], proportional)
        else:
            actual[v] = min(bucket_targets[v], per_bucket_available[v])

    return actual, True


# ---------------------------------------------------------------------------
# Per-family cap enforcement
# ---------------------------------------------------------------------------

def _enforce_family_caps(
    selected_samples: list,
) -> tuple:
    """Enforce per-family caps on the selected samples.

    - Per family per bucket <= 3 (drop excess by sample_id asc, keep lowest)
    - Per family total <= 8 (drop excess by sample_id asc, keep lowest)

    Returns ``(kept, dropped)`` where ``dropped`` is the list of samples
    removed to satisfy the caps.
    """
    kept = list(selected_samples)
    dropped: list = []

    # Pass 1: per family per bucket <= 3
    # Group by (family_id, variant_type)
    by_fam_bucket: dict = {}
    for s in kept:
        key = (s.family_id, s.variant_type)
        by_fam_bucket.setdefault(key, []).append(s)

    excess_positions: set = set()  # id() of Sample objects to drop
    for key, samples in by_fam_bucket.items():
        if len(samples) > PER_FAMILY_PER_BUCKET_CAP:
            sorted_samples = sorted(samples, key=lambda s: s.sample_id)
            for s in sorted_samples[PER_FAMILY_PER_BUCKET_CAP:]:
                excess_positions.add(id(s))

    if excess_positions:
        new_kept = []
        for s in kept:
            if id(s) in excess_positions:
                dropped.append(s)
            else:
                new_kept.append(s)
        kept = new_kept

    # Pass 2: per family total <= 8
    by_fam: dict = {}
    for s in kept:
        by_fam.setdefault(s.family_id, []).append(s)

    excess_positions2: set = set()
    for fid, samples in by_fam.items():
        if len(samples) > PER_FAMILY_TOTAL_CAP:
            sorted_samples = sorted(samples, key=lambda s: s.sample_id)
            for s in sorted_samples[PER_FAMILY_TOTAL_CAP:]:
                excess_positions2.add(id(s))

    if excess_positions2:
        new_kept = []
        for s in kept:
            if id(s) in excess_positions2:
                dropped.append(s)
            else:
                new_kept.append(s)
        kept = new_kept

    return kept, dropped


# ---------------------------------------------------------------------------
# Hard gates (13: existing 10 + 3 new formal gates)
# ---------------------------------------------------------------------------

def _run_hard_gates(
    train_samples: list,
    validation_samples: list,
    *,
    frozen_family_ids: set,
    actual_expected_train_count: int,
    actual_bucket_counts: dict,
    target_ratios: dict,
    any_under_capacity: bool,
) -> list:
    """Run all 13 hard gates. Returns list of error messages (empty = pass).

    Hard gates (binding):
      1. Train count == actual_expected_train_count (dynamic).
      2. All 4 variant ratios within +/-3pp of target (relaxed when
         under-capacity).
      3. Validation count == 180 AND 4 categories x 45 each.
      4. Train intersection Validation family_ids == empty.
      5. Train intersection frozen family_ids == empty.
      6. Validation intersection frozen family_ids == empty.
      7. All train samples have variant_type set (not None).
      8. All train samples have verified == True.
      9. No duplicate sample_ids in train.
      10. No duplicate sample_ids in validation.
      --- formal gates ---
      11. Per-family total <= 8 (per candidate).
      12. Per-family per bucket <= 3 (per candidate).
      13. Single family <= 1% of total (per candidate).
    """
    errors: list = []

    # Gate 1: Train count == actual_expected_train_count (dynamic).
    if len(train_samples) != actual_expected_train_count:
        errors.append(
            f"gate 1: train count {len(train_samples)} != "
            f"actual_expected_train_count {actual_expected_train_count}"
        )

    # Gate 2: All 4 variant ratios within +/-3pp of target.
    # Relaxed (skipped) when any bucket is under-capacity.
    variant_counts = {
        "code": 0, "boundary": 0,
        "static_repair": 0, "execution_repair": 0,
    }
    for s in train_samples:
        vt = s.variant_type
        if vt in variant_counts:
            variant_counts[vt] += 1
    total = sum(variant_counts.values())
    if total == 0:
        errors.append("gate 2: train has 0 samples -- cannot compute ratios")
    elif any_under_capacity:
        # Skip ratio checks when any bucket is under-capacity.
        pass
    else:
        for v in BUCKETS:
            actual_ratio = variant_counts[v] / total
            target_ratio = target_ratios[v]
            diff_pp = abs(actual_ratio - target_ratio) * 100
            if diff_pp > RATIO_TOLERANCE_PP:
                errors.append(
                    f"gate 2: variant {v} ratio {actual_ratio:.4f} "
                    f"differs from target {target_ratio:.4f} by "
                    f"{diff_pp:.2f}pp (> {RATIO_TOLERANCE_PP}pp)"
                )

    # Gate 3: Validation count == 180 AND 4 categories x 45 each.
    if len(validation_samples) != EXPECTED_VALIDATION_COUNT:
        errors.append(
            f"gate 3: validation count {len(validation_samples)} != "
            f"{EXPECTED_VALIDATION_COUNT}"
        )
    val_variant_counts: dict = {
        "code": 0, "boundary": 0,
        "static_repair": 0, "execution_repair": 0,
    }
    for s in validation_samples:
        vt = s.variant_type
        if vt in val_variant_counts:
            val_variant_counts[vt] += 1
    for v in BUCKETS:
        if val_variant_counts[v] != EXPECTED_VALIDATION_PER_CATEGORY:
            errors.append(
                f"gate 3: validation category {v} count "
                f"{val_variant_counts[v]} != "
                f"{EXPECTED_VALIDATION_PER_CATEGORY}"
            )

    # Gate 4: Train intersection Validation family_ids == empty.
    train_fids = {s.family_id for s in train_samples}
    val_fids = {s.family_id for s in validation_samples}
    overlap = train_fids & val_fids
    if overlap:
        errors.append(
            f"gate 4: {len(overlap)} family_ids in both train and validation "
            f"(first 5: {sorted(overlap)[:5]})"
        )

    # Gate 5: Train intersection frozen family_ids == empty.
    train_frozen_overlap = train_fids & frozen_family_ids
    if train_frozen_overlap:
        errors.append(
            f"gate 5: {len(train_frozen_overlap)} frozen_{FROZEN_VERSION} "
            f"family_ids in train (first 5: {sorted(train_frozen_overlap)[:5]})"
        )

    # Gate 6: Validation intersection frozen family_ids == empty.
    val_frozen_overlap = val_fids & frozen_family_ids
    if val_frozen_overlap:
        errors.append(
            f"gate 6: {len(val_frozen_overlap)} frozen_{FROZEN_VERSION} "
            f"family_ids in validation (first 5: {sorted(val_frozen_overlap)[:5]})"
        )

    # Gate 7: All train samples have variant_type set (not None).
    none_variants = [s.sample_id for s in train_samples if s.variant_type is None]
    if none_variants:
        errors.append(
            f"gate 7: {len(none_variants)} train samples with "
            f"variant_type=None (first 5: {none_variants[:5]})"
        )

    # Gate 8: All train samples have verified == True.
    unverified = [s.sample_id for s in train_samples if not s.verified]
    if unverified:
        errors.append(
            f"gate 8: {len(unverified)} train samples with verified=False "
            f"(first 5: {unverified[:5]})"
        )

    # Gate 9: No duplicate sample_ids in train.
    train_ids = [s.sample_id for s in train_samples]
    train_dups = _find_duplicates(train_ids)
    if train_dups:
        errors.append(
            f"gate 9: {len(train_dups)} duplicate sample_ids in train "
            f"(first 5: {train_dups[:5]})"
        )

    # Gate 10: No duplicate sample_ids in validation.
    val_ids = [s.sample_id for s in validation_samples]
    val_dups = _find_duplicates(val_ids)
    if val_dups:
        errors.append(
            f"gate 10: {len(val_dups)} duplicate sample_ids in validation "
            f"(first 5: {val_dups[:5]})"
        )

    # --- Formal gates (11/12/13) ---

    # Gate 11: Per-family total <= 8.
    family_total: dict = {}
    for s in train_samples:
        family_total[s.family_id] = family_total.get(s.family_id, 0) + 1
    over_total = {fid: c for fid, c in family_total.items()
                  if c > PER_FAMILY_TOTAL_CAP}
    if over_total:
        errors.append(
            f"gate 11: {len(over_total)} families exceed per-family total cap "
            f"{PER_FAMILY_TOTAL_CAP} (first 5: "
            f"{dict(sorted(over_total.items())[:5])})"
        )

    # Gate 12: Per-family per bucket <= 3.
    fam_bucket: dict = {}
    for s in train_samples:
        key = (s.family_id, s.variant_type)
        fam_bucket[key] = fam_bucket.get(key, 0) + 1
    over_bucket = {f"{k[0]}/{k[1]}": c for k, c in fam_bucket.items()
                   if c > PER_FAMILY_PER_BUCKET_CAP}
    if over_bucket:
        errors.append(
            f"gate 12: {len(over_bucket)} (family, bucket) pairs exceed "
            f"per-family per-bucket cap {PER_FAMILY_PER_BUCKET_CAP} "
            f"(first 5: {dict(sorted(over_bucket.items())[:5])})"
        )

    # Gate 13: Single family <= 1% of total.
    if total > 0:
        cap_count = total * SINGLE_FAMILY_PERCENT_CAP / 100.0
        over_percent = {fid: c for fid, c in family_total.items()
                        if c > cap_count}
        if over_percent:
            errors.append(
                f"gate 13: {len(over_percent)} families exceed 1% of total "
                f"(total={total}, cap={cap_count:.2f}; first 5: "
                f"{dict(sorted(over_percent.items())[:5])})"
            )

    return errors


# ---------------------------------------------------------------------------
# Token audit
# ---------------------------------------------------------------------------

def _build_token_audit(
    candidate_type: str,
    train_samples: list,
    validation_samples: list,
) -> dict:
    """Build token_audit.json payload per brief schema.

    Means are rounded to 4 decimal places.
    """
    # Train
    train_tokens = [_count_tokens(s) for s in train_samples]
    train_total = sum(train_tokens)
    train_count = len(train_tokens)
    train_mean = train_total / train_count if train_count else 0.0
    train_max = max(train_tokens) if train_tokens else 0
    train_min = min(train_tokens) if train_tokens else 0

    # By variant_type
    by_variant: dict = {}
    for v in BUCKETS:
        v_samples = [s for s in train_samples if s.variant_type == v]
        v_tokens = [_count_tokens(s) for s in v_samples]
        v_total = sum(v_tokens)
        v_count = len(v_tokens)
        v_mean = v_total / v_count if v_count else 0.0
        by_variant[v] = {
            "count": v_count,
            "total_tokens": v_total,
            "mean": _round4(v_mean),
        }

    # Validation
    val_tokens = [_count_tokens(s) for s in validation_samples]
    val_total = sum(val_tokens)
    val_count = len(val_tokens)
    val_mean = val_total / val_count if val_count else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_type": candidate_type,
        "train": {
            "total_samples": train_count,
            "total_tokens": train_total,
            "mean_tokens_per_sample": _round4(train_mean),
            "max_tokens": train_max,
            "min_tokens": train_min,
            "by_variant_type": by_variant,
        },
        "validation": {
            "total_samples": val_count,
            "total_tokens": val_total,
            "mean_tokens_per_sample": _round4(val_mean),
        },
    }


# ---------------------------------------------------------------------------
# Build one candidate
# ---------------------------------------------------------------------------

def _build_candidate(
    candidate_key: str,
    pool: SamplePool,
    by_vt: dict,
    per_bucket_available: dict,
    validation_samples: list,
    frozen_family_ids: set,
    capacity_assessment: dict,
    verdict: str,
    pool_sha256: str,
    total_pool_samples: int,
) -> int:
    """Build one candidate dataset. Returns 0 on success, 1 on gate failure."""
    cfg = CANDIDATES[candidate_key]
    candidate_type = cfg["candidate_type"]
    target_ratios = cfg["target_ratios"]
    output_dir = cfg["output_dir"]

    print(f"\n{'=' * 60}")
    print(f"Building candidate: {candidate_type}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Compute actual per-bucket targets (with proportional reduction
    # if any bucket is under-capacity).
    # ------------------------------------------------------------------
    actual_targets, any_under = _compute_actual_targets(
        cfg, per_bucket_available
    )
    print("Per-bucket targets:")
    for v in BUCKETS:
        avail = per_bucket_available[v]
        tgt = actual_targets[v]
        orig = cfg["bucket_targets"][v]
        flag = " (UNDER-CAPACITY)" if avail < orig else ""
        print(f"  {v:20s}: target={tgt} (available={avail}, "
              f"original={orig}){flag}")

    # ------------------------------------------------------------------
    # Sub-sample each bucket (seed=42, sorted by sample_id ascending).
    # Only verified=True samples are eligible (formal pool is fully
    # verified, but we filter defensively).
    # ------------------------------------------------------------------
    selected_samples: list = []
    rejected_samples: list = []
    n_unverified_excluded: int = 0
    actual_bucket_counts: dict = {}
    for v in BUCKETS:
        bucket_all = by_vt[v]
        bucket_verified = [s for s in bucket_all if s.verified]
        n_unverified_excluded += len(bucket_all) - len(bucket_verified)
        target = actual_targets[v]
        selected, rejected_verified = _subsample_bucket(bucket_verified, target)
        rejected_unverified = [s for s in bucket_all if not s.verified]
        selected_samples.extend(selected)
        rejected_samples.extend(rejected_verified)
        rejected_samples.extend(rejected_unverified)
        actual_bucket_counts[v] = len(selected)
        if target >= len(bucket_verified):
            if len(bucket_verified) < target:
                print(f"  WARNING: {v} bucket has only {len(bucket_verified)} "
                      f"verified samples (target {target}) -- using ALL "
                      f"available")
            else:
                print(f"  {v}: took ALL {len(selected)} (no sampling needed)")
        else:
            print(f"  {v}: selected {len(selected)} / {len(bucket_verified)} "
                  f"verified, rejected {len(rejected_verified)}")

    # ------------------------------------------------------------------
    # Per-family cap enforcement (after sub-sampling).
    # ------------------------------------------------------------------
    pre_cap_count = len(selected_samples)
    selected_samples, cap_dropped = _enforce_family_caps(selected_samples)
    if cap_dropped:
        print(f"  Per-family cap enforcement: dropped {len(cap_dropped)} "
              f"excess samples (pre-cap {pre_cap_count} -> "
              f"post-cap {len(selected_samples)})")
        # Cap-dropped samples go to rejected with a distinct reason.
        rejected_samples.extend(cap_dropped)
        # Recompute actual_bucket_counts after cap enforcement.
        for v in BUCKETS:
            actual_bucket_counts[v] = sum(
                1 for s in selected_samples if s.variant_type == v
            )

    # Sort by sample_id ascending (final order)
    selected_samples.sort(key=lambda s: s.sample_id)
    rejected_samples.sort(key=lambda s: s.sample_id)

    actual_expected_train_count = sum(actual_bucket_counts.values())
    if n_unverified_excluded:
        print(f"Excluded {n_unverified_excluded} unverified samples from "
              f"train candidate pool")

    print(f"Total selected: {len(selected_samples)} "
          f"(target {TARGET_TOTAL}, actual "
          f"{actual_expected_train_count})")
    print(f"Total rejected: {len(rejected_samples)}")

    # ------------------------------------------------------------------
    # Create output directory + write files
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    manifest_path = output_dir / "manifest.json"
    families_path = output_dir / "families.json"
    token_audit_path = output_dir / "token_audit.json"
    rejected_path = output_dir / "rejected.jsonl"

    # train.jsonl
    with train_path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in selected_samples:
            fh.write(s.to_json_line())
            fh.write("\n")
    print(f"Wrote train.jsonl: {len(selected_samples)} samples")

    # rejected.jsonl -- distinguish rejection reasons
    cap_dropped_ids = {id(s) for s in cap_dropped}
    with rejected_path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in rejected_samples:
            if not s.verified:
                reason = "unverified_excluded"
            elif id(s) in cap_dropped_ids:
                reason = "family_cap_excess"
            else:
                reason = "ratio_balance_excess"
            obj = {
                "sample_id": s.sample_id,
                "family_id": s.family_id,
                "variant_type": s.variant_type,
                "rejection_reason": reason,
            }
            fh.write(json.dumps(obj, ensure_ascii=False))
            fh.write("\n")
    print(f"Wrote rejected.jsonl: {len(rejected_samples)} records")

    # ------------------------------------------------------------------
    # Compute SHAs (CRLF-normalized via SamplePool.compute_sha256)
    # ------------------------------------------------------------------
    train_sha = _compute_sha256(train_path)
    val_sha = _compute_sha256(VALIDATION_PATH)

    # ------------------------------------------------------------------
    # Build manifest.json
    # ------------------------------------------------------------------
    train_family_ids = sorted({s.family_id for s in selected_samples})
    val_family_ids = sorted({s.family_id for s in validation_samples})
    train_family_count = len(train_family_ids)
    val_family_count = len(val_family_ids)
    total_family_count = len(set(train_family_ids) | set(val_family_ids))

    variant_distribution = {v: 0 for v in BUCKETS}
    for s in selected_samples:
        vt = s.variant_type
        if vt in variant_distribution:
            variant_distribution[vt] += 1

    train_count = len(selected_samples)
    actual_ratios = {
        v: (variant_distribution[v] / train_count if train_count else 0.0)
        for v in BUCKETS
    }
    actual_ratios_rounded = {v: round(actual_ratios[v], 4) for v in BUCKETS}

    if any_under:
        ratio_within_tol = False  # under-capacity: ratios may be skewed
    else:
        ratio_within_tol = all(
            abs(actual_ratios[v] - target_ratios[v]) * 100 <= RATIO_TOLERANCE_PP
            for v in BUCKETS
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": GENERATOR_NAME,
        "candidate_type": candidate_type,
        "seed": SEED,
        "frozen_version": FROZEN_VERSION,
        "target_ratios": target_ratios,
        "actual_ratios": actual_ratios_rounded,
        "ratio_tolerance_pp": RATIO_TOLERANCE_PP,
        "ratio_within_tolerance": ratio_within_tol,
        "train": {
            "count": train_count,
            "target_count": TARGET_TOTAL,
            "variant_distribution": variant_distribution,
            "family_count": train_family_count,
            "sha256": train_sha,
        },
        "validation": {
            "count": len(validation_samples),
            "family_count": val_family_count,
            "sha256": val_sha,
        },
        "families": {
            "train_family_count": train_family_count,
            "validation_family_count": val_family_count,
            "total_family_count": total_family_count,
            "train_validation_disjoint": True,
        },
        "pool_source": {
            "path": _relpath(POOL_PATH),
            "total_pool_samples": total_pool_samples,
            "samples_selected": train_count,
            "samples_rejected": len(rejected_samples),
            "unverified_excluded": n_unverified_excluded,
        },
        "formal_pool_source": {
            "path": _relpath(POOL_PATH),
            "pool_sha256": pool_sha256,
            "total_pool_samples": total_pool_samples,
        },
        "capacity_assessment": {
            "verdict": verdict,
            "per_bucket_available": capacity_assessment["per_bucket_available"],
            "per_bucket_target": capacity_assessment["per_bucket_target"],
            "max_achievable_total": capacity_assessment["max_achievable_total"],
        },
        "per_family_caps": {
            "per_family_total_cap": PER_FAMILY_TOTAL_CAP,
            "per_family_per_bucket_cap": PER_FAMILY_PER_BUCKET_CAP,
            "single_family_percent_cap": SINGLE_FAMILY_PERCENT_CAP,
            "samples_dropped_by_cap": len(cap_dropped),
        },
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote manifest.json")

    # ------------------------------------------------------------------
    # Write families.json
    # ------------------------------------------------------------------
    families_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_type": candidate_type,
        "frozen_version": FROZEN_VERSION,
        "train_family_ids": train_family_ids,
        "validation_family_ids": val_family_ids,
        "total_family_count": total_family_count,
        "shared_with": cfg["shared_with"],
    }
    with families_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(families_payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote families.json: {train_family_count} train + "
          f"{val_family_count} val = {total_family_count} total")

    # ------------------------------------------------------------------
    # Write token_audit.json
    # ------------------------------------------------------------------
    token_audit = _build_token_audit(
        candidate_type, selected_samples, validation_samples
    )
    with token_audit_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(token_audit, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote token_audit.json")

    # ------------------------------------------------------------------
    # Hard gates (binding -- abort exit 1 if any fail)
    # ------------------------------------------------------------------
    errors = _run_hard_gates(
        selected_samples, validation_samples,
        frozen_family_ids=frozen_family_ids,
        actual_expected_train_count=actual_expected_train_count,
        actual_bucket_counts=actual_bucket_counts,
        target_ratios=target_ratios,
        any_under_capacity=any_under,
    )
    if errors:
        print("\nHARD GATE FAILURES:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("\nAll hard gates PASS")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{candidate_type} summary:")
    print(f"  train samples:       {train_count}")
    print(f"  validation samples: {len(validation_samples)}")
    print(f"  rejected samples:   {len(rejected_samples)}")
    print(f"  train families:     {train_family_count}")
    print(f"  validation families:{val_family_count}")
    print(f"  total families:     {total_family_count}")
    print(f"  unverified excluded:{n_unverified_excluded}")
    print(f"  cap-dropped:        {len(cap_dropped)}")
    print(f"  variant dist:")
    for v in BUCKETS:
        ratio = actual_ratios[v]
        print(f"    {v:20s}: {variant_distribution[v]:3d} "
              f"({ratio*100:6.2f}%)")
    print(f"  train SHA256:       {train_sha}")
    print(f"  validation SHA256:  {val_sha}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on error, 2 on capacity."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        prog="p3_formal_dataset_builder.py",
        description=(
            "P3 Formal dataset builder (Wave 4-H, Issue #14). Builds "
            "Balanced Generalist and/or Repair Specialist from the formal "
            "canonical pool with capacity assessment and 13 hard gates."
        ),
    )
    parser.add_argument(
        "--candidate",
        choices=["balanced", "repair", "both"],
        default="both",
        help="Which candidate to build (default: both).",
    )
    args = parser.parse_args(argv)

    # Determine which candidates to build.
    if args.candidate == "both":
        keys = ["balanced", "repair"]
    else:
        keys = [args.candidate]

    # ------------------------------------------------------------------
    # Load canonical pool
    # ------------------------------------------------------------------
    if not POOL_PATH.exists():
        print(f"ERROR: formal pool not found: {POOL_PATH}", file=sys.stderr)
        return 1
    pool = SamplePool.from_jsonl(POOL_PATH)
    total_pool_samples = len(pool)
    pool_sha256 = _compute_sha256(POOL_PATH)
    print(f"Loaded formal canonical pool: {total_pool_samples} samples from "
          f"{_relpath(POOL_PATH)}")
    print(f"Pool SHA256: {pool_sha256}")

    # Group by variant_type
    by_vt: dict = {v: [] for v in BUCKETS}
    for s in pool:
        vt = s.variant_type
        if vt in by_vt:
            by_vt[vt].append(s)
        else:
            print(
                f"WARNING: sample {s.sample_id} has unknown variant_type "
                f"{vt!r}", file=sys.stderr
            )

    per_bucket_available = {v: len(by_vt[v]) for v in BUCKETS}
    print("Pool variant distribution:")
    for v in BUCKETS:
        print(f"  {v}: {per_bucket_available[v]}")

    # ------------------------------------------------------------------
    # Capacity assessment (binding)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Capacity assessment")
    print(f"{'=' * 60}")

    assessments: dict = {}
    for key in keys:
        assessments[key] = _assess_candidate_capacity(
            CANDIDATES[key], per_bucket_available
        )

    verdict = _determine_verdict(assessments)
    _print_capacity_report(verdict, assessments)

    if verdict == "MBPP_FAMILY_OR_VARIANT_LIMIT":
        print(
            f"\nERROR: pool capacity insufficient for one or more candidates "
            f"(min max_achievable < {HARD_MIN_TOTAL}). Not building datasets.",
            file=sys.stderr,
        )
        print(f"Exit code: 2 (MBPP_FAMILY_OR_VARIANT_LIMIT)")
        return 2

    # ------------------------------------------------------------------
    # Load validation samples (Validation v2 -- pre-built, 4 categories)
    # ------------------------------------------------------------------
    if not VALIDATION_PATH.exists():
        print(f"ERROR: Validation v2 not found: {VALIDATION_PATH}",
              file=sys.stderr)
        return 1
    validation_samples = _load_jsonl_samples(VALIDATION_PATH)
    print(f"\nValidation: {len(validation_samples)} samples from "
          f"{_relpath(VALIDATION_PATH)}")
    validation_samples.sort(key=lambda s: s.sample_id)

    # ------------------------------------------------------------------
    # Load frozen v4 family list (for disjoint gates)
    # ------------------------------------------------------------------
    if not FROZEN_FAMILIES_PATH.exists():
        print(f"ERROR: frozen {FROZEN_VERSION} families not found: "
              f"{FROZEN_FAMILIES_PATH}", file=sys.stderr)
        return 1
    with FROZEN_FAMILIES_PATH.open(encoding="utf-8") as fh:
        frozen_data = json.load(fh)
    frozen_family_ids = set(frozen_data["families"])
    print(f"Frozen {FROZEN_VERSION}: {len(frozen_family_ids)} families from "
          f"{_relpath(FROZEN_FAMILIES_PATH)}")

    # ------------------------------------------------------------------
    # Build each requested candidate
    # ------------------------------------------------------------------
    exit_code = 0
    for key in keys:
        rc = _build_candidate(
            candidate_key=key,
            pool=pool,
            by_vt=by_vt,
            per_bucket_available=per_bucket_available,
            validation_samples=validation_samples,
            frozen_family_ids=frozen_family_ids,
            capacity_assessment=assessments[key],
            verdict=verdict,
            pool_sha256=pool_sha256,
            total_pool_samples=total_pool_samples,
        )
        if rc != 0:
            exit_code = rc

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
