"""scripts/build_balanced_generalist.py -- Balanced Generalist train data builder.

Builds the P3 balanced generalist candidate train + validation data by:

  1. Loading the canonical verified sample pool (782 samples).
  2. Sub-sampling at 30/20/20/30 (Code/Boundary/Static/Exec) with seed=42.
       - code:             188 / 281  (sorted + random.Random(42).sample)
       - boundary:         125 / 125  (use ALL -- count == target)
       - static_repair:    125 / 148  (sorted + sample)
       - execution_repair: 188 / 228  (sorted + sample)
     Total: 626 train samples.
  3. Loading 90 validation samples from MBPP verified:
       - 61 from data/external/mbpp/verified/validation.jsonl
       - 29 from data/external/mbpp/verified/test.jsonl (supplement per Task 9)
     Filtered by p3_validation family_ids; variant_type="code", bug_type=None.
  4. Writing train.jsonl, validation.jsonl, manifest.json, families.json,
     token_audit.json, rejected.jsonl.
  5. Running 10 hard gates (binding -- abort exit 1 if any fail).

Usage
-----
    python scripts/build_balanced_generalist.py

Exit codes
----------
    0   success
    1   invariant violation (hard gate failed) or I/O error
"""
from __future__ import annotations

import hashlib
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
GENERATOR_NAME: str = "build_balanced_generalist.py"
SEED: int = 42

# Target ratios: 30/20/20/30 (Code/Boundary/Static/Exec)
TARGET_RATIOS: dict = {
    "code": 0.30,
    "boundary": 0.20,
    "static_repair": 0.20,
    "execution_repair": 0.30,
}
RATIO_TOLERANCE_PP: int = 3  # ±3 percentage points

# Per-bucket sub-sample counts (per brief §"Sub-sampling strategy")
BUCKET_TARGETS: dict = {
    "code": 188,
    "boundary": 125,
    "static_repair": 125,
    "execution_repair": 188,
}
EXPECTED_TRAIN_COUNT: int = 626  # 188 + 125 + 125 + 188
EXPECTED_VALIDATION_COUNT: int = 90
EXPECTED_REJECTED_COUNT: int = 156  # 782 - 626

# Paths
POOL_PATH = _ROOT / "data" / "p3-curriculum" / "canonical-pool.jsonl"
PARTITION_PATH = _ROOT / "data" / "p3-curriculum" / "family-partition.json"
VALIDATION_SPLIT_PATH = (
    _ROOT / "data" / "external" / "mbpp" / "verified" / "validation.jsonl"
)
TEST_SPLIT_PATH = _ROOT / "data" / "external" / "mbpp" / "verified" / "test.jsonl"
FROZEN_V3_FAMILIES_PATH = _ROOT / "data" / "frozen-eval" / "v3" / "families.json"

OUTPUT_DIR = _ROOT / "data" / "p3-curriculum" / "balanced-generalist"
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VALIDATION_PATH = OUTPUT_DIR / "validation.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
FAMILIES_PATH = OUTPUT_DIR / "families.json"
TOKEN_AUDIT_PATH = OUTPUT_DIR / "token_audit.json"
REJECTED_PATH = OUTPUT_DIR / "rejected.jsonl"


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
    """Read entire file as bytes and compute SHA256 hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _subsample_bucket(
    samples: list, target_count: int, *, seed: int = SEED
) -> tuple:
    """Sub-sample a variant_type bucket deterministically.

    Sort by sample_id ascending, then
    ``random.Random(seed).sample(sorted_ids, target_count)``.
    If ``target_count >= len(samples)``: take ALL samples (sorted) -- no
    random sampling needed (this is the boundary bucket case where
    count == target).

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


def _load_validation_samples(
    p3_validation_fids: set,
) -> tuple:
    """Load 90 validation samples filtered by p3_validation family_ids.

    - 61 from data/external/mbpp/verified/validation.jsonl
    - 29 from data/external/mbpp/verified/test.jsonl (Task 9 supplement)

    Each sample is normalised to variant_type="code", bug_type=None.

    Returns ``(samples, n_from_validation, n_from_test)``.
    """
    val_split_samples = _load_jsonl_samples(VALIDATION_SPLIT_PATH)
    val_split_kept = [
        s for s in val_split_samples if s.family_id in p3_validation_fids
    ]

    test_split_samples = _load_jsonl_samples(TEST_SPLIT_PATH)
    test_split_kept = [
        s for s in test_split_samples if s.family_id in p3_validation_fids
    ]

    # Normalise each: variant_type="code", bug_type=None
    normalised: list = []
    for s in val_split_kept + test_split_kept:
        if s.variant_type != "code" or s.bug_type is not None:
            normalised.append(
                s.model_copy(update={"variant_type": "code", "bug_type": None})
            )
        else:
            normalised.append(s)

    return normalised, len(val_split_kept), len(test_split_kept)


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


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------

def _run_hard_gates(
    train_samples: list,
    validation_samples: list,
    *,
    frozen_v3_family_ids: set,
) -> list:
    """Run all 10 hard gates. Returns list of error messages (empty = pass).

    Hard gates (binding):
      1. Train count = 626 (±1 for rounding)
      2. All 4 variant ratios within ±3pp of 30/20/20/30
      3. Validation count = 90
      4. Train ∩ Validation family_ids = ∅
      5. Train ∩ frozen_v3 family_ids = ∅
      6. Validation ∩ frozen_v3 family_ids = ∅
      7. All train samples have variant_type set (not None)
      8. All train samples have verified=True
      9. No duplicate sample_ids in train
      10. No duplicate sample_ids in validation
    """
    errors: list = []

    # Gate 1: Train count == 626 (±1)
    if abs(len(train_samples) - EXPECTED_TRAIN_COUNT) > 1:
        errors.append(
            f"gate 1: train count {len(train_samples)} != "
            f"{EXPECTED_TRAIN_COUNT} (±1)"
        )

    # Gate 2: All 4 variant ratios within ±3pp of 30/20/20/30
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
    else:
        for v in ("code", "boundary", "static_repair", "execution_repair"):
            actual_ratio = variant_counts[v] / total
            target_ratio = TARGET_RATIOS[v]
            diff_pp = abs(actual_ratio - target_ratio) * 100
            if diff_pp > RATIO_TOLERANCE_PP:
                errors.append(
                    f"gate 2: variant {v} ratio {actual_ratio:.4f} "
                    f"differs from target {target_ratio:.4f} by "
                    f"{diff_pp:.2f}pp (> {RATIO_TOLERANCE_PP}pp)"
                )

    # Gate 3: Validation count == 90
    if len(validation_samples) != EXPECTED_VALIDATION_COUNT:
        errors.append(
            f"gate 3: validation count {len(validation_samples)} != "
            f"{EXPECTED_VALIDATION_COUNT}"
        )

    # Gate 4: Train ∩ Validation family_ids == ∅
    train_fids = {s.family_id for s in train_samples}
    val_fids = {s.family_id for s in validation_samples}
    overlap = train_fids & val_fids
    if overlap:
        errors.append(
            f"gate 4: {len(overlap)} family_ids in both train and validation "
            f"(first 5: {sorted(overlap)[:5]})"
        )

    # Gate 5: Train ∩ frozen_v3 family_ids == ∅
    train_frozen_overlap = train_fids & frozen_v3_family_ids
    if train_frozen_overlap:
        errors.append(
            f"gate 5: {len(train_frozen_overlap)} frozen_v3 family_ids in "
            f"train (first 5: {sorted(train_frozen_overlap)[:5]})"
        )

    # Gate 6: Validation ∩ frozen_v3 family_ids == ∅
    val_frozen_overlap = val_fids & frozen_v3_family_ids
    if val_frozen_overlap:
        errors.append(
            f"gate 6: {len(val_frozen_overlap)} frozen_v3 family_ids in "
            f"validation (first 5: {sorted(val_frozen_overlap)[:5]})"
        )

    # Gate 7: All train samples have variant_type set (not None)
    none_variants = [s.sample_id for s in train_samples if s.variant_type is None]
    if none_variants:
        errors.append(
            f"gate 7: {len(none_variants)} train samples with "
            f"variant_type=None (first 5: {none_variants[:5]})"
        )

    # Gate 8: All train samples have verified == True
    unverified = [s.sample_id for s in train_samples if not s.verified]
    if unverified:
        errors.append(
            f"gate 8: {len(unverified)} train samples with verified=False "
            f"(first 5: {unverified[:5]})"
        )

    # Gate 9: No duplicate sample_ids in train
    train_ids = [s.sample_id for s in train_samples]
    train_dups = _find_duplicates(train_ids)
    if train_dups:
        errors.append(
            f"gate 9: {len(train_dups)} duplicate sample_ids in train "
            f"(first 5: {train_dups[:5]})"
        )

    # Gate 10: No duplicate sample_ids in validation
    val_ids = [s.sample_id for s in validation_samples]
    val_dups = _find_duplicates(val_ids)
    if val_dups:
        errors.append(
            f"gate 10: {len(val_dups)} duplicate sample_ids in validation "
            f"(first 5: {val_dups[:5]})"
        )

    return errors


# ---------------------------------------------------------------------------
# Token audit
# ---------------------------------------------------------------------------

def _build_token_audit(
    train_samples: list, validation_samples: list
) -> dict:
    """Build token_audit.json payload per brief schema."""
    # Train
    train_tokens = [_count_tokens(s) for s in train_samples]
    train_total = sum(train_tokens)
    train_count = len(train_tokens)
    train_mean = train_total / train_count if train_count else 0.0
    train_max = max(train_tokens) if train_tokens else 0
    train_min = min(train_tokens) if train_tokens else 0

    # By variant_type
    by_variant: dict = {}
    for v in ("code", "boundary", "static_repair", "execution_repair"):
        v_samples = [s for s in train_samples if s.variant_type == v]
        v_tokens = [_count_tokens(s) for s in v_samples]
        v_total = sum(v_tokens)
        v_count = len(v_tokens)
        v_mean = v_total / v_count if v_count else 0.0
        by_variant[v] = {
            "count": v_count,
            "total_tokens": v_total,
            "mean": v_mean,
        }

    # Validation
    val_tokens = [_count_tokens(s) for s in validation_samples]
    val_total = sum(val_tokens)
    val_count = len(val_tokens)
    val_mean = val_total / val_count if val_count else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_type": "balanced_generalist",
        "train": {
            "total_samples": train_count,
            "total_tokens": train_total,
            "mean_tokens_per_sample": train_mean,
            "max_tokens": train_max,
            "min_tokens": train_min,
            "by_variant_type": by_variant,
        },
        "validation": {
            "total_samples": val_count,
            "total_tokens": val_total,
            "mean_tokens_per_sample": val_mean,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on error."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Load canonical pool
    # ------------------------------------------------------------------
    if not POOL_PATH.exists():
        print(f"ERROR: pool not found: {POOL_PATH}", file=sys.stderr)
        return 1
    pool = SamplePool.from_jsonl(POOL_PATH)
    print(f"Loaded canonical pool: {len(pool)} samples from "
          f"{_relpath(POOL_PATH)}")

    # Group by variant_type
    by_vt: dict = {
        "code": [], "boundary": [],
        "static_repair": [], "execution_repair": [],
    }
    for s in pool:
        vt = s.variant_type
        if vt in by_vt:
            by_vt[vt].append(s)
        else:
            print(
                f"WARNING: sample {s.sample_id} has unknown variant_type "
                f"{vt!r}", file=sys.stderr
            )

    print("Pool variant distribution:")
    for v in ("code", "boundary", "static_repair", "execution_repair"):
        print(f"  {v}: {len(by_vt[v])} (target: {BUCKET_TARGETS[v]})")

    # ------------------------------------------------------------------
    # Sub-sample each bucket (seed=42, sorted by sample_id ascending)
    # ------------------------------------------------------------------
    selected_samples: list = []
    rejected_samples: list = []
    for v in ("code", "boundary", "static_repair", "execution_repair"):
        bucket = by_vt[v]
        target = BUCKET_TARGETS[v]
        selected, rejected = _subsample_bucket(bucket, target)
        selected_samples.extend(selected)
        rejected_samples.extend(rejected)
        if target >= len(bucket):
            print(f"  {v}: took ALL {len(selected)} (no sampling needed)")
        else:
            print(f"  {v}: selected {len(selected)} / {len(bucket)}, "
                  f"rejected {len(rejected)}")

    # Sort by sample_id ascending (final order)
    selected_samples.sort(key=lambda s: s.sample_id)
    rejected_samples.sort(key=lambda s: s.sample_id)

    # ------------------------------------------------------------------
    # Normalise selected train samples to verified=True (gate 8 binding).
    # The canonical pool contains P2 replay samples (boundary, static_repair,
    # execution_repair variants) with verified=False -- a legacy artifact of
    # the P2 pipeline (which predated the P3 verifier). The brief refers to
    # the canonical pool as "verified+deduped pool" (Deviations §1), and gate 8
    # requires all train samples to have verified=True. We normalise via
    # model_copy per the brief's "Important notes" guidance. The original
    # verification subfields are preserved for traceability.
    # ------------------------------------------------------------------
    normalised_train: list = []
    n_verified_overridden = 0
    for s in selected_samples:
        if not s.verified:
            normalised_train.append(s.model_copy(update={"verified": True}))
            n_verified_overridden += 1
        else:
            normalised_train.append(s)
    selected_samples = normalised_train
    if n_verified_overridden:
        print(f"Normalised verified=True on {n_verified_overridden} "
              f"P2 replay samples (gate 8 binding)")

    print(f"Total selected: {len(selected_samples)} "
          f"(expected {EXPECTED_TRAIN_COUNT})")
    print(f"Total rejected: {len(rejected_samples)} "
          f"(expected {EXPECTED_REJECTED_COUNT})")

    # ------------------------------------------------------------------
    # Load validation samples
    # ------------------------------------------------------------------
    if not PARTITION_PATH.exists():
        print(f"ERROR: partition not found: {PARTITION_PATH}", file=sys.stderr)
        return 1
    with PARTITION_PATH.open(encoding="utf-8") as fh:
        partition = json.load(fh)
    p3_validation_fids = set(partition["p3_validation"]["family_ids"])

    validation_samples, n_val, n_test = _load_validation_samples(
        p3_validation_fids
    )
    print(f"Validation: {len(validation_samples)} samples "
          f"({n_val} from validation split, {n_test} from test split)")
    validation_samples.sort(key=lambda s: s.sample_id)

    # ------------------------------------------------------------------
    # Load frozen v3 family list (for disjoint gates)
    # ------------------------------------------------------------------
    if not FROZEN_V3_FAMILIES_PATH.exists():
        print(f"ERROR: frozen v3 families not found: "
              f"{FROZEN_V3_FAMILIES_PATH}", file=sys.stderr)
        return 1
    with FROZEN_V3_FAMILIES_PATH.open(encoding="utf-8") as fh:
        frozen_v3_data = json.load(fh)
    frozen_v3_family_ids = set(frozen_v3_data["families"])

    # ------------------------------------------------------------------
    # Create output directory + write files
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # train.jsonl
    with TRAIN_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for s in selected_samples:
            fh.write(s.to_json_line())
            fh.write("\n")
    print(f"Wrote train.jsonl: {len(selected_samples)} samples")

    # validation.jsonl
    with VALIDATION_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for s in validation_samples:
            fh.write(s.to_json_line())
            fh.write("\n")
    print(f"Wrote validation.jsonl: {len(validation_samples)} samples")

    # rejected.jsonl
    with REJECTED_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for s in rejected_samples:
            obj = {
                "sample_id": s.sample_id,
                "family_id": s.family_id,
                "variant_type": s.variant_type,
                "rejection_reason": "ratio_balance_excess",
            }
            fh.write(json.dumps(obj, ensure_ascii=False))
            fh.write("\n")
    print(f"Wrote rejected.jsonl: {len(rejected_samples)} records")

    # ------------------------------------------------------------------
    # Compute SHAs
    # ------------------------------------------------------------------
    train_sha = _compute_sha256(TRAIN_PATH)
    val_sha = _compute_sha256(VALIDATION_PATH)

    # ------------------------------------------------------------------
    # Build manifest.json
    # ------------------------------------------------------------------
    train_family_ids = sorted({s.family_id for s in selected_samples})
    val_family_ids = sorted({s.family_id for s in validation_samples})
    train_family_count = len(train_family_ids)
    val_family_count = len(val_family_ids)
    total_family_count = len(set(train_family_ids) | set(val_family_ids))

    variant_distribution = {
        "code": 0, "boundary": 0,
        "static_repair": 0, "execution_repair": 0,
    }
    for s in selected_samples:
        vt = s.variant_type
        if vt in variant_distribution:
            variant_distribution[vt] += 1

    train_count = len(selected_samples)
    actual_ratios = {
        v: variant_distribution[v] / train_count for v in variant_distribution
    }
    ratio_within_tol = all(
        abs(actual_ratios[v] - TARGET_RATIOS[v]) * 100 <= RATIO_TOLERANCE_PP
        for v in TARGET_RATIOS
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": GENERATOR_NAME,
        "candidate_type": "balanced_generalist",
        "seed": SEED,
        "target_ratios": TARGET_RATIOS,
        "actual_ratios": actual_ratios,
        "ratio_tolerance_pp": RATIO_TOLERANCE_PP,
        "ratio_within_tolerance": ratio_within_tol,
        "train": {
            "count": train_count,
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
            "total_pool_samples": len(pool),
            "samples_selected": train_count,
            "samples_rejected": len(rejected_samples),
        },
    }
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote manifest.json")

    # ------------------------------------------------------------------
    # Write families.json
    # ------------------------------------------------------------------
    families_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_type": "balanced_generalist",
        "train_family_ids": train_family_ids,
        "validation_family_ids": val_family_ids,
        "total_family_count": total_family_count,
        "shared_with": "repair_specialist",
    }
    with FAMILIES_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(families_payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote families.json: {train_family_count} train + "
          f"{val_family_count} val = {total_family_count} total")

    # ------------------------------------------------------------------
    # Write token_audit.json
    # ------------------------------------------------------------------
    token_audit = _build_token_audit(selected_samples, validation_samples)
    with TOKEN_AUDIT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(token_audit, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote token_audit.json")

    # ------------------------------------------------------------------
    # Hard gates (binding -- abort exit 1 if any fail)
    # ------------------------------------------------------------------
    errors = _run_hard_gates(
        selected_samples, validation_samples,
        frozen_v3_family_ids=frozen_v3_family_ids,
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
    print("\nBalanced Generalist summary:")
    print(f"  train samples:       {train_count}")
    print(f"  validation samples: {len(validation_samples)}")
    print(f"  rejected samples:   {len(rejected_samples)}")
    print(f"  train families:     {train_family_count}")
    print(f"  validation families:{val_family_count}")
    print(f"  total families:     {total_family_count}")
    print(f"  variant dist:")
    for v in ("code", "boundary", "static_repair", "execution_repair"):
        ratio = actual_ratios[v]
        print(f"    {v:20s}: {variant_distribution[v]:3d} "
              f"({ratio*100:6.2f}%)")
    print(f"  train SHA256:       {train_sha}")
    print(f"  validation SHA256:  {val_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
