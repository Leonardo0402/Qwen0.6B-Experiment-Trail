"""scripts/p3_formal_readiness_gate.py -- P3 Formal Readiness Gate v2 (Wave 5-J, Issue #14).

Executes 20 checks (check6 split into 6a CPU mandatory + 6b GPU deferrable = 21
table rows) verifying whether the FORMAL training data, configs, and
infrastructure are ready for formal P3 training.

Adapts the existing 17 checks from ``p3_readiness_gate.py`` to formal data
paths and adds 3 new formal-specific checks (check17 pool SHA lock, check18
config validity, check19 per-family cap enforcement, check20 capacity verdict
-- 4 new checks bringing the total to 21 rows / 20 check numbers).

Three-state verdict (plus PENDING_DATASET_BUILD for the pre-build case):
  - GO_FOR_P3_TRAINING: all mandatory checks PASS AND both candidates'
        formal capacity >= 2300
  - MBPP_FAMILY_OR_VARIANT_LIMIT: all checks PASS but any candidate's formal
        capacity < 2300 (or capacity verdict is MBPP_FAMILY_OR_VARIANT_LIMIT)
  - FIX_FIRST: any mandatory check FAILS
  - PENDING_DATASET_BUILD: all checks PASS (incl. SKIP) but formal datasets
        not built yet -- capacity cannot be verified. The gate can be run
        before datasets are built to verify infrastructure, and again after
        to verify data.

Resilience: if formal datasets don't exist yet
(``data/p3-formal/balanced-generalist/train.jsonl`` missing), checks
3, 4, 10, 11, 12, 13, 19, 20 return SKIP (not FAIL). SKIP does not fail
the gate.

Report output: ``reports/p3/p3-formal-readiness-report.md``

Usage
-----
    py -3.11 scripts/p3_formal_readiness_gate.py
    py -3.11 scripts/p3_formal_readiness_gate.py --output /custom/path.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Project-root import guard
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.family_registry import FamilyRegistry  # noqa: E402
from src.schemas import Sample, to_chatml  # noqa: E402

# Reuse helpers, constants, and identical check functions from the existing gate
from scripts.p3_readiness_gate import (  # noqa: E402
    # Helpers
    _read_jsonl,
    _read_bytes,
    _count_variant_types,
    _load_family_set,
    _load_family_ids_from_jsonl,
    _load_all_family_sets,
    _estimate_tokens,
    _sample_full_text,
    _check_assistant_retention_one,
    _status_token,
    _format_details_short as _format_details_short_existing,
    # Shared constants
    FROZEN_V4_DIR,
    MANIFEST_PATH,
    FAMILIES_PATH,
    TEST_RAW_PATH,
    REJECTED_PATH,
    FROZEN_V4_LOCK_PATH,
    REGISTRY_PATH,
    FAMILY_PARTITION_PATH,
    VALIDATION_V2_DIR,
    VALIDATION_V2_PATH,
    VALIDATION_V2_FAMILIES_PATH,
    VALIDATION_V2_MANIFEST_PATH,
    BASELINE_LOCK_PATH,
    MAX_SEQ_LENGTH,
    MIN_TRAIN_SAMPLES_FOR_FULL,
    MAX_TRAIN_SAMPLES_FOR_FULL,
    EXPECTED_BASELINE_MODELS,
    BASELINE_REQUIRED_FIELDS,
    P3_TEST_FILES,
    # Reused check functions (identical for formal gate)
    check1_frozen_v4_sha_locked,
    check5_canary_all_fail,
    check6a_cpu_smoke,
    check6b_gpu_smoke,
    check8_cpu_ci_green,
    check9_baseline_lock_present,
    check14_composite_evaluator_complete,
    check15_v4_coverage_gate,
    check16_validation_v2_gate,
)

# ---------------------------------------------------------------------------
# Formal-specific constants
# ---------------------------------------------------------------------------

FORMAL_POOL_PATH = _ROOT / "data" / "p3-formal" / "canonical-pool.jsonl"
FORMAL_POOL_MANIFEST_PATH = _ROOT / "data" / "p3-formal" / "canonical-pool-manifest.json"
FORMAL_FAMILY_PARTITION_PATH = _ROOT / "data" / "p3-formal" / "family-partition.json"

FORMAL_BALANCED_DIR = _ROOT / "data" / "p3-formal" / "balanced-generalist"
FORMAL_REPAIR_DIR = _ROOT / "data" / "p3-formal" / "repair-specialist"
FORMAL_BALANCED_TRAIN_PATH = FORMAL_BALANCED_DIR / "train.jsonl"
FORMAL_REPAIR_TRAIN_PATH = FORMAL_REPAIR_DIR / "train.jsonl"
FORMAL_BALANCED_MANIFEST_PATH = FORMAL_BALANCED_DIR / "manifest.json"
FORMAL_REPAIR_MANIFEST_PATH = FORMAL_REPAIR_DIR / "manifest.json"

FORMAL_BALANCED_CONFIG_PATH = _ROOT / "configs" / "p3" / "balanced-generalist-formal-v1.yaml"
FORMAL_REPAIR_CONFIG_PATH = _ROOT / "configs" / "p3" / "repair-specialist-formal-v1.yaml"

FORMAL_BALANCED_OUTPUT_DIR = _ROOT / "adapters" / "p3" / "balanced-generalist-formal-v1"
FORMAL_REPAIR_OUTPUT_DIR = _ROOT / "adapters" / "p3" / "repair-specialist-formal-v1"

FORMAL_REPORT_PATH = _ROOT / "reports" / "p3" / "p3-formal-readiness-report.md"

# Per-family caps (mirror p3_formal_dataset_builder.py)
PER_FAMILY_TOTAL_CAP = 8
PER_FAMILY_PER_BUCKET_CAP = 3
SINGLE_FAMILY_PERCENT_CAP = 1.0  # 1%

# Valid capacity verdicts from the formal dataset builder
VALID_CAPACITY_VERDICTS = ("FORMAL_CAPACITY_FEASIBLE", "FORMAL_CAPACITY_AT_RISK")

# Candidate bucket targets (mirror p3_formal_dataset_builder.py CANDIDATES dict)
# Used for pool-based capacity fallback in check20.
_FORMAL_CANDIDATE_TARGETS: dict = {
    "balanced": {
        "code": 750, "boundary": 500, "static_repair": 500, "execution_repair": 750,
    },
    "repair": {
        "code": 375, "boundary": 375, "static_repair": 750, "execution_repair": 1000,
    },
}
_FORMAL_BUCKETS = ("code", "boundary", "static_repair", "execution_repair")


# ---------------------------------------------------------------------------
# Helper: SKIP result constructor
# ---------------------------------------------------------------------------

def _skip(reason: str, **extra) -> Tuple[bool, dict]:
    """Construct a SKIP result. SKIP does not fail the gate."""
    details = {"reason": f"SKIP: {reason}", "skipped": True}
    details.update(extra)
    return True, details


def _formal_datasets_exist() -> bool:
    """True if at least one formal train.jsonl exists."""
    return FORMAL_BALANCED_TRAIN_PATH.exists() or FORMAL_REPAIR_TRAIN_PATH.exists()


def _formal_manifests_exist() -> bool:
    """True if at least one formal manifest.json exists."""
    return FORMAL_BALANCED_MANIFEST_PATH.exists() or FORMAL_REPAIR_MANIFEST_PATH.exists()


# ---------------------------------------------------------------------------
# Check 2: Family isolation (formal train vs eval sets -- pairwise disjoint)
# ---------------------------------------------------------------------------

def _load_all_family_sets_formal() -> "dict[str, set[str]]":
    """Load family sets with ``formal_train`` from FORMAL train paths.

    Mirrors ``_load_all_family_sets`` from the existing gate but overrides
    ``formal_train`` to read from ``data/p3-formal/{balanced,repair}/train.jsonl``
    instead of the curriculum paths.
    """
    sets = _load_all_family_sets()
    sets["formal_train"] = (
        _load_family_ids_from_jsonl(FORMAL_BALANCED_TRAIN_PATH)
        | _load_family_ids_from_jsonl(FORMAL_REPAIR_TRAIN_PATH)
    )
    return sets


def check2_family_isolation_formal() -> Tuple[bool, dict]:
    """Verify pairwise disjoint family sets (formal train vs eval sets).

    The following 5 sets must be pairwise disjoint:
      - formal_train (families in FORMAL balanced + repair train.jsonl)
      - validation_v2 (validation-v2/families.json)
      - frozen_v4 (frozen-eval/v4/families.json)
      - historical_frozen (v1 + v3 + p2-frozen-v2)
      - historical_validation (p2 stage validation + curriculum-v2)

    Whitelist:
      - p3_train_replay ⊆ p2_train
      - formal_train ∩ p2_train ⊆ p3_train_replay
    """
    sets = _load_all_family_sets_formal()

    main_sets = {
        "formal_train": sets["formal_train"],
        "validation_v2": sets["validation_v2"],
        "frozen_v4": sets["frozen_v4"],
        "historical_frozen": sets["historical_frozen"],
        "historical_validation": sets["historical_validation"],
    }

    violations: list[dict] = []
    keys = list(main_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            overlap = main_sets[a] & main_sets[b]
            if overlap:
                violations.append({
                    "pair": (a, b),
                    "count": len(overlap),
                    "samples": sorted(overlap)[:10],
                })

    p3_replay = sets["p3_train_replay"]
    p2_train = sets["p2_train"]
    whitelist_intersection = p3_replay & p2_train
    whitelist_complete = (whitelist_intersection == p3_replay)
    formal_p2_overlap = sets["formal_train"] & p2_train
    formal_p2_clean = formal_p2_overlap <= p3_replay

    counts = {k: len(v) for k, v in main_sets.items()}
    counts["p3_train_replay"] = len(p3_replay)
    counts["p2_train"] = len(p2_train)
    counts["whitelist_intersection"] = len(whitelist_intersection)
    counts["formal_train_p2_overlap"] = len(formal_p2_overlap)

    passed = (
        len(violations) == 0
        and whitelist_complete
        and formal_p2_clean
    )
    details = {
        "counts": counts,
        "violations": violations,
        "whitelist": {
            "pair": ("p3_train_replay", "p2_train"),
            "replay_count": len(p3_replay),
            "intersection_count": len(whitelist_intersection),
            "all_replay_in_p2_train": whitelist_complete,
            "formal_train_p2_overlap_clean": formal_p2_clean,
            "formal_train_p2_overlap_count": len(formal_p2_overlap),
        },
    }
    return passed, details


# ---------------------------------------------------------------------------
# Check 3: Assistant retention = 100% (formal)
# ---------------------------------------------------------------------------

def check3_assistant_retention_formal() -> Tuple[bool, dict]:
    """Verify 100% of FORMAL train samples preserve the full target_code.

    SKIP if formal datasets don't exist yet.
    """
    if not _formal_datasets_exist():
        return _skip("formal train datasets not built yet")
    paths = [FORMAL_BALANCED_TRAIN_PATH, FORMAL_REPAIR_TRAIN_PATH]
    total = 0
    retained = 0
    failures: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    sample = Sample.from_json_line(line)
                except Exception as e:  # pragma: no cover -- defensive
                    failures.append(f"{path.name}: parse_error: {e}")
                    continue
                if _check_assistant_retention_one(sample):
                    retained += 1
                else:
                    failures.append(sample.sample_id)
    if total == 0:
        return _skip("formal train datasets exist but are empty")
    passed = retained == total
    details = {"checked": total, "retained": retained}
    if failures:
        details["first_failures"] = failures[:5]
    return passed, details


# ---------------------------------------------------------------------------
# Check 4: Silent target truncation = 0 (formal)
# ---------------------------------------------------------------------------

def check4_silent_truncation_zero_formal() -> Tuple[bool, dict]:
    """Flag any FORMAL train sample whose assistant target would be silently truncated.

    SKIP if formal datasets don't exist yet.
    PASS iff ``real_silent_truncations == 0`` (same policy as existing gate).
    """
    if not _formal_datasets_exist():
        return _skip("formal train datasets not built yet")
    paths = [FORMAL_BALANCED_TRAIN_PATH, FORMAL_REPAIR_TRAIN_PATH]
    total = 0
    potential = 0
    real_silent = 0
    flagged_potential: list[str] = []
    flagged_real: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    sample = Sample.from_json_line(line)
                except Exception:  # pragma: no cover -- defensive
                    continue
                full_est = _estimate_tokens(_sample_full_text(sample))
                assistant_est = _estimate_tokens(sample.target_code or "")
                if full_est > MAX_SEQ_LENGTH:
                    potential += 1
                    if len(flagged_potential) < 5:
                        flagged_potential.append(sample.sample_id)
                if assistant_est >= MAX_SEQ_LENGTH:
                    real_silent += 1
                    if len(flagged_real) < 5:
                        flagged_real.append(sample.sample_id)
    if total == 0:
        return _skip("formal train datasets exist but are empty")
    passed = real_silent == 0
    details = {
        "checked": total,
        "potential_truncations": potential,
        "real_silent_truncations": real_silent,
        "max_seq_length": MAX_SEQ_LENGTH,
        "estimate_method": "len(text) // 4",
        "first_flagged_potential": flagged_potential,
        "first_flagged_real": flagged_real,
        "policy": "preserve_assistant",
    }
    return passed, details


# ---------------------------------------------------------------------------
# Check 7: Formal output dirs don't exist (no overwrite)
# ---------------------------------------------------------------------------

def check7_formal_output_dirs_dont_exist() -> Tuple[bool, dict]:
    """Verify formal adapter output dirs don't exist (formal training not started)."""
    checked = [
        str(FORMAL_BALANCED_OUTPUT_DIR.relative_to(_ROOT)),
        str(FORMAL_REPAIR_OUTPUT_DIR.relative_to(_ROOT)),
    ]
    existing = []
    for p in (FORMAL_BALANCED_OUTPUT_DIR, FORMAL_REPAIR_OUTPUT_DIR):
        if p.exists():
            existing.append(str(p.relative_to(_ROOT)))
    passed = not existing
    return passed, {
        "checked": checked,
        "existing": existing,
        "none_exist": passed,
    }


# ---------------------------------------------------------------------------
# Check 10: Formal train capacity (>= 2300 per candidate)
# ---------------------------------------------------------------------------

def check10_formal_train_capacity() -> Tuple[bool, dict]:
    """Per-candidate FORMAL train capacity vs 2300 threshold.

    SKIP if formal datasets don't exist yet.
    Sets ``capacity_status`` in details for verdict computation:
      - "SKIP" if datasets not built
      - "LIMIT" if any candidate < 2300
      - "FULL" if both candidates >= 2300
    """
    if not _formal_datasets_exist():
        return _skip("formal train datasets not built yet", capacity_status="SKIP")

    def _count(p: Path) -> int:
        if not p.exists():
            return 0
        n = 0
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return n

    balanced_train = _count(FORMAL_BALANCED_TRAIN_PATH)
    repair_train = _count(FORMAL_REPAIR_TRAIN_PATH)

    if balanced_train == 0 and repair_train == 0:
        return _skip("formal train datasets exist but are empty", capacity_status="SKIP")

    balanced_ok = balanced_train >= MIN_TRAIN_SAMPLES_FOR_FULL
    repair_ok = repair_train >= MIN_TRAIN_SAMPLES_FOR_FULL
    capacity_status = "FULL" if (balanced_ok and repair_ok) else "LIMIT"
    passed = balanced_ok and repair_ok

    return passed, {
        "balanced_train": balanced_train,
        "repair_train": repair_train,
        "min_threshold": MIN_TRAIN_SAMPLES_FOR_FULL,
        "max_threshold": MAX_TRAIN_SAMPLES_FOR_FULL,
        "balanced_verdict": "FULL" if balanced_ok else "LIMIT",
        "repair_verdict": "FULL" if repair_ok else "LIMIT",
        "capacity_status": capacity_status,
    }


# ---------------------------------------------------------------------------
# Check 11: verified consistency (formal)
# ---------------------------------------------------------------------------

def check11_verified_consistency_formal() -> Tuple[bool, dict]:
    """Verify ``verified`` field consistency with ``verification`` subfields (formal).

    SKIP if formal datasets don't exist yet.
    Rule: verified=True ⟺ syntax_ok AND pytest_ok.
    """
    if not _formal_datasets_exist():
        return _skip("formal train datasets not built yet")

    def _load(p: Path) -> list:
        if not p.exists():
            return []
        samples = []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    all_samples = _load(FORMAL_BALANCED_TRAIN_PATH) + _load(FORMAL_REPAIR_TRAIN_PATH)
    if not all_samples:
        return _skip("formal train datasets exist but are empty")

    inconsistent = []
    for s in all_samples:
        verified = s.get("verified", False)
        ver = s.get("verification") or {}
        syntax_ok = ver.get("syntax_ok", False)
        pytest_ok = ver.get("pytest_ok", False)
        if verified:
            if not (syntax_ok and pytest_ok):
                inconsistent.append({
                    "sample_id": s.get("sample_id", "?"),
                    "family_id": s.get("family_id", "?"),
                    "issue": "verified=True but syntax_ok/pytest_ok not both True",
                })
        else:
            if syntax_ok and pytest_ok:
                inconsistent.append({
                    "sample_id": s.get("sample_id", "?"),
                    "family_id": s.get("family_id", "?"),
                    "issue": "verified=False but syntax_ok AND pytest_ok both True",
                })

    passed = len(inconsistent) == 0
    details = {
        "checked": len(all_samples),
        "inconsistent_count": len(inconsistent),
        "inconsistent_sample_ids": [r["sample_id"] for r in inconsistent[:20]],
        "rule": "verified=True ⟺ syntax_ok AND pytest_ok",
    }
    return passed, details


# ---------------------------------------------------------------------------
# Check 12: Candidate ratio within tolerance (formal)
# ---------------------------------------------------------------------------

def check12_candidate_ratio_within_tolerance_formal() -> Tuple[bool, dict]:
    """Check both FORMAL candidates' variant_type ratios are within +/-3pp.

    SKIP if formal datasets don't exist yet.
    """
    if not _formal_datasets_exist():
        return _skip("formal train datasets not built yet")

    balanced_targets = {"code": 0.30, "boundary": 0.20, "static_repair": 0.20, "execution_repair": 0.30}
    repair_targets = {"code": 0.15, "boundary": 0.15, "static_repair": 0.30, "execution_repair": 0.40}
    tolerance = 0.03

    balanced_counts = _count_variant_types(FORMAL_BALANCED_TRAIN_PATH)
    repair_counts = _count_variant_types(FORMAL_REPAIR_TRAIN_PATH)

    if sum(balanced_counts.values()) == 0 and sum(repair_counts.values()) == 0:
        return _skip("formal train datasets exist but are empty")

    details = {
        "balanced_counts": balanced_counts,
        "repair_counts": repair_counts,
        "tolerance_pp": 3,
        "violations": [],
    }
    all_ok = True
    for label, counts, targets in [
        ("balanced", balanced_counts, balanced_targets),
        ("repair", repair_counts, repair_targets),
    ]:
        total = sum(counts.values())
        if total == 0:
            details["violations"].append(f"{label}: total=0")
            all_ok = False
            continue
        for bucket, target in targets.items():
            actual = counts.get(bucket, 0) / total
            diff = abs(actual - target)
            if diff > tolerance:
                details["violations"].append(
                    f"{label}.{bucket}: actual={actual:.2%} target={target:.2%} diff={diff:.2%}"
                )
                all_ok = False
    return all_ok, details


# ---------------------------------------------------------------------------
# Check 13: All required buckets non-empty (formal)
# ---------------------------------------------------------------------------

def check13_all_buckets_non_empty_formal() -> Tuple[bool, dict]:
    """Check all 4 variant_type buckets non-empty for both FORMAL candidates.

    SKIP if formal datasets don't exist yet.
    """
    if not _formal_datasets_exist():
        return _skip("formal train datasets not built yet")

    required = ["code", "boundary", "static_repair", "execution_repair"]
    balanced_counts = _count_variant_types(FORMAL_BALANCED_TRAIN_PATH)
    repair_counts = _count_variant_types(FORMAL_REPAIR_TRAIN_PATH)

    if sum(balanced_counts.values()) == 0 and sum(repair_counts.values()) == 0:
        return _skip("formal train datasets exist but are empty")

    empty = []
    for bucket in required:
        if balanced_counts.get(bucket, 0) == 0:
            empty.append(f"balanced.{bucket}")
        if repair_counts.get(bucket, 0) == 0:
            empty.append(f"repair.{bucket}")

    details = {
        "balanced_counts": balanced_counts,
        "repair_counts": repair_counts,
        "required_buckets": required,
        "empty_buckets": empty,
    }
    return len(empty) == 0, details


# ---------------------------------------------------------------------------
# Check 17: Formal pool SHA lock (NEW)
# ---------------------------------------------------------------------------

def check17_formal_pool_sha_lock() -> Tuple[bool, dict]:
    """Verify formal canonical-pool-manifest.json exists and pool SHA matches.

    Checks:
      - ``data/p3-formal/canonical-pool-manifest.json`` exists
      - Has valid ``pool_sha256`` field (non-empty string)
      - Actual SHA256 of ``canonical-pool.jsonl`` matches ``pool_sha256``
        (CRLF normalized to LF for cross-platform consistency)
    """
    if not FORMAL_POOL_MANIFEST_PATH.exists():
        return False, {"error": f"manifest not found: {FORMAL_POOL_MANIFEST_PATH}"}
    if not FORMAL_POOL_PATH.exists():
        return False, {"error": f"pool file not found: {FORMAL_POOL_PATH}"}

    with FORMAL_POOL_MANIFEST_PATH.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    pool_sha = manifest.get("pool_sha256", "")
    if not pool_sha or not isinstance(pool_sha, str):
        return False, {"error": "pool_sha256 missing or invalid in manifest"}

    actual_sha = hashlib.sha256(_read_bytes(FORMAL_POOL_PATH)).hexdigest()
    if actual_sha != pool_sha:
        return False, {
            "error": "pool_sha256 mismatch",
            "manifest_sha": pool_sha,
            "actual_sha": actual_sha,
        }
    return True, {
        "pool_sha256": pool_sha,
        "actual_sha": actual_sha,
        "sha_match": True,
        "total_samples": manifest.get("total_samples"),
        "total_families": manifest.get("total_families"),
        "bucket_counts": manifest.get("bucket_counts"),
        "schema_version": manifest.get("schema_version"),
    }


# ---------------------------------------------------------------------------
# Check 18: Formal config validity (NEW)
# ---------------------------------------------------------------------------

def check18_formal_config_validity() -> Tuple[bool, dict]:
    """Verify both ``*-formal-v1.yaml`` configs exist and have required fields.

    Required fields per config:
      - ``training_mode: independent``
      - ``initial_adapter: null``
      - ``num_train_epochs: 2``
      - ``train_file`` references ``data/p3-formal/...``
    """
    import yaml

    configs = [
        ("balanced", FORMAL_BALANCED_CONFIG_PATH),
        ("repair", FORMAL_REPAIR_CONFIG_PATH),
    ]
    errors: list[str] = []
    details_per: dict = {}

    for name, path in configs:
        if not path.exists():
            errors.append(f"{name}: config missing: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        per_errors: list[str] = []
        if cfg.get("training_mode") != "independent":
            per_errors.append(
                f"training_mode={cfg.get('training_mode')!r} (expected 'independent')"
            )
        if cfg.get("initial_adapter") is not None:
            per_errors.append(
                f"initial_adapter={cfg.get('initial_adapter')!r} (expected null)"
            )
        if cfg.get("num_train_epochs") != 2:
            per_errors.append(
                f"num_train_epochs={cfg.get('num_train_epochs')!r} (expected 2)"
            )
        train_file = cfg.get("train_file", "")
        if not train_file or "data/p3-formal/" not in str(train_file):
            per_errors.append(
                f"train_file={train_file!r} (expected data/p3-formal/...)"
            )

        if per_errors:
            errors.extend([f"{name}: {e}" for e in per_errors])
        details_per[name] = {
            "training_mode": cfg.get("training_mode"),
            "initial_adapter": cfg.get("initial_adapter"),
            "num_train_epochs": cfg.get("num_train_epochs"),
            "train_file": train_file,
            "output_dir": cfg.get("output_dir"),
            "errors": per_errors,
        }

    passed = len(errors) == 0
    return passed, {"errors": errors, "configs": details_per}


# ---------------------------------------------------------------------------
# Check 19: Per-family cap enforcement (NEW)
# ---------------------------------------------------------------------------

def check19_per_family_cap_enforcement() -> Tuple[bool, dict]:
    """Verify per-family caps in FORMAL datasets.

    SKIP if formal datasets don't exist yet.

    Caps (mirror ``p3_formal_dataset_builder.py``):
      - Per family total <= 8 samples
      - Per family per bucket <= 3 samples
      - Single family <= 1% of total dataset
    """
    if not _formal_datasets_exist():
        return _skip("formal train datasets not built yet")

    paths = [FORMAL_BALANCED_TRAIN_PATH, FORMAL_REPAIR_TRAIN_PATH]
    errors: list[str] = []
    details_per: dict = {}

    for path in paths:
        name = path.parent.name  # "balanced-generalist" or "repair-specialist"
        if not path.exists():
            continue
        samples = _read_jsonl(path)
        total = len(samples)
        if total == 0:
            continue

        family_total: "dict[str, int]" = {}
        family_bucket: "dict[tuple, int]" = {}
        for s in samples:
            fid = s.get("family_id", "?")
            family_total[fid] = family_total.get(fid, 0) + 1
            vt = s.get("variant_type", "unknown")
            key = (fid, vt)
            family_bucket[key] = family_bucket.get(key, 0) + 1

        over_total = {
            fid: c for fid, c in family_total.items()
            if c > PER_FAMILY_TOTAL_CAP
        }
        over_bucket = {
            f"{k[0]}/{k[1]}": c for k, c in family_bucket.items()
            if c > PER_FAMILY_PER_BUCKET_CAP
        }
        pct_threshold = SINGLE_FAMILY_PERCENT_CAP / 100.0
        over_pct = {
            fid: c for fid, c in family_total.items()
            if c / total > pct_threshold
        }

        if over_total:
            errors.append(
                f"{name}: {len(over_total)} families exceed per-family total cap "
                f"{PER_FAMILY_TOTAL_CAP} (first 5: "
                f"{dict(sorted(over_total.items())[:5])})"
            )
        if over_bucket:
            errors.append(
                f"{name}: {len(over_bucket)} (family, bucket) pairs exceed "
                f"per-bucket cap {PER_FAMILY_PER_BUCKET_CAP} "
                f"(first 5: {dict(sorted(over_bucket.items())[:5])})"
            )
        if over_pct:
            errors.append(
                f"{name}: {len(over_pct)} families exceed "
                f"{SINGLE_FAMILY_PERCENT_CAP}% of total "
                f"(first 5: {dict(sorted(over_pct.items())[:5])})"
            )

        details_per[name] = {
            "total_samples": total,
            "family_count": len(family_total),
            "max_family_total": max(family_total.values()) if family_total else 0,
            "over_total_cap_count": len(over_total),
            "over_bucket_cap_count": len(over_bucket),
            "over_pct_cap_count": len(over_pct),
        }

    if not details_per:
        return _skip("formal train datasets exist but are empty")

    passed = len(errors) == 0
    return passed, {
        "errors": errors,
        "per_candidate": details_per,
        "caps": {
            "per_family_total": PER_FAMILY_TOTAL_CAP,
            "per_family_per_bucket": PER_FAMILY_PER_BUCKET_CAP,
            "single_family_pct": SINGLE_FAMILY_PERCENT_CAP,
        },
    }


# ---------------------------------------------------------------------------
# Check 20: Capacity verdict (NEW)
# ---------------------------------------------------------------------------

def check20_capacity_verdict() -> Tuple[bool, dict]:
    """Verify formal dataset builder's capacity assessment verdict.

    When formal dataset manifests exist, read the capacity_assessment from
    them (normal post-build path).

    When formal dataset manifests do NOT exist but the pool manifest exists,
    compute capacity directly from the pool's bucket_counts (pre-build path).
    This allows the gate to return MBPP_FAMILY_OR_VARIANT_LIMIT without
    requiring the dataset builder to have run -- which is critical when the
    pool is already known to be insufficient (the builder exits early without
    producing manifests).

    Sets ``capacity_status`` in details for verdict computation:
      - "SKIP" if manifests not built AND pool capacity is sufficient
        (datasets still need to be built)
      - "LIMIT" if any candidate's max_achievable < 2300
      - "OK" if all verdicts are FEASIBLE or AT_RISK
    """
    if not _formal_manifests_exist():
        # Fallback: compute capacity from pool manifest's bucket_counts.
        if not FORMAL_POOL_MANIFEST_PATH.exists():
            return _skip("pool manifest not built yet", capacity_status="SKIP")

        with FORMAL_POOL_MANIFEST_PATH.open(encoding="utf-8") as fh:
            pool_manifest = json.load(fh)
        pool_buckets = pool_manifest.get("bucket_counts", {})
        if not pool_buckets:
            return _skip("pool manifest has no bucket_counts",
                         capacity_status="SKIP")

        # Compute max_achievable for each candidate.
        details_per: dict = {}
        capacity_status = "OK"
        notes: list[str] = []
        for name, targets in _FORMAL_CANDIDATE_TARGETS.items():
            per_bucket_max = {
                v: min(targets[v], pool_buckets.get(v, 0))
                for v in _FORMAL_BUCKETS
            }
            max_achievable = sum(per_bucket_max.values())
            details_per[name] = {
                "max_achievable_total": max_achievable,
                "per_bucket_max": per_bucket_max,
                "source": "pool_fallback",
            }
            if max_achievable < MIN_TRAIN_SAMPLES_FOR_FULL:
                notes.append(
                    f"{name}: max_achievable={max_achievable} < "
                    f"{MIN_TRAIN_SAMPLES_FOR_FULL} (pool-based fallback)"
                )
                capacity_status = "LIMIT"

        # passed=True even when LIMIT: the check correctly identified the
        # capacity issue. capacity_status="LIMIT" drives the verdict, not
        # has_fail. Setting passed=False would trigger FIX_FIRST which is
        # wrong for a capacity insufficiency.
        return True, {
            "notes": notes,
            "per_candidate": details_per,
            "capacity_status": capacity_status,
            "source": "pool_fallback",
        }

    manifests = [
        ("balanced", FORMAL_BALANCED_MANIFEST_PATH),
        ("repair", FORMAL_REPAIR_MANIFEST_PATH),
    ]
    errors: list[str] = []  # genuine check failures (unexpected verdicts)
    notes: list[str] = []   # capacity insufficiency notes (not failures)
    details_per: dict = {}
    capacity_status = "OK"

    for name, path in manifests:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        ca = manifest.get("capacity_assessment", {})
        verdict = ca.get("verdict", "")
        max_achievable = ca.get("max_achievable_total", 0)

        if verdict == "MBPP_FAMILY_OR_VARIANT_LIMIT":
            # Capacity insufficiency is NOT a check failure -- it's a
            # capacity_status="LIMIT" that drives the verdict.
            notes.append(
                f"{name}: verdict={verdict} (capacity insufficient, "
                f"max_achievable={max_achievable} < {MIN_TRAIN_SAMPLES_FOR_FULL})"
            )
            capacity_status = "LIMIT"
        elif verdict in VALID_CAPACITY_VERDICTS:
            pass  # OK
        else:
            errors.append(
                f"{name}: verdict={verdict!r} (unexpected, expected one of "
                f"{VALID_CAPACITY_VERDICTS} or MBPP_FAMILY_OR_VARIANT_LIMIT)"
            )
            capacity_status = "LIMIT"

        details_per[name] = {
            "verdict": verdict,
            "max_achievable_total": max_achievable,
        }

    if not details_per:
        return _skip("formal dataset manifests exist but are empty", capacity_status="SKIP")

    # passed=False only for genuine errors (unexpected verdicts), not for
    # MBPP_FAMILY_OR_VARIANT_LIMIT which is a valid capacity assessment.
    passed = len(errors) == 0
    return passed, {
        "errors": errors,
        "notes": notes,
        "per_candidate": details_per,
        "capacity_status": capacity_status,
    }


# ---------------------------------------------------------------------------
# Check registry + verdict
# ---------------------------------------------------------------------------

CHECK_NAMES = [
    ("check1_frozen_v4_sha_locked", "Frozen v4 SHA locked"),
    ("check2_family_isolation_formal", "Family isolation (formal train disjoint)"),
    ("check3_assistant_retention_formal", "Assistant retention = 100% (formal)"),
    ("check4_silent_truncation_zero_formal", "Silent truncation = 0 (formal)"),
    ("check5_canary_all_fail", "Canary all fail (v4)"),
    ("check6a_cpu_smoke", "CPU smoke (mandatory)"),
    ("check6b_gpu_smoke", "GPU smoke (deferrable)"),
    ("check7_formal_output_dirs_dont_exist", "Formal output dirs don't exist"),
    ("check8_cpu_ci_green", "CPU CI green"),
    ("check9_baseline_lock_present", "P3 baseline lock present"),
    ("check10_formal_train_capacity", "Formal train capacity (>=2300/candidate)"),
    ("check11_verified_consistency_formal", "verified ⟺ verification (formal)"),
    ("check12_candidate_ratio_within_tolerance_formal", "Candidate ratio +/-3pp (formal)"),
    ("check13_all_buckets_non_empty_formal", "All buckets non-empty (formal)"),
    ("check14_composite_evaluator_complete", "Composite evaluator complete (5 comp)"),
    ("check15_v4_coverage_gate", "Frozen v4 coverage gate"),
    ("check16_validation_v2_gate", "Validation v2 gate"),
    ("check17_formal_pool_sha_lock", "Formal pool SHA lock"),
    ("check18_formal_config_validity", "Formal config validity"),
    ("check19_per_family_cap_enforcement", "Per-family cap enforcement (formal)"),
    ("check20_capacity_verdict", "Capacity verdict (formal builder)"),
]

# Check names that are reused from the existing gate (use existing formatter)
_REUSED_CHECK_NAMES = {
    "check1_frozen_v4_sha_locked",
    "check5_canary_all_fail",
    "check6a_cpu_smoke",
    "check6b_gpu_smoke",
    "check8_cpu_ci_green",
    "check9_baseline_lock_present",
    "check14_composite_evaluator_complete",
    "check15_v4_coverage_gate",
    "check16_validation_v2_gate",
}


def compute_verdict(results: list[Tuple[bool, dict]]) -> str:
    """Four-state verdict for the formal readiness gate.

    - FIX_FIRST: any mandatory check FAILS
    - PENDING_DATASET_BUILD: all checks PASS (incl. SKIP) but formal datasets
          not built yet -- capacity cannot be verified
    - MBPP_FAMILY_OR_VARIANT_LIMIT: all checks PASS but any candidate's formal
          capacity < 2300 (or capacity verdict is MBPP_FAMILY_OR_VARIANT_LIMIT)
    - GO_FOR_P3_TRAINING: all checks PASS and both candidates' capacity >= 2300
    """
    has_fail = False
    capacity_skip = False
    capacity_limit = False

    for passed, details in results:
        if not passed:
            has_fail = True
        cs = details.get("capacity_status")
        if cs == "SKIP":
            capacity_skip = True
        elif cs == "LIMIT":
            capacity_limit = True

    if has_fail:
        return "FIX_FIRST"
    # LIMIT takes priority over SKIP: if we definitively know capacity is
    # insufficient (e.g. from pool bucket_counts), MBPP_FAMILY_OR_VARIANT_LIMIT
    # is the correct terminal state -- not PENDING_DATASET_BUILD.
    if capacity_limit:
        return "MBPP_FAMILY_OR_VARIANT_LIMIT"
    if capacity_skip:
        return "PENDING_DATASET_BUILD"
    return "GO_FOR_P3_TRAINING"


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _format_details_short(name: str, passed: bool, details: dict) -> str:
    """One-line summary of the details for the table."""
    # Reused checks: delegate to existing formatter
    if name in _REUSED_CHECK_NAMES:
        return _format_details_short_existing(name, passed, details)

    # SKIP-aware helper
    if details.get("skipped") is True:
        return details.get("reason", "SKIP")

    if name == "check2_family_isolation_formal":
        if passed:
            counts = details.get("counts", {})
            wl = details.get("whitelist", {})
            return (
                f"formal={counts.get('formal_train', 0)} "
                f"val_v2={counts.get('validation_v2', 0)} "
                f"frozen_v4={counts.get('frozen_v4', 0)} "
                f"hist_frozen={counts.get('historical_frozen', 0)} "
                f"hist_val={counts.get('historical_validation', 0)} "
                f"wl=replay∩p2({wl.get('intersection_count', 0)})"
            )
        return f"violations={details.get('violations', [])[:2]}"

    if name == "check3_assistant_retention_formal":
        return f"{details.get('retained')}/{details.get('checked')} samples"

    if name == "check4_silent_truncation_zero_formal":
        return (
            f"real_silent={details.get('real_silent_truncations', 0)} "
            f"potential={details.get('potential_truncations', 0)} "
            f"/ {details.get('checked', 0)} (preserve_assistant)"
        )

    if name == "check7_formal_output_dirs_dont_exist":
        if passed:
            return f"{len(details.get('checked', []))} paths checked, none exist"
        return f"existing={details.get('existing', [])}"

    if name == "check10_formal_train_capacity":
        cs = details.get("capacity_status", "?")
        return (
            f"balanced={details.get('balanced_train', 0)}"
            f"[{details.get('balanced_verdict', '?')}] "
            f"repair={details.get('repair_train', 0)}"
            f"[{details.get('repair_verdict', '?')}] "
            f"status={cs}"
        )

    if name == "check11_verified_consistency_formal":
        return (
            f"{details.get('inconsistent_count', 0)}/"
            f"{details.get('checked', 0)} inconsistent"
        )

    if name == "check12_candidate_ratio_within_tolerance_formal":
        if passed:
            return (
                f"balanced={details.get('balanced_counts', {})} "
                f"repair={details.get('repair_counts', {})} "
                f"tol=+/-{details.get('tolerance_pp', 3)}pp"
            )
        return f"violations={details.get('violations', [])}"

    if name == "check13_all_buckets_non_empty_formal":
        if passed:
            return "all 8 buckets non-empty"
        return f"empty={details.get('empty_buckets', [])}"

    if name == "check17_formal_pool_sha_lock":
        if passed:
            return (
                f"sha={details.get('pool_sha256', '')[:16]}... "
                f"samples={details.get('total_samples')} "
                f"families={details.get('total_families')}"
            )
        return f"error={details.get('error', 'unknown')}"

    if name == "check18_formal_config_validity":
        if passed:
            cfgs = details.get("configs", {})
            return f"{len(cfgs)} configs valid (independent, null adapter, 2 epochs)"
        return f"errors={details.get('errors', [])[:3]}"

    if name == "check19_per_family_cap_enforcement":
        if passed:
            per = details.get("per_candidate", {})
            parts = [
                f"{k}: fam={v.get('family_count', 0)} "
                f"max={v.get('max_family_total', 0)}"
                for k, v in per.items()
            ]
            return "; ".join(parts)
        return f"errors={details.get('errors', [])[:2]}"

    if name == "check20_capacity_verdict":
        per = details.get("per_candidate", {})
        source = details.get("source", "manifest")
        parts = []
        for k, v in per.items():
            verdict = v.get("verdict", "?")
            max_a = v.get("max_achievable_total", 0)
            if source == "pool_fallback":
                parts.append(f"{k}: max={max_a}")
            else:
                parts.append(f"{k}: {verdict}(max={max_a})")
        status = details.get("capacity_status", "?")
        if parts:
            return "; ".join(parts) + f" status={status} src={source}"
        return f"status={status}"

    return json.dumps(details, ensure_ascii=False)[:80]


def render_report(results: list[Tuple[bool, dict]], verdict: str) -> str:
    """Render the formal readiness report as Markdown."""
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# P3 Formal Readiness Gate v2 Report")
    lines.append("")
    lines.append(f"**Generated**: {now}")
    lines.append("**Branch**: feat/p3-capability-expansion-v2")
    lines.append("**Wave**: 5-J (Issue #14)")
    lines.append("**Scope**: Formal training data + configs + infrastructure readiness.")
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    lines.append(f"## {len(CHECK_NAMES)} Checks (check6 split into 6a/6b)")
    lines.append("")
    lines.append("| # | Check | Status | Details |")
    lines.append("|---|---|---|---|")
    for i, (name, label) in enumerate(CHECK_NAMES, 1):
        passed, details = results[i - 1]
        status = _status_token(passed, details)
        short = _format_details_short(name, passed, details)
        lines.append(f"| {i} | {label} | {status} | {short} |")
    lines.append("")

    # SKIP summary
    skipped = [
        (i, name, label, details.get("reason", ""))
        for i, (name, label) in enumerate(CHECK_NAMES, 1)
        for _, details in [results[i - 1]]
        if details.get("skipped") is True
    ]
    if skipped:
        lines.append("## SKIP Summary")
        lines.append("")
        lines.append("The following checks were SKIPped (datasets not built yet). ")
        lines.append("SKIP does not fail the gate. Run the formal dataset builder, then re-run this gate.")
        lines.append("")
        for idx, name, label, reason in skipped:
            lines.append(f"- **Check {idx} ({label})**: {reason}")
        lines.append("")

    # Verdict logic explanation
    lines.append("## Verdict Logic")
    lines.append("")
    lines.append("- **GO_FOR_P3_TRAINING**: all mandatory checks PASS AND both candidates' formal capacity >= 2300.")
    lines.append("- **MBPP_FAMILY_OR_VARIANT_LIMIT**: all checks PASS but any candidate's formal capacity < 2300 (or capacity verdict is MBPP_FAMILY_OR_VARIANT_LIMIT).")
    lines.append("- **FIX_FIRST**: any mandatory check FAILS.")
    lines.append("- **PENDING_DATASET_BUILD**: all checks PASS (incl. SKIP) but formal datasets not built yet -- capacity cannot be verified.")
    lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    if verdict == "GO_FOR_P3_TRAINING":
        lines.append("**GO_FOR_P3_TRAINING** -- all checks PASS and both candidates' formal capacity >= 2300.")
        lines.append("Formal P3 training may proceed under user approval.")
        lines.append("")
        lines.append("Pre-flight checklist before launching training:")
        lines.append("- Confirm formal configs (`configs/p3/*-formal-v1.yaml`) are correct.")
        lines.append("- Confirm GPU smoke (Check 6b) PASSed (or SKIP is acceptable if GPU was verified previously).")
        lines.append("- Confirm formal output dirs are empty/non-existent (Check 7).")
        lines.append("- Confirm per-family caps enforced (Check 19).")
    elif verdict == "MBPP_FAMILY_OR_VARIANT_LIMIT":
        lines.append("**MBPP_FAMILY_OR_VARIANT_LIMIT** -- all checks PASS but at least one candidate's formal capacity < 2300.")
        lines.append("")
        lines.append("The MBPP family/variant supply is insufficient for formal training at the 2300-sample threshold.")
        lines.append("Options:")
        lines.append("- Expand the candidate pool (more families or more variants per family).")
        lines.append("- Accept PILOT_ONLY training with reduced capacity (results must NOT be reported as formal capability).")
        lines.append("- Re-run the formal dataset builder after pool expansion.")
    elif verdict == "PENDING_DATASET_BUILD":
        lines.append("**PENDING_DATASET_BUILD** -- infrastructure checks PASS but formal datasets not built yet.")
        lines.append("")
        lines.append("The formal data/config/infrastructure is ready. Next steps:")
        lines.append("1. Run the formal pool builder: `py -3.11 scripts/p3_formal_pool_builder.py`")
        lines.append("2. Run the formal dataset builder: `py -3.11 scripts/p3_formal_dataset_builder.py --candidate both`")
        lines.append("3. Re-run this gate: `py -3.11 scripts/p3_formal_readiness_gate.py`")
        lines.append("")
        lines.append("After datasets are built, checks 3/4/10/11/12/13/19/20 will run (no longer SKIP) and the verdict")
        lines.append("will be determined by the actual data quality and capacity.")
    else:
        lines.append("**FIX_FIRST** -- at least one mandatory check FAILed.")
        lines.append("")
        lines.append("Failed checks:")
        lines.append("")
        for i, (name, label) in enumerate(CHECK_NAMES, 1):
            passed, details = results[i - 1]
            if not passed:
                short = _format_details_short(name, passed, details)
                lines.append(f"- **Check {i} ({label})**: {short}")
        lines.append("")
        lines.append("Fix the failures above, then re-run `py -3.11 scripts/p3_formal_readiness_gate.py`")
        lines.append("until the verdict becomes GO_FOR_P3_TRAINING or PENDING_DATASET_BUILD.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all checks, print summary, write report. Returns 0/1 exit code."""
    parser = argparse.ArgumentParser(
        description="P3 Formal Readiness Gate v2 (Wave 5-J, Issue #14). "
                    "Verifies formal training data, configs, and infrastructure readiness.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FORMAL_REPORT_PATH,
        help=f"Report output path (default: {FORMAL_REPORT_PATH.relative_to(_ROOT)})",
    )
    args = parser.parse_args()

    check_fns = [
        check1_frozen_v4_sha_locked,
        check2_family_isolation_formal,
        check3_assistant_retention_formal,
        check4_silent_truncation_zero_formal,
        check5_canary_all_fail,
        check6a_cpu_smoke,
        check6b_gpu_smoke,
        check7_formal_output_dirs_dont_exist,
        check8_cpu_ci_green,
        check9_baseline_lock_present,
        check10_formal_train_capacity,
        check11_verified_consistency_formal,
        check12_candidate_ratio_within_tolerance_formal,
        check13_all_buckets_non_empty_formal,
        check14_composite_evaluator_complete,
        check15_v4_coverage_gate,
        check16_validation_v2_gate,
        check17_formal_pool_sha_lock,
        check18_formal_config_validity,
        check19_per_family_cap_enforcement,
        check20_capacity_verdict,
    ]

    results: list[Tuple[bool, dict]] = []
    total_checks = len(CHECK_NAMES)
    print("=" * 78)
    print(f"P3 Formal Readiness Gate v2 -- {total_checks} Checks (Wave 5-J, Issue #14)")
    print("=" * 78)
    print()

    for i, (name, label) in enumerate(CHECK_NAMES, 1):
        fn = check_fns[i - 1]
        print(f"[{i}/{total_checks}] {label} ...", flush=True)
        try:
            passed, details = fn()
        except Exception as e:  # pragma: no cover -- defensive
            passed = False
            details = {"error": f"{type(e).__name__}: {e}"}
        results.append((passed, details))
        status = _status_token(passed, details)
        print(f"      -> {status}: {_format_details_short(name, passed, details)}")
        print()

    verdict = compute_verdict(results)

    # Print summary table
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"| # | Check                                      | Status |")
    print(f"|---|---|---|")
    for i, (name, label) in enumerate(CHECK_NAMES, 1):
        passed, details = results[i - 1]
        status = _status_token(passed, details)
        print(f"| {i} | {label:<42} | {status:5} |")
    print()
    print(f"Verdict: {verdict}")
    print()

    # Write report
    report_md = render_report(results, verdict)
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md, encoding="utf-8", newline="\n")
    print(f"Report written: {output_path}")
    print()

    # Exit code: 0 for GO_FOR_P3_TRAINING and PENDING_DATASET_BUILD (green light
    # for next step); 1 for MBPP_FAMILY_OR_VARIANT_LIMIT and FIX_FIRST (needs
    # attention).
    return 0 if verdict in ("GO_FOR_P3_TRAINING", "PENDING_DATASET_BUILD") else 1


if __name__ == "__main__":
    sys.exit(main())
