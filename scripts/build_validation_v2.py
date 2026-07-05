"""scripts/build_validation_v2.py -- Build Validation v2 (4 categories).

Generates a 4-category validation set (code/boundary/static_repair/
execution_repair) with 45 samples per category (180 total) from the
p3_validation families.

Pipeline
--------
1. Load the 90 code validation samples from MBPP verified (filtered by
   p3_validation family_ids -- same loader as build_balanced_generalist).
2. Sub-sample 45 code samples (seed=42, sorted by sample_id ascending,
   then random.Random(42).sample).
3. For each of the 45 code samples:
   a. Keep the code sample (variant_type="code").
   b. Generate boundary variant via generate_boundary_variant().
   c. Generate repair pairs via build_repair_samples().
4. Post-process variant_type/bug_type/source_split, normalize public_tests
   for pytest, then verify each generated sample with verify_sample() and
   verify_broken_is_broken() (for repair samples).
5. Collect verified samples per category until 45 each is reached. If a
   source sample's first variant fails verification, fall back to the next
   pair (for repair) or skip (for boundary).
6. Write validation.jsonl (180 samples), manifest.json (with SHA256 + per-
   category counts), families.json (p3_validation family_ids), and
   rejected.jsonl (samples that failed verification).
7. Run hard gates (4 categories × 45, all verified=True, family disjoint
   from p3_train + frozen_v3, no duplicate sample_ids).

Usage
-----
    python scripts/build_validation_v2.py

Exit codes
----------
    0   success
    1   invariant violation (hard gate failed) or I/O error
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
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
# Python 3.8 compatibility: ast.unparse was added in Python 3.9, but this
# environment runs Python 3.8.10. inject_bugs._try_transform and
# mutate_code._try_transform both call ast.unparse at runtime. Patch once.
# ---------------------------------------------------------------------------
if not hasattr(ast, "unparse"):  # pragma: no cover - environment-specific
    try:
        import astunparse  # type: ignore[import-not-found]

        ast.unparse = astunparse.unparse  # type: ignore[attr-defined]
    except ImportError:
        pass  # let inject_bugs fail loudly when ast.unparse is missing

from scripts.build_execution_repair import build_repair_samples  # noqa: E402
from scripts.generate_boundary_variants import generate_boundary_variant  # noqa: E402
from src.hidden_test_padding import normalize_public_tests_for_pytest  # noqa: E402
from src.schemas import Sample  # noqa: E402
from src.validators import verify_broken_is_broken, verify_sample  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
GENERATOR_NAME: str = "build_validation_v2.py"
SEED: int = 42

TARGET_PER_CATEGORY: int = 45
TOTAL_TARGET: int = TARGET_PER_CATEGORY * 4  # 180
CATEGORIES: tuple = ("code", "boundary", "static_repair", "execution_repair")

PYTEST_TIMEOUT_S: float = 10.0

# Paths
PARTITION_PATH = _ROOT / "data" / "p3-curriculum" / "family-partition.json"
VALIDATION_SPLIT_PATH = (
    _ROOT / "data" / "external" / "mbpp" / "verified" / "validation.jsonl"
)
TEST_SPLIT_PATH = _ROOT / "data" / "external" / "mbpp" / "verified" / "test.jsonl"
FROZEN_V3_FAMILIES_PATH = _ROOT / "data" / "frozen-eval" / "v3" / "families.json"

OUTPUT_DIR = _ROOT / "data" / "p3-curriculum" / "validation-v2"
VALIDATION_PATH = OUTPUT_DIR / "validation.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
FAMILIES_PATH = OUTPUT_DIR / "families.json"
REJECTED_PATH = OUTPUT_DIR / "rejected.jsonl"


# ---------------------------------------------------------------------------
# Bug-type extraction + variant_type post-processing
# ---------------------------------------------------------------------------

import re

_BUG_TYPE_RE = re.compile(r".*_(sr|er)_(.+)$")


def extract_bug_type(sample_id: str) -> Optional[str]:
    """Extract bug_type from *sample_id* suffix (mirrors frozen v3 helper)."""
    m = _BUG_TYPE_RE.match(sample_id)
    if m:
        return m.group(2)
    return None


def post_process_variant_type(sample: Sample, source: str) -> Sample:
    """Set variant_type, bug_type, source_split on *sample* (mirrors frozen v3)."""
    updates: dict[str, Any] = {"source_split": "test"}
    if source == "code":
        updates["variant_type"] = "code"
        updates["bug_type"] = None
    elif source == "boundary":
        updates["variant_type"] = "boundary"
        updates["bug_type"] = None
    elif source == "static_repair":
        updates["variant_type"] = "static_repair"
        updates["bug_type"] = extract_bug_type(sample.sample_id)
    elif source == "execution_repair":
        updates["variant_type"] = "execution_repair"
        updates["bug_type"] = extract_bug_type(sample.sample_id)
    return sample.model_copy(update=updates)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relpath(path: Path) -> str:
    """Return path relative to _ROOT if possible, else absolute string."""
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


def _load_code_validation_samples(p3_validation_fids: set) -> list:
    """Load 90 code validation samples filtered by p3_validation family_ids.

    Mirrors build_balanced_generalist._load_validation_samples but returns
    only the kept samples (no count tuple).
    """
    val_split_samples = _load_jsonl_samples(VALIDATION_SPLIT_PATH)
    val_split_kept = [
        s for s in val_split_samples if s.family_id in p3_validation_fids
    ]
    test_split_samples = _load_jsonl_samples(TEST_SPLIT_PATH)
    test_split_kept = [
        s for s in test_split_samples if s.family_id in p3_validation_fids
    ]
    normalised: list = []
    for s in val_split_kept + test_split_kept:
        if s.variant_type != "code" or s.bug_type is not None:
            normalised.append(
                s.model_copy(update={"variant_type": "code", "bug_type": None})
            )
        else:
            normalised.append(s)
    return normalised


def _subsample_code(samples: list, target: int, *, seed: int = SEED) -> list:
    """Sort by sample_id ascending, then random.Random(seed).sample target."""
    sorted_samples = sorted(samples, key=lambda s: s.sample_id)
    if target >= len(sorted_samples):
        return list(sorted_samples)
    sorted_ids = [s.sample_id for s in sorted_samples]
    chosen_ids = set(random.Random(seed).sample(sorted_ids, target))
    return [s for s in sorted_samples if s.sample_id in chosen_ids]


# ---------------------------------------------------------------------------
# Per-sample variant generation + verification
# ---------------------------------------------------------------------------

def _verify_reference(sample: Sample) -> tuple[Optional[Sample], list[dict]]:
    """Verify a code/boundary reference sample.

    Returns (verified_sample_or_None, rejected_records).
    """
    sv = verify_sample(sample, pytest_timeout_s=PYTEST_TIMEOUT_S)
    updated = sample.model_copy(update={
        "verified": sv.is_accepted,
        "verification": sv.verification,
    })
    if sv.is_accepted:
        return updated, []
    return updated, [{
        "sample_id": sample.sample_id,
        "family_id": sample.family_id,
        "variant_type": sample.variant_type,
        "rejection_reason": "reference_verification_failed",
        "verification_messages": sv.messages,
    }]


def _verify_repair(sample: Sample) -> tuple[Optional[Sample], list[dict]]:
    """Verify a static_repair / execution_repair sample.

    Returns (verified_sample_or_None, rejected_records).

    Acceptance requires BOTH:
      - verify_sample.is_accepted (target_code passes public+hidden tests)
      - verify_broken_is_broken (broken_code genuinely fails >= 1 test)
    For execution_repair, also requires non-empty execution_feedback.
    """
    reasons: list[str] = []
    sv = verify_sample(sample, pytest_timeout_s=PYTEST_TIMEOUT_S)
    updated = sample.model_copy(update={
        "verified": sv.is_accepted,
        "verification": sv.verification,
    })
    if not sv.is_accepted:
        reasons.append("repair_verification_failed")
    try:
        broken_ok = verify_broken_is_broken(
            sample, pytest_timeout_s=PYTEST_TIMEOUT_S
        )
    except ValueError:
        broken_ok = False
        reasons.append("broken_code_empty")
    if not broken_ok:
        reasons.append("broken_code_not_broken")
    if sample.variant_type == "execution_repair":
        if not (sample.execution_feedback or "").strip():
            reasons.append("execution_feedback_empty")

    if reasons:
        return updated, [{
            "sample_id": sample.sample_id,
            "family_id": sample.family_id,
            "variant_type": sample.variant_type,
            "rejection_reason": ",".join(reasons),
            "verification_messages": sv.messages,
        }]
    return updated, []


def _process_source_sample(
    source: Sample,
) -> tuple[
    Optional[Sample],          # code
    Optional[Sample],          # boundary
    list[Sample],              # static_repair candidates
    list[Sample],              # execution_repair candidates
    list[dict],                # rejected records
]:
    """Generate + post-process + verify all variants for one source sample.

    Returns (code, boundary, static_candidates, exec_candidates, rejected).
    Each verified sample has verified=True and real verification subfields.
    Candidates lists contain verified repair samples (ready for selection).
    """
    rejected: list[dict] = []

    # 1. Code (reference) -- post-process then verify
    code_pp = post_process_variant_type(source, "code")
    code_pp = code_pp.model_copy(update={
        "public_tests": normalize_public_tests_for_pytest(code_pp.public_tests),
    })
    code_verified, code_rej = _verify_reference(code_pp)
    rejected.extend(code_rej)
    code_out = code_verified if code_verified is not None and code_verified.verified else None

    # 2. Boundary variant
    boundary_out: Optional[Sample] = None
    bv = generate_boundary_variant(source)
    if bv is None:
        rejected.append({
            "sample_id": f"{source.sample_id}_boundary",
            "family_id": source.family_id,
            "variant_type": "boundary",
            "rejection_reason": "boundary_variant_generation_failed",
            "verification_messages": [],
        })
    else:
        bv_pp = post_process_variant_type(bv, "boundary")
        bv_pp = bv_pp.model_copy(update={
            "public_tests": normalize_public_tests_for_pytest(bv_pp.public_tests),
        })
        bv_verified, bv_rej = _verify_reference(bv_pp)
        rejected.extend(bv_rej)
        if bv_verified is not None and bv_verified.verified:
            boundary_out = bv_verified

    # 3. Repair pairs (static + execution)
    static_candidates: list[Sample] = []
    exec_candidates: list[Sample] = []
    try:
        pairs = build_repair_samples(
            source, timeout_s=PYTEST_TIMEOUT_S, seed=SEED
        )
    except Exception as exc:
        pairs = []
        rejected.append({
            "sample_id": source.sample_id,
            "family_id": source.family_id,
            "variant_type": None,
            "rejection_reason": f"build_repair_samples_exception: {exc}",
            "verification_messages": [],
        })

    for sr, er in pairs:
        if sr is not None:
            sr_pp = post_process_variant_type(sr, "static_repair")
            sr_pp = sr_pp.model_copy(update={
                "public_tests": normalize_public_tests_for_pytest(sr_pp.public_tests),
            })
            sr_verified, sr_rej = _verify_repair(sr_pp)
            rejected.extend(sr_rej)
            if sr_verified is not None and sr_verified.verified:
                static_candidates.append(sr_verified)
        if er is not None:
            er_pp = post_process_variant_type(er, "execution_repair")
            er_pp = er_pp.model_copy(update={
                "public_tests": normalize_public_tests_for_pytest(er_pp.public_tests),
            })
            er_verified, er_rej = _verify_repair(er_pp)
            rejected.extend(er_rej)
            if er_verified is not None and er_verified.verified:
                exec_candidates.append(er_verified)

    return code_out, boundary_out, static_candidates, exec_candidates, rejected


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------

def _run_hard_gates(
    samples: list,
    *,
    p3_train_fids: set,
    frozen_v3_fids: set,
) -> list:
    """Run hard gates. Returns list of error messages (empty = pass).

    Gates:
      1. Total count == 180
      2. Each category has exactly 45 samples
      3. All samples have verified=True
      4. Family IDs disjoint from p3_train
      5. Family IDs disjoint from frozen_v3
      6. No duplicate sample_ids
    """
    errors: list = []

    # Gate 1: total count
    if len(samples) != TOTAL_TARGET:
        errors.append(
            f"gate 1: total count {len(samples)} != {TOTAL_TARGET}"
        )

    # Gate 2: 45 per category
    by_cat: dict = {c: 0 for c in CATEGORIES}
    for s in samples:
        vt = s.variant_type
        if vt in by_cat:
            by_cat[vt] += 1
    for c in CATEGORIES:
        if by_cat[c] != TARGET_PER_CATEGORY:
            errors.append(
                f"gate 2: category {c} count {by_cat[c]} != "
                f"{TARGET_PER_CATEGORY}"
            )

    # Gate 3: all verified=True
    unverified = [s.sample_id for s in samples if not s.verified]
    if unverified:
        errors.append(
            f"gate 3: {len(unverified)} samples with verified=False "
            f"(first 5: {unverified[:5]})"
        )

    # Gate 4: disjoint from p3_train
    val_fids = {s.family_id for s in samples}
    train_overlap = val_fids & p3_train_fids
    if train_overlap:
        errors.append(
            f"gate 4: {len(train_overlap)} family_ids in both validation "
            f"and p3_train (first 5: {sorted(train_overlap)[:5]})"
        )

    # Gate 5: disjoint from frozen_v3
    frozen_overlap = val_fids & frozen_v3_fids
    if frozen_overlap:
        errors.append(
            f"gate 5: {len(frozen_overlap)} frozen_v3 family_ids in "
            f"validation (first 5: {sorted(frozen_overlap)[:5]})"
        )

    # Gate 6: no duplicate sample_ids
    sids = [s.sample_id for s in samples]
    dups = _find_duplicates(sids)
    if dups:
        errors.append(
            f"gate 6: {len(dups)} duplicate sample_ids "
            f"(first 5: {dups[:5]})"
        )

    return errors


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
    # Load partition + frozen v3 families
    # ------------------------------------------------------------------
    if not PARTITION_PATH.exists():
        print(f"ERROR: partition not found: {PARTITION_PATH}", file=sys.stderr)
        return 1
    with PARTITION_PATH.open(encoding="utf-8") as fh:
        partition = json.load(fh)
    p3_validation_fids = set(partition["p3_validation"]["family_ids"])
    p3_train_fids = set(partition["p3_train_new"]["family_ids"]) | set(
        partition["p3_train_replay"]["family_ids"]
    )

    if not FROZEN_V3_FAMILIES_PATH.exists():
        print(f"ERROR: frozen v3 families not found: "
              f"{FROZEN_V3_FAMILIES_PATH}", file=sys.stderr)
        return 1
    with FROZEN_V3_FAMILIES_PATH.open(encoding="utf-8") as fh:
        frozen_v3_data = json.load(fh)
    frozen_v3_fids = set(frozen_v3_data["families"])

    # ------------------------------------------------------------------
    # Load + sub-sample 45 code samples
    # ------------------------------------------------------------------
    code_pool = _load_code_validation_samples(p3_validation_fids)
    print(f"Loaded {len(code_pool)} code validation samples from "
          f"p3_validation families")
    if len(code_pool) < TARGET_PER_CATEGORY:
        print(f"ERROR: code pool {len(code_pool)} < target "
              f"{TARGET_PER_CATEGORY}", file=sys.stderr)
        return 1

    code_selected = _subsample_code(code_pool, TARGET_PER_CATEGORY)
    print(f"Sub-sampled {len(code_selected)} code samples (seed={SEED})")

    # ------------------------------------------------------------------
    # Generate + verify variants per source sample
    # ------------------------------------------------------------------
    code_samples: list[Sample] = []
    boundary_samples: list[Sample] = []
    static_samples: list[Sample] = []
    exec_samples: list[Sample] = []
    all_rejected: list[dict] = []

    for i, source in enumerate(code_selected, 1):
        code, boundary, static_cand, exec_cand, rejected = (
            _process_source_sample(source)
        )
        all_rejected.extend(rejected)
        if code is not None and len(code_samples) < TARGET_PER_CATEGORY:
            code_samples.append(code)
        if boundary is not None and len(boundary_samples) < TARGET_PER_CATEGORY:
            boundary_samples.append(boundary)
        for sr in static_cand:
            if len(static_samples) >= TARGET_PER_CATEGORY:
                break
            static_samples.append(sr)
        for er in exec_cand:
            if len(exec_samples) >= TARGET_PER_CATEGORY:
                break
            exec_samples.append(er)
        print(f"[{i}/{len(code_selected)}] {source.sample_id}: "
              f"code={code is not None}, boundary={boundary is not None}, "
              f"static={len(static_cand)}, exec={len(exec_cand)} | "
              f"totals: code={len(code_samples)}, boundary={len(boundary_samples)}, "
              f"static={len(static_samples)}, exec={len(exec_samples)}")

    # ------------------------------------------------------------------
    # Check category counts
    # ------------------------------------------------------------------
    counts = {
        "code": len(code_samples),
        "boundary": len(boundary_samples),
        "static_repair": len(static_samples),
        "execution_repair": len(exec_samples),
    }
    print(f"\nCategory counts: {counts}")
    print(f"Total: {sum(counts.values())} (target {TOTAL_TARGET})")

    insufficient = [c for c, n in counts.items() if n < TARGET_PER_CATEGORY]
    if insufficient:
        print(f"\nWARNING: insufficient samples for categories: {insufficient}")
        print("Proceeding with reduced counts (PILOT ONLY)")

    # ------------------------------------------------------------------
    # Combine + sort by sample_id
    # ------------------------------------------------------------------
    all_samples = (
        code_samples + boundary_samples
        + static_samples + exec_samples
    )
    all_samples.sort(key=lambda s: s.sample_id)
    all_rejected.sort(key=lambda r: r.get("sample_id", ""))

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # validation.jsonl
    with VALIDATION_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for s in all_samples:
            fh.write(s.to_json_line())
            fh.write("\n")
    print(f"\nWrote validation.jsonl: {len(all_samples)} samples")

    # rejected.jsonl
    with REJECTED_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for r in all_rejected:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")
    print(f"Wrote rejected.jsonl: {len(all_rejected)} records")

    # Compute SHA256
    val_sha = _compute_sha256(VALIDATION_PATH)
    rejected_sha = _compute_sha256(REJECTED_PATH)

    # ------------------------------------------------------------------
    # Build manifest.json
    # ------------------------------------------------------------------
    val_family_ids = sorted({s.family_id for s in all_samples})
    val_family_count = len(val_family_ids)

    variant_distribution = {c: 0 for c in CATEGORIES}
    for s in all_samples:
        vt = s.variant_type
        if vt in variant_distribution:
            variant_distribution[vt] += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": GENERATOR_NAME,
        "seed": SEED,
        "target_per_category": TARGET_PER_CATEGORY,
        "total_samples": len(all_samples),
        "variant_distribution": variant_distribution,
        "families": {
            "validation_family_count": val_family_count,
            "validation_family_ids": val_family_ids,
            "disjoint_from": ["p3_train", "frozen_v3"],
        },
        "validation": {
            "count": len(all_samples),
            "sha256": val_sha,
            "path": _relpath(VALIDATION_PATH),
        },
        "rejected": {
            "count": len(all_rejected),
            "sha256": rejected_sha,
            "path": _relpath(REJECTED_PATH),
        },
        "source": {
            "code_pool_size": len(code_pool),
            "code_subsampled": len(code_selected),
            "p3_validation_family_count": len(p3_validation_fids),
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
        "generator": GENERATOR_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_family_count": val_family_count,
        "validation_family_ids": val_family_ids,
        "shared_with": ["balanced_generalist", "repair_specialist"],
        "disjoint_from": ["p3_train", "frozen_v3"],
    }
    with FAMILIES_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(families_payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote families.json: {val_family_count} validation families")

    # ------------------------------------------------------------------
    # Hard gates
    # ------------------------------------------------------------------
    errors = _run_hard_gates(
        all_samples,
        p3_train_fids=p3_train_fids,
        frozen_v3_fids=frozen_v3_fids,
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
    print("\nValidation v2 summary:")
    print(f"  total samples:    {len(all_samples)}")
    print(f"  variant dist:")
    for c in CATEGORIES:
        print(f"    {c:20s}: {variant_distribution[c]:3d}")
    print(f"  validation families: {val_family_count}")
    print(f"  rejected records:    {len(all_rejected)}")
    print(f"  validation SHA256:   {val_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
