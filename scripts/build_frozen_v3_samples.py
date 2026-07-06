"""scripts/build_frozen_v3_samples.py -- Build Frozen v3 samples + verify + freeze.

Generates samples on the 120 Frozen v3 candidate families, verifies them
with REAL pytest execution, applies the hidden>=3 hard gate (with
synthetic boundary-test padding), and freezes 80-100 qualified families
as the immutable Frozen v3 evaluation set.

Per P3 plan Global Constraints #6 (Frozen v3 write-once immutability),
#12 (public>=2/hidden>=3 + verify_broken_is_broken + real
execution_feedback), #19 (variant_type/bug_type optional fields).

Usage
-----
    python scripts/build_frozen_v3_samples.py \\
        --candidates data/frozen-eval/v3/candidates.json \\
        --mbpp-verified-dir data/external/mbpp/verified \\
        --output-dir data/frozen-eval/v3 \\
        --registry data/family-registry.json \\
        --seed 42 \\
        --timeout 10.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_p2_curriculum import generate_all_variants  # noqa: E402
from src.family_registry import FamilyRegistry  # noqa: E402
from src.hidden_test_padding import (  # noqa: E402
    normalize_public_tests_for_pytest,
    pad_hidden_tests,
)
from src.schemas import Sample, Verification  # noqa: E402
from src.validators import verify_broken_is_broken, verify_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
GENERATOR_NAME: str = "build_frozen_v3_samples.py"
CANDIDATE_TAG: str = "frozen_v3_candidate"
FROZEN_TAG: str = "frozen_v3"
MIN_FROZEN: int = 80
MAX_FROZEN: int = 100
TARGET_HIDDEN_COUNT: int = 3
CANARY_CODE: str = (
    'def canary_always_fails():\n'
    '    raise AssertionError("canary")\n'
)

# Reuse the placeholder Verification from generate_boundary_variants pattern.
_PLACEHOLDER_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)


# ---------------------------------------------------------------------------
# Bug-type extraction + variant_type post-processing
# ---------------------------------------------------------------------------

_BUG_TYPE_RE = re.compile(r".*_(sr|er)_(.+)$")


def extract_bug_type(sample_id: str) -> Optional[str]:
    """Extract bug_type from *sample_id* suffix.

    Parses ``.*_(sr|er)_(.+)$`` and returns group 2 (the bug_type).
    Returns ``None`` if the regex does not match.
    """
    m = _BUG_TYPE_RE.match(sample_id)
    if m:
        return m.group(2)
    return None


def post_process_variant_type(sample: Sample, source: str) -> Sample:
    """Set *variant_type*, *bug_type*, and *source_split* on *sample*.

    *source* is one of ``"code"``, ``"boundary"``, ``"static_repair"``,
    ``"execution_repair"``. The mapping follows the brief's table:

    | Source           | variant_type      | bug_type                       |
    |------------------|-------------------|--------------------------------|
    | code             | "code"            | None                           |
    | boundary         | "boundary"        | None                           |
    | static_repair    | "static_repair"   | extracted from sample_id suffix |
    | execution_repair | "execution_repair"| extracted from sample_id suffix |
    """
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
# Canary sample construction
# ---------------------------------------------------------------------------

def make_canary(sample: Sample) -> Sample:
    """Create a canary sample that MUST FAIL all tests.

    The canary is a negative-control sanity check: its target_code is a
    stub that does not define the real function under test, so any test
    calling the real function will raise ``NameError``.
    """
    return sample.model_copy(update={
        "sample_id": f"{sample.sample_id}_canary",
        "target_code": CANARY_CODE,
        "variant_type": "canary",
        "bug_type": None,
        "task_type": "code_generation",
        "source_split": "test",
        "broken_code": None,
        "execution_feedback": None,
        "verified": False,
        "verification": _PLACEHOLDER_VER,
    })


# ---------------------------------------------------------------------------
# Per-family verification
# ---------------------------------------------------------------------------

def verify_family(
    family_id: str,
    samples_by_source: dict[str, list[Sample]],
    canary: Sample,
    *,
    pytest_timeout_s: float = 10.0,
) -> tuple[bool, list[Sample], list[dict]]:
    """Verify all samples for one family.

    Returns ``(qualified, all_samples, rejected_records)``.

    *qualified* is True iff every sample passes its verification checks.
    *all_samples* includes every sample with updated ``verified`` /
    ``verification`` fields plus the canary (only meaningful when
    *qualified* is True).
    *rejected_records* includes only the failing samples.
    """
    all_samples: list[Sample] = []
    rejected: list[dict] = []

    # ---- Reference samples (code, boundary) ----
    for source in ("code", "boundary"):
        for s in samples_by_source.get(source, []):
            sv = verify_sample(s, pytest_timeout_s=pytest_timeout_s)
            updated = s.model_copy(update={
                "verified": sv.is_accepted,
                "verification": sv.verification,
            })
            all_samples.append(updated)
            if not sv.is_accepted:
                rejected.append({
                    "sample_id": s.sample_id,
                    "family_id": family_id,
                    "variant_type": s.variant_type,
                    "rejection_reason": "reference_verification_failed",
                    "verification_messages": sv.messages,
                })
                continue
            # Hard gates: public>=2, hidden>=3
            pub_count = s.public_tests.count("assert ")
            hid_count = s.hidden_tests.count("assert ")
            if pub_count < 2:
                rejected.append({
                    "sample_id": s.sample_id,
                    "family_id": family_id,
                    "variant_type": s.variant_type,
                    "rejection_reason": f"public_assertions_count_{pub_count}_lt_2",
                    "verification_messages": sv.messages,
                })
            if hid_count < 3:
                rejected.append({
                    "sample_id": s.sample_id,
                    "family_id": family_id,
                    "variant_type": s.variant_type,
                    "rejection_reason": f"hidden_assertions_count_{hid_count}_lt_3",
                    "verification_messages": sv.messages,
                })

    # ---- Repair samples (static_repair, execution_repair) ----
    for source in ("static_repair", "execution_repair"):
        for s in samples_by_source.get(source, []):
            sv = verify_sample(s, pytest_timeout_s=pytest_timeout_s)
            updated = s.model_copy(update={
                "verified": sv.is_accepted,
                "verification": sv.verification,
            })
            all_samples.append(updated)
            reasons: list[str] = []
            if not sv.is_accepted:
                reasons.append("repair_verification_failed")
            try:
                broken_ok = verify_broken_is_broken(
                    s, pytest_timeout_s=pytest_timeout_s
                )
            except ValueError:
                broken_ok = False
                reasons.append("broken_code_empty")
            if not broken_ok:
                reasons.append("broken_code_not_broken")
            if source == "execution_repair":
                if not (s.execution_feedback or "").strip():
                    reasons.append("execution_feedback_empty")
            if reasons:
                rejected.append({
                    "sample_id": s.sample_id,
                    "family_id": family_id,
                    "variant_type": s.variant_type,
                    "rejection_reason": ",".join(reasons),
                    "verification_messages": sv.messages,
                })

    # ---- Canary (must fail) ----
    cv = verify_sample(canary, pytest_timeout_s=pytest_timeout_s)
    updated_canary = canary.model_copy(update={
        "verified": cv.is_accepted,  # False for a valid canary
        "verification": cv.verification,
    })
    all_samples.append(updated_canary)
    if cv.is_accepted:
        rejected.append({
            "sample_id": canary.sample_id,
            "family_id": family_id,
            "variant_type": "canary",
            "rejection_reason": "canary_passed_verification",
            "verification_messages": cv.messages,
        })

    qualified = len(rejected) == 0
    return qualified, all_samples, rejected


# ---------------------------------------------------------------------------
# Freeze decision + registry update
# ---------------------------------------------------------------------------

def apply_freeze_decision(
    qualified_family_ids: list[str],
    *,
    min_count: int = MIN_FROZEN,
    max_count: int = MAX_FROZEN,
) -> tuple[list[str], list[str], str, str]:
    """Apply the freeze decision.

    Returns ``(frozen_ids, surplus_ids, decision, note)``.

    - qualified >= 100: freeze first 100 by family_id ascending;
      surplus reverts to available. *decision* = ``"freeze_100"``.
    - 80-99 qualified: freeze all. *decision* = ``"freeze_actual"``.
    - < 80 qualified: FIX_FIRST. *decision* = ``"fix_first"``.
    """
    sorted_ids = sorted(qualified_family_ids)
    n = len(sorted_ids)
    if n >= max_count:
        frozen = sorted_ids[:max_count]
        surplus = sorted_ids[max_count:]
        note = (
            f"Froze first {max_count} of {n} qualified families "
            f"(surplus {len(surplus)} reverted to available pool)."
        )
        return frozen, surplus, "freeze_100", note
    if n >= min_count:
        note = f"Froze all {n} qualified families."
        return sorted_ids, [], "freeze_actual", note
    note = (
        f"FIX_FIRST: only {n} qualified families, need >= {min_count}. "
        f"Pipeline must be fixed before re-running."
    )
    return [], sorted_ids, "fix_first", note


def update_registry(
    registry: FamilyRegistry,
    frozen_ids: list[str],
    surplus_ids: list[str],
    rejected_family_ids: list[str],
) -> None:
    """Update *registry* in-place.

    - Frozen families: claim ``frozen_v3`` (keep ``frozen_v3_candidate``).
    - Surplus qualified families: unclaim ``frozen_v3_candidate``.
    - Rejected families: unclaim ``frozen_v3_candidate``.
    """
    for fid in frozen_ids:
        registry.claim(fid, FROZEN_TAG)
    for fid in surplus_ids:
        registry.unclaim(fid, CANDIDATE_TAG)
    for fid in rejected_family_ids:
        registry.unclaim(fid, CANDIDATE_TAG)


# ---------------------------------------------------------------------------
# Output file writers
# ---------------------------------------------------------------------------

def write_families_json(
    path: Path,
    frozen_ids: list[str],
    decision: str,
    note: str,
) -> None:
    """Write ``families.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "frozen_family_count": len(frozen_ids),
        "freeze_decision": decision,
        "freeze_decision_note": note,
        "families": list(frozen_ids),
    }
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def write_test_raw_jsonl(path: Path, samples: list[Sample]) -> None:
    """Write ``test_raw.jsonl`` (one Sample per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(s.to_json_line() + "\n")


def write_rejected_jsonl(path: Path, records: list[dict]) -> None:
    """Write ``rejected.jsonl`` (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_sha_lock(
    families_path: Path,
    test_raw_path: Path,
    rejected_path: Path,
) -> str:
    """Compute ``sha256(families.json + test_raw.jsonl + rejected.jsonl)``.

    Per the brief's chicken-and-egg resolution: ``sha_lock`` is computed over
    the 3 non-manifest files (families.json + test_raw.jsonl + rejected.jsonl).
    The manifest's own ``immutability.sha_lock`` field stores this value.

    CRLF→LF normalization ensures cross-platform SHA consistency.
    """
    h = hashlib.sha256()
    for p in (families_path, test_raw_path, rejected_path):
        with p.open("rb") as fh:
            data = fh.read()
        # Normalize CRLF to LF for cross-platform consistency
        h.update(data.replace(b"\r\n", b"\n"))
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    """SHA256 hex digest of *path*'s raw bytes (CRLF normalized to LF)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        data = fh.read()
    # Normalize CRLF to LF for cross-platform consistency
    h.update(data.replace(b"\r\n", b"\n"))
    return h.hexdigest()


def write_manifest_json(
    path: Path,
    *,
    frozen_family_count: int,
    total_sample_count: int,
    variant_breakdown: dict[str, int],
    test_raw_path: Path,
    families_path: Path,
    rejected_path: Path,
    rejected_count: int,
    decision: str,
    note: str,
) -> str:
    """Write ``manifest.json`` and return the ``sha_lock`` value.

    The ``sha_lock`` is computed over the 3 non-manifest files (families,
    test_raw, rejected) and stored in ``immutability.sha_lock``.
    """
    sha_lock = compute_sha_lock(families_path, test_raw_path, rejected_path)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "frozen_version": "v3",
        "frozen_family_count": frozen_family_count,
        "total_sample_count": total_sample_count,
        "variant_breakdown": variant_breakdown,
        "test_raw_sha256": sha256_file(test_raw_path),
        "families_sha256": sha256_file(families_path),
        "rejected_count": rejected_count,
        "rejected_sha256": sha256_file(rejected_path),
        "freeze_decision": decision,
        "freeze_decision_note": note,
        "immutability": {
            "write_once": True,
            "any_change_requires": "v4",
            "sha_lock": sha_lock,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return sha_lock


def verify_sha_lock(output_dir: Path) -> None:
    """Re-read the 3 non-manifest files + manifest.json and assert that
    the recomputed ``sha_lock`` matches what is stored in the manifest.
    """
    families_path = output_dir / "families.json"
    test_raw_path = output_dir / "test_raw.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    manifest_path = output_dir / "manifest.json"

    recomputed = compute_sha_lock(families_path, test_raw_path, rejected_path)
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    stored = manifest["immutability"]["sha_lock"]
    if recomputed != stored:
        raise AssertionError(
            f"sha_lock mismatch: stored={stored} recomputed={recomputed}"
        )


def variant_breakdown(samples: list[Sample]) -> dict[str, int]:
    """Count samples per ``variant_type``."""
    out: dict[str, int] = {
        "code": 0,
        "boundary": 0,
        "static_repair": 0,
        "execution_repair": 0,
        "canary": 0,
    }
    for s in samples:
        vt = s.variant_type or "code"
        out[vt] = out.get(vt, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Source-sample loading
# ---------------------------------------------------------------------------

def load_candidates(path: Path) -> list[str]:
    """Load the candidate family_ids from ``candidates.json``."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data["candidates"])


def load_source_samples(verified_dir: Path) -> dict[str, Sample]:
    """Load ``verified/test.jsonl`` into ``{family_id: Sample}``."""
    test_jsonl = verified_dir / "test.jsonl"
    out: dict[str, Sample] = {}
    with test_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = Sample.from_json_line(line)
            out[s.family_id] = s
    return out


# ---------------------------------------------------------------------------
# Per-family processing
# ---------------------------------------------------------------------------

def process_family(
    family_id: str,
    source: Sample,
    *,
    timeout_s: float,
    seed: int,
) -> tuple[bool, list[Sample], list[dict]]:
    """Generate, post-process, pad, and verify all samples for one family.

    Returns ``(qualified, all_samples, rejected_records)``.
    """
    # 1. Generate variants.
    code_gen, boundary, static_repair, execution_repair, gen_rejected = (
        generate_all_variants([source], timeout_s=timeout_s, seed=seed)
    )

    rejected_records: list[dict] = []
    for r in gen_rejected:
        rejected_records.append({
            "sample_id": r["sample_id"],
            "family_id": family_id,
            "variant_type": None,
            "rejection_reason": r["reason"],
            "verification_messages": [],
        })

    # 2. Post-process variant types + source_split.
    code_gen = [post_process_variant_type(s, "code") for s in code_gen]
    boundary = [post_process_variant_type(s, "boundary") for s in boundary]
    static_repair = [post_process_variant_type(s, "static_repair") for s in static_repair]
    execution_repair = [post_process_variant_type(s, "execution_repair") for s in execution_repair]

    # 2b. Normalize public_tests for mixed-format variants.
    # generate_boundary_variant appends pytest test functions (with
    # `from solution import`) after the original bare asserts, creating a
    # mixed file where bare asserts run before the import line. Prepend
    # `from solution import *` so bare asserts resolve. No-op for pure
    # bare-assert samples (code_gen, repair variants without `def test`).
    for lst in (code_gen, boundary, static_repair, execution_repair):
        for i, s in enumerate(lst):
            normalized = normalize_public_tests_for_pytest(s.public_tests)
            if normalized != s.public_tests:
                lst[i] = s.model_copy(update={"public_tests": normalized})

    # 3. Pad hidden tests (skip canary — canary is created later).
    samples_by_source: dict[str, list[Sample]] = {
        "code": [],
        "boundary": [],
        "static_repair": [],
        "execution_repair": [],
    }
    for source_list, source_name in (
        (code_gen, "code"),
        (boundary, "boundary"),
        (static_repair, "static_repair"),
        (execution_repair, "execution_repair"),
    ):
        for s in source_list:
            padded, pad_reason = pad_hidden_tests(s, target_count=TARGET_HIDDEN_COUNT)
            if pad_reason is not None:
                rejected_records.append({
                    "sample_id": s.sample_id,
                    "family_id": family_id,
                    "variant_type": s.variant_type,
                    "rejection_reason": pad_reason,
                    "verification_messages": [],
                })
                # Padding failed: sample cannot pass hidden>=3 gate.
                # Still add it to samples_by_source so verify_family can
                # record an additional rejection if needed; but the family
                # is already rejected via rejected_records.
            samples_by_source[source_name].append(padded if pad_reason is None else s)

    # 4. Generate canary (skip padding).
    canary = make_canary(source)

    # 5. Verify all samples.
    qualified, all_samples, family_rejected = verify_family(
        family_id, samples_by_source, canary, pytest_timeout_s=timeout_s
    )
    rejected_records.extend(family_rejected)
    return qualified, all_samples, rejected_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build Frozen v3 samples with REAL pytest verification and "
            "freeze 80-100 qualified families as the immutable eval set."
        )
    )
    p.add_argument("--candidates", required=True, help="Path to candidates.json.")
    p.add_argument(
        "--mbpp-verified-dir",
        required=True,
        help="Directory containing test.jsonl (e.g. data/external/mbpp/verified).",
    )
    p.add_argument("--output-dir", required=True, help="Output directory for frozen v3 files.")
    p.add_argument("--registry", required=True, help="Path to family-registry.json.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--timeout", type=float, default=10.0, help="Per-pytest timeout (s).")
    return p


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    candidates_path = Path(args.candidates)
    verified_dir = Path(args.mbpp_verified_dir)
    output_dir = Path(args.output_dir)
    registry_path = Path(args.registry)

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    try:
        candidates = load_candidates(candidates_path)
    except Exception as exc:
        print(f"ERROR: cannot load candidates: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(candidates)} candidates from {candidates_path}")

    try:
        source_samples = load_source_samples(verified_dir)
    except Exception as exc:
        print(f"ERROR: cannot load source samples: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(source_samples)} source samples from {verified_dir}")

    try:
        registry = FamilyRegistry.from_path(registry_path)
    except Exception as exc:
        print(f"ERROR: cannot load registry: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Process each candidate family
    # ------------------------------------------------------------------
    qualified_samples: dict[str, list[Sample]] = {}
    rejected_family_ids: list[str] = []
    all_rejected_records: list[dict] = []

    for i, fid in enumerate(candidates, 1):
        source = source_samples.get(fid)
        if source is None:
            all_rejected_records.append({
                "sample_id": fid,
                "family_id": fid,
                "variant_type": None,
                "rejection_reason": "source_sample_not_found",
                "verification_messages": [],
            })
            rejected_family_ids.append(fid)
            print(f"[{i}/{len(candidates)}] {fid}: REJECTED (source_sample_not_found)")
            continue

        try:
            qualified, all_samples, fam_rejected = process_family(
                fid, source, timeout_s=args.timeout, seed=args.seed
            )
        except Exception as exc:
            all_rejected_records.append({
                "sample_id": fid,
                "family_id": fid,
                "variant_type": None,
                "rejection_reason": f"processing_exception: {type(exc).__name__}: {exc}",
                "verification_messages": [],
            })
            rejected_family_ids.append(fid)
            print(f"[{i}/{len(candidates)}] {fid}: REJECTED (exception: {exc})")
            continue

        all_rejected_records.extend(fam_rejected)
        if qualified:
            qualified_samples[fid] = all_samples
            print(f"[{i}/{len(candidates)}] {fid}: QUALIFIED ({len(all_samples)} samples)")
        else:
            rejected_family_ids.append(fid)
            print(f"[{i}/{len(candidates)}] {fid}: REJECTED ({len(fam_rejected)} failures)")

    # ------------------------------------------------------------------
    # Apply freeze decision
    # ------------------------------------------------------------------
    frozen_ids, surplus_ids, decision, note = apply_freeze_decision(
        list(qualified_samples.keys())
    )

    print(f"\n=== Freeze Decision ===")
    print(f"  Qualified families: {len(qualified_samples)}")
    print(f"  Decision: {decision}")
    print(f"  Frozen: {len(frozen_ids)}")
    print(f"  Surplus: {len(surplus_ids)}")
    print(f"  Rejected: {len(rejected_family_ids)}")
    print(f"  Note: {note}")

    # ------------------------------------------------------------------
    # FIX_FIRST branch: write only rejected.jsonl, exit 1
    # ------------------------------------------------------------------
    if decision == "fix_first":
        rejected_path = output_dir / "rejected.jsonl"
        write_rejected_jsonl(rejected_path, all_rejected_records)
        print(f"\nWrote rejected.jsonl ({len(all_rejected_records)} records) to {rejected_path}")
        print("FIX_FIRST", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Success branch: write all 4 output files
    # ------------------------------------------------------------------
    frozen_samples: list[Sample] = []
    for fid in frozen_ids:
        frozen_samples.extend(qualified_samples[fid])

    families_path = output_dir / "families.json"
    test_raw_path = output_dir / "test_raw.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    manifest_path = output_dir / "manifest.json"

    write_families_json(families_path, frozen_ids, decision, note)
    write_test_raw_jsonl(test_raw_path, frozen_samples)
    write_rejected_jsonl(rejected_path, all_rejected_records)

    vb = variant_breakdown(frozen_samples)
    write_manifest_json(
        manifest_path,
        frozen_family_count=len(frozen_ids),
        total_sample_count=len(frozen_samples),
        variant_breakdown=vb,
        test_raw_path=test_raw_path,
        families_path=families_path,
        rejected_path=rejected_path,
        rejected_count=len(all_rejected_records),
        decision=decision,
        note=note,
    )

    # Verify sha_lock consistency.
    verify_sha_lock(output_dir)

    print(f"\n=== Output Files ===")
    print(f"  {families_path} ({len(frozen_ids)} families)")
    print(f"  {test_raw_path} ({len(frozen_samples)} samples)")
    print(f"  {manifest_path}")
    print(f"  {rejected_path} ({len(all_rejected_records)} records)")
    print(f"\n  variant_breakdown: {vb}")
    print(f"  sha_lock: verified OK")

    # ------------------------------------------------------------------
    # Update registry
    # ------------------------------------------------------------------
    update_registry(registry, frozen_ids, surplus_ids, rejected_family_ids)
    registry.to_path(registry_path)
    print(f"\n  registry updated: {registry_path}")
    print(f"    frozen_v3 claimed: {len(frozen_ids)}")
    print(f"    frozen_v3_candidate unclaimed (surplus+rejected): "
          f"{len(surplus_ids) + len(rejected_family_ids)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
