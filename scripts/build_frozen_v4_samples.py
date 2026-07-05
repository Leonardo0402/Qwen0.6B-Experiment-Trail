"""scripts/build_frozen_v4_samples.py -- Build Frozen v4 from MBPP verified sources.

Issue #12 P4 compliant rebuild: v4 is constructed INDEPENDENTLY from v3.
Families are selected from the family-registry such that they have ZERO
overlap with every historical dataset (frozen-eval v1/v3, p3-curriculum
balanced/repair/validation-v2, p3 family-partition p3_validation, and the
p2-curriculum stages + frozen-eval-v2).

For each chosen family the builder produces:
  - 1 code sample            (variant_type="code")
  - 1 boundary sample        (variant_type="boundary")  [from a subset]
  - 1 static_repair sample   (variant_type="static_repair")
  - 1 execution_repair sample(variant_type="execution_repair")
  - 1 canary sample          (variant_type="canary", verified=False)

Canary samples are stored in test_raw.jsonl but are NOT counted in
manifest.total_sample_count.

Target distribution (Issue #12 P4):
  - 80-100 new families
  - 360-700 non-canary samples
  - Code 25-30%, Boundary 15-20%, Static 25-30%, Exec 25-30%
  - SHA locked, immutable (any_change_requires="v5")

Usage
-----
    py -3.11 scripts/build_frozen_v4_samples.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_execution_repair import (  # noqa: E402
    _DATASET_VERSION as _ER_DATASET_VERSION,
    _GENERATOR as _ER_GENERATOR,
    compress_feedback,
    verify_bugged_fails,
)
from scripts.generate_boundary_variants import generate_boundary_variant  # noqa: E402
from scripts.inject_bugs import inject_all_bugs  # noqa: E402
from src.family_registry import FamilyRegistry  # noqa: E402
from src.schemas import Sample, Verification  # noqa: E402
from src.validators import compile_check, verify_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
GENERATOR_NAME: str = "build_frozen_v4_samples.py"
SEED: int = 42
PYTEST_TIMEOUT_S: float = 10.0

MIN_FAMILIES: int = 80
MAX_FAMILIES: int = 100
TARGET_FAMILIES: int = 100          # aim for 100 to clear the 360-sample floor
TARGET_BOUNDARY_COUNT: int = 65     # 65 / 365 = 17.8% (within 15-20%)

FROZEN_TAG: str = "frozen_v4"

CANARY_CODE: str = (
    'def canary_always_fails():\n'
    '    raise AssertionError("canary")\n'
)

_PLACEHOLDER_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)

V4_DIR = _ROOT / "data" / "frozen-eval" / "v4"
REGISTRY_PATH = _ROOT / "data" / "family-registry.json"
MBPP_VERIFIED_DIR = _ROOT / "data" / "external" / "mbpp" / "verified"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(s.to_json_line() + "\n")


def _write_rejected_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Historical family set (mirrors tests/test_frozen_v4_compliance.py)
# ---------------------------------------------------------------------------

def _all_historical_families() -> set[str]:
    """Collect ALL family IDs used in any historical dataset.

    Mirrors the logic in tests/test_frozen_v4_compliance.py so the build
    stays in sync with the compliance gate.
    """
    used: set[str] = set()

    # Frozen v1, v3
    for ver in ("v1", "v3"):
        fp = _ROOT / "data" / "frozen-eval" / ver / "families.json"
        if fp.exists():
            data = _load_json(fp)
            for item in data.get("families", []):
                if isinstance(item, str):
                    used.add(item)
                elif isinstance(item, dict):
                    fid = item.get("family_id") or item.get("id")
                    if isinstance(fid, str):
                        used.add(fid)

    # P3 validation partition
    fp = _ROOT / "data" / "p3-curriculum" / "family-partition.json"
    if fp.exists():
        part = _load_json(fp)
        used.update(part.get("p3_validation", {}).get("family_ids", []))

    # P3 candidates (balanced + repair + validation-v2)
    for sub in ("balanced-generalist", "repair-specialist", "validation-v2"):
        sub_path = _ROOT / "data" / "p3-curriculum" / sub
        if not sub_path.exists():
            continue
        fp = sub_path / "families.json"
        if fp.exists():
            data = _load_json(fp)
            for item in data.get("families", []):
                if isinstance(item, str):
                    used.add(item)
        for fname in ("train.jsonl", "validation.jsonl"):
            fp = sub_path / fname
            if fp.exists():
                for obj in _load_jsonl(fp):
                    fid = obj.get("family_id")
                    if isinstance(fid, str):
                        used.add(fid)

    # P2 datasets
    p2_root = _ROOT / "data" / "p2-curriculum"
    if p2_root.exists():
        fp = p2_root / "frozen-eval-v2" / "families.json"
        if fp.exists():
            data = _load_json(fp)
            for item in data.get("families", []):
                if isinstance(item, str):
                    used.add(item)
        for stage in ("stage1-code", "stage2-boundary", "stage3-repair"):
            stage_path = p2_root / stage
            if not stage_path.exists():
                continue
            fp = stage_path / "families.json"
            if fp.exists():
                data = _load_json(fp)
                for item in data.get("families", []):
                    if isinstance(item, str):
                        used.add(item)
            for fname in ("train.jsonl", "validation.jsonl"):
                fp = stage_path / fname
                if fp.exists():
                    for obj in _load_jsonl(fp):
                        fid = obj.get("family_id")
                        if isinstance(fid, str):
                            used.add(fid)

    return used


# ---------------------------------------------------------------------------
# Source sample loading
# ---------------------------------------------------------------------------

def _load_verified_sources() -> dict[str, Sample]:
    """Load all MBPP verified samples, indexed by family_id.

    Searches test.jsonl, train.jsonl, validation.jsonl in that order; the
    first hit per family_id wins (test split preferred).
    """
    out: dict[str, Sample] = {}
    for split in ("test", "train", "validation"):
        fp = MBPP_VERIFIED_DIR / f"{split}.jsonl"
        if not fp.exists():
            continue
        for obj in _load_jsonl(fp):
            fid = obj.get("family_id")
            if not isinstance(fid, str):
                continue
            if fid in out:
                continue
            try:
                out[fid] = Sample(**obj)
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------

def _make_code_sample(source: Sample) -> Sample:
    """Code variant: the original reference solution."""
    return source.model_copy(update={
        "variant_type": "code",
        "bug_type": None,
        "source_split": source.source_split or "test",
    })


def _make_canary(source: Sample) -> Sample:
    """Canary sample: broken_code is the canary stub, verified=False.

    The canary is a negative-control sanity check. Its broken_code is the
    CANARY_CODE stub (raises AssertionError when called) and its target_code
    is the real reference solution. verified is forced to False because the
    canary's broken_code is intentionally broken.
    """
    return Sample(
        sample_id=f"{source.sample_id}_canary",
        family_id=source.family_id,
        difficulty=source.difficulty,
        task_type="static_repair",
        language="python",
        skill_tags=list(source.skill_tags),
        instruction=source.instruction,
        broken_code=CANARY_CODE,
        execution_feedback=None,
        target_code=source.target_code,
        public_tests=source.public_tests,
        hidden_tests=source.hidden_tests,
        verified=False,
        verification=_PLACEHOLDER_VER,
        generator=GENERATOR_NAME,
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_version="v4",
        variant_type="canary",
        bug_type=None,
        source_split=source.source_split or "test",
    )


def _verify_and_tag(sample: Sample, variant_type: str) -> tuple[Optional[Sample], list[str]]:
    """Run verify_sample and return (updated_sample_or_None, reasons).

    On success the returned sample has verified=True and the real
    Verification embedded. On failure returns (None, reasons).
    """
    sv = verify_sample(sample, pytest_timeout_s=PYTEST_TIMEOUT_S)
    if not sv.is_accepted:
        return None, ["reference_verification_failed"]
    updated = sample.model_copy(update={
        "verified": True,
        "verification": sv.verification,
        "variant_type": variant_type,
    })
    return updated, []


def _verify_repair(sample: Sample, variant_type: str) -> tuple[Optional[Sample], list[str]]:
    """Verify a repair sample's target_code passes pytest.

    The broken_code's failure was already confirmed by ``verify_bugged_fails``
    in ``_build_first_repair_pair``, so we do NOT re-run
    ``verify_broken_is_broken`` here (that would duplicate pytest work).
    """
    sv = verify_sample(sample, pytest_timeout_s=PYTEST_TIMEOUT_S)
    if not sv.is_accepted:
        return None, ["repair_verification_failed"]

    if variant_type == "execution_repair":
        if not (sample.execution_feedback or "").strip():
            return None, ["execution_feedback_empty"]

    updated = sample.model_copy(update={
        "verified": True,
        "verification": sv.verification,
        "variant_type": variant_type,
    })
    return updated, []


# ---------------------------------------------------------------------------
# Fast repair pair builder (stops after first valid bug)
# ---------------------------------------------------------------------------

def _build_first_repair_pair(
    source: Sample,
) -> tuple[Optional[Sample], Optional[Sample], list[dict]]:
    """Find the first bug that produces a failing variant and build (sr, er).

    Iterates over ``inject_all_bugs`` output and returns the first (sr, er)
    pair where ``verify_bugged_fails`` confirms the broken_code fails at
    least one test. This is much faster than ``build_repair_samples`` which
    exhaustively checks all 8 bug types.

    Returns (static_repair, execution_repair, rejected_records).
    Either or both of the samples may be None if construction fails.
    """
    rejected: list[dict] = []
    variants = inject_all_bugs(source.target_code, seed=SEED)
    created_at = datetime.now(timezone.utc).isoformat()

    for bug_type, bugged_code, _description in variants:
        ok, _ = compile_check(bugged_code)
        if not ok:
            rejected.append({
                "sample_id": f"{source.sample_id}_{bug_type}",
                "family_id": source.family_id,
                "variant_type": None,
                "rejection_reason": f"compile_failed:{bug_type}",
            })
            continue

        is_broken, feedback = verify_bugged_fails(
            bugged_code,
            source.public_tests,
            source.hidden_tests,
            timeout_s=PYTEST_TIMEOUT_S,
        )
        if not is_broken:
            rejected.append({
                "sample_id": f"{source.sample_id}_{bug_type}",
                "family_id": source.family_id,
                "variant_type": None,
                "rejection_reason": f"bug_did_not_fail:{bug_type}",
            })
            continue

        # Found a genuinely failing bug — construct sr + er and return.
        sr_id = f"{source.sample_id}_sr_{bug_type}"
        er_id = f"{source.sample_id}_er_{bug_type}"

        repair_instruction = (
            f"{source.instruction}\n\n"
            "以下代码存在错误，请找出并修复，使其能通过所有测试用例。"
        )
        exec_instruction = (
            f"{source.instruction}\n\n"
            "以下代码存在错误，执行后出现以下问题，请修复代码。"
        )

        sr = Sample(
            sample_id=sr_id,
            family_id=source.family_id,
            difficulty=source.difficulty,
            task_type="static_repair",
            language="python",
            skill_tags=list(source.skill_tags) + [bug_type],
            instruction=repair_instruction,
            broken_code=bugged_code,
            execution_feedback=None,
            target_code=source.target_code,
            public_tests=source.public_tests,
            hidden_tests=source.hidden_tests,
            verified=False,
            verification=_PLACEHOLDER_VER,
            generator=_ER_GENERATOR,
            created_at=created_at,
            dataset_version=_ER_DATASET_VERSION,
        )

        er: Optional[Sample] = None
        if feedback.strip():
            er = Sample(
                sample_id=er_id,
                family_id=source.family_id,
                difficulty=3,
                task_type="execution_repair",
                language="python",
                skill_tags=list(source.skill_tags) + [bug_type],
                instruction=exec_instruction,
                broken_code=bugged_code,
                execution_feedback=feedback,
                target_code=source.target_code,
                public_tests=source.public_tests,
                hidden_tests=source.hidden_tests,
                verified=False,
                verification=_PLACEHOLDER_VER,
                generator=_ER_GENERATOR,
                created_at=created_at,
                dataset_version=_ER_DATASET_VERSION,
            )

        return sr, er, rejected

    return None, None, rejected


# ---------------------------------------------------------------------------
# Per-family processing
# ---------------------------------------------------------------------------

def _process_family(
    family_id: str,
    source: Sample,
) -> tuple[bool, dict, list[dict]]:
    """Generate + verify all variants for one family.

    Returns (qualified, samples_by_variant, rejected_records).

    *qualified* is True iff the family produces a valid code sample AND at
    least one (static_repair, execution_repair) pair where both members
    verify. *samples_by_variant* maps variant_type -> Sample (or list for
    repairs).
    """
    rejected: list[dict] = []
    samples_by_variant: dict[str, Optional[Sample]] = {
        "code": None,
        "boundary": None,
        "static_repair": None,
        "execution_repair": None,
    }

    # ---- Code sample ----
    code_sample = _make_code_sample(source)
    code_verified, code_reasons = _verify_and_tag(code_sample, "code")
    if code_verified is None:
        rejected.append({
            "sample_id": code_sample.sample_id,
            "family_id": family_id,
            "variant_type": "code",
            "rejection_reason": ",".join(code_reasons),
        })
        return False, samples_by_variant, rejected
    samples_by_variant["code"] = code_verified

    # ---- Boundary sample (best-effort; not required for qualification) ----
    try:
        bv = generate_boundary_variant(source)
    except Exception:
        bv = None
    if bv is not None:
        bv_verified, bv_reasons = _verify_and_tag(bv, "boundary")
        if bv_verified is not None:
            samples_by_variant["boundary"] = bv_verified
        else:
            rejected.append({
                "sample_id": bv.sample_id,
                "family_id": family_id,
                "variant_type": "boundary",
                "rejection_reason": ",".join(bv_reasons),
            })

    # ---- Repair pairs (static_repair + execution_repair) ----
    try:
        sr_candidate, er_candidate, pair_rejected = _build_first_repair_pair(source)
    except Exception as exc:
        rejected.append({
            "sample_id": source.sample_id,
            "family_id": family_id,
            "variant_type": "static_repair",
            "rejection_reason": f"build_repair_error: {exc}",
        })
        return False, samples_by_variant, rejected
    rejected.extend(pair_rejected)

    if sr_candidate is None or er_candidate is None:
        rejected.append({
            "sample_id": source.sample_id,
            "family_id": family_id,
            "variant_type": "static_repair",
            "rejection_reason": (
                f"no_valid_repair_pair sr={'ok' if sr_candidate else 'missing'}"
                f" er={'ok' if er_candidate else 'missing'}"
            ),
        })
        return False, samples_by_variant, rejected

    sr_accepted: Optional[Sample] = None
    er_accepted: Optional[Sample] = None

    sr_v, sr_reasons = _verify_repair(sr_candidate, "static_repair")
    if sr_v is not None:
        sr_accepted = sr_v
    else:
        rejected.append({
            "sample_id": sr_candidate.sample_id,
            "family_id": family_id,
            "variant_type": "static_repair",
            "rejection_reason": ",".join(sr_reasons),
        })

    er_v, er_reasons = _verify_repair(er_candidate, "execution_repair")
    if er_v is not None:
        er_accepted = er_v
    else:
        rejected.append({
            "sample_id": er_candidate.sample_id,
            "family_id": family_id,
            "variant_type": "execution_repair",
            "rejection_reason": ",".join(er_reasons),
        })

    if sr_accepted is None or er_accepted is None:
        rejected.append({
            "sample_id": source.sample_id,
            "family_id": family_id,
            "variant_type": "static_repair",
            "rejection_reason": (
                f"no_valid_repair_pair sr={'ok' if sr_accepted else 'missing'}"
                f" er={'ok' if er_accepted else 'missing'}"
            ),
        })
        return False, samples_by_variant, rejected

    samples_by_variant["static_repair"] = sr_accepted
    samples_by_variant["execution_repair"] = er_accepted
    return True, samples_by_variant, rejected


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_families_json(path: Path, frozen_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "frozen_version": "v4",
        "frozen_family_count": len(frozen_ids),
        "families": list(frozen_ids),
    }
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _write_manifest_json(
    path: Path,
    *,
    frozen_ids: list[str],
    total_sample_count: int,
    variant_breakdown: dict[str, int],
    canary_count: int,
    rejected_count: int,
    test_raw_path: Path,
    families_path: Path,
    rejected_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "frozen_version": "v4",
        "base_version": "mbpp_verified_direct",
        "generator": GENERATOR_NAME,
        "frozen_family_count": len(frozen_ids),
        "total_sample_count": total_sample_count,
        "variant_breakdown": variant_breakdown,
        "canary_count": canary_count,
        "canary_excluded_from_total": True,
        "test_raw_sha256": _sha256_file(test_raw_path),
        "families_sha256": _sha256_file(families_path),
        "rejected_count": rejected_count,
        "rejected_sha256": _sha256_file(rejected_path) if rejected_path.exists() else "",
        "immutability": {
            "write_once": True,
            "any_change_requires": "v5",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    print("=" * 70, flush=True)
    print("Building Frozen v4 (Issue #12 P4 compliant rebuild)", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------
    # 1. Compute candidate family set
    # ------------------------------------------------------------------
    historical = _all_historical_families()
    print(f"Historical families (exclusion set): {len(historical)}", flush=True)

    registry = FamilyRegistry.from_path(REGISTRY_PATH)
    registry_families = set(registry.families.keys())
    candidates = sorted(registry_families - historical)
    print(f"Candidates (registry - historical): {len(candidates)}", flush=True)

    # ------------------------------------------------------------------
    # 2. Load verified MBPP sources
    # ------------------------------------------------------------------
    sources = _load_verified_sources()
    print(f"Loaded {len(sources)} MBPP verified source samples", flush=True)

    candidates_with_source = [fid for fid in candidates if fid in sources]
    print(f"Candidates with MBPP verified source: {len(candidates_with_source)}", flush=True)

    if len(candidates_with_source) < MIN_FAMILIES:
        print(
            f"ERROR: only {len(candidates_with_source)} candidates have source "
            f"data, need >= {MIN_FAMILIES}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # 3. Process each candidate family
    # ------------------------------------------------------------------
    qualified: list[tuple[str, dict, Sample]] = []  # (family_id, samples_by_variant, source)
    all_rejected: list[dict] = []
    boundary_pool: list[tuple[str, Sample]] = []  # (family_id, boundary_sample)

    target = min(TARGET_FAMILIES, len(candidates_with_source))
    print(f"\nTarget: {target} qualified families ({TARGET_BOUNDARY_COUNT} boundary samples)", flush=True)
    print("-" * 70, flush=True)

    for i, fid in enumerate(candidates_with_source, 1):
        if len(qualified) >= target:
            break

        source = sources[fid]
        try:
            ok, samples_by_variant, fam_rejected = _process_family(fid, source)
        except Exception as exc:
            all_rejected.append({
                "sample_id": source.sample_id,
                "family_id": fid,
                "variant_type": None,
                "rejection_reason": f"processing_exception: {type(exc).__name__}: {exc}",
            })
            print(f"[{i}/{len(candidates_with_source)}] {fid}: REJECTED (exception: {exc})", flush=True)
            continue

        all_rejected.extend(fam_rejected)
        if not ok:
            print(f"[{i}/{len(candidates_with_source)}] {fid}: REJECTED ({len(fam_rejected)} issues)", flush=True)
            continue

        qualified.append((fid, samples_by_variant, source))
        if samples_by_variant["boundary"] is not None:
            boundary_pool.append((fid, samples_by_variant["boundary"]))
        print(f"[{i}/{len(candidates_with_source)}] {fid}: QUALIFIED (total={len(qualified)})", flush=True)

    print("-" * 70, flush=True)
    print(f"Qualified families: {len(qualified)}", flush=True)
    print(f"Boundary-eligible families: {len(boundary_pool)}", flush=True)
    print(f"Rejected records: {len(all_rejected)}", flush=True)

    if len(qualified) < MIN_FAMILIES:
        print(
            f"ERROR: only {len(qualified)} families qualified, need >= {MIN_FAMILIES}",
            file=sys.stderr,
        )
        # Still write rejected.jsonl for diagnostics
        V4_DIR.mkdir(parents=True, exist_ok=True)
        _write_rejected_jsonl(V4_DIR / "rejected.jsonl", all_rejected)
        return 1

    # Cap at MAX_FAMILIES
    if len(qualified) > MAX_FAMILIES:
        qualified = qualified[:MAX_FAMILIES]
        # Rebuild boundary_pool to only include kept families
        kept_ids = {fid for fid, _, _ in qualified}
        boundary_pool = [(fid, s) for fid, s in boundary_pool if fid in kept_ids]

    frozen_ids = sorted(fid for fid, _, _ in qualified)

    # ------------------------------------------------------------------
    # 4. Select boundary samples
    # ------------------------------------------------------------------
    # Sort boundary pool by family_id for determinism, take TARGET_BOUNDARY_COUNT.
    boundary_pool.sort(key=lambda x: x[0])
    selected_boundary_fids = set()
    selected_boundary_samples: list[Sample] = []
    for fid, bv in boundary_pool:
        if len(selected_boundary_samples) >= TARGET_BOUNDARY_COUNT:
            break
        selected_boundary_fids.add(fid)
        selected_boundary_samples.append(bv)
    print(f"Selected boundary samples: {len(selected_boundary_samples)}")

    # ------------------------------------------------------------------
    # 5. Assemble final sample list
    # ------------------------------------------------------------------
    final_samples: list[Sample] = []
    for fid, samples_by_variant, source in qualified:
        # Code
        assert samples_by_variant["code"] is not None
        final_samples.append(samples_by_variant["code"])
        # Boundary (only for selected families)
        if fid in selected_boundary_fids:
            assert samples_by_variant["boundary"] is not None
            final_samples.append(samples_by_variant["boundary"])
        # Static repair
        assert samples_by_variant["static_repair"] is not None
        final_samples.append(samples_by_variant["static_repair"])
        # Execution repair
        assert samples_by_variant["execution_repair"] is not None
        final_samples.append(samples_by_variant["execution_repair"])
        # Canary (verified=False, excluded from total_sample_count)
        final_samples.append(_make_canary(source))

    # ------------------------------------------------------------------
    # 6. Compute variant breakdown (canary counted separately)
    # ------------------------------------------------------------------
    non_canary = [
        s for s in final_samples
        if s.variant_type not in ("canary", "canary_repair")
    ]
    canary_count = len(final_samples) - len(non_canary)
    variant_breakdown: dict[str, int] = {}
    for s in final_samples:
        vt = s.variant_type or "unknown"
        variant_breakdown[vt] = variant_breakdown.get(vt, 0) + 1

    total_sample_count = len(non_canary)

    print(f"\nFinal assembly:")
    print(f"  Families:          {len(frozen_ids)}")
    print(f"  Non-canary samples:{total_sample_count}")
    print(f"  Canary samples:    {canary_count}")
    print(f"  Variant breakdown: {variant_breakdown}")
    if total_sample_count > 0:
        for vt in ("code", "boundary", "static_repair", "execution_repair"):
            n = variant_breakdown.get(vt, 0)
            print(f"    {vt}: {n} ({n / total_sample_count:.2%})")

    # ------------------------------------------------------------------
    # 7. Ratio sanity check (advisory; does not fail the build)
    # ------------------------------------------------------------------
    if total_sample_count > 0:
        ratios = {
            "code": variant_breakdown.get("code", 0) / total_sample_count,
            "boundary": variant_breakdown.get("boundary", 0) / total_sample_count,
            "static_repair": variant_breakdown.get("static_repair", 0) / total_sample_count,
            "execution_repair": variant_breakdown.get("execution_repair", 0) / total_sample_count,
        }
        ranges = {
            "code": (0.25, 0.30),
            "boundary": (0.15, 0.20),
            "static_repair": (0.25, 0.30),
            "execution_repair": (0.25, 0.30),
        }
        for vt, (lo, hi) in ranges.items():
            r = ratios[vt]
            in_range = lo <= r <= hi
            print(f"  ratio {vt}: {r:.2%} [{'OK' if in_range else 'OUT OF RANGE'}]")

    # ------------------------------------------------------------------
    # 8. Write output files
    # ------------------------------------------------------------------
    V4_DIR.mkdir(parents=True, exist_ok=True)

    families_path = V4_DIR / "families.json"
    test_raw_path = V4_DIR / "test_raw.jsonl"
    rejected_path = V4_DIR / "rejected.jsonl"
    manifest_path = V4_DIR / "manifest.json"

    _write_families_json(families_path, frozen_ids)
    _write_jsonl(test_raw_path, final_samples)
    _write_rejected_jsonl(rejected_path, all_rejected)
    _write_manifest_json(
        manifest_path,
        frozen_ids=frozen_ids,
        total_sample_count=total_sample_count,
        variant_breakdown=variant_breakdown,
        canary_count=canary_count,
        rejected_count=len(all_rejected),
        test_raw_path=test_raw_path,
        families_path=families_path,
        rejected_path=rejected_path,
    )

    print(f"\nOutput files:")
    print(f"  {families_path} ({len(frozen_ids)} families)")
    print(f"  {test_raw_path} ({len(final_samples)} samples, {total_sample_count} non-canary)")
    print(f"  {manifest_path}")
    print(f"  {rejected_path} ({len(all_rejected)} records)")

    # ------------------------------------------------------------------
    # 9. Update family-registry.json with frozen_v4 tags
    # ------------------------------------------------------------------
    for fid in frozen_ids:
        registry.claim(fid, FROZEN_TAG)
    registry.to_path(REGISTRY_PATH)
    print(f"\nRegistry updated: claimed '{FROZEN_TAG}' for {len(frozen_ids)} families")

    print("\n" + "=" * 70)
    print("Frozen v4 build complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
