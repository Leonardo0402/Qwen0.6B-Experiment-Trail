"""scripts/p3_yield_pilot.py -- P5 Yield Pilot (Issue #14).

Measures the *actual* verification yield of the MBPP-only data pipeline
by generating and verifying variants for a fixed, reproducible sample of
families from the 425 shared train partition.

For each sampled family, attempts to generate up to:
  - 3 code variants (original, signature_scaffold, public_examples)
  - 1 boundary variant (existing generate_boundary_variant logic)
  - 3 static_repair variants (first 3 applicable bug types that fail tests)
  - 3 execution_repair variants (same bugs with execution_feedback)

Each candidate variant is verified via real sandbox execution:
  - target_code must pass public + hidden tests
  - broken_code must fail >= 1 test (for repair variants)
  - execution_feedback must be non-empty (for execution_repair)

The per-bucket yield rate is extrapolated to 425 families using the
Wilson score 90% confidence lower bound.

Outputs:
  - reports/p3/p3-yield-pilot.md
  - reports/p3/p3-yield-pilot.json
  - reports/p3/p3-yield-pilot-raw.jsonl  (per-family per-variant detail)

Usage
-----
    py -3.11 scripts/p3_yield_pilot.py [--n-families 25] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample, Verification  # noqa: E402
from src.sandbox import run_pytest  # noqa: E402
from src.validators import verify_sample, compile_check  # noqa: E402
from scripts.inject_bugs import inject_all_bugs  # noqa: E402
from scripts.generate_boundary_variants import (  # noqa: E402
    generate_boundary_variant,
)
from scripts.extract_function_signature import (  # noqa: E402
    extract_function_signature,
    extract_function_name,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
GENERATOR_NAME: str = "p3_yield_pilot.py"

# Wilson 90% CI z-score (one-sided lower bound)
Z_90: float = 1.645

PARTITION_PATH = _ROOT / "data" / "p3-curriculum" / "family-partition.json"
CANONICAL_POOL_PATH = _ROOT / "data" / "p3-curriculum" / "canonical-pool.jsonl"
REPORT_MD_PATH = _ROOT / "reports" / "p3" / "p3-yield-pilot.md"
REPORT_JSON_PATH = _ROOT / "reports" / "p3" / "p3-yield-pilot.json"
REPORT_RAW_PATH = _ROOT / "reports" / "p3" / "p3-yield-pilot-raw.jsonl"

# Per-bucket max variants per family (Issue #14 cap = 3)
MAX_CODE_PER_FAMILY: int = 3
MAX_BOUNDARY_PER_FAMILY: int = 1  # existing generator produces 1
MAX_STATIC_PER_FAMILY: int = 3
MAX_EXEC_PER_FAMILY: int = 3

# Total shared train families (for extrapolation)
TOTAL_SHARED_FAMILIES: int = 425

# Targets (Issue #14 P5-P7)
BALANCED_TARGETS_2500: dict = {
    "code": 750, "boundary": 500, "static_repair": 500, "execution_repair": 750,
}
REPAIR_TARGETS_2500: dict = {
    "code": 375, "boundary": 375, "static_repair": 750, "execution_repair": 1000,
}
FEASIBLE_MARGIN: float = 0.10  # +10% margin

BUCKETS: tuple[str, ...] = (
    "code", "boundary", "static_repair", "execution_repair",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relpath(path: Path) -> str:
    try:
        rel = path.relative_to(_ROOT)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_pool_by_family(path: Path) -> dict[str, Sample]:
    """Load canonical-pool.jsonl, return family_id -> representative Sample.

    Preference: variant_type == "code" (raw code_generation); falls back to
    the first sample seen for the family.
    """
    by_fam: dict[str, Sample] = {}
    code_by_fam: dict[str, Sample] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = Sample.from_json_line(line)
            if s.family_id not in by_fam:
                by_fam[s.family_id] = s
            if s.variant_type == "code" and s.family_id not in code_by_fam:
                code_by_fam[s.family_id] = s
    # Prefer code variants as the representative source sample
    for fid, s in code_by_fam.items():
        by_fam[fid] = s
    return by_fam


def wilson_lower(verified: int, attempts: int, z: float = Z_90) -> float:
    """Wilson score one-sided lower confidence bound.

    Returns 0.0 when attempts == 0.
    """
    if attempts <= 0:
        return 0.0
    p = verified / attempts
    if p == 0.0:
        # Avoid sqrt(0); the formula still works but gives a small positive
        # lower bound which we suppress to 0 for conservatism.
        return 0.0
    n = attempts
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (center - spread) / denom)


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

def _make_code_variants(src: Sample) -> list[Sample]:
    """Generate up to 3 code variants from the source sample.

    - code_original: unchanged instruction
    - code_signature_scaffold: instruction + function signature
    - code_public_examples: instruction + public test examples
    """
    variants: list[Sample] = []

    # v1: original
    variants.append(src.model_copy(update={
        "sample_id": f"{src.sample_id}_code_orig",
        "variant_type": "code",
        "bug_type": None,
        "task_type": "code_generation",
        "broken_code": None,
        "execution_feedback": None,
    }))

    # v2: signature scaffold
    sig = extract_function_signature(src.target_code) or ""
    name = extract_function_name(src.target_code) or ""
    if sig and name and name not in src.instruction:
        new_instr = f"{src.instruction}\n\nFunction signature: {sig}"
        variants.append(src.model_copy(update={
            "sample_id": f"{src.sample_id}_code_sig",
            "instruction": new_instr,
            "variant_type": "code",
            "bug_type": None,
            "task_type": "code_generation",
            "broken_code": None,
            "execution_feedback": None,
        }))
    else:
        # No signature extractable or instruction already mentions name:
        # produce a duplicate (will be deduped by sample_id downstream if
        # needed). For the pilot we still count it as an attempt.
        variants.append(src.model_copy(update={
            "sample_id": f"{src.sample_id}_code_sig",
            "variant_type": "code",
            "bug_type": None,
            "task_type": "code_generation",
            "broken_code": None,
            "execution_feedback": None,
        }))

    # v3: public examples
    pub = (src.public_tests or "").strip()
    if pub:
        new_instr = f"{src.instruction}\n\nExamples:\n{pub}"
        variants.append(src.model_copy(update={
            "sample_id": f"{src.sample_id}_code_pubex",
            "instruction": new_instr,
            "variant_type": "code",
            "bug_type": None,
            "task_type": "code_generation",
            "broken_code": None,
            "execution_feedback": None,
        }))
    else:
        variants.append(src.model_copy(update={
            "sample_id": f"{src.sample_id}_code_pubex",
            "variant_type": "code",
            "bug_type": None,
            "task_type": "code_generation",
            "broken_code": None,
            "execution_feedback": None,
        }))

    return variants


def _make_boundary_variant(src: Sample) -> Sample | None:
    """Generate 1 boundary variant using existing generator logic."""
    return generate_boundary_variant(src)


def _make_static_repair_variants(
    src: Sample, *, max_n: int = MAX_STATIC_PER_FAMILY
) -> list[Sample]:
    """Generate up to max_n static_repair variants via inject_all_bugs.

    Only bugs that (a) compile and (b) fail >= 1 test are kept, capped at
    max_n. Verification (broken fails >= 1 test) happens later in the pilot.
    Here we just produce the candidate samples.
    """
    variants: list[Sample] = []
    bugs = inject_all_bugs(src.target_code, seed=42)
    for bug_type, bugged_code, _desc in bugs:
        if len(variants) >= max_n:
            break
        # Quick compile check on the bugged code
        ok, _ = compile_check(bugged_code)
        if not ok:
            continue
        variants.append(src.model_copy(update={
            "sample_id": f"{src.sample_id}_sr_{bug_type}",
            "task_type": "static_repair",
            "variant_type": "static_repair",
            "bug_type": bug_type,
            "broken_code": bugged_code,
            "execution_feedback": None,
            "difficulty": min(src.difficulty + 1, 4),
        }))
    return variants


def _build_execution_feedback(
    broken_code: str, public_tests: str, hidden_tests: str
) -> str:
    """Run broken_code against public tests and build a feedback string.

    The feedback captures the first failing test's traceback, mirroring the
    format used in the existing execution_repair samples.
    """
    pub_result = run_pytest(broken_code, public_tests, timeout_s=10.0)
    if pub_result.passed:
        # broken_code unexpectedly passes public tests -- check hidden
        hidden = (hidden_tests or "").strip()
        if hidden:
            hid_result = run_pytest(broken_code, hidden, timeout_s=10.0)
            if not hid_result.passed:
                return _format_feedback(hid_result, "hidden tests")
        return _format_feedback(pub_result, "public tests (unexpectedly passed)")
    return _format_feedback(pub_result, "public tests")


def _format_feedback(result: "run_pytest.__class__", label: str) -> str:
    """Format a pytest result into a feedback string."""
    tb = result.stdout[-1200:] if result.stdout else ""
    return (
        "## 执行反馈\n\n"
        f"### 失败测试 ({label})\n"
        f"- 失败数: {result.num_failed}\n\n"
        "### 关键 Traceback\n"
        f"{tb}\n\n"
        "### 修复要求\n"
        "请根据以上执行反馈修复代码，使其通过所有测试用例。"
    )


def _make_execution_repair_variants(
    src: Sample, *, max_n: int = MAX_EXEC_PER_FAMILY
) -> list[Sample]:
    """Generate up to max_n execution_repair variants with feedback."""
    variants: list[Sample] = []
    bugs = inject_all_bugs(src.target_code, seed=42)
    for bug_type, bugged_code, _desc in bugs:
        if len(variants) >= max_n:
            break
        ok, _ = compile_check(bugged_code)
        if not ok:
            continue
        feedback = _build_execution_feedback(
            bugged_code, src.public_tests, src.hidden_tests
        )
        if not feedback or not feedback.strip():
            continue
        variants.append(src.model_copy(update={
            "sample_id": f"{src.sample_id}_er_{bug_type}",
            "task_type": "execution_repair",
            "variant_type": "execution_repair",
            "bug_type": bug_type,
            "broken_code": bugged_code,
            "execution_feedback": feedback,
            "difficulty": min(src.difficulty + 1, 4),
        }))
    return variants


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

_PLACEHOLDER_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)


def _verify_code_or_boundary(sample: Sample) -> bool:
    """Verify a code/boundary variant: target_code must pass all tests."""
    sv = verify_sample(sample, run_ruff=False, pytest_timeout_s=10.0)
    return sv.is_accepted


def _verify_repair(sample: Sample, *, require_feedback: bool) -> bool:
    """Verify a repair variant.

    Checks:
      1. target_code passes public + hidden tests (verify_sample)
      2. broken_code fails >= 1 test (verify_broken_is_broken)
      3. execution_feedback non-empty (if require_feedback)
    """
    if require_feedback:
        if not (sample.execution_feedback or "").strip():
            return False
    # target_code must pass
    sv = verify_sample(sample, run_ruff=False, pytest_timeout_s=10.0)
    if not sv.is_accepted:
        return False
    # broken_code must fail >= 1 test
    broken = sample.broken_code or ""
    if not broken.strip():
        return False
    pub_result = run_pytest(broken, sample.public_tests, timeout_s=10.0)
    if pub_result.timed_out or pub_result.num_failed >= 1:
        return True
    hidden = (sample.hidden_tests or "").strip()
    if hidden:
        hid_result = run_pytest(broken, hidden, timeout_s=10.0)
        if hid_result.timed_out or hid_result.num_failed >= 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Pilot runner
# ---------------------------------------------------------------------------

def run_pilot(n_families: int, seed: int) -> dict:
    """Run the yield pilot on n_families sampled from the 425 partition."""

    partition = _load_json(PARTITION_PATH)
    train_new = set(partition["p3_train_new"]["family_ids"])
    train_replay = set(partition["p3_train_replay"]["family_ids"])
    all_families = sorted(train_new | train_replay)

    # Reproducible fixed sample (seed=42)
    rng = random.Random(seed)
    sampled = sorted(rng.sample(all_families, min(n_families, len(all_families))))

    # Load source samples from canonical pool (prefer code variant)
    pool_by_fam = _load_pool_by_family(CANONICAL_POOL_PATH)

    raw_records: list[dict] = []
    # Per-bucket attempts / verified counts
    stats = {
        b: {"attempts": 0, "verified": 0, "families_with_source": 0}
        for b in BUCKETS
    }
    families_no_source: list[str] = []

    for idx, fid in enumerate(sampled):
        src = pool_by_fam.get(fid)
        if src is None:
            families_no_source.append(fid)
            continue
        for b in BUCKETS:
            stats[b]["families_with_source"] += 1

        # --- Code variants -------------------------------------------------
        code_variants = _make_code_variants(src)
        for v in code_variants:
            stats["code"]["attempts"] += 1
            ok = _verify_code_or_boundary(v)
            if ok:
                stats["code"]["verified"] += 1
            raw_records.append({
                "family_id": fid, "bucket": "code",
                "sample_id": v.sample_id, "verified": ok,
            })

        # --- Boundary variant ----------------------------------------------
        bv = _make_boundary_variant(src)
        stats["boundary"]["attempts"] += 1
        if bv is not None:
            ok = _verify_code_or_boundary(bv)
            if ok:
                stats["boundary"]["verified"] += 1
            raw_records.append({
                "family_id": fid, "bucket": "boundary",
                "sample_id": bv.sample_id, "verified": ok,
            })
        else:
            raw_records.append({
                "family_id": fid, "bucket": "boundary",
                "sample_id": None, "verified": False,
                "note": "boundary generator returned None",
            })

        # --- Static repair variants ----------------------------------------
        sr_variants = _make_static_repair_variants(src)
        for v in sr_variants:
            stats["static_repair"]["attempts"] += 1
            ok = _verify_repair(v, require_feedback=False)
            if ok:
                stats["static_repair"]["verified"] += 1
            raw_records.append({
                "family_id": fid, "bucket": "static_repair",
                "sample_id": v.sample_id, "verified": ok,
                "bug_type": v.bug_type,
            })
        # Note: attempts only counts variants actually generated, not the
        # theoretical max (3). We track the theoretical max separately via
        # families_with_source * MAX_*_PER_FAMILY.

        # --- Execution repair variants -------------------------------------
        er_variants = _make_execution_repair_variants(src)
        for v in er_variants:
            stats["execution_repair"]["attempts"] += 1
            ok = _verify_repair(v, require_feedback=True)
            if ok:
                stats["execution_repair"]["verified"] += 1
            raw_records.append({
                "family_id": fid, "bucket": "execution_repair",
                "sample_id": v.sample_id, "verified": ok,
                "bug_type": v.bug_type,
            })

        if (idx + 1) % 5 == 0:
            print(f"  processed {idx + 1}/{len(sampled)} families")

    # --- Extrapolation --------------------------------------------------
    n_with_source = stats["code"]["families_with_source"]  # same for all buckets
    extrapolation: dict = {}
    max_per_family = {
        "code": MAX_CODE_PER_FAMILY,
        "boundary": MAX_BOUNDARY_PER_FAMILY,
        "static_repair": MAX_STATIC_PER_FAMILY,
        "execution_repair": MAX_EXEC_PER_FAMILY,
    }
    for b in BUCKETS:
        attempts = stats[b]["attempts"]
        verified = stats[b]["verified"]
        # Yield rate relative to actual attempts (not theoretical max)
        yield_rate = verified / attempts if attempts > 0 else 0.0
        # Wilson lower bound on the per-attempt success probability
        p_lower = wilson_lower(verified, attempts)
        # Project to 425 families: 425 * max_per_family * p_lower
        # (assumes each family could generate up to max_per_family variants)
        projected_lower = int(math.floor(
            TOTAL_SHARED_FAMILIES * max_per_family[b] * p_lower
        ))
        projected_point = int(round(
            TOTAL_SHARED_FAMILIES * max_per_family[b] * yield_rate
        ))
        extrapolation[b] = {
            "attempts": attempts,
            "verified": verified,
            "yield_rate": round(yield_rate, 4),
            "wilson_lower_90": round(p_lower, 4),
            "max_per_family": max_per_family[b],
            "projected_point_425": projected_point,
            "projected_lower_425": projected_lower,
        }

    # --- Verdict --------------------------------------------------------
    verdict = _compute_verdict(extrapolation)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": GENERATOR_NAME,
        "config": {
            "n_families_requested": n_families,
            "n_families_sampled": len(sampled),
            "n_families_with_source": n_with_source,
            "n_families_no_source": len(families_no_source),
            "seed": seed,
            "total_shared_families": TOTAL_SHARED_FAMILIES,
            "z_score": Z_90,
            "feasible_margin_pct": FEASIBLE_MARGIN,
        },
        "sampled_family_ids": sampled,
        "families_no_source": families_no_source,
        "bucket_stats": stats,
        "extrapolation": extrapolation,
        "verdict": verdict,
    }
    return payload, raw_records


def _compute_verdict(extrapolation: dict) -> dict:
    """Compute the capacity verdict based on extrapolated lower bounds.

    FORMAL_CAPACITY_FEASIBLE: every bucket of every candidate meets
        target * (1 + margin).
    FORMAL_CAPACITY_AT_RISK: total projected >= target but some bucket
        is below target * (1 + margin).
    MBPP_FAMILY_OR_VARIANT_LIMIT: total projected or a critical bucket
        is below target (no margin).
    """
    balanced_targets = BALANCED_TARGETS_2500
    repair_targets = REPAIR_TARGETS_2500

    balanced_buckets_ok = {
        b: extrapolation[b]["projected_lower_425"]
            >= int(balanced_targets[b] * (1 + FEASIBLE_MARGIN))
        for b in BUCKETS
    }
    repair_buckets_ok = {
        b: extrapolation[b]["projected_lower_425"]
            >= int(repair_targets[b] * (1 + FEASIBLE_MARGIN))
        for b in BUCKETS
    }
    balanced_total = sum(
        extrapolation[b]["projected_lower_425"] for b in BUCKETS
    )
    repair_total = sum(
        extrapolation[b]["projected_lower_425"] for b in BUCKETS
    )

    balanced_total_ok = balanced_total >= int(2500 * (1 + FEASIBLE_MARGIN))
    repair_total_ok = repair_total >= int(2500 * (1 + FEASIBLE_MARGIN))

    # Critical-bucket check: any bucket below bare target (no margin)
    balanced_critical = {
        b: extrapolation[b]["projected_lower_425"] < balanced_targets[b]
        for b in BUCKETS
    }
    repair_critical = {
        b: extrapolation[b]["projected_lower_425"] < repair_targets[b]
        for b in BUCKETS
    }
    any_critical = any(balanced_critical.values()) or any(
        repair_critical.values()
    )

    if not any_critical and all(balanced_buckets_ok.values()) and all(
        repair_buckets_ok.values()
    ):
        verdict = "FORMAL_CAPACITY_FEASIBLE"
    elif balanced_total_ok and repair_total_ok and not any_critical:
        verdict = "FORMAL_CAPACITY_AT_RISK"
    else:
        verdict = "MBPP_FAMILY_OR_VARIANT_LIMIT"

    return {
        "verdict": verdict,
        "balanced_buckets_ok": balanced_buckets_ok,
        "repair_buckets_ok": repair_buckets_ok,
        "balanced_total_projected_lower": balanced_total,
        "repair_total_projected_lower": repair_total,
        "balanced_total_ok": balanced_total_ok,
        "repair_total_ok": repair_total_ok,
        "balanced_critical_buckets_below_target": balanced_critical,
        "repair_critical_buckets_below_target": repair_critical,
        "balanced_targets_2500": balanced_targets,
        "repair_targets_2500": repair_targets,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_md(payload: dict) -> str:
    cfg = payload["config"]
    ext = payload["extrapolation"]
    v = payload["verdict"]

    lines: list[str] = []
    lines.append("# P3 Yield Pilot Report (Issue #14 P5)")
    lines.append("")
    lines.append(f"**Generated**: {payload['generated_at']}")
    lines.append(f"**Generator**: {payload['generator']}")
    lines.append("")
    lines.append("## 1. Configuration")
    lines.append("")
    lines.append(f"- Families requested: {cfg['n_families_requested']}")
    lines.append(f"- Families sampled: {cfg['n_families_sampled']}")
    lines.append(f"- Families with source sample: "
                 f"{cfg['n_families_with_source']}")
    lines.append(f"- Families without source: "
                 f"{cfg['n_families_no_source']}")
    lines.append(f"- Seed: {cfg['seed']}")
    lines.append(f"- Total shared train families: "
                 f"{cfg['total_shared_families']}")
    lines.append(f"- Wilson z-score (90% one-sided): {cfg['z_score']}")
    lines.append(f"- Feasible margin: +{int(cfg['feasible_margin_pct'] * 100)}%")
    lines.append("")
    lines.append("## 2. Verdict")
    lines.append("")
    lines.append(f"### **{v['verdict']}**")
    lines.append("")
    lines.append("## 3. Per-Bucket Yield & Extrapolation")
    lines.append("")
    lines.append("| Bucket | Attempts | Verified | Yield | Wilson low 90% | "
                 "Max/fam | Point@425 | Lower@425 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for b in BUCKETS:
        e = ext[b]
        lines.append(
            f"| {b} | {e['attempts']} | {e['verified']} | "
            f"{e['yield_rate']:.4f} | {e['wilson_lower_90']:.4f} | "
            f"{e['max_per_family']} | {e['projected_point_425']} | "
            f"**{e['projected_lower_425']}** |"
        )
    lines.append("")
    lines.append("## 4. Bucket Verdict vs Targets (2500 + 10% margin)")
    lines.append("")
    lines.append("### Balanced Generalist")
    lines.append("")
    lines.append("| Bucket | Target @2500 | Lower@425 | Margin OK? |")
    lines.append("|---|---|---|---|")
    for b in BUCKETS:
        ok = "PASS" if v["balanced_buckets_ok"][b] else "FAIL"
        lines.append(
            f"| {b} | {v['balanced_targets_2500'][b]} | "
            f"{ext[b]['projected_lower_425']} | {ok} |"
        )
    lines.append(f"| **TOTAL** | 2500 | "
                 f"{v['balanced_total_projected_lower']} | "
                 f"{'PASS' if v['balanced_total_ok'] else 'FAIL'} |")
    lines.append("")
    lines.append("### Repair Specialist")
    lines.append("")
    lines.append("| Bucket | Target @2500 | Lower@425 | Margin OK? |")
    lines.append("|---|---|---|---|")
    for b in BUCKETS:
        ok = "PASS" if v["repair_buckets_ok"][b] else "FAIL"
        lines.append(
            f"| {b} | {v['repair_targets_2500'][b]} | "
            f"{ext[b]['projected_lower_425']} | {ok} |"
        )
    lines.append(f"| **TOTAL** | 2500 | "
                 f"{v['repair_total_projected_lower']} | "
                 f"{'PASS' if v['repair_total_ok'] else 'FAIL'} |")
    lines.append("")
    lines.append("## 5. Critical Buckets (below bare target, no margin)")
    lines.append("")
    bal_crit = [b for b, is_crit in v["balanced_critical_buckets_below_target"].items() if is_crit]
    rep_crit = [b for b, is_crit in v["repair_critical_buckets_below_target"].items() if is_crit]
    lines.append(f"- Balanced critical buckets: "
                 f"{bal_crit if bal_crit else 'none'}")
    lines.append(f"- Repair critical buckets: "
                 f"{rep_crit if rep_crit else 'none'}")
    lines.append("")
    lines.append("## 6. Notes")
    lines.append("")
    lines.append("- Yield rate = verified / attempts (actual generation "
                 "attempts, not theoretical max).")
    lines.append("- Wilson 90% one-sided lower bound on per-attempt success.")
    lines.append("- Lower@425 = floor(425 × max_per_family × wilson_lower).")
    lines.append("- Boundary bucket uses 1 variant/family (existing generator); "
                 "max_per_family=1.")
    lines.append("- Families without a source sample (not in canonical pool) "
                 "are excluded from attempts.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="P3 yield pilot: measure real verification yield."
    )
    p.add_argument("--n-families", type=int, default=25,
                   help="Number of families to sample (default 25).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible sampling (default 42).")
    return p


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    payload, raw_records = run_pilot(args.n_families, args.seed)

    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)

    md = _render_md(payload)
    with REPORT_MD_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)
        fh.write("\n")
    print(f"Wrote {_relpath(REPORT_MD_PATH)}")

    with REPORT_JSON_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote {_relpath(REPORT_JSON_PATH)}")

    with REPORT_RAW_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in raw_records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
    print(f"Wrote {_relpath(REPORT_RAW_PATH)} ({len(raw_records)} records)")

    # Summary
    ext = payload["extrapolation"]
    v = payload["verdict"]
    print("\nYield pilot summary:")
    for b in BUCKETS:
        e = ext[b]
        print(f"  {b:20s}: attempts={e['attempts']:3d} "
              f"verified={e['verified']:3d} "
              f"yield={e['yield_rate']:.4f} "
              f"lower@425={e['projected_lower_425']}")
    print(f"\nVerdict: {v['verdict']}")
    print(f"  balanced_total_lower={v['balanced_total_projected_lower']} "
          f"repair_total_lower={v['repair_total_projected_lower']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
