"""scripts/p3_allocation_proof.py -- Per-bucket allocation proof (Issue #14).

Generates an auditable allocation proof for each formal candidate, showing:
  - per-bucket available count
  - per-bucket selected count at max-feasible total
  - ratio and tolerance used
  - binding capacity constraint
  - unique sample count and family coverage

The proof computes the LP-feasible maximum total subject to:
  1. Per-bucket availability: selected_v <= available_v
  2. Ratio lower bound: selected_v >= lb_v * T  (lb_v = target_ratio - tol)
  3. Ratio upper bound: selected_v <= ub_v * T  (ub_v = target_ratio + tol)
  4. Sum constraint: sum(selected_v) = T

The max T is bounded by the tightest lower-bound constraint:
  T <= available_v / lb_v  for each bucket v

The binding bucket is the one with the smallest available_v / lb_v.

Usage:
  py -3.11 scripts/p3_allocation_proof.py
  py -3.11 scripts/p3_allocation_proof.py --tolerance 0.02
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

POOL_MANIFEST_PATH = _ROOT / "data" / "p3-formal" / "canonical-pool-manifest.json"
POOL_PATH = _ROOT / "data" / "p3-formal" / "canonical-pool.jsonl"
OUTPUT_PATH = _ROOT / "reports" / "p3" / "p3-allocation-proof.json"

BUCKETS = ("code", "boundary", "static_repair", "execution_repair")

CANDIDATES: dict = {
    "balanced_generalist": {
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
    },
    "repair_specialist": {
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
    },
}

HARD_MIN_TOTAL = 2300
FEASIBLE_MIN_TOTAL = 2530  # 2300 + 10%


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _compute_lp_feasible_max(
    available: dict[str, int],
    target_ratios: dict[str, float],
    tolerance: float,
) -> dict:
    """Compute LP-feasible max total with ratio constraints.

    Returns dict with:
      - max_total: int
      - binding_bucket: str (bucket that constrains the total)
      - per_bucket_selected: dict[str, int]
      - per_bucket_ratio: dict[str, float]
      - per_bucket_ratio_ok: dict[str, bool]
      - feasibility_verified: bool
    """
    lb = {v: target_ratios[v] - tolerance for v in BUCKETS}
    ub = {v: target_ratios[v] + tolerance for v in BUCKETS}

    # Upper bound on T from lower-bound ratio constraints:
    # T <= available_v / lb_v for each v
    t_upper_bounds = {}
    for v in BUCKETS:
        if lb[v] > 0:
            t_upper_bounds[v] = int(math.floor(available[v] / lb[v]))
        else:
            t_upper_bounds[v] = float("inf")

    # The binding bucket has the smallest upper bound.
    binding_bucket = min(t_upper_bounds, key=lambda v: t_upper_bounds[v])
    max_total = t_upper_bounds[binding_bucket]

    # Also cap by sum of available (trivial bound).
    sum_available = sum(available.values())
    if max_total > sum_available:
        max_total = sum_available

    # Verify feasibility at max_total:
    # For each bucket, selected_v must be in [lb_v * T, min(ub_v * T, avail_v)]
    # and sum of selected must equal T.
    per_bucket_selected = {}
    per_bucket_ratio = {}
    per_bucket_ratio_ok = {}

    # Greedy allocation: start with lower bounds, then fill from buckets
    # with the most slack (available - lb*T) until we reach T.
    selected = {v: int(math.ceil(lb[v] * max_total)) for v in BUCKETS}
    # Cap at available
    for v in BUCKETS:
        selected[v] = min(selected[v], available[v])

    current_sum = sum(selected.values())
    remaining = max_total - current_sum

    if remaining > 0:
        # Fill remaining from buckets with slack (avail - current selected)
        # prioritizing buckets that are below their upper bound.
        slack = {}
        for v in BUCKETS:
            upper = min(int(math.floor(ub[v] * max_total)), available[v])
            slack[v] = upper - selected[v]
        # Sort by slack descending
        for v in sorted(slack, key=slack.get, reverse=True):
            if remaining <= 0:
                break
            take = min(remaining, slack[v])
            selected[v] += take
            remaining -= take

    if remaining < 0:
        # Over-allocated (shouldn't happen if math is correct), trim from
        # buckets with most excess above lower bound.
        for v in BUCKETS:
            excess = selected[v] - int(math.ceil(lb[v] * max_total))
            trim = min(-remaining, excess)
            selected[v] -= trim
            remaining += trim

    # Verify ratios
    for v in BUCKETS:
        ratio = selected[v] / max_total if max_total > 0 else 0.0
        per_bucket_selected[v] = selected[v]
        per_bucket_ratio[v] = round(ratio, 6)
        per_bucket_ratio_ok[v] = (lb[v] - 1e-9 <= ratio <= ub[v] + 1e-9)

    feasibility_verified = (
        sum(per_bucket_selected.values()) == max_total
        and all(per_bucket_ratio_ok.values())
    )

    return {
        "max_total": max_total,
        "binding_bucket": binding_bucket,
        "binding_constraint": f"available_{binding_bucket} / lb_{binding_bucket} = {available[binding_bucket]} / {lb[binding_bucket]:.4f} = {max_total}",
        "per_bucket_available": dict(available),
        "per_bucket_selected": per_bucket_selected,
        "per_bucket_ratio": per_bucket_ratio,
        "per_bucket_ratio_ok": per_bucket_ratio_ok,
        "per_bucket_lower_bound": {v: round(lb[v], 6) for v in BUCKETS},
        "per_bucket_upper_bound": {v: round(ub[v], 6) for v in BUCKETS},
        "feasibility_verified": feasibility_verified,
        "tolerance": tolerance,
    }


def _count_unique_and_families(samples: list[dict]) -> dict:
    """Count unique samples by sample_id and unique families."""
    sample_ids = set()
    families = set()
    for s in samples:
        sid = s.get("sample_id")
        if sid:
            sample_ids.add(sid)
        fid = s.get("family_id")
        if fid:
            families.add(fid)
    return {
        "unique_sample_ids": len(sample_ids),
        "unique_families": len(families),
    }


def main():
    tolerance = 0.03  # ±3pp default
    if "--tolerance" in sys.argv:
        idx = sys.argv.index("--tolerance")
        tolerance = float(sys.argv[idx + 1])

    if not POOL_MANIFEST_PATH.exists():
        print(f"ERROR: pool manifest not found: {POOL_MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    with POOL_MANIFEST_PATH.open(encoding="utf-8") as fh:
        pool_manifest = json.load(fh)

    pool_buckets = pool_manifest.get("bucket_counts", {})
    pool_sha = pool_manifest.get("pool_sha256", "")
    total_samples = pool_manifest.get("total_samples", 0)
    total_families = pool_manifest.get("total_families", 0)

    # Load pool samples for unique/family counts
    pool_samples = _read_jsonl(POOL_PATH) if POOL_PATH.exists() else []
    pool_stats = _count_unique_and_families(pool_samples)

    # Per-bucket family coverage (unique families contributing to each bucket)
    bucket_families = {v: set() for v in BUCKETS}
    for s in pool_samples:
        vt = s.get("variant_type", "")
        fid = s.get("family_id", "")
        if vt in bucket_families and fid:
            bucket_families[vt].add(fid)

    proof = {
        "schema_version": 1,
        "generated_at": pool_manifest.get("generated_at", ""),
        "pool_sha256": pool_sha,
        "pool_total_samples": total_samples,
        "pool_total_families": total_families,
        "pool_unique_sample_ids": pool_stats["unique_sample_ids"],
        "pool_unique_families": pool_stats["unique_families"],
        "tolerance": tolerance,
        "pool_bucket_counts": dict(pool_buckets),
        "pool_bucket_family_coverage": {
            v: len(bucket_families[v]) for v in BUCKETS
        },
        "hard_min_total": HARD_MIN_TOTAL,
        "feasible_min_total": FEASIBLE_MIN_TOTAL,
        "candidates": {},
        "overall_verdict": "",
    }

    all_feasible = True
    any_limit = False

    for name, cfg in CANDIDATES.items():
        available = {v: pool_buckets.get(v, 0) for v in BUCKETS}
        result = _compute_lp_feasible_max(
            available, cfg["target_ratios"], tolerance
        )
        result["target_ratios"] = dict(cfg["target_ratios"])
        result["bucket_targets"] = dict(cfg["bucket_targets"])

        if result["max_total"] < HARD_MIN_TOTAL:
            result["verdict"] = "MBPP_FAMILY_OR_VARIANT_LIMIT"
            any_limit = True
            all_feasible = False
        elif result["max_total"] < FEASIBLE_MIN_TOTAL:
            result["verdict"] = "FORMAL_CAPACITY_AT_RISK"
            all_feasible = False
        else:
            result["verdict"] = "FORMAL_CAPACITY_FEASIBLE"

        proof["candidates"][name] = result

    if any_limit:
        proof["overall_verdict"] = "MBPP_FAMILY_OR_VARIANT_LIMIT"
    elif not all_feasible:
        proof["overall_verdict"] = "FORMAL_CAPACITY_AT_RISK"
    else:
        proof["overall_verdict"] = "FORMAL_CAPACITY_FEASIBLE"

    # Write proof
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(proof, fh, indent=2, ensure_ascii=False)

    # Print summary
    print("=" * 70)
    print("P3 Allocation Proof (Issue #14)")
    print("=" * 70)
    print(f"Pool SHA256: {pool_sha[:16]}...")
    print(f"Pool total: {total_samples} samples, {total_families} families")
    print(f"Pool unique sample_ids: {pool_stats['unique_sample_ids']}")
    print(f"Tolerance: ±{tolerance*100:.0f}pp")
    print()
    print("Pool bucket counts:")
    for v in BUCKETS:
        print(f"  {v:20s}: {pool_buckets.get(v, 0):4d}  (families: {len(bucket_families[v])})")
    print()
    print(f"{'Candidate':<25s} {'Max':>6s} {'Binding':<12s} {'Verdict':<30s}")
    print("-" * 80)
    for name, result in proof["candidates"].items():
        print(f"{name:<25s} {result['max_total']:>6d} {result['binding_bucket']:<12s} {result['verdict']:<30s}")
    print()
    print(f"Overall verdict: {proof['overall_verdict']}")
    print()
    print("Per-candidate detail:")
    for name, result in proof["candidates"].items():
        print(f"\n  {name}:")
        print(f"    binding: {result['binding_constraint']}")
        print(f"    feasibility_verified: {result['feasibility_verified']}")
        print(f"    {'bucket':<20s} {'avail':>6s} {'select':>6s} {'ratio':>8s} {'lb':>6s} {'ub':>6s} {'ok':>4s}")
        for v in BUCKETS:
            ok = "✓" if result["per_bucket_ratio_ok"][v] else "✗"
            print(f"    {v:<20s} {result['per_bucket_available'][v]:>6d} {result['per_bucket_selected'][v]:>6d} {result['per_bucket_ratio'][v]:>8.4f} {result['per_bucket_lower_bound'][v]:>6.2f} {result['per_bucket_upper_bound'][v]:>6.2f} {ok:>4s}")

    print(f"\nProof written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
