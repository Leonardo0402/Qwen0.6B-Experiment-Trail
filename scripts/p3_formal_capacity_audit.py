"""scripts/p3_formal_capacity_audit.py -- P5 Static Capacity Audit (Issue #14).

Computes the *theoretical* capacity of the MBPP-only data pipeline to
feed both P3 candidates (balanced_generalist + repair_specialist) at
>= 2300 verified samples each.

Static (no sandbox execution). Reads:
  - data/family-registry.json
  - data/p3-curriculum/family-partition.json
  - data/p3-curriculum/canonical-pool.jsonl  (current state, for reference)
  - data/frozen-eval/v4/families.json
  - data/p3-curriculum/validation-v2/manifest.json

Writes:
  - reports/p3/p3-formal-capacity-audit.md
  - reports/p3/p3-formal-capacity-audit.json

Theoretical yield per family (Issue #14 P5 brief):
  code             : 3 variants (original, signature_scaffold, public_examples)
  boundary         : 3 variants (distinct boundary test subsets)
  static_repair    : 3 variants (first 3 applicable bug types that fail tests)
  execution_repair : 3 variants (same bugs with execution_feedback)
  Total            : 12 variants / family

With 425 shared train families the theoretical max is 425 * 12 = 5100.
Per-bucket cap (3) gives 425 * 3 = 1275 max per bucket.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.family_registry import FamilyRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (Issue #14 P5 brief)
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
GENERATOR_NAME: str = "p3_formal_capacity_audit.py"

# Theoretical variants per family per bucket (cap = 3 per Issue #14)
VARIANTS_PER_FAMILY_PER_BUCKET: int = 3
BUCKETS: tuple[str, ...] = (
    "code", "boundary", "static_repair", "execution_repair",
)
TOTAL_VARIANTS_PER_FAMILY: int = VARIANTS_PER_FAMILY_PER_BUCKET * len(BUCKETS)  # 12

# Issue #14 P5-P7 capacity targets
TARGET_TOTAL_2300: int = 2300
TARGET_TOTAL_2500: int = 2500
TARGET_MARGIN_PCT: float = 0.10  # +10% margin for FEASIBLE verdict

# Per-bucket targets at 2300 and 2500 total (per brief)
BALANCED_BUCKET_TARGETS_2300: dict = {
    "code": 690, "boundary": 460, "static_repair": 460, "execution_repair": 690,
}
BALANCED_BUCKET_TARGETS_2500: dict = {
    "code": 750, "boundary": 500, "static_repair": 500, "execution_repair": 750,
}
REPAIR_BUCKET_TARGETS_2300: dict = {
    "code": 345, "boundary": 345, "static_repair": 690, "execution_repair": 920,
}
REPAIR_BUCKET_TARGETS_2500: dict = {
    "code": 375, "boundary": 375, "static_repair": 750, "execution_repair": 1000,
}

# Paths
REGISTRY_PATH = _ROOT / "data" / "family-registry.json"
PARTITION_PATH = _ROOT / "data" / "p3-curriculum" / "family-partition.json"
CANONICAL_POOL_PATH = _ROOT / "data" / "p3-curriculum" / "canonical-pool.jsonl"
FROZEN_V4_FAMILIES_PATH = _ROOT / "data" / "frozen-eval" / "v4" / "families.json"
VALIDATION_V2_MANIFEST_PATH = (
    _ROOT / "data" / "p3-curriculum" / "validation-v2" / "manifest.json"
)
REPORT_MD_PATH = _ROOT / "reports" / "p3" / "p3-formal-capacity-audit.md"
REPORT_JSON_PATH = _ROOT / "reports" / "p3" / "p3-formal-capacity-audit.json"


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


def _count_jsonl(path: Path) -> tuple[int, dict[str, int], dict[str, int]]:
    """Return (total, variant_distribution, bug_type_distribution)."""
    total = 0
    variant_dist: dict[str, int] = {}
    bug_dist: dict[str, int] = {}
    if not path.exists():
        return 0, variant_dist, bug_dist
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            data = json.loads(line)
            vt = data.get("variant_type") or "none"
            bt = data.get("bug_type") or "none"
            variant_dist[vt] = variant_dist.get(vt, 0) + 1
            bug_dist[bt] = bug_dist.get(bt, 0) + 1
    return total, variant_dist, bug_dist


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def run_audit() -> dict:
    """Run the static capacity audit and return the report payload."""

    # --- Load registry --------------------------------------------------
    registry = FamilyRegistry.from_path(REGISTRY_PATH)
    total_families = len(registry.families)
    total_quarantined = registry._count_quarantined()
    total_new_available = registry._count_new_available()
    total_p2_used = registry._count_p2_used()

    # --- Load partition -------------------------------------------------
    partition = _load_json(PARTITION_PATH)
    p3_train_new_fids = set(partition["p3_train_new"]["family_ids"])
    p3_train_replay_fids = set(partition["p3_train_replay"]["family_ids"])
    p3_validation_fids = set(partition["p3_validation"]["family_ids"])
    shared_train_fids = p3_train_new_fids | p3_train_replay_fids

    n_train_new = len(p3_train_new_fids)
    n_train_replay = len(p3_train_replay_fids)
    n_shared = len(shared_train_fids)
    n_validation = len(p3_validation_fids)

    # --- Frozen v4 ------------------------------------------------------
    frozen_v4_fids: set[str] = set()
    if FROZEN_V4_FAMILIES_PATH.exists():
        frozen_v4_data = _load_json(FROZEN_V4_FAMILIES_PATH)
        frozen_v4_fids = set(frozen_v4_data.get("families", []))
    n_frozen_v4 = len(frozen_v4_fids)

    # --- Validation v2 --------------------------------------------------
    # validation_v2 manifest stores family count under
    # families.validation_family_count (45 families x 4 categories = 180 samples)
    n_validation_v2 = 0
    if VALIDATION_V2_MANIFEST_PATH.exists():
        val_v2_manifest = _load_json(VALIDATION_V2_MANIFEST_PATH)
        fams_block = val_v2_manifest.get("families", {})
        n_validation_v2 = fams_block.get("validation_family_count", 0)

    # --- Canonical pool (current state) ---------------------------------
    pool_total, pool_variant_dist, pool_bug_dist = _count_jsonl(
        CANONICAL_POOL_PATH
    )

    # --- Theoretical yield ----------------------------------------------
    theoretical_per_bucket = n_shared * VARIANTS_PER_FAMILY_PER_BUCKET
    theoretical_total = n_shared * TOTAL_VARIANTS_PER_FAMILY

    # --- Build payload --------------------------------------------------
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": GENERATOR_NAME,
        "family_counts": {
            "total_in_registry": total_families,
            "shared_train": n_shared,
            "p3_train_new": n_train_new,
            "p3_train_replay": n_train_replay,
            "p3_validation": n_validation,
            "validation_v2": n_validation_v2,
            "frozen_v4": n_frozen_v4,
            "quarantined": total_quarantined,
            "remaining_new_available": total_new_available,
            "p2_used": total_p2_used,
        },
        "theoretical_yield": {
            "variants_per_family_per_bucket": VARIANTS_PER_FAMILY_PER_BUCKET,
            "buckets": list(BUCKETS),
            "total_variants_per_family": TOTAL_VARIANTS_PER_FAMILY,
            "shared_train_families": n_shared,
            "max_per_bucket": theoretical_per_bucket,
            "max_total": theoretical_total,
        },
        "current_canonical_pool": {
            "path": _relpath(CANONICAL_POOL_PATH),
            "total_samples": pool_total,
            "variant_distribution": pool_variant_dist,
            "bug_type_distribution": pool_bug_dist,
        },
        "capacity_targets": {
            "balanced_2300": BALANCED_BUCKET_TARGETS_2300,
            "balanced_2500": BALANCED_BUCKET_TARGETS_2500,
            "repair_2300": REPAIR_BUCKET_TARGETS_2300,
            "repair_2500": REPAIR_BUCKET_TARGETS_2500,
            "margin_pct_for_feasible": TARGET_MARGIN_PCT,
        },
        "theoretical_feasibility": {
            "balanced_2300_total": theoretical_total >= TARGET_TOTAL_2300,
            "balanced_2500_total": theoretical_total >= TARGET_TOTAL_2500,
            "per_bucket_meets_balanced_2500": {
                b: theoretical_per_bucket >= BALANCED_BUCKET_TARGETS_2500[b]
                for b in BUCKETS
            },
            "per_bucket_meets_repair_2500": {
                b: theoretical_per_bucket >= REPAIR_BUCKET_TARGETS_2500[b]
                for b in BUCKETS
            },
        },
    }
    return payload


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_md(payload: dict) -> str:
    fc = payload["family_counts"]
    ty = payload["theoretical_yield"]
    cp = payload["current_canonical_pool"]
    ct = payload["capacity_targets"]
    tf = payload["theoretical_feasibility"]

    lines: list[str] = []
    lines.append("# P3 Formal Capacity Audit (Issue #14 P5)")
    lines.append("")
    lines.append(f"**Generated**: {payload['generated_at']}")
    lines.append(f"**Generator**: {payload['generator']}")
    lines.append("")
    lines.append("## 1. Family Inventory")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("|---|---|")
    lines.append(f"| Total in registry | {fc['total_in_registry']} |")
    lines.append(f"| Shared train (new + replay) | {fc['shared_train']} |")
    lines.append(f"| &nbsp;&nbsp;p3_train_new | {fc['p3_train_new']} |")
    lines.append(f"| &nbsp;&nbsp;p3_train_replay | {fc['p3_train_replay']} |")
    lines.append(f"| p3_validation | {fc['p3_validation']} |")
    lines.append(f"| validation_v2 | {fc['validation_v2']} |")
    lines.append(f"| frozen_v4 | {fc['frozen_v4']} |")
    lines.append(f"| Quarantined | {fc['quarantined']} |")
    lines.append(f"| Remaining new available | {fc['remaining_new_available']} |")
    lines.append(f"| P2 used (any tag) | {fc['p2_used']} |")
    lines.append("")
    lines.append("## 2. Theoretical Yield (no verification)")
    lines.append("")
    lines.append(f"- Variants per family per bucket: **{ty['variants_per_family_per_bucket']}**")
    lines.append(f"- Buckets: {', '.join(ty['buckets'])}")
    lines.append(f"- Total variants per family: **{ty['total_variants_per_family']}**")
    lines.append(f"- Shared train families: **{ty['shared_train_families']}**")
    lines.append(f"- Max per bucket: **{ty['max_per_bucket']}** "
                 f"(families × {ty['variants_per_family_per_bucket']})")
    lines.append(f"- Max total: **{ty['max_total']}** "
                 f"(families × {ty['total_variants_per_family']})")
    lines.append("")
    lines.append("## 3. Current Canonical Pool (reference)")
    lines.append("")
    lines.append(f"- Path: `{cp['path']}`")
    lines.append(f"- Total samples: **{cp['total_samples']}**")
    lines.append(f"- Variant distribution: {cp['variant_distribution']}")
    lines.append(f"- Bug type distribution: {cp['bug_type_distribution']}")
    lines.append("")
    lines.append("## 4. Capacity Targets (Issue #14 P5-P7)")
    lines.append("")
    lines.append("### Balanced Generalist")
    lines.append("")
    lines.append("| Bucket | @2300 | @2500 | Theoretical max |")
    lines.append("|---|---|---|---|")
    for b in BUCKETS:
        lines.append(
            f"| {b} | {ct['balanced_2300'][b]} | "
            f"{ct['balanced_2500'][b]} | {ty['max_per_bucket']} |"
        )
    lines.append("")
    lines.append("### Repair Specialist")
    lines.append("")
    lines.append("| Bucket | @2300 | @2500 | Theoretical max |")
    lines.append("|---|---|---|---|")
    for b in BUCKETS:
        lines.append(
            f"| {b} | {ct['repair_2300'][b]} | "
            f"{ct['repair_2500'][b]} | {ty['max_per_bucket']} |"
        )
    lines.append("")
    lines.append("## 5. Theoretical Feasibility (ignoring verification yield)")
    lines.append("")
    lines.append(f"- Total >= 2300 (balanced/repair): "
                 f"**{tf['balanced_2300_total']}**")
    lines.append(f"- Total >= 2500 (balanced/repair): "
                 f"**{tf['balanced_2500_total']}**")
    lines.append("")
    lines.append("Per-bucket vs Balanced @2500 targets:")
    lines.append("")
    lines.append("| Bucket | Target @2500 | Theoretical max | Sufficient? |")
    lines.append("|---|---|---|---|")
    for b in BUCKETS:
        ok = "PASS" if tf["per_bucket_meets_balanced_2500"][b] else "FAIL"
        lines.append(
            f"| {b} | {ct['balanced_2500'][b]} | "
            f"{ty['max_per_bucket']} | {ok} |"
        )
    lines.append("")
    lines.append("Per-bucket vs Repair @2500 targets:")
    lines.append("")
    lines.append("| Bucket | Target @2500 | Theoretical max | Sufficient? |")
    lines.append("|---|---|---|---|")
    for b in BUCKETS:
        ok = "PASS" if tf["per_bucket_meets_repair_2500"][b] else "FAIL"
        lines.append(
            f"| {b} | {ct['repair_2500'][b]} | "
            f"{ty['max_per_bucket']} | {ok} |"
        )
    lines.append("")
    lines.append("## 6. Caveats")
    lines.append("")
    lines.append("- Theoretical yield assumes 100% verification pass rate.")
    lines.append("- Actual yield is measured by `scripts/p3_yield_pilot.py`.")
    lines.append("- Boundary bucket currently produces 1 variant/family via "
                 "`generate_boundary_variants.py`; reaching 3/family requires "
                 "additional boundary-test strategies.")
    lines.append("- Repair Specialist execution_repair target (920-1000) "
                 "requires 2.16-2.35 verified variants per family on average "
                 f"(cap = {VARIANTS_PER_FAMILY_PER_BUCKET}/family).")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    payload = run_audit()

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

    # Summary
    fc = payload["family_counts"]
    ty = payload["theoretical_yield"]
    print("\nStatic capacity audit summary:")
    print(f"  shared_train_families: {fc['shared_train']} "
          f"(new={fc['p3_train_new']} replay={fc['p3_train_replay']})")
    print(f"  quarantined: {fc['quarantined']}  remaining_new: "
          f"{fc['remaining_new_available']}")
    print(f"  theoretical max per bucket: {ty['max_per_bucket']}")
    print(f"  theoretical max total:      {ty['max_total']}")
    print(f"  current canonical pool:     {payload['current_canonical_pool']['total_samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
