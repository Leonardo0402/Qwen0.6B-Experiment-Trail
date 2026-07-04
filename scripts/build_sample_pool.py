"""scripts/build_sample_pool.py -- Canonical verified sample pool builder.

Builds the P3 canonical sample pool by:

  1. Loading partition family_ids from data/p3-curriculum/family-partition.json
     (p3_train_new.family_ids = 219, p3_train_replay.family_ids = 206).
  2. Loading P2 replay samples from the 3 P2 stage train.jsonl files,
     filtered to the 206 replay families.
  3. Loading P3 new train samples from data/external/mbpp/verified/test.jsonl,
     filtered to the 219 train_new families. These samples are normalised
     to variant_type="code", bug_type=None, source_split="test".
  4. Normalising variant_type + bug_type on ALL samples.
  5. Deduplicating by sample_id (first occurrence wins; logs duplicates).
  6. Applying per-family cap (default=7).
  7. Writing data/p3-curriculum/canonical-pool.jsonl (sorted by sample_id).
  8. Writing data/p3-curriculum/canonical-pool-manifest.json (statistics).
  9. Running hard gates (aborts with exit 1 if any fail).

Usage
-----
    python scripts/build_sample_pool.py
    python scripts/build_sample_pool.py --cap 7 \\
        --partition data/p3-curriculum/family-partition.json \\
        --registry data/family-registry.json \\
        --output-pool data/p3-curriculum/canonical-pool.jsonl \\
        --output-manifest data/p3-curriculum/canonical-pool-manifest.json

Exit codes
----------
    0   success
    1   invariant violation (hard gate failed) or I/O error
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root import guard (so the script works from any cwd)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sample_pool import ALLOWED_VARIANT_TYPES, SamplePool  # noqa: E402
from src.schemas import Sample  # noqa: E402
from src.family_registry import FamilyRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
GENERATOR_NAME: str = "build_sample_pool.py"

DEFAULT_CAP: int = 7
POOL_MIN_TOTAL: int = 400

DEFAULT_PARTITION = _ROOT / "data" / "p3-curriculum" / "family-partition.json"
DEFAULT_REGISTRY = _ROOT / "data" / "family-registry.json"
DEFAULT_OUTPUT_POOL = _ROOT / "data" / "p3-curriculum" / "canonical-pool.jsonl"
DEFAULT_OUTPUT_MANIFEST = _ROOT / "data" / "p3-curriculum" / "canonical-pool-manifest.json"

# P2 replay source files (Amendment A6 -- original stage outputs only,
# NOT stage3-repair-v3 which is a remix).
P2_STAGE1_CODE = _ROOT / "data" / "p2-curriculum" / "stage1-code" / "train.jsonl"
P2_STAGE2_BOUNDARY = _ROOT / "data" / "p2-curriculum" / "stage2-boundary" / "train.jsonl"
P2_STAGE3_REPAIR = _ROOT / "data" / "p2-curriculum" / "stage3-repair" / "train.jsonl"
P3_VERIFIED_TEST = _ROOT / "data" / "external" / "mbpp" / "verified" / "test.jsonl"

TAG_QUARANTINE: str = "quarantine"


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def _load_jsonl_samples(path: Path) -> list[Sample]:
    """Stream a JSONL file into a list of Sample objects."""
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(Sample.from_json_line(line))
    return samples


def _load_p2_stage(path: Path, allowed_families: set[str]) -> tuple[int, int, list[Sample]]:
    """Load a P2 stage train.jsonl and filter to ``allowed_families``.

    Returns ``(loaded, after_filter, samples)``.
    """
    samples = _load_jsonl_samples(path)
    loaded = len(samples)
    kept = [s for s in samples if s.family_id in allowed_families]
    return loaded, len(kept), kept


def _relpath(path: Path) -> str:
    """Return ``path`` relative to ``_ROOT`` if possible, else absolute string.

    Uses forward slashes for cross-platform consistency in the manifest.
    Python 3.8 doesn't have ``Path.is_relative_to`` so we use ``try/except``.
    """
    try:
        rel = path.relative_to(_ROOT)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def _load_p3_verified(path: Path, allowed_families: set[str]) -> tuple[int, int, list[Sample]]:
    """Load the P3 verified MBPP test split and filter to ``allowed_families``.

    Per the brief, P3 new train samples are normalised to:
      - variant_type = "code"
      - bug_type = None
      - source_split = "test"

    Returns ``(loaded, after_filter, samples)``.
    """
    samples = _load_jsonl_samples(path)
    loaded = len(samples)
    kept: list[Sample] = []
    for s in samples:
        if s.family_id not in allowed_families:
            continue
        # Force-override variant_type / bug_type / source_split per the brief.
        new_vt = "code"
        new_bt = None
        new_split = "test"
        if (s.variant_type != new_vt
                or s.bug_type != new_bt
                or s.source_split != new_split):
            kept.append(
                s.model_copy(update={
                    "variant_type": new_vt,
                    "bug_type": new_bt,
                    "source_split": new_split,
                })
            )
        else:
            kept.append(s)
    return loaded, len(kept), kept


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------

def _run_hard_gates(
    pool: SamplePool,
    *,
    cap: int,
    partition_family_ids: set[str],
    quarantine_family_ids: set[str],
) -> list[str]:
    """Run all hard gates. Returns list of error messages (empty = pass).

    Hard gates (binding):
      1. Pool total >= 400
      2. All variant_type values in {code, boundary, static_repair, execution_repair}
      3. No duplicate sample_ids
      4. No family has > cap samples
      5. All families in pool are in the 425 partition families
      6. No quarantined family in pool
    """
    errors: list[str] = []

    # Gate 1: minimum pool size
    if len(pool) < POOL_MIN_TOTAL:
        errors.append(
            f"gate 1: pool total {len(pool)} < {POOL_MIN_TOTAL}"
        )

    # Gate 2: variant_type allowed set + Gate 3: no duplicate sample_ids
    seen: set[str] = set()
    bad_variants: dict[str, int] = {}
    duplicates: list[str] = []
    for s in pool:
        vt = s.variant_type
        if vt not in ALLOWED_VARIANT_TYPES:
            bad_variants[vt or "None"] = bad_variants.get(vt or "None", 0) + 1
        if s.sample_id in seen:
            duplicates.append(s.sample_id)
        seen.add(s.sample_id)
    if bad_variants:
        errors.append(
            f"gate 2: variant_type values outside allowed set: {bad_variants}"
        )
    if duplicates:
        errors.append(
            f"gate 3: {len(duplicates)} duplicate sample_ids (first 5: "
            f"{duplicates[:5]})"
        )

    # Gate 4: per-family cap
    family_counts: dict[str, int] = {}
    for s in pool:
        family_counts[s.family_id] = family_counts.get(s.family_id, 0) + 1
    over_cap = {fid: c for fid, c in family_counts.items() if c > cap}
    if over_cap:
        errors.append(
            f"gate 4: {len(over_cap)} families exceed cap={cap} "
            f"(first 5: {dict(list(over_cap.items())[:5])})"
        )

    # Gate 5: all families in pool are in the 425 partition families
    pool_families = set(family_counts.keys())
    outside = pool_families - partition_family_ids
    if outside:
        errors.append(
            f"gate 5: {len(outside)} families in pool not in partition "
            f"(first 5: {sorted(outside)[:5]})"
        )

    # Gate 6: no quarantined family in pool
    quarantined_in_pool = pool_families & quarantine_family_ids
    if quarantined_in_pool:
        errors.append(
            f"gate 6: {len(quarantined_in_pool)} quarantined families in pool: "
            f"{sorted(quarantined_in_pool)[:5]}"
        )

    return errors


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _build_manifest(
    *,
    sources: dict,
    total_loaded: int,
    total_after_family_filter: int,
    duplicates_removed: int,
    total_after_dedup: int,
    families_capped: int,
    samples_dropped_by_cap: int,
    total_after_cap: int,
    cap: int,
    partition_family_count: int,
    pool_stats: dict,
    pool_sha256: str,
) -> dict:
    """Build the manifest payload per task-10-brief.md schema."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": GENERATOR_NAME,
        "sources": sources,
        "total_loaded": total_loaded,
        "total_after_family_filter": total_after_family_filter,
        "total_after_dedup": total_after_dedup,
        "duplicates_removed": duplicates_removed,
        "total_after_cap": total_after_cap,
        "families_capped": families_capped,
        "samples_dropped_by_cap": samples_dropped_by_cap,
        "per_family_cap": cap,
        "family_count": pool_stats["family_count"],
        "partition_family_count": partition_family_count,
        "variant_distribution": pool_stats["variant_distribution"],
        "bug_type_distribution": pool_stats["bug_type_distribution"],
        "family_distribution": pool_stats["family_distribution"],
        "pool_sha256": pool_sha256,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build the canonical verified sample pool: load P2 replay + P3 "
            "new train, dedup, normalise, per-family cap, write jsonl + manifest."
        ),
    )
    p.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_CAP,
        help=f"Per-family sample cap (default: {DEFAULT_CAP}).",
    )
    p.add_argument(
        "--partition",
        default=str(DEFAULT_PARTITION),
        help="Path to data/p3-curriculum/family-partition.json (input).",
    )
    p.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help="Path to data/family-registry.json (read-only, for quarantine check).",
    )
    p.add_argument(
        "--output-pool",
        default=str(DEFAULT_OUTPUT_POOL),
        help="Output path for canonical-pool.jsonl.",
    )
    p.add_argument(
        "--output-manifest",
        default=str(DEFAULT_OUTPUT_MANIFEST),
        help="Output path for canonical-pool-manifest.json.",
    )
    return p


def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on error."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    cap: int = args.cap
    partition_path = Path(args.partition)
    registry_path = Path(args.registry)
    output_pool_path = Path(args.output_pool)
    output_manifest_path = Path(args.output_manifest)

    # ------------------------------------------------------------------
    # Load partition family_ids
    # ------------------------------------------------------------------
    try:
        with partition_path.open(encoding="utf-8") as fh:
            partition = json.load(fh)
    except Exception as exc:
        print(f"ERROR: cannot load partition {partition_path}: {exc}", file=sys.stderr)
        return 1

    p3_train_new_fids = set(partition["p3_train_new"]["family_ids"])
    p3_train_replay_fids = set(partition["p3_train_replay"]["family_ids"])
    partition_fids = p3_train_new_fids | p3_train_replay_fids
    n_train_new = len(p3_train_new_fids)
    n_train_replay = len(p3_train_replay_fids)
    n_partition = len(partition_fids)
    print(f"Partition: train_new={n_train_new} replay={n_train_replay} "
          f"union={n_partition}")

    # ------------------------------------------------------------------
    # Load registry for quarantine check
    # ------------------------------------------------------------------
    try:
        registry = FamilyRegistry.from_path(registry_path)
    except Exception as exc:
        print(f"ERROR: cannot load registry {registry_path}: {exc}", file=sys.stderr)
        return 1
    quarantine_fids = set(registry.families_with_usage(TAG_QUARANTINE))
    print(f"Quarantine families: {len(quarantine_fids)}")

    # ------------------------------------------------------------------
    # Load sources
    # ------------------------------------------------------------------
    sources_payload: dict = {}
    total_loaded = 0
    total_after_family_filter = 0
    pool = SamplePool()
    duplicates_seen = 0  # samples rejected by add() due to duplicate sample_id

    def _add_all(samples: list[Sample]) -> None:
        nonlocal duplicates_seen
        for s in samples:
            if not pool.add(s):
                duplicates_seen += 1

    # P2 stages -> replay families
    p2_sources = [
        ("p2_stage1_code", P2_STAGE1_CODE),
        ("p2_stage2_boundary", P2_STAGE2_BOUNDARY),
        ("p2_stage3_repair", P2_STAGE3_REPAIR),
    ]
    for name, path in p2_sources:
        loaded, after_filter, samples = _load_p2_stage(path, p3_train_replay_fids)
        sources_payload[name] = {
            "path": _relpath(path),
            "loaded": loaded,
            "after_filter": after_filter,
        }
        total_loaded += loaded
        total_after_family_filter += after_filter
        _add_all(samples)
        print(f"  {name}: loaded={loaded} after_filter={after_filter}")

    # P3 verified test -> train_new families
    loaded, after_filter, samples = _load_p3_verified(
        P3_VERIFIED_TEST, p3_train_new_fids
    )
    sources_payload["p3_verified_test"] = {
        "path": _relpath(P3_VERIFIED_TEST),
        "loaded": loaded,
        "after_filter": after_filter,
    }
    total_loaded += loaded
    total_after_family_filter += after_filter
    _add_all(samples)
    print(f"  p3_verified_test: loaded={loaded} after_filter={after_filter}")

    # Defensive: filter_families to the union of partition families
    # (in case any source contains samples outside the targeted subset).
    kept_after_filter = pool.filter_families(partition_fids)
    print(f"Total loaded: {total_loaded}, "
          f"after family filter: {total_after_family_filter} "
          f"(duplicates so far: {duplicates_seen}, "
          f"after partition-union filter: {kept_after_filter})")

    # ------------------------------------------------------------------
    # Normalise variant_type + bug_type
    # ------------------------------------------------------------------
    n_normalised = pool.normalize_variant_type()
    print(f"Normalised variant_type + bug_type on {n_normalised} samples")

    # ------------------------------------------------------------------
    # Dedup: add() already rejected duplicate sample_ids (counted in
    # ``duplicates_seen``). Call dedup() defensively to clean up any
    # stragglers that may have been introduced by filter_families() or
    # other internal mutations (no-op in the typical case).
    # ------------------------------------------------------------------
    defensive_dedup = pool.dedup()
    duplicates_removed = duplicates_seen + defensive_dedup
    total_after_dedup = len(pool)
    print(f"Dedup: removed {duplicates_removed} duplicates "
          f"({duplicates_seen} from add() + {defensive_dedup} defensive), "
          f"total_after_dedup={total_after_dedup}")

    # ------------------------------------------------------------------
    # Per-family cap
    # ------------------------------------------------------------------
    # Compute family counts BEFORE cap (to log capped families)
    pre_cap_counts: dict[str, int] = {}
    for s in pool:
        pre_cap_counts[s.family_id] = pre_cap_counts.get(s.family_id, 0) + 1
    families_over_cap = {
        fid: c for fid, c in pre_cap_counts.items() if c > cap
    }

    samples_dropped_by_cap = pool.apply_family_cap(cap=cap)
    total_after_cap = len(pool)
    print(f"Per-family cap ({cap}): dropped {samples_dropped_by_cap} samples "
          f"across {len(families_over_cap)} families; "
          f"total_after_cap={total_after_cap}")
    if families_over_cap:
        for fid, c in sorted(families_over_cap.items()):
            print(f"    capped family {fid}: {c} -> {cap} (dropped {c - cap})")

    # ------------------------------------------------------------------
    # Write canonical-pool.jsonl
    # ------------------------------------------------------------------
    try:
        pool.to_jsonl(output_pool_path)
    except Exception as exc:
        print(f"ERROR: cannot write pool to {output_pool_path}: {exc}", file=sys.stderr)
        return 1
    print(f"Pool written: {output_pool_path} ({len(pool)} samples)")

    # ------------------------------------------------------------------
    # Compute pool SHA256
    # ------------------------------------------------------------------
    try:
        pool_sha = SamplePool.compute_sha256(output_pool_path)
    except Exception as exc:
        print(f"ERROR: cannot compute SHA256 of pool: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Build manifest
    # ------------------------------------------------------------------
    pool_stats = pool.stats()
    manifest = _build_manifest(
        sources=sources_payload,
        total_loaded=total_loaded,
        total_after_family_filter=total_after_family_filter,
        duplicates_removed=duplicates_removed,
        total_after_dedup=total_after_dedup,
        families_capped=len(families_over_cap),
        samples_dropped_by_cap=samples_dropped_by_cap,
        total_after_cap=total_after_cap,
        cap=cap,
        partition_family_count=n_partition,
        pool_stats=pool_stats,
        pool_sha256=pool_sha,
    )

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    try:
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with output_manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except Exception as exc:
        print(f"ERROR: cannot write manifest: {exc}", file=sys.stderr)
        return 1
    print(f"Manifest written: {output_manifest_path}")

    # ------------------------------------------------------------------
    # Hard gates (binding -- abort exit 1 if any fail)
    # ------------------------------------------------------------------
    errors = _run_hard_gates(
        pool,
        cap=cap,
        partition_family_ids=partition_fids,
        quarantine_family_ids=quarantine_fids,
    )
    if errors:
        print("\nHARD GATE FAILURES:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("\nHard gates: ALL PASS")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    vd = pool_stats["variant_distribution"]
    print(f"\nPool summary:")
    print(f"  total samples:    {len(pool)}")
    print(f"  family count:     {pool_stats['family_count']} / {n_partition} partition")
    print(f"  variant dist:     code={vd.get('code', 0)} "
          f"boundary={vd.get('boundary', 0)} "
          f"static_repair={vd.get('static_repair', 0)} "
          f"execution_repair={vd.get('execution_repair', 0)}")
    print(f"  duplicates removed: {duplicates_removed}")
    print(f"  samples dropped (cap): {samples_dropped_by_cap} across "
          f"{len(families_over_cap)} families")
    print(f"  pool SHA256:       {pool_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
