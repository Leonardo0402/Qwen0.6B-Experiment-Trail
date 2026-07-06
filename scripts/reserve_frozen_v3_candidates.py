"""scripts/reserve_frozen_v3_candidates.py -- Reserve Frozen v3 candidates.

Samples 120 candidate families from the MBPP test source pool,
stratified proportionally by ``difficulty``, and claims them with the
``frozen_v3_candidate`` tag in ``data/family-registry.json``. Writes
``data/frozen-eval/v3/candidates.json``.

Per Task 7 brief (.superpowers/sdd/task-7-brief.md). Idempotent:
re-running produces the same 120 candidates (same seed, same source
pool) and re-claiming is a no-op. Existing ``frozen_v3_candidate``
families are NOT excluded from the source pool (per task resolution
notes); the same seed deterministically reproduces the same 120.

Usage
-----
    python scripts/reserve_frozen_v3_candidates.py \\
        --registry data/family-registry.json \\
        --mbpp-verified-dir data/external/mbpp/verified \\
        --output-candidates data/frozen-eval/v3/candidates.json \\
        --output-registry data/family-registry.json \\
        --seed 42 \\
        --count 120

Exit codes
----------
    0   success
    1   invariant violation, insufficient pool, or I/O error
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

from src.family_registry import FamilyRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANDIDATE_TAG: str = "frozen_v3_candidate"
P2_TAGS: tuple[str, ...] = ("p2_train", "p2_validation", "p2_frozen_v2")
DIFFICULTY_BUCKETS: tuple[int, ...] = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Source pool
# ---------------------------------------------------------------------------

def build_source_pool(registry: FamilyRegistry) -> list[str]:
    """Return the sorted list of test-pool family_ids matching the
    source-pool definition.

    Conditions (binding, per task brief):
      - ``source_split == "test"``
      - ``"quarantine"`` NOT in ``usage``
      - none of ``p2_train`` / ``p2_validation`` / ``p2_frozen_v2`` in ``usage``
      - ``"frozen_v3"`` NOT in ``usage``

    Note (per task resolution): ``frozen_v3_candidate`` is NOT filtered
    out so re-runs reproduce the same 120 candidates deterministically.
    Re-claiming is a no-op (``FamilyEntry.claim`` is idempotent).
    """
    pool: list[str] = []
    for fid, entry in registry.families.items():
        if entry.source_split != "test":
            continue
        u = entry.usage
        if "quarantine" in u:
            continue
        if any(t in u for t in P2_TAGS):
            continue
        if "frozen_v3" in u:
            continue
        pool.append(fid)
    return sorted(pool)


# ---------------------------------------------------------------------------
# Difficulty map
# ---------------------------------------------------------------------------

def load_difficulty_map(verified_dir: Path) -> dict[str, int]:
    """Build ``{family_id: int difficulty}`` from ``verified/test.jsonl``.

    Defensive: families missing from the JSONL are simply absent from
    the returned map; :func:`sample_candidates` defaults them to 0.
    """
    test_jsonl = Path(verified_dir) / "test.jsonl"
    out: dict[str, int] = {}
    with test_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            fid = d.get("family_id")
            diff = d.get("difficulty")
            if fid is not None and diff is not None:
                out[fid] = int(diff)
    return out


# ---------------------------------------------------------------------------
# Proportional allocation
# ---------------------------------------------------------------------------

def allocate_proportionally(bucket_sizes: dict, total: int) -> dict:
    """Proportional allocation with floor + remainder distribution.

    Floors are computed with integer division (no floating point).
    Remainder slots (``total - sum_of_floors``) are distributed one at
    a time to the LARGEST buckets, breaking ties by bucket id ascending.

    The remainder is always strictly less than the number of non-zero
    buckets (sum of fractional remainders < N_nonzero), so 0-sized
    buckets never receive a remainder slot.
    """
    pool_total = sum(bucket_sizes.values())
    if pool_total == 0:
        return {b: 0 for b in bucket_sizes}
    allocation = {b: (bucket_sizes[b] * total) // pool_total for b in bucket_sizes}
    remainder = total - sum(allocation.values())
    if remainder > 0:
        sorted_buckets = sorted(
            bucket_sizes.keys(),
            key=lambda b: (-bucket_sizes[b], b),
        )
        for i in range(remainder):
            allocation[sorted_buckets[i % len(sorted_buckets)]] += 1
    return allocation


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def sample_candidates(
    source_pool: list[str],
    difficulty_map: dict[str, int],
    count: int,
    seed: int,
) -> tuple[list[str], dict]:
    """Stratified sampling by ``difficulty`` (0-4).

    Returns ``(candidates, stratification)`` where ``candidates`` is
    the sorted list of sampled family_ids and ``stratification`` is the
    per-bucket ``pool_size`` / ``allocated`` metadata dict.
    """
    # Group by difficulty (all 5 buckets always present).
    buckets: dict[int, list[str]] = {d: [] for d in DIFFICULTY_BUCKETS}
    for fid in source_pool:
        diff = difficulty_map.get(fid, 0)
        buckets[diff].append(fid)
    # Sort each bucket by family_id for stable shuffle input.
    for diff in buckets:
        buckets[diff].sort()
    # Proportional allocation.
    bucket_sizes = {d: len(buckets[d]) for d in DIFFICULTY_BUCKETS}
    allocation = allocate_proportionally(bucket_sizes, count)
    # Sample from each bucket using a single seeded RNG.
    rng = random.Random(seed)
    sampled: list[str] = []
    for diff in DIFFICULTY_BUCKETS:
        bucket = list(buckets[diff])  # copy (shuffle is in-place)
        rng.shuffle(bucket)
        sampled.extend(bucket[:allocation[diff]])
    # Final sort by family_id ascending (stable output).
    sampled.sort()
    stratification = {
        "method": "proportional_by_difficulty",
        "buckets": {
            str(d): {
                "pool_size": len(buckets[d]),
                "allocated": allocation[d],
            }
            for d in DIFFICULTY_BUCKETS
        },
    }
    return sampled, stratification


# ---------------------------------------------------------------------------
# Hard assertions
# ---------------------------------------------------------------------------

def assert_pre_claim_invariants(
    registry: FamilyRegistry,
    candidates: list[str],
    count: int,
) -> None:
    """Hard assertions 1-6 (before claiming). Raise AssertionError."""
    # 1. len(candidates) == count
    assert len(candidates) == count, (
        f"expected {count} candidates, got {len(candidates)}"
    )
    # 2. All candidates are in registry AND have source_split == "test"
    for fid in candidates:
        entry = registry.get(fid)
        assert entry is not None, f"candidate {fid} not in registry"
        assert entry.source_split == "test", (
            f"candidate {fid} has source_split={entry.source_split!r}, "
            f"expected 'test'"
        )
    # 3. No candidate has "quarantine" tag (BEFORE claiming)
    for fid in candidates:
        entry = registry.get(fid)
        assert "quarantine" not in entry.usage, (
            f"candidate {fid} has quarantine tag"
        )
    # 4. No candidate has any P2 tag
    for fid in candidates:
        entry = registry.get(fid)
        for tag in P2_TAGS:
            assert tag not in entry.usage, (
                f"candidate {fid} has {tag} tag"
            )
    # 5. No candidate has "frozen_v3" tag
    for fid in candidates:
        entry = registry.get(fid)
        assert "frozen_v3" not in entry.usage, (
            f"candidate {fid} has frozen_v3 tag"
        )
    # 6. All candidates are unique
    assert len(set(candidates)) == len(candidates), (
        f"candidates has {len(candidates) - len(set(candidates))} duplicates"
    )


def assert_post_claim_invariants(
    registry_path: Path,
    candidates: list[str],
) -> None:
    """Hard assertions 7-8 (after claiming + saving).

    Reloads the registry from disk to verify persistence, then asserts
    pairwise disjointness with the P2+quarantine whitelist (Task 6
    overlap).
    """
    # 7. Re-load and verify families_with_usage returns exactly these 120.
    reloaded = FamilyRegistry.from_path(registry_path)
    claimed = reloaded.families_with_usage(CANDIDATE_TAG)
    assert sorted(claimed) == sorted(candidates), (
        f"registry has {len(claimed)} claimed, expected {len(candidates)}; "
        f"differs by: {sorted(set(claimed) ^ set(candidates))}"
    )
    # 8. Pairwise disjoint (with P2+quarantine whitelist from Task 6).
    reloaded.assert_pairwise_disjoint(
        ["p2_train", "p2_validation", "p2_frozen_v2",
         CANDIDATE_TAG, "quarantine"],
        whitelist=[
            ("p2_train", "quarantine"),
            ("p2_validation", "quarantine"),
            ("p2_frozen_v2", "quarantine"),
        ],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Reserve 120 Frozen v3 candidate families from the MBPP test "
            "source pool, stratified by difficulty."
        ),
    )
    p.add_argument(
        "--registry",
        required=True,
        help="Path to data/family-registry.json (input).",
    )
    p.add_argument(
        "--mbpp-verified-dir",
        required=True,
        help="Directory containing test.jsonl "
             "(e.g. data/external/mbpp/verified).",
    )
    p.add_argument(
        "--output-candidates",
        required=True,
        help="Output path for candidates.json.",
    )
    p.add_argument(
        "--output-registry",
        required=True,
        help="Path to write updated registry (same as --registry for in-place).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42).",
    )
    p.add_argument(
        "--count",
        type=int,
        default=120,
        help="Number of candidates to sample (default: 120).",
    )
    return p


def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on error."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    registry_path = Path(args.registry)
    verified_dir = Path(args.mbpp_verified_dir)
    output_candidates = Path(args.output_candidates)
    output_registry = Path(args.output_registry)
    seed = args.seed
    count = args.count

    # ------------------------------------------------------------------
    # Load registry and build source pool
    # ------------------------------------------------------------------
    try:
        registry = FamilyRegistry.from_path(registry_path)
    except Exception as exc:
        print(
            f"ERROR: cannot load registry from {registry_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    source_pool = build_source_pool(registry)
    source_pool_size = len(source_pool)

    if source_pool_size < count:
        print(
            f"FIX_FIRST: only {source_pool_size} test-pool families "
            f"available, need {count}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Load difficulty map and sample
    # ------------------------------------------------------------------
    try:
        difficulty_map = load_difficulty_map(verified_dir)
    except Exception as exc:
        print(
            f"ERROR: cannot load difficulty map from {verified_dir}: {exc}",
            file=sys.stderr,
        )
        return 1

    candidates, stratification = sample_candidates(
        source_pool, difficulty_map, count, seed,
    )

    # ------------------------------------------------------------------
    # Hard assertions 1-6 (pre-claim)
    # ------------------------------------------------------------------
    try:
        assert_pre_claim_invariants(registry, candidates, count)
    except AssertionError as exc:
        print(f"INVARIANT VIOLATION: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Claim candidates in registry (idempotent)
    # ------------------------------------------------------------------
    for fid in candidates:
        registry.claim(fid, CANDIDATE_TAG)

    # ------------------------------------------------------------------
    # Save registry (in-place if --output-registry == --registry)
    # ------------------------------------------------------------------
    try:
        registry.to_path(output_registry)
    except Exception as exc:
        print(
            f"ERROR: cannot save registry to {output_registry}: {exc}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Hard assertions 7-8 (post-claim, post-save)
    # ------------------------------------------------------------------
    try:
        assert_post_claim_invariants(output_registry, candidates)
    except AssertionError as exc:
        print(f"INVARIANT VIOLATION (post-claim): {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: cannot reload registry: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Write candidates.json
    # ------------------------------------------------------------------
    try:
        output_candidates.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "reserve_frozen_v3_candidates.py",
            "schema_version": 1,
            "seed": seed,
            "source_pool_size": source_pool_size,
            "source_pool_definition": (
                "families with source_split=='test' AND not quarantine "
                "AND not p2_train AND not p2_validation AND not "
                "p2_frozen_v2 AND not frozen_v3"
            ),
            "candidate_count": len(candidates),
            "stratification": stratification,
            "candidates": candidates,
        }
        with output_candidates.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
    except Exception as exc:
        print(f"ERROR: cannot write candidates.json: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"Frozen v3 candidates reserved: {output_candidates}")
    print(f"  source_pool_size:    {source_pool_size}")
    print(f"  candidate_count:     {len(candidates)}")
    for d in DIFFICULTY_BUCKETS:
        b = stratification["buckets"][str(d)]
        print(
            f"  difficulty {d}: pool={b['pool_size']}, "
            f"allocated={b['allocated']}"
        )
    print(f"  registry updated:    {output_registry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
