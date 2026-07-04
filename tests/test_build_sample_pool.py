"""tests/test_build_sample_pool.py -- Integration tests for Task 10.

Covers the 6 integration tests specified in ``.superpowers/sdd/task-10-brief.md``:

  9.  test_pool_loads_from_all_sources
  10. test_pool_no_duplicate_sample_ids
  11. test_pool_variant_distribution
  12. test_pool_family_cap_enforced
  13. test_pool_manifest_correct
  14. test_pool_only_partition_families

These tests use the REAL data files (data/p2-curriculum/, data/external/mbpp/,
data/p3-curriculum/, data/family-registry.json) and read the canonical-pool.jsonl
+ canonical-pool-manifest.json produced by ``scripts/build_sample_pool.py``.

If the pool/manifest files do not exist yet, the tests are SKIPPED with a
clear instruction to run the orchestrator first.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sample_pool import SamplePool, ALLOWED_VARIANT_TYPES  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POOL_PATH = _ROOT / "data" / "p3-curriculum" / "canonical-pool.jsonl"
MANIFEST_PATH = _ROOT / "data" / "p3-curriculum" / "canonical-pool-manifest.json"
PARTITION_PATH = _ROOT / "data" / "p3-curriculum" / "family-partition.json"
REGISTRY_PATH = _ROOT / "data" / "family-registry.json"
SCRIPT_PATH = _ROOT / "scripts" / "build_sample_pool.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated_pool() -> SamplePool:
    """Ensure the pool exists (run the orchestrator if missing) and return
    a SamplePool loaded from canonical-pool.jsonl."""
    if not POOL_PATH.exists():
        # Run the orchestrator once to produce the pool + manifest.
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        if result.returncode != 0:
            pytest.fail(
                f"build_sample_pool.py failed (exit {result.returncode}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    if not POOL_PATH.exists():
        pytest.skip("canonical-pool.jsonl not produced by the orchestrator")
    return SamplePool.from_jsonl(POOL_PATH)


@pytest.fixture(scope="module")
def manifest() -> dict:
    """Load the canonical-pool-manifest.json produced by the orchestrator."""
    if not MANIFEST_PATH.exists():
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        if result.returncode != 0:
            pytest.fail(
                f"build_sample_pool.py failed (exit {result.returncode}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    if not MANIFEST_PATH.exists():
        pytest.skip("canonical-pool-manifest.json not produced by the orchestrator")
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def partition() -> dict:
    """Load the family-partition.json (Task 9 output)."""
    if not PARTITION_PATH.exists():
        pytest.skip("family-partition.json missing (Task 9 must run first)")
    with PARTITION_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Test 9: pool loads from all sources
# ---------------------------------------------------------------------------

def test_pool_loads_from_all_sources(generated_pool, manifest):
    """Pool has samples from P2 stages + P3 verified.

    We verify by:
      - Pool size > 0
      - All 4 sources in manifest report loaded > 0
      - Pool size == manifest['total_after_cap']
    """
    pool = generated_pool
    assert len(pool) > 0, "pool is empty"
    assert len(pool) == manifest["total_after_cap"], (
        f"pool size {len(pool)} != manifest total_after_cap "
        f"{manifest['total_after_cap']}"
    )
    sources = manifest["sources"]
    expected_sources = {
        "p2_stage1_code", "p2_stage2_boundary",
        "p2_stage3_repair", "p3_verified_test",
    }
    assert set(sources.keys()) == expected_sources, (
        f"sources mismatch: {set(sources.keys())}"
    )
    for name, info in sources.items():
        assert info["loaded"] > 0, f"{name} loaded=0"
        assert info["after_filter"] > 0, f"{name} after_filter=0"


# ---------------------------------------------------------------------------
# Test 10: no duplicate sample_ids
# ---------------------------------------------------------------------------

def test_pool_no_duplicate_sample_ids(generated_pool):
    """All sample_ids in the pool are unique."""
    sample_ids = [s.sample_id for s in generated_pool]
    assert len(sample_ids) == len(set(sample_ids)), (
        f"{len(sample_ids) - len(set(sample_ids))} duplicate sample_ids in pool"
    )


# ---------------------------------------------------------------------------
# Test 11: variant distribution (all 4 types present)
# ---------------------------------------------------------------------------

def test_pool_variant_distribution(generated_pool, manifest):
    """All 4 variant types (code / boundary / static_repair /
    execution_repair) are present in the pool."""
    vd = manifest["variant_distribution"]
    for v in ("code", "boundary", "static_repair", "execution_repair"):
        assert vd.get(v, 0) > 0, f"variant_type {v!r} missing from distribution"
    # Also verify against the actual pool (not just the manifest)
    actual_vd = {"code": 0, "boundary": 0, "static_repair": 0, "execution_repair": 0}
    for s in generated_pool:
        assert s.variant_type in ALLOWED_VARIANT_TYPES, (
            f"sample {s.sample_id} has invalid variant_type {s.variant_type!r}"
        )
        actual_vd[s.variant_type] += 1
    for v in actual_vd:
        assert actual_vd[v] == vd.get(v, 0), (
            f"variant {v}: actual={actual_vd[v]} manifest={vd.get(v, 0)}"
        )


# ---------------------------------------------------------------------------
# Test 12: family cap enforced (no family > cap)
# ---------------------------------------------------------------------------

def test_pool_family_cap_enforced(generated_pool, manifest):
    """No family has > cap samples in the pool."""
    cap = manifest["per_family_cap"]
    family_counts: dict[str, int] = {}
    for s in generated_pool:
        family_counts[s.family_id] = family_counts.get(s.family_id, 0) + 1
    over_cap = {fid: c for fid, c in family_counts.items() if c > cap}
    assert not over_cap, (
        f"{len(over_cap)} families exceed cap={cap}: "
        f"{dict(list(over_cap.items())[:5])}"
    )


# ---------------------------------------------------------------------------
# Test 13: manifest counts match actual pool
# ---------------------------------------------------------------------------

def test_pool_manifest_correct(generated_pool, manifest):
    """Manifest counts match the actual pool."""
    pool = generated_pool
    # total_after_cap == pool size
    assert manifest["total_after_cap"] == len(pool), (
        f"manifest total_after_cap={manifest['total_after_cap']} "
        f"!= pool size {len(pool)}"
    )
    # family_count == number of unique families in pool
    actual_family_count = len({s.family_id for s in pool})
    assert manifest["family_count"] == actual_family_count, (
        f"manifest family_count={manifest['family_count']} "
        f"!= actual {actual_family_count}"
    )
    # variant_distribution sums to pool size
    vd = manifest["variant_distribution"]
    assert sum(vd.values()) == len(pool), (
        f"variant_distribution sum={sum(vd.values())} != pool size {len(pool)}"
    )
    # pool_sha256 matches recomputed SHA
    from src.sample_pool import SamplePool as _SP
    actual_sha = _SP.compute_sha256(POOL_PATH)
    assert manifest["pool_sha256"] == actual_sha, (
        f"manifest sha={manifest['pool_sha256']} != actual {actual_sha}"
    )
    # Hard-gate thresholds (manifest records, not just verified against pool):
    assert manifest["total_loaded"] == sum(
        s["loaded"] for s in manifest["sources"].values()
    )
    assert manifest["total_after_family_filter"] == sum(
        s["after_filter"] for s in manifest["sources"].values()
    )
    assert manifest["total_after_dedup"] == (
        manifest["total_after_family_filter"] - manifest["duplicates_removed"]
    )
    assert manifest["total_after_cap"] == (
        manifest["total_after_dedup"] - manifest["samples_dropped_by_cap"]
    )


# ---------------------------------------------------------------------------
# Test 14: only partition families in pool
# ---------------------------------------------------------------------------

def test_pool_only_partition_families(generated_pool, partition):
    """All families in pool are in the 425 partition families
    (train_new ∪ train_replay)."""
    train_new = set(partition["p3_train_new"]["family_ids"])
    train_replay = set(partition["p3_train_replay"]["family_ids"])
    partition_families = train_new | train_replay
    pool_families = {s.family_id for s in generated_pool}
    outside = pool_families - partition_families
    assert not outside, (
        f"{len(outside)} families in pool not in partition: "
        f"{sorted(outside)[:5]}"
    )
    # Also verify pool size <= 425 (partition count)
    assert len(pool_families) <= len(partition_families)


# ---------------------------------------------------------------------------
# Bonus: no quarantined family in pool
# ---------------------------------------------------------------------------

def test_pool_no_quarantined_family(generated_pool):
    """No quarantined family appears in the pool (hard gate 6)."""
    if not REGISTRY_PATH.exists():
        pytest.skip("family-registry.json missing")
    with REGISTRY_PATH.open(encoding="utf-8") as fh:
        registry_data = json.load(fh)
    quarantine_families = {
        fid for fid, entry in registry_data.get("families", {}).items()
        if "quarantine" in entry.get("usage", [])
    }
    pool_families = {s.family_id for s in generated_pool}
    overlap = pool_families & quarantine_families
    assert not overlap, (
        f"{len(overlap)} quarantined families in pool: {sorted(overlap)[:5]}"
    )
