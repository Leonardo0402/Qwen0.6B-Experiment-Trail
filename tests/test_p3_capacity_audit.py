"""tests/test_p3_capacity_audit.py -- Issue #14 P5 Capacity Audit tests.

Covers three areas specified in the P5 brief:

  (a) Static audit output format correctness
      - reports/p3/p3-formal-capacity-audit.{md,json} structure & values
  (b) Yield pilot sampling reproducibility with seed=42
      - re-running random.Random(42).sample(...) reproduces the same
        family_ids recorded in reports/p3/p3-yield-pilot.json
  (c) Verdict logic (FEASIBLE / AT_RISK / LIMIT)
      - wilson_lower boundary behaviour
      - _compute_verdict on synthetic extrapolations covering all 3 states
      - the actual pilot verdict matches the recorded output

These tests read the REAL report files produced by
``scripts/p3_formal_capacity_audit.py`` and ``scripts/p3_yield_pilot.py``.
If the report files do not exist yet, the corresponding tests are SKIPPED
with a clear instruction to run the scripts first.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import p3_formal_capacity_audit as audit  # noqa: E402
import p3_yield_pilot as pilot  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STATIC_AUDIT_JSON = _ROOT / "reports" / "p3" / "p3-formal-capacity-audit.json"
STATIC_AUDIT_MD = _ROOT / "reports" / "p3" / "p3-formal-capacity-audit.md"
YIELD_PILOT_JSON = _ROOT / "reports" / "p3" / "p3-yield-pilot.json"
YIELD_PILOT_MD = _ROOT / "reports" / "p3" / "p3-yield-pilot.md"
YIELD_PILOT_RAW = _ROOT / "reports" / "p3" / "p3-yield-pilot-raw.jsonl"
PARTITION_PATH = _ROOT / "data" / "p3-curriculum" / "family-partition.json"

BUCKETS = ("code", "boundary", "static_repair", "execution_repair")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def static_audit_payload() -> dict:
    if not STATIC_AUDIT_JSON.exists():
        pytest.skip(
            f"{STATIC_AUDIT_JSON} missing -- run "
            f"`py -3.11 scripts/p3_formal_capacity_audit.py` first"
        )
    with STATIC_AUDIT_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def yield_pilot_payload() -> dict:
    if not YIELD_PILOT_JSON.exists():
        pytest.skip(
            f"{YIELD_PILOT_JSON} missing -- run "
            f"`py -3.11 scripts/p3_yield_pilot.py --n-families 25 --seed 42` first"
        )
    with YIELD_PILOT_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# (a) Static audit output format correctness
# ===========================================================================

def test_static_audit_json_exists():
    assert STATIC_AUDIT_JSON.exists(), (
        f"{STATIC_AUDIT_JSON} missing -- run "
        f"`py -3.11 scripts/p3_formal_capacity_audit.py`"
    )


def test_static_audit_md_exists():
    assert STATIC_AUDIT_MD.exists(), (
        f"{STATIC_AUDIT_MD} missing -- run "
        f"`py -3.11 scripts/p3_formal_capacity_audit.py`"
    )


def test_static_audit_schema_version(static_audit_payload):
    assert static_audit_payload["schema_version"] == 1
    assert static_audit_payload["generator"] == "p3_formal_capacity_audit.py"


def test_static_audit_family_counts(static_audit_payload):
    fc = static_audit_payload["family_counts"]
    # Snapshot values verified against data/p3-curriculum/family-partition.json
    # and data/family-registry.json.
    assert fc["shared_train"] == 425
    assert fc["p3_train_new"] == 219
    assert fc["p3_train_replay"] == 206
    assert fc["p3_train_new"] + fc["p3_train_replay"] == fc["shared_train"]
    assert fc["quarantined"] == 50
    assert fc["frozen_v4"] == 100
    assert fc["validation_v2"] == 45
    # remaining_new_available == 0 (all new families are allocated)
    assert fc["remaining_new_available"] == 0


def test_static_audit_theoretical_yield(static_audit_payload):
    ty = static_audit_payload["theoretical_yield"]
    assert ty["variants_per_family_per_bucket"] == 3
    assert ty["total_variants_per_family"] == 12
    assert ty["shared_train_families"] == 425
    assert ty["max_per_bucket"] == 1275  # 425 * 3
    assert ty["max_total"] == 5100  # 425 * 12
    assert tuple(ty["buckets"]) == BUCKETS


def test_static_audit_capacity_targets(static_audit_payload):
    ct = static_audit_payload["capacity_targets"]
    # Balanced @2500
    assert ct["balanced_2500"] == {
        "code": 750, "boundary": 500,
        "static_repair": 500, "execution_repair": 750,
    }
    # Repair @2500
    assert ct["repair_2500"] == {
        "code": 375, "boundary": 375,
        "static_repair": 750, "execution_repair": 1000,
    }
    # Margin for FEASIBLE verdict
    assert ct["margin_pct_for_feasible"] == 0.10


def test_static_audit_theoretical_feasibility(static_audit_payload):
    tf = static_audit_payload["theoretical_feasibility"]
    # Theoretical max (5100) is well above both 2300 and 2500 totals.
    assert tf["balanced_2300_total"] is True
    assert tf["balanced_2500_total"] is True
    # Per-bucket theoretical max (1275) meets every @2500 target.
    for b in BUCKETS:
        assert tf["per_bucket_meets_balanced_2500"][b] is True, (
            f"balanced bucket {b} should be theoretically feasible"
        )
        assert tf["per_bucket_meets_repair_2500"][b] is True, (
            f"repair bucket {b} should be theoretically feasible"
        )


def test_static_audit_md_headers(static_audit_payload):
    md = STATIC_AUDIT_MD.read_text(encoding="utf-8")
    assert "# P3 Formal Capacity Audit (Issue #14 P5)" in md
    assert "## 1. Family Inventory" in md
    assert "## 2. Theoretical Yield (no verification)" in md
    assert "## 4. Capacity Targets (Issue #14 P5-P7)" in md
    assert "## 5. Theoretical Feasibility" in md


# ===========================================================================
# (b) Yield pilot sampling reproducibility with seed=42
# ===========================================================================

def test_yield_pilot_json_exists():
    assert YIELD_PILOT_JSON.exists(), (
        f"{YIELD_PILOT_JSON} missing -- run "
        f"`py -3.11 scripts/p3_yield_pilot.py --n-families 25 --seed 42`"
    )


def test_yield_pilot_md_exists():
    assert YIELD_PILOT_MD.exists(), (
        f"{YIELD_PILOT_MD} missing -- run the yield pilot first"
    )


def test_yield_pilot_raw_jsonl_exists():
    assert YIELD_PILOT_RAW.exists(), (
        f"{YIELD_PILOT_RAW} missing -- run the yield pilot first"
    )


def test_yield_pilot_config(yield_pilot_payload):
    cfg = yield_pilot_payload["config"]
    assert cfg["n_families_requested"] == 25
    assert cfg["n_families_sampled"] == 25
    assert cfg["seed"] == 42
    assert cfg["total_shared_families"] == 425
    assert cfg["z_score"] == pytest.approx(1.645, abs=1e-3)
    assert cfg["feasible_margin_pct"] == 0.10
    # All sampled families should have a source sample in the canonical pool.
    assert cfg["n_families_with_source"] == cfg["n_families_sampled"]
    assert cfg["n_families_no_source"] == 0


def test_yield_pilot_sampling_reproducible(yield_pilot_payload):
    """Re-running random.Random(42).sample(sorted(all_families), 25) must
    reproduce the family_ids recorded in the pilot report."""
    with PARTITION_PATH.open(encoding="utf-8") as fh:
        partition = json.load(fh)
    train_new = set(partition["p3_train_new"]["family_ids"])
    train_replay = set(partition["p3_train_replay"]["family_ids"])
    all_families = sorted(train_new | train_replay)

    rng = random.Random(42)
    expected = sorted(rng.sample(all_families, 25))

    recorded = yield_pilot_payload["sampled_family_ids"]
    assert recorded == expected, (
        "seed=42 sampling did not reproduce the recorded family_ids; "
        f"recorded={recorded[:5]}... expected={expected[:5]}..."
    )


def test_yield_pilot_bucket_stats_consistency(yield_pilot_payload):
    """bucket_stats.attempts must equal the sum of raw records per bucket."""
    stats = yield_pilot_payload["bucket_stats"]
    # Load raw records and count per bucket
    per_bucket_counts = {b: 0 for b in BUCKETS}
    with YIELD_PILOT_RAW.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            b = rec.get("bucket")
            if b in per_bucket_counts:
                per_bucket_counts[b] += 1
    for b in BUCKETS:
        assert stats[b]["attempts"] == per_bucket_counts[b], (
            f"bucket {b}: stats.attempts={stats[b]['attempts']} but "
            f"raw records={per_bucket_counts[b]}"
        )
        # verified count must be <= attempts
        assert stats[b]["verified"] <= stats[b]["attempts"]


def test_yield_pilot_extrapolation_fields(yield_pilot_payload):
    ext = yield_pilot_payload["extrapolation"]
    for b in BUCKETS:
        e = ext[b]
        assert "attempts" in e
        assert "verified" in e
        assert "yield_rate" in e
        assert "wilson_lower_90" in e
        assert "max_per_family" in e
        assert "projected_point_425" in e
        assert "projected_lower_425" in e
        # yield_rate = verified / attempts
        if e["attempts"] > 0:
            expected_rate = e["verified"] / e["attempts"]
            assert e["yield_rate"] == pytest.approx(expected_rate, abs=1e-4)
        # projected_lower_425 = floor(425 * max_per_family * wilson_lower_90)
        # The script projects with full-precision wilson_lower but stores a
        # value rounded to 4 decimals, so the recomputed floor may be off by 1.
        expected_lower = int(math.floor(
            425 * e["max_per_family"] * e["wilson_lower_90"]
        ))
        assert abs(e["projected_lower_425"] - expected_lower) <= 1, (
            f"bucket {b}: projected_lower_425={e['projected_lower_425']} "
            f"expected~{expected_lower} (rounding tolerance +/-1)"
        )
        # wilson lower bound <= yield rate (lower bound is conservative)
        assert e["wilson_lower_90"] <= e["yield_rate"] + 1e-9


def test_yield_pilot_boundary_max_per_family(yield_pilot_payload):
    """Boundary bucket is capped at 1 variant/family (existing generator)."""
    ext = yield_pilot_payload["extrapolation"]
    assert ext["boundary"]["max_per_family"] == 1
    # Other buckets use cap=3 per Issue #14 brief.
    for b in ("code", "static_repair", "execution_repair"):
        assert ext[b]["max_per_family"] == 3


# ===========================================================================
# (c) Verdict logic -- wilson_lower and _compute_verdict
# ===========================================================================

def test_wilson_lower_zero_attempts():
    assert pilot.wilson_lower(0, 0) == 0.0


def test_wilson_lower_zero_verified():
    """All failures (p=0) -> lower bound is 0."""
    assert pilot.wilson_lower(0, 10) == 0.0
    assert pilot.wilson_lower(0, 100) == 0.0


def test_wilson_lower_all_success_below_one():
    """All successes -> lower bound is high but strictly < 1.0."""
    p = pilot.wilson_lower(10, 10)
    assert 0.5 < p < 1.0
    p2 = pilot.wilson_lower(100, 100)
    assert 0.9 < p2 < 1.0


def test_wilson_lower_partial_below_rate():
    """50% yield -> lower bound is below 0.5 (conservative)."""
    p = pilot.wilson_lower(5, 10)
    assert 0.0 < p < 0.5


def test_wilson_lower_monotone_with_success():
    """Holding attempts fixed, more successes -> higher lower bound."""
    n = 20
    prev = -1.0
    for k in range(0, n + 1):
        lb = pilot.wilson_lower(k, n)
        assert lb >= prev - 1e-9
        prev = lb


def _make_extrapolation(projected_lowers: dict) -> dict:
    """Build a minimal extrapolation dict for _compute_verdict tests."""
    return {
        b: {"projected_lower_425": projected_lowers[b]}
        for b in BUCKETS
    }


def test_compute_verdict_FEASIBLE():
    """All buckets of both candidates meet target * (1 + 10% margin)."""
    ext = _make_extrapolation({
        # balanced margins: code>=825, boundary>=550, static>=550, exec>=825
        # repair margins:   code>=412, boundary>=412, static>=825, exec>=1100
        #取最大: code>=825, boundary>=550, static>=825, exec>=1100
        "code": 900,
        "boundary": 600,
        "static_repair": 900,
        "execution_repair": 1200,
    })
    v = pilot._compute_verdict(ext)
    assert v["verdict"] == "FORMAL_CAPACITY_FEASIBLE"
    assert all(v["balanced_buckets_ok"].values())
    assert all(v["repair_buckets_ok"].values())
    assert not any(v["balanced_critical_buckets_below_target"].values())
    assert not any(v["repair_critical_buckets_below_target"].values())


def test_compute_verdict_AT_RISK():
    """Total passes, no critical bucket, but one bucket fails the +10%
    margin -> FORMAL_CAPACITY_AT_RISK."""
    ext = _make_extrapolation({
        # execution_repair = 1050 >= 1000 (bare target) but < 1100 (margin)
        # -> repair_buckets_ok[execution_repair] = False, but not critical
        "code": 900,
        "boundary": 600,
        "static_repair": 900,
        "execution_repair": 1050,
    })
    v = pilot._compute_verdict(ext)
    assert v["verdict"] == "FORMAL_CAPACITY_AT_RISK"
    # Totals still pass the 2500*1.1=2750 bar
    assert v["balanced_total_ok"] is True
    assert v["repair_total_ok"] is True
    # No critical buckets (all >= bare target)
    assert not any(v["balanced_critical_buckets_below_target"].values())
    assert not any(v["repair_critical_buckets_below_target"].values())
    # But execution_repair fails the repair +10% margin
    assert v["repair_buckets_ok"]["execution_repair"] is False


def test_compute_verdict_LIMIT_critical_bucket():
    """A bucket below the bare target (no margin) -> MBPP_FAMILY_OR_VARIANT_LIMIT."""
    ext = _make_extrapolation({
        # boundary = 300 < 375 (repair bare target) and < 500 (balanced target)
        "code": 900,
        "boundary": 300,
        "static_repair": 900,
        "execution_repair": 1200,
    })
    v = pilot._compute_verdict(ext)
    assert v["verdict"] == "MBPP_FAMILY_OR_VARIANT_LIMIT"
    assert v["balanced_critical_buckets_below_target"]["boundary"] is True
    assert v["repair_critical_buckets_below_target"]["boundary"] is True


def test_compute_verdict_LIMIT_total_below_target():
    """Total projected below 2500*1.1=2750 with all buckets above bare
    targets but well below margins -> still AT_RISK or LIMIT depending
    on total. With very low totals, the verdict is LIMIT."""
    ext = _make_extrapolation({
        "code": 400,         # >= 375 repair target, < 750 balanced target
        "boundary": 380,     # >= 375 repair, < 500 balanced
        "static_repair": 600,  # >= 500 balanced, < 750 repair
        "execution_repair": 800,  # >= 750 balanced, < 1000 repair
    })
    v = pilot._compute_verdict(ext)
    # Multiple critical buckets (code, boundary for balanced; static, exec for repair)
    assert v["verdict"] == "MBPP_FAMILY_OR_VARIANT_LIMIT"


def test_actual_pilot_verdict_recorded(yield_pilot_payload):
    """The recorded pilot verdict must be one of the three valid states."""
    v = yield_pilot_payload["verdict"]["verdict"]
    assert v in (
        "FORMAL_CAPACITY_FEASIBLE",
        "FORMAL_CAPACITY_AT_RISK",
        "MBPP_FAMILY_OR_VARIANT_LIMIT",
    ), f"unexpected verdict: {v}"


def test_actual_pilot_verdict_consistent_with_extrapolation(yield_pilot_payload):
    """Re-running _compute_verdict on the recorded extrapolation must
    reproduce the recorded verdict. This guards against drift between
    the verdict function and the verdict stored in the JSON."""
    ext = yield_pilot_payload["extrapolation"]
    recorded = yield_pilot_payload["verdict"]["verdict"]
    recomputed = pilot._compute_verdict(ext)["verdict"]
    assert recomputed == recorded, (
        f"verdict drift: recorded={recorded} recomputed={recomputed}"
    )
