"""tests/test_build_p3_partition.py -- Tests for Task 9.

Covers the 10 tests specified in ``.superpowers/sdd/task-9-brief.md``:

  1.  test_partition_counts
  2.  test_pairwise_disjoint_passes
  3.  test_pairwise_disjoint_no_whitelist_fails
  4.  test_quarantine_excluded_from_all_p3_splits
  5.  test_frozen_v3_disjoint_from_p3
  6.  test_p2_frozen_v2_disjoint_from_p3
  7.  test_registry_claims_consistent
  8.  test_deterministic_sampling
  9.  test_json_output_schema
  10. test_no_family_in_multiple_p3_sets

All tests use synthetic registries in ``tmp_path`` (no I/O on real data).
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

from src.family_registry import FamilyEntry, FamilyRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    family_id: str,
    *,
    source_split: str = "test",
    usage: list[str] | None = None,
) -> FamilyEntry:
    """Build a minimal FamilyEntry for tests."""
    sample_id = family_id.replace("mbpp_fam_", "mbpp_")
    return FamilyEntry(
        family_id=family_id,
        source_task_id=sample_id,
        source_split=source_split,
        usage=list(usage) if usage is not None else [],
        sample_ids=[sample_id],
    )


def _write_registry(path: Path, families: dict[str, FamilyEntry]) -> None:
    """Write a synthetic registry to *path*."""
    reg = FamilyRegistry()
    for fid, entry in families.items():
        reg.families[fid] = entry
    reg.to_path(path)


def _run_script(
    registry_path: Path,
    partition_path: Path,
    *,
    seed: int = 42,
) -> subprocess.CompletedProcess:
    """Invoke the partition script via subprocess."""
    return subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "build_p3_partition.py"),
            "--registry", str(registry_path),
            "--output-partition", str(partition_path),
            "--seed", str(seed),
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )


def _build_synthetic_registry() -> dict[str, FamilyEntry]:
    """Build a synthetic registry mirroring the brief's expected shape.

    Layout (matches the brief's "verified" numbers):
      - 100 frozen_v3 families (test split, ids 100-199)
      - 75 p2_validation families (validation split, ids 300-374)
      - 75 p2_frozen_v2 families (mixed splits, ids 400-474)
      - 50 quarantine families (mixed splits, ids 500-549)
        - 17 test, 7 validation, 26 train (per brief's table)
      - 224 p2_train families (train split, ids 600-823)
        - of which 26 are also quarantine (ids 600-625)
      - 61 available validation families (validation split, ids 900-960)
      - 248 available test families (test split, ids 1000-1247)

    Note: the brief states "26 of the 224 p2_train families are also
    quarantine". This synthetic registry matches the brief's expected
    shape so test_partition_counts can verify replay=198.
    """
    families: dict[str, FamilyEntry] = {}

    # 100 frozen_v3 (test split)
    for i in range(100, 200):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="test",
                              usage=["frozen_v3", "frozen_v3_candidate"])

    # 75 p2_validation (validation split)
    for i in range(300, 375):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="validation", usage=["p2_validation"])

    # 75 p2_frozen_v2 (train split, to keep things simple)
    for i in range(400, 475):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="train", usage=["p2_frozen_v2"])

    # 50 quarantine families: 17 test, 7 validation, 26 train
    # The 26 train-quarantine are ALSO p2_train (per brief's interpretation)
    quarantine_test_ids = [f"mbpp_fam_{i}" for i in range(500, 517)]
    quarantine_validation_ids = [f"mbpp_fam_{i}" for i in range(517, 524)]
    quarantine_train_ids = [f"mbpp_fam_{i}" for i in range(600, 626)]

    for fid in quarantine_test_ids:
        families[fid] = _entry(fid, source_split="test", usage=["quarantine"])
    for fid in quarantine_validation_ids:
        families[fid] = _entry(fid, source_split="validation", usage=["quarantine"])

    # 224 p2_train families: ids 600-823 (first 26 also quarantine)
    for i in range(600, 824):
        fid = f"mbpp_fam_{i}"
        usage = ["p2_train"]
        if i < 626:  # first 26 = quarantine
            usage.append("quarantine")
        families[fid] = _entry(fid, source_split="train", usage=usage)

    # 61 available validation families (empty usage, validation split)
    for i in range(900, 961):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="validation", usage=[])

    # 248 available test families (empty usage, test split)
    for i in range(1000, 1248):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="test", usage=[])

    return families


def _run_on_synthetic(tmp_path: Path) -> tuple[Path, Path, dict, FamilyRegistry]:
    """Run the script on a synthetic registry; return key objects.

    Returns ``(registry_path, partition_path, partition_json, reloaded_reg)``.
    """
    families = _build_synthetic_registry()
    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)
    partition_path = tmp_path / "family-partition.json"
    result = _run_script(reg_path, partition_path)
    assert result.returncode == 0, (
        f"script failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    with partition_path.open(encoding="utf-8") as fh:
        pdata = json.load(fh)
    reloaded = FamilyRegistry.from_path(reg_path)
    return reg_path, partition_path, pdata, reloaded


# ---------------------------------------------------------------------------
# Test 1: partition counts
# ---------------------------------------------------------------------------

def test_partition_counts(tmp_path):
    """validation=90, train_new=219, train_replay=198, total=417."""
    _, _, pdata, _ = _run_on_synthetic(tmp_path)
    assert pdata["p3_validation"]["count"] == 90, (
        f"p3_validation count={pdata['p3_validation']['count']}, want 90"
    )
    assert pdata["p3_train_new"]["count"] == 219, (
        f"p3_train_new count={pdata['p3_train_new']['count']}, want 219"
    )
    assert pdata["p3_train_replay"]["count"] == 198, (
        f"p3_train_replay count={pdata['p3_train_replay']['count']}, want 198"
    )
    assert pdata["p3_train_total"] == 417, (
        f"p3_train_total={pdata['p3_train_total']}, want 417"
    )


# ---------------------------------------------------------------------------
# Test 2: pairwise disjoint passes (with whitelist)
# ---------------------------------------------------------------------------

def test_pairwise_disjoint_passes(tmp_path):
    """assert_pairwise_disjoint with whitelist passes after partitioning."""
    _, _, _, reloaded = _run_on_synthetic(tmp_path)
    # Should NOT raise -- the (p3_train_replay, p2_train) overlap is whitelisted
    reloaded.assert_pairwise_disjoint(
        ["frozen_v3", "p3_validation", "p3_train", "p3_train_replay"],
        whitelist=[("p3_train_replay", "p2_train")],
    )


# ---------------------------------------------------------------------------
# Test 3: pairwise disjoint WITHOUT whitelist FAILS
# ---------------------------------------------------------------------------

def test_pairwise_disjoint_no_whitelist_fails(tmp_path):
    """Without the whitelist, the p3_train_replay ∩ p2_train overlap is
    detected as a violation (confirms the check is working)."""
    _, _, _, reloaded = _run_on_synthetic(tmp_path)
    with pytest.raises(AssertionError, match="pairwise disjoint violation"):
        reloaded.assert_pairwise_disjoint(
            # Include p2_train so the (p3_train_replay, p2_train) pair is
            # actually checked -- without p2_train in the list there is no
            # pair to evaluate the overlap against.
            ["frozen_v3", "p3_validation", "p3_train", "p3_train_replay",
             "p2_train"],
            # No whitelist: the p3_train_replay/p2_train overlap is a violation
            whitelist=None,
        )


# ---------------------------------------------------------------------------
# Test 4: quarantine excluded from all P3 splits
# ---------------------------------------------------------------------------

def test_quarantine_excluded_from_all_p3_splits(tmp_path):
    """No quarantine family appears in any P3 partition set."""
    _, _, pdata, reloaded = _run_on_synthetic(tmp_path)
    quarantine_ids = set(reloaded.families_with_usage("quarantine"))
    p3_val = set(pdata["p3_validation"]["family_ids"])
    p3_train_new = set(pdata["p3_train_new"]["family_ids"])
    p3_train_replay = set(pdata["p3_train_replay"]["family_ids"])
    assert not (p3_val & quarantine_ids), (
        f"p3_validation has quarantine: {p3_val & quarantine_ids}"
    )
    assert not (p3_train_new & quarantine_ids), (
        f"p3_train_new has quarantine: {p3_train_new & quarantine_ids}"
    )
    assert not (p3_train_replay & quarantine_ids), (
        f"p3_train_replay has quarantine: {p3_train_replay & quarantine_ids}"
    )


# ---------------------------------------------------------------------------
# Test 5: frozen_v3 disjoint from p3
# ---------------------------------------------------------------------------

def test_frozen_v3_disjoint_from_p3(tmp_path):
    """No frozen_v3 family appears in train or validation."""
    _, _, pdata, reloaded = _run_on_synthetic(tmp_path)
    frozen_v3_ids = set(reloaded.families_with_usage("frozen_v3"))
    p3_val = set(pdata["p3_validation"]["family_ids"])
    p3_train_new = set(pdata["p3_train_new"]["family_ids"])
    p3_train_replay = set(pdata["p3_train_replay"]["family_ids"])
    assert not (frozen_v3_ids & p3_val)
    assert not (frozen_v3_ids & p3_train_new)
    assert not (frozen_v3_ids & p3_train_replay)


# ---------------------------------------------------------------------------
# Test 6: p2_frozen_v2 disjoint from p3
# ---------------------------------------------------------------------------

def test_p2_frozen_v2_disjoint_from_p3(tmp_path):
    """No p2_frozen_v2 family appears in any P3 split."""
    _, _, pdata, reloaded = _run_on_synthetic(tmp_path)
    p2_frozen_v2_ids = set(reloaded.families_with_usage("p2_frozen_v2"))
    p3_val = set(pdata["p3_validation"]["family_ids"])
    p3_train_new = set(pdata["p3_train_new"]["family_ids"])
    p3_train_replay = set(pdata["p3_train_replay"]["family_ids"])
    assert not (p2_frozen_v2_ids & p3_val)
    assert not (p2_frozen_v2_ids & p3_train_new)
    assert not (p2_frozen_v2_ids & p3_train_replay)


# ---------------------------------------------------------------------------
# Test 7: registry claims consistent
# ---------------------------------------------------------------------------

def test_registry_claims_consistent(tmp_path):
    """Every family_id in the partition has the correct tag claimed in
    the registry."""
    _, _, pdata, reloaded = _run_on_synthetic(tmp_path)
    val_partition = set(pdata["p3_validation"]["family_ids"])
    train_new_partition = set(pdata["p3_train_new"]["family_ids"])
    train_replay_partition = set(pdata["p3_train_replay"]["family_ids"])

    val_claimed = set(reloaded.families_with_usage("p3_validation"))
    train_claimed = set(reloaded.families_with_usage("p3_train"))
    replay_claimed = set(reloaded.families_with_usage("p3_train_replay"))

    assert val_claimed == val_partition, (
        f"p3_validation mismatch: partition - claimed = "
        f"{val_partition - val_claimed}, claimed - partition = "
        f"{val_claimed - val_partition}"
    )
    assert train_claimed == train_new_partition, (
        f"p3_train mismatch: partition - claimed = "
        f"{train_new_partition - train_claimed}"
    )
    assert replay_claimed == train_replay_partition, (
        f"p3_train_replay mismatch: partition - claimed = "
        f"{train_replay_partition - replay_claimed}"
    )


# ---------------------------------------------------------------------------
# Test 8: deterministic sampling
# ---------------------------------------------------------------------------

def test_deterministic_sampling(tmp_path):
    """Re-running with seed=42 produces the same 29 supplemented
    validation family_ids."""
    families = _build_synthetic_registry()
    reg1 = tmp_path / "registry1.json"
    reg2 = tmp_path / "registry2.json"
    _write_registry(reg1, families)
    _write_registry(reg2, families)

    p1 = tmp_path / "partition1.json"
    p2 = tmp_path / "partition2.json"
    r1 = _run_script(reg1, p1, seed=42)
    r2 = _run_script(reg2, p2, seed=42)
    assert r1.returncode == 0, f"run1 failed:\n{r1.stderr}"
    assert r2.returncode == 0, f"run2 failed:\n{r2.stderr}"

    with p1.open(encoding="utf-8") as fh:
        d1 = json.load(fh)
    with p2.open(encoding="utf-8") as fh:
        d2 = json.load(fh)
    assert d1["p3_validation"]["family_ids"] == d2["p3_validation"]["family_ids"], (
        "validation family_ids differ across runs with same seed"
    )
    assert d1["p3_train_new"]["family_ids"] == d2["p3_train_new"]["family_ids"], (
        "train_new family_ids differ across runs with same seed"
    )
    assert d1["p3_train_replay"]["family_ids"] == d2["p3_train_replay"]["family_ids"], (
        "train_replay family_ids differ across runs with same seed"
    )


# ---------------------------------------------------------------------------
# Test 9: json output schema
# ---------------------------------------------------------------------------

def test_json_output_schema(tmp_path):
    """family-partition.json has all required fields per the brief."""
    _, _, pdata, _ = _run_on_synthetic(tmp_path)

    # Top-level required fields
    required_top = {
        "schema_version", "generated_at", "generator", "seed",
        "p3_validation", "p3_train_new", "p3_train_replay", "p3_train_total",
        "frozen_v3_count", "pairwise_disjoint", "quarantine_exclusion",
    }
    assert required_top.issubset(pdata.keys()), (
        f"missing top-level fields: {required_top - set(pdata.keys())}"
    )
    assert pdata["schema_version"] == 1
    assert pdata["generator"] == "build_p3_partition.py"
    assert pdata["seed"] == 42

    # p3_validation sub-fields
    val = pdata["p3_validation"]
    for k in ("count", "from_validation_split", "from_test_split_supplement", "family_ids"):
        assert k in val, f"p3_validation missing {k}"
    assert val["count"] == len(val["family_ids"])

    # p3_train_new sub-fields
    train_new = pdata["p3_train_new"]
    for k in ("count", "source", "family_ids"):
        assert k in train_new, f"p3_train_new missing {k}"
    assert train_new["count"] == len(train_new["family_ids"])

    # p3_train_replay sub-fields
    replay = pdata["p3_train_replay"]
    for k in ("count", "source", "excluded_quarantine_count", "family_ids"):
        assert k in replay, f"p3_train_replay missing {k}"
    assert replay["count"] == len(replay["family_ids"])

    # pairwise_disjoint sub-fields
    pd = pdata["pairwise_disjoint"]
    for k in ("tags_checked", "pairs_checked", "whitelist",
              "whitelist_intersection_count", "result"):
        assert k in pd, f"pairwise_disjoint missing {k}"
    assert pd["result"] == "PASS"
    assert pd["pairs_checked"] == 6
    assert pd["tags_checked"] == ["frozen_v3", "p3_validation", "p3_train", "p3_train_replay"]
    assert pd["whitelist"] == [["p3_train_replay", "p2_train"]]

    # quarantine_exclusion sub-fields
    qe = pdata["quarantine_exclusion"]
    for k in ("quarantine_total", "excluded_from_validation",
              "excluded_from_train_new", "excluded_from_train_replay",
              "excluded_from_frozen_v3"):
        assert k in qe, f"quarantine_exclusion missing {k}"


# ---------------------------------------------------------------------------
# Test 10: no family in multiple p3 sets
# ---------------------------------------------------------------------------

def test_no_family_in_multiple_p3_sets(tmp_path):
    """No family_id appears in both p3_validation and p3_train (new or replay)."""
    _, _, pdata, _ = _run_on_synthetic(tmp_path)
    p3_val = set(pdata["p3_validation"]["family_ids"])
    p3_train_new = set(pdata["p3_train_new"]["family_ids"])
    p3_train_replay = set(pdata["p3_train_replay"]["family_ids"])
    assert not (p3_val & p3_train_new), (
        f"validation ∩ train_new = {p3_val & p3_train_new}"
    )
    assert not (p3_val & p3_train_replay), (
        f"validation ∩ train_replay = {p3_val & p3_train_replay}"
    )
    assert not (p3_train_new & p3_train_replay), (
        f"train_new ∩ train_replay = {p3_train_new & p3_train_replay}"
    )


# ---------------------------------------------------------------------------
# Test 11 (bonus): idempotency -- re-running produces same output
# ---------------------------------------------------------------------------

def test_script_is_idempotent(tmp_path):
    """Re-running the script on an already-claimed registry produces the
    same partition (no FIX_FIRST error, same counts, same family_ids)."""
    families = _build_synthetic_registry()
    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)

    p1 = tmp_path / "partition1.json"
    p2 = tmp_path / "partition2.json"
    r1 = _run_script(reg_path, p1, seed=42)
    assert r1.returncode == 0, f"run1 failed:\n{r1.stderr}"
    # Second run on the SAME registry (already claimed)
    r2 = _run_script(reg_path, p2, seed=42)
    assert r2.returncode == 0, (
        f"run2 (idempotency) failed (exit {r2.returncode}):\n"
        f"stdout:\n{r2.stdout}\nstderr:\n{r2.stderr}"
    )
    with p1.open(encoding="utf-8") as fh:
        d1 = json.load(fh)
    with p2.open(encoding="utf-8") as fh:
        d2 = json.load(fh)
    assert d1["p3_validation"]["family_ids"] == d2["p3_validation"]["family_ids"]
    assert d1["p3_train_new"]["family_ids"] == d2["p3_train_new"]["family_ids"]
    assert d1["p3_train_replay"]["family_ids"] == d2["p3_train_replay"]["family_ids"]
    # Re-loaded registry should still have exactly one claim per family
    reloaded = FamilyRegistry.from_path(reg_path)
    val_claimed = reloaded.families_with_usage("p3_validation")
    assert len(val_claimed) == d2["p3_validation"]["count"], (
        f"idempotent run double-claimed: {len(val_claimed)} != "
        f"{d2['p3_validation']['count']}"
    )
