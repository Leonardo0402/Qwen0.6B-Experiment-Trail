"""tests/test_reserve_frozen_v3_candidates.py -- Tests for Task 7.

Covers the 10 tests specified in ``.superpowers/sdd/task-7-brief.md``:

  1.  test_source_pool_filters_correctly
  2.  test_candidate_count_120
  3.  test_seed_determinism
  4.  test_stratification_proportional
  5.  test_candidates_sorted
  6.  test_no_candidate_in_quarantine
  7.  test_no_candidate_in_p2
  8.  test_registry_claim_persists
  9.  test_pairwise_disjoint_holds
  10. test_insufficient_pool_aborts

All tests use synthetic registries + synthetic verified JSONL in
``tmp_path`` (no I/O on real data).
"""
from __future__ import annotations

import json
import shutil
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
    """Write a synthetic registry to *path* using FamilyRegistry.to_path."""
    reg = FamilyRegistry()
    for fid, entry in families.items():
        reg.families[fid] = entry
    reg.to_path(path)


def _write_verified_test_jsonl(
    path: Path, family_to_difficulty: dict[str, int]
) -> None:
    """Write a synthetic verified/test.jsonl mapping family_id -> difficulty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for fid, diff in family_to_difficulty.items():
        sample_id = fid.replace("mbpp_fam_", "mbpp_")
        lines.append(json.dumps({
            "sample_id": sample_id,
            "family_id": fid,
            "difficulty": diff,
            "source_split": "test",
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_script(
    registry_path: Path,
    verified_dir: Path,
    candidates_path: Path,
    output_registry_path: Path,
    *,
    count: int = 120,
    seed: int = 42,
) -> subprocess.CompletedProcess:
    """Invoke the script via subprocess. Returns the CompletedProcess."""
    return subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "reserve_frozen_v3_candidates.py"),
            "--registry", str(registry_path),
            "--mbpp-verified-dir", str(verified_dir),
            "--output-candidates", str(candidates_path),
            "--output-registry", str(output_registry_path),
            "--seed", str(seed),
            "--count", str(count),
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )


def _make_test_pool(start: int, n: int, difficulty: int = 0) -> tuple[
    dict[str, FamilyEntry], dict[str, int]
]:
    """Build n test+empty families (mbpp_fam_<start..start+n>) and a
    matching difficulty map entry."""
    fams: dict[str, FamilyEntry] = {}
    diff: dict[str, int] = {}
    for i in range(start, start + n):
        fid = f"mbpp_fam_{i}"
        fams[fid] = _entry(fid, source_split="test", usage=[])
        diff[fid] = difficulty
    return fams, diff


# ---------------------------------------------------------------------------
# Test 1: source pool filters correctly
# ---------------------------------------------------------------------------

def test_source_pool_filters_correctly(tmp_path):
    """Source pool excludes quarantined, P2, validation, train families;
    only test+empty-usage families are eligible."""
    families: dict[str, FamilyEntry] = {}
    diff_map: dict[str, int] = {}

    # 5 test+empty (eligible)
    f, d = _make_test_pool(11, 5, difficulty=0)
    families.update(f)
    diff_map.update(d)
    # 2 test+quarantine (excluded)
    for i in (21, 22):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="test", usage=["quarantine"])
    # 2 test+p2_train (excluded)
    for i in (31, 32):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="test", usage=["p2_train"])
    # 2 validation+empty (excluded by split)
    for i in (41, 42):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="validation", usage=[])
    # 2 train+empty (excluded by split)
    for i in (51, 52):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="train", usage=[])

    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)

    verified_dir = tmp_path / "verified"
    _write_verified_test_jsonl(verified_dir / "test.jsonl", diff_map)

    candidates_path = tmp_path / "candidates.json"
    result = _run_script(
        reg_path, verified_dir, candidates_path, reg_path, count=5,
    )
    assert result.returncode == 0, (
        f"script failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    with candidates_path.open(encoding="utf-8") as fh:
        cdata = json.load(fh)
    # Only the 5 test+empty families are in the source pool.
    assert cdata["source_pool_size"] == 5
    assert cdata["candidate_count"] == 5
    expected = {f"mbpp_fam_{i}" for i in range(11, 16)}
    assert set(cdata["candidates"]) == expected


# ---------------------------------------------------------------------------
# Test 2: candidate count == 120
# ---------------------------------------------------------------------------

def test_candidate_count_120(tmp_path):
    """Synthetic pool of 200 test families, sampling 120 -> exactly 120."""
    families, diff_map = _make_test_pool(11, 200, difficulty=0)
    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)
    verified_dir = tmp_path / "verified"
    _write_verified_test_jsonl(verified_dir / "test.jsonl", diff_map)

    candidates_path = tmp_path / "candidates.json"
    result = _run_script(
        reg_path, verified_dir, candidates_path, reg_path, count=120,
    )
    assert result.returncode == 0, (
        f"script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    with candidates_path.open(encoding="utf-8") as fh:
        cdata = json.load(fh)
    assert cdata["candidate_count"] == 120
    assert len(cdata["candidates"]) == 120


# ---------------------------------------------------------------------------
# Test 3: seed determinism
# ---------------------------------------------------------------------------

def test_seed_determinism(tmp_path):
    """Same pool, same seed -> identical candidate list across two runs."""
    families, diff_map = _make_test_pool(11, 200, difficulty=0)

    # Two independent registry copies so claims don't interfere.
    reg1 = tmp_path / "registry1.json"
    reg2 = tmp_path / "registry2.json"
    _write_registry(reg1, families)
    _write_registry(reg2, families)

    verified_dir = tmp_path / "verified"
    _write_verified_test_jsonl(verified_dir / "test.jsonl", diff_map)

    c1 = tmp_path / "candidates1.json"
    c2 = tmp_path / "candidates2.json"
    r1 = _run_script(reg1, verified_dir, c1, reg1, count=120, seed=42)
    r2 = _run_script(reg2, verified_dir, c2, reg2, count=120, seed=42)
    assert r1.returncode == 0, f"run1:\n{r1.stderr}"
    assert r2.returncode == 0, f"run2:\n{r2.stderr}"

    with c1.open(encoding="utf-8") as fh:
        d1 = json.load(fh)
    with c2.open(encoding="utf-8") as fh:
        d2 = json.load(fh)
    assert d1["candidates"] == d2["candidates"]


# ---------------------------------------------------------------------------
# Test 4: stratification proportional
# ---------------------------------------------------------------------------

def test_stratification_proportional(tmp_path):
    """100 difficulty-0 + 50 difficulty-1 + 50 difficulty-2 (total 200)
    -> 60 + 30 + 30 allocation."""
    families: dict[str, FamilyEntry] = {}
    diff_map: dict[str, int] = {}
    # 100 difficulty-0: mbpp_fam_11..110
    f, d = _make_test_pool(11, 100, difficulty=0)
    families.update(f)
    diff_map.update(d)
    # 50 difficulty-1: mbpp_fam_111..160
    f, d = _make_test_pool(111, 50, difficulty=1)
    families.update(f)
    diff_map.update(d)
    # 50 difficulty-2: mbpp_fam_161..210
    f, d = _make_test_pool(161, 50, difficulty=2)
    families.update(f)
    diff_map.update(d)

    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)
    verified_dir = tmp_path / "verified"
    _write_verified_test_jsonl(verified_dir / "test.jsonl", diff_map)

    candidates_path = tmp_path / "candidates.json"
    result = _run_script(
        reg_path, verified_dir, candidates_path, reg_path, count=120,
    )
    assert result.returncode == 0, (
        f"script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    with candidates_path.open(encoding="utf-8") as fh:
        cdata = json.load(fh)
    buckets = cdata["stratification"]["buckets"]
    assert buckets["0"] == {"pool_size": 100, "allocated": 60}
    assert buckets["1"] == {"pool_size": 50, "allocated": 30}
    assert buckets["2"] == {"pool_size": 50, "allocated": 30}
    assert buckets["3"] == {"pool_size": 0, "allocated": 0}
    assert buckets["4"] == {"pool_size": 0, "allocated": 0}
    # Sum of allocated == 120; sum of pool_size == source_pool_size.
    total_alloc = sum(buckets[str(d)]["allocated"] for d in range(5))
    total_pool = sum(buckets[str(d)]["pool_size"] for d in range(5))
    assert total_alloc == 120
    assert total_pool == cdata["source_pool_size"] == 200


# ---------------------------------------------------------------------------
# Test 5: candidates sorted
# ---------------------------------------------------------------------------

def test_candidates_sorted(tmp_path):
    """candidates list is sorted ascending by family_id."""
    families, diff_map = _make_test_pool(11, 200, difficulty=0)
    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)
    verified_dir = tmp_path / "verified"
    _write_verified_test_jsonl(verified_dir / "test.jsonl", diff_map)

    candidates_path = tmp_path / "candidates.json"
    result = _run_script(
        reg_path, verified_dir, candidates_path, reg_path, count=120,
    )
    assert result.returncode == 0, (
        f"script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    with candidates_path.open(encoding="utf-8") as fh:
        cdata = json.load(fh)
    cands = cdata["candidates"]
    assert cands == sorted(cands)


# ---------------------------------------------------------------------------
# Test 6: no candidate in quarantine
# ---------------------------------------------------------------------------

def test_no_candidate_in_quarantine(tmp_path):
    """After claiming, no candidate has quarantine tag in the registry."""
    families: dict[str, FamilyEntry] = {}
    diff_map: dict[str, int] = {}
    # 200 eligible test+empty families
    f, d = _make_test_pool(11, 200, difficulty=0)
    families.update(f)
    diff_map.update(d)
    # 5 test+quarantine families (excluded from source pool)
    for i in (211, 212, 213, 214, 215):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="test", usage=["quarantine"])

    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)
    verified_dir = tmp_path / "verified"
    _write_verified_test_jsonl(verified_dir / "test.jsonl", diff_map)

    candidates_path = tmp_path / "candidates.json"
    result = _run_script(
        reg_path, verified_dir, candidates_path, reg_path, count=120,
    )
    assert result.returncode == 0, (
        f"script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    with candidates_path.open(encoding="utf-8") as fh:
        cdata = json.load(fh)
    reloaded = FamilyRegistry.from_path(reg_path)
    for fid in cdata["candidates"]:
        entry = reloaded.get(fid)
        assert entry is not None
        assert "quarantine" not in entry.usage, (
            f"candidate {fid} has quarantine tag"
        )


# ---------------------------------------------------------------------------
# Test 7: no candidate in p2
# ---------------------------------------------------------------------------

def test_no_candidate_in_p2(tmp_path):
    """After claiming, no candidate has any P2 tag."""
    families: dict[str, FamilyEntry] = {}
    diff_map: dict[str, int] = {}
    # 200 eligible test+empty families
    f, d = _make_test_pool(11, 200, difficulty=0)
    families.update(f)
    diff_map.update(d)
    # P2-tagged test families (excluded from source pool)
    for i, tag in ((211, "p2_train"), (212, "p2_validation"),
                   (213, "p2_frozen_v2")):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(fid, source_split="test", usage=[tag])

    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)
    verified_dir = tmp_path / "verified"
    _write_verified_test_jsonl(verified_dir / "test.jsonl", diff_map)

    candidates_path = tmp_path / "candidates.json"
    result = _run_script(
        reg_path, verified_dir, candidates_path, reg_path, count=120,
    )
    assert result.returncode == 0, (
        f"script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    with candidates_path.open(encoding="utf-8") as fh:
        cdata = json.load(fh)
    reloaded = FamilyRegistry.from_path(reg_path)
    for fid in cdata["candidates"]:
        entry = reloaded.get(fid)
        assert entry is not None
        for tag in ("p2_train", "p2_validation", "p2_frozen_v2"):
            assert tag not in entry.usage, (
                f"candidate {fid} has {tag} tag"
            )


# ---------------------------------------------------------------------------
# Test 8: registry claim persists
# ---------------------------------------------------------------------------

def test_registry_claim_persists(tmp_path):
    """After running, re-loading the registry shows exactly the 120
    candidates have the frozen_v3_candidate tag."""
    families, diff_map = _make_test_pool(11, 200, difficulty=0)
    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)
    verified_dir = tmp_path / "verified"
    _write_verified_test_jsonl(verified_dir / "test.jsonl", diff_map)

    candidates_path = tmp_path / "candidates.json"
    result = _run_script(
        reg_path, verified_dir, candidates_path, reg_path, count=120,
    )
    assert result.returncode == 0, (
        f"script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    with candidates_path.open(encoding="utf-8") as fh:
        cdata = json.load(fh)
    reloaded = FamilyRegistry.from_path(reg_path)
    claimed = reloaded.families_with_usage("frozen_v3_candidate")
    assert sorted(claimed) == sorted(cdata["candidates"])


# ---------------------------------------------------------------------------
# Test 9: pairwise disjoint holds
# ---------------------------------------------------------------------------

def test_pairwise_disjoint_holds(tmp_path):
    """assert_pairwise_disjoint over [p2_*, frozen_v3_candidate, quarantine]
    does not raise after claiming (P2+quarantine overlap whitelisted)."""
    families: dict[str, FamilyEntry] = {}
    diff_map: dict[str, int] = {}
    # 200 eligible test+empty families
    f, d = _make_test_pool(11, 200, difficulty=0)
    families.update(f)
    diff_map.update(d)
    # P2+quarantine overlap families (Task 6 produces these): train split.
    for i in (211, 212):
        fid = f"mbpp_fam_{i}"
        families[fid] = _entry(
            fid, source_split="train",
            usage=["p2_train", "quarantine"],
        )
    # A pure p2_train (no quarantine) family.
    families["mbpp_fam_213"] = _entry(
        "mbpp_fam_213", source_split="train", usage=["p2_train"],
    )
    # A pure quarantine test family (excluded from candidates by filter).
    families["mbpp_fam_214"] = _entry(
        "mbpp_fam_214", source_split="test", usage=["quarantine"],
    )

    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)
    verified_dir = tmp_path / "verified"
    _write_verified_test_jsonl(verified_dir / "test.jsonl", diff_map)

    candidates_path = tmp_path / "candidates.json"
    result = _run_script(
        reg_path, verified_dir, candidates_path, reg_path, count=120,
    )
    assert result.returncode == 0, (
        f"script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    reloaded = FamilyRegistry.from_path(reg_path)
    # Should NOT raise (P2+quarantine overlap is whitelisted).
    reloaded.assert_pairwise_disjoint(
        ["p2_train", "p2_validation", "p2_frozen_v2",
         "frozen_v3_candidate", "quarantine"],
        whitelist=[
            ("p2_train", "quarantine"),
            ("p2_validation", "quarantine"),
            ("p2_frozen_v2", "quarantine"),
        ],
    )


# ---------------------------------------------------------------------------
# Test 10: insufficient pool aborts
# ---------------------------------------------------------------------------

def test_insufficient_pool_aborts(tmp_path):
    """Pool with only 100 test families (need 120) -> exit 1, clear
    error message, NO partial candidates.json written."""
    families, diff_map = _make_test_pool(11, 100, difficulty=0)
    reg_path = tmp_path / "registry.json"
    _write_registry(reg_path, families)
    verified_dir = tmp_path / "verified"
    _write_verified_test_jsonl(verified_dir / "test.jsonl", diff_map)

    candidates_path = tmp_path / "candidates.json"
    result = _run_script(
        reg_path, verified_dir, candidates_path, reg_path, count=120,
    )
    assert result.returncode == 1, (
        f"expected exit 1, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "FIX_FIRST" in result.stderr, (
        f"expected FIX_FIRST in stderr, got: {result.stderr!r}"
    )
    # No partial candidates.json should exist.
    assert not candidates_path.exists(), (
        f"candidates.json should not exist on abort, found {candidates_path}"
    )
