"""Tests for src/dataset_builder.py — TDD coverage for dataset building.

Covers:
- dedup removes content-identical samples (same content / different sample_id)
  and keeps distinct ones; stable order; first-seen wins.
- LEAK-PROOF split: family-id sets of train/val/test are pairwise DISJOINT.
  This is the safety-critical test: even a single family appearing in two
  splits is a hard failure.
- heldout families never appear in train/val/test.
- split approximates by-family fractions (within tolerance), deterministic.
- to_chatml_records produces well-formed 3-message records.
- write_jsonl, dataset_hash, write_split behave as specified.
"""
import json
import warnings
from datetime import datetime
from pathlib import Path

import pytest

from src.dataset_builder import (
    DatasetSplit,
    dataset_hash,
    dedup,
    split_by_family,
    to_chatml_records,
    write_jsonl,
    write_split,
)
from src.schemas import Sample


# ---------------------------------------------------------------------------
# Helpers — self-contained, no cross-test-module imports
# ---------------------------------------------------------------------------

def _verification() -> dict:
    return {"syntax_ok": True, "pytest_ok": True, "ruff_ok": True, "timeout": False}


def _make_sample(
    sample_id: str,
    family_id: str,
    difficulty: int = 0,
    instruction: str = "写一个测试函数。",
    target_code: str = "def foo(): pass",
) -> Sample:
    """Return a minimal valid code_generation Sample."""
    return Sample(
        sample_id=sample_id,
        family_id=family_id,
        difficulty=difficulty,
        task_type="code_generation",
        language="python",
        skill_tags=["basics"],
        instruction=instruction,
        broken_code=None,
        execution_feedback=None,
        target_code=target_code,
        public_tests="def test_foo():\n    foo()",
        hidden_tests="",
        verified=True,
        verification=_verification(),
        generator="test",
        created_at="2026-01-01T00:00:00Z",
        dataset_version="v1",
    )


def _make_family(
    family_id: str,
    n: int,
    difficulty: int = 0,
) -> list[Sample]:
    """Return *n* samples all belonging to *family_id*."""
    return [
        _make_sample(
            sample_id=f"{family_id}_s{i}",
            family_id=family_id,
            difficulty=difficulty,
        )
        for i in range(n)
    ]


def _large_pool(n_families: int = 20, samples_per_family: int = 3) -> list[Sample]:
    """Create a pool with *n_families* families, each having *samples_per_family* samples.

    Difficulty cycles through 0..3 per family (so the pool has varied
    difficulties, useful for mix_report tests).
    """
    samples: list[Sample] = []
    for i in range(n_families):
        fid = f"family_{i:03d}"
        samples.extend(_make_family(fid, samples_per_family, difficulty=i % 4))
    return samples


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------

class TestDedup:
    def test_removes_exact_content_duplicate(self):
        """Two samples with identical content but different sample_id → keep first."""
        s1 = _make_sample("id_1", "fam_a", instruction="同一指令", target_code="def f(): pass")
        s2 = _make_sample("id_2", "fam_a", instruction="同一指令", target_code="def f(): pass")
        result = dedup([s1, s2])
        assert len(result) == 1
        assert result[0].sample_id == "id_1"

    def test_keeps_distinct_samples(self):
        s1 = _make_sample("id_1", "fam_a", instruction="指令 A", target_code="def a(): pass")
        s2 = _make_sample("id_2", "fam_b", instruction="指令 B", target_code="def b(): pass")
        result = dedup([s1, s2])
        assert len(result) == 2

    def test_empty_list(self):
        assert dedup([]) == []

    def test_stable_insertion_order(self):
        """First-seen wins; original order is preserved for uniques."""
        samples = [
            _make_sample(f"id_{i}", "fam_a", instruction=f"instr_{i}", target_code=f"def f{i}(): pass")
            for i in range(5)
        ]
        result = dedup(samples)
        assert [s.sample_id for s in result] == ["id_0", "id_1", "id_2", "id_3", "id_4"]

    def test_three_duplicates_keeps_only_first(self):
        kw = {"instruction": "dup instr", "target_code": "def x(): return 1"}
        s1 = _make_sample("a", "fam", **kw)
        s2 = _make_sample("b", "fam", **kw)
        s3 = _make_sample("c", "fam", **kw)
        result = dedup([s1, s2, s3])
        assert len(result) == 1
        assert result[0].sample_id == "a"

    def test_different_target_code_is_not_duplicate(self):
        s1 = _make_sample("id_1", "fam", instruction="same", target_code="def a(): return 1")
        s2 = _make_sample("id_2", "fam", instruction="same", target_code="def a(): return 2")
        result = dedup([s1, s2])
        assert len(result) == 2

    def test_different_instruction_is_not_duplicate(self):
        s1 = _make_sample("id_1", "fam", instruction="instr A", target_code="def f(): pass")
        s2 = _make_sample("id_2", "fam", instruction="instr B", target_code="def f(): pass")
        result = dedup([s1, s2])
        assert len(result) == 2

    def test_single_sample_unchanged(self):
        s = _make_sample("only", "fam")
        result = dedup([s])
        assert len(result) == 1
        assert result[0].sample_id == "only"

    def test_none_and_empty_broken_code_are_duplicates(self):
        """broken_code=None vs broken_code="" (all else identical) → ONE kept.

        Both samples are code_generation, so empty broken_code is permitted by
        the schema; the content hash must normalise "" → None.
        """
        s1 = _make_sample("id_1", "fam", instruction="同一指令", target_code="def f(): pass")
        s1.broken_code = None
        s2 = _make_sample("id_2", "fam", instruction="同一指令", target_code="def f(): pass")
        s2.broken_code = ""
        result = dedup([s1, s2])
        assert len(result) == 1
        assert result[0].sample_id == "id_1"

    def test_none_and_empty_feedback_are_duplicates(self):
        """execution_feedback=None vs "" (all else identical) → ONE kept."""
        s1 = _make_sample("id_1", "fam", instruction="指令", target_code="def f(): pass")
        s1.execution_feedback = None
        s2 = _make_sample("id_2", "fam", instruction="指令", target_code="def f(): pass")
        s2.execution_feedback = ""
        result = dedup([s1, s2])
        assert len(result) == 1
        assert result[0].sample_id == "id_1"


# ---------------------------------------------------------------------------
# split_by_family — LEAK-PROOF guarantee
# ---------------------------------------------------------------------------

class TestSplitByFamily:

    def _pool_10_families(self) -> list[Sample]:
        """10 families × 3 samples = 30 total."""
        samples: list[Sample] = []
        for i in range(10):
            samples.extend(_make_family(f"fam_{i:02d}", 3))
        return samples

    # ---- CORE LEAK-PROOF TESTS ----

    def test_train_val_disjoint(self):
        """CRITICAL: train and val must share NO family_ids."""
        pool = self._pool_10_families()
        split = split_by_family(pool)
        train_fids = {s.family_id for s in split.train}
        val_fids   = {s.family_id for s in split.val}
        assert train_fids.isdisjoint(val_fids), (
            f"LEAK DETECTED: families in both train and val: "
            f"{train_fids & val_fids}"
        )

    def test_train_test_disjoint(self):
        """CRITICAL: train and test must share NO family_ids."""
        pool = self._pool_10_families()
        split = split_by_family(pool)
        train_fids = {s.family_id for s in split.train}
        test_fids  = {s.family_id for s in split.test}
        assert train_fids.isdisjoint(test_fids), (
            f"LEAK DETECTED: families in both train and test: "
            f"{train_fids & test_fids}"
        )

    def test_val_test_disjoint(self):
        """CRITICAL: val and test must share NO family_ids."""
        pool = self._pool_10_families()
        split = split_by_family(pool)
        val_fids  = {s.family_id for s in split.val}
        test_fids = {s.family_id for s in split.test}
        assert val_fids.isdisjoint(test_fids), (
            f"LEAK DETECTED: families in both val and test: "
            f"{val_fids & test_fids}"
        )

    def test_all_three_pairwise_disjoint_large_pool(self):
        """Comprehensive disjointness on a larger pool."""
        pool = _large_pool(30, 3)
        split = split_by_family(pool)
        train_fids = {s.family_id for s in split.train}
        val_fids   = {s.family_id for s in split.val}
        test_fids  = {s.family_id for s in split.test}
        assert train_fids.isdisjoint(val_fids),  "train ∩ val ≠ ∅"
        assert train_fids.isdisjoint(test_fids), "train ∩ test ≠ ∅"
        assert val_fids.isdisjoint(test_fids),   "val ∩ test ≠ ∅"

    # ---- HELDOUT ISOLATION ----

    def test_heldout_absent_from_train(self):
        pool = _large_pool(20, 3)
        held = {"family_000", "family_001"}
        split = split_by_family(pool, heldout_family_ids=held)
        assert {s.family_id for s in split.train}.isdisjoint(held)

    def test_heldout_absent_from_val(self):
        pool = _large_pool(20, 3)
        held = {"family_000", "family_001"}
        split = split_by_family(pool, heldout_family_ids=held)
        assert {s.family_id for s in split.val}.isdisjoint(held)

    def test_heldout_absent_from_test(self):
        pool = _large_pool(20, 3)
        held = {"family_000", "family_001"}
        split = split_by_family(pool, heldout_family_ids=held)
        assert {s.family_id for s in split.test}.isdisjoint(held)

    def test_heldout_contains_all_specified_families(self):
        pool = _large_pool(20, 3)
        held = {"family_000", "family_001"}
        split = split_by_family(pool, heldout_family_ids=held)
        heldout_fids = {s.family_id for s in split.heldout}
        assert heldout_fids == held

    # ---- COMPLETENESS (no samples lost or duplicated) ----

    def test_no_samples_lost(self):
        pool = _large_pool(20, 3)
        split = split_by_family(pool)
        all_ids = {s.sample_id for s in split.train + split.val + split.test + split.heldout}
        assert all_ids == {s.sample_id for s in pool}

    def test_no_samples_duplicated(self):
        pool = _large_pool(20, 3)
        split = split_by_family(pool)
        all_ids = [s.sample_id for s in split.train + split.val + split.test + split.heldout]
        assert len(all_ids) == len(set(all_ids)), "Some samples appear in multiple splits"

    # ---- FRACTION APPROXIMATION ----

    def test_approximates_fractions_by_family_count(self):
        """With 100 families, splits should be within 5% of target fractions."""
        pool = _large_pool(100, 3)
        split = split_by_family(pool, train=0.70, val=0.10, test=0.20, seed=42)
        n = 100
        train_fam = len({s.family_id for s in split.train})
        val_fam   = len({s.family_id for s in split.val})
        test_fam  = len({s.family_id for s in split.test})
        assert abs(train_fam / n - 0.70) < 0.05
        assert abs(val_fam   / n - 0.10) < 0.05
        assert abs(test_fam  / n - 0.20) < 0.05

    # ---- SMALL-N EMPTY-SPLIT WARNING ----

    def test_small_n_empty_val_warns(self):
        """n=8 active families with val=0.10 rounds val down to 0 → warn."""
        pool = _large_pool(8, 2)  # 8 families, none heldout
        with pytest.warns(UserWarning, match="val.*EMPTY"):
            split = split_by_family(pool, seed=42)
        # The empty val is still produced (algorithm unchanged); only a warning.
        assert len({s.family_id for s in split.val}) == 0
        # Disjointness must still hold.
        train_fids = {s.family_id for s in split.train}
        test_fids  = {s.family_id for s in split.test}
        assert train_fids.isdisjoint(test_fids)

    def test_sufficient_n_does_not_warn(self):
        """With 100 families, all splits populated → no warning."""
        pool = _large_pool(100, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            split_by_family(pool, seed=42)

    # ---- DETERMINISM ----

    def test_deterministic_given_seed(self):
        pool = _large_pool(20, 3)
        s1 = split_by_family(pool, seed=42)
        s2 = split_by_family(pool, seed=42)
        assert [s.sample_id for s in s1.train] == [s.sample_id for s in s2.train]
        assert [s.sample_id for s in s1.val]   == [s.sample_id for s in s2.val]
        assert [s.sample_id for s in s1.test]  == [s.sample_id for s in s2.test]

    def test_different_seeds_produce_different_splits(self):
        pool = _large_pool(30, 3)
        s1 = split_by_family(pool, seed=42)
        s2 = split_by_family(pool, seed=99)
        fids1 = {s.family_id for s in s1.train}
        fids2 = {s.family_id for s in s2.train}
        assert fids1 != fids2, "Expected different seeds to produce different train splits"

    # ---- INPUT VALIDATION ----

    def test_invalid_fractions_sum_raises(self):
        pool = _large_pool(10, 2)
        with pytest.raises(ValueError):
            split_by_family(pool, train=0.5, val=0.3, test=0.3)  # sums to 1.1

    # ---- family_ids() HELPER ----

    def test_family_ids_helper_returns_sets(self):
        pool = _large_pool(10, 3)
        split = split_by_family(pool)
        for name in ("train", "val", "test", "heldout"):
            result = split.family_ids(name)
            assert isinstance(result, set), f"family_ids({name!r}) should return a set"

    def test_family_ids_helper_invalid_raises(self):
        pool = _large_pool(12, 2)  # >=10 families so no small-n warning leaks
        split = split_by_family(pool)
        with pytest.raises(ValueError):
            split.family_ids("unknown")


# ---------------------------------------------------------------------------
# to_chatml_records
# ---------------------------------------------------------------------------

class TestToChatmlRecords:
    def test_returns_list_of_dicts(self):
        pool = _make_family("fam", 3)
        records = to_chatml_records(pool)
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)

    def test_length_equals_input(self):
        pool = _make_family("fam", 5)
        assert len(to_chatml_records(pool)) == 5

    def test_each_record_has_messages_key(self):
        pool = _make_family("fam", 3)
        for r in to_chatml_records(pool):
            assert "messages" in r

    def test_each_record_has_exactly_3_messages(self):
        pool = _make_family("fam", 3)
        for r in to_chatml_records(pool):
            assert len(r["messages"]) == 3

    def test_message_roles_system_user_assistant(self):
        pool = _make_family("fam", 2)
        for r in to_chatml_records(pool):
            roles = [m["role"] for m in r["messages"]]
            assert roles == ["system", "user", "assistant"]

    def test_empty_input(self):
        assert to_chatml_records([]) == []


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------

class TestWriteJsonl:
    def test_file_is_created(self, tmp_path: Path):
        write_jsonl([{"a": 1}], tmp_path / "out.jsonl")
        assert (tmp_path / "out.jsonl").exists()

    def test_one_line_per_record(self, tmp_path: Path):
        records = [{"x": 1}, {"x": 2}, {"x": 3}]
        path = tmp_path / "out.jsonl"
        write_jsonl(records, path)
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, tmp_path: Path):
        records = [{"messages": [{"role": "system"}]}, {"y": 42}]
        path = tmp_path / "out.jsonl"
        write_jsonl(records, path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parsed = json.loads(line)
                assert isinstance(parsed, dict)

    def test_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "out.jsonl"
        write_jsonl([{"k": "v"}], nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# dataset_hash
# ---------------------------------------------------------------------------

class TestDatasetHash:
    def test_stable_for_same_input(self):
        records = [{"a": 1}, {"b": 2}]
        assert dataset_hash(records) == dataset_hash(records)

    def test_order_independent(self):
        r1 = [{"a": 1}, {"b": 2}]
        r2 = [{"b": 2}, {"a": 1}]
        assert dataset_hash(r1) == dataset_hash(r2)

    def test_changes_for_different_content(self):
        assert dataset_hash([{"a": 1}]) != dataset_hash([{"a": 2}])

    def test_empty_list_returns_sha256_hex(self):
        h = dataset_hash([])
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_more_records_changes_hash(self):
        """Same records but different count → different hash."""
        r = [{"msg": "hello"}]
        assert dataset_hash(r) != dataset_hash(r * 2)


# ---------------------------------------------------------------------------
# write_split
# ---------------------------------------------------------------------------

class TestWriteSplit:

    def _make_split(self, n_families: int = 20) -> DatasetSplit:
        return split_by_family(_large_pool(n_families, 3), seed=42)

    def test_writes_train_file(self, tmp_path: Path):
        write_split(self._make_split(), tmp_path)
        assert (tmp_path / "train.jsonl").exists()

    def test_writes_validation_file(self, tmp_path: Path):
        write_split(self._make_split(), tmp_path)
        assert (tmp_path / "validation.jsonl").exists()

    def test_writes_test_file(self, tmp_path: Path):
        write_split(self._make_split(), tmp_path)
        assert (tmp_path / "test.jsonl").exists()

    def test_manifest_top_level_keys(self, tmp_path: Path):
        manifest = write_split(self._make_split(), tmp_path)
        for key in ("train", "validation", "test", "heldout",
                    "dataset_hash", "seed", "created_at"):
            assert key in manifest, f"Manifest missing key {key!r}"

    def test_manifest_split_sub_keys(self, tmp_path: Path):
        manifest = write_split(self._make_split(), tmp_path)
        for split_name in ("train", "validation", "test", "heldout"):
            sub = manifest[split_name]
            assert "sample_count"   in sub, f"{split_name}: missing sample_count"
            assert "difficulty_mix" in sub, f"{split_name}: missing difficulty_mix"
            assert "family_count"   in sub, f"{split_name}: missing family_count"

    def test_manifest_counts_match_split(self, tmp_path: Path):
        split = self._make_split()
        manifest = write_split(split, tmp_path)
        assert manifest["train"]["sample_count"]      == len(split.train)
        assert manifest["validation"]["sample_count"] == len(split.val)
        assert manifest["test"]["sample_count"]       == len(split.test)
        assert manifest["heldout"]["sample_count"]    == len(split.heldout)

    def test_file_line_count_matches_manifest(self, tmp_path: Path):
        split = self._make_split()
        manifest = write_split(split, tmp_path)
        for split_name in ("train", "validation", "test"):
            path = tmp_path / f"{split_name}.jsonl"
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            expected = manifest[split_name]["sample_count"]
            assert len(lines) == expected, (
                f"{split_name}: {len(lines)} lines in file but manifest says {expected}"
            )

    def test_manifest_dataset_hash_is_sha256(self, tmp_path: Path):
        manifest = write_split(self._make_split(), tmp_path)
        h = manifest["dataset_hash"]
        assert isinstance(h, str) and len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_dataset_hash_stable_same_data(self, tmp_path: Path):
        split = self._make_split()
        m1 = write_split(split, tmp_path / "run1")
        m2 = write_split(split, tmp_path / "run2")
        assert m1["dataset_hash"] == m2["dataset_hash"]

    def test_dataset_hash_changes_for_different_data(self, tmp_path: Path):
        m1 = write_split(split_by_family(_large_pool(10, 3), seed=42), tmp_path / "r1")
        m2 = write_split(split_by_family(_large_pool(15, 3), seed=42), tmp_path / "r2")
        assert m1["dataset_hash"] != m2["dataset_hash"]

    def test_manifest_created_at_is_iso8601(self, tmp_path: Path):
        manifest = write_split(self._make_split(), tmp_path)
        dt = datetime.fromisoformat(manifest["created_at"])
        assert dt is not None

    def test_manifest_seed_recorded(self, tmp_path: Path):
        manifest = write_split(self._make_split(), tmp_path, seed=777)
        assert manifest["seed"] == 777

    def test_heldout_file_written_when_nonempty(self, tmp_path: Path):
        pool = _large_pool(20, 3)
        held = {"family_000", "family_001"}
        split = split_by_family(pool, heldout_family_ids=held)
        write_split(split, tmp_path)
        assert (tmp_path / "heldout.jsonl").exists()

    def test_heldout_file_not_written_when_empty(self, tmp_path: Path):
        split = self._make_split()
        write_split(split, tmp_path)
        assert not (tmp_path / "heldout.jsonl").exists()
