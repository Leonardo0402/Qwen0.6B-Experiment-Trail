"""tests/test_p3_formal_pool_builder.py -- Tests for the Formal Canonical Pool
v2 builder (Issue #14 Wave 4-G1).

Covers:
  1. Sharding logic (25 families/shard, 17 shards for 425 families).
  2. Resumable execution (already-completed families are skipped).
  3. Deduplication (sample_id, instruction, target_code, broken_code,
     test_suite, AST structural hash).
  4. Per-family / per-bucket caps (8/family, 3/bucket).
  5. Output format (shard layout, manifest schema, verified/rejected files).

The shard-000 artifacts produced by G1.3 are used as the test fixture where
real data is needed; pure-logic tests use synthetic inputs.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample, Verification  # noqa: E402
from scripts.p3_formal_pool_builder import (  # noqa: E402
    BUCKETS,
    DEFAULT_SHARD_SIZE,
    DedupState,
    FORMAL_MANIFEST_PATH,
    FORMAL_POOL_PATH,
    FORMAL_ROOT,
    MAX_BOUNDARY_PER_FAMILY,
    MAX_CODE_PER_FAMILY,
    MAX_EXEC_PER_FAMILY,
    MAX_STATIC_PER_FAMILY,
    MAX_TOTAL_PER_FAMILY,
    SHARDS_ROOT,
    TOTAL_SHARED_FAMILIES,
    _ast_structural_hash,
    _load_cache_completed,
    _normalized_instruction,
    _sha256,
    consolidate_pool,
    load_train_families,
    shard_family_ids,
    total_shards,
)


# ---------------------------------------------------------------------------
# 1. Sharding logic
# ---------------------------------------------------------------------------

class TestSharding:
    """Shard partition: 25 families per shard, 17 shards for 425 families."""

    def test_total_train_families_is_425(self):
        """The shared train partition must contain exactly 425 families."""
        fams = load_train_families()
        assert len(fams) == TOTAL_SHARED_FAMILIES == 425

    def test_total_shards_25_per_shard(self):
        """ceil(425 / 25) == 17 shards."""
        assert total_shards(425, 25) == 17

    def test_shard_0_has_25_families(self):
        fams = load_train_families()
        shard0 = shard_family_ids(fams, 0, 25)
        assert len(shard0) == 25

    def test_last_shard_has_remainder(self):
        """425 = 17 * 25 exactly, so the last shard is also full (25)."""
        fams = load_train_families()
        last = shard_family_ids(fams, 16, 25)
        assert len(last) == 25

    def test_shards_are_disjoint(self):
        """No family appears in two shards."""
        fams = load_train_families()
        seen: set[str] = set()
        for i in range(total_shards(len(fams), 25)):
            shard = shard_family_ids(fams, i, 25)
            for fid in shard:
                assert fid not in seen, (
                    f"family {fid} appears in multiple shards"
                )
                seen.add(fid)
        assert len(seen) == len(fams)

    def test_shard_index_out_of_range_returns_empty(self):
        fams = load_train_families()
        assert shard_family_ids(fams, 999, 25) == []

    def test_shards_are_sorted_within(self):
        """Each shard's family list preserves the sorted global order."""
        fams = load_train_families()
        for i in range(total_shards(len(fams), 25)):
            shard = shard_family_ids(fams, i, 25)
            assert shard == sorted(shard), (
                f"shard {i} is not sorted: {shard[:3]}..."
            )


# ---------------------------------------------------------------------------
# 2. Dedup logic
# ---------------------------------------------------------------------------

class TestDedupState:
    """Multi-dimensional dedup tracker."""

    @staticmethod
    def _make_sample(
        sample_id: str = "s1",
        family_id: str = "fam1",
        instruction: str = "Write a function.",
        target_code: str = "def f():\n    return 1\n",
        public_tests: str = "assert f() == 1",
        hidden_tests: str = "assert f() == 1",
        broken_code: str | None = None,
    ) -> Sample:
        return Sample(
            sample_id=sample_id,
            family_id=family_id,
            difficulty=1,
            task_type="code_generation",
            language="python",
            skill_tags=["test"],
            instruction=instruction,
            broken_code=broken_code,
            execution_feedback=None,
            target_code=target_code,
            public_tests=public_tests,
            hidden_tests=hidden_tests,
            verified=True,
            verification=Verification(
                syntax_ok=True, pytest_ok=True, ruff_ok=False, timeout=False
            ),
            generator="test",
            created_at="2026-07-05T00:00:00+00:00",
            dataset_version="test-v1",
        )

    def test_first_sample_is_not_duplicate(self):
        d = DedupState()
        s = self._make_sample()
        assert d.is_duplicate(s) is None

    def test_same_sample_id_is_duplicate(self):
        d = DedupState()
        s1 = self._make_sample(sample_id="s1")
        s2 = self._make_sample(sample_id="s1", target_code="def g():\n    return 2\n")
        d.add(s1)
        reason = d.is_duplicate(s2)
        assert reason is not None
        assert "sample_id" in reason

    def test_same_target_code_different_instruction_not_duplicate(self):
        """Within-family code variants share target_code but have different
        instructions (orig vs signature-scaffold vs public-examples). They
        must NOT be deduped."""
        d = DedupState()
        s1 = self._make_sample(
            sample_id="s1", instruction="Write a function.",
        )
        s2 = self._make_sample(
            sample_id="s2",
            instruction="Write a function.\n\nFunction signature: def f():",
            target_code=s1.target_code,  # same target_code
        )
        d.add(s1)
        assert d.is_duplicate(s2) is None

    def test_same_instruction_different_target_not_duplicate(self):
        """Repair variants share the instruction but have different
        target_code (in principle). Must NOT be deduped."""
        d = DedupState()
        s1 = self._make_sample(sample_id="s1")
        s2 = self._make_sample(
            sample_id="s2",
            target_code="def g():\n    return 2\n",
        )
        d.add(s1)
        assert d.is_duplicate(s2) is None

    def test_full_content_signature_duplicate(self):
        """Same (instruction, target_code, broken_code, tests) → duplicate."""
        d = DedupState()
        s1 = self._make_sample(sample_id="s1")
        s2 = self._make_sample(sample_id="s2")  # identical content
        d.add(s1)
        reason = d.is_duplicate(s2)
        assert reason is not None
        assert "full content signature" in reason

    def test_structural_duplicate_same_ast_same_tests(self):
        """Same AST structure + same test suite + same instruction →
        duplicate (trivial whitespace-only differences in target_code are
        caught by the AST hash normalization)."""
        d = DedupState()
        # Same instruction is required: struct_sig now includes instr_h so
        # that two samples with the same code but different prompts are NOT
        # deduped (they are different tasks). The whitespace-only diff in
        # target_code is normalized by the AST hash, so struct_sig matches.
        s1 = self._make_sample(
            sample_id="s1",
            target_code="def f():\n    return 1\n",
            instruction="Write f.",
        )
        s2 = self._make_sample(
            sample_id="s2",
            target_code="def f():\n        return 1\n",  # whitespace-only diff
            instruction="Write f.",  # SAME instruction
        )
        d.add(s1)
        reason = d.is_duplicate(s2)
        assert reason is not None
        assert "structural" in reason

    def test_same_ast_different_tests_not_duplicate(self):
        """Boundary variants share target_code (hence AST) but have different
        test suites. They must NOT be deduped."""
        d = DedupState()
        s1 = self._make_sample(
            sample_id="s1",
            public_tests="assert f() == 1",
        )
        s2 = self._make_sample(
            sample_id="s2",
            public_tests="assert f() == 2",  # different tests
            instruction="Different.",
        )
        d.add(s1)
        assert d.is_duplicate(s2) is None

    def test_empty_broken_code_does_not_dedup(self):
        """Samples with no broken_code must NOT trigger broken_code dedup."""
        d = DedupState()
        s1 = self._make_sample(sample_id="s1", broken_code=None)
        s2 = self._make_sample(
            sample_id="s2", broken_code=None,
            target_code="def g():\n    return 2\n",
            instruction="Different instruction entirely.",
        )
        d.add(s1)
        # s2 has different target_code AND different instruction, so no dedup
        assert d.is_duplicate(s2) is None


# ---------------------------------------------------------------------------
# 3. Per-family / per-bucket caps
# ---------------------------------------------------------------------------

class TestCaps:
    """Caps: 8/family total, 3/bucket."""

    def test_total_cap_is_8(self):
        assert MAX_TOTAL_PER_FAMILY == 8

    def test_bucket_caps_are_3(self):
        assert MAX_CODE_PER_FAMILY == 3
        assert MAX_BOUNDARY_PER_FAMILY == 3
        assert MAX_STATIC_PER_FAMILY == 3
        assert MAX_EXEC_PER_FAMILY == 3

    def test_bucket_total_at_most_12_but_capped_at_8(self):
        """4 buckets * 3 = 12 theoretical max, but the per-family cap (8)
        is the binding constraint."""
        theoretical = (
            MAX_CODE_PER_FAMILY + MAX_BOUNDARY_PER_FAMILY
            + MAX_STATIC_PER_FAMILY + MAX_EXEC_PER_FAMILY
        )
        assert theoretical == 12
        assert MAX_TOTAL_PER_FAMILY < theoretical


# ---------------------------------------------------------------------------
# 4. Cache helpers (resumable execution)
# ---------------------------------------------------------------------------

class TestResumeCache:
    """verification-cache.jsonl loading."""

    def test_load_cache_completed_returns_empty_for_missing_file(self, tmp_path):
        cache = tmp_path / "verification-cache.jsonl"
        assert _load_cache_completed(cache) == set()

    def test_load_cache_completed_returns_empty_for_empty_file(self, tmp_path):
        cache = tmp_path / "verification-cache.jsonl"
        cache.write_text("", encoding="utf-8")
        assert _load_cache_completed(cache) == set()

    def test_load_cache_completed_returns_completed_family_ids(self, tmp_path):
        cache = tmp_path / "verification-cache.jsonl"
        cache.write_text(
            json.dumps({"family_id": "fam1", "completed": True}) + "\n"
            + json.dumps({"family_id": "fam2", "completed": False}) + "\n"
            + json.dumps({"family_id": "fam3", "completed": True}) + "\n",
            encoding="utf-8",
        )
        completed = _load_cache_completed(cache)
        assert completed == {"fam1", "fam3"}

    def test_load_cache_completed_skips_invalid_lines(self, tmp_path):
        cache = tmp_path / "verification-cache.jsonl"
        cache.write_text(
            "not json\n"
            + json.dumps({"family_id": "fam1", "completed": True}) + "\n"
            + "\n",
            encoding="utf-8",
        )
        completed = _load_cache_completed(cache)
        assert completed == {"fam1"}


# ---------------------------------------------------------------------------
# 5. Hash helpers
# ---------------------------------------------------------------------------

class TestHashHelpers:
    """Hash functions for dedup."""

    def test_sha256_is_deterministic(self):
        assert _sha256("hello") == _sha256("hello")
        assert _sha256("hello") != _sha256("world")

    def test_normalized_instruction_collapses_whitespace(self):
        a = _normalized_instruction("Write  a   function.")
        b = _normalized_instruction("write a function.")
        c = _normalized_instruction("  WRITE\tA\nFUNCTION.  ")
        assert a == b == c

    def test_ast_structural_hash_ignores_whitespace(self):
        code1 = "def f():\n    return 1\n"
        code2 = "def f():\n        return 1\n"
        code3 = "def f():\n    return 2\n"
        assert _ast_structural_hash(code1) == _ast_structural_hash(code2)
        # Different return value -> different AST
        assert _ast_structural_hash(code1) != _ast_structural_hash(code3)

    def test_ast_structural_hash_handles_syntax_error(self):
        """Falls back to raw SHA-256 when code cannot be parsed."""
        bad_code = "def f( :\n    return 1\n"
        h = _ast_structural_hash(bad_code)
        assert h == _sha256(bad_code)


# ---------------------------------------------------------------------------
# 6. Shard-000 real artifacts (produced by G1.3)
# ---------------------------------------------------------------------------

SHARD0_DIR = SHARDS_ROOT / "shard-000"
SHARD0_VERIFIED = SHARD0_DIR / "verified.jsonl"
SHARD0_REJECTED = SHARD0_DIR / "rejected.jsonl"
SHARD0_CANDIDATE = SHARD0_DIR / "candidate.jsonl"
SHARD0_CACHE = SHARD0_DIR / "verification-cache.jsonl"
SHARD0_MANIFEST = SHARD0_DIR / "manifest.json"


@pytest.mark.skipif(
    not SHARD0_MANIFEST.exists(),
    reason="shard-000 must be run first: "
           "py -3.11 scripts/p3_formal_pool_builder.py --shard 0 --shard-size 25",
)
class TestShard0Artifacts:
    """Verify the real shard-000 output files exist and are well-formed."""

    def test_all_shard_artifacts_exist(self):
        for p in (
            SHARD0_VERIFIED, SHARD0_REJECTED, SHARD0_CANDIDATE,
            SHARD0_CACHE, SHARD0_MANIFEST,
        ):
            assert p.exists(), f"missing shard artifact: {p}"

    def test_manifest_has_required_fields(self):
        with SHARD0_MANIFEST.open(encoding="utf-8") as fh:
            m = json.load(fh)
        for field in (
            "schema_version", "shard_id", "shard_index", "shard_size",
            "family_ids", "started_at", "completed_at", "duration_s",
            "candidate_count", "verified_count", "rejected_count",
            "bucket_counts", "generator",
        ):
            assert field in m, f"manifest missing field: {field}"

    def test_manifest_shard_id_matches(self):
        with SHARD0_MANIFEST.open(encoding="utf-8") as fh:
            m = json.load(fh)
        assert m["shard_id"] == "shard-000"
        assert m["shard_index"] == 0

    def test_manifest_family_count_is_25(self):
        with SHARD0_MANIFEST.open(encoding="utf-8") as fh:
            m = json.load(fh)
        assert len(m["family_ids"]) == 25

    def test_manifest_bucket_counts_keys(self):
        with SHARD0_MANIFEST.open(encoding="utf-8") as fh:
            m = json.load(fh)
        assert set(m["bucket_counts"].keys()) == set(BUCKETS)

    def test_verified_count_matches_file(self):
        with SHARD0_MANIFEST.open(encoding="utf-8") as fh:
            m = json.load(fh)
        n_lines = sum(
            1 for line in SHARD0_VERIFIED.open(encoding="utf-8")
            if line.strip()
        )
        assert m["verified_count"] == n_lines, (
            f"manifest verified_count={m['verified_count']} "
            f"but verified.jsonl has {n_lines} lines"
        )

    def test_verified_samples_are_valid_samples(self):
        """Every line in verified.jsonl must parse as a Sample with verified=True."""
        with SHARD0_VERIFIED.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                s = Sample.from_json_line(line)
                assert s.verified is True
                assert s.verification.syntax_ok is True
                assert s.verification.pytest_ok is True

    def test_no_family_exceeds_8_verified(self):
        """Per-family cap: at most 8 verified samples per family."""
        counts: dict[str, int] = {}
        with SHARD0_VERIFIED.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                s = Sample.from_json_line(line)
                counts[s.family_id] = counts.get(s.family_id, 0) + 1
        for fid, n in counts.items():
            assert n <= MAX_TOTAL_PER_FAMILY, (
                f"family {fid} has {n} verified samples "
                f"(cap={MAX_TOTAL_PER_FAMILY})"
            )

    def test_no_family_bucket_exceeds_3(self):
        """Per-bucket cap: at most 3 verified samples per family per bucket."""
        counts: dict[tuple[str, str], int] = {}
        with SHARD0_VERIFIED.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                s = Sample.from_json_line(line)
                # Map variant_type/task_type to bucket
                vt = s.variant_type or s.task_type
                if vt in ("code", "code_generation"):
                    bucket = "code"
                elif vt == "boundary":
                    bucket = "boundary"
                elif vt == "static_repair":
                    bucket = "static_repair"
                elif vt == "execution_repair":
                    bucket = "execution_repair"
                else:
                    continue
                key = (s.family_id, bucket)
                counts[key] = counts.get(key, 0) + 1
        for (fid, bucket), n in counts.items():
            assert n <= 3, (
                f"family {fid} bucket {bucket} has {n} samples (cap=3)"
            )

    def test_cache_marks_all_families_completed(self):
        """Every family in the shard should have a completed cache entry."""
        with SHARD0_MANIFEST.open(encoding="utf-8") as fh:
            m = json.load(fh)
        expected = set(m["family_ids"])
        completed = _load_cache_completed(SHARD0_CACHE)
        missing = expected - completed
        assert not missing, (
            f"{len(missing)} families missing completed cache entry: "
            f"first 5={sorted(missing)[:5]}"
        )

    def test_no_duplicate_sample_ids_in_verified(self):
        """No duplicate sample_id in verified.jsonl."""
        seen: set[str] = set()
        dupes: list[str] = []
        with SHARD0_VERIFIED.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                s = Sample.from_json_line(line)
                if s.sample_id in seen:
                    dupes.append(s.sample_id)
                seen.add(s.sample_id)
        assert not dupes, (
            f"{len(dupes)} duplicate sample_ids in verified.jsonl: "
            f"first 5={dupes[:5]}"
        )


# ---------------------------------------------------------------------------
# 7. Consolidation (only runs if --consolidate was invoked)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not FORMAL_POOL_PATH.exists(),
    reason="consolidate_pool() must be run first: "
           "py -3.11 scripts/p3_formal_pool_builder.py --consolidate",
)
class TestConsolidatedPool:
    """Verify the consolidated canonical-pool.jsonl + manifest."""

    def test_manifest_exists_and_has_sha(self):
        assert FORMAL_MANIFEST_PATH.exists()
        with FORMAL_MANIFEST_PATH.open(encoding="utf-8") as fh:
            m = json.load(fh)
        assert "pool_sha256" in m
        assert len(m["pool_sha256"]) == 64  # SHA-256 hex

    def test_pool_line_count_matches_manifest(self):
        with FORMAL_MANIFEST_PATH.open(encoding="utf-8") as fh:
            m = json.load(fh)
        n_lines = sum(
            1 for line in FORMAL_POOL_PATH.open(encoding="utf-8")
            if line.strip()
        )
        assert m["total_samples"] == n_lines

    def test_no_family_exceeds_1_percent(self):
        """Single family must occupy <=1% of the total pool.

        This cap is only enforceable on a full run (17 shards, ~2754 samples):
        with 8 samples/family, 8/2754 = 0.29% < 1%. A partial run (e.g. the
        1-shard verification with 162 samples) cannot satisfy this — 8/162 =
        4.94% — so we skip the assertion when shard_count < 17.
        """
        with FORMAL_MANIFEST_PATH.open(encoding="utf-8") as fh:
            m = json.load(fh)
        if m.get("shard_count", 0) < 17:
            pytest.skip(
                f"1% cap only enforceable on full run; shard_count="
                f"{m.get('shard_count')} < 17"
            )
        assert m["max_family_pct"] <= 1.0, (
            f"max_family_pct={m['max_family_pct']}% exceeds 1% cap"
        )
