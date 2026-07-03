"""Tests for scripts/import_mbpp.py

Covers:
- split_mbpp_tests: public/hidden partitioning of MBPP test_list
- extract_skill_tags: keyword extraction + general fallback
- infer_difficulty: string / int / None (code-length fallback)
- mbpp_record_to_sample: full Sample conversion correctness
- compute_sha256: known digests
- build_manifest: required keys + value preservation
- write_manifest / write_normalized_jsonl: file creation + JSON round-trip

No network calls: synthetic MBPP-like records only.  Real HuggingFace
download is exercised by the CLI (scripts/import_mbpp.py main()) and is NOT
covered here, so tests pass regardless of network availability.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.import_mbpp import (
    build_manifest,
    compute_sha256,
    extract_skill_tags,
    infer_difficulty,
    mbpp_record_to_sample,
    split_mbpp_tests,
    write_manifest,
    write_normalized_jsonl,
)
from src.schemas import Sample


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _mbpp_record(**kwargs) -> dict:
    """Return a synthetic MBPP record (task_id=601, 3 tests)."""
    base = {
        "task_id": 601,
        "text": "Write a function to add two numbers.",
        "code": "def add(a, b):\n    return a + b\n",
        "test_list": [
            "assert add(1, 2) == 3",
            "assert add(0, 0) == 0",
            "assert add(-1, 1) == 0",
        ],
        "difficulty": "Introductory",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# split_mbpp_tests
# ---------------------------------------------------------------------------

class TestSplitMbppTests:
    def test_three_tests_public_two_hidden_one(self) -> None:
        pub, hid = split_mbpp_tests(["t1", "t2", "t3"])
        assert pub == "t1\n\nt2"
        assert hid == "t3"

    def test_two_tests_all_public_no_hidden(self) -> None:
        pub, hid = split_mbpp_tests(["t1", "t2"])
        assert pub == "t1\n\nt2"
        assert hid == ""

    def test_single_test(self) -> None:
        pub, hid = split_mbpp_tests(["t1"])
        assert pub == "t1"
        assert hid == ""

    def test_empty_list(self) -> None:
        pub, hid = split_mbpp_tests([])
        assert pub == ""
        assert hid == ""

    def test_skips_blank_entries(self) -> None:
        pub, hid = split_mbpp_tests(["t1", "", "  ", "t2", "t3"])
        assert pub == "t1\n\nt2"
        assert hid == "t3"


# ---------------------------------------------------------------------------
# extract_skill_tags
# ---------------------------------------------------------------------------

class TestExtractSkillTags:
    def test_matches_known_keywords(self) -> None:
        tags = extract_skill_tags("sort a list of numbers")
        assert "sort" in tags
        assert "list" in tags

    def test_no_match_returns_general(self) -> None:
        assert extract_skill_tags("hello world foo") == ["general"]

    def test_keyword_listed_once(self) -> None:
        # The same keyword appearing many times must be listed only once.
        assert extract_skill_tags("list list list") == ["list"]

    def test_returns_unique_entries(self) -> None:
        tags = extract_skill_tags("sort then sorting a list")
        assert len(tags) == len(set(tags))


# ---------------------------------------------------------------------------
# infer_difficulty
# ---------------------------------------------------------------------------

class TestInferDifficulty:
    def test_string_introductory(self) -> None:
        assert infer_difficulty("Introductory", "x") == 1

    def test_string_competition(self) -> None:
        assert infer_difficulty("Competition", "x") == 3

    def test_string_case_insensitive(self) -> None:
        assert infer_difficulty("INTERVIEW", "x") == 2

    def test_int_in_range(self) -> None:
        assert infer_difficulty(2, "x") == 2

    def test_numeric_string_in_range(self) -> None:
        assert infer_difficulty("3", "x") == 3

    def test_int_out_of_range_falls_back(self) -> None:
        # 7 is outside 0..4 -> fall back to code-length bucket
        assert infer_difficulty(7, "x") == 0  # len("x") == 1 -> bucket 0

    def test_none_falls_back_to_code_length(self) -> None:
        assert infer_difficulty(None, "short") == 0
        assert infer_difficulty(None, "x" * 2500) == 4

    def test_code_length_buckets(self) -> None:
        assert infer_difficulty(None, "x" * 100) == 0
        assert infer_difficulty(None, "x" * 300) == 1
        assert infer_difficulty(None, "x" * 700) == 2
        assert infer_difficulty(None, "x" * 1500) == 3
        assert infer_difficulty(None, "x" * 2500) == 4

    def test_result_always_in_range(self) -> None:
        for d in ("Introductory", "Interview", "Competition", None, 0, 4):
            assert 0 <= infer_difficulty(d, "code") <= 4


# ---------------------------------------------------------------------------
# mbpp_record_to_sample
# ---------------------------------------------------------------------------

class TestMbppRecordToSample:
    def test_returns_sample_instance(self) -> None:
        assert isinstance(mbpp_record_to_sample(_mbpp_record()), Sample)

    def test_sample_id_format(self) -> None:
        s = mbpp_record_to_sample(_mbpp_record())
        assert s.sample_id == "mbpp_601"

    def test_family_id_format(self) -> None:
        s = mbpp_record_to_sample(_mbpp_record())
        assert s.family_id == "mbpp_fam_601"

    def test_task_type_is_code_generation(self) -> None:
        s = mbpp_record_to_sample(_mbpp_record())
        assert s.task_type == "code_generation"

    def test_language_is_python(self) -> None:
        assert mbpp_record_to_sample(_mbpp_record()).language == "python"

    def test_instruction_is_text_field(self) -> None:
        rec = _mbpp_record()
        assert mbpp_record_to_sample(rec).instruction == rec["text"]

    def test_instruction_falls_back_to_prompt_alias(self) -> None:
        rec = _mbpp_record()
        del rec["text"]
        rec["prompt"] = "Reverse a string."
        assert mbpp_record_to_sample(rec).instruction == "Reverse a string."

    def test_target_code_is_code_field(self) -> None:
        rec = _mbpp_record()
        assert mbpp_record_to_sample(rec).target_code == rec["code"]

    def test_broken_code_is_none(self) -> None:
        assert mbpp_record_to_sample(_mbpp_record()).broken_code is None

    def test_execution_feedback_is_none(self) -> None:
        assert mbpp_record_to_sample(_mbpp_record()).execution_feedback is None

    def test_public_tests_gets_first_two(self) -> None:
        rec = _mbpp_record()
        s = mbpp_record_to_sample(rec)
        assert "assert add(1, 2) == 3" in s.public_tests
        assert "assert add(0, 0) == 0" in s.public_tests
        # third assert is NOT public
        assert "assert add(-1, 1) == 0" not in s.public_tests

    def test_hidden_tests_gets_remainder(self) -> None:
        rec = _mbpp_record()
        s = mbpp_record_to_sample(rec)
        assert "assert add(-1, 1) == 0" in s.hidden_tests

    def test_hidden_tests_empty_when_two_tests(self) -> None:
        rec = _mbpp_record(test_list=["assert add(1, 2) == 3",
                                       "assert add(0, 0) == 0"])
        assert mbpp_record_to_sample(rec).hidden_tests == ""

    def test_verified_is_true(self) -> None:
        assert mbpp_record_to_sample(_mbpp_record()).verified is True

    def test_verification_flags(self) -> None:
        v = mbpp_record_to_sample(_mbpp_record()).verification
        assert v.syntax_ok is True
        assert v.pytest_ok is True
        assert v.ruff_ok is False
        assert v.timeout is False

    def test_generator_is_mbpp_importer(self) -> None:
        assert mbpp_record_to_sample(_mbpp_record()).generator == "mbpp-importer"

    def test_dataset_version(self) -> None:
        assert mbpp_record_to_sample(_mbpp_record()).dataset_version == "mbpp-v1"

    def test_created_at_passthrough(self) -> None:
        ts = "2026-01-01T00:00:00+00:00"
        assert mbpp_record_to_sample(_mbpp_record(), created_at=ts).created_at == ts

    def test_skill_tags_extracted_non_empty(self) -> None:
        s = mbpp_record_to_sample(
            _mbpp_record(text="Write a function to add two numbers.")
        )
        assert isinstance(s.skill_tags, list)
        assert len(s.skill_tags) >= 1

    def test_difficulty_in_range(self) -> None:
        s = mbpp_record_to_sample(_mbpp_record(difficulty="Competition"))
        assert 0 <= s.difficulty <= 4

    def test_difficulty_string_resolved(self) -> None:
        assert mbpp_record_to_sample(
            _mbpp_record(difficulty="Competition")).difficulty == 3

    def test_jsonl_roundtrip(self) -> None:
        s = mbpp_record_to_sample(_mbpp_record())
        restored = Sample.from_json_line(s.to_json_line())
        assert restored.sample_id == s.sample_id
        assert restored.target_code == s.target_code

    def test_sample_id_uses_task_id_from_record(self) -> None:
        assert mbpp_record_to_sample(
            _mbpp_record(task_id=42)).sample_id == "mbpp_42"


# ---------------------------------------------------------------------------
# compute_sha256
# ---------------------------------------------------------------------------

class TestComputeSha256:
    def test_known_hello(self, tmp_path: Path) -> None:
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        # sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        assert compute_sha256(p).startswith("2cf24dba")

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert compute_sha256(p) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_stable_across_calls(self, tmp_path: Path) -> None:
        p = tmp_path / "f.bin"
        p.write_bytes(b"abc")
        assert compute_sha256(p) == compute_sha256(p)


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def _make(self, **kwargs) -> dict:
        defaults = dict(
            source="google-research-datasets/mbpp",
            split="train",
            sample_count=10,
            sha256="abc123",
            license="Apache-2.0",
            imported_at="2026-01-01T00:00:00+00:00",
        )
        defaults.update(kwargs)
        return build_manifest(**defaults)

    def test_required_keys(self) -> None:
        m = self._make()
        for k in ("source", "split", "sample_count", "sha256",
                  "license", "imported_at"):
            assert k in m, f"missing key: {k}"

    def test_values_preserved(self) -> None:
        m = self._make(
            source="google-research-datasets/mbpp",
            split="train",
            sample_count=374,
            sha256="deadbeef",
            license="Apache-2.0",
            imported_at="2026-07-02T00:00:00+00:00",
        )
        assert m["source"] == "google-research-datasets/mbpp"
        assert m["split"] == "train"
        assert m["sample_count"] == 374
        assert m["sha256"] == "deadbeef"
        assert m["license"] == "Apache-2.0"
        assert m["imported_at"] == "2026-07-02T00:00:00+00:00"

    def test_imported_at_is_iso_parseable(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        m = self._make(imported_at=ts)
        datetime.fromisoformat(m["imported_at"])  # must not raise

    def test_license_always_apache(self) -> None:
        assert self._make()["license"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# write_manifest / write_normalized_jsonl
# ---------------------------------------------------------------------------

class TestWriteHelpers:
    def test_write_manifest_creates_file_with_parents(self, tmp_path: Path) -> None:
        p = tmp_path / "nested" / "dir" / "manifest.json"
        write_manifest({"a": 1}, p)
        assert p.exists()
        assert json.loads(p.read_text(encoding="utf-8"))["a"] == 1

    def test_write_manifest_preserves_unicode(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.json"
        write_manifest({"note": "中文"}, p)
        assert json.loads(p.read_text(encoding="utf-8"))["note"] == "中文"

    def test_write_normalized_jsonl_creates_file(self, tmp_path: Path) -> None:
        s = mbpp_record_to_sample(_mbpp_record())
        p = tmp_path / "normalized" / "train.jsonl"
        write_normalized_jsonl([s], p)
        assert p.exists()
        Sample.from_json_line(p.read_text(encoding="utf-8").strip())  # parses

    def test_write_normalized_jsonl_line_count(self, tmp_path: Path) -> None:
        samples = [mbpp_record_to_sample(_mbpp_record(task_id=i)) for i in range(5)]
        p = tmp_path / "train.jsonl"
        write_normalized_jsonl(samples, p)
        assert len(p.read_text(encoding="utf-8").splitlines()) == 5

    def test_jsonl_sha_matches_manifest(self, tmp_path: Path) -> None:
        """Integration: write JSONL, compute sha, build manifest, verify."""
        samples = [mbpp_record_to_sample(_mbpp_record(task_id=i)) for i in range(3)]
        p = tmp_path / "normalized" / "train.jsonl"
        write_normalized_jsonl(samples, p)
        sha = compute_sha256(p)
        m = build_manifest(
            source="google-research-datasets/mbpp",
            split="train",
            sample_count=len(samples),
            sha256=sha,
            license="Apache-2.0",
            imported_at=datetime.now(timezone.utc).isoformat(),
        )
        assert m["sample_count"] == 3
        assert m["sha256"] == sha
        assert len(m["sha256"]) == 64  # hex digest length
