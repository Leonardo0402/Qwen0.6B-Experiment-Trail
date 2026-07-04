"""Tests for the P3 import_mbpp + verify_imported_mbpp pipeline (Task 3).

Covers the 11 tests specified in `.superpowers/sdd/task-3-brief.md`:

Importer (5)
-----------
1. test_imported_samples_default_verified_false
2. test_per_split_manifest_no_overwrite
3. test_benchmark_contamination_flag
4. test_source_split_field_set
5. test_no_pytest_in_importer

Verifier (6)
------------
6.  test_verified_sample_passes_all_checks
7.  test_rejected_low_public_count
8.  test_rejected_low_hidden_count
9.  test_repair_sample_verify_broken_is_broken
10. test_execution_repair_feedback_check
11. test_manifest_updated_with_verified_fields

Importer tests mock ``datasets.load_dataset`` via ``scripts.import_mbpp._load_dataset``
so no network is hit.  Verifier tests use synthetic Samples and run the REAL
``verify_sample`` / ``verify_broken_is_broken`` (no mocks on the validators).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import import_mbpp
from scripts.import_mbpp import (
    _SOURCE_REPO,
    _SOURCE_REVISION,
    build_manifest,
    mbpp_record_to_sample,
    update_manifest_index,
    write_manifest,
)
from scripts.verify_imported_mbpp import (
    check_sample,
    update_manifest_with_verified,
    verify_split,
    write_verified_jsonl,
)
from src.schemas import Sample, Verification


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mbpp_record(**kwargs) -> dict:
    """Return a synthetic MBPP record (task_id=601, 5 tests)."""
    base = {
        "task_id": 601,
        "text": "Write a function to add two numbers.",
        "code": "def add(a, b):\n    return a + b\n",
        "test_list": [
            "assert add(1, 2) == 3",
            "assert add(0, 0) == 0",
            "assert add(-1, 1) == 0",
            "assert add(2, 3) == 5",
            "assert add(100, 200) == 300",
        ],
        "difficulty": "Introductory",
    }
    base.update(kwargs)
    return base


def _verification_all_false() -> Verification:
    return Verification(syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False)


def _make_sample(**kwargs) -> Sample:
    """Build a minimal valid code_generation sample with passing tests.

    Defaults satisfy the verifier hard checks:
    - public_tests has 2 ``assert `` occurrences (>= 2)
    - hidden_tests has 3 ``assert `` occurrences (>= 3)
    - target_code is correct, so verify_sample().is_accepted is True
    """
    defaults: dict = {
        "sample_id": "test-001",
        "family_id": "fam-001",
        "difficulty": 1,
        "task_type": "code_generation",
        "language": "python",
        "skill_tags": ["arithmetic"],
        "instruction": "Write an add function.",
        "broken_code": None,
        "execution_feedback": None,
        "target_code": "def add(a, b):\n    return a + b\n",
        "public_tests": "assert add(1, 2) == 3\n\nassert add(0, 0) == 0",
        "hidden_tests": (
            "assert add(-1, 1) == 0\n\n"
            "assert add(2, 3) == 5\n\n"
            "assert add(100, 200) == 300"
        ),
        "verified": False,
        "verification": _verification_all_false(),
        "generator": "test-harness",
        "created_at": "2026-01-01T00:00:00+00:00",
        "dataset_version": "v1",
        "source_split": "test",
    }
    defaults.update(kwargs)
    return Sample(**defaults)


def _run_importer_cli(monkeypatch, tmp_path: Path, split: str) -> int:
    """Invoke ``scripts.import_mbpp.main()`` with mocked datasets + CLI args.

    Returns the exit code from main().  ``_load_dataset`` is replaced with a
    callable returning a single synthetic MBPP record; ``_DATASETS_AVAILABLE``
    is forced True so the datasets-unavailable guard does not short-circuit.
    """
    monkeypatch.setattr(import_mbpp, "_load_dataset",
                        lambda *a, **kw: [_mbpp_record()])
    monkeypatch.setattr(import_mbpp, "_DATASETS_AVAILABLE", True)
    monkeypatch.setattr(
        sys, "argv",
        ["import_mbpp.py",
         "--output-dir", str(tmp_path),
         "--split", split],
    )
    return import_mbpp.main()


# ---------------------------------------------------------------------------
# Part 1: Importer tests (mock datasets, no network)
# ---------------------------------------------------------------------------

class TestImporterP3:
    """5 importer tests per the task brief."""

    # --- Test 1 ---
    def test_imported_samples_default_verified_false(self) -> None:
        """Imported Sample must default to verified=False with all-false
        Verification (no preset claims; P3 constraint #2)."""
        s = mbpp_record_to_sample(_mbpp_record(), source_split="train")
        assert s.verified is False
        assert s.verification.syntax_ok is False
        assert s.verification.pytest_ok is False
        assert s.verification.ruff_ok is False
        assert s.verification.timeout is False

    # --- Test 2 ---
    def test_per_split_manifest_no_overwrite(self, tmp_path: Path,
                                              monkeypatch) -> None:
        """Running --split train then --split test must produce BOTH
        manifest.train.json and manifest.test.json (neither overwritten),
        and manifest.index.json must list both splits (P3 constraint #3)."""
        rc1 = _run_importer_cli(monkeypatch, tmp_path, split="train")
        assert rc1 == 0
        rc2 = _run_importer_cli(monkeypatch, tmp_path, split="test")
        assert rc2 == 0

        # Both per-split manifests exist with correct content.
        train_manifest_path = tmp_path / "manifest.train.json"
        test_manifest_path = tmp_path / "manifest.test.json"
        assert train_manifest_path.exists()
        assert test_manifest_path.exists()

        train_m = json.loads(train_manifest_path.read_text(encoding="utf-8"))
        test_m = json.loads(test_manifest_path.read_text(encoding="utf-8"))
        assert train_m["split"] == "train"
        assert test_m["split"] == "test"

        # Index lists both splits and points to the right manifest files.
        index_path = tmp_path / "manifest.index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text(encoding="utf-8"))
        assert sorted(index["splits"]) == ["test", "train"]
        assert index["splits_detail"]["train"]["manifest"] == "manifest.train.json"
        assert index["splits_detail"]["test"]["manifest"] == "manifest.test.json"

    # --- Test 3 ---
    def test_benchmark_contamination_flag(self, tmp_path: Path,
                                           monkeypatch) -> None:
        """test split manifest has benchmark_contaminated=true;
        train split has false (P3 constraint #5)."""
        _run_importer_cli(monkeypatch, tmp_path, split="test")
        _run_importer_cli(monkeypatch, tmp_path, split="train")

        test_m = json.loads(
            (tmp_path / "manifest.test.json").read_text(encoding="utf-8"))
        train_m = json.loads(
            (tmp_path / "manifest.train.json").read_text(encoding="utf-8"))

        assert test_m["benchmark_contaminated"] is True
        assert test_m["standard_mbpp_test_claims_disallowed"] is True
        assert train_m["benchmark_contaminated"] is False
        assert train_m["standard_mbpp_test_claims_disallowed"] is False

    # --- Test 4 ---
    def test_source_split_field_set(self) -> None:
        """Every imported Sample must carry source_split=<split_name>
        (P3 constraint #6)."""
        for split in ("train", "test", "validation"):
            s = mbpp_record_to_sample(_mbpp_record(), source_split=split)
            assert s.source_split == split

    # --- Test 5 ---
    def test_no_pytest_in_importer(self) -> None:
        """Importer must NOT import or call verify_sample / run_pytest
        (P3 constraint #1).  Verified by module attribute inspection: if
        the importer never imports these names, it cannot call them."""
        import scripts.import_mbpp as mod
        assert not hasattr(mod, "verify_sample"), \
            "importer must not import verify_sample"
        assert not hasattr(mod, "run_pytest"), \
            "importer must not import run_pytest"
        assert not hasattr(mod, "verify_broken_is_broken"), \
            "importer must not import verify_broken_is_broken"
        # The src.validators module itself must not be imported by the
        # importer (no transitive pytest path).
        assert "src.validators" not in (getattr(mod, "__dict__", {}) or {}), \
            "importer must not import src.validators"


# ---------------------------------------------------------------------------
# Part 2: Verifier tests (synthetic samples, real validators)
# ---------------------------------------------------------------------------

class TestVerifierP3:
    """6 verifier tests per the task brief."""

    # --- Test 6 ---
    def test_verified_sample_passes_all_checks(self, tmp_path: Path) -> None:
        """Sample with valid code, public>=2, hidden>=3 -> written to
        verified/<split>.jsonl with verified=true (P3 constraint #7)."""
        sample = _make_sample()
        verified, rejected = verify_split([sample])
        assert len(verified) == 1
        assert len(rejected) == 0
        assert verified[0].verified is True
        # Real verification results from verify_sample
        assert verified[0].verification.syntax_ok is True
        assert verified[0].verification.pytest_ok is True

        # Write and read back to confirm verified=true lands in the file.
        out_path = tmp_path / "verified" / "test.jsonl"
        write_verified_jsonl(verified, out_path)
        lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["verified"] is True

    # --- Test 7 ---
    def test_rejected_low_public_count(self) -> None:
        """public_tests with only 1 assert -> rejected with reason
        mentioning 'public assertions' (P3 constraint #7)."""
        sample = _make_sample(
            public_tests="assert add(1, 2) == 3",  # 1 assert only
        )
        passed, reason = check_sample(sample)
        assert passed is False
        assert "public assertions" in reason
        assert "1" in reason

    # --- Test 8 ---
    def test_rejected_low_hidden_count(self) -> None:
        """hidden_tests with only 2 asserts -> rejected with reason
        mentioning 'hidden assertions' (P3 constraint #7)."""
        sample = _make_sample(
            hidden_tests="assert add(-1, 1) == 0\n\nassert add(2, 3) == 5",
            # 2 asserts only -- below the minimum of 3
        )
        passed, reason = check_sample(sample)
        assert passed is False
        assert "hidden assertions" in reason
        assert "2" in reason

    # --- Test 9 ---
    def test_repair_sample_verify_broken_is_broken(self) -> None:
        """static_repair sample where broken_code PASSES all tests ->
        rejected with reason 'broken_code passes all tests' (P3 constraint #8).

        Here broken_code is identical to target_code, so it passes pytest and
        verify_broken_is_broken returns False.
        """
        correct_code = "def add(a, b):\n    return a + b\n"
        sample = _make_sample(
            task_type="static_repair",
            broken_code=correct_code,  # passes -- NOT actually broken
        )
        passed, reason = check_sample(sample)
        assert passed is False
        assert "broken_code passes all tests" in reason

    # --- Test 10 ---
    def test_execution_repair_feedback_check(self) -> None:
        """execution_repair with non-failure execution_feedback -> rejected
        with reason 'execution_feedback lacks failure marker'
        (P3 constraint #9).

        broken_code is genuinely broken (subtracts instead of adds), so
        verify_broken_is_broken returns True and the sample reaches the
        execution_feedback hard check, where it is rejected.

        The feedback string is chosen to contain NONE of the failure
        markers (``error`` / ``assert`` / ``traceback`` / ``failed`` /
        ``exception`` / ``fail``) so the heuristic correctly rejects it.
        """
        sample = _make_sample(
            task_type="execution_repair",
            broken_code="def add(a, b):\n    return a - b\n",  # genuinely broken
            execution_feedback="all good -- program completed",  # no marker
        )
        passed, reason = check_sample(sample)
        assert passed is False
        assert "execution_feedback" in reason
        assert "failure marker" in reason

    # --- Test 11 ---
    def test_manifest_updated_with_verified_fields(self, tmp_path: Path) -> None:
        """After verification, manifest.<split>.json has verified_sha256,
        verified_count, rejected_count, rejected_sha256, verified_at
        filled in (P3 task brief Part B #3)."""
        manifest_path = tmp_path / "manifest.test.json"
        initial = build_manifest(
            source=_SOURCE_REPO,
            source_revision=_SOURCE_REVISION,
            dataset_fingerprint=None,
            split="test",
            sample_count=1,
            normalized_sha256="abc",
            normalized_file="normalized/test.jsonl",
            license="Apache-2.0",
            imported_at="2026-01-01T00:00:00+00:00",
            benchmark_contaminated=True,
            standard_mbpp_test_claims_disallowed=True,
        )
        write_manifest(initial, manifest_path)

        verified_at = datetime.now(timezone.utc).isoformat()
        updated = update_manifest_with_verified(
            manifest_path,
            verified_sha256="deadbeef",
            verified_count=1,
            rejected_count=0,
            rejected_sha256="",
            verified_at=verified_at,
        )
        assert updated["verified_sha256"] == "deadbeef"
        assert updated["verified_count"] == 1
        assert updated["rejected_count"] == 0
        assert updated["rejected_sha256"] == ""
        assert updated["verified_at"] == verified_at

        # Persisted to disk
        with manifest_path.open("r", encoding="utf-8") as fh:
            persisted = json.load(fh)
        assert persisted["verified_sha256"] == "deadbeef"
        assert persisted["verified_count"] == 1
        assert persisted["rejected_count"] == 0
        assert persisted["verified_at"] == verified_at
