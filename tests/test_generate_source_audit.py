"""Tests for scripts/generate_source_audit.py (Task 4).

Covers the audit logic against synthetic manifest + JSONL fixtures in a
temporary directory.  No network access, no real MBPP data.

Test list:
1. test_extract_task_id_parsing -- mbpp_<int> -> int, others -> None
2. test_find_missing_task_ids -- gaps in [min, max] range
3. test_find_duplicate_task_ids -- ids appearing >1 time
4. test_audit_split_correct_count_and_ranges -- single split audit fields
5. test_audit_split_missing_manifest_returns_present_false -- absent manifest
6. test_cross_split_overlap_disjoint -- no overlap when splits differ
7. test_cross_split_overlap_non_empty -- overlap detected when shared ids
8. test_conclusion_likely_feasible -- test verified >= 240
9. test_conclusion_infeasible -- test verified < 240
10. test_new_families_available_computation -- test + validation verified
11. test_generate_audit_full_report -- end-to-end on synthetic dir
12. test_main_writes_report_and_returns_zero -- CLI smoke test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import generate_source_audit as gsa
from scripts.generate_source_audit import (
    DEFAULT_SPLITS,
    _TEST_VERIFIED_THRESHOLD,
    audit_split,
    compute_conclusion,
    cross_split_overlap,
    extract_task_id,
    find_duplicate_task_ids,
    find_missing_task_ids,
    generate_audit,
    load_manifest,
    load_task_ids,
)


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------

def _write_manifest(
    *,
    path: Path,
    split: str,
    sample_count: int,
    normalized_sha256: str = "abcd",
    verified_sha256: str | None = None,
    verified_count: int | None = None,
    rejected_count: int | None = None,
    benchmark_contaminated: bool = False,
    dataset_fingerprint: str | None = None,
    source_revision: str = "main",
) -> None:
    """Write a synthetic per-split manifest JSON file."""
    manifest = {
        "source": "google-research-datasets/mbpp",
        "source_revision": source_revision,
        "dataset_fingerprint": dataset_fingerprint,
        "split": split,
        "sample_count": sample_count,
        "normalized_sha256": normalized_sha256,
        "normalized_file": f"normalized/{split}.jsonl",
        "license": "Apache-2.0",
        "imported_at": "2026-07-04T00:00:00+00:00",
        "benchmark_contaminated": benchmark_contaminated,
        "standard_mbpp_test_claims_disallowed": benchmark_contaminated,
        "verified_sha256": verified_sha256,
        "verified_count": verified_count,
        "rejected_count": rejected_count,
        "rejected_sha256": "" if verified_count is not None else None,
        "verified_at": "2026-07-04T01:00:00+00:00" if verified_count is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh)


def _write_normalized_jsonl(*, path: Path, task_ids: list[int]) -> None:
    """Write synthetic Sample JSONL records with given task_ids.

    Each record is a minimal dict carrying just ``sample_id`` (which is
    what the auditor reads).  Other Sample fields are not required by
    :func:`load_task_ids`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for tid in task_ids:
            record = {"sample_id": f"mbpp_{tid}"}
            fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# 1. extract_task_id parsing
# ---------------------------------------------------------------------------

class TestExtractTaskId:
    def test_extract_task_id_parsing(self) -> None:
        assert extract_task_id("mbpp_601") == 601
        assert extract_task_id("mbpp_1") == 1
        assert extract_task_id("mbpp_99999") == 99999

    def test_extract_task_id_none_for_invalid(self) -> None:
        assert extract_task_id("") is None
        assert extract_task_id(None) is None  # type: ignore[arg-type]
        assert extract_task_id("mbpp_") is None
        assert extract_task_id("mbpp_abc") is None
        assert extract_task_id("human_eval_1") is None
        assert extract_task_id("601") is None  # missing prefix

    def test_extract_task_id_case_insensitive_prefix(self) -> None:
        # The parser lowercases the prefix for matching.
        assert extract_task_id("MBPP_42") == 42


# ---------------------------------------------------------------------------
# 2-3. missing / duplicate detection
# ---------------------------------------------------------------------------

class TestMissingDuplicate:
    def test_find_missing_task_ids_no_gaps(self) -> None:
        assert find_missing_task_ids([1, 2, 3, 4, 5]) == []

    def test_find_missing_task_ids_with_gaps(self) -> None:
        # Range is [10, 14]; 12 and 13 are missing.
        assert find_missing_task_ids([10, 11, 14]) == [12, 13]

    def test_find_missing_task_ids_empty(self) -> None:
        assert find_missing_task_ids([]) == []

    def test_find_missing_task_ids_unsorted_input(self) -> None:
        # Gaps computed against min/max regardless of input order.
        assert find_missing_task_ids([14, 10, 11]) == [12, 13]

    def test_find_duplicate_task_ids_none(self) -> None:
        assert find_duplicate_task_ids([1, 2, 3]) == []

    def test_find_duplicate_task_ids_with_dupes(self) -> None:
        # 601 appears 3 times, 602 appears twice -- both reported once.
        assert find_duplicate_task_ids(
            [601, 601, 601, 602, 602, 603]
        ) == [601, 602]

    def test_find_duplicate_task_ids_empty(self) -> None:
        assert find_duplicate_task_ids([]) == []


# ---------------------------------------------------------------------------
# 4-5. audit_split fields
# ---------------------------------------------------------------------------

class TestAuditSplit:
    def test_audit_split_correct_count_and_ranges(self, tmp_path: Path) -> None:
        # Build a manifest + JSONL for split=test with task_ids 601..605
        # (no gaps, no dupes).
        _write_manifest(
            path=tmp_path / "manifest.test.json",
            split="test",
            sample_count=5,
            normalized_sha256="sha-test",
            verified_sha256="sha-verified",
            verified_count=4,
            rejected_count=1,
            benchmark_contaminated=True,
            dataset_fingerprint="fp-test",
            source_revision="main",
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "test.jsonl",
            task_ids=[601, 602, 603, 604, 605],
        )

        out = audit_split(split="test", output_dir=tmp_path)

        assert out["present"] is True
        assert out["sample_count"] == 5
        assert out["task_id_range"] == {"min": 601, "max": 605}
        assert out["missing_task_ids"] == []
        assert out["duplicate_task_ids"] == []
        assert out["normalized_sha256"] == "sha-test"
        assert out["verified_sha256"] == "sha-verified"
        assert out["verified_count"] == 4
        assert out["rejected_count"] == 1
        assert out["benchmark_contaminated"] is True
        assert out["dataset_fingerprint"] == "fp-test"
        assert out["source_revision"] == "main"

    def test_audit_split_with_gaps_and_duplicates(
        self, tmp_path: Path
    ) -> None:
        # task_ids = [601, 603, 605, 605] -> gap [602, 604], dup [605].
        _write_manifest(
            path=tmp_path / "manifest.train.json",
            split="train",
            sample_count=4,
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "train.jsonl",
            task_ids=[601, 603, 605, 605],
        )
        out = audit_split(split="train", output_dir=tmp_path)
        assert out["task_id_range"] == {"min": 601, "max": 605}
        assert out["missing_task_ids"] == [602, 604]
        assert out["duplicate_task_ids"] == [605]

    def test_audit_split_missing_manifest_returns_present_false(
        self, tmp_path: Path
    ) -> None:
        out = audit_split(split="validation", output_dir=tmp_path)
        assert out["present"] is False
        assert out["sample_count"] is None
        assert out["verified_count"] is None
        assert out["normalized_sha256"] is None
        assert out["task_id_range"] == {"min": None, "max": None}
        assert out["missing_task_ids"] == []
        assert out["duplicate_task_ids"] == []

    def test_audit_split_unverified_manifest_has_null_verified_fields(
        self, tmp_path: Path
    ) -> None:
        # Importer-only manifest: verified_* fields are None.
        _write_manifest(
            path=tmp_path / "manifest.test.json",
            split="test",
            sample_count=3,
            verified_sha256=None,
            verified_count=None,
            rejected_count=None,
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "test.jsonl",
            task_ids=[1, 2, 3],
        )
        out = audit_split(split="test", output_dir=tmp_path)
        assert out["verified_sha256"] is None
        assert out["verified_count"] is None
        assert out["rejected_count"] is None


# ---------------------------------------------------------------------------
# 6-7. cross_split_overlap
# ---------------------------------------------------------------------------

class TestCrossSplitOverlap:
    def test_cross_split_overlap_disjoint(self, tmp_path: Path) -> None:
        # train has 1..3, test has 100..102 -- no overlap.
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "train.jsonl",
            task_ids=[1, 2, 3],
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "test.jsonl",
            task_ids=[100, 101, 102],
        )
        overlap = cross_split_overlap(
            output_dir=tmp_path, splits=("train", "test"),
        )
        assert overlap["test_train"] == []

    def test_cross_split_overlap_non_empty(self, tmp_path: Path) -> None:
        # train has 1..5, test has 4..6 -> overlap {4, 5}.
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "train.jsonl",
            task_ids=[1, 2, 3, 4, 5],
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "test.jsonl",
            task_ids=[4, 5, 6],
        )
        overlap = cross_split_overlap(
            output_dir=tmp_path, splits=("train", "test"),
        )
        assert overlap["test_train"] == [4, 5]

    def test_cross_split_overlap_three_way(self, tmp_path: Path) -> None:
        # 3 splits -- pair keys are sorted alphabetically.
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "train.jsonl",
            task_ids=[10, 20, 30],
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "test.jsonl",
            task_ids=[20, 30, 40],
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "validation.jsonl",
            task_ids=[30, 40, 50],
        )
        overlap = cross_split_overlap(
            output_dir=tmp_path,
            splits=("train", "test", "validation"),
        )
        # Keys are sorted split pairs.
        assert set(overlap.keys()) == {
            "test_train", "test_validation", "train_validation",
        }
        assert overlap["test_train"] == [20, 30]
        assert overlap["test_validation"] == [30, 40]
        assert overlap["train_validation"] == [30]


# ---------------------------------------------------------------------------
# 8-10. conclusion + new_families_available
# ---------------------------------------------------------------------------

class TestConclusion:
    def test_conclusion_likely_feasible(self) -> None:
        # test verified_count >= threshold (default 240).
        splits_audit = {
            "test": {"verified_count": 250},
            "validation": {"verified_count": 80},
            "train": {"verified_count": 370},
        }
        conclusion, notes, new_fam = compute_conclusion(splits_audit=splits_audit)
        assert conclusion == "LIKELY_FEASIBLE"
        # new_families = test + validation (these are the NEW families)
        assert new_fam == 250 + 80
        assert str(_TEST_VERIFIED_THRESHOLD) in notes

    def test_conclusion_infeasible(self) -> None:
        # test verified_count below threshold.
        splits_audit = {
            "test": {"verified_count": 100},
            "validation": {"verified_count": 50},
            "train": {"verified_count": 0},
        }
        conclusion, notes, new_fam = compute_conclusion(splits_audit=splits_audit)
        assert conclusion == "INFEASIBLE"
        assert new_fam == 100 + 50
        assert "Stop and escalate" in notes

    def test_conclusion_threshold_boundary(self) -> None:
        # At exactly the threshold -> LIKELY_FEASIBLE (>= comparison).
        splits_audit = {
            "test": {"verified_count": _TEST_VERIFIED_THRESHOLD},
            "validation": {"verified_count": 0},
        }
        conclusion, _notes, _new_fam = compute_conclusion(splits_audit=splits_audit)
        assert conclusion == "LIKELY_FEASIBLE"

    def test_conclusion_handles_null_verified_count(self) -> None:
        # verified_count can be None when split not yet verified.
        splits_audit = {
            "test": {"verified_count": None},
            "validation": {"verified_count": None},
        }
        conclusion, _notes, new_fam = compute_conclusion(splits_audit=splits_audit)
        # None -> 0 -> INFEASIBLE
        assert conclusion == "INFEASIBLE"
        assert new_fam == 0

    def test_new_families_available_computation(self) -> None:
        # new_families = test verified + validation verified (NOT train,
        # because train was already used in P2).
        splits_audit = {
            "test": {"verified_count": 300},
            "validation": {"verified_count": 90},
            "train": {"verified_count": 374},
        }
        _conclusion, _notes, new_fam = compute_conclusion(splits_audit=splits_audit)
        # 300 + 90 = 390, NOT 300 + 90 + 374.
        assert new_fam == 390


# ---------------------------------------------------------------------------
# 11. end-to-end generate_audit
# ---------------------------------------------------------------------------

class TestGenerateAudit:
    def test_generate_audit_full_report(self, tmp_path: Path) -> None:
        # Three splits, all verified, test meets the threshold.
        _write_manifest(
            path=tmp_path / "manifest.train.json",
            split="train",
            sample_count=374,
            verified_count=370,
            rejected_count=4,
            benchmark_contaminated=False,
            dataset_fingerprint="fp-train",
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "train.jsonl",
            task_ids=list(range(601, 601 + 374)),
        )

        _write_manifest(
            path=tmp_path / "manifest.test.json",
            split="test",
            sample_count=500,
            verified_count=260,
            rejected_count=240,
            benchmark_contaminated=True,
            dataset_fingerprint="fp-test",
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "test.jsonl",
            task_ids=list(range(1, 1 + 500)),
        )

        _write_manifest(
            path=tmp_path / "manifest.validation.json",
            split="validation",
            sample_count=90,
            verified_count=85,
            rejected_count=5,
            benchmark_contaminated=False,
            dataset_fingerprint="fp-val",
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "validation.jsonl",
            task_ids=list(range(1000, 1000 + 90)),
        )

        audit = generate_audit(output_dir=tmp_path)

        # Top-level keys.
        assert "generated_at" in audit
        assert audit["source"] == "google-research-datasets/mbpp"
        assert audit["source_revision"] == "main"

        # Splits present.
        assert set(audit["splits"].keys()) == {"train", "test", "validation"}

        # Per-split values.
        assert audit["splits"]["train"]["sample_count"] == 374
        assert audit["splits"]["train"]["verified_count"] == 370
        assert audit["splits"]["train"]["task_id_range"] == {
            "min": 601, "max": 601 + 374 - 1,
        }
        assert audit["splits"]["train"]["missing_task_ids"] == []
        assert audit["splits"]["train"]["duplicate_task_ids"] == []
        assert audit["splits"]["train"]["benchmark_contaminated"] is False

        assert audit["splits"]["test"]["sample_count"] == 500
        assert audit["splits"]["test"]["verified_count"] == 260
        assert audit["splits"]["test"]["benchmark_contaminated"] is True

        assert audit["splits"]["validation"]["sample_count"] == 90
        assert audit["splits"]["validation"]["verified_count"] == 85

        # Cross-split overlap: all three ranges disjoint -> all empty.
        for key, val in audit["cross_split_task_id_overlap"].items():
            assert val == [], f"expected empty overlap for {key}, got {val}"

        # Totals.
        assert audit["total_samples"] == 374 + 500 + 90
        assert audit["total_verified"] == 370 + 260 + 85
        assert audit["total_rejected"] == 4 + 240 + 5

        # new_families_available = test verified + validation verified.
        assert audit["new_families_available"] == 260 + 85

        # Conclusion: test verified=260 >= 240 -> LIKELY_FEASIBLE.
        assert audit["conclusion"] == "LIKELY_FEASIBLE"

    def test_generate_audit_partial_progress(self, tmp_path: Path) -> None:
        # Only test imported; validation + train manifests missing.
        _write_manifest(
            path=tmp_path / "manifest.test.json",
            split="test",
            sample_count=500,
            verified_count=200,
            rejected_count=300,
            benchmark_contaminated=True,
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "test.jsonl",
            task_ids=list(range(1, 1 + 500)),
        )

        audit = generate_audit(output_dir=tmp_path)

        # test split present; train + validation not.
        assert audit["splits"]["test"]["present"] is True
        assert audit["splits"]["train"]["present"] is False
        assert audit["splits"]["validation"]["present"] is False

        # Totals: only test contributes.
        assert audit["total_samples"] == 500
        assert audit["total_verified"] == 200
        assert audit["total_rejected"] == 300

        # Conclusion: test verified=200 < 240 -> INFEASIBLE.
        assert audit["conclusion"] == "INFEASIBLE"

    def test_generate_audit_overlap_detected(self, tmp_path: Path) -> None:
        # train and test share task_id 42.
        _write_manifest(
            path=tmp_path / "manifest.train.json",
            split="train", sample_count=3,
        )
        _write_manifest(
            path=tmp_path / "manifest.test.json",
            split="test", sample_count=3,
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "train.jsonl",
            task_ids=[1, 2, 42],
        )
        _write_normalized_jsonl(
            path=tmp_path / "normalized" / "test.jsonl",
            task_ids=[42, 100, 101],
        )
        audit = generate_audit(output_dir=tmp_path)
        # Overlap key is the alphabetically-sorted pair.
        assert audit["cross_split_task_id_overlap"]["test_train"] == [42]


# ---------------------------------------------------------------------------
# 12. main() CLI smoke test
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_writes_report_and_returns_zero(self, tmp_path: Path) -> None:
        # Build a minimal valid input dir with one manifest.
        _write_manifest(
            path=tmp_path / "data" / "manifest.test.json",
            split="test",
            sample_count=2,
            verified_count=2,
        )
        _write_normalized_jsonl(
            path=tmp_path / "data" / "normalized" / "test.jsonl",
            task_ids=[1, 2],
        )
        report_dir = tmp_path / "reports" / "p3"

        original_argv = sys.argv
        try:
            sys.argv = [
                "generate_source_audit.py",
                "--output-dir", str(tmp_path / "data"),
                "--report-dir", str(report_dir),
                "--splits", "test",
            ]
            rc = gsa.main()
        finally:
            sys.argv = original_argv

        assert rc == 0
        report_path = report_dir / "mbpp-source-audit.json"
        assert report_path.exists()
        with report_path.open("r", encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["conclusion"] in {"LIKELY_FEASIBLE", "INFEASIBLE"}
        assert report["splits"]["test"]["present"] is True
        assert report["splits"]["test"]["sample_count"] == 2

    def test_main_returns_1_when_no_manifests(self, tmp_path: Path) -> None:
        # Empty output-dir: no manifests at all -> exit 1.
        (tmp_path / "data").mkdir()
        original_argv = sys.argv
        try:
            sys.argv = [
                "generate_source_audit.py",
                "--output-dir", str(tmp_path / "data"),
                "--report-dir", str(tmp_path / "reports"),
            ]
            rc = gsa.main()
        finally:
            sys.argv = original_argv
        assert rc == 1


# ---------------------------------------------------------------------------
# load_manifest / load_task_ids helpers
# ---------------------------------------------------------------------------

class TestLoaders:
    def test_load_manifest_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path / "does-not-exist.json") is None

    def test_load_manifest_returns_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.test.json"
        _write_manifest(path=path, split="test", sample_count=1)
        result = load_manifest(path)
        assert isinstance(result, dict)
        assert result["split"] == "test"
        assert result["sample_count"] == 1

    def test_load_task_ids_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_task_ids(tmp_path / "nope.jsonl") == []

    def test_load_task_ids_skips_unparseable_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "normalized" / "test.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write('{"sample_id":"mbpp_1"}\n')
            fh.write('not json\n')  # bad JSON -- skipped
            fh.write('{"sample_id":"mbpp_3"}\n')
            fh.write('{"sample_id":"unknown"}\n')  # unparsable id -- skipped
            fh.write("\n")  # blank -- skipped
        result = load_task_ids(path)
        assert result == [1, 3]
