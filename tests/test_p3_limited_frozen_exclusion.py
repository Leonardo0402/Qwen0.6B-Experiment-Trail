"""tests/test_p3_limited_frozen_exclusion.py -- TDD tests for Issue #16 Task 1.

Verifies that scripts/build_p3_limited.py:
  1. Excludes frozen v4 sample_ids from train files
  2. Excludes frozen v4 family_ids from train files
  3. Records frozen exclusion in manifest with overlap_check_passed=true
  4. Builds Balanced-Limited with exactly 1280 samples (384/256/256/384)
  5. Builds Repair-Limited with exactly 1280 samples (192/192/384/512)
  6. Has no duplicate sample_ids within each candidate

These tests run the build script via subprocess (integration-level) so they
exercise the real code path the user invokes with `py scripts/build_p3_limited.py`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

SCRIPT_PATH = _ROOT / "scripts" / "build_p3_limited.py"
FROZEN_V4_PATH = _ROOT / "data" / "frozen-eval" / "v4" / "test_raw.jsonl"

BALANCED_DIR = _ROOT / "data" / "p3-limited" / "balanced-limited"
REPAIR_DIR = _ROOT / "data" / "p3-limited" / "repair-limited"

EXPECTED_BALANCED_COUNTS = {
    "code": 384,
    "boundary": 256,
    "static_repair": 256,
    "execution_repair": 384,
}
EXPECTED_REPAIR_COUNTS = {
    "code": 192,
    "boundary": 192,
    "static_repair": 384,
    "execution_repair": 512,
}
EXPECTED_TOTAL = 1280


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_frozen_v4() -> tuple[set[str], set[str]]:
    """Return (sample_ids, family_ids) from frozen v4 test_raw.jsonl."""
    samples = _read_jsonl(FROZEN_V4_PATH)
    sids = {s["sample_id"] for s in samples}
    fams = {s["family_id"] for s in samples}
    return sids, fams


@pytest.fixture(scope="module")
def build_result() -> dict:
    """Run the build script once and return a dict of loaded artifacts."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    if result.returncode != 0:
        pytest.fail(
            f"build_p3_limited.py failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    balanced_train = _read_jsonl(BALANCED_DIR / "train.jsonl")
    repair_train = _read_jsonl(REPAIR_DIR / "train.jsonl")
    with (BALANCED_DIR / "manifest.json").open(encoding="utf-8") as fh:
        balanced_manifest = json.load(fh)
    with (REPAIR_DIR / "manifest.json").open(encoding="utf-8") as fh:
        repair_manifest = json.load(fh)
    return {
        "stdout": result.stdout,
        "balanced_train": balanced_train,
        "repair_train": repair_train,
        "balanced_manifest": balanced_manifest,
        "repair_manifest": repair_manifest,
    }


def test_build_excludes_frozen_sample_ids(build_result: dict) -> None:
    """Train files must contain zero sample_ids from frozen v4."""
    frozen_sids, _ = _load_frozen_v4()
    for name, train in (
        ("balanced-limited", build_result["balanced_train"]),
        ("repair-limited", build_result["repair_train"]),
    ):
        train_sids = {s["sample_id"] for s in train}
        overlap = train_sids & frozen_sids
        assert overlap == set(), (
            f"{name}: {len(overlap)} frozen v4 sample_ids leaked into train "
            f"(examples: {sorted(overlap)[:5]})"
        )


def test_build_excludes_frozen_family_ids(build_result: dict) -> None:
    """Train files must contain zero family_ids from frozen v4."""
    _, frozen_fams = _load_frozen_v4()
    for name, train in (
        ("balanced-limited", build_result["balanced_train"]),
        ("repair-limited", build_result["repair_train"]),
    ):
        train_fams = {s["family_id"] for s in train}
        overlap = train_fams & frozen_fams
        assert overlap == set(), (
            f"{name}: {len(overlap)} frozen v4 family_ids leaked into train "
            f"(examples: {sorted(overlap)[:5]})"
        )


def test_manifest_records_frozen_exclusion(build_result: dict) -> None:
    """Both manifests must have frozen_exclusion with overlap_check_passed=true."""
    for name, manifest in (
        ("balanced-limited", build_result["balanced_manifest"]),
        ("repair-limited", build_result["repair_manifest"]),
    ):
        assert "frozen_exclusion" in manifest, (
            f"{name}: missing frozen_exclusion field in manifest"
        )
        fe = manifest["frozen_exclusion"]
        assert fe.get("frozen_eval_path") == "data/frozen-eval/v4/test_raw.jsonl", (
            f"{name}: frozen_eval_path mismatch: {fe.get('frozen_eval_path')!r}"
        )
        assert fe.get("overlap_check_passed") is True, (
            f"{name}: overlap_check_passed is {fe.get('overlap_check_passed')!r}, expected True"
        )
        assert fe.get("frozen_sample_ids_in_train") == 0, (
            f"{name}: frozen_sample_ids_in_train={fe.get('frozen_sample_ids_in_train')}"
        )
        assert fe.get("frozen_family_ids_in_train") == 0, (
            f"{name}: frozen_family_ids_in_train={fe.get('frozen_family_ids_in_train')}"
        )
        assert isinstance(fe.get("excluded_frozen_sample_count"), int), (
            f"{name}: excluded_frozen_sample_count not an int"
        )
        assert isinstance(fe.get("excluded_frozen_family_count"), int), (
            f"{name}: excluded_frozen_family_count not an int"
        )


def test_balanced_limited_1280_samples(build_result: dict) -> None:
    """Balanced-Limited must have exactly 1280 samples with counts 384/256/256/384."""
    train = build_result["balanced_train"]
    manifest = build_result["balanced_manifest"]
    assert len(train) == EXPECTED_TOTAL, (
        f"balanced-limited: expected {EXPECTED_TOTAL} samples, got {len(train)}"
    )
    counts = {"code": 0, "boundary": 0, "static_repair": 0, "execution_repair": 0}
    for s in train:
        counts[s["variant_type"]] += 1
    assert counts == EXPECTED_BALANCED_COUNTS, (
        f"balanced-limited: bucket counts {counts} != expected {EXPECTED_BALANCED_COUNTS}"
    )
    assert manifest["total_samples"] == EXPECTED_TOTAL
    assert manifest["bucket_counts"] == EXPECTED_BALANCED_COUNTS


def test_repair_limited_1280_samples(build_result: dict) -> None:
    """Repair-Limited must have exactly 1280 samples with counts 192/192/384/512."""
    train = build_result["repair_train"]
    manifest = build_result["repair_manifest"]
    assert len(train) == EXPECTED_TOTAL, (
        f"repair-limited: expected {EXPECTED_TOTAL} samples, got {len(train)}"
    )
    counts = {"code": 0, "boundary": 0, "static_repair": 0, "execution_repair": 0}
    for s in train:
        counts[s["variant_type"]] += 1
    assert counts == EXPECTED_REPAIR_COUNTS, (
        f"repair-limited: bucket counts {counts} != expected {EXPECTED_REPAIR_COUNTS}"
    )
    assert manifest["total_samples"] == EXPECTED_TOTAL
    assert manifest["bucket_counts"] == EXPECTED_REPAIR_COUNTS


def test_no_duplicate_sample_ids_within_candidate(build_result: dict) -> None:
    """No duplicate sample_ids within each candidate's train file."""
    for name, train in (
        ("balanced-limited", build_result["balanced_train"]),
        ("repair-limited", build_result["repair_train"]),
    ):
        sids = [s["sample_id"] for s in train]
        assert len(sids) == len(set(sids)), (
            f"{name}: {len(sids) - len(set(sids))} duplicate sample_ids"
        )
