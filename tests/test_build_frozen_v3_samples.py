"""tests/test_build_frozen_v3_samples.py -- Tests for Task 8.

Covers the 12 tests specified in ``.superpowers/sdd/task-8-brief.md``:

  1.  test_pad_hidden_tests_passes_through_when_sufficient
  2.  test_pad_hidden_tests_extends_to_three
  3.  test_pad_hidden_tests_handles_syntax_error
  4.  test_variant_type_post_processing
  5.  test_bug_type_extraction_from_sample_id
  6.  test_canary_sample_fails_verification
  7.  test_qualified_family_passes_all_gates
  8.  test_rejected_family_recorded
  9.  test_freeze_decision_100_or_more
  10. test_freeze_decision_less_than_80_aborts
  11. test_manifest_sha_lock_consistent
  12. test_unclaim_removes_tag

All tests use synthetic Samples built in-memory. Tests that would otherwise
invoke slow pytest subprocesses (test 7, test 8) use small, trivial
``add(a, b)`` functions so each pytest run is sub-100ms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_frozen_v3_samples import (  # noqa: E402
    CANARY_CODE,
    apply_freeze_decision,
    compute_sha_lock,
    extract_bug_type,
    make_canary,
    post_process_variant_type,
    update_registry,
    verify_family,
    verify_sha_lock,
    write_families_json,
    write_manifest_json,
    write_rejected_jsonl,
    write_test_raw_jsonl,
)
from src.family_registry import FamilyEntry, FamilyRegistry  # noqa: E402
from src.hidden_test_padding import pad_hidden_tests  # noqa: E402
from src.schemas import Sample, Verification  # noqa: E402
from src.validators import verify_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)


def _make_sample(
    *,
    sample_id: str = "mbpp_42_test",
    family_id: str = "mbpp_fam_42",
    task_type: str = "code_generation",
    target_code: str = "def add(a, b):\n    return a + b\n",
    public_tests: str = (
        "assert add(1, 2) == 3\n"
        "assert add(0, 0) == 0\n"
    ),
    hidden_tests: str = (
        "assert add(-1, 1) == 0\n"
        "assert add(10, -5) == 5\n"
        "assert add(100, 200) == 300\n"
    ),
    broken_code: str | None = None,
    execution_feedback: str | None = None,
    variant_type: str | None = None,
    bug_type: str | None = None,
    source_split: str | None = None,
) -> Sample:
    """Build a minimal but valid Sample for tests."""
    return Sample(
        sample_id=sample_id,
        family_id=family_id,
        difficulty=1,
        task_type=task_type,
        language="python",
        skill_tags=["test"],
        instruction="Write a function.",
        broken_code=broken_code,
        execution_feedback=execution_feedback,
        target_code=target_code,
        public_tests=public_tests,
        hidden_tests=hidden_tests,
        verified=False,
        verification=_PLACEHOLDER_VER,
        generator="test",
        created_at="2026-01-01T00:00:00+00:00",
        dataset_version="mbpp-v1",
        variant_type=variant_type,
        bug_type=bug_type,
        source_split=source_split,
    )


def _add_code_sample(sample_id: str = "mbpp_42_code", family_id: str = "mbpp_fam_42") -> Sample:
    """A code_generation sample that passes all gates."""
    return _make_sample(
        sample_id=sample_id,
        family_id=family_id,
        task_type="code_generation",
        variant_type="code",
        source_split="test",
    )


def _add_boundary_sample(sample_id: str = "mbpp_42_boundary", family_id: str = "mbpp_fam_42") -> Sample:
    """A boundary sample that passes all gates."""
    return _make_sample(
        sample_id=sample_id,
        family_id=family_id,
        task_type="code_generation",
        variant_type="boundary",
        source_split="test",
    )


def _add_static_repair_sample(sample_id: str = "mbpp_42_sr_off_by_one", family_id: str = "mbpp_fam_42") -> Sample:
    """A static_repair sample with broken_code that fails at least one test."""
    return _make_sample(
        sample_id=sample_id,
        family_id=family_id,
        task_type="static_repair",
        variant_type="static_repair",
        source_split="test",
        broken_code="def add(a, b):\n    return a - b\n",
    )


def _add_execution_repair_sample(
    sample_id: str = "mbpp_42_er_off_by_one",
    family_id: str = "mbpp_fam_42",
) -> Sample:
    """An execution_repair sample with broken_code and execution_feedback."""
    return _make_sample(
        sample_id=sample_id,
        family_id=family_id,
        task_type="execution_repair",
        variant_type="execution_repair",
        source_split="test",
        broken_code="def add(a, b):\n    return a - b\n",
        execution_feedback=(
            "FAILED add(1, 2) == 3\n"
            "AssertionError: assert -1 == 3\n"
        ),
    )


# ---------------------------------------------------------------------------
# Tests 1-3: pad_hidden_tests
# ---------------------------------------------------------------------------

def test_pad_hidden_tests_passes_through_when_sufficient():
    """Sample with hidden_count >= 3 → returned unchanged."""
    s = _make_sample()  # default has 3 hidden asserts
    padded, reason = pad_hidden_tests(s, target_count=3)
    assert reason is None
    assert padded.hidden_tests == s.hidden_tests


def test_pad_hidden_tests_extends_to_three():
    """Sample with hidden_count=1 → padded to >= 3, new tests pass."""
    s = _make_sample(
        hidden_tests="assert add(-1, 1) == 0\n",
    )
    assert s.hidden_tests.count("assert ") == 1
    padded, reason = pad_hidden_tests(s, target_count=3)
    assert reason is None, f"unexpected rejection reason: {reason}"
    assert padded.hidden_tests.count("assert ") >= 3
    # Verify the new tests pass against target_code.
    sv = verify_sample(padded, run_ruff=False)
    assert sv.is_accepted, (
        f"padded hidden tests should pass against target_code; "
        f"messages: {sv.messages}"
    )


def test_pad_hidden_tests_handles_syntax_error():
    """Malformed target_code → returns sample unchanged + syntax_error reason."""
    s = _make_sample(
        target_code="def add(a, b\n    return a + b\n",  # malformed
        hidden_tests="assert add(1, 2) == 3\n",
    )
    padded, reason = pad_hidden_tests(s, target_count=3)
    assert reason == "hidden_padding_failed_syntax_error"
    assert padded.hidden_tests == s.hidden_tests


# ---------------------------------------------------------------------------
# Tests 4-5: variant_type / bug_type post-processing
# ---------------------------------------------------------------------------

def test_variant_type_post_processing():
    """Each source maps to the correct variant_type and bug_type."""
    code_s = _make_sample(
        sample_id="mbpp_42_code",
        task_type="code_generation",
    )
    boundary_s = _make_sample(
        sample_id="mbpp_42_boundary",
        task_type="code_generation",
    )
    static_s = _make_sample(
        sample_id="mbpp_42_sr_condition_error",
        task_type="static_repair",
        broken_code="def add(a, b):\n    return a - b\n",
    )
    exec_s = _make_sample(
        sample_id="mbpp_42_er_off_by_one",
        task_type="execution_repair",
        broken_code="def add(a, b):\n    return a - b\n",
        execution_feedback="FAILED",
    )

    code_out = post_process_variant_type(code_s, "code")
    boundary_out = post_process_variant_type(boundary_s, "boundary")
    static_out = post_process_variant_type(static_s, "static_repair")
    exec_out = post_process_variant_type(exec_s, "execution_repair")

    assert code_out.variant_type == "code"
    assert code_out.bug_type is None
    assert code_out.source_split == "test"

    assert boundary_out.variant_type == "boundary"
    assert boundary_out.bug_type is None

    assert static_out.variant_type == "static_repair"
    assert static_out.bug_type == "condition_error"

    assert exec_out.variant_type == "execution_repair"
    assert exec_out.bug_type == "off_by_one"


def test_bug_type_extraction_from_sample_id():
    """Regex extracts bug_type from sample_id suffix; None on no match."""
    assert extract_bug_type("mbpp_42_sr_condition_error") == "condition_error"
    assert extract_bug_type("mbpp_42_er_off_by_one") == "off_by_one"
    assert extract_bug_type("mbpp_42_plain") is None
    assert extract_bug_type("mbpp_42_sr") is None  # sr_ requires suffix


# ---------------------------------------------------------------------------
# Test 6: canary sample
# ---------------------------------------------------------------------------

def test_canary_sample_fails_verification():
    """Canary sample's verify_sample.is_accepted must be False."""
    source = _make_sample()
    canary = make_canary(source)
    assert canary.target_code == CANARY_CODE
    assert canary.variant_type == "canary"
    assert canary.task_type == "code_generation"
    assert canary.sample_id.endswith("_canary")
    sv = verify_sample(canary, run_ruff=False)
    assert sv.is_accepted is False, (
        f"canary must fail verification; messages: {sv.messages}"
    )


# ---------------------------------------------------------------------------
# Test 7: qualified family
# ---------------------------------------------------------------------------

def test_qualified_family_passes_all_gates():
    """Synthetic family with 5 valid samples → qualified, no rejections."""
    fid = "mbpp_fam_42"
    samples_by_source = {
        "code": [_add_code_sample(family_id=fid)],
        "boundary": [_add_boundary_sample(family_id=fid)],
        "static_repair": [_add_static_repair_sample(family_id=fid)],
        "execution_repair": [_add_execution_repair_sample(family_id=fid)],
    }
    canary = make_canary(_make_sample(family_id=fid))
    qualified, all_samples, rejected = verify_family(
        fid, samples_by_source, canary, pytest_timeout_s=10.0
    )
    assert qualified, f"family should be qualified; rejections: {rejected}"
    assert len(all_samples) == 5  # 4 source samples + 1 canary
    assert rejected == []


# ---------------------------------------------------------------------------
# Test 8: rejected family
# ---------------------------------------------------------------------------

def test_rejected_family_recorded(tmp_path):
    """Family with one failing sample → rejected, sample in rejected.jsonl."""
    fid = "mbpp_fam_43"
    # Construct a code sample with a broken target_code (syntax error).
    bad_code_sample = _make_sample(
        sample_id="mbpp_43_code",
        family_id=fid,
        target_code="def add(a, b):\n    return a +\n",  # SyntaxError
        variant_type="code",
        source_split="test",
    )
    samples_by_source = {
        "code": [bad_code_sample],
        "boundary": [],
        "static_repair": [],
        "execution_repair": [],
    }
    canary = make_canary(_make_sample(family_id=fid))
    qualified, all_samples, rejected = verify_family(
        fid, samples_by_source, canary, pytest_timeout_s=10.0
    )
    assert not qualified
    # The failing code sample must appear in rejected records.
    rejected_ids = {r["sample_id"] for r in rejected}
    assert "mbpp_43_code" in rejected_ids
    # Write rejected.jsonl and verify the records are present.
    rejected_path = tmp_path / "rejected.jsonl"
    write_rejected_jsonl(rejected_path, rejected)
    assert rejected_path.exists()
    lines = rejected_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(rejected)
    parsed = [json.loads(line) for line in lines]
    assert any(r["sample_id"] == "mbpp_43_code" for r in parsed)


# ---------------------------------------------------------------------------
# Tests 9-10: freeze decision
# ---------------------------------------------------------------------------

def test_freeze_decision_100_or_more():
    """110 qualified families → freeze first 100, surplus 10 revert to available."""
    reg = FamilyRegistry()
    fam_ids = [f"mbpp_fam_{i:03d}" for i in range(110)]
    for fid in fam_ids:
        reg.families[fid] = FamilyEntry(
            family_id=fid,
            source_task_id=fid.replace("mbpp_fam_", "mbpp_"),
            source_split="test",
            usage=["frozen_v3_candidate"],
        )

    frozen, surplus, decision, note = apply_freeze_decision(fam_ids)
    assert decision == "freeze_100"
    assert len(frozen) == 100
    assert len(surplus) == 10
    # Frozen list must be sorted ascending (first 100 by family_id).
    assert frozen == sorted(fam_ids)[:100]
    assert surplus == sorted(fam_ids)[100:]

    # Update registry: frozen get frozen_v3 tag (keep candidate), surplus lose candidate.
    update_registry(reg, frozen, surplus, rejected_family_ids=[])

    for fid in frozen:
        assert reg.families[fid].has_usage("frozen_v3")
        assert reg.families[fid].has_usage("frozen_v3_candidate")

    for fid in surplus:
        assert not reg.families[fid].has_usage("frozen_v3_candidate")
        assert not reg.families[fid].has_usage("frozen_v3")


def test_freeze_decision_less_than_80_aborts(tmp_path, monkeypatch):
    """70 qualified families → script exits 1, only rejected.jsonl written."""
    # Build minimal candidates.json with 70 entries.
    candidates = [f"mbpp_fam_{i:03d}" for i in range(70)]
    cand_path = tmp_path / "candidates.json"
    cand_path.write_text(
        json.dumps({"candidates": candidates}), encoding="utf-8"
    )

    # Build verified/test.jsonl with 70 source samples (one per candidate).
    verified_dir = tmp_path / "verified"
    verified_dir.mkdir()
    test_jsonl = verified_dir / "test.jsonl"
    with test_jsonl.open("w", encoding="utf-8") as fh:
        for fid in candidates:
            s = _make_sample(
                sample_id=fid.replace("mbpp_fam_", "mbpp_"),
                family_id=fid,
            )
            fh.write(s.to_json_line() + "\n")

    # Build registry with all 70 candidates tagged frozen_v3_candidate.
    reg_path = tmp_path / "registry.json"
    reg = FamilyRegistry()
    for fid in candidates:
        reg.families[fid] = FamilyEntry(
            family_id=fid,
            source_task_id=fid.replace("mbpp_fam_", "mbpp_"),
            source_split="test",
            usage=["frozen_v3_candidate"],
        )
    reg.to_path(reg_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Mock process_family to return qualified=True with one synthetic sample
    # for each family. This avoids slow real pytest runs while still
    # producing 70 qualified families (< MIN_FROZEN=80 → fix_first branch).
    def _mock_process(family_id, source, *, timeout_s, seed):
        s = _make_sample(
            sample_id=f"{family_id}_code",
            family_id=family_id,
            variant_type="code",
            source_split="test",
        )
        canary = make_canary(s)
        return True, [s, canary], []
    monkeypatch.setattr(
        "scripts.build_frozen_v3_samples.process_family", _mock_process
    )

    # Invoke main with CLI args.
    sys.argv = [
        "build_frozen_v3_samples.py",
        "--candidates", str(cand_path),
        "--mbpp-verified-dir", str(verified_dir),
        "--output-dir", str(out_dir),
        "--registry", str(reg_path),
        "--seed", "42",
        "--timeout", "10.0",
    ]
    from scripts.build_frozen_v3_samples import main
    rc = main()
    assert rc == 1
    # Only rejected.jsonl should exist in the fix_first branch.
    assert (out_dir / "rejected.jsonl").exists()
    assert not (out_dir / "families.json").exists()
    assert not (out_dir / "test_raw.jsonl").exists()
    assert not (out_dir / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Test 11: sha_lock consistency
# ---------------------------------------------------------------------------

def test_manifest_sha_lock_consistent(tmp_path):
    """After writing all 4 files, re-read and verify sha_lock matches."""
    fid = "mbpp_fam_42"
    samples = [_add_code_sample(family_id=fid)]

    families_path = tmp_path / "families.json"
    test_raw_path = tmp_path / "test_raw.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    manifest_path = tmp_path / "manifest.json"

    write_families_json(
        families_path, [fid], "freeze_actual", "Froze all 1."
    )
    write_test_raw_jsonl(test_raw_path, samples)
    write_rejected_jsonl(rejected_path, [])

    from scripts.build_frozen_v3_samples import variant_breakdown
    vb = variant_breakdown(samples)
    write_manifest_json(
        manifest_path,
        frozen_family_count=1,
        total_sample_count=len(samples),
        variant_breakdown=vb,
        test_raw_path=test_raw_path,
        families_path=families_path,
        rejected_path=rejected_path,
        rejected_count=0,
        decision="freeze_actual",
        note="Froze all 1.",
    )

    # verify_sha_lock reads manifest + 3 non-manifest files and asserts match.
    verify_sha_lock(tmp_path)  # Should not raise.

    # Also recompute manually and compare to the stored value.
    recomputed = compute_sha_lock(families_path, test_raw_path, rejected_path)
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    stored = manifest["immutability"]["sha_lock"]
    assert recomputed == stored


# ---------------------------------------------------------------------------
# Test 12: unclaim
# ---------------------------------------------------------------------------

def test_unclaim_removes_tag():
    """registry.unclaim removes tag from family's usage list; no-op if absent."""
    reg = FamilyRegistry()
    reg.families["mbpp_fam_42"] = FamilyEntry(
        family_id="mbpp_fam_42",
        source_task_id="mbpp_42",
        source_split="test",
        usage=["frozen_v3_candidate", "p2_train"],
    )
    # Remove existing tag.
    reg.unclaim("mbpp_fam_42", "frozen_v3_candidate")
    assert not reg.families["mbpp_fam_42"].has_usage("frozen_v3_candidate")
    assert reg.families["mbpp_fam_42"].has_usage("p2_train")

    # Remove again (no-op, idempotent).
    reg.unclaim("mbpp_fam_42", "frozen_v3_candidate")
    assert not reg.families["mbpp_fam_42"].has_usage("frozen_v3_candidate")

    # unclaim on unknown family_id is a silent no-op.
    reg.unclaim("nonexistent", "frozen_v3_candidate")
