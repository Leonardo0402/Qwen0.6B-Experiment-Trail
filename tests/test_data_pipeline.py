"""
tests/test_data_pipeline.py -- Integration tests for verify_samples + build_dataset.

Coverage
--------
verify_samples:
  - known-good sample → accepted (verified=True, verification.pytest_ok=True)
  - known-bad sample (target fails tests) → rejected
  - mixed batch → correct split into accepted/rejected

build_dataset:
  - end-to-end on a small in-memory verified pool → 3 split files + manifest.json
  - family disjointness: no family_id appears in both train and test splits
  - dedup: duplicate samples count as one
  - total sample count in manifest == verified pool size
"""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_dataset import build_and_write_dataset, load_samples  # noqa: E402
from scripts.generate_tasks import TASK_FAMILIES, family_to_sample  # noqa: E402
from scripts.verify_samples import VerifyResult, verify_jsonl_samples  # noqa: E402
from src.dataset_builder import dedup, split_by_family  # noqa: E402
from src.schemas import Sample, Verification  # noqa: E402
from src.validators import verify_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_CODE = "def add(a, b):\n    return a + b\n"
_GOOD_TESTS = (
    "from solution import add\n\n"
    "def test_add():\n"
    "    assert add(1, 2) == 3\n"
    "def test_add_zero():\n"
    "    assert add(0, 0) == 0\n"
)
_BAD_CODE = "def add(a, b):\n    return a - b\n"   # wrong operator → fails tests


def _blank_verification() -> Verification:
    return Verification(syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False)


def _make_sample(
    sample_id: str,
    family_id: str,
    difficulty: int = 1,
    target_code: str = _GOOD_CODE,
    public_tests: str = _GOOD_TESTS,
    hidden_tests: str = "",
) -> Sample:
    return Sample(
        sample_id=sample_id,
        family_id=family_id,
        difficulty=difficulty,
        task_type="code_generation",
        language="python",
        skill_tags=["arithmetic"],
        instruction="Write an add function.",
        broken_code=None,
        execution_feedback=None,
        target_code=target_code,
        public_tests=public_tests,
        hidden_tests=hidden_tests,
        verified=False,
        verification=_blank_verification(),
        generator="test",
        created_at="2024-01-01T00:00:00+00:00",
        dataset_version="v1.0",
    )


# ---------------------------------------------------------------------------
# verify_samples tests
# ---------------------------------------------------------------------------


def test_verify_good_sample_accepted():
    """A known-good sample lands in verified bucket with correct flags."""
    s = _make_sample("good-1", "fam-good")
    vr = verify_jsonl_samples([s], run_ruff=False)
    assert vr.n_verified == 1
    assert vr.n_rejected == 0
    accepted = vr.accepted[0]
    assert accepted.verified is True
    assert accepted.verification.syntax_ok is True
    assert accepted.verification.pytest_ok is True


def test_verify_bad_sample_rejected():
    """A sample whose target_code fails its own tests is rejected."""
    s = _make_sample("bad-1", "fam-bad", target_code=_BAD_CODE)
    vr = verify_jsonl_samples([s], run_ruff=False)
    assert vr.n_verified == 0
    assert vr.n_rejected == 1


def test_verify_mixed_batch():
    """Batch with 2 good + 1 bad → 2 accepted, 1 rejected."""
    good1 = _make_sample("good-1", "fam-a")
    good2 = _make_sample("good-2", "fam-b")
    bad = _make_sample("bad-1", "fam-c", target_code=_BAD_CODE)
    vr = verify_jsonl_samples([good1, good2, bad], run_ruff=False)
    assert vr.n_verified == 2
    assert vr.n_rejected == 1


def test_verify_result_properties():
    """VerifyResult n_in == n_verified + n_rejected."""
    good = _make_sample("g-1", "fam-g")
    bad = _make_sample("b-1", "fam-b", target_code=_BAD_CODE)
    vr = verify_jsonl_samples([good, bad], run_ruff=False)
    assert vr.n_in == 2
    assert vr.n_in == vr.n_verified + vr.n_rejected


def test_verify_reasons_histogram_non_empty_on_rejection():
    """A rejection leaves a non-empty reasons histogram."""
    bad = _make_sample("b-1", "fam-b", target_code=_BAD_CODE)
    vr = verify_jsonl_samples([bad], run_ruff=False)
    assert vr.n_rejected == 1
    assert len(vr.reasons) >= 1


def test_verify_accepted_sample_verified_flag_true():
    """After verify_jsonl_samples, accepted sample has verified=True."""
    s = _make_sample("g-1", "fam-g")
    assert s.verified is False  # pre-condition
    vr = verify_jsonl_samples([s], run_ruff=False)
    assert vr.accepted[0].verified is True


# ---------------------------------------------------------------------------
# build_dataset tests
# ---------------------------------------------------------------------------


def _build_verified_pool() -> list[Sample]:
    """Create a small verified pool from the actual task families."""
    samples = []
    for fam in TASK_FAMILIES:
        s = family_to_sample(fam)
        result = verify_sample(s, run_ruff=False)
        if result.is_accepted:
            s.verified = True
            s.verification = result.verification
            samples.append(s)
    return samples


def test_build_dataset_creates_split_files():
    """build_and_write_dataset creates train/validation/test JSONL files."""
    verified = _build_verified_pool()
    assert len(verified) >= 3, "Need at least 3 verified samples"

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manifest = build_and_write_dataset(
                verified, out, train=0.6, val=0.2, test=0.2, seed=42
            )

        assert (out / "train.jsonl").exists()
        assert (out / "validation.jsonl").exists()
        assert (out / "test.jsonl").exists()
        assert (out / "manifest.json").exists()


def test_build_dataset_manifest_structure():
    """manifest.json has expected keys."""
    verified = _build_verified_pool()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manifest = build_and_write_dataset(
                verified, out, train=0.6, val=0.2, test=0.2, seed=42
            )

        for key in ("train", "validation", "test", "dataset_hash", "seed", "created_at"):
            assert key in manifest, f"Missing manifest key: {key}"

        for split_name in ("train", "validation", "test"):
            split_meta = manifest[split_name]
            assert "sample_count" in split_meta
            assert "family_count" in split_meta
            assert "difficulty_mix" in split_meta


def test_build_dataset_total_count():
    """Total samples across splits equals the verified pool size (after dedup)."""
    verified = _build_verified_pool()
    deduped = dedup(verified)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manifest = build_and_write_dataset(
                deduped, out, train=0.6, val=0.2, test=0.2, seed=42
            )

        total = (
            manifest["train"]["sample_count"]
            + manifest["validation"]["sample_count"]
            + manifest["test"]["sample_count"]
        )
        assert total == len(deduped)


def test_build_dataset_family_disjoint():
    """No family_id appears in both train and test splits."""
    verified = _build_verified_pool()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build_and_write_dataset(
                verified, out, train=0.7, val=0.1, test=0.2, seed=42
            )

    # Verify via split_by_family directly (the guarantee is algebraic)
    split = split_by_family(dedup(verified), train=0.7, val=0.1, test=0.2, seed=42)
    train_fams = {s.family_id for s in split.train}
    test_fams = {s.family_id for s in split.test}
    assert train_fams.isdisjoint(test_fams), (
        f"Train/test families overlap: {train_fams & test_fams}"
    )


def test_build_dataset_dedup_removes_duplicates():
    """Two identical samples from the same family are counted as one."""
    s1 = _make_sample("dup-1", "fam-dup")
    s2 = _make_sample("dup-2", "fam-dup", target_code=_GOOD_CODE)
    # Same content (same target_code, same instruction, same task_type) → 1 after dedup
    deduped = dedup([s1, s2])
    assert len(deduped) == 1


def test_build_dataset_jsonl_parseable():
    """Each line in train.jsonl is valid JSON with 'messages' key."""
    verified = _build_verified_pool()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build_and_write_dataset(
                verified, out, train=0.6, val=0.2, test=0.2, seed=42
            )
        train_path = out / "train.jsonl"
        lines = train_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                record = json.loads(line)
                assert "messages" in record
                assert len(record["messages"]) == 3
                roles = [m["role"] for m in record["messages"]]
                assert roles == ["system", "user", "assistant"]


def test_load_samples_raises_on_missing_file():
    """load_samples raises FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_samples([Path("nonexistent_file_xyz.jsonl")])


def test_build_dataset_with_heldout():
    """Heldout families never appear in train/val/test."""
    verified = _build_verified_pool()
    if len(verified) < 3:
        pytest.skip("Not enough samples for heldout test")

    heldout_fam = verified[0].family_id
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manifest = build_and_write_dataset(
                verified,
                out,
                train=0.7,
                val=0.1,
                test=0.2,
                heldout={heldout_fam},
                seed=42,
            )

    # Heldout samples must not be in train/val/test
    split = split_by_family(
        dedup(verified),
        train=0.7,
        val=0.1,
        test=0.2,
        heldout_family_ids={heldout_fam},
        seed=42,
    )
    all_active_fams = (
        {s.family_id for s in split.train}
        | {s.family_id for s in split.val}
        | {s.family_id for s in split.test}
    )
    assert heldout_fam not in all_active_fams
