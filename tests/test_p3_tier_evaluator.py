"""Tests for src/p3_tier_evaluator.py (Issue #14 Wave 3-D).

Covers
------
- P3.2: ``select_probe_samples`` reproducibility (fixed seed → same SHA).
- P3.4: ``EarlyStoppingManager`` rules (NaN/Inf, syntax drop, timeout,
  bucket missing, pending→confirmed, max_epochs).
- P3.5: ``CheckpointEvidence`` format (SHA fields, CRLF→LF normalization,
  JSONL append).
- P3.2/P3.3: ``CompositeCoverageError`` on missing bucket (hard fail).
- Mock model.generate for Tier 2/3 execution (no real GPU needed).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Torch availability check for tests that call .run() (which lazy-imports torch).
# CPU-only CI environments may not have torch installed.
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_TORCH_SKIP_REASON = "torch not installed (CPU-only CI environment)"

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import EvalOutcome  # noqa: E402
from src.p3_checkpoint_evaluator import (  # noqa: E402
    CompositeCoverageError,
    CompositeScore,
    FullValidationResult,
    ProbeResult,
)
from src.p3_tier_evaluator import (  # noqa: E402
    PROBE_GENERATION_CONFIG,
    CheckpointEvidence,
    EarlyStoppingManager,
    Tier2Probe,
    Tier3FullValidation,
    build_eval_messages,
    compute_probe_sha,
    extract_code_block,
    select_probe_samples,
)
from src.schemas import Sample, Verification  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(
    sample_id: str = "s1",
    family_id: str = "fam1",
    variant_type: str = "code",
    task_type: str = "code_generation",
    *,
    broken_code: str | None = None,
    execution_feedback: str | None = None,
) -> Sample:
    """Build a minimal valid Sample for testing."""
    is_repair = task_type in ("static_repair", "execution_repair")
    if is_repair and broken_code is None:
        broken_code = "def solution():\n    pass"
    return Sample(
        sample_id=sample_id,
        family_id=family_id,
        difficulty=1,
        task_type=task_type,
        language="python",
        skill_tags=["test"],
        instruction="Write a function that returns 42.",
        broken_code=broken_code,
        execution_feedback=execution_feedback,
        target_code="def solution():\n    return 42",
        public_tests="def test_solution():\n    assert solution() == 42",
        hidden_tests="def test_hidden():\n    assert solution() == 42",
        verified=True,
        verification=Verification(
            syntax_ok=True, pytest_ok=True, ruff_ok=True, timeout=False,
        ),
        generator="test",
        created_at="2026-07-05T00:00:00+00:00",
        dataset_version="test-v1",
        variant_type=variant_type,
    )


def _make_validation_samples(n_per_bucket: int = 12) -> list[Sample]:
    """Build 4*n_per_bucket samples, evenly split across variant types."""
    samples: list[Sample] = []
    for i in range(n_per_bucket):
        samples.append(_make_sample(
            sample_id=f"code_{i}", family_id=f"fam_code_{i}",
            variant_type="code", task_type="code_generation",
        ))
    for i in range(n_per_bucket):
        samples.append(_make_sample(
            sample_id=f"boundary_{i}", family_id=f"fam_boundary_{i}",
            variant_type="boundary", task_type="code_generation",
        ))
    for i in range(n_per_bucket):
        samples.append(_make_sample(
            sample_id=f"static_{i}", family_id=f"fam_static_{i}",
            variant_type="static_repair", task_type="static_repair",
        ))
    for i in range(n_per_bucket):
        samples.append(_make_sample(
            sample_id=f"exec_{i}", family_id=f"fam_exec_{i}",
            variant_type="execution_repair", task_type="execution_repair",
            execution_feedback="AssertionError: expected 42",
        ))
    return samples


def _make_config(
    probe_size: int = 8,
    probe_seed: int = 42,
    probe_patience: int = 4,
    probe_min_delta: float = 0.005,
    max_epochs: int = 3,
) -> dict:
    """Build a minimal config dict for Tier2Probe / Tier3FullValidation."""
    return {
        "checkpoint_evaluator": {
            "tier1": {"interval_steps": 50, "metrics": ["train_loss"]},
            "tier2": {
                "interval_epoch_fraction": 0.25,
                "probe_size": probe_size,
                "probe_stratify_by": "variant_type",
                "probe_seed": probe_seed,
                "composite_score": True,
            },
            "tier3": {
                "interval_epochs": 1,
                "full_validation": True,
                "composite_score": True,
            },
        },
        "composite_score": {
            "code_generation_pass_at_1": 0.30,
            "boundary_pass_at_1": 0.15,
            "static_repair_success": 0.20,
            "execution_repair_success": 0.25,
            "hidden_pass_rate": 0.10,
            "hard_constraint": {
                "code_generation_drop_vs_p2_final_max_pct": 3.0,
            },
        },
        "early_stopping": {
            "enabled": True,
            "probe_patience": probe_patience,
            "probe_min_delta": probe_min_delta,
            "full_validation_confirm": True,
            "divergence_nan_inf": True,
            "max_epochs": max_epochs,
        },
        "best_checkpoint": {
            "selection_metric": "full_validation_composite",
            "never_use": ["frozen_v4", "probe"],
        },
    }


def _make_probe_result(
    step: int = 0,
    epoch: float = 0.25,
    composite_value: float = 0.5,
) -> ProbeResult:
    return ProbeResult(
        step=step,
        epoch=epoch,
        probe_sample_ids=["s1", "s2"],
        composite_score=CompositeScore(
            code_generation_pass_at_1=0.5,
            boundary_pass_at_1=0.5,
            static_repair_success=0.5,
            execution_repair_success=0.5,
            hidden_pass_rate=0.5,
        ),
        composite_value=composite_value,
        metrics={"pass_at_1": 0.5, "syntax_rate": 1.0, "timeout_rate": 0.0},
    )


def _make_full_result(
    step: int = 0,
    epoch: int = 1,
    composite_value: float = 0.5,
) -> FullValidationResult:
    return FullValidationResult(
        step=step,
        epoch=epoch,
        composite_score=CompositeScore(
            code_generation_pass_at_1=0.5,
            boundary_pass_at_1=0.5,
            static_repair_success=0.5,
            execution_repair_success=0.5,
            hidden_pass_rate=0.5,
        ),
        composite_value=composite_value,
        metrics={"pass_at_1": 0.5, "syntax_rate": 1.0, "timeout_rate": 0.0},
        hard_constraint_pass=True,
        hard_constraint_violations=[],
    )


# ---------------------------------------------------------------------------
# P3.2: select_probe_samples reproducibility
# ---------------------------------------------------------------------------


class TestSelectProbeSamples:
    """Probe sampling must be deterministic given fixed seed."""

    def test_same_seed_same_samples(self) -> None:
        samples = _make_validation_samples(12)
        a = select_probe_samples(samples, probe_size=8, seed=42)
        b = select_probe_samples(samples, probe_size=8, seed=42)
        assert [s.sample_id for s in a] == [s.sample_id for s in b]

    def test_same_seed_same_sha(self) -> None:
        samples = _make_validation_samples(12)
        a = select_probe_samples(samples, probe_size=8, seed=42)
        b = select_probe_samples(samples, probe_size=8, seed=42)
        sha_a = compute_probe_sha([s.sample_id for s in a], PROBE_GENERATION_CONFIG)
        sha_b = compute_probe_sha([s.sample_id for s in b], PROBE_GENERATION_CONFIG)
        assert sha_a == sha_b

    def test_different_seed_different_samples(self) -> None:
        samples = _make_validation_samples(12)
        a = select_probe_samples(samples, probe_size=8, seed=42)
        b = select_probe_samples(samples, probe_size=8, seed=99)
        # Very unlikely two different seeds pick the same sample set
        assert [s.sample_id for s in a] != [s.sample_id for s in b]

    def test_probe_size_close_to_target(self) -> None:
        samples = _make_validation_samples(12)
        # probe_size=8 → 2 per bucket → 8 total
        probe = select_probe_samples(samples, probe_size=8, seed=42)
        assert len(probe) == 8

    def test_probe_stratified_across_all_four_buckets(self) -> None:
        samples = _make_validation_samples(12)
        probe = select_probe_samples(samples, probe_size=8, seed=42)
        buckets = {s.variant_type for s in probe}
        assert buckets == {"code", "boundary", "static_repair", "execution_repair"}

    def test_probe_handles_small_pool(self) -> None:
        """If pool < target, min(target, len(pool)) is used per bucket."""
        samples = _make_validation_samples(1)  # 1 per bucket, 4 total
        probe = select_probe_samples(samples, probe_size=8, seed=42)
        # Each bucket has only 1 sample, so probe = 4
        assert len(probe) == 4

    def test_probe_sha_locks_generation_config(self) -> None:
        """Different generation config → different SHA."""
        ids = ["s1", "s2"]
        sha_a = compute_probe_sha(ids, {"temperature": 0, "max_new_tokens": 384})
        sha_b = compute_probe_sha(ids, {"temperature": 0, "max_new_tokens": 256})
        assert sha_a != sha_b

    def test_probe_sha_sorts_sample_ids(self) -> None:
        """SHA is invariant to sample_id order."""
        sha_a = compute_probe_sha(["s2", "s1"], {"temp": 0})
        sha_b = compute_probe_sha(["s1", "s2"], {"temp": 0})
        assert sha_a == sha_b


# ---------------------------------------------------------------------------
# P3.4: EarlyStoppingManager rules
# ---------------------------------------------------------------------------


class TestEarlyStoppingNaNInf:
    """NaN/Inf → immediate confirmed stop."""

    def test_nan_inf_immediate_stop(self) -> None:
        cfg = _make_config()
        mgr = EarlyStoppingManager(cfg)
        stop, reason = mgr.check_nan_inf(True)
        assert stop is True
        assert "nan" in reason.lower() or "inf" in reason.lower()
        assert mgr.confirmed_stop is True

    def test_no_nan_inf_no_stop(self) -> None:
        cfg = _make_config()
        mgr = EarlyStoppingManager(cfg)
        stop, _ = mgr.check_nan_inf(False)
        assert stop is False
        assert mgr.confirmed_stop is False


class TestEarlyStoppingBucketMissing:
    """Required bucket missing → immediate confirmed stop."""

    def test_bucket_missing_immediate_stop(self) -> None:
        cfg = _make_config()
        mgr = EarlyStoppingManager(cfg)
        err = CompositeCoverageError("missing bucket: code")
        mgr.record_bucket_missing(err)
        assert mgr.confirmed_stop is True
        assert "bucket" in mgr.stop_reason.lower()
        assert "FIX_FIRST" in mgr.stop_reason


class TestEarlyStoppingSyntaxDrop:
    """Syntax drop > 5pp → immediate stop."""

    def test_syntax_drop_above_5pp_stops(self) -> None:
        cfg = _make_config()
        mgr = EarlyStoppingManager(cfg)
        # First probe: syntax_rate = 1.0
        probe1 = _make_probe_result(epoch=0.25)
        mgr.record_probe(probe1, {"syntax_rate": 1.0, "timeout_rate": 0.0})
        assert not mgr.confirmed_stop
        # Second probe: syntax_rate = 0.90 (drop = 10pp > 5pp)
        probe2 = _make_probe_result(epoch=0.50)
        stop, reason = mgr.record_probe(probe2, {"syntax_rate": 0.90, "timeout_rate": 0.0})
        assert stop is True
        assert "syntax" in reason.lower()

    def test_syntax_drop_below_5pp_no_stop(self) -> None:
        cfg = _make_config()
        mgr = EarlyStoppingManager(cfg)
        probe1 = _make_probe_result(epoch=0.25)
        mgr.record_probe(probe1, {"syntax_rate": 1.0, "timeout_rate": 0.0})
        # Drop = 3pp < 5pp → no stop
        probe2 = _make_probe_result(epoch=0.50)
        stop, _ = mgr.record_probe(probe2, {"syntax_rate": 0.97, "timeout_rate": 0.0})
        assert stop is False


class TestEarlyStoppingTimeout:
    """Timeout > 8% → immediate stop."""

    def test_timeout_above_8pct_stops(self) -> None:
        cfg = _make_config()
        mgr = EarlyStoppingManager(cfg)
        probe = _make_probe_result(epoch=0.25)
        stop, reason = mgr.record_probe(probe, {"timeout_rate": 0.10, "syntax_rate": 1.0})
        assert stop is True
        assert "timeout" in reason.lower()

    def test_timeout_below_8pct_no_stop(self) -> None:
        cfg = _make_config()
        mgr = EarlyStoppingManager(cfg)
        probe = _make_probe_result(epoch=0.25)
        stop, _ = mgr.record_probe(probe, {"timeout_rate": 0.05, "syntax_rate": 1.0})
        assert stop is False


class TestEarlyStoppingProbePatience:
    """probe_patience consecutive no-improvement → pending stop."""

    def test_pending_stop_after_patience(self) -> None:
        cfg = _make_config(probe_patience=2, probe_min_delta=0.005)
        mgr = EarlyStoppingManager(cfg)
        # Probe 1: best = 0.50
        mgr.record_probe(_make_probe_result(epoch=0.25, composite_value=0.50),
                         {"syntax_rate": 1.0, "timeout_rate": 0.0})
        assert not mgr.pending_stop
        # Probe 2: no improvement (0.50, not > 0.50 + 0.005)
        mgr.record_probe(_make_probe_result(epoch=0.50, composite_value=0.50),
                         {"syntax_rate": 1.0, "timeout_rate": 0.0})
        assert mgr._consecutive_no_improve == 1
        assert not mgr.pending_stop
        # Probe 3: still no improvement → pending
        mgr.record_probe(_make_probe_result(epoch=0.75, composite_value=0.50),
                         {"syntax_rate": 1.0, "timeout_rate": 0.0})
        assert mgr._consecutive_no_improve == 2
        assert mgr.pending_stop is True
        assert "pending" in mgr.stop_reason.lower()

    def test_improvement_resets_counter(self) -> None:
        cfg = _make_config(probe_patience=2, probe_min_delta=0.005)
        mgr = EarlyStoppingManager(cfg)
        mgr.record_probe(_make_probe_result(epoch=0.25, composite_value=0.50),
                         {"syntax_rate": 1.0, "timeout_rate": 0.0})
        mgr.record_probe(_make_probe_result(epoch=0.50, composite_value=0.50),
                         {"syntax_rate": 1.0, "timeout_rate": 0.0})
        assert mgr._consecutive_no_improve == 1
        # Improvement: 0.52 > 0.50 + 0.005
        mgr.record_probe(_make_probe_result(epoch=0.75, composite_value=0.52),
                         {"syntax_rate": 1.0, "timeout_rate": 0.0})
        assert mgr._consecutive_no_improve == 0
        assert not mgr.pending_stop


class TestEarlyStoppingConfirmed:
    """Pending + full validation no improvement → confirmed."""

    def test_pending_then_full_no_improve_confirms(self) -> None:
        cfg = _make_config(probe_patience=1, probe_min_delta=0.005)
        mgr = EarlyStoppingManager(cfg)
        # Probe 1: best = 0.50
        mgr.record_probe(_make_probe_result(epoch=0.25, composite_value=0.50),
                         {"syntax_rate": 1.0, "timeout_rate": 0.0})
        # Probe 2: no improvement → pending
        mgr.record_probe(_make_probe_result(epoch=0.50, composite_value=0.50),
                         {"syntax_rate": 1.0, "timeout_rate": 0.0})
        assert mgr.pending_stop is True
        # Full validation 1: sets best_full
        mgr.record_full(_make_full_result(epoch=1, composite_value=0.50),
                        {"syntax_rate": 1.0, "timeout_rate": 0.0})
        # Full validation 2: no improvement → confirmed
        stop, reason = mgr.record_full(
            _make_full_result(epoch=2, composite_value=0.50),
            {"syntax_rate": 1.0, "timeout_rate": 0.0},
        )
        assert stop is True
        assert mgr.confirmed_stop is True
        assert "confirmed" in reason.lower()

    def test_pending_then_full_improvement_clears_pending(self) -> None:
        cfg = _make_config(probe_patience=1, probe_min_delta=0.005)
        mgr = EarlyStoppingManager(cfg)
        mgr.record_probe(_make_probe_result(epoch=0.25, composite_value=0.50),
                         {"syntax_rate": 1.0, "timeout_rate": 0.0})
        mgr.record_probe(_make_probe_result(epoch=0.50, composite_value=0.50),
                         {"syntax_rate": 1.0, "timeout_rate": 0.0})
        assert mgr.pending_stop is True
        # Full validation improves → clear pending
        mgr.record_full(_make_full_result(epoch=1, composite_value=0.60),
                        {"syntax_rate": 1.0, "timeout_rate": 0.0})
        assert mgr.pending_stop is False


class TestEarlyStoppingMaxEpochs:
    """max_epochs reached → confirmed stop."""

    def test_max_epochs_stops(self) -> None:
        cfg = _make_config(max_epochs=2)
        mgr = EarlyStoppingManager(cfg)
        stop, reason = mgr.record_full(
            _make_full_result(epoch=2, composite_value=0.50),
            {"syntax_rate": 1.0, "timeout_rate": 0.0},
        )
        assert stop is True
        assert "max_epochs" in reason.lower()


class TestEarlyStoppingPersistence:
    """to_dict / from_dict round-trip preserves state."""

    def test_round_trip_preserves_state(self) -> None:
        cfg = _make_config()
        mgr = EarlyStoppingManager(cfg)
        mgr._best_probe = 0.55
        mgr._best_full = 0.60
        mgr._consecutive_no_improve = 2
        mgr.pending_stop = True
        mgr.stop_reason = "test pending"

        d = mgr.to_dict()
        restored = EarlyStoppingManager.from_dict(cfg, d)

        assert restored._best_probe == 0.55
        assert restored._best_full == 0.60
        assert restored._consecutive_no_improve == 2
        assert restored.pending_stop is True
        assert restored.stop_reason == "test pending"


# ---------------------------------------------------------------------------
# P3.5: CheckpointEvidence
# ---------------------------------------------------------------------------


class TestCheckpointEvidence:
    """Checkpoint evidence records SHA fields and appends JSONL."""

    def test_record_appends_jsonl(self, tmp_path: Path) -> None:
        ev = CheckpointEvidence(tmp_path)
        # Create fake checkpoint files
        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapter_model.safetensors").write_bytes(b"weights")
        config_path = tmp_path / "config.yaml"
        config_path.write_bytes(b"config\n")
        train_file = tmp_path / "train.jsonl"
        train_file.write_bytes(b"train\n")
        val_file = tmp_path / "val.jsonl"
        val_file.write_bytes(b"val\n")

        ev.record(
            checkpoint_path=ckpt_dir,
            config_path=config_path,
            train_file=train_file,
            validation_file=val_file,
            generation_config=PROBE_GENERATION_CONFIG,
            metrics={"pass_at_1": 0.5},
        )

        evidence_path = tmp_path / "checkpoint-evidence.jsonl"
        assert evidence_path.exists()
        lines = evidence_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "weight_sha" in entry
        assert "config_sha" in entry
        assert "train_data_sha" in entry
        assert "validation_sha" in entry
        assert "generation_config_sha" in entry
        assert "metrics_sha" in entry
        assert "created_at" in entry
        assert entry["weight_sha"]  # non-empty

    def test_multiple_records_append(self, tmp_path: Path) -> None:
        ev = CheckpointEvidence(tmp_path)
        ckpt_dir1 = tmp_path / "checkpoint-100"
        ckpt_dir1.mkdir()
        (ckpt_dir1 / "adapter_model.safetensors").write_bytes(b"w1")
        ckpt_dir2 = tmp_path / "checkpoint-200"
        ckpt_dir2.mkdir()
        (ckpt_dir2 / "adapter_model.safetensors").write_bytes(b"w2")

        for ckpt in (ckpt_dir1, ckpt_dir2):
            ev.record(
                checkpoint_path=ckpt,
                config_path=tmp_path / "c.yaml",
                train_file=tmp_path / "t.jsonl",
                validation_file=tmp_path / "v.jsonl",
                generation_config=PROBE_GENERATION_CONFIG,
                metrics={},
            )

        evidence_path = tmp_path / "checkpoint-evidence.jsonl"
        lines = evidence_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_crlf_normalization(self, tmp_path: Path) -> None:
        """CRLF→LF normalization produces same SHA for same content."""
        ev = CheckpointEvidence(tmp_path)
        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir()
        # Write with CRLF
        (ckpt_dir / "adapter_model.safetensors").write_bytes(
            b"line1\r\nline2\r\n"
        )

        ev.record(
            checkpoint_path=ckpt_dir,
            config_path=tmp_path / "c.yaml",
            train_file=tmp_path / "t.jsonl",
            validation_file=tmp_path / "v.jsonl",
            generation_config=PROBE_GENERATION_CONFIG,
            metrics={},
        )

        evidence_path = tmp_path / "checkpoint-evidence.jsonl"
        entry = json.loads(evidence_path.read_text(encoding="utf-8").strip())
        # SHA of "line1\nline2\n"
        import hashlib
        expected = hashlib.sha256(b"line1\nline2\n").hexdigest()
        assert entry["weight_sha"] == expected

    def test_missing_file_returns_empty_sha(self, tmp_path: Path) -> None:
        """Missing file → empty SHA string (not crash)."""
        ev = CheckpointEvidence(tmp_path)
        ev.record(
            checkpoint_path=tmp_path / "nonexistent",
            config_path=tmp_path / "also_missing.yaml",
            train_file=tmp_path / "missing_train.jsonl",
            validation_file=tmp_path / "missing_val.jsonl",
            generation_config=PROBE_GENERATION_CONFIG,
            metrics={},
        )
        evidence_path = tmp_path / "checkpoint-evidence.jsonl"
        entry = json.loads(evidence_path.read_text(encoding="utf-8").strip())
        assert entry["weight_sha"] == ""
        assert entry["config_sha"] == ""


# ---------------------------------------------------------------------------
# P3.2/P3.3: CompositeCoverageError on missing bucket
# ---------------------------------------------------------------------------


class TestTier2ProbeBucketMissing:
    """Tier2Probe._compute_composite raises on missing bucket."""

    def test_missing_bucket_raises(self) -> None:
        cfg = _make_config(probe_size=8)
        samples = _make_validation_samples(12)
        probe = Tier2Probe(cfg, samples, output_dir=Path("/tmp"))
        # Remove a bucket
        outcomes = {
            "code": [EvalOutcome(
                task_type="code_generation", syntax_ok=True, public_passed=True,
                public_tests_collected=1, hidden_passed=True,
                hidden_tests_present=True, hidden_tests_collected=1,
                format_ok=True, timed_out=False, is_repair=False,
                repair_succeeded=None, broke_other_tests=None,
            )],
            "boundary": [],
            "static_repair": [],
            "execution_repair": [],
        }
        with pytest.raises(CompositeCoverageError) as excinfo:
            probe._compute_composite(outcomes)
        assert "FIX_FIRST" in str(excinfo.value)


class TestTier3FullValidationBucketMissing:
    """Tier3FullValidation._compute_composite raises on missing bucket."""

    def test_missing_bucket_raises(self) -> None:
        cfg = _make_config()
        samples = _make_validation_samples(12)
        validator = Tier3FullValidation(cfg, samples, output_dir=Path("/tmp"))
        outcomes = {
            "code": [],
            "boundary": [],
            "static_repair": [],
            "execution_repair": [],
        }
        with pytest.raises(CompositeCoverageError):
            validator._compute_composite(outcomes)


# ---------------------------------------------------------------------------
# P3.2/P3.3: Tier 2 / Tier 3 execution with mock model
# ---------------------------------------------------------------------------


class TestTier2ProbeRunWithMock:
    """Tier2Probe.run with a mock model (no real GPU)."""

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason=_TORCH_SKIP_REASON)
    def test_run_produces_probe_result(self, tmp_path: Path) -> None:
        cfg = _make_config(probe_size=8)
        samples = _make_validation_samples(12)
        probe = Tier2Probe(cfg, samples, output_dir=tmp_path)

        # Mock model + tokenizer
        mock_model = MagicMock()
        mock_model.training = False
        mock_model.device = "cpu"
        # generate returns a tensor [1, seq_len] with input_len + new_tokens
        mock_model.generate.return_value = MagicMock(
            __getitem__=lambda self, idx: MagicMock(
                shape=[10]  # arbitrary
            )
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.return_value = MagicMock(
            to=lambda device: {"input_ids": MagicMock(shape=[1, 5])}
        )
        # Make __getitem__ on the returned input_ids work
        mock_tokenizer.return_value["input_ids"] = MagicMock(shape=[1, 5])
        mock_tokenizer.decode.return_value = "```python\ndef solution():\n    return 42\n```"
        mock_tokenizer.eos_token_id = 0

        # Patch run_pytest to always pass
        with patch("src.p3_tier_evaluator.run_pytest") as mock_pytest:
            mock_pytest.return_value = MagicMock(
                passed=True, num_collected=1, num_passed=1,
                num_failed=0, timed_out=False,
            )
            result = probe.run(mock_model, mock_tokenizer, step=50, epoch=0.25)

        assert isinstance(result, ProbeResult)
        assert result.step == 50
        assert result.epoch == 0.25
        assert len(result.probe_sample_ids) > 0

        # Check files were created
        samples_path = tmp_path / "probe_step50_samples.jsonl"
        report_path = tmp_path / "probe_step50_report.json"
        assert samples_path.exists()
        assert report_path.exists()

        # Validate report content
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["step"] == 50
        assert "probe_sha" in report
        assert "composite_value" in report
        assert "generation_config" in report


class TestTier3FullValidationRunWithMock:
    """Tier3FullValidation.run with a mock model."""

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason=_TORCH_SKIP_REASON)
    def test_run_produces_full_result(self, tmp_path: Path) -> None:
        cfg = _make_config()
        # Use 4 samples (1 per bucket) for speed
        samples = _make_validation_samples(1)
        validator = Tier3FullValidation(cfg, samples, output_dir=tmp_path)

        mock_model = MagicMock()
        mock_model.training = False
        mock_model.device = "cpu"
        mock_model.generate.return_value = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.return_value = MagicMock(
            to=lambda device: {"input_ids": MagicMock(shape=[1, 5])}
        )
        mock_tokenizer.return_value["input_ids"] = MagicMock(shape=[1, 5])
        mock_tokenizer.decode.return_value = "```python\ndef solution():\n    return 42\n```"
        mock_tokenizer.eos_token_id = 0

        with patch("src.p3_tier_evaluator.run_pytest") as mock_pytest:
            mock_pytest.return_value = MagicMock(
                passed=True, num_collected=1, num_passed=1,
                num_failed=0, timed_out=False,
            )
            result = validator.run(mock_model, mock_tokenizer, step=100, epoch=1)

        assert isinstance(result, FullValidationResult)
        assert result.epoch == 1
        assert result.hard_constraint_pass is True

        # Check files
        samples_path = tmp_path / "fullval_epoch1_samples.jsonl"
        report_path = tmp_path / "fullval_epoch1_report.json"
        assert samples_path.exists()
        assert report_path.exists()

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "per_bucket" in report
        assert "hard_constraint_pass" in report


# ---------------------------------------------------------------------------
# extract_code_block + build_eval_messages
# ---------------------------------------------------------------------------


class TestExtractCodeBlock:
    def test_extracts_python_block(self) -> None:
        text = "Here:\n```python\ndef foo():\n    pass\n```\nDone."
        code = extract_code_block(text)
        assert code is not None
        assert "def foo" in code

    def test_returns_none_when_no_block(self) -> None:
        text = "No code here."
        assert extract_code_block(text) is None


class TestBuildEvalMessages:
    def test_code_generation_messages(self) -> None:
        s = _make_sample(task_type="code_generation", variant_type="code")
        msgs = build_eval_messages(s)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "完整代码" in msgs[1]["content"]

    def test_execution_repair_includes_feedback(self) -> None:
        s = _make_sample(
            task_type="execution_repair",
            variant_type="execution_repair",
            execution_feedback="AssertionError: expected 42",
        )
        msgs = build_eval_messages(s)
        assert "AssertionError" in msgs[1]["content"]
        assert "修复后的完整代码" in msgs[1]["content"]
