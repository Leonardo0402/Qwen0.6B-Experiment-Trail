"""Tests for src/p3_checkpoint_evaluator.py (P3 Task 13).

Coverage
--------
1. Config schema for both YAMLs (required fields present).
2. Composite-score weights sum to 1.0 and match candidate ratios.
3. ``check_bf16_support`` returns ``(bool, str)``.
4. ``CompositeScore.compute`` returns weighted sum in [0, 1].
5. Tier scheduling booleans for tier1/tier2/tier3.
6. Probe sample selection returns 75+/-1 stratified samples.
7. Best checkpoint only uses full validation (never probe).
8. Early stop triggers: NaN/Inf immediate, probe-patience confirmed.
9. Hard-constraint check detects code_generation drop > 3pp.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.p3_checkpoint_evaluator import (  # noqa: E402
    CheckpointEvaluator,
    CompositeScore,
    FullValidationResult,
    ProbeResult,
    check_bf16_support,
)
from src.metrics import (  # noqa: E402
    METRICS_SCHEMA_VERSION,
    normalize_baseline_key,
)

BALANCED_CONFIG_PATH = _ROOT / "configs" / "p3" / "balanced-generalist.yaml"
REPAIR_CONFIG_PATH = _ROOT / "configs" / "p3" / "repair-specialist.yaml"


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_evaluator_config() -> dict:
    """Minimal config dict for CheckpointEvaluator (balanced-generalist)."""
    return {
        "checkpoint_evaluator": {
            "tier1": {
                "interval_steps": 50,
                "metrics": ["train_loss", "eval_loss", "lr", "gpu_mem_mb",
                            "nan_inf_detected"],
            },
            "tier2": {
                "interval_epoch_fraction": 0.25,
                "probe_size": 75,
                "probe_stratify_by": "variant_type",
                "probe_seed": 42,
                "composite_score": True,
            },
            "tier3": {
                "interval_epochs": 1,
                "full_validation": True,
                "composite_score": True,
            },
        },
        "early_stopping": {
            "enabled": True,
            "probe_patience": 4,
            "probe_min_delta": 0.005,
            "full_validation_confirm": True,
            "divergence_nan_inf": True,
            "max_epochs": 3,
        },
        "composite_score": {
            "code_generation_pass_at_1": 0.30,
            "boundary_pass_at_1": 0.20,
            "static_repair_success": 0.20,
            "execution_repair_success": 0.30,
            "hard_constraint": {
                "code_generation_drop_vs_p2_final_max_pct": 3.0,
            },
        },
        "best_checkpoint": {
            "selection_metric": "full_validation_composite",
            "never_use": ["frozen_v3", "probe"],
        },
    }


def _make_mock_samples(n_per_bucket: int = 25) -> list:
    """Build mock training samples stratified by variant_type."""
    samples = []
    for variant in ["code", "boundary", "static_repair", "execution_repair"]:
        for i in range(n_per_bucket):
            samples.append(
                SimpleNamespace(
                    sample_id=f"{variant}_{i:03d}",
                    variant_type=variant,
                )
            )
    return samples


def _make_full_result(composite_value: float, epoch: int = 1) -> FullValidationResult:
    return FullValidationResult(
        step=epoch * 100,
        epoch=epoch,
        composite_score=CompositeScore(0.0, 0.0, 0.0, 0.0),
        composite_value=composite_value,
        metrics={},
        hard_constraint_pass=True,
        hard_constraint_violations=[],
    )


def _make_probe_result(composite_value: float, epoch: float) -> ProbeResult:
    return ProbeResult(
        step=int(epoch * 100),
        epoch=epoch,
        probe_sample_ids=[],
        composite_score=CompositeScore(0.0, 0.0, 0.0, 0.0),
        composite_value=composite_value,
        metrics={},
    )


# ---------------------------------------------------------------------------
# 1. Config schema tests
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_config_schema_balanced(self) -> None:
        cfg = _load_yaml(BALANCED_CONFIG_PATH)
        # Top-level required fields
        for key in [
            "training_mode", "model_name_or_path", "train_file", "eval_file",
            "output_dir", "dataset_manifest", "max_seq_length", "num_train_epochs",
            "learning_rate", "per_device_train_batch_size",
            "gradient_accumulation_steps",
        ]:
            assert key in cfg, f"missing top-level field: {key}"
        # lora block
        assert cfg["lora"]["rank"] == 16
        assert cfg["lora"]["alpha"] == 32
        assert cfg["lora"]["dropout"] == 0.05
        assert len(cfg["lora"]["target_modules"]) == 7
        # bf16 block
        assert cfg["bf16"]["enabled"] is True
        assert cfg["bf16"]["runtime_check"] is True
        assert cfg["fp16"]["enabled"] is False
        # checkpoint_evaluator with tier1/tier2/tier3
        ce = cfg["checkpoint_evaluator"]
        for tier in ("tier1", "tier2", "tier3"):
            assert tier in ce, f"missing {tier}"
        # composite_score with 4 weights
        cs = cfg["composite_score"]
        for w in [
            "code_generation_pass_at_1", "boundary_pass_at_1",
            "static_repair_success", "execution_repair_success",
        ]:
            assert w in cs, f"missing composite weight: {w}"
        assert "hard_constraint" in cs
        # early_stopping
        es = cfg["early_stopping"]
        assert es["enabled"] is True
        assert es["full_validation_confirm"] is True
        # best_checkpoint
        bc = cfg["best_checkpoint"]
        assert bc["selection_metric"] == "full_validation_composite"
        assert bc["never_use"] == ["frozen_v3", "probe"]

    def test_config_schema_repair(self) -> None:
        cfg = _load_yaml(REPAIR_CONFIG_PATH)
        for key in [
            "training_mode", "model_name_or_path", "train_file", "eval_file",
            "output_dir", "dataset_manifest", "max_seq_length", "num_train_epochs",
            "learning_rate", "per_device_train_batch_size",
            "gradient_accumulation_steps",
        ]:
            assert key in cfg, f"missing top-level field: {key}"
        assert cfg["lora"]["rank"] == 16
        assert cfg["lora"]["alpha"] == 32
        assert cfg["lora"]["dropout"] == 0.05
        assert len(cfg["lora"]["target_modules"]) == 7
        assert cfg["bf16"]["enabled"] is True
        assert cfg["bf16"]["runtime_check"] is True
        assert cfg["fp16"]["enabled"] is False
        ce = cfg["checkpoint_evaluator"]
        for tier in ("tier1", "tier2", "tier3"):
            assert tier in ce
        cs = cfg["composite_score"]
        for w in [
            "code_generation_pass_at_1", "boundary_pass_at_1",
            "static_repair_success", "execution_repair_success",
        ]:
            assert w in cs
        assert "hard_constraint" in cs
        es = cfg["early_stopping"]
        assert es["enabled"] is True
        assert es["full_validation_confirm"] is True
        bc = cfg["best_checkpoint"]
        assert bc["selection_metric"] == "full_validation_composite"
        assert bc["never_use"] == ["frozen_v3", "probe"]
        # Repair-specific paths
        assert "repair-specialist" in cfg["train_file"]
        assert "repair-specialist" in cfg["eval_file"]
        assert "repair-specialist" in cfg["output_dir"]
        assert "repair-specialist" in cfg["dataset_manifest"]


# ---------------------------------------------------------------------------
# 2. Composite-score weight tests
# ---------------------------------------------------------------------------


class TestCompositeWeights:
    def test_config_ratio_weights_balanced(self) -> None:
        cfg = _load_yaml(BALANCED_CONFIG_PATH)
        cs = cfg["composite_score"]
        weights = [
            cs["code_generation_pass_at_1"],
            cs["boundary_pass_at_1"],
            cs["static_repair_success"],
            cs["execution_repair_success"],
        ]
        # Sum to 1.0 within tolerance
        assert abs(sum(weights) - 1.0) < 0.01
        # Match 30/20/20/30 within +-0.01
        assert abs(cs["code_generation_pass_at_1"] - 0.30) < 0.01
        assert abs(cs["boundary_pass_at_1"] - 0.20) < 0.01
        assert abs(cs["static_repair_success"] - 0.20) < 0.01
        assert abs(cs["execution_repair_success"] - 0.30) < 0.01

    def test_config_ratio_weights_repair(self) -> None:
        cfg = _load_yaml(REPAIR_CONFIG_PATH)
        cs = cfg["composite_score"]
        weights = [
            cs["code_generation_pass_at_1"],
            cs["boundary_pass_at_1"],
            cs["static_repair_success"],
            cs["execution_repair_success"],
        ]
        assert abs(sum(weights) - 1.0) < 0.01
        # Match 15/15/30/40 within +-0.01
        assert abs(cs["code_generation_pass_at_1"] - 0.15) < 0.01
        assert abs(cs["boundary_pass_at_1"] - 0.15) < 0.01
        assert abs(cs["static_repair_success"] - 0.30) < 0.01
        assert abs(cs["execution_repair_success"] - 0.40) < 0.01


# ---------------------------------------------------------------------------
# 3. BF16 check
# ---------------------------------------------------------------------------


class TestBf16Check:
    def test_bf16_check_returns_bool_and_message(self) -> None:
        result = check_bf16_support()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
        assert len(result[1]) > 0


# ---------------------------------------------------------------------------
# 4. CompositeScore.compute
# ---------------------------------------------------------------------------


class TestCompositeScoreCompute:
    def test_composite_score_compute(self) -> None:
        cs = CompositeScore(
            code_generation_pass_at_1=0.5,
            boundary_pass_at_1=0.4,
            static_repair_success=0.6,
            execution_repair_success=0.7,
        )
        weights = {
            "code_generation_pass_at_1": 0.30,
            "boundary_pass_at_1": 0.20,
            "static_repair_success": 0.20,
            "execution_repair_success": 0.30,
        }
        result = cs.compute(weights)
        expected = 0.5 * 0.3 + 0.4 * 0.2 + 0.6 * 0.2 + 0.7 * 0.3  # = 0.56
        assert abs(result - expected) < 1e-9
        assert 0.0 <= result <= 1.0

    def test_composite_score_compute_zero(self) -> None:
        cs = CompositeScore(0.0, 0.0, 0.0, 0.0)
        weights = {
            "code_generation_pass_at_1": 0.30,
            "boundary_pass_at_1": 0.20,
            "static_repair_success": 0.20,
            "execution_repair_success": 0.30,
        }
        assert cs.compute(weights) == 0.0

    def test_composite_score_compute_max(self) -> None:
        cs = CompositeScore(1.0, 1.0, 1.0, 1.0)
        weights = {
            "code_generation_pass_at_1": 0.30,
            "boundary_pass_at_1": 0.20,
            "static_repair_success": 0.20,
            "execution_repair_success": 0.30,
        }
        assert abs(cs.compute(weights) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 5. Tier scheduling
# ---------------------------------------------------------------------------


class TestTierScheduling:
    def test_tier_scheduling(self) -> None:
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)

        # Tier 1: every 50 steps
        for step in (50, 100, 150, 200, 250):
            assert evaluator.should_run_tier1(step) is True, f"tier1 at step {step}"
        for step in (0, 25, 75, 125, 175):
            if step == 0:
                assert evaluator.should_run_tier1(step) is False
            else:
                assert evaluator.should_run_tier1(step) is False, f"no tier1 at step {step}"

        # Tier 2: every 0.25 epoch boundary
        for epoch in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
            assert evaluator.should_run_tier2(epoch) is True, f"tier2 at epoch {epoch}"
        for epoch in (0.0, 0.1, 0.3, 0.6, 0.9, 1.1, 1.7):
            assert evaluator.should_run_tier2(epoch) is False, f"no tier2 at epoch {epoch}"

        # Tier 3: integer epoch >= 1
        for epoch in (1, 2, 3, 4, 10):
            assert evaluator.should_run_tier3(epoch) is True, f"tier3 at epoch {epoch}"
        for epoch in (0, -1, -5):
            assert evaluator.should_run_tier3(epoch) is False, f"no tier3 at epoch {epoch}"
        # Floats (even whole-number floats) are not integers
        assert evaluator.should_run_tier3(0.5) is False
        assert evaluator.should_run_tier3(1.0) is False


# ---------------------------------------------------------------------------
# 6. Probe sample selection
# ---------------------------------------------------------------------------


class TestProbeSampleSelection:
    def test_probe_sample_selection(self) -> None:
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)

        samples = _make_mock_samples(n_per_bucket=25)
        selected = evaluator.select_probe_samples(samples, seed=42)

        # Total 75 +/- 1
        assert 74 <= len(selected) <= 76, f"expected ~75 samples, got {len(selected)}"

        # Stratified by variant_type (4 buckets)
        variant_counts: dict[str, int] = {}
        for s in selected:
            v = s.variant_type
            variant_counts[v] = variant_counts.get(v, 0) + 1
        assert len(variant_counts) == 4
        for v, count in variant_counts.items():
            # Each bucket should have ~18-19 samples
            assert 17 <= count <= 20, f"variant {v}: {count} samples"

    def test_probe_sample_selection_deterministic(self) -> None:
        """Same seed produces same selection."""
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)
        samples = _make_mock_samples(n_per_bucket=25)
        selected1 = evaluator.select_probe_samples(samples, seed=42)
        selected2 = evaluator.select_probe_samples(samples, seed=42)
        ids1 = [s.sample_id for s in selected1]
        ids2 = [s.sample_id for s in selected2]
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# 7. Best checkpoint
# ---------------------------------------------------------------------------


class TestBestCheckpoint:
    def test_best_checkpoint_only_uses_full_validation(self) -> None:
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)

        # First full validation: composite_value=0.5 -> new best
        result1 = _make_full_result(composite_value=0.5, epoch=1)
        assert evaluator.update_best_checkpoint(result1) is True
        assert evaluator._best_composite_value == 0.5

        # Lower composite_value -> not new best
        result2 = _make_full_result(composite_value=0.3, epoch=2)
        assert evaluator.update_best_checkpoint(result2) is False
        assert evaluator._best_composite_value == 0.5

        # Higher composite_value -> new best
        result3 = _make_full_result(composite_value=0.7, epoch=3)
        assert evaluator.update_best_checkpoint(result3) is True
        assert evaluator._best_composite_value == 0.7

        # Equal composite_value -> not new best (strictly greater required)
        result4 = _make_full_result(composite_value=0.7, epoch=4)
        assert evaluator.update_best_checkpoint(result4) is False
        assert evaluator._best_composite_value == 0.7


# ---------------------------------------------------------------------------
# 8. Early stop triggers
# ---------------------------------------------------------------------------


class TestEarlyStop:
    def test_early_stop_nan_inf(self) -> None:
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)
        should_stop, reason = evaluator.check_early_stop(
            probe_history=[], full_history=[], nan_inf_detected=True
        )
        assert should_stop is True
        assert "nan" in reason.lower() or "inf" in reason.lower()

    def test_early_stop_probe_patience(self) -> None:
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)

        # 4 consecutive probe drops (each lower than previous by > 0.005)
        probe_history = [
            _make_probe_result(composite_value=0.50, epoch=0.25),
            _make_probe_result(composite_value=0.49, epoch=0.50),  # drop 1
            _make_probe_result(composite_value=0.48, epoch=0.75),  # drop 2
            _make_probe_result(composite_value=0.47, epoch=1.00),  # drop 3
        ]

        # Case 1: with full_validation_confirm=True but no full_history
        # -> should NOT stop (awaiting confirmation)
        should_stop, reason = evaluator.check_early_stop(
            probe_history=probe_history, full_history=[],
            nan_inf_detected=False
        )
        assert should_stop is False
        assert "awaiting" in reason.lower() or "confirm" in reason.lower()

        # Case 2: with full_validation_confirm=True AND a confirming full_history
        # (full_history[-1] < full_history[-2] - probe_min_delta) -> should stop
        full_history = [
            _make_full_result(composite_value=0.50, epoch=1),
            _make_full_result(composite_value=0.45, epoch=2),  # drop 0.05 > 0.005
        ]
        should_stop, reason = evaluator.check_early_stop(
            probe_history=probe_history, full_history=full_history,
            nan_inf_detected=False
        )
        assert should_stop is True
        assert "probe_patience" in reason or "confirmed" in reason

    def test_early_stop_no_trigger(self) -> None:
        """No drops and no max_epochs -> no stop."""
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)

        # Probes increasing -> no drop
        probe_history = [
            _make_probe_result(composite_value=0.40, epoch=0.25),
            _make_probe_result(composite_value=0.50, epoch=0.50),
            _make_probe_result(composite_value=0.55, epoch=0.75),
            _make_probe_result(composite_value=0.60, epoch=1.00),
        ]
        full_history = [_make_full_result(composite_value=0.55, epoch=1)]
        should_stop, reason = evaluator.check_early_stop(
            probe_history=probe_history, full_history=full_history,
            nan_inf_detected=False
        )
        assert should_stop is False

    def test_early_stop_max_epochs(self) -> None:
        """max_epochs reached -> stop."""
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)

        full_history = [_make_full_result(composite_value=0.55, epoch=3)]
        should_stop, reason = evaluator.check_early_stop(
            probe_history=[], full_history=full_history,
            nan_inf_detected=False
        )
        assert should_stop is True
        assert "max_epochs" in reason.lower()


# ---------------------------------------------------------------------------
# 9. Hard constraint check
# ---------------------------------------------------------------------------


class TestHardConstraint:
    def test_hard_constraint_check(self) -> None:
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)

        # P2 final codegen_pass1 = 0.1724 (Stage3-v3-Antiforget baseline)
        baseline = {"codegen_pass1": 0.1724}

        # Case 1: drop > 3pp -> violation
        # current pass_at_1 = 0.10, drop = (0.1724 - 0.10) * 100 = 7.24pp > 3.0
        metrics = {"pass_at_1": 0.10}
        passed, violations = evaluator.check_hard_constraint(metrics, baseline)
        assert passed is False
        assert len(violations) == 1
        assert "dropped" in violations[0].lower() or "drop" in violations[0].lower()

        # Case 2: drop <= 3pp -> pass
        # current pass_at_1 = 0.16, drop = (0.1724 - 0.16) * 100 = 1.24pp <= 3.0
        metrics = {"pass_at_1": 0.16}
        passed, violations = evaluator.check_hard_constraint(metrics, baseline)
        assert passed is True
        assert len(violations) == 0

        # Case 3: improvement (current > baseline) -> pass
        metrics = {"pass_at_1": 0.20}
        passed, violations = evaluator.check_hard_constraint(metrics, baseline)
        assert passed is True
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# 10. Fix 3: Tightened check_early_stop trigger 2 (Issue #10)
# ---------------------------------------------------------------------------


class TestCheckEarlyStopFix3:
    """Tests for tightened trigger 2: full validation must also drop."""

    def _make_dropping_probes(self) -> list[ProbeResult]:
        """4 consecutive probe drops (each < prev - 0.005)."""
        return [
            _make_probe_result(composite_value=0.50, epoch=0.25),
            _make_probe_result(composite_value=0.49, epoch=0.50),  # drop 0.01
            _make_probe_result(composite_value=0.48, epoch=0.75),  # drop 0.01
            _make_probe_result(composite_value=0.47, epoch=1.00),  # drop 0.01
        ]

    def test_check_early_stop_probe_drops_no_full_history_returns_false(self) -> None:
        """probe drops + full_history=[] -> False (awaiting 2 full validations)."""
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)
        should_stop, reason = evaluator.check_early_stop(
            probe_history=self._make_dropping_probes(),
            full_history=[],
            nan_inf_detected=False,
        )
        assert should_stop is False
        assert "awaiting" in reason.lower() or "2 full" in reason.lower()

    def test_check_early_stop_probe_drops_one_full_returns_false(self) -> None:
        """probe drops + full_history has only 1 item -> False."""
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)
        full_history = [_make_full_result(composite_value=0.50, epoch=1)]
        should_stop, reason = evaluator.check_early_stop(
            probe_history=self._make_dropping_probes(),
            full_history=full_history,
            nan_inf_detected=False,
        )
        assert should_stop is False
        assert "awaiting" in reason.lower() or "2 full" in reason.lower()

    def test_check_early_stop_probe_drops_full_also_drops_returns_true(self) -> None:
        """probe drops + full_history[-1] < [-2] - probe_min_delta -> True."""
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)
        full_history = [
            _make_full_result(composite_value=0.50, epoch=1),
            _make_full_result(composite_value=0.45, epoch=2),  # drop 0.05 > 0.005
        ]
        should_stop, reason = evaluator.check_early_stop(
            probe_history=self._make_dropping_probes(),
            full_history=full_history,
            nan_inf_detected=False,
        )
        assert should_stop is True
        assert "confirmed" in reason.lower() or "probe" in reason.lower()

    def test_check_early_stop_probe_drops_full_does_not_drop_returns_false(self) -> None:
        """probe drops + full_history[-1] >= [-2] - probe_min_delta -> False."""
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)
        full_history = [
            _make_full_result(composite_value=0.45, epoch=1),
            _make_full_result(composite_value=0.50, epoch=2),  # increase, no drop
        ]
        should_stop, reason = evaluator.check_early_stop(
            probe_history=self._make_dropping_probes(),
            full_history=full_history,
            nan_inf_detected=False,
        )
        assert should_stop is False
        assert "did not confirm" in reason.lower() or "not confirm" in reason.lower()

    def test_check_early_stop_nan_inf_immediate_stop(self) -> None:
        """NaN/Inf -> immediate stop (unchanged behavior)."""
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)
        should_stop, reason = evaluator.check_early_stop(
            probe_history=[], full_history=[], nan_inf_detected=True
        )
        assert should_stop is True
        assert "nan" in reason.lower() or "inf" in reason.lower()


# ---------------------------------------------------------------------------
# 11. Fix 4: Metrics Schema unification (Issue #10)
# ---------------------------------------------------------------------------


class TestMetricsSchemaFix4:
    def test_normalize_baseline_key_renames_codegen_pass1(self) -> None:
        baseline = {"codegen_pass1": 0.5, "other": "x"}
        normalized = normalize_baseline_key(baseline)
        # codegen_pass1 renamed to pass_at_1
        assert "codegen_pass1" not in normalized
        assert normalized["pass_at_1"] == 0.5
        # Other fields preserved
        assert normalized["other"] == "x"
        # Original dict not mutated
        assert baseline == {"codegen_pass1": 0.5, "other": "x"}

    def test_metrics_schema_version_constant(self) -> None:
        assert METRICS_SCHEMA_VERSION == "1.0.0"

    def test_check_hard_constraint_uses_normalized_key(self) -> None:
        """baseline uses codegen_pass1, metrics uses pass_at_1 - unified comparison."""
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)
        # baseline codegen_pass1=0.5, current pass_at_1=0.4
        # drop = (0.5 - 0.4) * 100 = 10pp > 3.0pp max -> FAIL
        baseline = {"codegen_pass1": 0.5}
        metrics = {"pass_at_1": 0.4}
        passed, violations = evaluator.check_hard_constraint(metrics, baseline)
        assert passed is False
        assert len(violations) >= 1
        assert any(
            "dropped" in v.lower() or "drop" in v.lower() for v in violations
        )

    def test_check_hard_constraint_schema_version_mismatch_warning(self) -> None:
        """schema_version mismatch recorded as warning (not FAIL by itself)."""
        cfg = _make_evaluator_config()
        evaluator = CheckpointEvaluator(cfg, total_train_samples=626)
        # baseline has schema_version="0.0.9" (mismatch), no drop_pct violation
        baseline = {"codegen_pass1": 0.5, "schema_version": "0.0.9"}
        metrics = {"pass_at_1": 0.5}
        passed, violations = evaluator.check_hard_constraint(metrics, baseline)
        # passed is True because only schema_version mismatch (warning only)
        assert passed is True
        # But violations list contains schema_version warning
        assert any("schema_version" in v.lower() for v in violations)
