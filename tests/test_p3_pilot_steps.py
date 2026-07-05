"""tests/test_p3_pilot_steps.py — Tests for compliant pilot step calculation.

Issue #14 P0.3: Compliant Pilot step count formula.
"""
from __future__ import annotations

import math

import pytest

from src.p3_pilot_steps import PilotStepCalc, compute_pilot_max_steps


class TestBalancedPilotSteps:
    """Balanced: 622 samples, batch=1, grad_accum=8, world=1."""

    def test_balanced_steps_per_epoch(self):
        # ceil(622 / 8) = ceil(77.75) = 78
        r = compute_pilot_max_steps(622, 1, 8, 1)
        assert r.steps_per_epoch == 78

    def test_balanced_pilot_max_steps(self):
        # min(50, floor(0.25 * 78)) = min(50, floor(19.5)) = min(50, 19) = 19
        r = compute_pilot_max_steps(622, 1, 8, 1)
        assert r.pilot_max_steps == 19

    def test_balanced_effective_batch_size(self):
        r = compute_pilot_max_steps(622, 1, 8, 1)
        assert r.effective_batch_size == 8


class TestRepairPilotSteps:
    """Repair: 490 samples, batch=1, grad_accum=8, world=1."""

    def test_repair_steps_per_epoch(self):
        # ceil(490 / 8) = ceil(61.25) = 62
        r = compute_pilot_max_steps(490, 1, 8, 1)
        assert r.steps_per_epoch == 62

    def test_repair_pilot_max_steps(self):
        # min(50, floor(0.25 * 62)) = min(50, floor(15.5)) = min(50, 15) = 15
        r = compute_pilot_max_steps(490, 1, 8, 1)
        assert r.pilot_max_steps == 15


class TestPositiveMaxStepsOverridesEpochs:
    """When max_steps > 0, it overrides num_train_epochs (HuggingFace Trainer behavior)."""

    def test_pilot_max_steps_is_positive(self):
        r = compute_pilot_max_steps(622, 1, 8, 1)
        assert r.pilot_max_steps > 0

    def test_pilot_max_steps_le_50(self):
        r = compute_pilot_max_steps(622, 1, 8, 1)
        assert r.pilot_max_steps <= 50

    def test_pilot_max_steps_le_quarter_epoch(self):
        """pilot_max_steps must not exceed 0.25 * steps_per_epoch."""
        r = compute_pilot_max_steps(622, 1, 8, 1)
        assert r.pilot_max_steps <= int(math.floor(0.25 * r.steps_per_epoch))


class TestSmallDataset:
    """Small dataset must produce at least 1 step."""

    def test_single_sample(self):
        r = compute_pilot_max_steps(1, 1, 8, 1)
        # steps_per_epoch = ceil(1/8) = 1
        # pilot_max = min(50, floor(0.25*1)) = min(50, 0) = 0 -> bumped to 1
        assert r.steps_per_epoch == 1
        assert r.pilot_max_steps == 1

    def test_very_small_dataset(self):
        r = compute_pilot_max_steps(3, 1, 8, 1)
        # steps_per_epoch = ceil(3/8) = 1
        # pilot_max = min(50, floor(0.25*1)) = 0 -> bumped to 1
        assert r.steps_per_epoch == 1
        assert r.pilot_max_steps == 1

    def test_eight_samples(self):
        r = compute_pilot_max_steps(8, 1, 8, 1)
        # steps_per_epoch = ceil(8/8) = 1
        # pilot_max = min(50, floor(0.25*1)) = 0 -> bumped to 1
        assert r.steps_per_epoch == 1
        assert r.pilot_max_steps == 1

    def test_nine_samples(self):
        r = compute_pilot_max_steps(9, 1, 8, 1)
        # steps_per_epoch = ceil(9/8) = 2
        # pilot_max = min(50, floor(0.25*2)) = min(50, 0) = 0 -> bumped to 1
        assert r.steps_per_epoch == 2
        assert r.pilot_max_steps == 1


class TestEdgeCases:
    """Edge cases and validation."""

    def test_zero_samples_raises(self):
        with pytest.raises(ValueError, match="num_train_samples"):
            compute_pilot_max_steps(0)

    def test_negative_samples_raises(self):
        with pytest.raises(ValueError, match="num_train_samples"):
            compute_pilot_max_steps(-1)

    def test_zero_batch_raises(self):
        with pytest.raises(ValueError, match="per_device_train_batch_size"):
            compute_pilot_max_steps(622, 0)

    def test_zero_grad_accum_raises(self):
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            compute_pilot_max_steps(622, 1, 0)

    def test_zero_world_size_raises(self):
        with pytest.raises(ValueError, match="world_size"):
            compute_pilot_max_steps(622, 1, 8, 0)

    def test_invalid_epoch_fraction_zero(self):
        with pytest.raises(ValueError, match="epoch_fraction"):
            compute_pilot_max_steps(622, epoch_fraction=0)

    def test_invalid_epoch_fraction_above_one(self):
        with pytest.raises(ValueError, match="epoch_fraction"):
            compute_pilot_max_steps(622, epoch_fraction=1.5)

    def test_world_size_2_doubles_batch(self):
        r = compute_pilot_max_steps(622, 1, 8, 2)
        # effective_batch = 1*8*2 = 16
        # steps_per_epoch = ceil(622/16) = 39
        # pilot_max = min(50, floor(0.25*39)) = min(50, 9) = 9
        assert r.effective_batch_size == 16
        assert r.steps_per_epoch == 39
        assert r.pilot_max_steps == 9


class TestAsDict:
    def test_as_dict_contains_all_fields(self):
        r = compute_pilot_max_steps(622, 1, 8, 1)
        d = r.as_dict()
        assert set(d.keys()) == {
            "steps_per_epoch",
            "pilot_max_steps",
            "effective_batch_size",
            "num_train_samples",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "world_size",
        }

    def test_pilot_step_calc_is_frozen(self):
        r = compute_pilot_max_steps(622, 1, 8, 1)
        with pytest.raises(AttributeError):
            r.pilot_max_steps = 100  # type: ignore[misc]
