"""Tests for src/training_callbacks.py (Issue #12 P4).

Verifies CheckpointEvaluatorCallback:
- Tier 1 logging on on_log
- Tier 2/3 scheduling on on_epoch_end
- NaN/Inf detection triggers early stop
- Evaluator state saved on on_train_end
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training_callbacks import CheckpointEvaluatorCallback  # noqa: E402


def _make_evaluator_mock():
    """Mock CheckpointEvaluator with should_run_tier1/2/3 and check_early_stop."""
    ev = MagicMock()
    ev.should_run_tier1.return_value = True
    ev.should_run_tier2.return_value = False
    ev.should_run_tier3.return_value = False
    ev.check_early_stop.return_value = (False, "")
    return ev


def _make_trainer_state(global_step=50, epoch=0.5):
    """Mock TrainerState with global_step and epoch."""
    state = MagicMock()
    state.global_step = global_step
    state.epoch = epoch
    return state


def _make_control():
    """Mock TrainerControl."""
    control = MagicMock()
    control.should_training_stop = False
    return control


class TestTier1Logging:
    def test_on_log_records_tier1_entry(self, tmp_path):
        """on_log should append to tier1_log."""
        ev = _make_evaluator_mock()
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=True)
        state = _make_trainer_state(global_step=50, epoch=0.5)
        control = _make_control()
        logs = {"loss": 1.5, "learning_rate": 0.0001}

        cb.on_log(MagicMock(), state, control, logs=logs)

        assert len(cb.tier1_log) == 1
        entry = cb.tier1_log[0]
        assert entry["step"] == 50
        assert entry["train_loss"] == 1.5
        assert entry["lr"] == 0.0001
        assert entry["nan_inf_detected"] is False

    def test_on_log_detects_nan_loss(self, tmp_path):
        """NaN in loss should set nan_inf_detected=True."""
        ev = _make_evaluator_mock()
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=True)
        state = _make_trainer_state(global_step=50, epoch=0.5)
        control = _make_control()
        logs = {"loss": float("nan")}

        cb.on_log(MagicMock(), state, control, logs=logs)

        assert cb.nan_inf_detected is True

    def test_on_log_detects_inf_eval_loss(self, tmp_path):
        """Inf in eval_loss should set nan_inf_detected=True."""
        ev = _make_evaluator_mock()
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=True)
        state = _make_trainer_state(global_step=50, epoch=0.5)
        control = _make_control()
        logs = {"eval_loss": float("inf")}

        cb.on_log(MagicMock(), state, control, logs=logs)

        assert cb.nan_inf_detected is True

    def test_on_log_skips_none_logs(self, tmp_path):
        """on_log with logs=None should be a no-op."""
        ev = _make_evaluator_mock()
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=True)
        state = _make_trainer_state()

        cb.on_log(MagicMock(), state, _make_control(), logs=None)

        assert len(cb.tier1_log) == 0


class TestTier2Tier3Scheduling:
    def test_on_epoch_end_tier2_scheduled_pilot(self, tmp_path):
        """In pilot mode, Tier 2 should be SCHEDULED_PILOT_DEFERRED."""
        ev = _make_evaluator_mock()
        ev.should_run_tier2.return_value = True
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=True)
        state = _make_trainer_state(global_step=100, epoch=0.25)
        control = _make_control()

        cb.on_epoch_end(MagicMock(), state, control)

        assert len(cb.tier2_log) == 1
        assert cb.tier2_log[0]["status"] == "SCHEDULED_PILOT_DEFERRED"
        assert cb.tier2_log[0]["epoch"] == 0.25

    def test_on_epoch_end_tier2_running_non_pilot(self, tmp_path):
        """In non-pilot mode, Tier 2 should be RUNNING."""
        ev = _make_evaluator_mock()
        ev.should_run_tier2.return_value = True
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=False)
        state = _make_trainer_state(global_step=100, epoch=0.25)
        control = _make_control()

        cb.on_epoch_end(MagicMock(), state, control)

        assert len(cb.tier2_log) == 1
        assert cb.tier2_log[0]["status"] == "RUNNING"

    def test_on_epoch_end_tier3_scheduled_pilot(self, tmp_path):
        """In pilot mode, Tier 3 should be SCHEDULED_PILOT_DEFERRED."""
        ev = _make_evaluator_mock()
        ev.should_run_tier3.return_value = True
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=True)
        state = _make_trainer_state(global_step=200, epoch=1.0)
        control = _make_control()

        cb.on_epoch_end(MagicMock(), state, control)

        assert len(cb.tier3_log) == 1
        assert cb.tier3_log[0]["status"] == "SCHEDULED_PILOT_DEFERRED"
        assert cb.tier3_log[0]["epoch"] == 1

    def test_on_epoch_end_tier2_and_tier3_both_scheduled(self, tmp_path):
        """Both Tier 2 and Tier 3 can be scheduled at epoch=1.0."""
        ev = _make_evaluator_mock()
        ev.should_run_tier2.return_value = True
        ev.should_run_tier3.return_value = True
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=True)
        state = _make_trainer_state(global_step=200, epoch=1.0)
        control = _make_control()

        cb.on_epoch_end(MagicMock(), state, control)

        assert len(cb.tier2_log) == 1
        assert len(cb.tier3_log) == 1


class TestEarlyStop:
    def test_nan_inf_triggers_early_stop(self, tmp_path):
        """NaN/Inf detected should trigger early stop via check_early_stop."""
        ev = _make_evaluator_mock()
        ev.check_early_stop.return_value = (True, "nan_or_inf_detected: immediate stop")
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=True)
        cb.nan_inf_detected = True
        state = _make_trainer_state(global_step=50, epoch=0.5)
        control = _make_control()

        cb.on_epoch_end(MagicMock(), state, control)

        assert cb.early_stop_triggered is True
        assert "nan_or_inf" in cb.early_stop_reason
        assert control.should_training_stop is True

    def test_no_early_stop_when_healthy(self, tmp_path):
        """No early stop when check_early_stop returns False."""
        ev = _make_evaluator_mock()
        ev.check_early_stop.return_value = (False, "")
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=True)
        state = _make_trainer_state(global_step=50, epoch=0.5)
        control = _make_control()

        cb.on_epoch_end(MagicMock(), state, control)

        assert cb.early_stop_triggered is False
        assert control.should_training_stop is False


class TestEvaluatorStateSave:
    def test_on_train_end_saves_evaluator_state_json(self, tmp_path):
        """on_train_end should write evaluator_state.json."""
        ev = _make_evaluator_mock()
        cb = CheckpointEvaluatorCallback(ev, {}, tmp_path, pilot_mode=True)
        cb.tier1_log.append({"step": 10, "train_loss": 1.0})
        cb.tier2_log.append({"epoch": 0.25, "status": "SCHEDULED_PILOT_DEFERRED"})

        cb.on_train_end(MagicMock(), _make_trainer_state(), _make_control())

        state_path = tmp_path / "evaluator_state.json"
        assert state_path.exists()
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        assert state["tier1_entries"] == 1
        assert state["tier2_entries"] == 1
        assert state["pilot_mode"] is True
        assert state["nan_inf_detected"] is False
        assert state["early_stop_triggered"] is False

    def test_on_train_end_creates_output_dir(self, tmp_path):
        """on_train_end should create output_dir if it doesn't exist."""
        ev = _make_evaluator_mock()
        new_dir = tmp_path / "subdir"
        cb = CheckpointEvaluatorCallback(ev, {}, new_dir, pilot_mode=True)

        cb.on_train_end(MagicMock(), _make_trainer_state(), _make_control())

        assert new_dir.exists()
        assert (new_dir / "evaluator_state.json").exists()
