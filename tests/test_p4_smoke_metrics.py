"""Tests for §2.3 smoke success metrics enrichment.

Verifies that _aggregate_metrics computes the 8 required fields from
trajectory data, and that the model trajectory collection report
contains all 11 required fields per config.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.collect_model_trajectories import _aggregate_metrics


# ---------------------------------------------------------------------------
# §2.3 required metric fields per configuration
# ---------------------------------------------------------------------------
REQUIRED_METRIC_FIELDS = [
    "model_load_ok",
    "adapter_load_ok",
    "generation_ok",
    "json_parse_rate",
    "schema_valid_rate",
    "safety_valid_rate",
    "action_type_valid_rate",
    "arguments_valid_rate",
    "forbidden_action_count",
    "tool_dispatch_ok",
    "max_step_stop_ok",
    "runtime_crash_count",
]


def _make_diag(json_parse_ok=True, schema_valid=True, safety_valid=True,
               action_type_valid=True, arguments_valid=True):
    """Create a minimal step_diagnostics dict matching ModelStepDiagnostics."""
    return {
        "raw_output": "",
        "json_parse_ok": json_parse_ok,
        "schema_valid": schema_valid,
        "safety_valid": safety_valid,
        "action_type_valid": action_type_valid,
        "arguments_valid": arguments_valid,
        "repair_attempted": False,
        "repair_success": False,
        "latency_ms": 0,
    }


def _make_traj(success=True, step_diags=None, forbidden=0, tool_error_rate=0.0,
               max_step_exceeded=0):
    """Create a minimal trajectory dict for testing."""
    if step_diags is None:
        step_diags = [_make_diag()]
    return {
        "trajectory_id": "test_001",
        "task_id": "task_001",
        "success": success,
        "metrics": {
            "forbidden_action_count": forbidden,
            "tool_error_rate": tool_error_rate,
            "max_step_exceeded_count": max_step_exceeded,
        },
        "step_diagnostics": step_diags,
    }


# ---------------------------------------------------------------------------
# Test: _aggregate_metrics returns all required fields
# ---------------------------------------------------------------------------
def test_aggregate_metrics_returns_all_required_fields():
    """§2.3: all 11 required fields must be present in the aggregated metrics."""
    trajectories = [_make_traj()]
    metrics = _aggregate_metrics(trajectories, crashes=0)
    for field in REQUIRED_METRIC_FIELDS:
        assert field in metrics, f"missing required field: {field}"


# ---------------------------------------------------------------------------
# Test: json_parse_rate / schema_valid_rate / etc. computed from diagnostics
# ---------------------------------------------------------------------------
def test_rates_computed_from_diagnostics():
    """Rates should be computed from step_diagnostics across all trajectories."""
    traj1 = _make_traj(step_diags=[
        _make_diag(json_parse_ok=True, schema_valid=True, safety_valid=True,
                   action_type_valid=True, arguments_valid=True),
        _make_diag(json_parse_ok=True, schema_valid=False, safety_valid=True,
                   action_type_valid=True, arguments_valid=False),
    ])
    traj2 = _make_traj(step_diags=[
        _make_diag(json_parse_ok=False, schema_valid=False, safety_valid=False,
                   action_type_valid=False, arguments_valid=False),
    ])
    metrics = _aggregate_metrics([traj1, traj2], crashes=0)
    # 3 total diagnostics
    # json_parse_ok: 2/3
    assert metrics["json_parse_rate"] == 2.0 / 3.0
    # schema_valid: 1/3
    assert metrics["schema_valid_rate"] == 1.0 / 3.0
    # safety_valid: 2/3
    assert metrics["safety_valid_rate"] == 2.0 / 3.0
    # action_type_valid: 2/3
    assert metrics["action_type_valid_rate"] == 2.0 / 3.0
    # arguments_valid: 1/3
    assert metrics["arguments_valid_rate"] == 1.0 / 3.0


def test_rates_zero_when_no_diagnostics():
    """When there are no step_diagnostics, rates should be 0.0 (not div-by-zero)."""
    traj = _make_traj(step_diags=[])
    metrics = _aggregate_metrics([traj], crashes=0)
    assert metrics["json_parse_rate"] == 0.0
    assert metrics["schema_valid_rate"] == 0.0
    assert metrics["safety_valid_rate"] == 0.0
    assert metrics["action_type_valid_rate"] == 0.0
    assert metrics["arguments_valid_rate"] == 0.0


# ---------------------------------------------------------------------------
# Test: forbidden_action_count summed across trajectories
# ---------------------------------------------------------------------------
def test_forbidden_action_count_summed():
    """forbidden_action_count should be the sum across all trajectories."""
    trajectories = [
        _make_traj(forbidden=2),
        _make_traj(forbidden=0),
        _make_traj(forbidden=3),
    ]
    metrics = _aggregate_metrics(trajectories, crashes=0)
    assert metrics["forbidden_action_count"] == 5


# ---------------------------------------------------------------------------
# Test: tool_dispatch_ok computed from tool_error_rate
# ---------------------------------------------------------------------------
def test_tool_dispatch_ok_from_tool_error_rate():
    """tool_dispatch_ok = 1 - mean(tool_error_rate) across trajectories."""
    trajectories = [
        _make_traj(tool_error_rate=0.0),  # all tools succeeded
        _make_traj(tool_error_rate=0.5),  # half failed
    ]
    metrics = _aggregate_metrics(trajectories, crashes=0)
    # mean tool_error_rate = (0.0 + 0.5) / 2 = 0.25
    # tool_dispatch_ok = 1 - 0.25 = 0.75
    assert metrics["tool_dispatch_ok"] == 0.75


# ---------------------------------------------------------------------------
# Test: max_step_stop_ok counts trajectories that hit max_steps
# ---------------------------------------------------------------------------
def test_max_step_stop_ok_counts_max_step_hits():
    """max_step_stop_ok = count of trajectories that hit max_steps and
    stopped cleanly (sum of max_step_exceeded_count)."""
    trajectories = [
        _make_traj(max_step_exceeded=1),  # hit max steps
        _make_traj(max_step_exceeded=0),  # finished naturally
        _make_traj(max_step_exceeded=1),  # hit max steps
    ]
    metrics = _aggregate_metrics(trajectories, crashes=0)
    assert metrics["max_step_stop_ok"] == 2


# ---------------------------------------------------------------------------
# Test: runtime_crash_count from crashes parameter
# ---------------------------------------------------------------------------
def test_runtime_crash_count():
    """runtime_crash_count should reflect the crashes parameter."""
    metrics = _aggregate_metrics([], crashes=3)
    assert metrics["runtime_crash_count"] == 3


# ---------------------------------------------------------------------------
# Test: model_load_ok / adapter_load_ok / generation_ok passed through
# ---------------------------------------------------------------------------
def test_load_and_generation_flags_passed_through():
    """model_load_ok, adapter_load_ok, generation_ok should be passed through."""
    metrics = _aggregate_metrics(
        [_make_traj()], crashes=0,
        model_load_ok=True, adapter_load_ok=False, generation_ok=True,
    )
    assert metrics["model_load_ok"] is True
    assert metrics["adapter_load_ok"] is False
    assert metrics["generation_ok"] is True


# ---------------------------------------------------------------------------
# Test: existing report contains all required fields
# ---------------------------------------------------------------------------
def test_existing_report_has_all_required_fields():
    """§2.3: the model-trajectory-collection-report.json must contain all
    11 required metric fields per config."""
    report_path = _ROOT / "reports" / "p4" / "model-trajectory-collection-report.json"
    if not report_path.exists():
        import pytest
        pytest.skip("report not generated yet")
    reports = json.loads(report_path.read_text(encoding="utf-8"))
    for entry in reports:
        for field in REQUIRED_METRIC_FIELDS:
            assert field in entry, (
                f"config '{entry.get('config', '?')}' missing field: {field}"
            )
