# tests/test_protocol_ablation.py
"""Integration tests for protocol ablation script.

Tests use MockProtocol to avoid loading the actual model.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.protocols.base import ProtocolBase, ProtocolDiagnostics
from src.agent_model_provider import SentinelAction


class MockProtocol(ProtocolBase):
    """Protocol that returns a predefined action for testing."""
    def __init__(self, action_type="list_files"):
        self._action_type = action_type

    @property
    def name(self):
        return "mock"

    def build_system_prompt(self, task_context):
        return f"Mock prompt for {task_context}"

    def parse_output(self, raw):
        diag = ProtocolDiagnostics(
            raw_output=raw, format_parse_ok=True, schema_valid=True,
            safety_valid=True, action_type_valid=True, arguments_valid=True,
            repair_attempted=False, repair_success=False, latency_ms=1,
        )
        if self._action_type == "invalid":
            return SentinelAction(reason="mock invalid"), diag
        from src.agent_actions import ListFilesAction, SafetyFlags
        action = ListFilesAction(
            action_id="mock_a1", reason_short="mock",
            expected_observation="mock",
            safety_flags=SafetyFlags(
                modifies_workspace=False, executes_code=False,
                network_required=False, reads_sensitive_path=False,
                is_terminal=False,
            ),
        )
        return action, diag


def test_baseline_lock_records_all_fields():
    from scripts.run_protocol_ablation import baseline_lock
    lock = baseline_lock()
    assert "commit_sha" in lock
    assert "micro_task_manifest_sha256" in lock
    assert "model_path" in lock
    assert "adapter_path" in lock
    assert "generation_config" in lock
    assert lock["generation_config"]["temperature"] == 0.0
    assert lock["generation_config"]["do_sample"] is False
    assert "total_tasks" in lock
    assert lock["total_tasks"] == 40


def test_aggregate_metrics_computes_all_fields():
    from scripts.run_protocol_ablation import aggregate_metrics
    trajectories = [
        {
            "step_diagnostics": [
                {"format_parse_ok": True, "schema_valid": True,
                 "safety_valid": True, "action_type_valid": True,
                 "arguments_valid": True, "failure_class": None},
                {"format_parse_ok": True, "schema_valid": False,
                 "safety_valid": False, "action_type_valid": False,
                 "arguments_valid": False, "failure_class": "SCHEMA_VALIDATION_FAIL"},
            ],
            "metrics": {"forbidden_action_count": 0, "tool_error_rate": 0.0,
                        "max_step_exceeded_count": 1},
            "success": False,
            "finish_claim_mismatch": True,
            "steps_executed": 12,
        },
    ]
    metrics = aggregate_metrics(trajectories, crashes=0)
    # Spec §6.2 requires 12+ metrics
    assert "format_parse_rate" in metrics
    assert "schema_valid_rate" in metrics
    assert "safety_valid_rate" in metrics
    assert "action_type_valid_rate" in metrics
    assert "arguments_valid_rate" in metrics
    assert "forbidden_action_count" in metrics
    assert "unknown_action_count" in metrics
    assert "task_success_rate" in metrics
    assert "finish_without_tests_count" in metrics
    assert "finish_claim_mismatch_count" in metrics
    assert "max_steps_hit_rate" in metrics
    assert "runtime_crash_count" in metrics
    assert metrics["format_parse_rate"] == 1.0
    assert metrics["schema_valid_rate"] == 0.5
    assert metrics["unknown_action_count"] == 1  # one step had action_type_valid=False
    assert metrics["finish_claim_mismatch_count"] == 1


def test_classify_failures_returns_taxonomy():
    from scripts.run_protocol_ablation import classify_failures
    trajectories = [
        {
            "step_diagnostics": [
                {"failure_class": "FORMAT_PARSE_FAIL"},
                {"failure_class": "SCHEMA_VALIDATION_FAIL"},
                {"failure_class": "UNKNOWN_ACTION_TYPE"},
                {"failure_class": None},  # success
            ],
            "actions": [
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
            ],
        },
    ]
    taxonomy = classify_failures(trajectories)
    assert taxonomy["FORMAT_PARSE_FAIL"] == 1
    assert taxonomy["SCHEMA_VALIDATION_FAIL"] == 1
    assert taxonomy["UNKNOWN_ACTION_TYPE"] == 1
    assert taxonomy["REPEATED_ACTION_LOOP"] == 1  # detected from actions


def test_run_combination_with_mock_protocol():
    """Test that run_combination works with a mock protocol (no model loading)."""
    from scripts.run_protocol_ablation import run_combination, _TASKS_DIR
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    proto = MockProtocol(action_type="list_files")
    config = {"name": "mock-base", "model_path": "fake", "adapter_path": None}

    # Use only first 2 tasks for speed
    manifest = json.loads((_TASKS_DIR / "manifest.json").read_text(encoding="utf-8"))
    task_ids = [t["task_id"] for t in manifest["tasks"][:2]]

    with patch("scripts.run_protocol_ablation.ModelActionProvider") as MockProvider:
        # Create a mock provider that uses our protocol
        def make_provider(*args, **kwargs):
            provider = MagicMock()
            provider._protocol = kwargs.get("protocol")
            provider._model = MagicMock()
            provider._load_model = MagicMock()  # succeed, don't raise
            provider.diagnostics = []

            # Iterator state — reset() recreates it so each task gets fresh actions
            actions_state = {"iter": iter(["list_files", "finish"])}

            def reset():
                actions_state["iter"] = iter(["list_files", "finish"])
                provider.diagnostics = []

            def next_action(state):
                at = next(actions_state["iter"])
                if at == "list_files":
                    from src.agent_actions import ListFilesAction, SafetyFlags
                    return ListFilesAction(
                        action_id="a1", reason_short="test",
                        expected_observation="files",
                        safety_flags=SafetyFlags(
                            modifies_workspace=False, executes_code=False,
                            network_required=False, reads_sensitive_path=False,
                            is_terminal=False,
                        ),
                    )
                else:
                    from src.agent_actions import FinishAction, FinishArgs, SafetyFlags, TaskSuccessCriterion
                    return FinishAction(
                        action_id="a2", reason_short="done",
                        expected_observation="finished",
                        safety_flags=SafetyFlags(
                            modifies_workspace=False, executes_code=False,
                            network_required=False, reads_sensitive_path=False,
                            is_terminal=True,
                        ),
                        arguments=FinishArgs(
                            success_criterion=TaskSuccessCriterion.TEST_PASS,
                            tests_passed=False, identification_verified=False,
                            summary="mock finish",
                        ),
                    )
            provider.next_action = next_action
            provider.reset = reset
            return provider
        MockProvider.side_effect = make_provider

        result = run_combination(proto, config, task_ids, max_steps=5)
        assert result["config"] == "mock-base"
        assert result["trajectories_written"] == 2
        assert "metrics" in result


def test_generate_report_contains_all_protocols():
    from scripts.run_protocol_ablation import generate_report
    results = [
        {"protocol": "json", "config": "base", "metrics": {"schema_valid_rate": 0.0}},
        {"protocol": "json", "config": "repair-lora", "metrics": {"schema_valid_rate": 0.0}},
        {"protocol": "tag", "config": "base", "metrics": {"schema_valid_rate": 0.5}},
        {"protocol": "tag", "config": "repair-lora", "metrics": {"schema_valid_rate": 0.6}},
        {"protocol": "dsl", "config": "base", "metrics": {"schema_valid_rate": 0.3}},
        {"protocol": "dsl", "config": "repair-lora", "metrics": {"schema_valid_rate": 0.4}},
    ]
    taxonomy = {"FORMAT_PARSE_FAIL": 10, "SCHEMA_VALIDATION_FAIL": 20}
    report = generate_report(results, taxonomy)
    assert "json" in report
    assert "tag" in report
    assert "dsl" in report
    assert "schema_valid_rate" in report
    assert "FORMAT_PARSE_FAIL" in report


def test_generate_report_has_markdown_table():
    from scripts.run_protocol_ablation import generate_report
    results = [
        {"protocol": "json", "config": "base",
         "metrics": {"format_parse_rate": 1.0, "schema_valid_rate": 0.0,
                     "safety_valid_rate": 0.0, "action_type_valid_rate": 0.5,
                     "arguments_valid_rate": 0.0, "forbidden_action_count": 0,
                     "task_success_rate": 0.0, "max_steps_hit_rate": 1.0,
                     "runtime_crash_count": 0}},
    ]
    taxonomy = {"FORMAT_PARSE_FAIL": 5}
    report = generate_report(results, taxonomy)
    assert "|" in report  # markdown table
    assert "format_parse_rate" in report


def test_verdict_keep_action_json_when_json_best():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "KEEP_ACTION_JSON"


def test_verdict_try_tag_when_tag_significantly_better():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "TRY_TAG_PROTOCOL_FOR_P4_2"


def test_verdict_try_dsl_when_dsl_significantly_better():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.6, "safety_valid_rate": 0.6, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.6, "safety_valid_rate": 0.6, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "TRY_DSL_FOR_P4_2"


def test_verdict_fix_prompt_when_all_below_30pct():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "FIX_PROMPT_FIRST"


def test_verdict_fix_evaluator_when_model_load_fails():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "FIX_EVALUATOR_FIRST"


def test_verdict_fix_evaluator_when_high_crash():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True,
         "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 30}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "FIX_EVALUATOR_FIRST"


def test_verdict_is_valid_enum():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True,
         "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True,
         "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    allowed = {
        "KEEP_ACTION_JSON", "TRY_TAG_PROTOCOL_FOR_P4_2", "TRY_DSL_FOR_P4_2",
        "FIX_PROMPT_FIRST", "FIX_EVALUATOR_FIRST", "STOP_PROTOCOL_CHANGE",
    }
    assert verdict in allowed


def test_verdict_stop_protocol_change_on_fallback():
    """Rule 5: fallback when no clear winner and JSON not best."""
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.4, "safety_valid_rate": 0.4, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.4, "safety_valid_rate": 0.4, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.45, "safety_valid_rate": 0.45, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.45, "safety_valid_rate": 0.45, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.3, "safety_valid_rate": 0.3, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.3, "safety_valid_rate": 0.3, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "STOP_PROTOCOL_CHANGE"
