# tests/test_trust_repair.py
"""Issue #32 — P4.1b Trust Repair regression test matrix.

Covers:
- Schema tests (11): extra="forbid" on all Args/SafetyFlags/ActionBase models
- Repair tests (7): format-only repair must not change action semantics
- Metrics tests (8): independent dimension counts, numerator/denominator
- Protocol tests (5): JSON/Tag/DSL consistency on illegal actions

These tests exist because the original P4.1b ablation silently dropped
unknown fields (Pydantic default extra="ignore"), inflating schema_valid_rate
to a reported 96.25% that was not trustworthy.
"""
from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from src.agent_actions import (
    SafetyFlags, ActionBase, ListFilesArgs, ReadFileArgs, SearchTextArgs,
    InspectTaskArgs, ProposePatchArgs, ApplyPatchArgs, RollbackPatchArgs,
    RunTestsArgs, InspectErrorArgs, FinishArgs, ListFilesAction,
    ReadFileAction, FinishAction, P4ForbiddenActionError,
)
from src.protocols.base import ProtocolBase, ProtocolDiagnostics
from src.protocols.json_protocol import JsonProtocol
from src.protocols.tag_protocol import TagProtocol
from src.protocols.dsl_protocol import DslProtocol
from src.agent_model_provider import SentinelAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_safety() -> dict:
    return {
        "modifies_workspace": False,
        "executes_code": False,
        "network_required": False,
        "reads_sensitive_path": False,
        "is_terminal": False,
    }


def _valid_list_files_payload(**overrides) -> dict:
    data = {
        "action_type": "list_files",
        "action_id": "a1",
        "reason_short": "list files",
        "expected_observation": "files",
        "safety_flags": _valid_safety(),
        "arguments": {},
    }
    data.update(overrides)
    return data


def _valid_read_file_payload(**overrides) -> dict:
    data = {
        "action_type": "read_file",
        "action_id": "a1",
        "reason_short": "read file",
        "expected_observation": "contents",
        "safety_flags": _valid_safety(),
        "arguments": {"path": "solution.py"},
    }
    data.update(overrides)
    return data


def _valid_finish_payload(**overrides) -> dict:
    data = {
        "action_type": "finish",
        "action_id": "a1",
        "reason_short": "done",
        "expected_observation": "finished",
        "safety_flags": {**_valid_safety(), "is_terminal": True},
        "arguments": {
            "success_criterion": "test_pass",
            "tests_passed": True,
            "identification_verified": True,
            "summary": "all tests pass",
        },
    }
    data.update(overrides)
    return data


# ===========================================================================
# Section 1 — Schema tests (11)
# ===========================================================================

class TestStrictSchema:
    """Issue #32 Task A: all Pydantic models must use extra='forbid'."""

    def test_list_files_arguments_path_hard_fail(self):
        """list_files.arguments.path must be rejected (correct field is pattern)."""
        with pytest.raises(ValidationError) as exc:
            ListFilesArgs.model_validate({"path": "solution.py"})
        assert "path" in str(exc.value) or "Extra" in str(exc.value)

    def test_read_file_arguments_pattern_hard_fail(self):
        """read_file.arguments.pattern must be rejected (correct field is path)."""
        with pytest.raises(ValidationError) as exc:
            ReadFileArgs.model_validate({"pattern": "*.py"})
        assert "pattern" in str(exc.value) or "Extra" in str(exc.value)

    def test_inspect_task_unknown_argument_hard_fail(self):
        """inspect_task must reject any unknown argument."""
        with pytest.raises(ValidationError) as exc:
            InspectTaskArgs.model_validate({"path": "README.md"})
        assert "path" in str(exc.value) or "Extra" in str(exc.value)

    def test_finish_unknown_argument_hard_fail(self):
        """finish must reject unknown arguments."""
        with pytest.raises(ValidationError) as exc:
            FinishArgs.model_validate({
                "success_criterion": "test_pass",
                "tests_passed": True,
                "identification_verified": True,
                "summary": "done",
                "unexpected_arg": True,
            })
        assert "unexpected_arg" in str(exc.value) or "Extra" in str(exc.value)

    def test_unknown_top_level_action_field_hard_fail(self):
        """Unknown top-level field on ActionBase must be rejected."""
        data = _valid_list_files_payload(unexpected_top_level=True)
        action = ProtocolBase.validate_action(data)
        assert action is None, "Unknown top-level field must cause validation failure"

    def test_unknown_safety_flag_hard_fail(self):
        """Unknown field in safety_flags must be rejected."""
        data = _valid_list_files_payload()
        data["safety_flags"]["unknown_flag"] = False
        assert not ProtocolBase.check_safety_valid(data)
        assert ProtocolBase.validate_action(data) is None

    def test_valid_action_still_passes(self):
        """A fully valid action must still pass all checks."""
        data = _valid_read_file_payload()
        assert ProtocolBase.check_action_type_valid(data)
        assert ProtocolBase.check_safety_valid(data)
        assert ProtocolBase.check_arguments_valid(data)
        assert ProtocolBase.check_schema_valid(data)
        action = ProtocolBase.validate_action(data)
        assert action is not None

    def test_valid_optional_field_still_passes(self):
        """Optional fields (e.g. list_files.pattern=None) must still pass."""
        data = _valid_list_files_payload(arguments={"pattern": "*.py"})
        assert ProtocolBase.check_arguments_valid(data)
        assert ProtocolBase.validate_action(data) is not None

    def test_path_traversal_still_hard_fail(self):
        """Path traversal in read_file must still be rejected."""
        data = _valid_read_file_payload(arguments={"path": "../../../etc/passwd"})
        # Path validation runs in model_validator, so validate_action returns None
        assert ProtocolBase.validate_action(data) is None

    def test_network_required_true_hard_fail(self):
        """safety_flags.network_required=True must be rejected (P4.0 hard rule)."""
        from pydantic import TypeAdapter
        from src.agent_actions import Action
        adapter = TypeAdapter(Action)
        data = _valid_list_files_payload()
        data["safety_flags"]["network_required"] = True
        assert not ProtocolBase.check_safety_valid(data)
        # validate_action swallows the exception (returns None)
        assert ProtocolBase.validate_action(data) is None
        # Direct adapter call must raise P4ForbiddenActionError
        with pytest.raises(P4ForbiddenActionError):
            adapter.validate_python(data)

    def test_reads_sensitive_path_true_hard_fail(self):
        """safety_flags.reads_sensitive_path=True must be rejected."""
        from pydantic import TypeAdapter
        from src.agent_actions import Action
        adapter = TypeAdapter(Action)
        data = _valid_list_files_payload()
        data["safety_flags"]["reads_sensitive_path"] = True
        assert not ProtocolBase.check_safety_valid(data)
        assert ProtocolBase.validate_action(data) is None
        with pytest.raises(P4ForbiddenActionError):
            adapter.validate_python(data)


# ===========================================================================
# Section 2 — Repair semantic tests (7)
# ===========================================================================

class TestRepairSemantics:
    """Issue #32 Task B: format repair must not change action semantics."""

    def test_trailing_comma_can_be_repaired(self):
        """Trailing comma is a syntax issue, repair should fix it."""
        raw = json.dumps(_valid_read_file_payload())[:-2] + ",}"
        raw = raw.replace('"}', '"}')
        # Use JSON protocol which has repair logic
        proto = JsonProtocol()
        action, diag = proto.parse_output(raw)
        # Should parse (repair fixes trailing comma)
        assert diag.format_parse_ok
        # Either repaired successfully or schema_valid (depends on exact JSON)
        # The key check: repair_attempted is True
        assert diag.repair_attempted

    def test_markdown_fence_can_be_extracted(self):
        """Markdown fences are format-only, extraction should work."""
        payload = _valid_read_file_payload()
        raw = f"```json\n{json.dumps(payload)}\n```"
        proto = JsonProtocol()
        action, diag = proto.parse_output(raw)
        assert diag.format_parse_ok
        assert action is not None or diag.repair_attempted

    def test_repair_does_not_modify_action_type(self):
        """Repair must not change action_type."""
        payload = _valid_read_file_payload()
        # Add trailing comma to trigger repair
        raw = json.dumps(payload).replace('"}', '",}')
        proto = JsonProtocol()
        action, diag = proto.parse_output(raw)
        if action is not None and not isinstance(action, SentinelAction):
            assert action.action_type == "read_file"

    def test_repair_does_not_delete_unknown_argument(self):
        """Repair must not delete unknown arguments to make payload valid."""
        payload = _valid_read_file_payload()
        payload["arguments"]["unknown_arg"] = "value"
        raw = json.dumps(payload)
        proto = JsonProtocol()
        action, diag = proto.parse_output(raw)
        # Unknown arg must cause schema_valid=False, not be silently dropped
        assert diag.schema_valid is False
        assert diag.arguments_valid is False

    def test_repair_does_not_rename_unknown_argument(self):
        """Repair must not rename unknown arguments to known ones."""
        payload = _valid_list_files_payload()
        # 'path' is wrong for list_files (should be 'pattern')
        payload["arguments"] = {"path": "solution.py"}
        raw = json.dumps(payload)
        proto = JsonProtocol()
        action, diag = proto.parse_output(raw)
        # Must NOT be renamed to pattern and pass
        assert diag.schema_valid is False
        assert diag.arguments_valid is False

    def test_repair_does_not_inject_business_parameters(self):
        """Repair must not inject default business values for missing required args."""
        # read_file requires 'path' — repair must not inject it
        payload = _valid_read_file_payload()
        del payload["arguments"]["path"]
        raw = json.dumps(payload)
        proto = JsonProtocol()
        action, diag = proto.parse_output(raw)
        assert diag.schema_valid is False
        assert diag.arguments_valid is False

    def test_repaired_payload_still_requires_extra_forbid(self):
        """After format repair, the payload must still pass extra='forbid' checks."""
        payload = _valid_read_file_payload()
        payload["arguments"]["unknown"] = "x"
        # Wrap in markdown fence + trailing comma to force repair
        raw = f"```json\n{json.dumps(payload)},\n```"
        proto = JsonProtocol()
        action, diag = proto.parse_output(raw)
        # Even after repair, unknown field must cause failure
        assert diag.schema_valid is False


# ===========================================================================
# Section 3 — Metrics aggregation tests (8)
# ===========================================================================

class TestMetricsAggregation:
    """Issue #32 Tasks D/E/F: aggregation semantics."""

    def _make_traj(self, diags: list[dict], **kwargs) -> dict:
        traj = {
            "step_diagnostics": diags,
            "metrics": {"forbidden_action_count": 0, "finish_without_tests_count": 0},
            "success": False,
            "finish_claim_mismatch": False,
            "max_steps_hit": False,
        }
        traj.update(kwargs)
        return traj

    def test_format_parse_fail_not_counted_as_unknown_action(self):
        """FORMAT_PARSE_FAIL steps must not be counted as UNKNOWN_ACTION_TYPE."""
        from scripts.run_protocol_ablation import aggregate_metrics
        traj = self._make_traj([
            {"format_parse_ok": False, "schema_valid": False,
             "safety_valid": False, "action_type_valid": False,
             "arguments_valid": False, "failure_class": "FORMAT_PARSE_FAIL"},
        ])
        m = aggregate_metrics([traj], crashes=0)
        assert m["unknown_action_count"] == 0
        assert m["format_parse_success_steps"] == 0

    def test_unknown_action_type_correctly_counted(self):
        """Only failure_class==UNKNOWN_ACTION_TYPE counts as unknown action."""
        from scripts.run_protocol_ablation import aggregate_metrics
        traj = self._make_traj([
            {"format_parse_ok": True, "schema_valid": False,
             "safety_valid": True, "action_type_valid": False,
             "arguments_valid": False, "failure_class": "UNKNOWN_ACTION_TYPE"},
        ])
        m = aggregate_metrics([traj], crashes=0)
        assert m["unknown_action_count"] == 1

    def test_schema_failure_correctly_counted(self):
        """SCHEMA_VALIDATION_FAIL steps are counted in failure taxonomy."""
        from scripts.run_protocol_ablation import classify_failures
        traj = self._make_traj([
            {"failure_class": "SCHEMA_VALIDATION_FAIL"},
            {"failure_class": "SCHEMA_VALIDATION_FAIL"},
        ])
        taxonomy = classify_failures([traj])
        assert taxonomy["SCHEMA_VALIDATION_FAIL"] == 2

    def test_finish_without_tests_counted_in_failed_trajectory(self):
        """Failed trajectories that finished without tests must still be counted."""
        from scripts.run_protocol_ablation import aggregate_metrics
        traj = self._make_traj(
            [{"format_parse_ok": True, "schema_valid": True,
              "safety_valid": True, "action_type_valid": True,
              "arguments_valid": True, "failure_class": None}],
            success=False,
            metrics={"forbidden_action_count": 0, "finish_without_tests_count": 1},
        )
        m = aggregate_metrics([traj], crashes=0)
        assert m["finish_without_tests_count"] == 1
        # Task E: must NOT use `success and not tests_passed`
        # This trajectory is success=False but still counts

    def test_numerator_denominator_consistent_with_rate(self):
        """Rate must equal numerator/denominator for all step-level metrics."""
        from scripts.run_protocol_ablation import aggregate_metrics
        traj = self._make_traj([
            {"format_parse_ok": True, "schema_valid": True,
             "safety_valid": True, "action_type_valid": True,
             "arguments_valid": True, "failure_class": None},
            {"format_parse_ok": True, "schema_valid": False,
             "safety_valid": False, "action_type_valid": True,
             "arguments_valid": False, "failure_class": "SCHEMA_VALIDATION_FAIL"},
        ])
        m = aggregate_metrics([traj], crashes=0)
        assert m["total_steps"] == 2
        assert m["schema_valid_steps"] == 1
        assert m["schema_valid_rate"] == 0.5
        assert m["format_parse_success_steps"] == 2
        assert m["format_parse_rate"] == 1.0

    def test_zero_denominator_well_defined(self):
        """When total_steps=0, rates must be 0.0 (not raise)."""
        from scripts.run_protocol_ablation import aggregate_metrics
        m = aggregate_metrics([], crashes=0)
        assert m["total_steps"] == 0
        assert m["format_parse_rate"] == 0.0
        assert m["schema_valid_rate"] == 0.0
        assert m["task_success_rate"] == 0.0

    def test_repeated_loop_independently_counted(self):
        """REPEATED_ACTION_LOOP is detected at trajectory level."""
        from scripts.run_protocol_ablation import classify_failures
        traj = self._make_traj(
            [{"failure_class": None}] * 5,
            actions=[
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
                {"action_type": "read_file", "arguments": {"path": "a.py"}},
            ],
        )
        taxonomy = classify_failures([traj])
        assert taxonomy["REPEATED_ACTION_LOOP"] == 1

    def test_max_step_hit_independently_counted(self):
        """max_steps_hit is a trajectory-level metric, separate from step metrics."""
        from scripts.run_protocol_ablation import aggregate_metrics
        trajectories = [
            self._make_traj([{"failure_class": None}], max_steps_hit=True),
            self._make_traj([{"failure_class": None}], max_steps_hit=False),
        ]
        m = aggregate_metrics(trajectories, crashes=0)
        assert m["max_steps_hit_count"] == 1
        assert m["total_trajectories"] == 2
        assert m["max_steps_hit_rate"] == 0.5


# ===========================================================================
# Section 4 — Protocol consistency tests (5)
# ===========================================================================

class TestProtocolConsistency:
    """Issue #32 Task B: JSON/Tag/DSL must give consistent semantic judgments."""

    def test_json_unknown_field_hard_fail(self):
        """JSON protocol must reject unknown arguments fields."""
        proto = JsonProtocol()
        payload = _valid_list_files_payload()
        payload["arguments"] = {"path": "solution.py"}  # wrong for list_files
        raw = json.dumps(payload)
        action, diag = proto.parse_output(raw)
        assert diag.format_parse_ok
        assert not diag.arguments_valid
        assert not diag.schema_valid
        assert isinstance(action, SentinelAction)

    def test_tag_unknown_field_hard_fail(self):
        """Tag protocol must reject unknown argument keys."""
        proto = TagProtocol()
        # list_files with 'path' (wrong — should be 'pattern' or omitted)
        raw = "<action>\ntool: list_files\npath: solution.py\n</action>"
        action, diag = proto.parse_output(raw)
        assert diag.format_parse_ok
        assert diag.action_type_valid
        # 'path' is not a valid argument for list_files
        assert not diag.arguments_valid
        assert not diag.schema_valid

    def test_dsl_unknown_field_hard_fail(self):
        """DSL protocol must reject unknown key=value pairs in arguments."""
        proto = DslProtocol()
        raw = "ACTION list_files path=solution.py"
        action, diag = proto.parse_output(raw)
        assert diag.format_parse_ok
        assert diag.action_type_valid
        # 'path' is not valid for list_files
        assert not diag.arguments_valid
        assert not diag.schema_valid

    def test_three_protocols_consistent_on_illegal_action(self):
        """All three protocols must agree that an illegal action is illegal."""
        # list_files with 'path' argument is illegal in all protocols
        payloads = {
            "json": json.dumps(_valid_list_files_payload(
                arguments={"path": "solution.py"})),
            "tag": "<action>\ntool: list_files\npath: solution.py\n</action>",
            "dsl": "ACTION list_files path=solution.py",
        }
        results = {}
        for name, raw in payloads.items():
            if name == "json":
                proto = JsonProtocol()
            elif name == "tag":
                proto = TagProtocol()
            else:
                proto = DslProtocol()
            action, diag = proto.parse_output(raw)
            results[name] = (diag.schema_valid, diag.arguments_valid)

        # All three must agree: schema_valid=False, arguments_valid=False
        for name, (sv, av) in results.items():
            assert sv is False, f"{name} should reject unknown field"
            assert av is False, f"{name} should reject unknown arguments"

    def test_parser_does_not_produce_silent_noop(self):
        """When model outputs illegal field, parser must not silently
        drop it and return a valid action. The action must be rejected."""
        # The critical regression: list_files.arguments.path was silently
        # dropped, and the action was recorded as schema_valid=true.
        for proto_cls in (JsonProtocol, TagProtocol, DslProtocol):
            proto = proto_cls()
            if proto_cls is JsonProtocol:
                raw = json.dumps(_valid_list_files_payload(
                    arguments={"path": "solution.py"}))
            elif proto_cls is TagProtocol:
                raw = "<action>\ntool: list_files\npath: solution.py\n</action>"
            else:
                raw = "ACTION list_files path=solution.py"
            action, diag = proto.parse_output(raw)
            # Must NOT silently succeed
            assert diag.schema_valid is False, (
                f"{proto.name} silently accepted unknown field "
                f"(schema_valid=True but arguments had 'path' for list_files)"
            )
            assert isinstance(action, SentinelAction), (
                f"{proto.name} returned a real Action instead of SentinelAction "
                f"for illegal arguments"
            )


# ===========================================================================
# Section 5 — Finish defaults tests (Issue #32 Final Trust Repair)
# ===========================================================================

class TestFinishDefaults:
    """Issue #32 Final Trust Repair: finish business parameters must be
    explicitly provided by the model. No semantic defaults injected."""

    def test_tag_finish_missing_success_criterion_fails(self):
        """Tag finish without success_criterion must fail."""
        proto = TagProtocol()
        raw = (
            "<action>\ntool: finish\ntests_passed: true\n"
            "identification_verified: true\nsummary: done\n</action>"
        )
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_tag_finish_missing_tests_passed_fails(self):
        """Tag finish without tests_passed must fail."""
        proto = TagProtocol()
        raw = (
            "<action>\ntool: finish\nsuccess_criterion: test_pass\n"
            "identification_verified: true\nsummary: done\n</action>"
        )
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_tag_finish_missing_identification_verified_fails(self):
        """Tag finish without identification_verified must fail."""
        proto = TagProtocol()
        raw = (
            "<action>\ntool: finish\nsuccess_criterion: test_pass\n"
            "tests_passed: true\nsummary: done\n</action>"
        )
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_dsl_finish_missing_success_criterion_fails(self):
        """DSL finish without success_criterion must fail."""
        proto = DslProtocol()
        raw = (
            "ACTION finish tests_passed=true "
            "identification_verified=true summary=done"
        )
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_dsl_finish_missing_tests_passed_fails(self):
        """DSL finish without tests_passed must fail."""
        proto = DslProtocol()
        raw = (
            "ACTION finish success_criterion=test_pass "
            "identification_verified=true summary=done"
        )
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_dsl_finish_missing_identification_verified_fails(self):
        """DSL finish without identification_verified must fail."""
        proto = DslProtocol()
        raw = (
            "ACTION finish success_criterion=test_pass "
            "tests_passed=true summary=done"
        )
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_complete_tag_finish_passes(self):
        """Complete Tag finish with all required fields must pass."""
        proto = TagProtocol()
        raw = (
            "<action>\ntool: finish\nsuccess_criterion: test_pass\n"
            "tests_passed: true\nidentification_verified: true\n"
            "summary: all tests pass\n</action>"
        )
        action, diag = proto.parse_output(raw)
        assert not isinstance(action, SentinelAction)
        assert diag.schema_valid

    def test_complete_dsl_finish_passes(self):
        """Complete DSL finish with all required fields must pass."""
        proto = DslProtocol()
        raw = (
            "ACTION finish success_criterion=test_pass "
            "tests_passed=true identification_verified=true "
            "summary=all tests pass"
        )
        action, diag = proto.parse_output(raw)
        assert not isinstance(action, SentinelAction)
        assert diag.schema_valid

    def test_tag_parser_does_not_inject_finish_business_fields(self):
        """Tag parser must not inject success_criterion/tests_passed/
        identification_verified when model omits them."""
        proto = TagProtocol()
        raw = "<action>\ntool: finish\nsummary: done\n</action>"
        action, diag = proto.parse_output(raw)
        # Must fail — not pass via injected defaults
        assert isinstance(action, SentinelAction)
        assert not diag.arguments_valid


# ===========================================================================
# Section 6 — Boolean strictness tests (Issue #32 Final Trust Repair)
# ===========================================================================

class TestBooleanStrictness:
    """Issue #32 Final Trust Repair: strict boolean parsing.

    Only true/false/yes/no/1/0 are valid. Everything else hard-fails.
    Invalid booleans must NOT be silently converted to False.
    """

    def _tag_finish_with_tests_passed(self, value: str):
        proto = TagProtocol()
        raw = (
            f"<action>\ntool: finish\nsuccess_criterion: test_pass\n"
            f"tests_passed: {value}\nidentification_verified: true\n"
            f"summary: done\n</action>"
        )
        return proto.parse_output(raw)

    def _dsl_finish_with_tests_passed(self, value: str):
        proto = DslProtocol()
        raw = (
            f"ACTION finish success_criterion=test_pass "
            f"tests_passed={value} identification_verified=true "
            f"summary=done"
        )
        return proto.parse_output(raw)

    def test_tag_tests_passed_true(self):
        action, diag = self._tag_finish_with_tests_passed("true")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.tests_passed is True

    def test_tag_tests_passed_false(self):
        action, diag = self._tag_finish_with_tests_passed("false")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.tests_passed is False

    def test_tag_tests_passed_yes(self):
        action, diag = self._tag_finish_with_tests_passed("yes")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.tests_passed is True

    def test_tag_tests_passed_no(self):
        action, diag = self._tag_finish_with_tests_passed("no")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.tests_passed is False

    def test_tag_tests_passed_1(self):
        action, diag = self._tag_finish_with_tests_passed("1")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.tests_passed is True

    def test_tag_tests_passed_0(self):
        action, diag = self._tag_finish_with_tests_passed("0")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.tests_passed is False

    def test_tag_tests_passed_banana_hard_fails(self):
        """tests_passed=banana must hard-fail, NOT become False."""
        action, diag = self._tag_finish_with_tests_passed("banana")
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_tests_passed_2_hard_fails(self):
        """tests_passed=2 must hard-fail (only 0/1 are valid)."""
        action, diag = self._tag_finish_with_tests_passed("2")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_tests_passed_negative1_hard_fails(self):
        """tests_passed=-1 must hard-fail."""
        action, diag = self._tag_finish_with_tests_passed("-1")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_tests_passed_empty_hard_fails(self):
        """Empty tests_passed must hard-fail."""
        action, diag = self._tag_finish_with_tests_passed("")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_invalid_boolean_not_converted_to_false(self):
        """Critical: banana must NOT silently become False.

        If it became False, a complete finish action would PASS.
        The fact that it FAILS proves no silent conversion happened.
        """
        action, diag = self._tag_finish_with_tests_passed("banana")
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_dsl_tests_passed_true(self):
        action, diag = self._dsl_finish_with_tests_passed("true")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.tests_passed is True

    def test_dsl_tests_passed_false(self):
        action, diag = self._dsl_finish_with_tests_passed("false")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.tests_passed is False

    def test_dsl_tests_passed_yes(self):
        action, diag = self._dsl_finish_with_tests_passed("yes")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.tests_passed is True

    def test_dsl_tests_passed_no(self):
        action, diag = self._dsl_finish_with_tests_passed("no")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.tests_passed is False

    def test_dsl_tests_passed_banana_hard_fails(self):
        """DSL tests_passed=banana must hard-fail."""
        action, diag = self._dsl_finish_with_tests_passed("banana")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_dsl_invalid_boolean_not_converted_to_false(self):
        """DSL: banana must NOT silently become False."""
        action, diag = self._dsl_finish_with_tests_passed("banana")
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid


# ===========================================================================
# Section 7 — Numeric strictness tests (Issue #32 Final Trust Repair)
# ===========================================================================

class TestNumericStrictness:
    """Issue #32 Final Trust Repair: strict integer/float parsing.

    Invalid numbers must hard-fail, not be kept as strings or use defaults.
    """

    def _tag_search_with_max_results(self, value: str):
        proto = TagProtocol()
        raw = f"<action>\ntool: search_text\nquery: def\nmax_results: {value}\n</action>"
        return proto.parse_output(raw)

    def _dsl_search_with_max_results(self, value: str):
        proto = DslProtocol()
        raw = f"ACTION search_text query=def max_results={value}"
        return proto.parse_output(raw)

    def _tag_run_tests_with_timeout(self, value: str):
        proto = TagProtocol()
        raw = f"<action>\ntool: run_tests\ntimeout_s: {value}\n</action>"
        return proto.parse_output(raw)

    def _dsl_run_tests_with_timeout(self, value: str):
        proto = DslProtocol()
        raw = f"ACTION run_tests timeout_s={value}"
        return proto.parse_output(raw)

    def test_tag_max_results_20_correct(self):
        action, diag = self._tag_search_with_max_results("20")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.max_results == 20

    def test_tag_max_results_abc_hard_fails(self):
        action, diag = self._tag_search_with_max_results("abc")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_max_results_float_hard_fails(self):
        """1.5 is not a valid integer."""
        action, diag = self._tag_search_with_max_results("1.5")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_max_results_empty_hard_fails(self):
        action, diag = self._tag_search_with_max_results("")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_timeout_s_1_5_correct(self):
        action, diag = self._tag_run_tests_with_timeout("1.5")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.timeout_s == 1.5

    def test_tag_timeout_s_10_correct(self):
        action, diag = self._tag_run_tests_with_timeout("10")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.timeout_s == 10.0

    def test_tag_timeout_s_abc_hard_fails(self):
        action, diag = self._tag_run_tests_with_timeout("abc")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_timeout_s_nan_hard_fails(self):
        action, diag = self._tag_run_tests_with_timeout("NaN")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_timeout_s_infinity_hard_fails(self):
        action, diag = self._tag_run_tests_with_timeout("Infinity")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_dsl_max_results_20_correct(self):
        action, diag = self._dsl_search_with_max_results("20")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.max_results == 20

    def test_dsl_max_results_abc_hard_fails(self):
        action, diag = self._dsl_search_with_max_results("abc")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_dsl_timeout_s_1_5_correct(self):
        action, diag = self._dsl_run_tests_with_timeout("1.5")
        assert not isinstance(action, SentinelAction)
        assert action.arguments.timeout_s == 1.5

    def test_dsl_timeout_s_abc_hard_fails(self):
        action, diag = self._dsl_run_tests_with_timeout("abc")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_invalid_number_does_not_use_default(self):
        """max_results=abc must fail, not use default 20.

        If default were used, the action would pass with max_results=20.
        The fact that it fails proves no default substitution happened.
        """
        action, diag = self._tag_search_with_max_results("abc")
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid


# ===========================================================================
# Section 8 — Semantic preservation tests (Issue #32 Final Trust Repair)
# ===========================================================================

class TestSemanticPreservation:
    """Issue #32 Final Trust Repair: parser must not alter model semantics.

    No field renaming, no field deletion, no business param injection,
    no action replacement, no silent no-op.
    """

    def test_tag_does_not_rename_fields(self):
        """Tag must not rename 'path' to 'pattern' for list_files."""
        proto = TagProtocol()
        raw = "<action>\ntool: list_files\npath: solution.py\n</action>"
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_dsl_does_not_rename_fields(self):
        """DSL must not rename 'path' to 'pattern' for list_files."""
        proto = DslProtocol()
        raw = "ACTION list_files path=solution.py"
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_tag_does_not_delete_unknown_fields(self):
        """Tag must not delete unknown fields to make payload valid."""
        proto = TagProtocol()
        raw = "<action>\ntool: read_file\npath: solution.py\nbadkey: value\n</action>"
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_dsl_does_not_delete_unknown_fields(self):
        """DSL must not delete unknown key=value pairs."""
        proto = DslProtocol()
        raw = "ACTION read_file path=solution.py badkey=value"
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_tag_does_not_inject_business_params(self):
        """Tag must not inject finish business params."""
        proto = TagProtocol()
        raw = "<action>\ntool: finish\nsummary: done\n</action>"
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_dsl_does_not_inject_business_params(self):
        """DSL must not inject finish business params."""
        proto = DslProtocol()
        raw = "ACTION finish summary=done"
        action, diag = proto.parse_output(raw)
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_tag_semantic_error_returns_sentinel(self):
        """All Tag semantic errors must return SentinelAction."""
        proto = TagProtocol()
        # Unknown action type
        action, diag = proto.parse_output(
            "<action>\ntool: run_terminal\ncommand: rm\n</action>"
        )
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "UNKNOWN_ACTION_TYPE"
        # Invalid scalar
        action, diag = proto.parse_output(
            "<action>\ntool: finish\nsuccess_criterion: test_pass\n"
            "tests_passed: banana\nidentification_verified: true\n"
            "summary: done\n</action>"
        )
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_dsl_semantic_error_returns_sentinel(self):
        """All DSL semantic errors must return SentinelAction."""
        proto = DslProtocol()
        # Unknown action type
        action, diag = proto.parse_output("ACTION run_terminal command=rm")
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "UNKNOWN_ACTION_TYPE"
        # Invalid scalar
        action, diag = proto.parse_output(
            "ACTION finish success_criterion=test_pass "
            "tests_passed=banana identification_verified=true summary=done"
        )
        assert isinstance(action, SentinelAction)
        assert diag.failure_class == "SCHEMA_VALIDATION_FAIL"

    def test_tag_no_silent_coercion(self):
        """Tag must not silently coerce invalid values to valid ones."""
        proto = TagProtocol()
        # banana must not become False
        action, diag = proto.parse_output(
            "<action>\ntool: finish\nsuccess_criterion: test_pass\n"
            "tests_passed: banana\nidentification_verified: true\n"
            "summary: done\n</action>"
        )
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid

    def test_dsl_no_silent_coercion(self):
        """DSL must not silently coerce invalid values to valid ones."""
        proto = DslProtocol()
        action, diag = proto.parse_output(
            "ACTION finish success_criterion=test_pass "
            "tests_passed=banana identification_verified=true summary=done"
        )
        assert isinstance(action, SentinelAction)
        assert not diag.schema_valid


# ===========================================================================
# Section 9 — Reproducibility tests (Issue #32 Final Trust Repair)
# ===========================================================================

class TestReproducibility:
    """Issue #32 Final Trust Repair: experiment must be reproducible from
    a committed state. No uncommitted working-tree code allowed."""

    def test_git_worktree_clean_check_returns_bool(self):
        """_git_worktree_clean_for_experiment must return a bool."""
        from scripts.run_protocol_ablation import _git_worktree_clean_for_experiment
        result = _git_worktree_clean_for_experiment()
        assert isinstance(result, bool)

    def test_assert_clean_experiment_state_exists(self):
        """assert_clean_experiment_state must be callable."""
        from scripts.run_protocol_ablation import assert_clean_experiment_state
        assert callable(assert_clean_experiment_state)

    def test_assert_clean_experiment_state_exits_on_dirty(self):
        """assert_clean_experiment_state must sys.exit(1) if worktree dirty."""
        from scripts.run_protocol_ablation import assert_clean_experiment_state
        from unittest.mock import patch, MagicMock
        with patch("scripts.run_protocol_ablation._git_worktree_clean_for_experiment",
                   return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                assert_clean_experiment_state()
            assert exc_info.value.code == 1

    def test_assert_clean_experiment_state_passes_on_clean(self):
        """assert_clean_experiment_state must not exit if worktree clean."""
        from scripts.run_protocol_ablation import assert_clean_experiment_state
        from unittest.mock import patch
        with patch("scripts.run_protocol_ablation._git_worktree_clean_for_experiment",
                   return_value=True):
            # Should not raise SystemExit
            assert_clean_experiment_state()

    def test_baseline_lock_includes_worktree_clean_field(self):
        """baseline_lock must include git_worktree_clean_for_experiment."""
        from scripts.run_protocol_ablation import baseline_lock
        lock = baseline_lock()
        assert "git_worktree_clean_for_experiment" in lock
        assert isinstance(lock["git_worktree_clean_for_experiment"], bool)

    def test_baseline_lock_commit_sha_matches_head(self):
        """baseline_lock.experiment_commit_sha must equal git HEAD."""
        import subprocess
        from scripts.run_protocol_ablation import baseline_lock, _ROOT
        lock = baseline_lock()
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        head_sha = result.stdout.strip()
        assert lock["experiment_commit_sha"] == head_sha

    def test_artifact_manifest_excludes_self(self):
        """generate_artifact_manifest must not include artifact-manifest.json."""
        import tempfile
        from pathlib import Path
        from scripts.run_protocol_ablation import generate_artifact_manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            # Create some test files
            (report_dir / "baseline-lock.json").write_text('{"test": 1}')
            (report_dir / "trajectories").mkdir()
            (report_dir / "trajectories" / "json-base.jsonl").write_text(
                '{"trajectory_id": "test"}\n'
            )
            # Generate manifest
            manifest = generate_artifact_manifest(report_dir)
            # Verify artifact-manifest.json is NOT in the artifacts list
            for artifact in manifest["artifacts"]:
                assert artifact["relative_path"] != "artifact-manifest.json"
            assert manifest["artifact_count"] == 2  # baseline-lock + jsonl


# ===========================================================================
# Section 10 — Scalar parser unit tests (Issue #32 Final Trust Repair)
# ===========================================================================

class TestScalarParsers:
    """Direct unit tests for _parse_bool, _parse_int, _parse_float."""

    def test_parse_bool_true(self):
        from src.protocols.base import _parse_bool
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("yes") is True
        assert _parse_bool("1") is True

    def test_parse_bool_false(self):
        from src.protocols.base import _parse_bool
        assert _parse_bool("false") is False
        assert _parse_bool("False") is False
        assert _parse_bool("no") is False
        assert _parse_bool("0") is False

    def test_parse_bool_invalid_raises(self):
        from src.protocols.base import _parse_bool
        for invalid in ("banana", "truthy", "null", "none", "2", "-1", ""):
            with pytest.raises(ValueError):
                _parse_bool(invalid)

    def test_parse_bool_whitespace_trimmed(self):
        from src.protocols.base import _parse_bool
        assert _parse_bool("  true  ") is True
        assert _parse_bool("  false  ") is False

    def test_parse_int_valid(self):
        from src.protocols.base import _parse_int
        assert _parse_int("0") == 0
        assert _parse_int("1") == 1
        assert _parse_int("20") == 20
        assert _parse_int("-1") == -1

    def test_parse_int_invalid_raises(self):
        from src.protocols.base import _parse_int
        for invalid in ("1.5", "abc", "1px", ""):
            with pytest.raises(ValueError):
                _parse_int(invalid)

    def test_parse_float_valid(self):
        from src.protocols.base import _parse_float
        assert _parse_float("0") == 0.0
        assert _parse_float("1") == 1.0
        assert _parse_float("1.5") == 1.5
        assert _parse_float("10.0") == 10.0

    def test_parse_float_invalid_raises(self):
        from src.protocols.base import _parse_float
        for invalid in ("abc", "1s", ""):
            with pytest.raises(ValueError):
                _parse_float(invalid)

    def test_parse_float_non_finite_raises(self):
        from src.protocols.base import _parse_float
        for invalid in ("NaN", "Infinity", "-inf", "inf"):
            with pytest.raises(ValueError):
                _parse_float(invalid)
