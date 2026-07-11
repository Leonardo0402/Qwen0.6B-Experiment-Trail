"""Tests for ProtocolBase and ProtocolDiagnostics."""
import pytest
from src.protocols.base import ProtocolBase, ProtocolDiagnostics


def test_protocol_diagnostics_has_all_fields():
    diag = ProtocolDiagnostics(
        raw_output="test",
        format_parse_ok=True,
        schema_valid=False,
        safety_valid=False,
        action_type_valid=True,
        arguments_valid=False,
        repair_attempted=False,
        repair_success=False,
        latency_ms=42,
    )
    assert diag.failure_class is None
    assert diag.latency_ms == 42
    assert diag.raw_output == "test"


def test_protocol_diagnostics_failure_class_set():
    diag = ProtocolDiagnostics(
        raw_output="",
        format_parse_ok=False,
        schema_valid=False,
        safety_valid=False,
        action_type_valid=False,
        arguments_valid=False,
        repair_attempted=False,
        repair_success=False,
        latency_ms=0,
        failure_class="FORMAT_PARSE_FAIL",
    )
    assert diag.failure_class == "FORMAT_PARSE_FAIL"


def test_protocol_diagnostics_model_dump_works():
    """ProtocolDiagnostics must support model_dump() for trajectory recording."""
    diag = ProtocolDiagnostics(
        raw_output="x", format_parse_ok=True, schema_valid=True,
        safety_valid=True, action_type_valid=True, arguments_valid=True,
        repair_attempted=False, repair_success=False, latency_ms=5,
    )
    d = diag.model_dump()
    assert d["format_parse_ok"] is True
    assert d["schema_valid"] is True
    assert "failure_class" in d


def test_validate_action_returns_none_for_invalid():
    result = ProtocolBase.validate_action({"action_type": "nonexistent"})
    assert result is None


def test_validate_action_returns_none_for_empty():
    result = ProtocolBase.validate_action({})
    assert result is None


def test_is_valid_action_type_recognizes_11_types():
    for at in ["list_files", "read_file", "search_text", "inspect_task",
               "propose_patch", "apply_patch", "rollback_patch", "run_tests",
               "inspect_error", "write_memory", "finish"]:
        assert ProtocolBase.is_valid_action_type(at), f"{at} should be valid"


def test_is_valid_action_type_rejects_unknown():
    assert not ProtocolBase.is_valid_action_type("run_terminal")
    assert not ProtocolBase.is_valid_action_type("")
    assert not ProtocolBase.is_valid_action_type("READ_FILE")
