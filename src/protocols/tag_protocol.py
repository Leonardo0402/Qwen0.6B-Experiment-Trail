# src/protocols/tag_protocol.py
"""Protocol B — XML Tag action protocol.

Lower-entropy format: model outputs key-value pairs inside <action> tags.
Parser fills in default safety_flags and other required ActionBase fields
to reduce the model's formatting burden.

Format:
    <action>
    tool: read_file
    path: solution.py
    </action>

Multiline content uses <content>...</content> subtags:
    <action>
    tool: propose_patch
    file_path: solution.py
    old_text: <content>def old():
        pass</content>
    new_text: <content>def new():
        return 42</content>
    </action>

<think>...</think> blocks are ignored.
Unknown keys hard-fail (not silently ignored).
"""
from __future__ import annotations

import re
import time

from src.agent_actions import Action
from src.agent_model_provider import SentinelAction
from src.protocols.base import ProtocolBase, ProtocolDiagnostics, _parse_bool, _parse_int, _parse_float

_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_CONTENT_RE = re.compile(r"<content>(.*?)</content>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_TOOL_LIST = (
    "list_files, read_file, search_text, inspect_task, "
    "propose_patch, apply_patch, rollback_patch, run_tests, "
    "inspect_error, write_memory, finish"
)

# Keys recognized by the tag protocol.
# "tool" maps to action_type; others map to ActionBase fields or arguments.
_BASE_FIELD_MAP = {
    "action_id": "action_id",
    "reason": "reason_short",
    "expected": "expected_observation",
}

_ARGUMENT_KEYS = frozenset({
    "path", "file_path", "query", "file_glob", "max_results",
    "old_text", "new_text", "action_id_ref", "test_path", "timeout_s",
    "error_source", "notes", "hypothesis",
    "success_criterion", "tests_passed", "identification_verified",
    "summary", "pattern", "start_line", "end_line",
    "expected_before_sha256",
})

_FINISH_KEYS = frozenset({
    "success_criterion", "tests_passed", "identification_verified", "summary",
})


def _default_safety_flags(action_type: str) -> dict:
    """Infer safe safety_flags from action_type."""
    return {
        "modifies_workspace": action_type in ("apply_patch", "rollback_patch"),
        "executes_code": action_type == "run_tests",
        "network_required": False,
        "reads_sensitive_path": False,
        "is_terminal": action_type == "finish",
    }


def _parse_kv_pairs(block: str) -> dict[str, str]:
    """Parse key-value pairs from an <action> block.

    Handles <content>...</content> subtags for multiline values.
    """
    # Extract content subtags and replace with placeholders
    contents: list[str] = []

    def _store_content(m):
        contents.append(m.group(1))
        return f"__CONTENT_{len(contents) - 1}__"

    block = _CONTENT_RE.sub(_store_content, block)

    result: dict[str, str] = {}
    for line in block.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Restore content placeholders
        for i, content in enumerate(contents):
            value = value.replace(f"__CONTENT_{i}__", content)
        result[key] = value
    return result


class TagProtocol(ProtocolBase):
    """XML Tag protocol — lower-entropy key-value format."""

    @property
    def name(self) -> str:
        return "tag"

    def build_system_prompt(self, task_context: str) -> str:
        return "\n".join([
            f"Task: {task_context}",
            "",
            f"Choose ONE action from: {_TOOL_LIST}",
            "",
            "Format your action as:",
            "<action>",
            "tool: <action_type>",
            "<argument_key>: <value>",
            "</action>",
            "",
            "Rules:",
            "- 'tool' is the action type (e.g. read_file, run_tests, finish)",
            "- Provide only the arguments needed for that action type",
            "- For multiline content (old_text, new_text), use: <content>...</content>",
            "- You may use <think>...</think> to reason before the action",
            "",
            "Examples:",
            "<action>",
            "tool: read_file",
            "path: solution.py",
            "</action>",
            "",
            "<action>",
            "tool: propose_patch",
            "file_path: solution.py",
            "old_text: <content>def old():\n    pass</content>",
            "new_text: <content>def new():\n    return 42</content>",
            "</action>",
            "",
            "Output ONLY the <action> block. No preamble, no explanation.",
        ])

    def parse_output(self, raw: str) -> tuple[Action | SentinelAction, ProtocolDiagnostics]:
        t0 = time.monotonic()
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Strip <think> blocks
        cleaned = _THINK_RE.sub("", raw)

        m = _ACTION_TAG_RE.search(cleaned)
        format_parse_ok = m is not None

        diag = ProtocolDiagnostics(
            raw_output=raw,
            format_parse_ok=format_parse_ok,
            schema_valid=False,
            safety_valid=False,
            action_type_valid=False,
            arguments_valid=False,
            repair_attempted=False,
            repair_success=False,
            latency_ms=latency_ms,
        )

        if not format_parse_ok:
            diag.failure_class = "FORMAT_PARSE_FAIL"
            return SentinelAction(reason="no <action> tag found"), diag

        block = m.group(1)
        kv = _parse_kv_pairs(block)

        # Extract action_type
        action_type = kv.pop("tool", "")
        diag.action_type_valid = self.is_valid_action_type(action_type)

        if not diag.action_type_valid:
            diag.failure_class = "UNKNOWN_ACTION_TYPE"
            return SentinelAction(reason=f"unknown action type: {action_type}"), diag

        # Build data dict for Pydantic validation
        data: dict = {
            "action_type": action_type,
            "action_id": kv.pop("action_id", f"tag_{action_type}"),
            "reason_short": kv.pop("reason", "no reason provided"),
            "expected_observation": kv.pop("expected", "unknown"),
            "safety_flags": _default_safety_flags(action_type),
        }

        # Map remaining keys to arguments.
        # Issue #32 Trust Repair: ALL keys are passed through to arguments,
        # including unknown ones. Previously unknown keys were silently
        # filtered out (like Pydantic's extra="ignore"). Now they are kept
        # and rejected by check_arguments_valid via extra="forbid" on Args.
        arguments: dict = {}
        for key, value in kv.items():
            arguments[key] = value

        # Issue #32 Final Trust Repair: finish business parameters
        # (success_criterion, tests_passed, identification_verified) must
        # be explicitly provided by the model. No semantic defaults are
        # injected. Missing required fields cause Pydantic validation to
        # fail naturally via FinishArgs (which has no defaults for these).

        # Type coercion for known argument fields (strict — invalid values
        # raise ValueError and produce SentinelAction)
        try:
            arguments = self._coerce_arguments(action_type, arguments)
        except ValueError as e:
            diag.failure_class = "SCHEMA_VALIDATION_FAIL"
            return SentinelAction(reason=f"invalid scalar argument: {e}"), diag
        if arguments:
            data["arguments"] = arguments

        # Issue #32 Trust Repair: compute each dimension independently.
        # Unknown keys in arguments will cause arguments_valid=False and
        # schema_valid=False (via extra="forbid" on Args models).
        diag.safety_valid = self.check_safety_valid(data)
        diag.arguments_valid = self.check_arguments_valid(data)
        action = self.validate_action(data)
        diag.schema_valid = action is not None
        if action is not None:
            return action, diag

        diag.failure_class = self.classify_failure(data, format_parse_ok=True)
        return SentinelAction(reason="tag action failed schema validation"), diag

    @staticmethod
    def _coerce_arguments(action_type: str, arguments: dict) -> dict:
        """Coerce string values to correct types for known argument fields.

        Issue #32 Final Trust Repair: strict scalar parsing.
        - Boolean: only true/false/yes/no/1/0 (case-insensitive) accepted.
        - Integer: only valid integer strings accepted.
        - Float: only valid finite float strings accepted.
        Any invalid value raises ValueError, which the caller converts to
        SentinelAction with failure_class=SCHEMA_VALIDATION_FAIL.
        """
        coerced = dict(arguments)
        # Boolean coercion (strict — no silent False mapping)
        for bool_key in ("tests_passed", "identification_verified"):
            if bool_key in coerced:
                coerced[bool_key] = _parse_bool(coerced[bool_key])
        # Integer coercion (strict — no silent pass-through)
        for int_key in ("max_results", "start_line", "end_line"):
            if int_key in coerced:
                coerced[int_key] = _parse_int(coerced[int_key])
        # Float coercion (strict — rejects NaN/Infinity)
        for float_key in ("timeout_s",):
            if float_key in coerced:
                coerced[float_key] = _parse_float(coerced[float_key])
        return coerced
