"""Protocol A — Action JSON (baseline).

Extracts existing parsing logic from ModelActionProvider into a protocol
class. Computes diagnostics dimensions independently (fixes P4.1 issue
where schema_valid/safety_valid/action_type_valid/arguments_valid were
all set to True together).

Also fixes the repair-path bug (P4.1 agent_model_provider.py lines 220-230):
when repair succeeds, the validated action is now returned instead of
falling through to SentinelAction.
"""
from __future__ import annotations

import json
import time

from src.agent_actions import Action
from src.agent_model_provider import SentinelAction, extract_json, repair_json
from src.protocols.base import ProtocolBase, ProtocolDiagnostics

_TOOL_LIST = (
    "list_files, read_file, search_text, inspect_task, "
    "propose_patch, apply_patch, rollback_patch, run_tests, "
    "inspect_error, write_memory, finish"
)


class JsonProtocol(ProtocolBase):
    """Baseline JSON protocol — same format as P4.1 ModelActionProvider."""

    @property
    def name(self) -> str:
        return "json"

    def build_system_prompt(self, task_context: str) -> str:
        return "\n".join([
            f"Task: {task_context}",
            "",
            f"Actions: {_TOOL_LIST}",
            "",
            'Example: {"action_type":"read_file","action_id":"a1","reason_short":"inspect file","expected_observation":"file contents","safety_flags":{"modifies_workspace":false,"executes_code":false,"network_required":false,"reads_sensitive_path":false,"is_terminal":false},"arguments":{"path":"solution.py"}}',
            "",
            "Output ONLY ONE JSON object like the example. No preamble, no markdown, no explanation.",
        ])

    def parse_output(self, raw: str) -> tuple[Action | SentinelAction, ProtocolDiagnostics]:
        t0 = time.monotonic()
        json_str = extract_json(raw)
        latency_ms = int((time.monotonic() - t0) * 1000)

        format_parse_ok = json_str is not None

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
            return SentinelAction(reason="json parse failed"), diag

        # Parse JSON and check action_type independently.
        # On JSONDecodeError, fall through to repair (do NOT return early —
        # the original P4.1 code used `except: pass` for this reason).
        data = None
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, Exception):
            pass  # Fall through to repair path

        # Inject structural defaults for missing base fields (consistent
        # with Tag/DSL protocols). Only structural metadata — never
        # business parameters (success_criterion, tests_passed, etc.).
        if data is not None and isinstance(data, dict):
            at = data.get("action_type", "")
            data.setdefault("action_id", f"json_{at}")
            data.setdefault("reason_short", "no reason provided")
            data.setdefault("expected_observation", "unknown")

        # Try direct validation (only if json.loads succeeded)
        if data is not None and isinstance(data, dict):
            # Issue #32 Trust Repair: compute each dimension independently.
            # Previously all four were set to True together when validate_action
            # succeeded, and all False when it failed. Unknown fields were
            # silently dropped by Pydantic's default extra="ignore".
            diag.action_type_valid = self.check_action_type_valid(data)
            diag.safety_valid = self.check_safety_valid(data)
            diag.arguments_valid = self.check_arguments_valid(data)
            action = self.validate_action(data)
            diag.schema_valid = action is not None
            if action is not None:
                return action, diag

        # Attempt format-only repair (syntax only, never changes semantics)
        diag.repair_attempted = True
        repaired = repair_json(json_str)
        try:
            data = json.loads(repaired)
        except (json.JSONDecodeError, Exception):
            data = None

        if data is not None and isinstance(data, dict):
            # Inject structural defaults (same as primary path)
            at = data.get("action_type", "")
            data.setdefault("action_id", f"json_{at}")
            data.setdefault("reason_short", "no reason provided")
            data.setdefault("expected_observation", "unknown")
            # Recompute dimensions on repaired data
            diag.action_type_valid = self.check_action_type_valid(data)
            diag.safety_valid = self.check_safety_valid(data)
            diag.arguments_valid = self.check_arguments_valid(data)
            action = self.validate_action(data)
            diag.schema_valid = action is not None
            if action is not None:
                # FIX: P4.1 bug — was missing `return action` here
                diag.repair_success = True
                return action, diag

        # Classify failure using independent dimension checks
        if data is None or not isinstance(data, dict):
            diag.failure_class = "FORMAT_PARSE_FAIL"
        else:
            diag.failure_class = self.classify_failure(data, format_parse_ok=True)
        return SentinelAction(reason="schema validation failed after repair"), diag
