"""P4.1 Phase E — ModelActionProvider: prompt builder, JSON extraction,
format-only repair, and structured diagnostics.

This module does NOT load the model (that's in the GPU tests / collection
script). It provides the building blocks that the ModelActionProvider class
composes.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, TypeAdapter

from src.agent_actions import Action, SafetyFlags
from src.agent_evaluator import AgentState, ActionProvider

if TYPE_CHECKING:
    from src.protocols.base import ProtocolBase, ProtocolDiagnostics

_ACTION_ADAPTER = TypeAdapter(Action)


class ModelStepDiagnostics(BaseModel):
    """Diagnostics recorded for each model.generate() call."""
    raw_output: str
    json_parse_ok: bool
    schema_valid: bool
    safety_valid: bool
    action_type_valid: bool
    arguments_valid: bool
    repair_attempted: bool
    repair_success: bool
    latency_ms: int


class SentinelAction(BaseModel):
    """Marker returned when the model output cannot be parsed into a valid
    Action. Not a real Action — the evaluator records it as action_invalid."""
    is_invalid: bool = True
    reason: str = ""

    @property
    def action_type(self) -> str:
        return "invalid"

    @property
    def safety_flags(self) -> SafetyFlags:
        return SafetyFlags(
            modifies_workspace=False,
            executes_code=False,
            network_required=False,
            reads_sensitive_path=False,
            is_terminal=False,
        )


def build_prompt(
    state: AgentState,
    task_description: str,
    last_observation: dict | None,
) -> str:
    """Build the prompt for model.generate()."""
    lines = [
        f"Task ID: {state.task_id}",
        f"Step: {state.step_count}",
        f"Task: {task_description}",
    ]
    if state.memory.notes:
        lines.append(f"Notes: {state.memory.notes}")
    if state.memory.hypothesis:
        lines.append(f"Hypothesis: {state.memory.hypothesis}")
    if last_observation:
        lines.append(f"Last observation: {last_observation}")
    lines.append(
        "Choose ONE action from: list_files, read_file, search_text, "
        "inspect_task, propose_patch, apply_patch, rollback_patch, run_tests, "
        "inspect_error, write_memory, finish."
    )
    lines.append("Respond with a single JSON object.")
    return "\n".join(lines)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


def extract_json(raw: str) -> str | None:
    """Extract the first JSON object from raw model output. Returns the
    JSON string or None if no JSON found."""
    m = _JSON_FENCE_RE.search(raw)
    if m:
        return m.group(1)
    m = _BARE_JSON_RE.search(raw)
    if m:
        return m.group(1)
    return None


def repair_json(raw: str) -> str:
    """Format-only repair. NEVER alters action semantics (action_type,
    arguments values). Only fixes: markdown fences, trailing commas,
    unbalanced braces (best-effort)."""
    result = raw
    # Strip markdown fences
    result = re.sub(r"```(?:json)?\s*", "", result)
    result = result.replace("```", "")
    # Remove trailing commas before } or ]
    result = re.sub(r",\s*([}\]])", r"\1", result)
    # Best-effort brace balancing (append missing closing braces)
    opens = result.count("{")
    closes = result.count("}")
    if opens > closes:
        result = result + ("}" * (opens - closes))
    return result.strip()


class ModelActionProvider(ActionProvider):
    """Loads Qwen3-0.6B and generates actions. GPU required.

    When protocol is None: uses legacy JSON parsing (backward-compatible).
    When protocol is set: delegates prompt building and output parsing
    to the protocol object.
    """

    def __init__(
        self,
        model_path: str = "models/Qwen3-0.6B",
        adapter_path: str | None = None,
        max_new_tokens: int = 256,
        protocol: "ProtocolBase | None" = None,
    ):
        self._model_path = model_path
        self._adapter_path = adapter_path
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._protocol = protocol
        self._diagnostics: list[ModelStepDiagnostics] = []
        self._protocol_diagnostics: list["ProtocolDiagnostics"] = []

    def _load_model(self):
        """Lazy-load the model. Called on first next_action."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            dtype=torch.float16,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
        )
        if self._adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
            self._model = self._model.merge_and_unload()
        self._model.config.use_cache = True
        self._model.eval()

    def _generate(self, prompt: str) -> str:
        """Generate raw text from the model. Override in tests."""
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    def _build_user_prompt(self, state: AgentState) -> str:
        """Build user prompt for protocol path (state info only)."""
        lines = [
            f"Task ID: {state.task_id}",
            f"Step: {state.step_count}",
        ]
        if state.memory.notes:
            lines.append(f"Notes: {state.memory.notes}")
        if state.memory.hypothesis:
            lines.append(f"Hypothesis: {state.memory.hypothesis}")
        lines.append("What is your next action?")
        return "\n".join(lines)

    def next_action(self, state: AgentState) -> Action | SentinelAction:
        if self._model is None:
            self._load_model()

        if self._protocol is not None:
            return self._next_action_protocol(state)
        return self._next_action_legacy(state)

    def _next_action_protocol(self, state: AgentState) -> Action | SentinelAction:
        """Protocol path: delegate to protocol for prompt and parsing."""
        system_prompt = self._protocol.build_system_prompt(state.task_id)
        user_prompt = self._build_user_prompt(state)
        prompt = system_prompt + "\n\n" + user_prompt
        raw_output = self._generate(prompt)
        action, diag = self._protocol.parse_output(raw_output)
        self._protocol_diagnostics.append(diag)
        return action

    def _next_action_legacy(self, state: AgentState) -> Action | SentinelAction:
        """Legacy path: existing JSON parsing (backward-compatible).

        Issue #32 Trust Repair: each diagnostic dimension is computed
        independently via ProtocolBase.check_* methods. Previously all
        four were set to True together when validate_action succeeded.
        """
        from src.protocols.base import ProtocolBase

        prompt = build_prompt(state, task_description="", last_observation=None)
        t0 = time.monotonic()
        raw_output = self._generate(prompt)
        latency_ms = int((time.monotonic() - t0) * 1000)

        json_str = extract_json(raw_output)
        diag = ModelStepDiagnostics(
            raw_output=raw_output,
            json_parse_ok=json_str is not None,
            schema_valid=False,
            safety_valid=False,
            action_type_valid=False,
            arguments_valid=False,
            repair_attempted=False,
            repair_success=False,
            latency_ms=latency_ms,
        )

        if json_str is None:
            self._diagnostics.append(diag)
            return SentinelAction(reason="json parse failed")

        # Try direct validation
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                # Issue #32: independent dimension checks
                diag.action_type_valid = ProtocolBase.check_action_type_valid(data)
                diag.safety_valid = ProtocolBase.check_safety_valid(data)
                diag.arguments_valid = ProtocolBase.check_arguments_valid(data)
                action = _validate_action(data)
                diag.schema_valid = action is not None
                if action is not None:
                    self._diagnostics.append(diag)
                    return action
        except (json.JSONDecodeError, Exception):
            pass

        # Attempt repair
        diag.repair_attempted = True
        repaired = repair_json(json_str)
        try:
            data = json.loads(repaired)
            if isinstance(data, dict):
                diag.action_type_valid = ProtocolBase.check_action_type_valid(data)
                diag.safety_valid = ProtocolBase.check_safety_valid(data)
                diag.arguments_valid = ProtocolBase.check_arguments_valid(data)
                action = _validate_action(data)
                diag.schema_valid = action is not None
                if action is not None:
                    diag.repair_success = True
                    self._diagnostics.append(diag)
                    return action  # FIX: was missing return (P4.1 bug)
        except (json.JSONDecodeError, Exception):
            pass

        self._diagnostics.append(diag)
        return SentinelAction(reason="schema validation failed after repair")

    def reset(self) -> None:
        self._diagnostics.clear()
        self._protocol_diagnostics.clear()

    @property
    def diagnostics(self):
        """Returns ProtocolDiagnostics if protocol set, else ModelStepDiagnostics."""
        if self._protocol is not None:
            return list(self._protocol_diagnostics)
        return list(self._diagnostics)


def _validate_action(data: dict) -> Action | None:
    """Validate a dict against the Action union. Returns the Action or None."""
    try:
        return _ACTION_ADAPTER.validate_python(data)
    except Exception:
        return None
