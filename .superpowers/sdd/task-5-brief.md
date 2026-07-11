### Task 5: ModelActionProvider Protocol Adapter

**Files:**
- Modify: `src/agent_model_provider.py` (add `protocol` param, protocol delegation, diagnostics property)
- Modify: `tests/test_agent_model_provider.py` (update 3 existing tests that use `__new__` to set new attributes)
- Test: `tests/test_agent_model_provider_protocol.py` (new, 6 tests)

**Interfaces:**
- Consumes: `ProtocolBase`, `ProtocolDiagnostics` from Task 1; `JsonProtocol` from Task 2; `TagProtocol` from Task 3
- Produces: `ModelActionProvider(protocol=...)` — when protocol is set, delegates prompt/parsing to protocol; when `None`, uses existing code path (backward-compatible)

---

**IMPORTANT FIX (not in original plan):** The existing tests in `tests/test_agent_model_provider.py` use `ModelActionProvider.__new__()` to bypass `__init__` and manually set attributes. After T5 adds `_protocol` and `_protocol_diagnostics`, these tests will fail with `AttributeError` because `next_action` checks `self._protocol`. You MUST update these 3 tests to add the new attributes.

---

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_model_provider_protocol.py
"""Tests for ModelActionProvider protocol adapter.

Verifies that:
1. protocol=None preserves existing behavior (backward compat)
2. protocol set delegates to protocol.build_system_prompt and parse_output
3. ProtocolDiagnostics are recorded when protocol is used
4. Existing ModelStepDiagnostics are recorded when protocol=None
"""
import pytest
from unittest.mock import patch, MagicMock
from src.agent_model_provider import ModelActionProvider, SentinelAction
from src.agent_evaluator import AgentState
from src.agent_state import AgentMemory
from src.protocols import JsonProtocol, TagProtocol
from src.protocols.base import ProtocolDiagnostics


def _make_state():
    return AgentState(
        memory=AgentMemory(),
        step_count=0,
        task_id="task_001",
        workspace_id="test_ws",
    )


def test_protocol_none_uses_legacy_path():
    """When protocol=None, existing code path is used (ModelStepDiagnostics)."""
    provider = ModelActionProvider(model_path="fake", protocol=None)
    # Mock _generate to return invalid JSON
    with patch.object(provider, "_generate", return_value="not json at all"):
        provider._model = MagicMock()  # skip model loading
        action = provider.next_action(_make_state())
        assert isinstance(action, SentinelAction)
        # Legacy path stores ModelStepDiagnostics
        assert len(provider.diagnostics) == 1
        # ModelStepDiagnostics has json_parse_ok, not format_parse_ok
        d = provider.diagnostics[0].model_dump()
        assert "json_parse_ok" in d


def test_protocol_set_uses_protocol_path():
    """When protocol is set, protocol.parse_output is used."""
    proto = JsonProtocol()
    provider = ModelActionProvider(model_path="fake", protocol=proto)
    # Mock _generate to return empty (will fail parse)
    with patch.object(provider, "_generate", return_value=""):
        provider._model = MagicMock()
        action = provider.next_action(_make_state())
        assert isinstance(action, SentinelAction)
        # Protocol path stores ProtocolDiagnostics
        assert len(provider.diagnostics) == 1
        d = provider.diagnostics[0].model_dump()
        assert "format_parse_ok" in d
        assert "json_parse_ok" not in d


def test_protocol_set_builds_system_prompt():
    """When protocol is set, build_system_prompt is called with task context."""
    proto = JsonProtocol()
    provider = ModelActionProvider(model_path="fake", protocol=proto)
    captured_prompt = []

    def fake_generate(prompt):
        captured_prompt.append(prompt)
        return ""

    with patch.object(provider, "_generate", side_effect=fake_generate):
        provider._model = MagicMock()
        provider.next_action(_make_state())
        assert len(captured_prompt) == 1
        # System prompt should contain format instructions from protocol
        assert "JSON" in captured_prompt[0] or "action_type" in captured_prompt[0]
        # Should contain task context
        assert "task_001" in captured_prompt[0]


def test_protocol_set_records_protocol_diagnostics():
    """ProtocolDiagnostics are recorded with independent dimensions."""
    proto = TagProtocol()
    provider = ModelActionProvider(model_path="fake", protocol=proto)
    with patch.object(provider, "_generate", return_value="<action>\ntool: read_file\npath: solution.py\n</action>"):
        provider._model = MagicMock()
        action = provider.next_action(_make_state())
        assert not isinstance(action, SentinelAction)
        assert len(provider.diagnostics) == 1
        diag = provider.diagnostics[0]
        assert isinstance(diag, ProtocolDiagnostics)
        assert diag.format_parse_ok
        assert diag.schema_valid
        assert diag.action_type_valid


def test_reset_clears_protocol_diagnostics():
    proto = JsonProtocol()
    provider = ModelActionProvider(model_path="fake", protocol=proto)
    with patch.object(provider, "_generate", return_value=""):
        provider._model = MagicMock()
        provider.next_action(_make_state())
        assert len(provider.diagnostics) == 1
    provider.reset()
    assert len(provider.diagnostics) == 0


def test_existing_tests_still_pass_with_protocol_none():
    """Smoke check: protocol=None doesn't break existing ModelActionProvider."""
    provider = ModelActionProvider(model_path="fake")
    assert provider._protocol is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_agent_model_provider_protocol.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'protocol'`

- [ ] **Step 3: Update existing tests in `tests/test_agent_model_provider.py`**

The 3 tests that use `ModelActionProvider.__new__()` need `_protocol` and `_protocol_diagnostics` attributes added. For each of these 3 tests, add two lines after `provider._diagnostics = []`:

```python
    provider._protocol = None
    provider._protocol_diagnostics = []
```

The 3 tests to update are:
1. `test_model_provider_extracts_valid_json_mocked` (after line `provider._diagnostics = []`)
2. `test_model_provider_records_diagnostics_on_invalid_json` (after line `provider._diagnostics = []`)
3. `test_model_provider_repair_strips_fences_then_validates` (after line `provider._diagnostics = []`)

- [ ] **Step 4: Modify `src/agent_model_provider.py`**

The changes are:

1. Add `TYPE_CHECKING` import for protocol types (after existing imports, ~line 13)
2. Add `protocol` parameter to `__init__` (line 124-135)
3. Add `_build_user_prompt` method (before `next_action`)
4. Split `next_action` into dispatch + `_next_action_protocol` + `_next_action_legacy`
5. Update `reset` and `diagnostics` property

```python
# At top, after existing imports (line 13, after "from typing import Any"), add:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.protocols.base import ProtocolBase, ProtocolDiagnostics


# Modified __init__ (replace lines 124-135):
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
        max_new_tokens: int = 128,
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


# Add new method before next_action (after _generate, ~line 177):
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


# Replace next_action (lines 178-233) with dispatch + two methods:
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
        """Legacy path: existing JSON parsing (backward-compatible)."""
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
            action = _validate_action(data)
            if action is not None:
                diag.schema_valid = True
                diag.safety_valid = True
                diag.action_type_valid = True
                diag.arguments_valid = True
                self._diagnostics.append(diag)
                return action
        except (json.JSONDecodeError, Exception):
            pass

        # Attempt repair
        diag.repair_attempted = True
        repaired = repair_json(json_str)
        try:
            data = json.loads(repaired)
            action = _validate_action(data)
            if action is not None:
                diag.repair_success = True
                diag.schema_valid = True
                diag.safety_valid = True
                diag.action_type_valid = True
                diag.arguments_valid = True
                self._diagnostics.append(diag)
                return action  # FIX: was missing return (P4.1 bug)
        except (json.JSONDecodeError, Exception):
            pass

        self._diagnostics.append(diag)
        return SentinelAction(reason="schema validation failed after repair")


# Replace reset and diagnostics (lines 235-240):
    def reset(self) -> None:
        self._diagnostics.clear()
        self._protocol_diagnostics.clear()

    @property
    def diagnostics(self):
        """Returns ProtocolDiagnostics if protocol set, else ModelStepDiagnostics."""
        if self._protocol is not None:
            return list(self._protocol_diagnostics)
        return list(self._diagnostics)
```

- [ ] **Step 5: Run new tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_agent_model_provider_protocol.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Run existing tests to verify no regression**

Run: `py -3.11 -m pytest tests/test_agent_model_provider.py -v`
Expected: PASS (all existing tests, including the 3 updated ones)

- [ ] **Step 7: Run broader test suite (non-GPU)**

Run: `py -3.11 -m pytest tests/ -p no:warnings --tb=no -q -m "not gpu" --timeout=120 --ignore=tests/test_data_pipeline.py --ignore=tests/test_p3_readiness_gate.py`
Expected: PASS (all non-GPU tests)

- [ ] **Step 8: Commit**

```bash
git add src/agent_model_provider.py tests/test_agent_model_provider_protocol.py tests/test_agent_model_provider.py
git commit -m "feat(agent-model): add protocol adapter to ModelActionProvider (P4.1b T5)

- protocol=None: legacy JSON path (backward-compatible)
- protocol set: delegates to protocol.build_system_prompt and parse_output
- Fixes P4.1 repair-path bug (validated action was not returned)
- ProtocolDiagnostics computed independently per dimension
- Updates 3 existing tests to set _protocol and _protocol_diagnostics attributes"
```
