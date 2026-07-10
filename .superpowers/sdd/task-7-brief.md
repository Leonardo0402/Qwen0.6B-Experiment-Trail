## Task 7: Phase E — ModelActionProvider mocked generation + diagnostics tests

**Files:**
- Modify: `tests/test_agent_model_provider.py` (+3 tests for mocked generation flow)

**Interfaces:**
- Consumes: `ModelActionProvider`, `ModelStepDiagnostics`, `SentinelAction` from Task 6

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_model_provider.py`:

```python
from unittest.mock import patch, MagicMock
from src.agent_model_provider import ModelActionProvider, SentinelAction
from src.agent_actions import ListFilesAction


def test_model_provider_extracts_valid_json_mocked():
    """Mock _generate to return valid Action JSON → provider returns a
    valid Action, diagnostics record schema_valid=True."""
    provider = ModelActionProvider.__new__(ModelActionProvider)
    provider._model = MagicMock()  # skip _load_model
    provider._tokenizer = MagicMock()
    provider._max_new_tokens = 512
    provider._diagnostics = []
    provider._model_path = "fake"
    provider._adapter_path = None

    valid_json = '{"action_type": "list_files", "action_id": "a1", "reason_short": "list", "expected_observation": "files", "safety_flags": {"modifies_workspace": false, "executes_code": false, "network_required": false, "reads_sensitive_path": false, "is_terminal": false}}'

    with patch.object(provider, "_generate", return_value=valid_json):
        state = AgentState(
            memory=AgentMemory(), step_count=0,
            task_id="t1", workspace_id="w1",
        )
        action = provider.next_action(state)

    assert not isinstance(action, SentinelAction), "expected valid Action, got SentinelAction"
    assert provider.diagnostics[0].json_parse_ok
    assert provider.diagnostics[0].schema_valid


def test_model_provider_records_diagnostics_on_invalid_json():
    """Mock _generate to return garbage → provider returns SentinelAction,
    diagnostics record json_parse_ok=False."""
    provider = ModelActionProvider.__new__(ModelActionProvider)
    provider._model = MagicMock()
    provider._tokenizer = MagicMock()
    provider._max_new_tokens = 512
    provider._diagnostics = []
    provider._model_path = "fake"
    provider._adapter_path = None

    with patch.object(provider, "_generate", return_value="I cannot do that."):
        state = AgentState(
            memory=AgentMemory(), step_count=0,
            task_id="t1", workspace_id="w1",
        )
        action = provider.next_action(state)

    assert isinstance(action, SentinelAction)
    assert not provider.diagnostics[0].json_parse_ok
    assert not provider.diagnostics[0].schema_valid


def test_model_provider_repair_strips_fences_then_validates():
    """Mock _generate to return fenced JSON → repair strips fences,
    validation succeeds."""
    provider = ModelActionProvider.__new__(ModelActionProvider)
    provider._model = MagicMock()
    provider._tokenizer = MagicMock()
    provider._max_new_tokens = 512
    provider._diagnostics = []
    provider._model_path = "fake"
    provider._adapter_path = None

    fenced = '```json\n{"action_type": "list_files", "action_id": "a1", "reason_short": "list", "expected_observation": "files", "safety_flags": {"modifies_workspace": false, "executes_code": false, "network_required": false, "reads_sensitive_path": false, "is_terminal": false}}\n```'

    with patch.object(provider, "_generate", return_value=fenced):
        state = AgentState(
            memory=AgentMemory(), step_count=0,
            task_id="t1", workspace_id="w1",
        )
        action = provider.next_action(state)

    # extract_json should handle fences, but if not, repair kicks in
    assert not isinstance(action, SentinelAction), \
        f"expected valid Action after repair, got SentinelAction; diag: {provider.diagnostics[0]}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.11 -m pytest tests/test_agent_model_provider.py::test_model_provider_extracts_valid_json_mocked tests/test_agent_model_provider.py::test_model_provider_records_diagnostics_on_invalid_json tests/test_agent_model_provider.py::test_model_provider_repair_strips_fences_then_validates -v -p no:warnings`
Expected: May partially pass (Task 6 implemented the logic), but verify all 3 pass. If any fail, the mock setup or Action validation needs adjustment.

- [ ] **Step 3: Fix any issues (minimal)**

If the Action union validation fails on the test JSON, check that the field names match `agent_actions.py` exactly. The `ListFilesAction` should not require `arguments` — verify the schema.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_agent_model_provider.py -v -p no:warnings`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_agent_model_provider.py
git commit -m "feat(p4-1): Phase E — ModelActionProvider mocked generation + diagnostics tests"
```

---

