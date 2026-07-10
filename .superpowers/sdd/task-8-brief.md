## Task 8: Phase E — GPU smoke tests (base + repair-lora)

**Files:**
- Create: `tests/test_agent_model_provider_gpu.py`

**Interfaces:**
- Consumes: `ModelActionProvider` from Task 6, `MicroTaskWorkspace.from_task`, micro-tasks from `data/p4-agent/micro-tasks-v0/`

- [ ] **Step 1: Write the GPU smoke tests**

```python
# tests/test_agent_model_provider_gpu.py
"""GPU smoke tests for ModelActionProvider.

Marked @pytest.mark.gpu — skipped in CI (CI uses -m "not gpu").
Run manually on the RTX 3050 before PR merge.
"""
import pytest
from pathlib import Path

pytestmark = pytest.mark.gpu

_ROOT = Path(__file__).resolve().parent.parent
_TASKS_DIR = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"


def test_model_provider_smoke_base():
    """Load base Qwen3-0.6B, run 1 micro-task, assert:
    - no runtime crash
    - forbidden_action_count == 0
    - at least 1 schema-valid action OR structured diagnostics recorded

    Note: invalid_action_count > 0 is acceptable (model may produce invalid
    JSON → SentinelAction → invalid_action_count). forbidden_action_count == 0
    is still required (no unknown action types should slip through).
    """
    from src.agent_model_provider import ModelActionProvider, SentinelAction
    from src.agent_evaluator import AgentEvaluator, AgentState
    from src.agent_workspace import MicroTaskWorkspace
    import os
    os.environ.setdefault("P4_ALLOW_NETWORK", "0")

    task_dir = _TASKS_DIR / "task_001"
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        provider = ModelActionProvider(
            model_path="models/Qwen3-0.6B",
            adapter_path=None,
        )
        evaluator = AgentEvaluator(ws, provider, "task_001", max_steps=12)
        result = evaluator.run()

        # Minimum bar (user decision #3)
        # forbidden_action_count must be 0 — unknown action types are not
        # acceptable. invalid_action_count > 0 is OK (model may emit bad JSON).
        assert result.metrics.get("forbidden_action_count", 0) == 0, \
            f"forbidden_action_count must be 0, got {result.metrics.get('forbidden_action_count')}"
        # At least one diagnostic recorded (even if all invalid)
        assert len(provider.diagnostics) > 0, "no diagnostics recorded"
    finally:
        ws.cleanup()


def test_model_provider_smoke_repair_lora():
    """Load Qwen3-0.6B + Repair-Limited LoRA, run 1 micro-task, same bar.

    Note: invalid_action_count > 0 is acceptable (model may produce invalid
    JSON). forbidden_action_count == 0 is still required.
    """
    from src.agent_model_provider import ModelActionProvider
    from src.agent_evaluator import AgentEvaluator
    from src.agent_workspace import MicroTaskWorkspace
    import os
    os.environ.setdefault("P4_ALLOW_NETWORK", "0")

    task_dir = _TASKS_DIR / "task_001"
    ws = MicroTaskWorkspace.from_task(task_dir)
    try:
        provider = ModelActionProvider(
            model_path="models/Qwen3-0.6B",
            adapter_path="adapters/p3/repair-limited",
        )
        evaluator = AgentEvaluator(ws, provider, "task_001", max_steps=12)
        result = evaluator.run()

        assert result.metrics.get("forbidden_action_count", 0) == 0
        assert len(provider.diagnostics) > 0
    finally:
        ws.cleanup()
```

- [ ] **Step 2: Verify tests are collected (will skip without GPU)**

Run: `py -3.11 -m pytest tests/test_agent_model_provider_gpu.py --collect-only -p no:warnings`
Expected: 2 tests collected.

Run: `py -3.11 -m pytest tests/test_agent_model_provider_gpu.py -v -p no:warnings -m "not gpu"`
Expected: 2 SKIPPED (no GPU marker active).

- [ ] **Step 3: Commit (tests will run on GPU before PR merge)**

```bash
git add tests/test_agent_model_provider_gpu.py
git commit -m "feat(p4-1): Phase E — GPU smoke tests (base + repair-lora)"
```

---

