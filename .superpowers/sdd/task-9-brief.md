## Task 9: Phase F — Model trajectory collection script (RecordingProvider + action-list JSONL)

**Files:**
- Create: `scripts/collect_model_trajectories.py`

**Interfaces:**
- Consumes: `ModelActionProvider` (Task 6, commit 37c4ef2 + T7 fix 2f244cd), `AgentEvaluator` / `ActionProvider` / `AgentState` (P4.0, src/agent_evaluator.py), 40 micro-tasks from `data/p4-agent/micro-tasks-v0/`
- Produces: trajectory JSONL files at `data/p4-agent/trajectories-v1/model-base.jsonl` and `data/p4-agent/trajectories-v1/model-repair-lora.jsonl`, plus collection report at `reports/p4/model-trajectory-collection-report.json` (generated at runtime on GPU — NOT committed)

**Design note:** Trajectories are written as JSONL with an `actions` field (list of action dicts via `action.model_dump()`), NOT as P4.0 `Trajectory` objects. The P4.0 `Trajectory` schema requires `TrajectoryStep` objects with complex fields and a restrictive `source` Literal — too complex to construct from evaluator runtime data, and `src/agent_trajectory.py` is FROZEN. Replay (T13) uses `_ListActionProvider(actions)`, NOT `ReplayActionProvider(Trajectory)`.

**Global Constraints:**
1. ONLY create `scripts/collect_model_trajectories.py`. Do NOT touch any source files in `src/`.
2. `src/agent_trajectory.py` is FROZEN — do not modify.
3. The script requires GPU to actually run. In this environment (no GPU), only verify syntax via `ast.parse`. The actual trajectory collection runs on the RTX 3050 before PR merge.
4. RecordingProvider goes INSIDE the script file, NOT in a separate `src/recording_provider.py` module.
5. No emojis, no incidental docstrings beyond what's in the brief code, no refactors.
6. Use `py -3.11` for all Python commands.

**Verified dependencies (pre-flight):**
- `ModelActionProvider._load_model()` — exists at src/agent_model_provider.py:137, takes no args (besides self), uses torch.float16 + device_map={"": "cuda:0"}
- `ModelActionProvider.reset()` — exists at src/agent_model_provider.py:235
- `ModelActionProvider.next_action(state)` — exists at src/agent_model_provider.py:178, returns Action | SentinelAction
- `ActionProvider`, `AgentState`, `AgentEvaluator` — all in src/agent_evaluator.py (ActionProvider at line 80, AgentState at line 47, AgentEvaluator at line 228)
- `manifest.json` — at data/p4-agent/micro-tasks-v0/manifest.json, has `tasks` array with `task_id` fields
- `MicroTaskWorkspace.from_task(task_dir)` — used in T8 tests, exists

- [ ] **Step 1: Write the collection script**

Create `scripts/collect_model_trajectories.py` with EXACTLY this content:

```python
# scripts/collect_model_trajectories.py
"""Phase F: collect model trajectories on the 40 micro-tasks.

Runs ModelActionProvider (base + repair-lora configs) through the
AgentEvaluator on all 40 tasks. Uses a RecordingProvider wrapper to capture
each action returned by the model, then writes trajectories as JSONL with
an `actions` list field (for replay via _ListActionProvider in T13).

Usage:
    py -3.11 scripts/collect_model_trajectories.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("P4_ALLOW_NETWORK", "0")

from src.agent_model_provider import ModelActionProvider, SentinelAction
from src.agent_evaluator import AgentEvaluator, ActionProvider, AgentState
from src.agent_workspace import MicroTaskWorkspace


class RecordingProvider(ActionProvider):
    """Wraps an inner ActionProvider and records each action returned (as a
    dict via action.model_dump()) for later replay. SentinelActions are
    recorded with a `__sentinel__` marker so they can be reconstructed."""

    def __init__(self, inner: ActionProvider):
        self._inner = inner
        self._recorded: list[dict] = []

    def next_action(self, state: AgentState):
        action = self._inner.next_action(state)
        if isinstance(action, SentinelAction):
            self._recorded.append({
                "__sentinel__": True,
                "is_invalid": True,
                "reason": action.reason,
            })
        else:
            self._recorded.append(action.model_dump())
        return action

    @property
    def recorded_actions(self) -> list[dict]:
        return list(self._recorded)

    @property
    def diagnostics(self):
        return self._inner.diagnostics if hasattr(self._inner, 'diagnostics') else []

    def reset(self) -> None:
        self._recorded.clear()
        if hasattr(self._inner, 'reset'):
            self._inner.reset()


_CONFIGS = [
    {"name": "base", "model_path": "models/Qwen3-0.6B", "adapter_path": None},
    {"name": "repair-lora", "model_path": "models/Qwen3-0.6B",
     "adapter_path": "adapters/p3/repair-limited"},
]

_TASKS_DIR = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"
_TRAJ_DIR = _ROOT / "data" / "p4-agent" / "trajectories-v1"
_REPORT = _ROOT / "reports" / "p4" / "model-trajectory-collection-report.json"
MAX_STEPS = 12


def _load_task_ids():
    manifest = json.loads((_TASKS_DIR / "manifest.json").read_text(encoding="utf-8"))
    return [t["task_id"] for t in manifest["tasks"]]


def _run_config(config, task_ids):
    trajectories = []
    crashes = 0
    model_load_ok = False
    adapter_load_ok = config["adapter_path"] is not None

    inner_provider = ModelActionProvider(
        model_path=config["model_path"],
        adapter_path=config["adapter_path"],
    )
    # Try to load the model once
    try:
        inner_provider._load_model()
        model_load_ok = True
    except Exception as e:
        print(f"[{config['name']}] model load failed: {e}")
        return {
            "config": config["name"],
            "total_tasks": len(task_ids),
            "trajectories_written": 0,
            "model_load_ok": False,
            "adapter_load_ok": adapter_load_ok,
            "generation_ok": False,
            "crashes": len(task_ids),
            "aggregate_metrics": {},
            "trajectories": [],
        }

    for i, task_id in enumerate(task_ids):
        task_dir = _TASKS_DIR / task_id
        ws = MicroTaskWorkspace.from_task(task_dir)
        try:
            inner_provider.reset()
            provider = RecordingProvider(inner_provider)
            evaluator = AgentEvaluator(ws, provider, task_id, max_steps=MAX_STEPS)
            result = evaluator.run()
            source = "model_self_run_success" if result.success else "model_self_run_failure"
            trajectories.append({
                "trajectory_id": f"{config['name']}_{task_id}",
                "task_id": task_id,
                "config": config["name"],
                "source": source,
                "success": result.success,
                "finish_claim_mismatch": result.finish_claim_mismatch,
                "metrics": result.metrics,
                "steps_executed": result.steps_executed,
                "actions": provider.recorded_actions,
                "step_diagnostics": [d.model_dump() for d in inner_provider.diagnostics],
            })
        except Exception:
            crashes += 1
            traceback.print_exc()
        finally:
            ws.cleanup()
        print(f"\r[{config['name']}] {i+1}/{len(task_ids)} {task_id}", end="", flush=True)
    print()

    return {
        "config": config["name"],
        "total_tasks": len(task_ids),
        "trajectories_written": len(trajectories),
        "model_load_ok": model_load_ok,
        "adapter_load_ok": adapter_load_ok,
        "generation_ok": len(trajectories) > 0,
        "crashes": crashes,
        "trajectories": trajectories,
    }


def main():
    _TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    task_ids = _load_task_ids()
    reports = []
    for config in _CONFIGS:
        print(f"\n=== Config: {config['name']} ===")
        report = _run_config(config, task_ids)
        # Write trajectories JSONL
        out_file = _TRAJ_DIR / f"model-{config['name']}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for traj in report["trajectories"]:
                f.write(json.dumps(traj) + "\n")
        # Strip trajectories from report (keep only summary)
        summary = {k: v for k, v in report.items() if k != "trajectories"}
        reports.append(summary)

    _REPORT.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nWrote {_REPORT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is importable (no syntax errors)**

Run: `py -3.11 -c "import ast; ast.parse(open('scripts/collect_model_trajectories.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit (script runs on GPU before PR merge)**

```bash
git add scripts/collect_model_trajectories.py
git commit -m "feat(p4-1): Phase F — model trajectory collection script (RecordingProvider + action-list JSONL)"
```

---
