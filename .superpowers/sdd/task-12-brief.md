## Task 12: Phase G — failed_patch_recovery augmentation generator

**Files:**
- Create: `scripts/augment_failed_patch_recovery.py`

**Interfaces:**
- Consumes: P4.0 `scripted.jsonl` (40 trajectories), `AgentEvaluator`, `_ListActionProvider` (inline), micro-tasks
- Produces: `data/p4-agent/trajectories-v1/failed-patch-recovery.jsonl` with ~80 trajectories — generated at runtime, NOT committed

**Concept:** Take each scripted trajectory, truncate it right after the first `apply_patch` step (simulating patch failure), then append a recovery sequence: `rollback_patch` → `propose_patch` (correct) → `apply_patch` → `run_tests` → `finish`. The recovery actions come from the original trajectory's later steps. 40 scripted × ~2 patchable steps = ~80 failed_patch_recovery trajectories.

**Global Constraints:**
1. ONLY create `scripts/augment_failed_patch_recovery.py`. Do NOT touch any source files in `src/`.
2. `src/agent_trajectory.py` is FROZEN — do not modify.
3. The script does NOT require GPU (uses _ListActionProvider for replay, no model inference). Only verify syntax via `ast.parse` in this environment. The actual run happens before PR merge.
4. `_ListActionProvider` goes INSIDE the script file (defined inline, same pattern as T10/T11).
5. No emojis, no incidental docstrings beyond what's in the brief code, no refactors.
6. Use `py -3.11` for all Python commands.

**Verified dependencies (pre-flight):**
- `load_trajectories(path)` — at src/agent_trajectory.py:188 (verified in T10)
- `Trajectory` has `.task_id` and `.steps` (each step has `.action`) — verified in T10
- `scripted.jsonl` exists at data/p4-agent/trajectories-v0/scripted.jsonl — verified in T10
- `Action` in src/agent_actions.py — verified
- `AgentEvaluator`, `ActionProvider`, `AgentState` in src/agent_evaluator.py — verified
- `MicroTaskWorkspace.from_task` exists — verified
- No SentinelAction import needed (T12 imports only `Action` from `src.agent_actions`)

- [ ] **Step 1: Write the augmentation script**

Create `scripts/augment_failed_patch_recovery.py` with EXACTLY this content:

```python
# scripts/augment_failed_patch_recovery.py
"""Phase G: failed_patch_recovery augmentation generator.

For each scripted trajectory, truncates right after the first apply_patch
step (simulating patch failure), then appends a recovery sequence:
rollback_patch → propose_patch (correct) → apply_patch → run_tests → finish.
The recovery actions come from the original trajectory's later steps.

Output: data/p4-agent/trajectories-v1/failed-patch-recovery.jsonl

Usage:
    py -3.11 scripts/augment_failed_patch_recovery.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("P4_ALLOW_NETWORK", "0")

from src.agent_trajectory import load_trajectories
from src.agent_evaluator import AgentEvaluator, ActionProvider, AgentState
from src.agent_actions import Action
from src.agent_workspace import MicroTaskWorkspace


class _ListActionProvider(ActionProvider):
    """Replays a list of Action objects. Yields them in order."""

    def __init__(self, actions: list):
        self._actions = list(actions)
        self._index = 0

    def next_action(self, state: AgentState):
        if self._index >= len(self._actions):
            raise StopIteration("no more actions in list")
        action = self._actions[self._index]
        self._index += 1
        return action


_SCRIPTED = _ROOT / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"
_TASKS_DIR = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"
_OUT = _ROOT / "data" / "p4-agent" / "trajectories-v1" / "failed-patch-recovery.jsonl"


def _build_recovery_sequence(traj, patch_step_idx):
    """Build a recovery sequence: truncate after apply_patch, then append
    rollback_patch → propose_patch → apply_patch → run_tests → finish,
    drawing from the original trajectory's later steps."""
    original_actions = [s.action for s in traj.steps]
    # Prefix: actions up to and including the apply_patch
    prefix = original_actions[:patch_step_idx + 1]
    # Recovery: find rollback, propose_patch, apply_patch, run_tests, finish
    # from the remaining original actions
    remaining = original_actions[patch_step_idx + 1:]
    recovery_types = {"rollback_patch", "propose_patch", "apply_patch",
                      "run_tests", "finish"}
    recovery = [a for a in remaining if a.action_type in recovery_types]
    return prefix + recovery


def main():
    scripted_trajs = load_trajectories(_SCRIPTED)
    print(f"Loaded {len(scripted_trajs)} scripted trajectories")

    results = []
    for traj in scripted_trajs:
        task_id = traj.task_id
        task_dir = _TASKS_DIR / task_id
        if not task_dir.exists():
            continue

        # Find all apply_patch steps
        patch_steps = [
            i for i, s in enumerate(traj.steps)
            if s.action.action_type == "apply_patch"
        ]

        for patch_idx in patch_steps:
            recovery_actions = _build_recovery_sequence(traj, patch_idx)
            ws = MicroTaskWorkspace.from_task(task_dir)
            try:
                provider = _ListActionProvider(recovery_actions)
                evaluator = AgentEvaluator(ws, provider, task_id, max_steps=20)
                result = evaluator.run()
                if result.success:
                    results.append({
                        "trajectory_id": f"failed_patch_{task_id}_s{patch_idx}",
                        "task_id": task_id,
                        "config": "failed_patch",
                        "source": "failed_patch_recovery",
                        "success": True,
                        "finish_claim_mismatch": result.finish_claim_mismatch,
                        "metrics": result.metrics,
                        "steps_executed": result.steps_executed,
                        "actions": [a.model_dump() for a in recovery_actions],
                        "step_diagnostics": [],
                    })
            except Exception:
                pass  # skip failed recoveries
            finally:
                ws.cleanup()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        for traj in results:
            f.write(json.dumps(traj) + "\n")
    print(f"Wrote {len(results)} failed_patch_recovery trajectories to {_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is importable**

Run: `py -3.11 -c "import ast; ast.parse(open('scripts/augment_failed_patch_recovery.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/augment_failed_patch_recovery.py
git commit -m "feat(p4-1): Phase G — failed_patch_recovery augmentation generator"
```

---
