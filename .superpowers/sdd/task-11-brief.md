## Task 11: Phase G — corrupted_recovered augmentation generator

**Files:**
- Create: `scripts/augment_corrupted_recovered.py`

**Interfaces:**
- Consumes: P4.0 `scripted.jsonl` (40 trajectories), `CorruptedActionProvider`, `Corruption`, `CorruptionType` from `src/agent_evaluator.py`, micro-tasks
- Produces: `data/p4-agent/trajectories-v1/corrupted-recovered.jsonl` with ~600+ trajectories — generated at runtime, NOT committed

**Concept:** Take each scripted trajectory, apply each of the 5 `CorruptionType` values at each patchable step (and at steps 1, 2, 3 for step-index variants), run the evaluator with `CorruptedActionProvider`. If the trajectory still reaches success despite the corruption, it's a `corrupted_recovered` trajectory. The recorded actions are the ORIGINAL uncorrupted sequence (for SFT training). 40 scripted × 5 corruption types × ~3 step indices = ~600 corrupted_recovered trajectories.

**Global Constraints:**
1. ONLY create `scripts/augment_corrupted_recovered.py`. Do NOT touch any source files in `src/`.
2. `src/agent_trajectory.py` is FROZEN — do not modify.
3. The script does NOT require GPU (uses CorruptedActionProvider for replay, no model inference). Only verify syntax via `ast.parse` in this environment. The actual run happens before PR merge.
4. `_ListActionProvider` goes INSIDE the script file (defined inline, same pattern as T10).
5. No emojis, no incidental docstrings beyond what's in the brief code, no refactors.
6. Use `py -3.11` for all Python commands.

**Verified dependencies (pre-flight):**
- `CorruptionType` — enum at src/agent_evaluator.py:67 with 5 values: WRONG_ACTION_TYPE, INVALID_PATH, WRONG_PATCH, SKIP_TESTS_BEFORE_FINISH, EXCEED_MAX_STEPS
- `Corruption` — BaseModel at src/agent_evaluator.py:75 with `step_index: int` and `type: CorruptionType`
- `CorruptedActionProvider(base: Trajectory, corruption: Corruption)` — at src/agent_evaluator.py:138, replays all actions from trajectory, corrupts action at step_index, continues with original actions for other steps
- `load_trajectories(path)` — at src/agent_trajectory.py:188
- `Trajectory` has `.task_id` and `.steps` (each step has `.action`)
- `scripted.jsonl` exists at data/p4-agent/trajectories-v0/scripted.jsonl
- `Action` in src/agent_actions.py
- `AgentEvaluator`, `ActionProvider`, `AgentState` in src/agent_evaluator.py
- `MicroTaskWorkspace.from_task` exists
- No SentinelAction import needed (T11 imports only `Action` from `src.agent_actions`)

- [ ] **Step 1: Write the augmentation script**

Create `scripts/augment_corrupted_recovered.py` with EXACTLY this content:

```python
# scripts/augment_corrupted_recovered.py
"""Phase G: corrupted_recovered augmentation generator.

For each scripted trajectory, applies each of the 5 CorruptionType values
at multiple step indices (1, 2, 3 and any patchable steps), runs the
evaluator with CorruptedActionProvider for the corrupted prefix, then
appends the original uncorrupted remaining actions. If the trajectory
still reaches success, it's a corrupted_recovered trajectory.

Output: data/p4-agent/trajectories-v1/corrupted-recovered.jsonl

Usage:
    py -3.11 scripts/augment_corrupted_recovered.py
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
from src.agent_evaluator import (
    AgentEvaluator, ActionProvider, AgentState,
    CorruptedActionProvider, Corruption, CorruptionType,
)
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
_OUT = _ROOT / "data" / "p4-agent" / "trajectories-v1" / "corrupted-recovered.jsonl"

# Step indices to try corruption at (1, 2, 3 + any patchable steps)
_STEP_INDICES = [1, 2, 3]
_CORRUPTION_TYPES = list(CorruptionType)


def main():
    scripted_trajs = load_trajectories(_SCRIPTED)
    print(f"Loaded {len(scripted_trajs)} scripted trajectories")

    results = []
    for traj in scripted_trajs:
        task_id = traj.task_id
        task_dir = _TASKS_DIR / task_id
        if not task_dir.exists():
            continue

        # Determine patchable steps and merge with fixed step indices
        patchable_steps = [
            i for i, s in enumerate(traj.steps)
            if s.action.action_type in ("apply_patch", "propose_patch")
        ]
        step_indices = sorted(set(_STEP_INDICES + patchable_steps))
        # Filter to valid range
        step_indices = [i for i in step_indices if 0 <= i < len(traj.steps)]

        for step_idx in step_indices:
            for ctype in _CORRUPTION_TYPES:
                ws = MicroTaskWorkspace.from_task(task_dir)
                try:
                    # Run corrupted prefix
                    corruption = Corruption(step_index=step_idx, type=ctype)
                    corrupted_provider = CorruptedActionProvider(traj, corruption)
                    evaluator = AgentEvaluator(
                        ws, corrupted_provider, task_id, max_steps=20
                    )
                    result = evaluator.run()

                    # If the corrupted run still succeeded, record it
                    if result.success:
                        # Record the original action sequence (uncorrupted)
                        # as the replayable trajectory
                        actions = [s.action for s in traj.steps]
                        results.append({
                            "trajectory_id": f"corrupted_{task_id}_s{step_idx}_{ctype.name}",
                            "task_id": task_id,
                            "config": "corrupted",
                            "source": "corrupted_recovered",
                            "success": True,
                            "finish_claim_mismatch": result.finish_claim_mismatch,
                            "metrics": result.metrics,
                            "steps_executed": result.steps_executed,
                            "actions": [a.model_dump() for a in actions],
                            "step_diagnostics": [],
                        })
                except Exception:
                    pass  # skip failed corruptions
                finally:
                    ws.cleanup()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        for traj in results:
            f.write(json.dumps(traj) + "\n")
    print(f"Wrote {len(results)} corrupted_recovered trajectories to {_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is importable**

Run: `py -3.11 -c "import ast; ast.parse(open('scripts/augment_corrupted_recovered.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/augment_corrupted_recovered.py
git commit -m "feat(p4-1): Phase G — corrupted_recovered augmentation generator"
```

---
