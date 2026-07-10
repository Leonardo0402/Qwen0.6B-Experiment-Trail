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
