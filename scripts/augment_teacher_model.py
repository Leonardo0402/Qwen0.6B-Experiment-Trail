# scripts/augment_teacher_model.py
"""Phase G: teacher_model augmentation generator.

For each scripted trajectory, applies its action sequence to 3-4 other
tasks of the same task_type (cross-task transfer). If replay succeeds
(tests pass), the trajectory is kept as a `teacher_model` trajectory.

Output: data/p4-agent/trajectories-v1/teacher-model.jsonl

Usage:
    py -3.11 scripts/augment_teacher_model.py
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
    """Replays a list of Action objects (or SentinelAction). Yields them in
    order. Used for replay-verify in T10/T11/T12/T13."""

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
_MANIFEST = _ROOT / "data" / "p4-agent" / "micro-tasks-v0" / "manifest.json"
_TASKS_DIR = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"
_OUT = _ROOT / "data" / "p4-agent" / "trajectories-v1" / "teacher-model.jsonl"


def _load_manifest():
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def _task_type_map(manifest):
    return {t["task_id"]: t["task_type"] for t in manifest["tasks"]}


def _tasks_by_type(manifest):
    by_type: dict[str, list[str]] = {}
    for t in manifest["tasks"]:
        by_type.setdefault(t["task_type"], []).append(t["task_id"])
    return by_type


def main():
    manifest = _load_manifest()
    type_map = _task_type_map(manifest)
    by_type = _tasks_by_type(manifest)

    scripted_trajs = load_trajectories(_SCRIPTED)
    print(f"Loaded {len(scripted_trajs)} scripted trajectories")

    results = []
    for traj in scripted_trajs:
        src_task_id = traj.task_id
        src_type = type_map.get(src_task_id, "unknown")
        # Candidate target tasks: same type, different task_id
        candidates = [tid for tid in by_type.get(src_type, []) if tid != src_task_id]
        # Apply to up to 4 other tasks of the same type
        for target_task_id in candidates[:4]:
            task_dir = _TASKS_DIR / target_task_id
            if not task_dir.exists():
                continue
            ws = MicroTaskWorkspace.from_task(task_dir)
            try:
                actions = [s.action for s in traj.steps]
                provider = _ListActionProvider(actions)
                evaluator = AgentEvaluator(ws, provider, target_task_id, max_steps=20)
                result = evaluator.run()
                if result.success:
                    results.append({
                        "trajectory_id": f"teacher_{src_task_id}_{target_task_id}",
                        "task_id": target_task_id,
                        "config": "teacher",
                        "source": "teacher_model",
                        "success": True,
                        "finish_claim_mismatch": result.finish_claim_mismatch,
                        "metrics": result.metrics,
                        "steps_executed": result.steps_executed,
                        "actions": [a.model_dump() for a in actions],
                        "step_diagnostics": [],
                    })
            except Exception:
                pass  # skip failed transfers
            finally:
                ws.cleanup()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        for traj in results:
            f.write(json.dumps(traj) + "\n")
    print(f"Wrote {len(results)} teacher_model trajectories to {_OUT}")


if __name__ == "__main__":
    main()
