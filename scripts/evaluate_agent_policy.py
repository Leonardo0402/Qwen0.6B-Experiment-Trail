"""Evaluate scripted agent trajectories through the real tool layer.

Run: set P4_ALLOW_NETWORK=0 && py -3.11 scripts/evaluate_agent_policy.py
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("P4_ALLOW_NETWORK", "0")
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.agent_workspace import MicroTaskWorkspace
from src.agent_trajectory import Trajectory
from src.agent_evaluator import AgentEvaluator, ReplayActionProvider

TRAJ_PATH = _ROOT / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"
TASKS_DIR = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"
OUT_PATH = _ROOT / "reports" / "p4" / "agent-eval-report.json"


def _load_trajectories() -> list[Trajectory]:
    trajectories = []
    with TRAJ_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(Trajectory.model_validate_json(line))
    return trajectories


def main():
    trajectories = _load_trajectories()
    all_results = []

    for traj in trajectories:
        task_dir = TASKS_DIR / traj.task_id
        ws = MicroTaskWorkspace.from_task(task_dir)
        try:
            provider = ReplayActionProvider(traj)
            evaluator = AgentEvaluator(
                workspace=ws,
                provider=provider,
                task_id=traj.task_id,
                max_steps=20,
            )
            result = evaluator.run()
            result.trajectory_id = traj.trajectory_id
            all_results.append(result)
            print(f"  {traj.task_id}: success={result.success}, steps={result.steps_executed}")
        finally:
            ws.cleanup()

    # Aggregate metrics
    agg = {}
    metric_keys = all_results[0].metrics.keys()
    for key in metric_keys:
        values = [r.metrics[key] for r in all_results]
        if isinstance(values[0], float):
            agg[key] = sum(values) / len(values)
        else:
            agg[key] = sum(values)

    # Compute eval hash
    eval_hash = hashlib.sha256(
        json.dumps(agg, sort_keys=True).encode()
    ).hexdigest()[:16]

    report = {
        "eval_hash": eval_hash,
        "config": {"max_steps": 20, "timeout_s": 10},
        "total_tasks": len(all_results),
        "metrics": agg,
        "per_task": [
            {
                "task_id": r.task_id,
                "trajectory_id": r.trajectory_id,
                "steps_executed": r.steps_executed,
                "success": r.success,
                "metrics": r.metrics,
                "errors": r.errors,
                "max_steps_hit": r.max_steps_hit,
            }
            for r in all_results
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEval complete: {len(all_results)} tasks")
    print(f"task_success_rate: {agg['task_success_rate']:.1%}")
    print(f"Report: {OUT_PATH}")


if __name__ == "__main__":
    main()
