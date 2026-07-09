# scripts/build_agent_sft_dataset.py
"""Phase G: build the Agent SFT dataset.

Aggregates trajectories from 6 sources (scripted_variant, model_self_run,
teacher_model, corrupted_recovered, failed_patch_recovery), labels them,
splits by task family, and replay-verifies every trajectory via
_ListActionProvider before inclusion.

Usage:
    py -3.11 scripts/build_agent_sft_dataset.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from pydantic import TypeAdapter

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("P4_ALLOW_NETWORK", "0")

from src.agent_trajectory import load_trajectories
from src.agent_evaluator import AgentEvaluator, ActionProvider, AgentState
from src.agent_actions import Action
from src.agent_model_provider import SentinelAction
from src.agent_workspace import MicroTaskWorkspace


class _ListActionProvider(ActionProvider):
    """Replays a list of Action objects (or SentinelAction). Yields them
    in order. Used for replay-verify — same pattern as _FixedProvider in
    tests, but without the test helpers."""

    def __init__(self, actions: list):
        self._actions = list(actions)
        self._index = 0

    def next_action(self, state: AgentState):
        if self._index >= len(self._actions):
            raise StopIteration("no more actions in list")
        action = self._actions[self._index]
        self._index += 1
        return action


_ACTION_ADAPTER = TypeAdapter(Action)

_SCRIPTED = _ROOT / "data" / "p4-agent" / "trajectories-v0" / "scripted.jsonl"
_MODEL_BASE = _ROOT / "data" / "p4-agent" / "trajectories-v1" / "model-base.jsonl"
_MODEL_REPAIR = _ROOT / "data" / "p4-agent" / "trajectories-v1" / "model-repair-lora.jsonl"
_TEACHER = _ROOT / "data" / "p4-agent" / "trajectories-v1" / "teacher-model.jsonl"
_CORRUPTED = _ROOT / "data" / "p4-agent" / "trajectories-v1" / "corrupted-recovered.jsonl"
_FAILED_PATCH = _ROOT / "data" / "p4-agent" / "trajectories-v1" / "failed-patch-recovery.jsonl"
_MANIFEST = _ROOT / "data" / "p4-agent" / "micro-tasks-v0" / "manifest.json"
_OUT_DIR = _ROOT / "data" / "p4-agent" / "sft-v1"
_FAILURES = _ROOT / "reports" / "p4" / "sft-dataset-replay-failures.jsonl"

# Task type split (user decision #6)
HELDOUT_TYPE = "avoid_editing_tests"
VALIDATION_TYPE = "recover_from_failed_patch"
# train = all other types

SOURCES = {
    "scripted_variant", "teacher_model", "corrupted_recovered",
    "failed_patch_recovery", "model_self_run_success", "model_self_run_failure",
}


def _load_manifest():
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def _task_type_map(manifest):
    return {t["task_id"]: t["task_type"] for t in manifest["tasks"]}


def _split_for_type(task_type):
    if task_type == HELDOUT_TYPE:
        return "heldout-agent-eval"
    if task_type == VALIDATION_TYPE:
        return "validation"
    return "train"


def _reconstruct_actions(action_dicts: list) -> list:
    """Reconstruct Action objects (or SentinelAction) from a list of dicts.
    SentinelActions are detected via the `__sentinel__` marker."""
    actions = []
    for d in action_dicts:
        if d.get("__sentinel__"):
            actions.append(SentinelAction(
                reason=d.get("reason", ""),
                is_invalid=d.get("is_invalid", True),
            ))
        else:
            actions.append(_ACTION_ADAPTER.validate_python(d))
    return actions


def _replay_verify(traj_data, task_dir, expected_success):
    """Replay-verify a trajectory using _ListActionProvider.
    Returns (ok, result_or_error)."""
    try:
        if "actions" in traj_data:
            # P4.1 JSONL format — reconstruct from actions list
            actions = _reconstruct_actions(traj_data["actions"])
        else:
            # P4.0 Trajectory format — shouldn't reach here (handled by caller)
            return (False, "unexpected trajectory format (no actions field)")

        ws = MicroTaskWorkspace.from_task(task_dir)
        try:
            provider = _ListActionProvider(actions)
            evaluator = AgentEvaluator(ws, provider, traj_data.get("task_id", ""), max_steps=20)
            result = evaluator.run()
            ok = (result.success == expected_success
                  and result.metrics.get("forbidden_action_count", 0) == 0)
            return (ok, result)
        finally:
            ws.cleanup()
    except Exception as e:
        return (False, str(e))


def _load_jsonl(path):
    if not path.exists():
        return []
    result = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _FAILURES.parent.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()
    task_types = _task_type_map(manifest)
    tasks_dir = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"

    all_trajectories = []

    # Source 1: scripted_variant (from P4.0 scripted.jsonl — Trajectory format)
    # Extract action list from traj.steps[i].action, convert to JSONL format
    scripted_trajs = load_trajectories(_SCRIPTED)
    for traj in scripted_trajs:
        actions = [s.action.model_dump() for s in traj.steps]
        all_trajectories.append({
            "trajectory_id": f"scripted_{traj.trajectory_id}",
            "task_id": traj.task_id,
            "config": "scripted",
            "source": "scripted_variant",
            "success": traj.final_success,
            "finish_claim_mismatch": False,
            "metrics": {},
            "steps_executed": len(traj.steps),
            "actions": actions,
            "step_diagnostics": [],
        })

    # Sources 5 & 6: model_self_run_success / model_self_run_failure (T9)
    for path in [_MODEL_BASE, _MODEL_REPAIR]:
        for traj in _load_jsonl(path):
            # source already set by collection script
            all_trajectories.append(traj)

    # Source 2: teacher_model (T10)
    for traj in _load_jsonl(_TEACHER):
        all_trajectories.append(traj)

    # Source 3: corrupted_recovered (T11)
    for traj in _load_jsonl(_CORRUPTED):
        all_trajectories.append(traj)

    # Source 4: failed_patch_recovery (T12)
    for traj in _load_jsonl(_FAILED_PATCH):
        all_trajectories.append(traj)

    print(f"Loaded {len(all_trajectories)} total trajectories from all sources")

    # Replay-verify + split
    train, validation, heldout, failures = [], [], [], []
    failure_lines = []

    for traj in all_trajectories:
        task_id = traj.get("task_id", "")
        task_type = task_types.get(task_id, "unknown")
        split = _split_for_type(task_type)
        task_dir = tasks_dir / task_id
        expected_success = traj.get("success", False)

        ok, result = _replay_verify(traj, task_dir, expected_success)
        if not ok:
            failure_lines.append({
                "trajectory_id": traj.get("trajectory_id", ""),
                "task_id": task_id,
                "error": str(result),
            })
            continue

        traj["split"] = split
        traj["task_type"] = task_type

        # Failure trajectories go to failure-diagnostics, not train/val/heldout
        if traj.get("source") == "model_self_run_failure":
            failures.append(traj)
        else:
            if split == "train":
                train.append(traj)
            elif split == "validation":
                validation.append(traj)
            else:
                heldout.append(traj)

    # Write outputs
    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    write_jsonl(_OUT_DIR / "train.jsonl", train)
    write_jsonl(_OUT_DIR / "validation.jsonl", validation)
    write_jsonl(_OUT_DIR / "heldout-agent-eval.jsonl", heldout)
    write_jsonl(_OUT_DIR / "failure-diagnostics.jsonl", failures)
    write_jsonl(_FAILURES, failure_lines)

    dataset_manifest = {
        "schema_version": 1,
        "total_trajectories": len(train) + len(validation) + len(heldout) + len(failures),
        "train_count": len(train),
        "validation_count": len(validation),
        "heldout_count": len(heldout),
        "failure_count": len(failures),
        "replay_failures": len(failure_lines),
        "splits": {
            "train": {"task_types": [t for t in sorted(set(task_types.values()))
                                     if t not in (HELDOUT_TYPE, VALIDATION_TYPE)]},
            "validation": {"task_types": [VALIDATION_TYPE]},
            "heldout-agent-eval": {"task_types": [HELDOUT_TYPE]},
        },
        "sources": sorted(SOURCES),
    }
    (_OUT_DIR / "manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2), encoding="utf-8"
    )
    print(f"train={len(train)} val={len(validation)} heldout={len(heldout)} "
          f"failures={len(failures)} replay_failures={len(failure_lines)}")


if __name__ == "__main__":
    main()
