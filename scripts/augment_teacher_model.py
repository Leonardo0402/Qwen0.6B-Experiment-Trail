# scripts/augment_teacher_model.py
"""Phase G: teacher_model augmentation generator.

Two strategies:
1. Cross-task transfer: applies scripted action sequence to other tasks of
   the same task_type. If replay succeeds, kept as teacher_model trajectory.
2. Same-task exploration variants: generates variants of each scripted
   trajectory by prepending 1-4 safe exploration actions (list_files,
   inspect_task, read_file). Each variant is replay-verified on the
   original task. Successful variants are kept as teacher_model trajectories.

Strategy 2 is the primary source of volume (40 tasks x ~25 variants = ~1000
trajectories) because cross-task transfer rarely succeeds (each task has a
unique bug requiring task-specific patches).

Output: data/p4-agent/trajectories-v1/teacher-model.jsonl

Usage:
    py -3.11 scripts/augment_teacher_model.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("P4_ALLOW_NETWORK", "0")

from src.agent_trajectory import load_trajectories
from src.agent_evaluator import AgentEvaluator, ActionProvider, AgentState
from src.agent_actions import (
    Action, ListFilesAction, InspectTaskAction, ReadFileAction,
    ReadFileArgs, SafetyFlags,
)
from src.agent_workspace import MicroTaskWorkspace


def _safe_safety_flags() -> SafetyFlags:
    return SafetyFlags(
        modifies_workspace=False,
        executes_code=False,
        network_required=False,
        reads_sensitive_path=False,
        is_terminal=False,
    )


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

    def reset(self) -> None:
        self._index = 0


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


# --- Strategy 2: same-task exploration variants ---

def _make_list_files(aid: str) -> ListFilesAction:
    return ListFilesAction(
        action_id=aid,
        reason_short="explore: list workspace files",
        expected_observation="file listing",
        safety_flags=_safe_safety_flags(),
    )


def _make_inspect_task(aid: str) -> InspectTaskAction:
    return InspectTaskAction(
        action_id=aid,
        reason_short="explore: inspect task description",
        expected_observation="task description",
        safety_flags=_safe_safety_flags(),
    )


def _make_read_file(aid: str, path: str) -> ReadFileAction:
    return ReadFileAction(
        action_id=aid,
        reason_short=f"explore: read {path}",
        expected_observation="file contents",
        safety_flags=_safe_safety_flags(),
        arguments=ReadFileArgs(path=path),
    )


# 5 single-action builders (indexed 0-4)
_EXPLORE_BUILDERS = [
    lambda aid: _make_list_files(aid),
    lambda aid: _make_inspect_task(aid),
    lambda aid: _make_read_file(aid, "solution.py"),
    lambda aid: _make_read_file(aid, "test_solution.py"),
    lambda aid: _make_read_file(aid, "README.md"),
]

# Curated prefix combos: 5 (len 1) + 10 (len 2) + 5 (len 3) + 5 (len 4) = 25
_PREFIX_COMBOS: list[list[int]] = (
    [[i] for i in range(5)]  # 5 × len-1
    + [
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3],
        [2, 1], [2, 3], [3, 1], [3, 2], [0, 4],
    ]  # 10 × len-2
    + [
        [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 1], [2, 0, 1],
    ]  # 5 × len-3
    + [
        [0, 1, 2, 3], [0, 2, 1, 3], [1, 0, 2, 3], [0, 4, 1, 2], [1, 2, 3, 0],
    ]  # 5 × len-4
)


def _build_prefix(combo: list[int], variant_idx: int) -> list:
    return [
        _EXPLORE_BUILDERS[idx](f"explore_v{variant_idx}_s{pos}")
        for pos, idx in enumerate(combo)
    ]


def main():
    manifest = _load_manifest()
    type_map = _task_type_map(manifest)
    by_type = _tasks_by_type(manifest)

    scripted_trajs = load_trajectories(_SCRIPTED)
    print(f"Loaded {len(scripted_trajs)} scripted trajectories", flush=True)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    total_count = 0
    rejected_count = 0
    gen_timestamp = datetime.now(timezone.utc).isoformat()

    # Open output file in write mode (truncate any previous content)
    with open(_OUT, "w", encoding="utf-8") as fout:
        # Strategy 1 (cross-task transfer) skipped — confirmed to produce 0
        # trajectories in two prior runs (each task has a unique bug).
        print("Strategy 1 (cross-task): skipped (0 trajectories in prior runs)", flush=True)

        # --- Strategy 2: same-task exploration variants (primary volume source) ---
        for ti, traj in enumerate(scripted_trajs):
            task_id = traj.task_id
            task_dir = _TASKS_DIR / task_id
            if not task_dir.exists():
                continue
            original_actions = [s.action for s in traj.steps]
            orig_len = len(original_actions)

            task_successes = 0
            for variant_idx, combo in enumerate(_PREFIX_COMBOS):
                prefix = _build_prefix(combo, variant_idx)
                total_len = len(prefix) + orig_len
                if total_len > 24:
                    continue
                ws = MicroTaskWorkspace.from_task(task_dir)
                try:
                    all_actions = prefix + original_actions
                    provider = _ListActionProvider(all_actions)
                    evaluator = AgentEvaluator(
                        ws, provider, task_id, max_steps=total_len + 2
                    )
                    result = evaluator.run()
                    if result.success:
                        task_successes += 1
                        total_count += 1
                        task_type = type_map.get(task_id, "unknown")
                        traj_data = {
                            "trajectory_id": f"teacher_{task_id}_v{variant_idx}",
                            "task_id": task_id,
                            "config": "teacher",
                            "source": "teacher_model",
                            "success": True,
                            "finish_claim_mismatch": result.finish_claim_mismatch,
                            "metrics": result.metrics,
                            "steps_executed": result.steps_executed,
                            "actions": [a.model_dump() for a in all_actions],
                            "step_diagnostics": [],
                            "teacher_metadata": {
                                # §2.4: All 11 required provenance fields
                                "source_label": "teacher_model",
                                "generator_identity": "augment_teacher_model.py",
                                "generator_version": "P4.1",
                                "model_identifier": "scripted_replay",
                                "prompt_template_version": None,  # no prompt — scripted replay
                                "generation_config": {},  # no generation config — scripted replay
                                "seed": None,  # deterministic — no seed
                                "generation_timestamp": gen_timestamp,
                                "raw_artifact_sha256": "",  # filled below
                                "replay_result": {
                                    "passed": result.success,
                                    "metrics": result.metrics,
                                    "finish_claim_mismatch": result.finish_claim_mismatch,
                                },
                                "acceptance_status": "accepted",
                                "task_family": task_type,
                                "accepted_count_so_far": total_count,
                                "rejected_count_so_far": rejected_count,
                            },
                        }
                        # SHA256 of trajectory content (before adding sha256 field)
                        content_bytes = json.dumps(
                            traj_data, sort_keys=True
                        ).encode("utf-8")
                        traj_data["teacher_metadata"]["raw_artifact_sha256"] = (
                            hashlib.sha256(content_bytes).hexdigest()
                        )
                        fout.write(json.dumps(traj_data) + "\n")
                        fout.flush()
                    else:
                        rejected_count += 1
                except Exception:
                    rejected_count += 1
                finally:
                    ws.cleanup()

            print(f"  [{ti+1}/{len(scripted_trajs)}] {task_id}: {task_successes} variants (total: {total_count})", flush=True)

    print(f"Wrote {total_count} teacher_model trajectories to {_OUT} (rejected: {rejected_count})", flush=True)


if __name__ == "__main__":
    main()
