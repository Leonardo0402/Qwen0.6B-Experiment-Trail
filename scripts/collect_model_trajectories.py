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


def _aggregate_metrics(
    trajectories: list[dict],
    crashes: int,
    model_load_ok: bool = True,
    adapter_load_ok: bool = False,
    generation_ok: bool = True,
) -> dict:
    """Compute §2.3 smoke success metrics from collected trajectory data.

    Aggregates per-trajectory ``step_diagnostics`` and ``metrics`` into the
    11 required fields listed in roadmap §2.3:

    - model_load_ok / adapter_load_ok / generation_ok (passed through)
    - json_parse_rate / schema_valid_rate / safety_valid_rate /
      action_type_valid_rate / arguments_valid_rate (from step_diagnostics)
    - forbidden_action_count (summed from per-trajectory metrics)
    - tool_dispatch_ok (1 - mean(tool_error_rate))
    - max_step_stop_ok (count of trajectories that hit max_steps cleanly)
    - runtime_crash_count (from ``crashes``)
    """
    # Collect all step_diagnostics across trajectories
    all_diags: list[dict] = []
    for traj in trajectories:
        all_diags.extend(traj.get("step_diagnostics", []))

    total_diags = len(all_diags)
    if total_diags > 0:
        json_parse_rate = sum(
            1 for d in all_diags if d.get("json_parse_ok")
        ) / total_diags
        schema_valid_rate = sum(
            1 for d in all_diags if d.get("schema_valid")
        ) / total_diags
        safety_valid_rate = sum(
            1 for d in all_diags if d.get("safety_valid")
        ) / total_diags
        action_type_valid_rate = sum(
            1 for d in all_diags if d.get("action_type_valid")
        ) / total_diags
        arguments_valid_rate = sum(
            1 for d in all_diags if d.get("arguments_valid")
        ) / total_diags
    else:
        json_parse_rate = 0.0
        schema_valid_rate = 0.0
        safety_valid_rate = 0.0
        action_type_valid_rate = 0.0
        arguments_valid_rate = 0.0

    # forbidden_action_count: sum across trajectories
    forbidden_action_count = sum(
        t.get("metrics", {}).get("forbidden_action_count", 0) for t in trajectories
    )

    # tool_dispatch_ok: 1 - mean(tool_error_rate) across trajectories
    tool_error_rates = [
        t.get("metrics", {}).get("tool_error_rate", 0.0) for t in trajectories
    ]
    if tool_error_rates:
        mean_tool_error_rate = sum(tool_error_rates) / len(tool_error_rates)
    else:
        mean_tool_error_rate = 0.0
    tool_dispatch_ok = 1.0 - mean_tool_error_rate

    # max_step_stop_ok: count of trajectories that hit max_steps and stopped cleanly
    max_step_stop_ok = sum(
        t.get("metrics", {}).get("max_step_exceeded_count", 0) for t in trajectories
    )

    return {
        "model_load_ok": model_load_ok,
        "adapter_load_ok": adapter_load_ok,
        "generation_ok": generation_ok,
        "json_parse_rate": json_parse_rate,
        "schema_valid_rate": schema_valid_rate,
        "safety_valid_rate": safety_valid_rate,
        "action_type_valid_rate": action_type_valid_rate,
        "arguments_valid_rate": arguments_valid_rate,
        "forbidden_action_count": forbidden_action_count,
        "tool_dispatch_ok": tool_dispatch_ok,
        "max_step_stop_ok": max_step_stop_ok,
        "runtime_crash_count": crashes,
    }


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
            **_aggregate_metrics(
                [], crashes=len(task_ids),
                model_load_ok=False, adapter_load_ok=adapter_load_ok,
                generation_ok=False,
            ),
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
        **_aggregate_metrics(
            trajectories, crashes=crashes,
            model_load_ok=model_load_ok, adapter_load_ok=adapter_load_ok,
            generation_ok=len(trajectories) > 0,
        ),
        "trajectories": trajectories,
    }


def main():
    _TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    all_task_ids = _load_task_ids()
    task_ids = list(all_task_ids)
    limit = os.environ.get("P4_T9_TASK_LIMIT")
    if limit:
        task_ids = task_ids[:int(limit)]
        print(f"P4_T9_TASK_LIMIT={limit}: running first {len(task_ids)} tasks")
    limited_smoke = len(task_ids) < len(all_task_ids)
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
        summary["limited_smoke"] = limited_smoke
        if limited_smoke:
            summary["full_task_count"] = len(all_task_ids)
        reports.append(summary)

    _REPORT.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nWrote {_REPORT}")


if __name__ == "__main__":
    main()
