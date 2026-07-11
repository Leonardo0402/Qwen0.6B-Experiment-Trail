# scripts/run_protocol_ablation.py
"""P4.1b Protocol Ablation: run 3 protocols x 2 configs on 40 micro-tasks.

Produces:
- reports/p4/protocol-ablation/baseline-lock.json
- reports/p4/protocol-ablation/trajectories/{protocol}-{config}.jsonl
- reports/p4/protocol-ablation/comparison-matrix.json
- reports/p4/protocol-ablation/failure-taxonomy.json
- reports/p4/protocol-ablation/comparison-report.md

Usage:
    py -3.11 scripts/run_protocol_ablation.py
    py -3.11 scripts/run_protocol_ablation.py --task-limit 5  # quick smoke
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("P4_ALLOW_NETWORK", "0")

from src.protocols import JsonProtocol, TagProtocol, DslProtocol
from src.agent_model_provider import ModelActionProvider, SentinelAction
from src.agent_evaluator import AgentEvaluator, ActionProvider, AgentState
from src.agent_workspace import MicroTaskWorkspace

_PROTOCOLS = [
    {"name": "json", "class": JsonProtocol},
    {"name": "tag", "class": TagProtocol},
    {"name": "dsl", "class": DslProtocol},
]

_CONFIGS = [
    {"name": "base", "model_path": "models/Qwen3-0.6B", "adapter_path": None},
    {"name": "repair-lora", "model_path": "models/Qwen3-0.6B",
     "adapter_path": "adapters/p3/repair-limited"},
]

_TASKS_DIR = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"
_REPORT_DIR = _ROOT / "reports" / "p4" / "protocol-ablation"
MAX_STEPS = 12

_FAILURE_CLASSES = [
    "FORMAT_PARSE_FAIL", "SCHEMA_VALIDATION_FAIL", "UNKNOWN_ACTION_TYPE",
    "FORBIDDEN_ACTION", "INVALID_PATH", "EMPTY_OR_USELESS_ACTION",
    "MODEL_REFUSAL_OR_CHATTER", "REPEATED_ACTION_LOOP",
]


def _load_task_ids():
    manifest = json.loads((_TASKS_DIR / "manifest.json").read_text(encoding="utf-8"))
    return [t["task_id"] for t in manifest["tasks"]]


def _git_sha():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(_ROOT),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def baseline_lock() -> dict:
    """Record experiment starting state for reproducibility."""
    manifest_path = _TASKS_DIR / "manifest.json"
    return {
        "commit_sha": _git_sha(),
        "micro_task_manifest_path": str(manifest_path.relative_to(_ROOT)),
        "micro_task_manifest_sha256": _file_sha256(manifest_path),
        "model_path": "models/Qwen3-0.6B",
        "adapter_path": "adapters/p3/repair-limited",
        "generation_config": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 128,
            "dtype": "float16",
        },
        "max_steps": MAX_STEPS,
        "total_tasks": 40,
        "protocols": [p["name"] for p in _PROTOCOLS],
        "configs": [c["name"] for c in _CONFIGS],
        "total_combinations": len(_PROTOCOLS) * len(_CONFIGS),
        "total_runs": len(_PROTOCOLS) * len(_CONFIGS) * 40,
    }


class RecordingProvider(ActionProvider):
    """Wraps an ActionProvider and records each action for later replay."""

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
        return self._inner.diagnostics if hasattr(self._inner, "diagnostics") else []

    def reset(self) -> None:
        self._recorded.clear()
        if hasattr(self._inner, "reset"):
            self._inner.reset()


def run_combination(protocol, config, task_ids, max_steps=MAX_STEPS):
    """Run one protocol x config combination on all task_ids."""
    trajectories = []
    crashes = 0
    model_load_ok = False

    inner_provider = ModelActionProvider(
        model_path=config["model_path"],
        adapter_path=config["adapter_path"],
        protocol=protocol,
    )

    # Try to load the model once
    try:
        inner_provider._load_model()
        model_load_ok = True
    except Exception as e:
        print(f"[{protocol.name}/{config['name']}] model load failed: {e}")
        return {
            "protocol": protocol.name,
            "config": config["name"],
            "total_tasks": len(task_ids),
            "trajectories_written": 0,
            "model_load_ok": False,
            "metrics": {},
            "trajectories": [],
        }

    for i, task_id in enumerate(task_ids):
        task_dir = _TASKS_DIR / task_id
        ws = MicroTaskWorkspace.from_task(task_dir)
        try:
            inner_provider.reset()
            provider = RecordingProvider(inner_provider)
            evaluator = AgentEvaluator(ws, provider, task_id, max_steps=max_steps)
            result = evaluator.run()
            trajectories.append({
                "trajectory_id": f"{protocol.name}_{config['name']}_{task_id}",
                "task_id": task_id,
                "protocol": protocol.name,
                "config": config["name"],
                "success": result.success,
                "finish_claim_mismatch": result.finish_claim_mismatch,
                "metrics": result.metrics,
                "steps_executed": result.steps_executed,
                "max_steps_hit": result.max_steps_hit,
                "actions": provider.recorded_actions,
                "step_diagnostics": [d.model_dump() for d in inner_provider.diagnostics],
            })
        except Exception:
            crashes += 1
            traceback.print_exc()
        finally:
            ws.cleanup()
        print(f"\r[{protocol.name}/{config['name']}] {i+1}/{len(task_ids)} {task_id}", end="", flush=True)
    print()

    metrics = aggregate_metrics(trajectories, crashes, model_load_ok)
    return {
        "protocol": protocol.name,
        "config": config["name"],
        "total_tasks": len(task_ids),
        "trajectories_written": len(trajectories),
        "model_load_ok": model_load_ok,
        "metrics": metrics,
        "trajectories": trajectories,
    }


def aggregate_metrics(trajectories: list[dict], crashes: int = 0,
                      model_load_ok: bool = True) -> dict:
    """Compute aggregated metrics from trajectory step_diagnostics.

    Returns 13 metrics per spec §6.2 (12+ required by acceptance criteria).
    """
    all_diags: list[dict] = []
    for traj in trajectories:
        all_diags.extend(traj.get("step_diagnostics", []))

    total_diags = len(all_diags)
    if total_diags > 0:
        format_parse_rate = sum(1 for d in all_diags if d.get("format_parse_ok")) / total_diags
        schema_valid_rate = sum(1 for d in all_diags if d.get("schema_valid")) / total_diags
        safety_valid_rate = sum(1 for d in all_diags if d.get("safety_valid")) / total_diags
        action_type_valid_rate = sum(1 for d in all_diags if d.get("action_type_valid")) / total_diags
        arguments_valid_rate = sum(1 for d in all_diags if d.get("arguments_valid")) / total_diags
    else:
        format_parse_rate = 0.0
        schema_valid_rate = 0.0
        safety_valid_rate = 0.0
        action_type_valid_rate = 0.0
        arguments_valid_rate = 0.0

    forbidden_action_count = sum(
        t.get("metrics", {}).get("forbidden_action_count", 0) for t in trajectories
    )
    unknown_action_count = sum(1 for d in all_diags if not d.get("action_type_valid"))
    total_tasks = len(trajectories)
    task_success_rate = sum(1 for t in trajectories if t.get("success")) / total_tasks if total_tasks else 0.0
    max_steps_hit_rate = sum(1 for t in trajectories if t.get("max_steps_hit")) / total_tasks if total_tasks else 0.0

    # Finish-related metrics (from evaluator)
    finish_without_tests_count = sum(
        1 for t in trajectories
        if t.get("success") and not t.get("metrics", {}).get("tests_passed", False)
    )
    finish_claim_mismatch_count = sum(
        1 for t in trajectories if t.get("finish_claim_mismatch")
    )

    return {
        "format_parse_rate": format_parse_rate,
        "schema_valid_rate": schema_valid_rate,
        "safety_valid_rate": safety_valid_rate,
        "action_type_valid_rate": action_type_valid_rate,
        "arguments_valid_rate": arguments_valid_rate,
        "forbidden_action_count": forbidden_action_count,
        "unknown_action_count": unknown_action_count,
        "task_success_rate": task_success_rate,
        "finish_without_tests_count": finish_without_tests_count,
        "finish_claim_mismatch_count": finish_claim_mismatch_count,
        "max_steps_hit_rate": max_steps_hit_rate,
        "runtime_crash_count": crashes,
        "model_load_ok": model_load_ok,
    }


def classify_failures(trajectories: list[dict]) -> dict:
    """Classify all failed steps into failure taxonomy.

    Also detects REPEATED_ACTION_LOOP at trajectory level (3+ consecutive
    identical actions in a single trajectory).
    """
    taxonomy = {fc: 0 for fc in _FAILURE_CLASSES}
    for traj in trajectories:
        for d in traj.get("step_diagnostics", []):
            fc = d.get("failure_class")
            if fc and fc in taxonomy:
                taxonomy[fc] += 1
        # Trajectory-level: detect repeated action loops
        actions = traj.get("actions", [])
        if _detect_repeated_loop(actions):
            taxonomy["REPEATED_ACTION_LOOP"] += 1
    return taxonomy


def _detect_repeated_loop(actions: list[dict]) -> bool:
    """Return True if 3+ consecutive identical actions found."""
    if len(actions) < 3:
        return False
    for i in range(len(actions) - 2):
        a1 = actions[i]
        a2 = actions[i + 1]
        a3 = actions[i + 2]
        if (a1.get("action_type") == a2.get("action_type") == a3.get("action_type")
                and a1.get("arguments") == a2.get("arguments") == a3.get("arguments")):
            return True
    return False


def generate_report(results: list[dict], taxonomy: dict) -> str:
    """Generate markdown comparison report from ablation results."""
    lines = [
        "# P4.1b Protocol Ablation — Comparison Report",
        "",
        "## Overview",
        "",
        f"- Protocols: {len(set(r['protocol'] for r in results))}",
        f"- Configs: {len(set(r['config'] for r in results))}",
        f"- Total combinations: {len(results)}",
        "",
        "## Metrics by Protocol x Config",
        "",
        "| Protocol | Config | format_parse_rate | schema_valid_rate | safety_valid_rate | action_type_valid_rate | arguments_valid_rate | forbidden_count | unknown_count | task_success_rate | finish_no_tests | finish_mismatch | max_steps_hit_rate | crashes |",
        "|----------|--------|-------------------|-------------------|-------------------|------------------------|----------------------|-----------------|----------------|-------------------|------------------|-----------------|---------------------|---------|",
    ]

    for r in sorted(results, key=lambda x: (x["protocol"], x["config"])):
        m = r.get("metrics", {})
        lines.append(
            f"| {r['protocol']} | {r['config']} "
            f"| {m.get('format_parse_rate', 0):.2%} "
            f"| {m.get('schema_valid_rate', 0):.2%} "
            f"| {m.get('safety_valid_rate', 0):.2%} "
            f"| {m.get('action_type_valid_rate', 0):.2%} "
            f"| {m.get('arguments_valid_rate', 0):.2%} "
            f"| {m.get('forbidden_action_count', 0)} "
            f"| {m.get('unknown_action_count', 0)} "
            f"| {m.get('task_success_rate', 0):.2%} "
            f"| {m.get('finish_without_tests_count', 0)} "
            f"| {m.get('finish_claim_mismatch_count', 0)} "
            f"| {m.get('max_steps_hit_rate', 0):.2%} "
            f"| {m.get('runtime_crash_count', 0)} |"
        )

    lines.extend([
        "",
        "## Failure Taxonomy",
        "",
        "| Failure Class | Count |",
        "|---------------|-------|",
    ])
    for fc, count in sorted(taxonomy.items()):
        lines.append(f"| {fc} | {count} |")

    lines.extend([
        "",
        "## Protocol Comparison Summary",
        "",
    ])

    # Summarize by protocol (average across configs)
    protocols = sorted(set(r["protocol"] for r in results))
    for proto in protocols:
        proto_results = [r for r in results if r["protocol"] == proto]
        avg_schema = sum(r["metrics"].get("schema_valid_rate", 0) for r in proto_results) / len(proto_results)
        lines.append(f"- **{proto}**: avg schema_valid_rate = {avg_schema:.2%}")

    return "\n".join(lines)


def main():
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    traj_dir = _REPORT_DIR / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Baseline lock
    print("=== Step 1: Baseline Lock ===")
    lock = baseline_lock()
    (_REPORT_DIR / "baseline-lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    print(f"Wrote {_REPORT_DIR / 'baseline-lock.json'}")

    # Step 2: Run all 6 combinations
    print("\n=== Step 2: Run Protocol x Config Combinations ===")
    all_task_ids = _load_task_ids()
    task_limit = os.environ.get("P4_1B_TASK_LIMIT")
    if task_limit:
        all_task_ids = all_task_ids[:int(task_limit)]
        print(f"P4_1B_TASK_LIMIT={task_limit}: running first {len(all_task_ids)} tasks")

    all_results = []
    for proto_spec in _PROTOCOLS:
        for config in _CONFIGS:
            print(f"\n--- Protocol: {proto_spec['name']} | Config: {config['name']} ---")
            proto = proto_spec["class"]()
            result = run_combination(proto, config, all_task_ids)
            # Write trajectories JSONL
            out_file = traj_dir / f"{proto_spec['name']}-{config['name']}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for traj in result["trajectories"]:
                    f.write(json.dumps(traj) + "\n")
            # Strip trajectories from summary
            summary = {k: v for k, v in result.items() if k != "trajectories"}
            all_results.append(summary)
            print(f"  schema_valid_rate: {summary['metrics'].get('schema_valid_rate', 0):.2%}")

    # Step 3: Comparison matrix
    print("\n=== Step 3: Comparison Matrix ===")
    (_REPORT_DIR / "comparison-matrix.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    # Step 4: Failure taxonomy
    print("\n=== Step 4: Failure Taxonomy ===")
    # Reload trajectories for taxonomy
    all_trajectories = []
    for proto_spec in _PROTOCOLS:
        for config in _CONFIGS:
            traj_file = traj_dir / f"{proto_spec['name']}-{config['name']}.jsonl"
            if traj_file.exists():
                for line in traj_file.read_text(encoding="utf-8").strip().split("\n"):
                    if line:
                        all_trajectories.append(json.loads(line))
    taxonomy = classify_failures(all_trajectories)
    (_REPORT_DIR / "failure-taxonomy.json").write_text(
        json.dumps(taxonomy, indent=2), encoding="utf-8"
    )

    # Step 5: Generate report
    print("\n=== Step 5: Comparison Report ===")
    report = generate_report(all_results, taxonomy)
    (_REPORT_DIR / "comparison-report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {_REPORT_DIR / 'comparison-report.md'}")

    print(f"\nDone. Reports in {_REPORT_DIR}")


if __name__ == "__main__":
    main()
