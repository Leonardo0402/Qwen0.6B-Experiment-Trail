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
# Issue #32: parameterize report dir for v2 reruns
_REPORT_DIR_NAME = os.environ.get("P4_1B_REPORT_DIR", "protocol-ablation")
_REPORT_DIR = _ROOT / "reports" / "p4" / _REPORT_DIR_NAME
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


def _git_worktree_clean_for_experiment() -> bool:
    """Issue #32 Final Trust Repair: check that no experiment-affecting
    tracked files have uncommitted modifications.

    Experiment-affecting paths:
    - src/ (all protocol and action code)
    - scripts/run_protocol_ablation.py (experiment harness)
    - data/p4-agent/micro-tasks-v0/ (task definitions)

    Unrelated files (AGENTS.md, docs/, reports/p4/p4-1-*, data/p3-limited/*,
    test-results.xml) may have working-tree modifications without blocking
    the experiment.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            path = line[3:]
            if (path.startswith("src/")
                    or path == "scripts/run_protocol_ablation.py"
                    or path.startswith("data/p4-agent/")):
                return False
        return True
    except Exception:
        return False


def assert_clean_experiment_state() -> None:
    """Issue #32 Final Trust Repair: hard-fail if experiment-affecting
    tracked files have uncommitted modifications.

    Call this before running the experiment to guarantee reproducibility
    from the recorded experiment_commit_sha.
    """
    if not _git_worktree_clean_for_experiment():
        print("FATAL: working tree has uncommitted modifications to "
              "experiment-affecting files (src/, scripts/, data/p4-agent/).")
        print("Commit all experiment code before running the ablation.")
        print("Run: git status --porcelain -- src/ scripts/run_protocol_ablation.py data/p4-agent/")
        sys.exit(1)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def baseline_lock() -> dict:
    """Record experiment starting state for reproducibility.

    Issue #32: enhanced to record source file SHAs, task IDs, and
    environment details as required by Trust Repair spec.
    """
    manifest_path = _TASKS_DIR / "manifest.json"
    task_ids = _load_task_ids()

    # Source file SHAs
    src_files = {
        "agent_actions": _ROOT / "src" / "agent_actions.py",
        "agent_evaluator": _ROOT / "src" / "agent_evaluator.py",
        "agent_model_provider": _ROOT / "src" / "agent_model_provider.py",
        "protocols_base": _ROOT / "src" / "protocols" / "base.py",
        "protocols_json": _ROOT / "src" / "protocols" / "json_protocol.py",
        "protocols_tag": _ROOT / "src" / "protocols" / "tag_protocol.py",
        "protocols_dsl": _ROOT / "src" / "protocols" / "dsl_protocol.py",
    }
    src_shas = {}
    for name, path in src_files.items():
        if path.exists():
            src_shas[name] = _file_sha256(path)
        else:
            src_shas[name] = "NOT_FOUND"

    # Environment info
    import platform
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
            env_info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        env_info["torch_version"] = "NOT_INSTALLED"
    try:
        import transformers
        env_info["transformers_version"] = transformers.__version__
    except ImportError:
        env_info["transformers_version"] = "NOT_INSTALLED"

    return {
        "experiment_commit_sha": _git_sha(),
        "git_worktree_clean_for_experiment": _git_worktree_clean_for_experiment(),
        "report_dir_name": _REPORT_DIR_NAME,
        "micro_task_manifest_path": str(manifest_path.relative_to(_ROOT)),
        "micro_task_manifest_sha256": _file_sha256(manifest_path),
        "task_ids": task_ids,
        "model_path": "models/Qwen3-0.6B",
        "adapter_path_base": None,
        "adapter_path_repair_lora": "adapters/p3/repair-limited",
        "generation_config": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 128,
            "dtype": "float16",
        },
        "max_steps": MAX_STEPS,
        "total_tasks": len(task_ids),
        "protocols": [p["name"] for p in _PROTOCOLS],
        "configs": [c["name"] for c in _CONFIGS],
        "total_combinations": len(_PROTOCOLS) * len(_CONFIGS),
        "total_runs": len(_PROTOCOLS) * len(_CONFIGS) * len(task_ids),
        "source_file_shas": src_shas,
        "environment": env_info,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
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

    Issue #32 Trust Repair:
    - Task D: unknown_action_count uses failure_class=="UNKNOWN_ACTION_TYPE"
      (not `not action_type_valid`) to avoid double-counting FORMAT_PARSE_FAIL
      steps which also have action_type_valid=False.
    - Task E: finish_without_tests_count uses evaluator's explicit metric
      (not `success and not tests_passed`) so failed trajectories that
      finished without tests are also counted.
    - Task F: All rates include explicit numerator/denominator for transparency.
    """
    all_diags: list[dict] = []
    for traj in trajectories:
        all_diags.extend(traj.get("step_diagnostics", []))

    total_steps = len(all_diags)
    total_trajectories = len(trajectories)

    # Step-level metrics with numerator/denominator (Task F)
    format_parse_success_steps = sum(1 for d in all_diags if d.get("format_parse_ok"))
    schema_valid_steps = sum(1 for d in all_diags if d.get("schema_valid"))
    safety_valid_steps = sum(1 for d in all_diags if d.get("safety_valid"))
    action_type_valid_steps = sum(1 for d in all_diags if d.get("action_type_valid"))
    arguments_valid_steps = sum(1 for d in all_diags if d.get("arguments_valid"))

    format_parse_rate = format_parse_success_steps / total_steps if total_steps else 0.0
    schema_valid_rate = schema_valid_steps / total_steps if total_steps else 0.0
    safety_valid_rate = safety_valid_steps / total_steps if total_steps else 0.0
    action_type_valid_rate = action_type_valid_steps / total_steps if total_steps else 0.0
    arguments_valid_rate = arguments_valid_steps / total_steps if total_steps else 0.0

    forbidden_action_count = sum(
        t.get("metrics", {}).get("forbidden_action_count", 0) for t in trajectories
    )
    # Task D: Use failure_class to avoid double-counting.
    # FORMAT_PARSE_FAIL steps also have action_type_valid=False, but they
    # should NOT be counted as UNKNOWN_ACTION_TYPE.
    unknown_action_count = sum(
        1 for d in all_diags if d.get("failure_class") == "UNKNOWN_ACTION_TYPE"
    )

    # Trajectory-level metrics with numerator/denominator (Task F)
    successful_trajectories = sum(1 for t in trajectories if t.get("success"))
    max_steps_hit_count = sum(1 for t in trajectories if t.get("max_steps_hit"))
    task_success_rate = successful_trajectories / total_trajectories if total_trajectories else 0.0
    max_steps_hit_rate = max_steps_hit_count / total_trajectories if total_trajectories else 0.0

    # Task E: Use evaluator's explicit finish_without_tests_count.
    # Previously: `success and not tests_passed` — missed failed trajectories
    # that finished without tests (finish-without-tests usually causes failure).
    finish_without_tests_count = sum(
        t.get("metrics", {}).get("finish_without_tests_count", 0) for t in trajectories
    )
    finish_claim_mismatch_count = sum(
        1 for t in trajectories if t.get("finish_claim_mismatch")
    )

    return {
        # Step-level rates with numerator/denominator (Task F)
        "total_steps": total_steps,
        "format_parse_success_steps": format_parse_success_steps,
        "format_parse_rate": format_parse_rate,
        "schema_valid_steps": schema_valid_steps,
        "schema_valid_rate": schema_valid_rate,
        "safety_valid_steps": safety_valid_steps,
        "safety_valid_rate": safety_valid_rate,
        "action_type_valid_steps": action_type_valid_steps,
        "action_type_valid_rate": action_type_valid_rate,
        "arguments_valid_steps": arguments_valid_steps,
        "arguments_valid_rate": arguments_valid_rate,
        # Failure counts
        "forbidden_action_count": forbidden_action_count,
        "unknown_action_count": unknown_action_count,
        # Trajectory-level metrics with numerator/denominator (Task F)
        "total_trajectories": total_trajectories,
        "successful_trajectories": successful_trajectories,
        "task_success_rate": task_success_rate,
        "max_steps_hit_count": max_steps_hit_count,
        "max_steps_hit_rate": max_steps_hit_rate,
        # Finish-related metrics
        "finish_without_tests_count": finish_without_tests_count,
        "finish_claim_mismatch_count": finish_claim_mismatch_count,
        # Runtime
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
    """Generate markdown comparison report from ablation results.

    Issue #32 Task F: Show numerator/denominator alongside rates so
    percentages can be traced to raw counts (e.g. 462/480 = 96.25%).
    """
    lines = [
        f"# P4.1b Protocol Ablation — Comparison Report ({_REPORT_DIR_NAME})",
        "",
        "## Overview",
        "",
        f"- Protocols: {len(set(r['protocol'] for r in results))}",
        f"- Configs: {len(set(r['config'] for r in results))}",
        f"- Total combinations: {len(results)}",
        f"- Report dir: {_REPORT_DIR_NAME}",
        "",
        "## Metrics by Protocol x Config",
        "",
        "Rates shown as `numerator / denominator = rate`.",
        "",
        "| Protocol | Config | schema_valid | arguments_valid | task_success | max_steps_hit | unknown_actions | finish_no_tests | crashes |",
        "|----------|--------|--------------|-----------------|--------------|---------------|-----------------|-----------------|---------|",
    ]

    for r in sorted(results, key=lambda x: (x["protocol"], x["config"])):
        m = r.get("metrics", {})
        ts = m.get("total_steps", 0)
        tt = m.get("total_trajectories", 0)
        lines.append(
            f"| {r['protocol']} | {r['config']} "
            f"| {m.get('schema_valid_steps', 0)}/{ts} = {m.get('schema_valid_rate', 0):.2%} "
            f"| {m.get('arguments_valid_steps', 0)}/{ts} = {m.get('arguments_valid_rate', 0):.2%} "
            f"| {m.get('successful_trajectories', 0)}/{tt} = {m.get('task_success_rate', 0):.2%} "
            f"| {m.get('max_steps_hit_count', 0)}/{tt} = {m.get('max_steps_hit_rate', 0):.2%} "
            f"| {m.get('unknown_action_count', 0)} "
            f"| {m.get('finish_without_tests_count', 0)} "
            f"| {m.get('runtime_crash_count', 0)} |"
        )

    lines.extend([
        "",
        "## Detailed Step-Level Metrics",
        "",
        "| Protocol | Config | total_steps | format_parse | safety_valid | action_type_valid |",
        "|----------|--------|-------------|--------------|--------------|-------------------|",
    ])
    for r in sorted(results, key=lambda x: (x["protocol"], x["config"])):
        m = r.get("metrics", {})
        ts = m.get("total_steps", 0)
        lines.append(
            f"| {r['protocol']} | {r['config']} "
            f"| {ts} "
            f"| {m.get('format_parse_success_steps', 0)}/{ts} = {m.get('format_parse_rate', 0):.2%} "
            f"| {m.get('safety_valid_steps', 0)}/{ts} = {m.get('safety_valid_rate', 0):.2%} "
            f"| {m.get('action_type_valid_steps', 0)}/{ts} = {m.get('action_type_valid_rate', 0):.2%} |"
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


def generate_artifact_manifest(report_dir: Path) -> dict:
    """Generate manifest with SHA256, size, row count for each artifact.

    Issue #32: required for reproducibility audit.
    Issue #32 Final Trust Repair: the manifest must NOT include itself
    (方案 A). Self-reference causes the recorded SHA to become invalid
    once the manifest is written to disk.
    """
    artifacts = []
    for path in sorted(report_dir.rglob("*")):
        if path.is_file():
            # Skip self-reference: artifact-manifest.json must not
            # record its own SHA (it doesn't exist yet at generation
            # time, but skip it explicitly in case of re-runs).
            if path.name == "artifact-manifest.json":
                continue
            rel = path.relative_to(report_dir)
            sha = _file_sha256(path)
            size = path.stat().st_size
            entry = {
                "relative_path": str(rel),
                "sha256": sha,
                "size": size,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "experiment_commit": _git_sha(),
            }
            # Add row count for JSONL files
            if path.suffix == ".jsonl":
                with open(path, encoding="utf-8") as f:
                    row_count = sum(1 for line in f if line.strip())
                entry["row_count"] = row_count
            artifacts.append(entry)
    return {
        "report_dir": _REPORT_DIR_NAME,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


_ALLOWED_VERDICTS = {
    "KEEP_ACTION_JSON", "TRY_TAG_PROTOCOL_FOR_P4_2", "TRY_DSL_FOR_P4_2",
    "FIX_PROMPT_FIRST", "FIX_EVALUATOR_FIRST", "STOP_PROTOCOL_CHANGE",
}


def compute_verdict(results: list[dict]) -> str:
    """Apply T8 verdict decision rules.

    Rules (from spec §7.2), applied in this order:
    1. Evaluator issues (model load failed or >50% trajectories crashed)
       → FIX_EVALUATOR_FIRST
    2. All protocols schema_valid_rate < 30% → FIX_PROMPT_FIRST
    3. Any alternative protocol's schema_valid_rate >30% better than JSON
       AND safety_valid_rate not degraded → TRY_TAG/TRY_DSL
    4. JSON baseline has highest schema_valid_rate → KEEP_ACTION_JSON
    5. Fallback → STOP_PROTOCOL_CHANGE
    """
    # Average schema_valid_rate per protocol (across configs)
    proto_rates: dict[str, list[float]] = {}
    proto_safety: dict[str, list[float]] = {}
    for r in results:
        proto = r["protocol"]
        if proto not in proto_rates:
            proto_rates[proto] = []
            proto_safety[proto] = []
        proto_rates[proto].append(r["metrics"].get("schema_valid_rate", 0))
        proto_safety[proto].append(r["metrics"].get("safety_valid_rate", 0))

    avg_schema = {p: sum(v) / len(v) for p, v in proto_rates.items()}
    avg_safety = {p: sum(v) / len(v) for p, v in proto_safety.items()}

    json_rate = avg_schema.get("json", 0.0)
    json_safety = avg_safety.get("json", 0.0)

    # Rule 4: evaluator issues make metrics unreliable
    # - model_load_ok=False for all results, OR
    # - runtime_crash_count > 50% of total_tasks for any combination
    all_model_load_failed = all(
        not r.get("model_load_ok", True) for r in results
    )
    high_crash = any(
        r.get("metrics", {}).get("runtime_crash_count", 0) > r.get("total_tasks", 0) / 2
        for r in results
    )
    if all_model_load_failed or high_crash:
        return "FIX_EVALUATOR_FIRST"

    # Rule 2: all below 30%
    if all(rate < 0.30 for rate in avg_schema.values()):
        return "FIX_PROMPT_FIRST"

    # Rule 1: alternative protocol significantly better (>30% improvement)
    for proto in ("tag", "dsl"):
        if proto in avg_schema:
            improvement = avg_schema[proto] - json_rate
            if improvement > 0.30 and avg_safety[proto] >= json_safety:
                if proto == "tag":
                    return "TRY_TAG_PROTOCOL_FOR_P4_2"
                else:
                    return "TRY_DSL_FOR_P4_2"

    # Rule 3: JSON is best
    if json_rate >= max(avg_schema.values()):
        return "KEEP_ACTION_JSON"

    # Fallback
    return "STOP_PROTOCOL_CHANGE"


def main():
    # Issue #32 Final Trust Repair: ensure experiment runs from a clean
    # committed state so experiment_commit_sha is reproducible.
    assert_clean_experiment_state()

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

    # Step 6: Verdict
    print("\n=== Step 6: Verdict ===")
    verdict = compute_verdict(all_results)
    print(f"Verdict: {verdict}")
    # Append verdict to report
    report_path = _REPORT_DIR / "comparison-report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        report += f"\n\n## Verdict\n\n**{verdict}**\n"
        report_path.write_text(report, encoding="utf-8")
    # Write verdict as separate file for machine reading
    (_REPORT_DIR / "verdict.json").write_text(
        json.dumps({"verdict": verdict}, indent=2), encoding="utf-8"
    )

    # Step 7: Artifact manifest (Issue #32)
    print("\n=== Step 7: Artifact Manifest ===")
    manifest = generate_artifact_manifest(_REPORT_DIR)
    (_REPORT_DIR / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {_REPORT_DIR / 'artifact-manifest.json'}")
    print(f"  {manifest['artifact_count']} artifacts catalogued")

    print(f"\nDone. Reports in {_REPORT_DIR}")


if __name__ == "__main__":
    main()
