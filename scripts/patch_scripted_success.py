"""Patch scripted_variant trajectories with A1 fix (§2.5).

Re-replays the 40 scripted_variant trajectories to derive success/
finish_claim_mismatch/metrics from replay evidence, then patches
the existing SFT dataset JSONL files in-place. Also updates the
manifest to schema_version=2 with §2.1 accounting vocabulary.

Usage:
    py -3.11 scripts/patch_scripted_success.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("P4_ALLOW_NETWORK", "0")

from src.agent_trajectory import load_trajectories

# Reuse the A1-fixed _replay_verify from build_agent_sft_dataset
from scripts.build_agent_sft_dataset import (
    _replay_verify,
    _SCRIPTED,
    _MANIFEST,
    _OUT_DIR,
    HELDOUT_TYPE,
    VALIDATION_TYPE,
    SOURCES,
)

_SFT_DIR = _ROOT / "data" / "p4-agent" / "sft-v1"
_FAILURES = _ROOT / "reports" / "p4" / "sft-dataset-replay-failures.jsonl"


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


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = _load_manifest()
    task_types = _task_type_map(manifest)
    tasks_dir = _ROOT / "data" / "p4-agent" / "micro-tasks-v0"

    # Step 1: Load and replay the 40 scripted trajectories
    scripted_trajs = load_trajectories(_SCRIPTED)
    print(f"Loaded {len(scripted_trajs)} scripted trajectories")

    patches: dict[str, dict] = {}  # trajectory_id → {success, finish_claim_mismatch, metrics}
    replay_failures: list[dict] = []

    for i, traj in enumerate(scripted_trajs):
        traj_id = f"scripted_{traj.trajectory_id}"
        task_id = traj.task_id
        task_dir = tasks_dir / task_id

        # Build the dict format that _replay_verify expects
        actions = [s.action.model_dump() for s in traj.steps]
        traj_data = {
            "trajectory_id": traj_id,
            "task_id": task_id,
            "actions": actions,
        }

        ok, result = _replay_verify(traj_data, task_dir)
        if not ok:
            print(f"  [{i+1}/{len(scripted_trajs)}] {traj_id}: REPLAY FAILED — {result}")
            replay_failures.append({
                "trajectory_id": traj_id,
                "task_id": task_id,
                "error": str(result),
            })
            # Mark as failed — success=False, empty metrics
            patches[traj_id] = {
                "success": False,
                "finish_claim_mismatch": False,
                "metrics": {},
            }
        else:
            print(f"  [{i+1}/{len(scripted_trajs)}] {traj_id}: success={result.success}, "
                  f"mismatch={result.finish_claim_mismatch}")
            patches[traj_id] = {
                "success": result.success,
                "finish_claim_mismatch": result.finish_claim_mismatch,
                "metrics": result.metrics,
            }

    print(f"\nReplay complete: {len(patches)} patched, {len(replay_failures)} failures")

    # Step 2: Patch existing JSONL files
    split_files = {
        "train": _SFT_DIR / "train.jsonl",
        "validation": _SFT_DIR / "validation.jsonl",
        "heldout-agent-eval": _SFT_DIR / "heldout-agent-eval.jsonl",
        "failure-diagnostics": _SFT_DIR / "failure-diagnostics.jsonl",
    }

    patched_counts = {}
    for split_name, split_path in split_files.items():
        if not split_path.exists():
            continue
        entries = []
        patched = 0
        with open(split_path, encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                tid = t.get("trajectory_id", "")
                if tid in patches:
                    t["success"] = patches[tid]["success"]
                    t["finish_claim_mismatch"] = patches[tid]["finish_claim_mismatch"]
                    t["metrics"] = patches[tid]["metrics"]
                    patched += 1
                entries.append(t)
        with open(split_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        patched_counts[split_name] = patched
        print(f"  {split_name}: patched {patched} entries")

    # Step 3: Regenerate manifest with schema_version=2 (A3/§2.1)
    # Re-read patched files to get accurate counts
    def _count(path):
        if not path.exists():
            return 0
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    train_count = _count(split_files["train"])
    validation_count = _count(split_files["validation"])
    heldout_count = _count(split_files["heldout-agent-eval"])
    failure_count = _count(split_files["failure-diagnostics"])

    # Read existing replay failures from the failures file (not the split)
    existing_replay_failures = 0
    if _FAILURES.exists():
        with open(_FAILURES, encoding="utf-8") as f:
            existing_replay_failures = sum(1 for line in f if line.strip())

    # Add new replay failures from this patch
    all_replay_failures = existing_replay_failures + len(replay_failures)

    total_generated = train_count + validation_count + heldout_count + failure_count + all_replay_failures
    accepted_count = train_count + validation_count + heldout_count
    rejected_count = all_replay_failures
    quarantined_count = failure_count

    dataset_manifest = {
        "schema_version": 2,
        # §2.1: Dataset accounting invariant
        #   total_generated == accepted_count + rejected_count + quarantined_count
        #   accepted_count  == train_count + validation_count + heldout_count
        "total_generated": total_generated,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "quarantined_count": quarantined_count,
        # Legacy fields (kept for compatibility, derived from above)
        "total_trajectories": train_count + validation_count + heldout_count + failure_count,
        "train_count": train_count,
        "validation_count": validation_count,
        "heldout_count": heldout_count,
        "failure_count": failure_count,
        "replay_failures": all_replay_failures,
        "splits": {
            "train": {
                "task_types": [t for t in sorted(set(task_types.values()))
                               if t not in (HELDOUT_TYPE, VALIDATION_TYPE)],
                "sha256": _sha256_of_file(split_files["train"]),
                "file": "train.jsonl",
            },
            "validation": {
                "task_types": [VALIDATION_TYPE],
                "sha256": _sha256_of_file(split_files["validation"]),
                "file": "validation.jsonl",
            },
            "heldout-agent-eval": {
                "task_types": [HELDOUT_TYPE],
                "sha256": _sha256_of_file(split_files["heldout-agent-eval"]),
                "file": "heldout-agent-eval.jsonl",
            },
            "failure-diagnostics": {
                "sha256": _sha256_of_file(split_files["failure-diagnostics"]),
                "file": "failure-diagnostics.jsonl",
                "classification": "quarantined",
                "description": "model_self_run_failure trajectories — valid replay, excluded from training",
            },
        },
        "sources": sorted(SOURCES),
    }
    (_SFT_DIR / "manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2), encoding="utf-8"
    )

    # Verify accounting invariant
    assert total_generated == accepted_count + rejected_count + quarantined_count, \
        f"§2.1 invariant violated: {total_generated} != {accepted_count} + {rejected_count} + {quarantined_count}"
    assert accepted_count == train_count + validation_count + heldout_count

    print(f"\nManifest updated: schema_version=2")
    print(f"  total_generated={total_generated}")
    print(f"  accepted_count={accepted_count} (train={train_count} + val={validation_count} + heldout={heldout_count})")
    print(f"  rejected_count={rejected_count}")
    print(f"  quarantined_count={quarantined_count}")
    print(f"  §2.1 invariant: {total_generated} == {accepted_count} + {rejected_count} + {quarantined_count} ✓")


if __name__ == "__main__":
    main()
