"""Compute adapter SHA256 evidence for P2 continual training chain."""
import hashlib
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    stages = [
        ("stage1-code", "adapters/p2/continual/stage1-code-v1"),
        ("stage2-boundary", "adapters/p2/continual/stage2-boundary-v1"),
        ("stage3-repair", "adapters/p2/continual/stage3-repair-v1"),
    ]

    evidence = {}
    for name, dpath in stages:
        base = _ROOT / dpath
        am = base / "adapter_model.safetensors"
        cfg = base / "adapter_config.json"
        tc = base / "training_config.yaml"
        metrics = json.load(open(base / "metrics.json"))

        ev = {
            "path": dpath,
            "weight_file": am.name,
            "size_bytes": am.stat().st_size if am.exists() else 0,
            "sha256": sha256_file(am) if am.exists() else "",
            "config_sha256": sha256_file(cfg) if cfg.exists() else "",
            "training_config_sha256": sha256_file(tc) if tc.exists() else "",
            "parent_adapter_sha256": metrics.get("parent_adapter_sha256"),
            "training_mode": metrics.get("training_mode"),
            "initial_adapter": metrics.get("initial_adapter"),
            "train_hash": metrics.get("train_hash"),
            "eval_hash": metrics.get("eval_hash"),
            "started_at": metrics.get("started_at"),
            "finished_at": metrics.get("finished_at"),
            "train_duration_s": metrics.get("train_duration_s"),
            "trainable_params": metrics.get("trainable_params"),
            "total_params": metrics.get("total_params"),
            "peak_gpu_mib": metrics.get("peak_gpu_mib"),
            "token_audit": metrics.get("token_audit"),
            "assistant_only_loss": metrics.get("assistant_only_loss"),
            "truncation_policy": metrics.get("truncation_policy"),
            "lora_config": metrics.get("lora_config"),
            "max_seq_length": metrics.get("max_seq_length"),
        }
        evidence[name] = ev

        sha_short = ev["sha256"][:16]
        parent = ev.get("parent_adapter_sha256")
        parent_short = parent[:16] if parent else "None"
        print(f"{name}: SHA256={sha_short}... parent={parent_short}")

    # Verify all adapter hashes are different
    hashes = [evidence[s]["sha256"] for s, _ in stages]
    all_diff = len(set(hashes)) == len(hashes)
    print(f"All SHA256 different: {all_diff}")

    # Verify parent chain: parent_adapter_sha256 in metrics is the SHA256
    # of the parent's adapter_config.json (not the weight file)
    chain_ok = True
    for i in range(1, len(stages)):
        parent_name = stages[i - 1][0]
        child_name = stages[i][0]
        child_parent_sha = evidence[child_name].get("parent_adapter_sha256")
        parent_cfg_sha = evidence[parent_name].get("config_sha256")
        if child_parent_sha and parent_cfg_sha:
            match = child_parent_sha == parent_cfg_sha
            print(f"  {child_name} parent_config_sha == {parent_name} config_sha: {match}")
            if not match:
                chain_ok = False

    evidence["_verification"] = {
        "all_adapter_hashes_different": all_diff,
        "parent_chain_verified": chain_ok,
        "verification_time": str(__import__("datetime").datetime.now(__import__("datetime").timezone.utc)),
    }

    out = _ROOT / "reports" / "p2" / "adapter-evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
