"""Compute adapter SHA256 evidence for P2 continual training chain.

Issue #1 P0-6 requires splitting the legacy ``parent_adapter_sha256`` field
(which was actually the parent's *config* SHA) into two explicit fields:

- ``parent_adapter_weight_sha256``  -> SHA256 of parent's adapter_model.safetensors
- ``parent_adapter_config_sha256``  -> SHA256 of parent's adapter_config.json

The legacy ``parent_adapter_sha256`` field is preserved as an alias of
``parent_adapter_config_sha256`` for backward compatibility with existing
reports, but new code MUST use the explicit *_weight_* / *_config_* fields.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
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
        ("stage1-code", "adapters/p2/continual/stage1-code-v2"),
        ("stage2-boundary", "adapters/p2/continual/stage2-boundary-v2"),
        ("stage3-repair", "adapters/p2/continual/stage3-repair-v2"),
        # Issue #1 P1: Independent Stage3 (from Base, no parent)
        ("independent-stage3", "adapters/p2/independent/stage3-repair-v2"),
        # Issue #1 P2: Anti-forget Stage3-v3 (from Stage2-v2)
        ("stage3-v3-antiforget", "adapters/p2/continual/stage3-repair-v3"),
    ]

    evidence = {}
    for name, dpath in stages:
        base = _ROOT / dpath
        am = base / "adapter_model.safetensors"
        cfg = base / "adapter_config.json"
        tc = base / "training_config.yaml"
        metrics = json.load(open(base / "metrics.json"))

        weight_sha = sha256_file(am) if am.exists() else ""
        config_sha = sha256_file(cfg) if cfg.exists() else ""
        training_cfg_sha = sha256_file(tc) if tc.exists() else ""

        # Legacy parent_adapter_sha256 in metrics.json is actually the parent's
        # adapter_config.json SHA. Split into explicit weight + config fields.
        legacy_parent_sha = metrics.get("parent_adapter_sha256")
        parent_adapter_path = metrics.get("initial_adapter")
        parent_weight_sha = ""
        parent_config_sha = legacy_parent_sha or ""
        if parent_adapter_path:
            parent_dir = _ROOT / parent_adapter_path
            parent_am = parent_dir / "adapter_model.safetensors"
            parent_cfg = parent_dir / "adapter_config.json"
            if parent_am.exists():
                parent_weight_sha = sha256_file(parent_am)
            if parent_cfg.exists():
                parent_config_sha = sha256_file(parent_cfg)

        ev = {
            "path": dpath,
            "weight_file": am.name,
            "size_bytes": am.stat().st_size if am.exists() else 0,
            # Self hashes
            "sha256": weight_sha,                       # weight SHA
            "weight_sha256": weight_sha,                 # explicit alias
            "config_sha256": config_sha,
            "training_config_sha256": training_cfg_sha,
            # Parent chain (Issue #1 P0-6 split)
            "parent_adapter_weight_sha256": parent_weight_sha or None,
            "parent_adapter_config_sha256": parent_config_sha or None,
            # Legacy alias (deprecated; equals parent_adapter_config_sha256)
            "parent_adapter_sha256": parent_config_sha or None,
            "parent_adapter_path": parent_adapter_path,
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
        parent_w = ev.get("parent_adapter_weight_sha256")
        parent_c = ev.get("parent_adapter_config_sha256")
        print(f"{name}: weight_sha={sha_short}... "
              f"parent_weight={parent_w[:16] if parent_w else 'None'}... "
              f"parent_config={parent_c[:16] if parent_c else 'None'}...")

    # Verify all adapter WEIGHT hashes are different
    hashes = [evidence[s]["sha256"] for s, _ in stages]
    all_diff = len(set(hashes)) == len(hashes)
    print(f"All weight SHA256 different: {all_diff}")

    # Verify parent chain on BOTH weight and config SHAs.
    chain_ok = True
    for i in range(1, len(stages)):
        parent_name = stages[i - 1][0]
        child_name = stages[i][0]
        child = evidence[child_name]
        parent = evidence[parent_name]

        # Config chain (was the only check before P0-6)
        child_parent_cfg = child.get("parent_adapter_config_sha256")
        parent_cfg = parent.get("config_sha256")
        cfg_match = (child_parent_cfg == parent_cfg) if (child_parent_cfg and parent_cfg) else None

        # Weight chain (NEW in P0-6)
        child_parent_w = child.get("parent_adapter_weight_sha256")
        parent_w = parent.get("weight_sha256")
        weight_match = (child_parent_w == parent_w) if (child_parent_w and parent_w) else None

        print(f"  {child_name} -> {parent_name}: "
              f"config_match={cfg_match} weight_match={weight_match}")
        if cfg_match is False or weight_match is False:
            chain_ok = False

    evidence["_verification"] = {
        "all_adapter_weight_hashes_different": all_diff,
        "parent_chain_verified": chain_ok,
        "verification_time": datetime.now(timezone.utc).isoformat(),
        "field_note": (
            "Issue #1 P0-6: parent_adapter_sha256 was actually the parent's "
            "config SHA. Split into parent_adapter_weight_sha256 "
            "(adapter_model.safetensors) and parent_adapter_config_sha256 "
            "(adapter_config.json). Legacy parent_adapter_sha256 kept as "
            "alias of parent_adapter_config_sha256 for backward compat."
        ),
    }

    out = _ROOT / "reports" / "p2" / "adapter-evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
