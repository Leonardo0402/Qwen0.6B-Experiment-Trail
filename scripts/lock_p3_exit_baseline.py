"""Phase A: Lock P3 exit baseline — captures SHAs of all P3 artifacts.

Idempotent: re-running produces identical JSON (except `locked_at`).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT_JSON = _ROOT / "reports/p4/p3-exit-baseline-lock.json"
_OUT_MD = _ROOT / "reports/p4/p3-exit-summary.md"


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def main() -> None:
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    val_v2_path = _ROOT / "data/p3-curriculum/validation-v2/validation.jsonl"
    frozen_v4_path = _ROOT / "data/frozen-eval/v4/test_raw.jsonl"

    adapters = {}
    for cand in ("balanced_limited", "repair_limited"):
        fs_name = cand.replace("_", "-")
        manifest = json.loads((_ROOT / f"data/p3-limited/{fs_name}/manifest.json").read_text(encoding="utf-8"))
        metrics_path = _ROOT / f"adapters/p3/{fs_name}/metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        adapters[cand] = {
            "metrics_path": str(metrics_path.relative_to(_ROOT)),
            "metrics_sha256": _sha256_file(metrics_path),
            "train_hash_in_metrics": metrics["train_hash"],
        }
        assert metrics["train_hash"] == manifest["train_sha256"], f"train_hash mismatch for {cand}"

    bal_manifest = json.loads((_ROOT / "data/p3-limited/balanced-limited/manifest.json").read_text(encoding="utf-8"))
    rep_manifest = json.loads((_ROOT / "data/p3-limited/repair-limited/manifest.json").read_text(encoding="utf-8"))

    lock = {
        "schema_version": 1,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "p3_terminal_verdict": "MBPP_FAMILY_OR_VARIANT_LIMIT",
        "pr_15": {
            "merge_commit_sha": "d91586e0d31214f4ed3edbdce524e6b0e8067070",
            "title": "feat(p3-limited): controlled experiment (1280 samples, Issue #16 fixed) (#15)",
        },
        "p3_limited_datasets": {
            "balanced_limited": {
                "train_sha256": bal_manifest["train_sha256"],
                "manifest_path": "data/p3-limited/balanced-limited/manifest.json",
                "total_samples": 1280,
            },
            "repair_limited": {
                "train_sha256": rep_manifest["train_sha256"],
                "manifest_path": "data/p3-limited/repair-limited/manifest.json",
                "total_samples": 1280,
            },
        },
        "validation_v2": {
            "path": str(val_v2_path.relative_to(_ROOT)),
            "sha256": _sha256_file(val_v2_path),
            "sample_count": _count_jsonl(val_v2_path),
        },
        "frozen_v4": {
            "path": str(frozen_v4_path.relative_to(_ROOT)),
            "sha256": _sha256_file(frozen_v4_path),
            "sample_count": _count_jsonl(frozen_v4_path),
        },
        "adapters": adapters,
        "warnings": [
            "P3-Limited is NOT a formal capability claim — controlled comparison only.",
            "Best honest pass@1 = 0.61% (Repair-Limited); 2300-sample formal threshold unreachable.",
            "No model weights committed; adapter_model.safetensors excluded by .gitignore.",
        ],
    }

    _OUT_JSON.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")

    md = f"""# P3 Exit Baseline Summary

- **Verdict:** {lock['p3_terminal_verdict']}
- **PR #15 merge:** `{lock['pr_15']['merge_commit_sha']}`
- **Locked at:** {lock['locked_at']}

## Locked artifacts

| Artifact | SHA-256 (first 16) | Samples |
|---|---|---|
| validation-v2 | `{lock['validation_v2']['sha256'][:16]}...` | {lock['validation_v2']['sample_count']} |
| frozen-v4 | `{lock['frozen_v4']['sha256'][:16]}...` | {lock['frozen_v4']['sample_count']} |
| balanced-limited train | `{lock['p3_limited_datasets']['balanced_limited']['train_sha256'][:16]}...` | 1280 |
| repair-limited train | `{lock['p3_limited_datasets']['repair_limited']['train_sha256'][:16]}...` | 1280 |

## Warnings

{chr(10).join(f'- {w}' for w in lock['warnings'])}

## Next phase

P4.0 Agentic Coder Foundation begins. P3 artifacts are frozen.
"""
    _OUT_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {_OUT_JSON}")
    print(f"Wrote {_OUT_MD}")


if __name__ == "__main__":
    main()
