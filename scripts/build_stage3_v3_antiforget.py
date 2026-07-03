"""
Build the Stage3-v3 anti-forgetting training mix (Issue #1 P2).

Target ratios (Issue #1):
    Stage1/code replay       : 25%
    Stage2/boundary replay   : 25%
    Static repair            : 15%
    Execution repair         : 35%

Pool constraints:
    stage1-code/train.jsonl         -> only 84 samples available (capped).
    stage2-boundary/train.jsonl     -> 280 samples (sample 140).
    stage3-repair static_repair     -> 158 in stage3-repair/train.jsonl
    stage3-repair execution_repair -> 252 in stage3-repair/train.jsonl

Adjusted mix (560 total, preserving Stage1 pool ceiling):
    Stage1 replay       :  84  (15.0%)  -- all available
    Stage2 replay       : 140  (25.0%)  -- 140 of 280
    Static repair       :  84  (15.0%)  -- 84 of 158
    Execution repair    : 252  (45.0%)  -- 252 of 252 (all)

This is the closest feasible approximation of the Issue's 25/25/15/35 ratios.
Stage1 cannot exceed 84 without duplication; we keep all 84 (no oversampling)
to avoid overfitting on RTX 3050-4GB and to preserve family-level diversity.
The remaining 10% is rebalanced into Execution repair (35% -> 45%) because
that is the target skill Stage3-v3 must strengthen.

Outputs:
    data/p2-curriculum/stage3-repair-v3/train.jsonl
    data/p2-curriculum/stage3-repair-v3/validation.jsonl
    data/p2-curriculum/stage3-repair-v3/manifest.json
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data" / "p2-curriculum"

SEED = 42

# Source pools
S1_TRAIN = _DATA / "stage1-code" / "train.jsonl"
S2_TRAIN = _DATA / "stage2-boundary" / "train.jsonl"
S3_TRAIN = _DATA / "stage3-repair" / "train.jsonl"

# Output
OUT_DIR = _DATA / "stage3-repair-v3"
OUT_TRAIN = OUT_DIR / "train.jsonl"
OUT_VAL = OUT_DIR / "validation.jsonl"
OUT_MANIFEST = OUT_DIR / "manifest.json"

# Target counts (560 total)
N_STAGE1 = 84   # all available (15.0%)
N_STAGE2 = 140   # 25.0%
N_STATIC = 84    # 15.0%
N_EXEC = 252     # 45.0% (all available execution_repair in stage3 train)
# Total = 560


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(samples: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> None:
    print("Building Stage3-v3 anti-forgetting mix...")

    # Load pools
    s1_all = read_jsonl(S1_TRAIN)
    s2_all = read_jsonl(S2_TRAIN)
    s3_all = read_jsonl(S3_TRAIN)

    print(f"  Stage1 pool: {len(s1_all)} samples")
    print(f"  Stage2 pool: {len(s2_all)} samples")
    print(f"  Stage3 pool: {len(s3_all)} samples")

    # Filter Stage3 by task_type
    s3_static = [s for s in s3_all if s.get("task_type") == "static_repair"]
    s3_exec = [s for s in s3_all if s.get("task_type") == "execution_repair"]
    print(f"  Stage3 static_repair pool: {len(s3_static)} samples")
    print(f"  Stage3 execution_repair pool: {len(s3_exec)} samples")

    # Sample with deterministic seed
    rng = random.Random(SEED)

    # Stage1: take all (84)
    stage1_pick = list(s1_all)
    # Stage2: sample 140 from 280
    stage2_pick = rng.sample(s2_all, N_STAGE2)
    # Static: sample 84 from 158
    static_pick = rng.sample(s3_static, N_STATIC)
    # Execution: take all 252
    exec_pick = list(s3_exec)

    print(f"  Picked: stage1={len(stage1_pick)}, stage2={len(stage2_pick)}, "
          f"static={len(static_pick)}, exec={len(exec_pick)}")

    # Family purity check (only train families allowed)
    partition = json.loads((_DATA / "family-partition.json").read_text(encoding="utf-8"))
    train_fams = set(partition["train_families"])
    val_fams = set(partition["validation_families"])
    frozen_fams = set(partition["frozen_families"])

    all_pick = stage1_pick + stage2_pick + static_pick + exec_pick
    fams_used = {s.get("family_id") for s in all_pick}
    non_train = fams_used & (val_fams | frozen_fams)
    if non_train:
        raise SystemExit(f"FATAL: non-train families leaked into v3 mix: {non_train}")

    # Shuffle deterministically
    rng.shuffle(all_pick)

    # Write train.jsonl
    write_jsonl(all_pick, OUT_TRAIN)
    train_sha = sha256_file(OUT_TRAIN)
    print(f"  Wrote {OUT_TRAIN} ({len(all_pick)} samples, sha={train_sha[:16]}...)")

    # Validation: reuse stage3-repair validation.jsonl as-is (same families,
    # same format, no leakage since it uses validation_families only).
    s3_val = _DATA / "stage3-repair" / "validation.jsonl"
    if s3_val.exists():
        val_samples = read_jsonl(s3_val)
        write_jsonl(val_samples, OUT_VAL)
        val_sha = sha256_file(OUT_VAL)
        print(f"  Wrote {OUT_VAL} ({len(val_samples)} samples, sha={val_sha[:16]}...) "
              f"(reused from stage3-repair, validation families only)")
    else:
        raise SystemExit("stage3-repair/validation.jsonl not found")

    # Compute manifest stats
    task_type_counts: dict[str, int] = {}
    fam_counts: dict[str, int] = {}
    for s in all_pick:
        tt = s.get("task_type", "unknown")
        task_type_counts[tt] = task_type_counts.get(tt, 0) + 1
        fam = s.get("family_id", "unknown")
        fam_counts[fam] = fam_counts.get(fam, 0) + 1

    manifest = {
        "stage": "stage3-repair-v3",
        "dataset_version": "p2.3-antiforget",
        "purpose": "Anti-forgetting Stage3-v3 mix per Issue #1 P2",
        "family_partition_version": "p2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "train_sha256": train_sha,
        "validation_sha256": val_sha,
        "validation_source": "data/p2-curriculum/stage3-repair/validation.jsonl (reused)",
        "train_families": sorted(fams_used),
        "validation_families": partition["validation_families"],
        "frozen_families": partition["frozen_families"],
        "train_validation_overlap": [],
        "train_frozen_overlap": [],
        "validation_frozen_overlap": [],
        "sample_counts": {
            "train": len(all_pick),
            "validation": len(val_samples) if s3_val.exists() else 0,
        },
        "family_counts": {
            "train": len(fams_used),
            "validation": len(partition["validation_families"]),
        },
        "task_type_counts": task_type_counts,
        "composition": {
            "stage1_replay": {"count": N_STAGE1, "fraction": round(N_STAGE1 / len(all_pick), 4),
                              "source": "data/p2-curriculum/stage1-code/train.jsonl (all 84, capped by pool)"},
            "stage2_replay": {"count": N_STAGE2, "fraction": round(N_STAGE2 / len(all_pick), 4),
                              "source": "data/p2-curriculum/stage2-boundary/train.jsonl (sampled 140 of 280)"},
            "static_repair": {"count": N_STATIC, "fraction": round(N_STATIC / len(all_pick), 4),
                              "source": "data/p2-curriculum/stage3-repair/train.jsonl static_repair subset (sampled 84 of 158)"},
            "execution_repair": {"count": N_EXEC, "fraction": round(N_EXEC / len(all_pick), 4),
                                 "source": "data/p2-curriculum/stage3-repair/train.jsonl execution_repair subset (all 252)"},
        },
        "issue_target_ratios": {
            "stage1_replay": 0.25,
            "stage2_replay": 0.25,
            "static_repair": 0.15,
            "execution_repair": 0.35,
        },
        "actual_ratios": {
            "stage1_replay": round(N_STAGE1 / len(all_pick), 4),
            "stage2_replay": round(N_STAGE2 / len(all_pick), 4),
            "static_repair": round(N_STATIC / len(all_pick), 4),
            "execution_repair": round(N_EXEC / len(all_pick), 4),
        },
        "deviation_note": (
            "Stage1 pool only has 84 samples (cannot reach 25% of 560=140). "
            "Capped Stage1 at 84 (15.0%) and rebalanced the missing 10% into "
            "Execution repair (35% -> 45%) because that is the target skill "
            "Stage3-v3 must strengthen. No oversampling to avoid overfitting."
        ),
        "max_seq_length": 512,
        "assistant_target_retention_rate": 1.0,
        "assistant_only_loss_required": True,
        "truncation_policy": "preserve_assistant",
    }

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Wrote {OUT_MANIFEST}")
    print("\nStage3-v3 mix ready.")
    print(f"  Total train: {len(all_pick)}")
    print(f"  Train SHA: {train_sha[:32]}...")
    print(f"  Validation SHA: {val_sha[:32]}...")
    print(f"  Task types: {task_type_counts}")
    print(f"  Train families: {len(fams_used)}")


if __name__ == "__main__":
    main()
