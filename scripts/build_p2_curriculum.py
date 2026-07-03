"""
scripts/build_p2_curriculum.py -- P2.2 curriculum integration builder.

Integrates inject_bugs, build_execution_repair, and generate_boundary_variants
to produce the full P2 curriculum under data/p2-curriculum/.

Output structure
---------------
    data/p2-curriculum/
      stage1-code/          (train, validation, manifest, families, token_audit, rejected)
      stage2-boundary/      (same)
      stage3-repair/        (same)
      frozen-eval-v2/       (same)
      family-partition.json

Curriculum ratios (written to manifest)
---------------------------------------
  Stage 1 (code):      Code Gen 70%, Static Repair 20%, Format 10%
  Stage 2 (boundary):  Stage1 replay 30%, Boundary 50%, Static Repair 20%
  Stage 3 (repair):    Stage1 replay 15%, Stage2 replay 20%, Static Repair 20%, Execution Repair 45%

Family partition: train / validation / frozen — three-way disjoint.
Data isolation is prioritised over sample quantity.

Usage
-----
    python scripts/build_p2_curriculum.py --input <sample.jsonl> --seed 42
    python scripts/build_p2_curriculum.py --input <sample.jsonl> --output-dir data/p2-curriculum
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_execution_repair import build_repair_samples  # noqa: E402
from scripts.generate_boundary_variants import generate_boundary_variant  # noqa: E402
from src.schemas import Sample, Verification  # noqa: E402
from src.validators import verify_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
DATASET_VERSION = "p2.2"
MAX_SEQ_LENGTH = 512
ASSISTANT_TARGET_RETENTION_RATE = 1.0
VAL_FAMILY_FRACTION = 0.20
FROZEN_FAMILY_FRACTION = 0.20
MIN_VAL_FAMILIES = 2
MIN_FROZEN_FAMILIES = 2

DEFAULT_OUT_ROOT = _ROOT / "data" / "p2-curriculum"

# Per-stage target ratios.
# Stage 1 (code): Code Gen 70%, Static Repair 20%, Format 10%
# Stage 2 (boundary): Stage1 replay 30%, Boundary 50%, Static Repair 20%
# Stage 3 (repair): Stage1 replay 15%, Stage2 replay 20%, Static Repair 20%, Execution Repair 45%
STAGE_RATIOS: dict[str, dict[str, float]] = {
    "stage1-code": {
        "code_generation": 0.70,
        "static_repair": 0.20,
        "format": 0.10,
    },
    "stage2-boundary": {
        "stage1_replay": 0.30,
        "boundary": 0.50,
        "static_repair": 0.20,
    },
    "stage3-repair": {
        "stage1_replay": 0.15,
        "stage2_replay": 0.20,
        "static_repair": 0.20,
        "execution_repair": 0.45,
    },
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_samples_file(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(Sample.from_json_line(line))
    return samples


def write_samples_jsonl(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(s.to_json_line() + "\n")


def write_rejected_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Sample generation from code_generation samples
# ---------------------------------------------------------------------------

def generate_all_variants(
    samples: list[Sample],
    *,
    timeout_s: float = 10.0,
    seed: int = 42,
) -> tuple[
    list[Sample],       # code_gen (original)
    list[Sample],       # boundary variants
    list[Sample],       # static_repair
    list[Sample],       # execution_repair
    list[dict],         # rejected records
]:
    """Generate all variant samples from code_generation inputs.

    Returns (code_gen, boundary, static_repair, execution_repair, rejected).
    """
    code_gen: list[Sample] = []
    boundary: list[Sample] = []
    static_repair: list[Sample] = []
    execution_repair: list[Sample] = []
    rejected: list[dict] = []

    for sample in samples:
        # 1. Keep original code_generation sample
        code_gen.append(sample)

        # 2. Generate boundary variant
        bv = generate_boundary_variant(sample)
        if bv is not None:
            boundary.append(bv)
        else:
            rejected.append({
                "sample_id": sample.sample_id,
                "family_id": sample.family_id,
                "reason": "boundary_variant_generation_failed",
            })

        # 3. Generate repair samples (static + execution)
        pairs = build_repair_samples(sample, timeout_s=timeout_s, seed=seed)
        if not pairs:
            rejected.append({
                "sample_id": sample.sample_id,
                "family_id": sample.family_id,
                "reason": "no_failing_bug_variants",
            })
        for sr, er in pairs:
            if sr is not None:
                static_repair.append(sr)
            if er is not None:
                execution_repair.append(er)

    return code_gen, boundary, static_repair, execution_repair, rejected


# ---------------------------------------------------------------------------
# Family partition
# ---------------------------------------------------------------------------

def partition_families(
    all_samples: list[Sample],
    *,
    seed: int = SEED,
) -> tuple[set[str], set[str], set[str]]:
    """Partition family_ids into (train, validation, frozen) — disjoint.

    Tries to reuse the existing curriculum-v2 partition if available.
    """
    # Try to reuse existing partition
    existing = _ROOT / "data" / "curriculum-v2" / "family-partition.json"
    all_families = sorted({s.family_id for s in all_samples})
    all_set = set(all_families)

    if existing.exists():
        with existing.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        train_fams = set(doc.get("train_families", [])) & all_set
        val_fams = set(doc.get("validation_families", [])) & all_set
        frozen_fams = set(doc.get("frozen_families", [])) & all_set
        # Only reuse existing partition if it actually covers these families.
        # If most families are uncovered (e.g. new MBPP families), fall through
        # to a fresh partition instead of dumping everything into train.
        covered = train_fams | val_fams | frozen_fams
        if len(covered) < len(all_set) * 0.5:
            pass  # Fall through to fresh partition
        else:
            remaining = all_set - covered
            if remaining:
                train_fams |= remaining
            return train_fams, val_fams, frozen_fams

    # Fresh partition
    rng = random.Random(seed)
    shuffled = list(all_families)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_frozen = max(MIN_FROZEN_FAMILIES, round(n * FROZEN_FAMILY_FRACTION))
    n_val = max(MIN_VAL_FAMILIES, round(n * VAL_FAMILY_FRACTION))
    n_frozen = min(n_frozen, n - n_val - 1)
    n_val = min(n_val, n - n_frozen - 1)

    frozen = set(shuffled[:n_frozen])
    val = set(shuffled[n_frozen:n_frozen + n_val])
    train = set(shuffled[n_frozen + n_val:])
    return train, val, frozen


# ---------------------------------------------------------------------------
# Stage assembly
# ---------------------------------------------------------------------------

def _sample_by_count(pool: list[Sample], n: int, seed: int) -> list[Sample]:
    """Deterministically sample n items from pool."""
    if n <= 0 or not pool:
        return []
    n = min(n, len(pool))
    rng = random.Random(seed)
    return rng.sample(pool, n)


def _filter_by_families(samples: list[Sample], families: set[str]) -> list[Sample]:
    return [s for s in samples if s.family_id in families]


def _format_pool(samples: list[Sample]) -> list[Sample]:
    """Code generation at difficulty 0 (format-focused)."""
    return [s for s in samples if s.difficulty == 0 and s.task_type == "code_generation"]


def _codegen_pool(samples: list[Sample]) -> list[Sample]:
    """Code generation at difficulty >= 1."""
    return [s for s in samples if s.difficulty >= 1 and s.task_type == "code_generation"]


def assemble_stage(
    stage: str,
    train_families: set[str],
    val_families: set[str],
    code_gen: list[Sample],
    boundary: list[Sample],
    static_repair: list[Sample],
    execution_repair: list[Sample],
    stage1_mix: list[Sample],
    stage2_mix: list[Sample],
    seed: int = SEED,
) -> tuple[list[Sample], list[Sample], dict]:
    """Assemble the train+validation mix for one stage.

    Returns (train_samples, val_samples, assembly_report).
    """
    ratios = STAGE_RATIOS[stage]

    def _build_mix(families: set[str]) -> tuple[list[Sample], dict]:
        pools: dict[str, list[Sample]] = {}
        if stage == "stage1-code":
            pools["code_generation"] = _filter_by_families(_codegen_pool(code_gen), families)
            pools["format"] = _filter_by_families(_format_pool(code_gen), families)
            pools["static_repair"] = _filter_by_families(static_repair, families)
        elif stage == "stage2-boundary":
            pools["stage1_replay"] = _filter_by_families(stage1_mix, families)
            pools["boundary"] = _filter_by_families(boundary, families)
            pools["static_repair"] = _filter_by_families(static_repair, families)
        elif stage == "stage3-repair":
            pools["stage1_replay"] = _filter_by_families(stage1_mix, families)
            pools["stage2_replay"] = _filter_by_families(stage2_mix, families)
            pools["static_repair"] = _filter_by_families(static_repair, families)
            pools["execution_repair"] = _filter_by_families(execution_repair, families)

        # Calibrate total to the smallest binding constraint
        candidates: list[float] = []
        for source, pool in pools.items():
            frac = ratios.get(source, 0.0)
            if frac > 0.0 and len(pool) > 0:
                candidates.append(len(pool) / frac)
        total = int(math.floor(min(candidates))) if candidates else 0

        rng = random.Random(seed)
        chosen: list[Sample] = []
        report: dict = {"calibrated_total": total, "buckets": [], "shortfalls": {}}
        for source, pool in pools.items():
            frac = ratios.get(source, 0.0)
            target = round(total * frac) if total > 0 else 0
            if target > len(pool):
                report["shortfalls"][source] = target - len(pool)
                target = len(pool)
            if target > 0:
                picked = rng.sample(pool, target)
            else:
                picked = []
            chosen.extend(picked)
            report["buckets"].append({
                "source": source,
                "fraction": frac,
                "pool_size": len(pool),
                "target_count": round(total * frac) if total > 0 else 0,
                "actual_count": len(picked),
            })
        rng.shuffle(chosen)
        return chosen, report

    train_samples, train_report = _build_mix(train_families)
    val_samples, val_report = _build_mix(val_families)

    report = {
        "stage": stage,
        "train_report": train_report,
        "validation_report": val_report,
    }
    return train_samples, val_samples, report


# ---------------------------------------------------------------------------
# Token audit (character-length based)
# ---------------------------------------------------------------------------

def build_token_audit(samples: list[Sample]) -> dict:
    """Build a simplified token audit using character lengths as proxy."""
    lengths: list[int] = []
    by_task_type: dict[str, list[int]] = {}
    for s in samples:
        # Approximate: total chars of instruction + target_code + tests
        total = len(s.instruction) + len(s.target_code) + len(s.public_tests)
        if s.broken_code:
            total += len(s.broken_code)
        if s.execution_feedback:
            total += len(s.execution_feedback)
        lengths.append(total)
        by_task_type.setdefault(s.task_type, []).append(total)

    def _stats(vals: list[int]) -> dict:
        if not vals:
            return {"count": 0, "mean": 0, "max": 0}
        sv = sorted(vals)
        return {
            "count": len(vals),
            "mean": round(sum(vals) / len(vals), 2),
            "max": sv[-1],
            "p50": sv[len(sv) // 2],
            "p90": sv[int(len(sv) * 0.9)] if len(sv) > 10 else sv[-1],
        }

    return {
        "note": "Character-length based audit (tokenizer not loaded).",
        "overall": _stats(lengths),
        "by_task_type": {k: _stats(v) for k, v in sorted(by_task_type.items())},
    }


# ---------------------------------------------------------------------------
# Stage builder
# ---------------------------------------------------------------------------

def task_type_mix(samples: list[Sample]) -> dict[str, int]:
    out: dict[str, int] = {
        "code_generation": 0,
        "static_repair": 0,
        "execution_repair": 0,
    }
    for s in samples:
        out[s.task_type] = out.get(s.task_type, 0) + 1
    return out


def difficulty_mix(samples: list[Sample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in samples:
        key = str(s.difficulty)
        out[key] = out.get(key, 0) + 1
    return out


def family_set(samples: list[Sample]) -> set[str]:
    return {s.family_id for s in samples}


def build_stage_dir(
    stage: str,
    train_samples: list[Sample],
    val_samples: list[Sample],
    rejected: list[dict],
    train_families: list[str],
    val_families: list[str],
    frozen_families: list[str],
    assembly_report: dict,
    out_root: Path,
) -> dict:
    """Build one stage directory and return its manifest dict."""
    stage_dir = out_root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Write samples
    train_path = stage_dir / "train.jsonl"
    val_path = stage_dir / "validation.jsonl"
    write_samples_jsonl(train_samples, train_path)
    write_samples_jsonl(val_samples, val_path)

    # Write rejected
    rejected_path = stage_dir / "rejected.jsonl"
    write_rejected_jsonl(rejected, rejected_path)

    # Hashes
    train_sha = sha256_file(train_path)
    val_sha = sha256_file(val_path) if val_samples else ""

    # Families
    train_fams = sorted(family_set(train_samples))
    val_fams = sorted(family_set(val_samples))
    families_doc = {
        "stage": stage,
        "dataset_version": DATASET_VERSION,
        "family_count": len(train_fams),
        "family_ids": train_fams,
    }
    with (stage_dir / "families.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(families_doc, fh, indent=2, ensure_ascii=False)

    # Token audit
    audit = build_token_audit(train_samples + val_samples)
    with (stage_dir / "token_audit.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    # Manifest
    manifest = {
        "stage": stage,
        "dataset_version": DATASET_VERSION,
        "family_partition_version": "p2",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "seed": SEED,
        "train_sha256": train_sha,
        "validation_sha256": val_sha,
        "train_families": train_families,
        "validation_families": val_families,
        "frozen_families": frozen_families,
        "train_validation_overlap": [],
        "train_frozen_overlap": [],
        "validation_frozen_overlap": [],
        "sample_counts": {
            "train": len(train_samples),
            "validation": len(val_samples),
        },
        "family_counts": {
            "train": len(train_fams),
            "validation": len(val_fams),
        },
        "task_type_mix": task_type_mix(train_samples),
        "difficulty_mix": difficulty_mix(train_samples),
        "max_seq_length": MAX_SEQ_LENGTH,
        "assistant_target_retention_rate": ASSISTANT_TARGET_RETENTION_RATE,
        "curriculum_ratios": STAGE_RATIOS.get(stage, {}),
        "assembly_report": assembly_report,
    }
    with (stage_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build P2.2 curriculum data (code, boundary, repair stages)."
    )
    p.add_argument("--input", required=True, help="Input JSONL of code_generation samples.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT_ROOT), help="Output root directory.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--timeout", type=float, default=10.0, help="Per-pytest timeout (s).")
    return p


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    in_path = Path(args.input)
    out_root = Path(args.output_dir)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    # 1. Load input samples
    samples = load_samples_file(in_path)
    print(f"Loaded {len(samples)} code_generation samples from {in_path}")

    # 2. Generate all variants
    print("\nGenerating variants...")
    code_gen, boundary, static_repair, execution_repair, rejected = generate_all_variants(
        samples, timeout_s=args.timeout, seed=args.seed
    )
    print(f"  code_generation: {len(code_gen)}")
    print(f"  boundary: {len(boundary)}")
    print(f"  static_repair: {len(static_repair)}")
    print(f"  execution_repair: {len(execution_repair)}")
    print(f"  rejected: {len(rejected)}")

    # 3. Partition families
    print("\nPartitioning families...")
    all_samples = code_gen + boundary + static_repair + execution_repair
    train_fams_set, val_fams_set, frozen_fams_set = partition_families(all_samples, seed=args.seed)
    train_fams = sorted(train_fams_set)
    val_fams = sorted(val_fams_set)
    frozen_fams = sorted(frozen_fams_set)

    print(f"  train families: {len(train_fams)}")
    print(f"  validation families: {len(val_fams)}")
    print(f"  frozen families: {len(frozen_fams)}")

    # Overlap checks
    tv_overlap = train_fams_set & val_fams_set
    tf_overlap = train_fams_set & frozen_fams_set
    vf_overlap = val_fams_set & frozen_fams_set
    if tv_overlap or tf_overlap or vf_overlap:
        print("FATAL: Family overlap detected!", file=sys.stderr)
        return 1

    # 4. Write family-partition.json
    out_root.mkdir(parents=True, exist_ok=True)
    partition_doc = {
        "seed": args.seed,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "train_families": train_fams,
        "validation_families": val_fams,
        "frozen_families": frozen_fams,
        "train_validation_overlap": [],
        "train_frozen_overlap": [],
        "validation_frozen_overlap": [],
    }
    with (out_root / "family-partition.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(partition_doc, fh, indent=2, ensure_ascii=False)

    # 5. Build Stage 1 (code)
    print("\nBuilding Stage 1 (code)...")
    s1_train, s1_val, s1_report = assemble_stage(
        "stage1-code", train_fams_set, val_fams_set,
        code_gen, boundary, static_repair, execution_repair,
        [], [], seed=args.seed,
    )
    # Stage 1 mix = all samples used in stage 1 (for replay in later stages)
    stage1_mix = s1_train + s1_val
    s1_manifest = build_stage_dir(
        "stage1-code", s1_train, s1_val, rejected,
        train_fams, val_fams, frozen_fams, s1_report, out_root
    )
    print(f"  train: {len(s1_train)}, val: {len(s1_val)}")

    # 6. Build Stage 2 (boundary)
    print("\nBuilding Stage 2 (boundary)...")
    s2_train, s2_val, s2_report = assemble_stage(
        "stage2-boundary", train_fams_set, val_fams_set,
        code_gen, boundary, static_repair, execution_repair,
        stage1_mix, [], seed=args.seed,
    )
    stage2_mix = s2_train + s2_val
    s2_manifest = build_stage_dir(
        "stage2-boundary", s2_train, s2_val, rejected,
        train_fams, val_fams, frozen_fams, s2_report, out_root
    )
    print(f"  train: {len(s2_train)}, val: {len(s2_val)}")

    # 7. Build Stage 3 (repair)
    print("\nBuilding Stage 3 (repair)...")
    s3_train, s3_val, s3_report = assemble_stage(
        "stage3-repair", train_fams_set, val_fams_set,
        code_gen, boundary, static_repair, execution_repair,
        stage1_mix, stage2_mix, seed=args.seed,
    )
    s3_manifest = build_stage_dir(
        "stage3-repair", s3_train, s3_val, rejected,
        train_fams, val_fams, frozen_fams, s3_report, out_root
    )
    print(f"  train: {len(s3_train)}, val: {len(s3_val)}")

    # 8. Build frozen-eval-v2 (from frozen families)
    print("\nBuilding frozen-eval-v2...")
    frozen_code_gen = _filter_by_families(code_gen, frozen_fams_set)
    frozen_boundary = _filter_by_families(boundary, frozen_fams_set)
    frozen_static = _filter_by_families(static_repair, frozen_fams_set)
    frozen_exec = _filter_by_families(execution_repair, frozen_fams_set)
    frozen_all = frozen_code_gen + frozen_boundary + frozen_static + frozen_exec
    frozen_manifest = build_stage_dir(
        "frozen-eval-v2", frozen_all, [], rejected,
        train_fams, val_fams, frozen_fams, {}, out_root
    )
    print(f"  frozen samples: {len(frozen_all)}")

    # 9. Top-level summary
    all_ok = True
    for stage, manifest in [
        ("stage1-code", s1_manifest),
        ("stage2-boundary", s2_manifest),
        ("stage3-repair", s3_manifest),
        ("frozen-eval-v2", frozen_manifest),
    ]:
        train_path = out_root / stage / "train.jsonl"
        exists = train_path.exists() and train_path.stat().st_size > 0
        train_fams_in_stage = set(
            json.loads((out_root / stage / "families.json").read_text(encoding="utf-8")).get("family_ids", [])
        )
        # Check train families are subset of train_fams
        if stage != "frozen-eval-v2":
            subset_ok = train_fams_in_stage.issubset(train_fams_set)
        else:
            subset_ok = train_fams_in_stage.issubset(frozen_fams_set)
        all_ok = all_ok and exists and subset_ok
        print(f"\n  [{stage}] exists={exists}, families_subset_ok={subset_ok}")
        print(f"          train={manifest['sample_counts']['train']}, "
              f"val={manifest['sample_counts']['validation']}")

    print(f"\n  overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
