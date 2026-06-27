"""
scripts/build_dataset.py -- Final dataset builder.

Loads verified Sample JSONL(s), deduplicates, optionally assembles a
curriculum stage mix, splits by family, and writes ChatML JSONL files +
a manifest.

Usage
-----
    python scripts/build_dataset.py
        [--in FILE [FILE ...]]
        [--out-dir DIR]
        [--stage easy|boundary|repair]
        [--seed S]
        [--train F] [--val F] [--test F]
        [--heldout FAM_ID [FAM_ID ...]]

Smoke-run pipeline order (run from project root):

    1. generate_tasks.py     -- create raw task bank
    2. generate_reference.py -- validate references (the output is already
                                verified, so it can feed step 4 directly)
    3. verify_samples.py     -- OPTIONAL re-verify of reference samples; step 2
                                already accepts only self-verifying references,
                                so this is a redundant safety re-check, not a
                                required stage
    4. mutate_code.py        -- produce repair variants
    5. verify_samples.py     -- verify repair variants (REQUIRED: confirms each
                                broken/fixed variant accepts after mutation)
    6. build_dataset.py      -- dedup + split + write ChatML

Exit codes
----------
    0   success
    1   error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._io import load_samples  # noqa: E402,F401  (re-exported for callers/tests)
from src.curriculum import Stage, STAGE_MIX, build_stage_mix  # noqa: E402
from src.dataset_builder import (  # noqa: E402
    DatasetSplit,
    dedup,
    split_by_family,
    write_split,
)
from src.schemas import Sample  # noqa: E402


# ---------------------------------------------------------------------------
# Core helpers (importable for tests)
# ---------------------------------------------------------------------------
# load_samples is provided by scripts._io and re-exported above so the four
# data-factory scripts share ONE JSONL loader implementation.


def build_and_write_dataset(
    samples: list[Sample],
    out_dir: Path,
    *,
    train: float = 0.70,
    val: float = 0.10,
    test: float = 0.20,
    heldout: Optional[set[str]] = None,
    stage: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """Dedup, optionally stage-mix, split, write ChatML JSONL, return manifest.

    Parameters
    ----------
    samples:
        Full pool of verified samples.
    out_dir:
        Directory to write train/validation/test JSONL + manifest.json.
    train, val, test:
        Family-level split fractions (must sum to 1.0).
    heldout:
        Set of family_ids to exclude from train/val/test entirely.
    stage:
        If provided, assemble a curriculum mix first using
        ``curriculum.build_stage_mix``.  One of "easy", "boundary", "repair".
    seed:
        RNG seed for deduplication order-independence and family shuffle.

    Returns
    -------
    dict
        Manifest as produced by ``dataset_builder.write_split``.
    """
    # 1. Dedup
    samples = dedup(samples)

    # 2. Optional stage mix
    if stage is not None:
        if stage not in STAGE_MIX:
            raise ValueError(
                f"Unknown stage {stage!r}. Valid: {list(STAGE_MIX)}"
            )
        total = len(samples)
        samples = build_stage_mix(samples, stage, total, seed=seed)

    # 3. Split by family
    split: DatasetSplit = split_by_family(
        samples,
        train=train,
        val=val,
        test=test,
        heldout_family_ids=heldout,
        seed=seed,
    )

    # 4. Write splits
    manifest = write_split(split, out_dir, seed=seed)

    # 5. Write manifest.json alongside the splits
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build train/val/test JSONL splits from verified samples.",
    )
    p.add_argument(
        "--in", dest="in_paths", nargs="+", required=True,
        help="One or more input JSONL files of verified samples.",
    )
    p.add_argument("--out-dir", default="data/splits",
                   help="Output directory for split JSONL files.")
    p.add_argument(
        "--stage", choices=["easy", "boundary", "repair"], default=None,
        help="Curriculum stage for difficulty-mix sampling.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--train", type=float, default=0.70, help="Train fraction.")
    p.add_argument("--val", type=float, default=0.10, help="Val fraction.")
    p.add_argument("--test", type=float, default=0.20, help="Test fraction.")
    p.add_argument(
        "--heldout", nargs="*", default=None,
        help="Family IDs to exclude from train/val/test.",
    )
    return p


def main() -> int:
    """CLI entry point."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()

    in_paths = [Path(p) for p in args.in_paths]
    out_dir = Path(args.out_dir)

    try:
        samples = load_samples(in_paths)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not samples:
        print("ERROR: no samples loaded", file=sys.stderr)
        return 1

    print(f"build_dataset: {len(samples)} samples loaded")

    heldout_set: Optional[set[str]] = set(args.heldout) if args.heldout else None

    manifest = build_and_write_dataset(
        samples,
        out_dir,
        train=args.train,
        val=args.val,
        test=args.test,
        heldout=heldout_set,
        stage=args.stage,
        seed=args.seed,
    )

    tr = manifest["train"]
    va = manifest["validation"]
    te = manifest["test"]
    print(
        f"build_dataset: train={tr['sample_count']} "
        f"val={va['sample_count']} test={te['sample_count']}"
    )
    print(f"  dataset_hash: {manifest['dataset_hash'][:16]}...")
    print(f"  manifest written -> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
