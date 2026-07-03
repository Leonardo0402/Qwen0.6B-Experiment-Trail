"""
scripts/audit_p2_dataset.py -- P2 curriculum dataset auditor.

Scans ``data/p2-curriculum/`` and produces a structured audit covering:
  - Per-stage sample counts (train / validation)
  - Per-stage family counts
  - ``task_type`` distribution per stage
  - ``difficulty`` distribution per stage
  - Three-way family disjointness check (train / validation / frozen)

Output: ``reports/p2/dataset-audit.json``

The script is safe to run before the dataset has been built: when the data
directory or stage files are missing, the report records ``exists=False`` /
zero counts rather than raising.

Usage
-----
    python scripts/audit_p2_dataset.py
    python scripts/audit_p2_dataset.py --data-dir data/p2-curriculum \\
                                       --out reports/p2/dataset-audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so this script can be run directly (python scripts/audit_*.py)
# and still import the project's ``src`` package.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = _ROOT / "data" / "p2-curriculum"
DEFAULT_OUT_PATH = _ROOT / "reports" / "p2" / "dataset-audit.json"

STAGES = ("stage1-code", "stage2-boundary", "stage3-repair", "frozen-eval-v2")


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    """Yield parsed JSON objects from a JSONL file (blank lines skipped)."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_samples(path: Path) -> tuple[list[Sample], int]:
    """Load samples from a JSONL file.

    Returns ``(samples, parse_error_count)``.  Lines that fail Sample
    validation are counted but otherwise skipped, so a single malformed
    record does not abort the whole audit.
    """
    samples: list[Sample] = []
    parse_errors = 0
    if not path.exists():
        return samples, 0
    for raw in _iter_jsonl(path):
        try:
            samples.append(Sample(**raw))
        except Exception:  # noqa: BLE001
            parse_errors += 1
    return samples, parse_errors


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def task_type_distribution(samples: list[Sample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in samples:
        out[s.task_type] = out.get(s.task_type, 0) + 1
    return out


def difficulty_distribution(samples: list[Sample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in samples:
        key = str(s.difficulty)
        out[key] = out.get(key, 0) + 1
    return out


def family_set(samples: list[Sample]) -> set[str]:
    return {s.family_id for s in samples}


# ---------------------------------------------------------------------------
# Per-stage audit
# ---------------------------------------------------------------------------

def audit_stage(data_dir: Path, stage: str) -> dict:
    """Audit one stage directory and return its stats dict."""
    stage_dir = data_dir / stage
    train_path = stage_dir / "train.jsonl"
    val_path = stage_dir / "validation.jsonl"
    manifest_path = stage_dir / "manifest.json"

    train_samples, train_errors = load_samples(train_path)
    val_samples, val_errors = load_samples(val_path)

    manifest: dict = {}
    if manifest_path.exists():
        try:
            with manifest_path.open(encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception:  # noqa: BLE001
            manifest = {}

    train_fams = family_set(train_samples)
    val_fams = family_set(val_samples)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(_ROOT))
        except ValueError:
            return str(p)

    return {
        "stage": stage,
        "stage_dir": _rel(stage_dir),
        "exists": stage_dir.exists(),
        "sample_counts": {
            "train": len(train_samples),
            "validation": len(val_samples),
            "total": len(train_samples) + len(val_samples),
        },
        "family_counts": {
            "train": len(train_fams),
            "validation": len(val_fams),
        },
        "family_ids": {
            "train": sorted(train_fams),
            "validation": sorted(val_fams),
        },
        "task_type_distribution": {
            "train": task_type_distribution(train_samples),
            "validation": task_type_distribution(val_samples),
        },
        "difficulty_distribution": {
            "train": difficulty_distribution(train_samples),
            "validation": difficulty_distribution(val_samples),
        },
        "parse_errors": {
            "train": train_errors,
            "validation": val_errors,
        },
        "manifest_present": bool(manifest),
    }


# ---------------------------------------------------------------------------
# Family partition (three-way disjointness)
# ---------------------------------------------------------------------------

def audit_family_partition(data_dir: Path) -> dict:
    """Read ``family-partition.json`` and verify three-way disjointness."""
    partition_path = data_dir / "family-partition.json"
    if not partition_path.exists():
        return {
            "partition_file_present": False,
            "train_families": [],
            "validation_families": [],
            "frozen_families": [],
            "family_counts": {
                "train": 0,
                "validation": 0,
                "frozen": 0,
            },
            "train_validation_overlap": [],
            "train_frozen_overlap": [],
            "validation_frozen_overlap": [],
            "three_way_disjoint": False,
        }

    with partition_path.open(encoding="utf-8") as fh:
        doc = json.load(fh)

    train = set(doc.get("train_families", []))
    val = set(doc.get("validation_families", []))
    frozen = set(doc.get("frozen_families", []))

    tv_overlap = sorted(train & val)
    tf_overlap = sorted(train & frozen)
    vf_overlap = sorted(val & frozen)
    disjoint = not (tv_overlap or tf_overlap or vf_overlap)

    return {
        "partition_file_present": True,
        "train_families": sorted(train),
        "validation_families": sorted(val),
        "frozen_families": sorted(frozen),
        "family_counts": {
            "train": len(train),
            "validation": len(val),
            "frozen": len(frozen),
        },
        "train_validation_overlap": tv_overlap,
        "train_frozen_overlap": tf_overlap,
        "validation_frozen_overlap": vf_overlap,
        "three_way_disjoint": disjoint,
    }


# ---------------------------------------------------------------------------
# Top-level audit
# ---------------------------------------------------------------------------

def audit_dataset(data_dir: Path) -> dict:
    """Run the full P2 dataset audit and return the report dict."""
    stages_report: dict[str, dict] = {}
    for stage in STAGES:
        stages_report[stage] = audit_stage(data_dir, stage)

    partition_report = audit_family_partition(data_dir)

    total_train = sum(s["sample_counts"]["train"] for s in stages_report.values())
    total_val = sum(s["sample_counts"]["validation"] for s in stages_report.values())

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(_ROOT))
        except ValueError:
            return str(p)

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "data_dir": _rel(data_dir),
        "data_dir_exists": data_dir.exists(),
        "totals": {
            "train": total_train,
            "validation": total_val,
            "all": total_train + total_val,
        },
        "stages": stages_report,
        "family_partition": partition_report,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit P2 curriculum dataset (sample/family/task_type/difficulty stats).",
    )
    p.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR),
        help="P2 curriculum data root (default: data/p2-curriculum).",
    )
    p.add_argument(
        "--out", default=str(DEFAULT_OUT_PATH),
        help="Output JSON path (default: reports/p2/dataset-audit.json).",
    )
    return p


def main() -> int:
    """CLI entry point."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    report = audit_dataset(data_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # Console summary
    print(f"P2 dataset audit -> {out_path}")
    print(f"data_dir_exists: {report['data_dir_exists']}")
    print(f"totals: train={report['totals']['train']}, "
          f"validation={report['totals']['validation']}, "
          f"all={report['totals']['all']}")
    for stage, info in report["stages"].items():
        sc = info["sample_counts"]
        fc = info["family_counts"]
        print(f"  [{stage}] exists={info['exists']}  "
              f"train={sc['train']}, val={sc['validation']}  "
              f"families(train)={fc['train']}")
    fp = report["family_partition"]
    print(f"  family_partition present={fp['partition_file_present']}  "
          f"three_way_disjoint={fp['three_way_disjoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
