"""scripts/p3_evaluator_integration_smoke.py -- P3 3-Tier Evaluator smoke test.

Issue #14 Wave 3-D P3.6: Integration Smoke Test.

Runs a single Tier 2 probe against the base Qwen3-0.6B model using a small
stratified subset (8-16 samples per variant_type bucket) of Validation v2.
Outputs a smoke report to ``reports/p3/p3-evaluator-smoke-report.json``.

This proves the end-to-end pipeline works:
  config load → sample load → probe selection → model.generate →
  sandbox.run_pytest → CompositeScore → report write.

Usage
-----
    py -3.11 scripts/p3_evaluator_integration_smoke.py
    py -3.11 scripts/p3_evaluator_integration_smoke.py --probe-size 12
    py -3.11 scripts/p3_evaluator_integration_smoke.py --model models/Qwen3-0.6B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.p3_tier_evaluator import (  # noqa: E402
    PROBE_GENERATION_CONFIG,
    Tier2Probe,
    select_probe_samples,
)
from src.schemas import Sample  # noqa: E402


def _load_validation_samples(path: Path) -> list[Sample]:
    """Load Validation v2 JSONL as Sample objects."""
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            samples.append(Sample.model_validate(rec))
    return samples


def _make_smoke_config(probe_size: int) -> dict:
    """Build a minimal config for the smoke test."""
    return {
        "checkpoint_evaluator": {
            "tier1": {"interval_steps": 50, "metrics": ["train_loss"]},
            "tier2": {
                "interval_epoch_fraction": 0.25,
                "probe_size": probe_size,
                "probe_stratify_by": "variant_type",
                "probe_seed": 42,
                "composite_score": True,
            },
            "tier3": {
                "interval_epochs": 1,
                "full_validation": True,
                "composite_score": True,
            },
        },
        "composite_score": {
            "code_generation_pass_at_1": 0.30,
            "boundary_pass_at_1": 0.15,
            "static_repair_success": 0.20,
            "execution_repair_success": 0.25,
            "hidden_pass_rate": 0.10,
            "hard_constraint": {
                "code_generation_drop_vs_p2_final_max_pct": 3.0,
            },
        },
        "early_stopping": {
            "enabled": True,
            "probe_patience": 4,
            "probe_min_delta": 0.005,
            "full_validation_confirm": True,
            "divergence_nan_inf": True,
            "max_epochs": 3,
        },
        "best_checkpoint": {
            "selection_metric": "full_validation_composite",
            "never_use": ["frozen_v4", "probe"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P3 3-Tier Evaluator integration smoke test",
    )
    parser.add_argument(
        "--model", default="models/Qwen3-0.6B",
        help="Model path (default: models/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--validation-file",
        default="data/p3-curriculum/validation-v2/validation.jsonl",
        help="Validation v2 JSONL path",
    )
    parser.add_argument(
        "--probe-size", type=int, default=8,
        help="Probe size per bucket * 4 = total (default: 8 → 32 samples)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/p3",
        help="Where to write smoke report",
    )
    parser.add_argument(
        "--pytest-timeout", type=float, default=10.0,
        help="Per-sample pytest timeout in seconds",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    validation_file = _ROOT / args.validation_file
    output_dir = _ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1
    if not validation_file.exists():
        print(f"ERROR: validation file not found: {validation_file}", file=sys.stderr)
        return 1

    print(f"=== P3 Evaluator Integration Smoke Test ===")
    print(f"Model: {model_path}")
    print(f"Validation: {validation_file}")
    print(f"Probe size: {args.probe_size} per bucket "
          f"({args.probe_size * 4} total)")

    # Load validation samples
    print("\nLoading validation samples...")
    all_samples = _load_validation_samples(validation_file)
    print(f"Loaded {len(all_samples)} samples")

    # Verify 4-bucket distribution
    bucket_counts: dict[str, int] = {}
    for s in all_samples:
        vt = s.variant_type or "unknown"
        bucket_counts[vt] = bucket_counts.get(vt, 0) + 1
    print(f"Bucket distribution: {bucket_counts}")

    required = {"code", "boundary", "static_repair", "execution_repair"}
    missing = required - set(bucket_counts.keys())
    if missing:
        print(f"ERROR: missing required buckets: {missing}", file=sys.stderr)
        return 1

    # Select probe samples (stratified)
    probe_samples = select_probe_samples(
        all_samples, probe_size=args.probe_size * 4, seed=42,
    )
    print(f"Selected {len(probe_samples)} probe samples")
    probe_buckets: dict[str, int] = {}
    for s in probe_samples:
        vt = s.variant_type or "unknown"
        probe_buckets[vt] = probe_buckets.get(vt, 0) + 1
    print(f"Probe bucket distribution: {probe_buckets}")

    # Load model + tokenizer
    print("\nLoading model + tokenizer...")
    t0 = time.time()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Build config + Tier2Probe
    cfg = _make_smoke_config(probe_size=args.probe_size * 4)
    probe = Tier2Probe(cfg, all_samples, output_dir=output_dir)

    print(f"\nProbe SHA: {probe.probe_sha}")
    print(f"Generation config: {PROBE_GENERATION_CONFIG}")

    # Run probe
    print(f"\n=== Running Tier 2 Probe ({len(probe.probe_samples)} samples) ===")
    t0 = time.time()
    try:
        result = probe.run(
            model, tokenizer, step=0, epoch=0.0,
            pytest_timeout_s=args.pytest_timeout,
        )
    except Exception as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        elapsed = time.time() - t0
        smoke_report = {
            "status": "FAILED",
            "error": str(exc),
            "elapsed_s": round(elapsed, 2),
            "probe_sha": probe.probe_sha,
            "probe_size": len(probe.probe_samples),
            "model": str(model_path),
            "generation_config": PROBE_GENERATION_CONFIG,
        }
        report_path = output_dir / "p3-evaluator-smoke-report.json"
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(smoke_report, fh, indent=2, ensure_ascii=False)
        print(f"Smoke report: {report_path}")
        return 1

    elapsed = time.time() - t0

    print(f"\n=== Smoke Test Results ===")
    print(f"Composite value: {result.composite_value:.4f}")
    print(f"  code_generation_pass_at_1: {result.composite_score.code_generation_pass_at_1:.4f}")
    print(f"  boundary_pass_at_1:        {result.composite_score.boundary_pass_at_1:.4f}")
    print(f"  static_repair_success:     {result.composite_score.static_repair_success:.4f}")
    print(f"  execution_repair_success:  {result.composite_score.execution_repair_success:.4f}")
    print(f"  hidden_pass_rate:          {result.composite_score.hidden_pass_rate:.4f}")
    print(f"Metrics: {json.dumps(result.metrics, indent=2)}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / len(probe.probe_samples):.1f}s/sample)")

    # Write smoke report
    smoke_report = {
        "status": "PASSED",
        "model": str(model_path),
        "validation_file": str(validation_file),
        "probe_sha": probe.probe_sha,
        "probe_size": len(probe.probe_samples),
        "probe_bucket_distribution": probe_buckets,
        "generation_config": PROBE_GENERATION_CONFIG,
        "composite_value": result.composite_value,
        "composite_score": {
            "code_generation_pass_at_1": result.composite_score.code_generation_pass_at_1,
            "boundary_pass_at_1": result.composite_score.boundary_pass_at_1,
            "static_repair_success": result.composite_score.static_repair_success,
            "execution_repair_success": result.composite_score.execution_repair_success,
            "hidden_pass_rate": result.composite_score.hidden_pass_rate,
        },
        "metrics": result.metrics,
        "elapsed_s": round(elapsed, 2),
        "per_sample_s": round(elapsed / len(probe.probe_samples), 2),
    }

    report_path = output_dir / "p3-evaluator-smoke-report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(smoke_report, fh, indent=2, ensure_ascii=False)

    print(f"\nSmoke report: {report_path}")
    print(f"\n=== SMOKE TEST PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
