"""
scripts/compare_runs.py -- Compare evaluation runs with compatibility checks.

P0 fixes:
  - Validates dataset_sha256, sample_count, generation_config, sample_ids,
    task_type distribution, and canary.passed before generating report.
  - Refuses to generate report when results are incompatible.
  - Adds --allow-incompatible for dev use (marks report as INVALID).

Usage
-----
    python scripts/compare_runs.py
        --baseline evaluations/fixed-p0/baseline.json
        --candidate evaluations/fixed-p0/v3-repair.json
        --output evaluations/fixed-p0/comparison.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_FROZEN_EVAL = _ROOT / "data" / "frozen-eval" / "v1" / "test_raw.jsonl"

_METRIC_LABELS: dict[str, str] = {
    "pass_at_1": "Pass@1",
    "syntax_rate": "Syntax Rate",
    "hidden_pass_rate": "Boundary Pass",
    "format_compliance_rate": "Format Compliance",
    "timeout_rate": "Timeout Rate",
    "repair_success_rate": "Repair Success",
    "regression_rate": "Regression Rate",
}

_DISPLAY_ORDER = [
    "pass_at_1",
    "syntax_rate",
    "hidden_pass_rate",
    "format_compliance_rate",
    "timeout_rate",
    "repair_success_rate",
    "regression_rate",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _delta_str(baseline: float, candidate: float) -> str:
    delta = candidate - baseline
    if delta > 0.001:
        return f"+{delta:.3f}"
    elif delta < -0.001:
        return f"{delta:.3f}"
    else:
        return "0.000"


def _extract_sample_ids(data: dict) -> set[str]:
    """Extract sample_ids from outcomes list."""
    ids: set[str] = set()
    for o in data.get("outcomes", []):
        sid = o.get("sample_id")
        if sid:
            ids.add(sid)
    return ids


def _extract_task_type_counts(data: dict) -> dict[str, int]:
    """Extract task_type distribution from either top-level or outcomes."""
    if "task_type_counts" in data:
        return data["task_type_counts"]
    counts: dict[str, int] = {}
    for o in data.get("outcomes", []):
        tt = o.get("task_type", "?")
        counts[tt] = counts.get(tt, 0) + 1
    return counts


def _frozen_eval_sha256() -> str | None:
    """Compute SHA-256 of the frozen eval file, or None if missing."""
    if not _FROZEN_EVAL.exists():
        return None
    sha = hashlib.sha256()
    with _FROZEN_EVAL.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def check_compatibility(baseline: dict, candidate: dict) -> list[str]:
    """Return list of incompatibility reasons. Empty list = compatible."""
    issues: list[str] = []

    # 1. Dataset SHA256
    b_sha = baseline.get("dataset_sha256")
    c_sha = candidate.get("dataset_sha256")
    if b_sha and c_sha and b_sha != c_sha:
        issues.append(
            f"dataset_sha256 mismatch: {b_sha[:16]}... vs {c_sha[:16]}..."
        )
    elif not b_sha or not c_sha:
        issues.append("Missing dataset_sha256 in one or both results")

    # 1a. Validate against frozen eval SHA256
    frozen_sha = _frozen_eval_sha256()
    if frozen_sha is not None:
        if b_sha and b_sha != frozen_sha:
            issues.append(
                f"baseline dataset_sha256 does not match frozen eval: "
                f"{b_sha[:16]}... vs {frozen_sha[:16]}..."
            )
        if c_sha and c_sha != frozen_sha:
            issues.append(
                f"candidate dataset_sha256 does not match frozen eval: "
                f"{c_sha[:16]}... vs {frozen_sha[:16]}..."
            )

    # 2. Sample count
    b_count = baseline.get("sample_count")
    c_count = candidate.get("sample_count")
    if b_count != c_count:
        issues.append(
            f"sample_count mismatch: {b_count} vs {c_count}"
        )

    # 3. Generation config
    b_cfg = baseline.get("generation_config", {})
    c_cfg = candidate.get("generation_config", {})
    if b_cfg != c_cfg:
        issues.append(
            f"generation_config mismatch:\n"
            f"  baseline:  {json.dumps(b_cfg, sort_keys=True)}\n"
            f"  candidate: {json.dumps(c_cfg, sort_keys=True)}"
        )

    # 4. Sample ID sets
    b_ids = _extract_sample_ids(baseline)
    c_ids = _extract_sample_ids(candidate)
    if b_ids != c_ids:
        only_b = b_ids - c_ids
        only_c = c_ids - b_ids
        issues.append(
            f"sample_id sets differ: "
            f"only in baseline ({len(only_b)}), "
            f"only in candidate ({len(only_c)})"
        )

    # 5. Task type distribution
    b_tt = _extract_task_type_counts(baseline)
    c_tt = _extract_task_type_counts(candidate)
    if b_tt != c_tt:
        issues.append(
            f"task_type distribution mismatch:\n"
            f"  baseline:  {json.dumps(b_tt, sort_keys=True)}\n"
            f"  candidate: {json.dumps(c_tt, sort_keys=True)}"
        )

    # 6. Canary passed
    b_canary = baseline.get("canary", {})
    c_canary = candidate.get("canary", {})
    if not b_canary.get("passed"):
        issues.append("baseline canary did not pass (or was skipped)")
    if not c_canary.get("passed"):
        issues.append("candidate canary did not pass (or was skipped)")

    return issues


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    baseline: dict,
    candidate: dict,
    *,
    baseline_label: str = "Baseline",
    candidate_label: str = "Candidate",
    allow_incompatible: bool = False,
) -> str:
    """Generate a Markdown comparison report.

    When ``allow_incompatible`` is True, the report is generated but marked
    as INVALID FOR DIRECT MODEL COMPARISON.
    """
    issues = check_compatibility(baseline, candidate)
    is_compatible = len(issues) == 0

    if not is_compatible and not allow_incompatible:
        # Refuse to generate report
        lines = ["# COMPARISON REFUSED — Incompatible Results", ""]
        lines.append("The following incompatibilities were found:")
        lines.append("")
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. {issue}")
        lines.append("")
        lines.append(
            "Use --allow-incompatible to generate a report marked as "
            "INVALID FOR DIRECT MODEL COMPARISON."
        )
        return "\n".join(lines) + "\n"

    b_metrics = baseline.get("metrics", {})
    c_metrics = candidate.get("metrics", {})

    lines: list[str] = []

    if is_compatible:
        lines.append("# Model Comparison Report")
    else:
        lines.append("# INVALID FOR DIRECT MODEL COMPARISON")
        lines.append("")
        lines.append("> **WARNING**: Results are incompatible. "
                      "Deltas below are NOT meaningful.")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    if not is_compatible:
        lines.append("## Incompatibility Issues")
        lines.append("")
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. {issue}")
        lines.append("")

    lines.append("## Run Info")
    lines.append("")
    lines.append(f"| Field | {baseline_label} | {candidate_label} |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Model | {baseline.get('model_path', '?')} | "
        f"{candidate.get('model_path', '?')} |"
    )
    b_adapter = baseline.get("adapter") or "none"
    c_adapter = candidate.get("adapter") or "none"
    lines.append(f"| Adapter | {b_adapter} | {c_adapter} |")
    lines.append(
        f"| Dataset | {baseline.get('dataset', '?')} | "
        f"{candidate.get('dataset', '?')} |"
    )
    lines.append(
        f"| Dataset SHA256 | {baseline.get('dataset_sha256', '?')[:16]}... | "
        f"{candidate.get('dataset_sha256', '?')[:16]}... |"
    )
    b_total = b_metrics.get("n_total", baseline.get("sample_count", "?"))
    c_total = c_metrics.get("n_total", candidate.get("sample_count", "?"))
    lines.append(f"| Samples | {b_total} | {c_total} |")

    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"| Metric | {baseline_label} | {candidate_label} | Delta |")
    lines.append("|---|---|---|---|")

    for key in _DISPLAY_ORDER:
        label = _METRIC_LABELS.get(key, key)
        b_val = b_metrics.get(key, 0.0)
        c_val = c_metrics.get(key, 0.0)
        delta = _delta_str(b_val, c_val)
        lines.append(f"| {label} | {b_val:.3f} | {c_val:.3f} | {delta} |")

    lines.append("")
    lines.append("## Sample Counts")
    lines.append("")
    lines.append(f"| Category | {baseline_label} | {candidate_label} |")
    lines.append("|---|---|---|")
    for key in ("n_total", "n_generation", "n_repair"):
        label = key.replace("n_", "").capitalize()
        b_val = int(b_metrics.get(key, 0))
        c_val = int(c_metrics.get(key, 0))
        lines.append(f"| {label} | {b_val} | {c_val} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare two model evaluation runs with compatibility checks.",
    )
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline-label", default="Baseline")
    p.add_argument("--candidate-label", default="Candidate")
    p.add_argument(
        "--allow-incompatible",
        action="store_true",
        default=False,
        help="Generate report even when results are incompatible "
             "(marked as INVALID).",
    )
    return p


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    args = _build_parser().parse_args()

    try:
        baseline = _load_json(args.baseline)
    except Exception as exc:
        print(f"ERROR loading baseline: {exc}", file=sys.stderr)
        return 1

    try:
        candidate = _load_json(args.candidate)
    except Exception as exc:
        print(f"ERROR loading candidate: {exc}", file=sys.stderr)
        return 1

    report = generate_report(
        baseline, candidate,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        allow_incompatible=args.allow_incompatible,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Report written to: {output_path.resolve()}")
    print()
    print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
