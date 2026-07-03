"""Generate the unified Full-576 comparison Markdown report.

Reads three JSON inputs produced by upstream P2 scripts:
  - evaluations/p2/full576-comparison.json   (compare_p2_evals.py)
  - reports/p2/full576-paired-stats.json      (compute_paired_stats.py)
  - reports/p2/router-analysis.json           (compute_router_analysis.py)

Writes:
  - reports/p2/p2-full576-comparison-report.md

If any of the three input files is missing, prints an error to stderr and
exits with code 1 (does not crash with an unhandled exception).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = _ROOT / "evaluations" / "p2"
_REPORTS = _ROOT / "reports" / "p2"

COMPARISON_PATH = _EVAL_DIR / "full576-comparison.json"
PAIRED_STATS_PATH = _REPORTS / "full576-paired-stats.json"
ROUTER_ANALYSIS_PATH = _REPORTS / "router-analysis.json"
OUTPUT_PATH = _REPORTS / "p2-full576-comparison-report.md"

# Canonical model order per Issue #6 P3
MODELS = [
    ("full576-base", "Base"),
    ("full576-stage2-boundary", "Stage2-v2"),
    ("full576-stage3-repair", "Stage3-v2-Continual"),
    ("full576-independent-stage3", "Stage3-Independent"),
    ("full576-stage3-v3-antiforget", "Stage3-v3-Antiforget"),
]

TASK_TYPES = ["code_generation", "static_repair", "execution_repair"]


def load_json(p: Path) -> dict:
    return json.load(open(p))


def fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def generate_report(comparison: dict, paired_stats: dict, router_analysis: dict) -> str:
    lines: list[str] = []

    # 1. Header
    lines.append("# P2 Full-576 Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # 2. Evaluation Setup
    common_n = paired_stats.get("common_sample_count", 576)
    lines.append("## Evaluation Setup")
    lines.append("")
    lines.append("- Dataset: data/p2-curriculum/frozen-eval-v2/test_raw.jsonl")
    lines.append("- Samples: 576")
    lines.append("- Families: 75")
    lines.append("- Task types: code_generation (140), static_repair (218), execution_repair (218)")
    lines.append(f"- Common sample count (paired-stats): {common_n}")
    lines.append("")

    # 3. Overall Metrics
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Model | Pass@1 | Syntax | Repair | Hidden | Format | Timeout | Family Pass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, label in MODELS:
        if key not in comparison:
            continue
        m = comparison[key].get("metrics", {})
        fp = comparison[key].get("family_pass_count", 0)
        ft = comparison[key].get("family_total", 0)
        fam_rate = fp / ft if ft else 0.0
        lines.append(
            f"| {label} | {fmt_pct(m.get('pass_at_1', 0.0))} | "
            f"{fmt_pct(m.get('syntax_rate', 0.0))} | "
            f"{fmt_pct(m.get('repair_success_rate', 0.0))} | "
            f"{fmt_pct(m.get('hidden_pass_rate', 0.0))} | "
            f"{fmt_pct(m.get('format_compliance_rate', 0.0))} | "
            f"{fmt_pct(m.get('timeout_rate', 0.0))} | "
            f"{fmt_pct(fam_rate)} |"
        )
    lines.append("")

    # 4. Per-Task-Type Breakdown
    lines.append("## Per-Task-Type Breakdown")
    lines.append("")
    for t in TASK_TYPES:
        lines.append(f"### {t}")
        lines.append("")
        lines.append("| Model | Total | Passed | Rate | Syntax | Format |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for key, label in MODELS:
            if key not in comparison:
                continue
            tt = comparison[key].get("per_task_type", {}).get(t, {})
            total = tt.get("total", 0)
            passed = tt.get("passed", 0)
            syn = tt.get("syntax_ok", 0)
            fmt = tt.get("format_ok", 0)
            rate = passed / total if total else 0.0
            syn_rate = syn / total if total else 0.0
            fmt_rate = fmt / total if total else 0.0
            lines.append(
                f"| {label} | {total} | {passed} | {fmt_pct(rate)} | "
                f"{fmt_pct(syn_rate)} | {fmt_pct(fmt_rate)} |"
            )
        lines.append("")

    # 5. Family-Level Pass
    lines.append("## Family-Level Pass")
    lines.append("")
    lines.append("| Model | Families Passed | Total Families | Rate |")
    lines.append("|---|---:|---:|---:|")
    for key, label in MODELS:
        if key not in comparison:
            continue
        fp = comparison[key].get("family_pass_count", 0)
        ft = comparison[key].get("family_total", 0)
        rate = fp / ft if ft else 0.0
        lines.append(f"| {label} | {fp} | {ft} | {fmt_pct(rate)} |")
    lines.append("")
    # Stage3-v2-Continual vs Base family delta (from comparison.json if available)
    base = comparison.get("full576-base", {})
    s3 = comparison.get("full576-stage3-repair", {})
    if base and s3:
        base_fam = base.get("family_pass_count", 0)
        s3_fam = s3.get("family_pass_count", 0)
        delta = s3_fam - base_fam
        lines.append(f"- Stage3-v2-Continual vs Base family delta: {delta:+d}")
        lines.append("")

    # 6. Paired Statistics Summary
    lines.append("## Paired Statistics Summary")
    lines.append("")
    lines.append("| Pair | N | Win | Loss | Unchanged | Delta | McNemar b/c | p (2-sided) | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for p in paired_stats.get("pair_comparisons", []):
        pair = p.get("pair", [])
        if len(pair) != 2:
            continue
        a, b = pair[0], pair[1]
        s = p.get("sample_compare", {})
        ci = s.get("bootstrap_95ci", [0.0, 0.0])
        delta = s.get("delta", 0.0)
        lines.append(
            f"| {a} → {b} | {s.get('n_compared', 0)} | {s.get('win', 0)} | "
            f"{s.get('loss', 0)} | {s.get('unchanged', 0)} | "
            f"{delta * 100:+.1f}% | "
            f"{s.get('mcnemar_b', 0)}/{s.get('mcnemar_c', 0)} | "
            f"{s.get('mcnemar_p_two_sided', 1.0):.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )
    lines.append("")

    # 7. Bug-Type Repair Success Rate
    lines.append("## Bug-Type Repair Success Rate")
    lines.append("")
    bug_stats = paired_stats.get("per_model_bug_type_repair", {})
    bug_types = sorted({bt for d in bug_stats.values() for bt in d})
    header_labels = [label for _, label in MODELS]
    lines.append("| Bug Type | " + " | ".join(header_labels) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(header_labels)) + "|")
    for bt in bug_types:
        row = f"| {bt} |"
        for key, _ in MODELS:
            s = bug_stats.get(key, {}).get(bt, {"total": 0, "passed": 0, "pass_rate": 0.0})
            total = s.get("total", 0)
            if total == 0:
                row += " - |"
            else:
                passed = s.get("passed", 0)
                rate = s.get("pass_rate", 0.0)
                row += f" {passed}/{total} ({fmt_pct(rate)}) |"
        lines.append(row)
    lines.append("")

    # 8. Router Feasibility Summary
    lines.append("## Router Feasibility Summary")
    lines.append("")
    lines.append("| Model/Router | Type | Overall | Family | CodeGen | StaticRepair | ExecRepair | Lift vs Best |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in router_analysis.get("comparison_table", []):
        lines.append(
            f"| {row.get('name', '?')} | {row.get('type', '?')} | "
            f"{fmt_pct(row.get('overall_pass', 0.0))} | "
            f"{fmt_pct(row.get('family_pass', 0.0))} | "
            f"{fmt_pct(row.get('code_generation_pass', 0.0))} | "
            f"{fmt_pct(row.get('static_repair_pass', 0.0))} | "
            f"{fmt_pct(row.get('execution_repair_pass', 0.0))} | "
            f"{row.get('lift_vs_best_single', 0.0):+.4f} |"
        )
    lines.append("")

    # 9. P3 Decision Gate
    gate = router_analysis.get("decision_gate", {})
    verdict = gate.get("verdict", "UNKNOWN")
    reason = gate.get("reason", "")
    c = gate.get("criteria", {})
    lines.append("## P3 Decision Gate")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(reason)
    lines.append("")
    lines.append("### Gate Criteria")
    lines.append("")
    lines.append("| Criterion | Value | Threshold | Met |")
    lines.append("|-----------|-------|-----------|-----|")
    oracle_lift = c.get("oracle_lift", 0.0)
    deployable_lift = c.get("deployable_lift", 0.0)
    mcnemar_p = c.get("deployable_mcnemar_p", 1.0)
    ci = c.get("deployable_ci_95", [0.0, 0.0])
    lines.append(
        f"| Oracle lift | {oracle_lift * 100:.1f}pp | >= 5.0pp | "
        f"{'YES' if c.get('oracle_meaningful', False) else 'NO'} |"
    )
    lines.append(
        f"| Deployable lift | {deployable_lift * 100:.1f}pp | >= 5.0pp | "
        f"{'YES' if c.get('deployable_meaningful', False) else 'NO'} |"
    )
    lines.append(
        f"| McNemar p | {mcnemar_p:.4f} | < 0.05 | "
        f"{'YES' if mcnemar_p < 0.05 else 'NO'} |"
    )
    lines.append(
        f"| 95% CI | [{ci[0]:+.4f}, {ci[1]:+.4f}] | lower > 0 | "
        f"{'YES' if ci[0] > 0 else 'NO'} |"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    inputs = [
        (COMPARISON_PATH, "full576-comparison.json"),
        (PAIRED_STATS_PATH, "full576-paired-stats.json"),
        (ROUTER_ANALYSIS_PATH, "router-analysis.json"),
    ]
    missing = [(p, name) for p, name in inputs if not p.exists()]
    if missing:
        for p, name in missing:
            print(f"ERROR: missing input file: {name} ({p})", file=sys.stderr)
        return 1

    comparison = load_json(COMPARISON_PATH)
    paired_stats = load_json(PAIRED_STATS_PATH)
    router_analysis = load_json(ROUTER_ANALYSIS_PATH)

    _REPORTS.mkdir(parents=True, exist_ok=True)
    md = generate_report(comparison, paired_stats, router_analysis)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
