# Task 3: Write generate_full576_report.py

## Context

Issue #6 requires a comprehensive Full-576 comparison report that unifies data
from three JSON sources into a single Markdown document. This script reads the
outputs of `compare_p2_evals.py`, `compute_paired_stats.py`, and
`compute_router_analysis.py` and produces `p2-full576-comparison-report.md`.

This is a **new file** — `scripts/generate_full576_report.py` does not exist yet.
Follow the patterns in `scripts/generate_p2_reports.py` (load_json, fmt_pct, etc.).

## Deliverable

1. Create `scripts/generate_full576_report.py`
2. Create `tests/test_generate_full576_report.py`
3. Run the test suite and confirm all tests pass
4. Commit with message: `feat(scripts): add generate_full576_report.py for Full-576 comparison report`

## Input Files

All three input files are JSON. They may not exist yet (evaluations are still
running). The script must exit with code 1 and a clear message if any required
input is missing.

### 1. `evaluations/p2/full576-comparison.json`

Produced by `scripts/compare_p2_evals.py`. Structure:

```json
{
  "full576-base": {
    "label": "Base",
    "metrics": {
      "pass_at_1": 0.0,
      "syntax_rate": 0.0,
      "repair_success_rate": 0.0,
      "hidden_pass_rate": 0.0,
      "format_compliance_rate": 0.0,
      "timeout_rate": 0.0
    },
    "per_task_type": {
      "code_generation": {"total": 140, "passed": 0, "syntax_ok": 0, "format_ok": 0},
      "static_repair": {"total": 218, "passed": 0, "syntax_ok": 0, "format_ok": 0},
      "execution_repair": {"total": 218, "passed": 0, "syntax_ok": 0, "format_ok": 0}
    },
    "family_pass_count": 0,
    "family_total": 75
  },
  "full576-stage2-boundary": { ... same structure, "label": "Stage2-v2" },
  "full576-stage3-repair": { ... "label": "Stage3-v2-Continual" },
  "full576-independent-stage3": { ... "label": "Stage3-Independent" },
  "full576-stage3-v3-antiforget": { ... "label": "Stage3-v3-Antiforget" }
}
```

### 2. `reports/p2/full576-paired-stats.json`

Produced by `scripts/compute_paired_stats.py`. Structure:

```json
{
  "models": ["full576-base", "full576-stage2-boundary", ...],
  "common_sample_count": 576,
  "pair_comparisons": [
    {
      "pair": ["full576-base", "full576-stage2-boundary"],
      "sample_compare": {
        "n_compared": 576,
        "win": 0,
        "loss": 0,
        "unchanged": 576,
        "rate_a": 0.0,
        "rate_b": 0.0,
        "delta": 0.0,
        "mcnemar_b": 0,
        "mcnemar_c": 0,
        "mcnemar_p_two_sided": 1.0,
        "bootstrap_95ci": [0.0, 0.0]
      },
      "family_compare": {
        "n_families": 75,
        "families_gained": [],
        "families_lost": [],
        "net_gain": 0,
        "pass_a": 0,
        "pass_b": 0
      }
    }
  ],
  "per_model_bug_type_repair": {
    "full576-base": {
      "off_by_one": {"total": 0, "passed": 0, "pass_rate": 0.0},
      "return_value_error": {"total": 0, "passed": 0, "pass_rate": 0.0}
    }
  }
}
```

### 3. `reports/p2/router-analysis.json`

Produced by `scripts/compute_router_analysis.py`. Structure:

```json
{
  "models_loaded": ["full576-base", ...],
  "models_skipped": [],
  "common_sample_count": 576,
  "best_single": {
    "model": "Base",
    "model_key": "full576-base",
    "overall_pass": 0.0,
    "family_pass": 0.0,
    "per_task_type": { ... },
    "n_samples": 576
  },
  "oracle_router": {
    "overall_pass": 0.0,
    "family_pass": 0.0,
    "lift_vs_best_single": 0.0
  },
  "metadata_router": {
    "overall_pass": 0.0,
    "family_pass": 0.0,
    "routing_map": {"code_generation": "Base", "static_repair": "...", "execution_repair": "..."},
    "lift_vs_best_single": 0.0
  },
  "deployable_router": {
    "overall_pass": 0.0,
    "family_pass": 0.0,
    "routing_map": { ... },
    "lift_vs_best_single": 0.0
  },
  "decision_gate": {
    "verdict": "GO",
    "reason": "...",
    "criteria": {
      "lift_threshold_pp": 5.0,
      "oracle_lift": 0.0,
      "oracle_meaningful": false,
      "deployable_lift": 0.0,
      "deployable_meaningful": false,
      "deployable_mcnemar_p": 1.0,
      "deployable_ci_95": [0.0, 0.0],
      "deployable_ci_significant": false,
      "deployable_significant": false,
      "deployable_mcnemar_b": 0,
      "deployable_mcnemar_c": 0,
      "n_common": 576
    }
  },
  "comparison_table": [
    {
      "name": "Base",
      "type": "single_model",
      "overall_pass": 0.0,
      "family_pass": 0.0,
      "code_generation_pass": 0.0,
      "static_repair_pass": 0.0,
      "execution_repair_pass": 0.0,
      "lift_vs_best_single": 0.0
    }
  ]
}
```

## Output File

`reports/p2/p2-full576-comparison-report.md`

## Report Structure

The Markdown report must contain these sections in order:

### 1. Header
```
# P2 Full-576 Comparison Report

Generated: <ISO timestamp>
```

### 2. Evaluation Setup
- Dataset: `data/p2-curriculum/frozen-eval-v2/test_raw.jsonl`
- Samples: 576
- Families: 75
- Task types: code_generation (140), static_repair (218), execution_repair (218)
- Common sample count (from paired-stats.json)

### 3. Overall Metrics

Table with columns: Model | Pass@1 | Syntax | Repair | Hidden | Format | Timeout | Family Pass

Use `fmt_pct()` to format 0-1 ratios as percentages with 1 decimal.
Rows: all 5 models from comparison.json, in MODELS order:
  Base, Stage2-v2, Stage3-v2-Continual, Stage3-Independent, Stage3-v3-Antiforget

### 4. Per-Task-Type Breakdown

Three subsections (code_generation, static_repair, execution_repair), each with a table:
Model | Total | Passed | Rate | Syntax | Format

### 5. Family-Level Pass

Table: Model | Families Passed | Total Families | Rate

Also include Stage3-v2-Continual vs Base delta (from comparison.json if available,
or from paired-stats.json family_compare).

### 6. Paired Statistics Summary

Table from paired-stats.json pair_comparisons:
Pair | N | Win | Loss | Unchanged | Delta | McNemar b/c | p (2-sided) | 95% CI

Use the `pair` field labels. Format delta as `+0.0%` or `-0.0%`.
Format p-value to 4 decimals. Format CI as `[+0.0000, +0.0000]`.

### 7. Bug-Type Repair Success Rate

Table from paired-stats.json per_model_bug_type_repair:
Bug Type | Base | Stage2-v2 | Stage3-v2-Continual | Stage3-Independent | Stage3-v3-Antiforget

Each cell: `passed/total (rate%)` or `-` if total is 0.

### 8. Router Feasibility Summary

Table from router-analysis.json comparison_table:
Model/Router | Type | Overall | Family | CodeGen | StaticRepair | ExecRepair | Lift vs Best

### 9. P3 Decision Gate

Display the verdict prominently:
```
## P3 Decision Gate

**Verdict: <VERDICT>**

<reason>

### Gate Criteria

| Criterion | Value | Threshold | Met |
|-----------|-------|-----------|-----|
| Oracle lift | X.Xpp | >= 5.0pp | YES/NO |
| Deployable lift | X.Xpp | >= 5.0pp | YES/NO |
| McNemar p | 0.XXXX | < 0.05 | YES/NO |
| 95% CI | [+0.0000, +0.0000] | lower > 0 | YES/NO |
```

## Formatting Rules

- Percentages: `f"{v * 100:.1f}%"` (e.g., 25.0%, 3.5%)
- Delta: `f"{delta:+.4f}"` for raw, `f"{delta*100:+.1f}pp"` for percentage points
- p-value: `f"{p:.4f}"`
- CI: `f"[{lo:+.4f}, {hi:+.4f}]"`
- Use `datetime.now(timezone.utc).isoformat()` for the timestamp

## Error Handling

- If any of the 3 input files is missing, print an error message to stderr and
  `return 1` (or `raise SystemExit(1)`)
- Do NOT crash with an unhandled exception on missing files

## Test Requirements

Create `tests/test_generate_full576_report.py` with:

1. **test_generates_report_with_mock_data**: Create temporary JSON files with
   minimal valid data, run the report generator, verify the output Markdown file
   exists and contains expected section headers.

2. **test_missing_input_returns_error**: Verify the script exits with code 1
   when input files are missing.

3. **test_percentage_formatting**: Verify that pass rates are formatted as
   percentages with 1 decimal place in the output.

4. **test_decision_gate_verdict_displayed**: Verify the P3 Decision Gate
   verdict is present in the output.

5. **test_all_five_models_in_table**: Verify all 5 model labels appear in the
   overall metrics table.

Use `tmp_path` fixture for temp files. Import functions from the script module.
Follow the test style in `tests/test_router_gate.py`.

## Constraints

- Do NOT modify any existing files other than creating the two new files
- Do NOT add dependencies beyond stdlib (json, os, pathlib, datetime)
- Do NOT hardcode metric values — read everything from the JSON inputs
- The script must be runnable with: `python scripts/generate_full576_report.py`
- Match the code style of `scripts/generate_p2_reports.py`

## Self-Review Checklist

Before reporting DONE, verify:
- [ ] Script creates the output file at the correct path
- [ ] All 9 report sections are present
- [ ] All 5 models appear in the metrics tables
- [ ] P3 Decision Gate verdict is displayed
- [ ] Missing input files cause exit code 1 (not a crash)
- [ ] All tests pass: `python -m pytest tests/test_generate_full576_report.py -v`
- [ ] Existing tests still pass: `python -m pytest tests/ -x -q`
- [ ] No files other than the 2 new files were modified
