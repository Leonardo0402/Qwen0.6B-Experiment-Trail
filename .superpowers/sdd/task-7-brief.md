# Task 7: Comparison Report Generation

## Project Context

This is Task 7 of 8 in the P4.1b Protocol Ablation plan (Issue #29). T6 created the orchestration script (`scripts/run_protocol_ablation.py`) with `baseline_lock()`, `run_combination()`, `aggregate_metrics()`, `classify_failures()`, and `main()`. T7 adds the `generate_report()` function that produces a markdown comparison report from the results.

**Branch:** `feat/p4-1b-protocol-ablation`
**Base SHA:** `9ebfdd0` (T6 commit)
**Previous tasks COMPLETE:** T1-T6

## Task Summary

- **Modify:** `scripts/run_protocol_ablation.py` (add `generate_report` function before `main()`, update `main()` to call it)
- **Modify:** `tests/test_protocol_ablation.py` (add 2 report tests)
- **Output:** `reports/p4/protocol-ablation/comparison-report.md`

No proactive fixes needed — this is a pure additive change with no external interface dependencies.

## Plan Text (verbatim from plan, Task 7)

### Files
- Modify: `scripts/run_protocol_ablation.py` (add `generate_report` function)
- Modify: `tests/test_protocol_ablation.py` (add report tests)
- Output: `reports/p4/protocol-ablation/comparison-report.md`

### Interfaces
- Consumes: `aggregate_metrics` and `classify_failures` from Task 6
- Produces: `generate_report(results, taxonomy)` → markdown string

### Step 1: Write the failing tests

Add these 2 tests to `tests/test_protocol_ablation.py` (append to existing file, after the existing 4 tests):

```python
def test_generate_report_contains_all_protocols():
    from scripts.run_protocol_ablation import generate_report
    results = [
        {"protocol": "json", "config": "base", "metrics": {"schema_valid_rate": 0.0}},
        {"protocol": "json", "config": "repair-lora", "metrics": {"schema_valid_rate": 0.0}},
        {"protocol": "tag", "config": "base", "metrics": {"schema_valid_rate": 0.5}},
        {"protocol": "tag", "config": "repair-lora", "metrics": {"schema_valid_rate": 0.6}},
        {"protocol": "dsl", "config": "base", "metrics": {"schema_valid_rate": 0.3}},
        {"protocol": "dsl", "config": "repair-lora", "metrics": {"schema_valid_rate": 0.4}},
    ]
    taxonomy = {"FORMAT_PARSE_FAIL": 10, "SCHEMA_VALIDATION_FAIL": 20}
    report = generate_report(results, taxonomy)
    assert "json" in report
    assert "tag" in report
    assert "dsl" in report
    assert "schema_valid_rate" in report
    assert "FORMAT_PARSE_FAIL" in report


def test_generate_report_has_markdown_table():
    from scripts.run_protocol_ablation import generate_report
    results = [
        {"protocol": "json", "config": "base",
         "metrics": {"format_parse_rate": 1.0, "schema_valid_rate": 0.0,
                     "safety_valid_rate": 0.0, "action_type_valid_rate": 0.5,
                     "arguments_valid_rate": 0.0, "forbidden_action_count": 0,
                     "task_success_rate": 0.0, "max_steps_hit_rate": 1.0,
                     "runtime_crash_count": 0}},
    ]
    taxonomy = {"FORMAT_PARSE_FAIL": 5}
    report = generate_report(results, taxonomy)
    assert "|" in report  # markdown table
    assert "format_parse_rate" in report
```

### Step 2: Run test to verify it fails

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py::test_generate_report_contains_all_protocols -v`
Expected: FAIL with `ImportError: cannot import name 'generate_report'`

### Step 3: Add generate_report to run_protocol_ablation.py

Add this function to `scripts/run_protocol_ablation.py` (before `main()`):

```python
def generate_report(results: list[dict], taxonomy: dict) -> str:
    """Generate markdown comparison report from ablation results."""
    lines = [
        "# P4.1b Protocol Ablation — Comparison Report",
        "",
        "## Overview",
        "",
        f"- Protocols: {len(set(r['protocol'] for r in results))}",
        f"- Configs: {len(set(r['config'] for r in results))}",
        f"- Total combinations: {len(results)}",
        "",
        "## Metrics by Protocol x Config",
        "",
        "| Protocol | Config | format_parse_rate | schema_valid_rate | safety_valid_rate | action_type_valid_rate | arguments_valid_rate | forbidden_count | unknown_count | task_success_rate | finish_no_tests | finish_mismatch | max_steps_hit_rate | crashes |",
        "|----------|--------|-------------------|-------------------|-------------------|------------------------|----------------------|-----------------|----------------|-------------------|------------------|-----------------|---------------------|---------|",
    ]

    for r in sorted(results, key=lambda x: (x["protocol"], x["config"])):
        m = r.get("metrics", {})
        lines.append(
            f"| {r['protocol']} | {r['config']} "
            f"| {m.get('format_parse_rate', 0):.2%} "
            f"| {m.get('schema_valid_rate', 0):.2%} "
            f"| {m.get('safety_valid_rate', 0):.2%} "
            f"| {m.get('action_type_valid_rate', 0):.2%} "
            f"| {m.get('arguments_valid_rate', 0):.2%} "
            f"| {m.get('forbidden_action_count', 0)} "
            f"| {m.get('unknown_action_count', 0)} "
            f"| {m.get('task_success_rate', 0):.2%} "
            f"| {m.get('finish_without_tests_count', 0)} "
            f"| {m.get('finish_claim_mismatch_count', 0)} "
            f"| {m.get('max_steps_hit_rate', 0):.2%} "
            f"| {m.get('runtime_crash_count', 0)} |"
        )

    lines.extend([
        "",
        "## Failure Taxonomy",
        "",
        "| Failure Class | Count |",
        "|---------------|-------|",
    ])
    for fc, count in sorted(taxonomy.items()):
        lines.append(f"| {fc} | {count} |")

    lines.extend([
        "",
        "## Protocol Comparison Summary",
        "",
    ])

    # Summarize by protocol (average across configs)
    protocols = sorted(set(r["protocol"] for r in results))
    for proto in protocols:
        proto_results = [r for r in results if r["protocol"] == proto]
        avg_schema = sum(r["metrics"].get("schema_valid_rate", 0) for r in proto_results) / len(proto_results)
        lines.append(f"- **{proto}**: avg schema_valid_rate = {avg_schema:.2%}")

    return "\n".join(lines)
```

Also add to `main()` after writing failure-taxonomy.json:

```python
    # Step 5: Generate report
    print("\n=== Step 5: Comparison Report ===")
    report = generate_report(all_results, taxonomy)
    (_REPORT_DIR / "comparison-report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {_REPORT_DIR / 'comparison-report.md'}")
```

### Step 4: Run tests to verify they pass

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py -v`
Expected: PASS (6 tests: 4 from Task 6 + 2 from Task 7)

### Step 5: Commit

```bash
git add scripts/run_protocol_ablation.py tests/test_protocol_ablation.py
git commit -m "feat(p4-1b): add comparison report generation (P4.1b T7)"
```

## Report Contract

Write your full report to: `.superpowers/sdd/task-7-report.md`

Return only:
- Status: DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED
- Commit SHA
- One-line test summary (e.g., "6/6 tests pass")
- Any concerns

The report file should contain:
1. Files modified
2. Test results (full pytest output)
3. Any deviations from the brief and why
4. Self-review notes
