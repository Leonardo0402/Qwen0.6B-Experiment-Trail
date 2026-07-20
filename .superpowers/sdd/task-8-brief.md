# Task 8: Verdict Decision Logic

## Project Context

This is Task 8 of 8 (FINAL implementation task) in the P4.1b Protocol Ablation plan (Issue #29). T6 created the orchestration script, T7 added the report generator. T8 adds the `compute_verdict()` function that applies decision rules to determine which protocol to use for P4.2 Agent SFT training.

**Branch:** `feat/p4-1b-protocol-ablation`
**Base SHA:** `111c266` (T7 commit)
**Previous tasks COMPLETE:** T1-T7

## Task Summary

- **Modify:** `scripts/run_protocol_ablation.py` (add `_ALLOWED_VERDICTS` set + `compute_verdict()` function after `generate_report()`, update `main()` with Step 6)
- **Modify:** `tests/test_protocol_ablation.py` (add 7 verdict tests)
- **Output:** verdict in `comparison-report.md` + `verdict.json`

No proactive fixes needed — this is a pure additive change. All interfaces verified.

## Interface Confirmations (verified by controller)

- All 7 test files referenced in Step 5 exist:
  - `tests/test_protocol_base.py` ✅
  - `tests/test_protocol_json.py` ✅
  - `tests/test_protocol_tag.py` ✅
  - `tests/test_protocol_dsl.py` ✅
  - `tests/test_protocol_ablation.py` ✅
  - `tests/test_agent_model_provider_protocol.py` ✅
  - `tests/test_agent_model_provider.py` ✅
- `generate_report()` exists in `scripts/run_protocol_ablation.py` (T7) ✅
- `main()` has Step 5 (generate report) from T7 ✅

## Plan Text (verbatim from plan, Task 8)

### Files
- Modify: `scripts/run_protocol_ablation.py` (add `compute_verdict` function)
- Modify: `tests/test_protocol_ablation.py` (add verdict tests)
- Output: verdict in `comparison-report.md`

### Interfaces
- Consumes: `results` list from Task 6-7
- Produces: `compute_verdict(results)` → verdict string from allowed enum

### Step 1: Write the failing tests

Add these 7 tests to `tests/test_protocol_ablation.py` (append to existing file, after the 6 existing tests):

```python
def test_verdict_keep_action_json_when_json_best():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "KEEP_ACTION_JSON"


def test_verdict_try_tag_when_tag_significantly_better():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "TRY_TAG_PROTOCOL_FOR_P4_2"


def test_verdict_try_dsl_when_dsl_significantly_better():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.6, "safety_valid_rate": 0.6, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.6, "safety_valid_rate": 0.6, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "TRY_DSL_FOR_P4_2"


def test_verdict_fix_prompt_when_all_below_30pct():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.1, "safety_valid_rate": 0.1, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "base", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
        {"protocol": "dsl", "config": "repair-lora", "model_load_ok": True, "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.05, "safety_valid_rate": 0.05, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "FIX_PROMPT_FIRST"


def test_verdict_fix_evaluator_when_model_load_fails():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "base", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
        {"protocol": "tag", "config": "repair-lora", "model_load_ok": False,
         "total_tasks": 40, "metrics": {"schema_valid_rate": 0.0, "safety_valid_rate": 0.0, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "FIX_EVALUATOR_FIRST"


def test_verdict_fix_evaluator_when_high_crash():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True,
         "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 30}},
    ]
    verdict = compute_verdict(results)
    assert verdict == "FIX_EVALUATOR_FIRST"


def test_verdict_is_valid_enum():
    from scripts.run_protocol_ablation import compute_verdict
    results = [
        {"protocol": "json", "config": "base", "model_load_ok": True,
         "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
        {"protocol": "json", "config": "repair-lora", "model_load_ok": True,
         "total_tasks": 40,
         "metrics": {"schema_valid_rate": 0.5, "safety_valid_rate": 0.5, "runtime_crash_count": 0}},
    ]
    verdict = compute_verdict(results)
    allowed = {
        "KEEP_ACTION_JSON", "TRY_TAG_PROTOCOL_FOR_P4_2", "TRY_DSL_FOR_P4_2",
        "FIX_PROMPT_FIRST", "FIX_EVALUATOR_FIRST", "STOP_PROTOCOL_CHANGE",
    }
    assert verdict in allowed
```

### Step 2: Run test to verify it fails

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py::test_verdict_keep_action_json_when_json_best -v`
Expected: FAIL with `ImportError: cannot import name 'compute_verdict'`

### Step 3: Add compute_verdict to run_protocol_ablation.py

Add this code to `scripts/run_protocol_ablation.py` (after `generate_report`, before `main()`):

```python
_ALLOWED_VERDICTS = {
    "KEEP_ACTION_JSON", "TRY_TAG_PROTOCOL_FOR_P4_2", "TRY_DSL_FOR_P4_2",
    "FIX_PROMPT_FIRST", "FIX_EVALUATOR_FIRST", "STOP_PROTOCOL_CHANGE",
}


def compute_verdict(results: list[dict]) -> str:
    """Apply T8 verdict decision rules.

    Rules (from spec §7.2):
    1. Any alternative protocol's schema_valid_rate >30% better than JSON
       AND safety_valid_rate not degraded → TRY_TAG/TRY_DSL
    2. All protocols schema_valid_rate < 30% → FIX_PROMPT_FIRST
    3. JSON baseline has highest schema_valid_rate → KEEP_ACTION_JSON
    4. Evaluator issues (model load failed or >50% trajectories crashed)
       → FIX_EVALUATOR_FIRST
    5. Fallback → STOP_PROTOCOL_CHANGE
    """
    # Average schema_valid_rate per protocol (across configs)
    proto_rates: dict[str, float] = {}
    proto_safety: dict[str, float] = {}
    for r in results:
        proto = r["protocol"]
        if proto not in proto_rates:
            proto_rates[proto] = []
            proto_safety[proto] = []
        proto_rates[proto].append(r["metrics"].get("schema_valid_rate", 0))
        proto_safety[proto].append(r["metrics"].get("safety_valid_rate", 0))

    avg_schema = {p: sum(v) / len(v) for p, v in proto_rates.items()}
    avg_safety = {p: sum(v) / len(v) for p, v in proto_safety.items()}

    json_rate = avg_schema.get("json", 0.0)
    json_safety = avg_safety.get("json", 0.0)

    # Rule 4: evaluator issues make metrics unreliable
    # - model_load_ok=False for all results, OR
    # - runtime_crash_count > 50% of total_tasks for any combination
    all_model_load_failed = all(
        not r.get("model_load_ok", True) for r in results
    )
    high_crash = any(
        r.get("metrics", {}).get("runtime_crash_count", 0) > r.get("total_tasks", 0) / 2
        for r in results
    )
    if all_model_load_failed or high_crash:
        return "FIX_EVALUATOR_FIRST"

    # Rule 2: all below 30%
    if all(rate < 0.30 for rate in avg_schema.values()):
        return "FIX_PROMPT_FIRST"

    # Rule 1: alternative protocol significantly better (>30% improvement)
    for proto in ("tag", "dsl"):
        if proto in avg_schema:
            improvement = avg_schema[proto] - json_rate
            if improvement > 0.30 and avg_safety[proto] >= json_safety:
                if proto == "tag":
                    return "TRY_TAG_PROTOCOL_FOR_P4_2"
                else:
                    return "TRY_DSL_FOR_P4_2"

    # Rule 3: JSON is best
    if json_rate >= max(avg_schema.values()):
        return "KEEP_ACTION_JSON"

    # Fallback
    return "STOP_PROTOCOL_CHANGE"
```

Also update `main()` to include verdict in the report. Add this AFTER the Step 5 block (generate report), BEFORE the final `print(f"\nDone. Reports in {_REPORT_DIR}")` line:

```python
    # Step 6: Verdict
    print("\n=== Step 6: Verdict ===")
    verdict = compute_verdict(all_results)
    print(f"Verdict: {verdict}")
    # Append verdict to report
    report_path = _REPORT_DIR / "comparison-report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        report += f"\n\n## Verdict\n\n**{verdict}**\n"
        report_path.write_text(report, encoding="utf-8")
    # Write verdict as separate file for machine reading
    (_REPORT_DIR / "verdict.json").write_text(
        json.dumps({"verdict": verdict}, indent=2), encoding="utf-8"
    )
```

### Step 4: Run tests to verify they pass

Run: `py -3.11 -m pytest tests/test_protocol_ablation.py -v`
Expected: PASS (13 tests: 4 from Task 6 + 2 from Task 7 + 7 from Task 8)

### Step 5: Run full protocol test suite

Run: `py -3.11 -m pytest tests/test_protocol_base.py tests/test_protocol_json.py tests/test_protocol_tag.py tests/test_protocol_dsl.py tests/test_protocol_ablation.py tests/test_agent_model_provider_protocol.py tests/test_agent_model_provider.py -v`
Expected: PASS (all protocol + regression tests)

### Step 6: Commit

```bash
git add scripts/run_protocol_ablation.py tests/test_protocol_ablation.py
git commit -m "feat(p4-1b): add verdict decision logic (P4.1b T8)

Verdict rules:
- Alternative protocol >30% better than JSON → TRY_TAG/TRY_DSL
- All protocols <30% schema_valid → FIX_PROMPT_FIRST
- JSON highest → KEEP_ACTION_JSON
- Fallback → STOP_PROTOCOL_CHANGE"
```

## Report Contract

Write your full report to: `.superpowers/sdd/task-8-report.md`

Return only:
- Status: DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED
- Commit SHA
- One-line test summary (e.g., "13/13 ablation tests pass, full suite pass")
- Any concerns

The report file should contain:
1. Files modified
2. Test results (full pytest output for both Step 4 and Step 5)
3. Any deviations from the brief and why
4. Self-review notes
