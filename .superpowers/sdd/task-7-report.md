# Task 7 Report: Comparison Report Generation

## Status: DONE

## Commit
- SHA: `111c266`
- Message: `feat(p4-1b): add comparison report generation (P4.1b T7)`
- Branch: `feat/p4-1b-protocol-ablation`
- Base SHA: `9ebfdd0` (T6 commit)

## 1. Files Modified

### `scripts/run_protocol_ablation.py`
- **Added** `generate_report(results: list[dict], taxonomy: dict) -> str` function before `main()` (lines 298-356). Generates a markdown comparison report containing:
  - Overview section (protocol count, config count, total combinations)
  - Metrics by Protocol x Config markdown table (14 columns)
  - Failure Taxonomy markdown table
  - Protocol Comparison Summary (avg schema_valid_rate per protocol)
- **Updated** `main()` to add Step 5 block (lines 418-422) after the `failure-taxonomy.json` write and before the final `print(f"\nDone. Reports in {_REPORT_DIR}")` line. The new block calls `generate_report(all_results, taxonomy)` and writes the output to `reports/p4/protocol-ablation/comparison-report.md`.

### `tests/test_protocol_ablation.py`
- **Appended** 2 new tests after the existing 4 tests:
  - `test_generate_report_contains_all_protocols` — verifies the report contains all 3 protocol names ("json", "tag", "dsl"), the metric key "schema_valid_rate", and the failure class "FORMAT_PARSE_FAIL".
  - `test_generate_report_has_markdown_table` — verifies the report contains markdown table syntax (`|`) and the metric key "format_parse_rate".

## 2. Test Results

### Step 2: Verification that new test fails (before implementation)

Command: `py -3.11 -m pytest tests/test_protocol_ablation.py::test_generate_report_contains_all_protocols -v`

```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 1 item

tests\test_protocol_ablation.py F                                        [100%]

================================== FAILURES ===================================
_________________ test_generate_report_contains_all_protocols _________________

    def test_generate_report_contains_all_protocols():
>       from scripts.run_protocol_ablation import generate_report
E       ImportError: cannot import name 'generate_report' from 'scripts.run_protocol_ablation' (E:\agent\Qwen\qwen3-code-lab\scripts\run_protocol_ablation.py)

tests\test_protocol_ablation.py:194: ImportError
=========================== short test summary info ===========================
FAILED tests/test_protocol_ablation.py::test_generate_report_contains_all_protocols - ImportError: cannot import name 'generate_report' from 'scripts.run_protocol_ablation' (E:\agent\Qwen\qwen3-code-lab\scripts\run_protocol_ablation.py)
============================== 1 failed in 0.56s ==============================
```

Result: FAILED with `ImportError: cannot import name 'generate_report'` — as expected per Step 2 of the brief.

### Step 4: Full test suite (after implementation)

Command: `py -3.11 -m pytest tests/test_protocol_ablation.py -v -o "addopts="`

```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\20385\AppData\Local\Programs\Python\Python311\python.exe
cachedir: .pytest_cache
hypothesis profile 'ci' -> database=None, deadline=None, print_blob=True, derandomize=True, suppress_health_check=(HealthCheck.too_slow,)
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collecting ... collected 6 items

tests/test_protocol_ablation.py::test_baseline_lock_records_all_fields PASSED [ 16%]
tests/test_protocol_ablation.py::test_aggregate_metrics_computes_all_fields PASSED [ 33%]
tests/test_protocol_ablation.py::test_classify_failures_returns_taxonomy PASSED [ 50%]
tests/test_protocol_ablation.py::test_run_combination_with_mock_protocol PASSED [ 66%]
tests/test_protocol_ablation.py::test_generate_report_contains_all_protocols PASSED [ 83%]
tests/test_protocol_ablation.py::test_generate_report_has_markdown_table PASSED [100%]

============================== 6 passed in 1.72s ==============================
```

Result: **6/6 PASSED** (4 from T6 + 2 from T7).

Note: The `pyproject.toml` has `addopts = "-q"` which collapses pytest output to dots by default. To see individual test names, the override `-o "addopts="` was added. This is a display-only concern and does not affect test execution or results.

## 3. Deviations from the Brief

**None.** All code was added verbatim from the brief:
- The 2 tests were appended to `tests/test_protocol_ablation.py` exactly as specified.
- The `generate_report()` function was added before `main()` exactly as specified.
- The Step 5 block was inserted into `main()` after the `failure-taxonomy.json` write and before the final `print(f"\nDone. Reports in {_REPORT_DIR}")` line, exactly as specified.
- Git add was limited to `scripts/run_protocol_ablation.py` and `tests/test_protocol_ablation.py` as instructed (other modified files in the workspace were left unstaged).
- Commit message matches verbatim: `feat(p4-1b): add comparison report generation (P4.1b T7)`.

## 4. Self-Review Notes

- **Additive only**: No existing functions or tests were modified. The change is purely additive — a new `generate_report()` function and a new Step 5 block in `main()`.
- **Function placement**: `generate_report()` is placed between `_detect_repeated_loop()` (the last helper before `main()`) and `main()`, as required by the brief.
- **Step 5 placement**: Verified the Step 5 block sits between the `failure-taxonomy.json` write (Step 4 output) and the final `print(f"\nDone. Reports in {_REPORT_DIR}")` line — exactly per the brief's instruction.
- **Markdown table column count**: The header row has 14 columns (Protocol, Config, + 12 metrics) and the separator row has 14 dashes-aligned cells — verified alignment matches.
- **Metric defaults**: The function uses `.get(metric, 0)` for all metric lookups, so results with partial metric dicts (as in the tests) render correctly with 0 defaults.
- **Test coverage**: The 2 new tests verify (a) all protocol names appear in the report and key metric/failure-class tokens are present, and (b) markdown table syntax is present. This is sufficient for T7's scope.
- **No proactive fixes**: No other code in the script or tests was touched. Pre-existing modified files in the workspace (progress.md, task briefs/reports, data manifests) were left as-is and not staged.
- **Line ending warning**: Git emitted `LF will be replaced by CRLF` warnings on both staged files. This is a Windows-only line-ending normalization warning and does not affect file content or test execution. The repo's existing files use the same convention.
- **Imports**: The `generate_report()` function uses only built-in Python features (f-strings, `sorted`, `set`, `sum`, `len`) — no new imports were needed in the script.
