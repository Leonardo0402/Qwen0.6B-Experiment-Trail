# Task 3 Report: generate_full576_report.py

## Status: DONE

## What was implemented

Created `scripts/generate_full576_report.py` — a new script that reads three JSON
inputs produced by upstream P2 scripts and produces a unified Markdown report
at `reports/p2/p2-full576-comparison-report.md`.

The script reads:
- `evaluations/p2/full576-comparison.json` (from `compare_p2_evals.py`)
- `reports/p2/full576-paired-stats.json` (from `compute_paired_stats.py`)
- `reports/p2/router-analysis.json` (from `compute_router_analysis.py`)

The generated report contains all 9 sections in the required order:
1. Header (`# P2 Full-576 Comparison Report` + ISO timestamp)
2. Evaluation Setup (dataset, samples, families, task types, common sample count)
3. Overall Metrics (5 models × 8 columns incl. Family Pass)
4. Per-Task-Type Breakdown (3 subsections: code_generation, static_repair, execution_repair)
5. Family-Level Pass (+ Stage3-v2-Continual vs Base family delta)
6. Paired Statistics Summary (pair_comparisons table with McNemar + CI)
7. Bug-Type Repair Success Rate (bug_types × 5 models, `-` when total=0)
8. Router Feasibility Summary (router-analysis comparison_table)
9. P3 Decision Gate (verdict + reason + criteria table with YES/NO)

If any of the 3 input files is missing, the script prints an error to stderr
and returns exit code 1 (no unhandled exception).

Code style matches `scripts/generate_p2_reports.py` (load_json, fmt_pct helpers,
`lines.append(...)` pattern, `_ROOT`/`_REPORTS` module-level constants).

A test file with 5 tests was also created, covering: mock-data end-to-end run,
missing-input error path, percentage formatting, decision-gate verdict display,
and all-five-models presence.

## Files created

- `e:\agent\Qwen\qwen3-code-lab\scripts\generate_full576_report.py` (266 lines)
- `e:\agent\Qwen\qwen3-code-lab\tests\test_generate_full576_report.py` (223 lines)

No existing files were modified.

## Test results

### New tests

Command:
```
D:\Anaconda\envs\qwen3-code-lab\python.exe -m pytest tests/test_generate_full576_report.py -v
```

Output:
```
============================= test session starts =============================
platform win32 -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.14.1, hypothesis-6.155.7, timeout-2.4.0
collected 5 items

tests\test_generate_full576_report.py .....                              [100%]

======================== 5 passed in 77.82s (0:01:17) =========================
```

Result: **5 passed, 0 failed**.

The 5 tests:
1. `test_generates_report_with_mock_data` — writes 3 temp JSON files, runs `main()`, verifies the output `.md` exists and contains all 9 section headers.
2. `test_missing_input_returns_error` — points paths to non-existent files, asserts `main()` returns 1 and no output file is written.
3. `test_percentage_formatting` — calls `generate_report()` directly with mock dicts, asserts `25.0%`, `80.0%`, `5.0%` appear (1-decimal percentage format).
4. `test_decision_gate_verdict_displayed` — asserts `**Verdict: SIGNAL**`, the reason string, and the Gate Criteria table are present.
5. `test_all_five_models_in_table` — asserts all 5 model labels (Base, Stage2-v2, Stage3-v2-Continual, Stage3-Independent, Stage3-v3-Antiforget) appear in the output.

### Full test suite (no-breakage check)

Command:
```
D:\Anaconda\envs\qwen3-code-lab\python.exe -m pytest tests/ -x -q
```

Output (tail):
```
........................................................................ [ 11%]
........................................................................ [ 22%]
........................................................................ [ 33%]
........................................................................ [ 45%]
........................................................................ [ 56%]
...............................................................s........ [ 67%]
........................................................................ [ 79%]
........................................................................ [ 90%]
.............................................................            [100%]
============================== warnings summary ===============================
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
  ...
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

Exit code: **0**. Result: **708 passed, 1 skipped, 0 failed**. The single
skip is pre-existing (unrelated to this task). The 5 new tests are included
in this count.

## Commit SHA

Full SHA: `6670523e2e9d67af4dee55a272994d9adca56e7d`

```
commit 6670523e2e9d67af4dee55a272994d9adca56e7d
Author: Claude Code <noreply@anthropic.com>
Date:   Sat Jul 4 00:05:18 2026 +0800

    feat(scripts): add generate_full576_report.py for Full-576 comparison report

 scripts/generate_full576_report.py    | 266 ++++++++++++++++++++++++++++++++++
 tests/test_generate_full576_report.py | 223 ++++++++++++++++++++++++++++
 2 files changed, 489 insertions(+)
```

Only the 2 new files were committed; no existing files were modified.

## Self-review checklist

- [x] Script creates the output file at the correct path
      (`reports/p2/p2-full576-comparison-report.md`)
- [x] All 9 report sections are present (verified by
      `test_generates_report_with_mock_data`)
- [x] All 5 models appear in the metrics tables (verified by
      `test_all_five_models_in_table`)
- [x] P3 Decision Gate verdict is displayed (verified by
      `test_decision_gate_verdict_displayed`)
- [x] Missing input files cause exit code 1, not a crash (verified by
      `test_missing_input_returns_error`; `main()` checks `p.exists()` before
      opening and returns 1 after printing to stderr)
- [x] All new tests pass: `python -m pytest tests/test_generate_full576_report.py -v`
      → 5 passed
- [x] Existing tests still pass: `python -m pytest tests/ -x -q` → 708 passed,
      1 skipped, 0 failed, exit 0
- [x] No files other than the 2 new files were modified (verified by
      `git show --stat HEAD` — only 2 files in the commit, 489 insertions, 0
      deletions)
- [x] No dependencies beyond stdlib (`json`, `sys`, `datetime`, `pathlib`)
- [x] No hardcoded metric values — every metric is read from the JSON inputs
      via `.get(...)` with safe defaults
- [x] Script is runnable with `python scripts/generate_full576_report.py`
      (uses `if __name__ == "__main__": raise SystemExit(main())`)
- [x] Code style matches `generate_p2_reports.py` (load_json, fmt_pct,
      `lines.append` pattern, `_ROOT` / `_REPORTS` module constants)

## Concerns

None.

- The 3 upstream JSON inputs do not exist yet (evaluations are still running),
  so running the script directly today exits with code 1 and a clear stderr
  message — this is the intended graceful-degradation behavior per the brief.
- The Stage3-v2-Continual vs Base family delta in section 5 is computed from
  `comparison.json` only (the brief allowed falling back to `paired-stats.json`
  `family_compare`, but `comparison.json` is the primary source and will be
  present whenever the upstream `compare_p2_evals.py` has run). If both models
  are missing from `comparison.json`, the delta line is omitted rather than
  crashing.
- Section 6 delta is formatted as `+0.0%` per the brief's section-specific
  instruction; section 8 router lift uses `+.4f` (raw) to match the existing
  `compute_router_analysis.py` Markdown output. Both interpretations are
  documented in the brief's formatting rules.
