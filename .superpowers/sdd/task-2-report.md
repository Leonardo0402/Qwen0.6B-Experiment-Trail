# Task 2 Report: Extend compute_router_analysis.py with P3 Decision Gate

## Status: DONE

## What was implemented

Extended `scripts/compute_router_analysis.py` (and added `tests/test_router_gate.py`) with the P3 Decision Gate logic per the task brief.

### 1. Two inline helper functions (lines 107-137 of `scripts/compute_router_analysis.py`)

Added after `extract_bug_type`, in a new "Paired stats helpers" section:
- `mcnemar_exact(b: int, c: int) -> float` — two-sided exact McNemar p-value via binomial
- `paired_bootstrap_ci(pass_a, pass_b, n_boot=10000, seed=42) -> tuple[float, float]` — paired bootstrap 95% CI for difference in pass rate (b - a)

Both copied verbatim from the brief (which mirrors `scripts/compute_paired_stats.py` lines 69-108). They are duplicated inline (not imported) to keep `compute_router_analysis.py` self-contained, as required.

### 2. `apply_decision_gate(...)` function (lines 253-323)

Added before `main()` in a new "P3 Decision Gate" section. Copied verbatim from the brief. Pure function (no I/O, no global state) returning:

```python
{
    "verdict": "GO" | "NO-GO" | "SIGNAL",
    "reason": "<human-readable explanation>",
    "criteria": {
        "lift_threshold_pp": 5.0,
        "oracle_lift", "oracle_meaningful",
        "deployable_lift", "deployable_meaningful",
        "deployable_mcnemar_p",
        "deployable_ci_95": [lo, hi],
        "deployable_ci_significant",
        "deployable_significant",
        "deployable_mcnemar_b", "deployable_mcnemar_c",
        "n_common",
    },
}
```

### 3. Integration in `main()` (lines 493-527)

Inserted between the `deployable_router = {...}` block and the "Comparison table" section. The clean loop I wrote (NOT copied from the awkward example in the brief):

```python
# ------------------------------------------------------------------
# 5. P3 Decision Gate — Deployable Router vs Best Single
# ------------------------------------------------------------------
# McNemar convention: A = Best Single, B = Deployable Router
#   b = #samples where A passed, B failed
#   c = #samples where A failed, B passed
best_single_pass_map = {
    sid: _passed(by_model[best_single_key][sid]) for sid in common
}
gate_b = 0
gate_c = 0
pass_a: list[bool] = []  # best_single
pass_b: list[bool] = []  # deployable
for sid in common:
    bs_pass = best_single_pass_map[sid]
    dep_pass = deployable_pass[sid]
    if bs_pass and not dep_pass:
        gate_b += 1
    elif not bs_pass and dep_pass:
        gate_c += 1
    pass_a.append(bs_pass)
    pass_b.append(dep_pass)

mcnemar_p = mcnemar_exact(gate_b, gate_c)
ci_lo, ci_hi = paired_bootstrap_ci(pass_a, pass_b)
gate_result = apply_decision_gate(
    deployable_lift=deployable_router["lift_vs_best_single"],
    oracle_lift=oracle_router["lift_vs_best_single"],
    deployable_mcnemar_p=mcnemar_p,
    deployable_ci_lo=ci_lo,
    deployable_ci_hi=ci_hi,
    deployable_b=gate_b,
    deployable_c=gate_c,
    n_common=len(common),
)
```

The loop iterates `common` exactly once, computing both McNemar discordant counts (`gate_b`/`gate_c`) and the aligned `pass_a`/`pass_b` lists for the bootstrap CI in a single pass.

### 4. `decision_gate` key in JSON result (line 593)

Added `"decision_gate": gate_result,` to the `result` dict, immediately after `deployable_router` and before `comparison_table`.

### 5. "P3 Decision Gate" markdown section (lines 780-798)

Added after the "Methodology Notes" section and before the final `out_md.write_text(...)` call. Includes:
- The verdict in bold
- The human-readable reason
- A criteria table with columns: Criterion | Value | Threshold | Met

The markdown code matches the brief verbatim.

### 6. Test file `tests/test_router_gate.py` (new file, 117 lines)

Created with the verbatim test cases from the brief. Three test classes:
- `TestDecisionGate` — 9 tests covering GO/NO-GO/SIGNAL branches and edge cases
- `TestMcNemarExact` — 3 tests (no discordant pairs, symmetric, extreme asymmetry)
- `TestPairedBootstrapCI` — 2 tests (identical lists → zero CI, all-b-passes → positive CI)

Total: 14 test methods.

## Test results (all 3 verification commands)

### 1. New test file
```
$ D:\Anaconda\envs\qwen3-code-lab\python.exe -m pytest tests/test_router_gate.py -v
============================= test session starts =============================
platform win32 -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.14.1, hypothesis-6.155.7, timeout-2.4.0
collected 14 items

tests\test_router_gate.py ..............                                 [100%]

============================= 14 passed in 1.84s ==============================
```
**Result: 14/14 passed**

### 2. Import check
```
$ D:\Anaconda\envs\qwen3-code-lab\python.exe -c "from scripts.compute_router_analysis import apply_decision_gate, mcnemar_exact, paired_bootstrap_ci; print('imports OK')"
imports OK
```
**Result: imports OK**

### 3. Existing CI tests
```
$ D:\Anaconda\envs\qwen3-code-lab\python.exe -m pytest tests/test_p2_evidence_hardening.py tests/test_metrics.py tests/test_schemas.py tests/test_validators.py -v --tb=short
============================= test session starts =============================
platform win32 -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.14.1, hypothesis-6.155.7, timeout-2.4.0
collected 148 items

tests\test_p2_evidence_hardening.py .................................... [ 24%]
                                                                         [ 24%]
tests\test_metrics.py ...............................................    [ 56%]
tests\test_schemas.py ........................................           [ 83%]
tests\test_validators.py .........................                       [100%]

======================= 148 passed in 111.58s (0:01:51) =======================
```
**Result: 148/148 passed**

## Commit

- **SHA:** `5dbaf8b`
- **Subject:** `feat(router): add P3 Decision Gate with McNemar + bootstrap CI`
- **Files changed:** 2 files, 905 insertions (new: `scripts/compute_router_analysis.py`, new: `tests/test_router_gate.py`)
- **Parent:** `deb8db9` (the Task 1 / BASE commit)

Commit message used verbatim from the brief.

## Pre-existing repository state (NOT my changes)

Two pre-existing conditions observed during the task — left untouched per the brief's "Explicitly NOT in scope" rule:
1. `scripts/compute_paired_stats.py` had uncommitted modifications (renamed model keys from `base`/`stage2-boundary`/... to `full576-base`/`full576-stage2-boundary`/...). I did NOT stage or modify this file.
2. `scripts/compute_router_analysis.py` was untracked before my work (Task 1 left it on disk but did not commit it). My commit now includes it as a new file alongside my Task 2 changes.

## Self-review checklist

- [x] `apply_decision_gate` is a pure function (no I/O, no global state)
- [x] McNemar and bootstrap CI functions match the implementations in `compute_paired_stats.py` (copied verbatim from the brief, which mirrors compute_paired_stats.py lines 69-108)
- [x] `decision_gate` key added to the `result` dict in `main()` (line 593)
- [x] "P3 Decision Gate" section added to the markdown output (lines 780-798)
- [x] All tests in `tests/test_router_gate.py` pass (14/14 — brief mentioned "12 test cases" but lists 14 in the test code; all pass)
- [x] Existing CI tests (evidence_hardening, metrics, schemas, validators) still pass (148/148)
- [x] No existing router computation logic was modified (Best Single, Oracle, Metadata, Deployable sections untouched — only added new code in clean insertion points)
- [x] No reformatting of existing code (matched double-quoted strings, 4-space indent, type hints on public functions)

## Notes

- The brief's commit message says "12 test cases" but the test file actually contains 14 test methods (9 + 3 + 2). I kept the commit message verbatim as instructed by the brief. The brief's own test code listing shows 14 methods, so this is a brief-internal inconsistency, not an implementation error.
- The `paired_bootstrap_ci` in the brief is slightly simpler than the one in `compute_paired_stats.py` (no `observed` variable computed). I copied the brief's version verbatim per instructions — they are functionally identical for CI computation.
