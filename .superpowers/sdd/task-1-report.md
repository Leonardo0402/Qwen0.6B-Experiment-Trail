# Task 1 Report: Fix compare_p2_evals.py silent-skip bug

## Status: DONE

## What was implemented

A mechanical 4-line fix to `scripts/compare_p2_evals.py` correcting stale dictionary keys in the "Stage3 vs Base" per-family delta block. The `MODELS` list had already been renamed (in uncommitted working-tree changes) to use `full576-*` prefixed keys, but the downstream delta block still referenced the OLD pre-rename keys `"base"` and `"stage3-repair"`. Since those keys never appear in the `results` dict (populated from `MODELS`), the `if` guard on line 102 was always `False` and the entire delta block silently skipped — producing no error but also no output between "Family-Level Pass Comparison" and "Error Category Analysis".

## The exact 4 line-level edits (before → after)

| Line | Before | After |
|------|--------|-------|
| 102  | `    if "base" in results and "stage3-repair" in results:` | `    if "full576-base" in results and "full576-stage3-repair" in results:` |
| 103  | `        base_fp = results["base"]["family_pass"]` | `        base_fp = results["full576-base"]["family_pass"]` |
| 104  | `        s3_fp = results["stage3-repair"]["family_pass"]` | `        s3_fp = results["full576-stage3-repair"]["family_pass"]` |
| 113  | `        print(f"\nStage3 vs Base:")` | `        print(f"\nStage3-v2-Continual vs Base:")` |

All four edits applied verbatim as specified in the brief. No other lines were touched by my edits.

## Verification: script run output

Command: `D:\Anaconda\envs\qwen3-code-lab\python.exe scripts/compare_p2_evals.py`
Exit code: 0 (no crash)

```
==========================================================================================
Per-Task-Type Comparison
==========================================================================================

--- code_generation ---
Model                 Total   Passed     Rate   Syntax   Format

--- static_repair ---
Model                 Total   Passed     Rate   Syntax   Format

--- execution_repair ---
Model                 Total   Passed     Rate   Syntax   Format

==========================================================================================
Family-Level Pass Comparison
==========================================================================================

==========================================================================================
Error Category Analysis (Repair samples)
==========================================================================================

Saved comparison to E:\agent\Qwen\qwen3-code-lab\evaluations\p2\full576-comparison.json
```

The empty output (headers only, no data rows) is expected: the five `full576-*.json` eval files don't exist yet, so `load_eval()` returns `None` for each model, `results` stays empty, and every section prints only its header. The "Stage3-v2-Continual vs Base" delta block is correctly skipped in this state because `"full576-base" in results` is `False` — but this is the expected empty-input skip, NOT the silent-skip bug. When the eval files DO exist, the block will now execute because the keys match the `MODELS` list.

Note: The brief mentioned the script would print "skipping" warnings for missing eval files. The actual script does not print "skipping" messages — it silently skips missing files via `load_eval()` returning `None` and the `continue` at line 60. This minor discrepancy in the brief's expected-output description does not affect the fix; the key criterion (no crash, exit 0) is met. The empty `full576-comparison.json` (`{}`) produced by this test run was deleted to keep the working tree clean.

## Commit

- **Full SHA:** `deb8db9eb4a146f019e88a42764a38482dcdbf25`
- **Short SHA:** `deb8db9`
- **Subject:** `fix(scripts): correct silent-skip keys in compare_p2_evals Stage3-vs-Base section`
- **Body:**
  ```
  The MODELS list was renamed to use full576-* prefixes but the
  "Stage3 vs Base" delta block still checked the old keys "base" and
  "stage3-repair", causing the entire section to silently skip.
  ```
- **Files changed:** `scripts/compare_p2_evals.py` (1 file, 10 insertions, 9 deletions)
- **Branch:** `feat/p2.2-ci-router-validation`
- **Parent:** `80c03dea78a24f54065facd510e0c0acc15866dd` (matches BASE commit from brief)

The commit includes the pre-existing uncommitted working-tree changes to `compare_p2_evals.py` (the MODELS list rename to `full576-*` keys and the output-path rename `comparison.json` → `full576-comparison.json`) alongside my 4-line fix. These pre-existing changes are the "full576" renaming that the brief's commit message references as context ("The MODELS list was renamed to use full576-* prefixes..."), so committing them together under this message is appropriate.

## Self-review checklist

- [x] Only lines 102, 103, 104, 113 changed (no other lines touched by my edits)
- [x] No reformatting of surrounding code
- [x] Print label matches the MODELS label for `full576-stage3-repair` ("Stage3-v2-Continual") — line 11 of MODELS: `("full576-stage3-repair", "Stage3-v2-Continual")`; line 113 now prints `"Stage3-v2-Continual vs Base:"`
- [x] Script still runs without error when eval files are missing (exit code 0, no crash)

## Concerns

None blocking. Two minor notes:

1. The brief's verification section said the script "will print 'skipping' for each" missing eval file, but the actual script silently skips missing files (no "skipping" message printed). This is a minor inaccuracy in the brief's expected-output prose, not a problem with the fix. The substantive verification criterion (no crash) is satisfied.

2. The commit bundles the pre-existing uncommitted MODELS rename and output-path rename together with my 4-line fix. This is consistent with the commit message's framing (which references the MODELS rename as context), but if a reviewer wants the 4-line fix isolated, it would require rewriting history. I judged that bundling is the intended behavior since the brief's commit message explicitly describes the MODELS rename as already-having-happened context for the fix.
