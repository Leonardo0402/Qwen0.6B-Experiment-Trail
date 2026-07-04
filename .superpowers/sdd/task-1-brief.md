# Task 1: Fix compare_p2_evals.py silent-skip bug

## Location
- File: `scripts/compare_p2_evals.py`
- Working directory: `e:\agent\Qwen\qwen3-code-lab`
- Current git branch: `feat/p2.2-ci-router-validation`

## The Bug

The file's `MODELS` list (lines 8-14) uses these keys:
```python
MODELS = [
    ("full576-base", "Base"),
    ("full576-stage2-boundary", "Stage2-v2"),
    ("full576-stage3-repair", "Stage3-v2-Continual"),
    ("full576-independent-stage3", "Stage3-Independent"),
    ("full576-stage3-v3-antiforget", "Stage3-v3-Antiforget"),
]
```

The `results` dict in `main()` is populated with these full keys (line 62: `results[name] = {...}` where `name` comes from `MODELS`).

However, the "Stage3 vs Base" delta section at lines 101-116 checks for the OLD (pre-rename) keys:

```python
# Per-family delta (Stage3 vs Base)
if "base" in results and "stage3-repair" in results:    # line 102 — ALWAYS FALSE
    base_fp = results["base"]["family_pass"]            # line 103 — never reached
    s3_fp = results["stage3-repair"]["family_pass"]     # line 104 — never reached
    improved = []
    regressed = []
    for f in base_fp:
        if f in s3_fp:
            if s3_fp[f] and not base_fp[f]:
                improved.append(f)
            elif not s3_fp[f] and base_fp[f]:
                regressed.append(f)
    print(f"\nStage3 vs Base:")
    print(f"  New passing families: {len(improved)}")
    print(f"  Regressed families:   {len(regressed)}")
    print(f"  Net improvement:      {len(improved) - len(regressed)}")
```

Because `"base"` and `"stage3-repair"` are NEVER keys in `results` (the real keys are `"full576-base"` and `"full576-stage3-repair"`), the `if` guard is always `False` and the entire delta block silently skips. The user sees no output between "Family-Level Pass Comparison" and "Error Category Analysis" — a silent failure.

## Required Fix (minimal, surgical)

1. On line 102, change the `if` guard keys:
   - `"base"` → `"full576-base"`
   - `"stage3-repair"` → `"full576-stage3-repair"`

2. On line 103, change `results["base"]` → `results["full576-base"]`

3. On line 104, change `results["stage3-repair"]` → `results["full576-stage3-repair"]`

4. On line 113, change the print label `"Stage3 vs Base:"` → `"Stage3-v2-Continual vs Base:"` (matches the label in MODELS for clarity).

## Explicitly NOT in scope

- Do NOT extend the section to compare other Stage3 variants (Independent, Antiforget) vs Base. The Full-576 markdown report (separate task) will cover those comparisons via `compute_paired_stats.py`.
- Do NOT add per-family delta info to the JSON output (`full576-comparison.json`). That information lives in `paired-stats.json`.
- Do NOT touch any other section of the file (per-task-type, family-level pass, error category analysis, JSON save).
- Do NOT reformat, reorder, or rename anything outside the four lines listed above.

## Verification

After the fix, run:
```
D:\Anaconda\envs\qwen3-code-lab\python.exe scripts\compare_p2_evals.py
```

Expected behavior (currently the 5 `full576-*.json` eval files don't exist yet, so the script will print "skipping" for each — that's fine). The fix is verified by:

1. Reading the patched file and confirming lines 102-104 use `full576-base` / `full576-stage3-repair` and line 113 uses the new label.
2. Confirming there is NO silent skip — i.e., when the eval files DO exist, the section will execute. (You can construct a tiny synthetic test by creating two minimal fake eval JSONs in a temp dir, but that is optional — the key fix is the key-name correction.)
3. Optional: add an assertion-style sanity check by running the script in dry mode (no eval files) and confirming it doesn't crash.

## Test file

There is no existing test file for `compare_p2_evals.py`. Do NOT create one — the fix is mechanical (4 line edits) and the script is a reporting utility, not a library function. The Full-576 report task will exercise the script end-to-end after evaluations complete.

## Commit message

```
fix(scripts): correct silent-skip keys in compare_p2_evals Stage3-vs-Base section

The MODELS list was renamed to use full576-* prefixes but the
"Stage3 vs Base" delta block still checked the old keys "base" and
"stage3-repair", causing the entire section to silently skip.
```

## Self-review checklist

- [ ] Only lines 102, 103, 104, 113 changed (no other lines touched)
- [ ] No reformatting of surrounding code
- [ ] Print label matches the MODELS label for `full576-stage3-repair` ("Stage3-v2-Continual")
- [ ] Script still runs without error when eval files are missing
