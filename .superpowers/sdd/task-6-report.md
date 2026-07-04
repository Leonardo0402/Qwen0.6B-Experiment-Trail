# Task 6 Report: Build Family Registry

## Status: DONE (v2.1 rework — rebuilt on pad-then-verify verified set)

The builder is correct and all 10 tests pass. Rebuilt on the 714
pad-then-verify verified samples (was 955) and 50 quarantined families
(was 58).

## Files (4 — unchanged from original Task 6)

1. `e:\agent\Qwen\qwen3-code-lab\src\family_registry.py`
2. `e:\agent\Qwen\qwen3-code-lab\scripts\build_family_registry.py`
3. `e:\agent\Qwen\qwen3-code-lab\tests\test_family_registry.py`
4. `e:\agent\Qwen\qwen3-code-lab\data\family-registry.json` (regenerated)

## Actual Counts from the Redo Builder Run

```
Family registry built: data\family-registry.json
  total_families:      807
  total_p2_used:       374
  total_quarantined:   50
  total_new_available: 409
```

| Field                 | v2.1 Redo | Original (e88146c) |
|-----------------------|-----------|--------------------|
| total_families        | 807       | 958                |
| total_p2_used         | 374       | 374                |
| total_quarantined     | 50        | 58                 |
| total_new_available   | 409       | 556                |

## Builder Invariants (all pass)

The builder asserts 5 invariants with input-derived expected values:

1. `total_p2_used == expected_p2_count` → 374 == 374. **PASS**
2. `total_quarantined == expected_quarantine_count` → 50 == 50 (length
   of `quarantined_families` in the Task 5-redo quarantine JSON). **PASS**
3. `total_new_available == count of families with empty usage` → 409 == 409
   (tautological — both sides count the same set). **PASS**
4. `total_families == len(p2_ids | verified_ids)` → 807 == 807 (374 P2
   family_ids ∪ 714 verified JSONL family_ids; 281 of the verified
   train families overlap with P2). **PASS**
5. (future-guard) No family has both `quarantine` AND a P3-exclusive
   tag. Trivially true at builder time. **PASS**

The builder exited with code 0.

## Pairwise Disjoint Assertion (P2 tags)

Loaded the produced registry via `FamilyRegistry.from_path()` and ran:

```python
registry.assert_pairwise_disjoint(["p2_train", "p2_validation", "p2_frozen_v2"])
```

Result: **PASS** (no exception raised).

Per-tag family counts:
- `p2_train`: 224 families
- `p2_validation`: 75 families
- `p2_frozen_v2`: 75 families
- Total: 374 (matches `total_p2_used`)

The three P2 sets are pairwise disjoint because they come from three
disjoint lists in `data/p2-curriculum/family-partition.json`.

## Test Summary

```
$ python -m pytest tests/test_family_registry.py -v
============================= test session starts =============================
platform win32 -- Python 3.8.10, pytest-8.3.5, pluggy-1.5.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
collected 10 items

tests\test_family_registry.py ..........                                 [100%]

============================= 10 passed in 0.26s ==============================
```

**10/10 tests pass.**

## Concern (carried over from original Task 6)

The original Task 6 noted a `total_new_available` discrepancy (556 vs
brief's 526) caused by P2/quarantine set overlap (30 families in
both). The builder correctly uses input-derived expected values and
the empty-usage definition, so the discrepancy was a brief arithmetic
error, not a builder bug. The same logic applies to the redo: P2 and
quarantine sets still overlap (some P2-train families are also
quarantined), so `total_new_available = 409` is the count of families
with empty `usage` lists, not a simple `total - p2 - quarantine`.

## Commit (redo)

Redo commit supersedes e88146c (which ran on 955 soft-warning samples):

```
feat(p3): redo family registry on pad-then-verify verified set (v2.1)
```

Staged files:
- `data/family-registry.json` (regenerated)
- `.superpowers/sdd/task-6-report.md`
- `.superpowers/sdd/progress.md`

## No Other Concerns

- All 10 tests pass (including the builder-correctness test on
  synthetic inputs).
- The builder exits 0 on the real data.
- The pairwise disjoint assertion for P2 tags passes.
- The registry file is well-formed JSON with sorted keys and a
  trailing newline (stable diffs).
- The 409 new-available families is the pool Task 7 will claim 120
  `frozen_v3_candidate` tags from (409 > 120, sufficient).
