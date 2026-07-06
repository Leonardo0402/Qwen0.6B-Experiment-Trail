# Task 7 Brief: Frozen v3 Candidate Reservation

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Tasks 1-6 complete)
- Plan file: .superpowers/sdd/p3-plan.md (Global Constraints #6, #8 bind this task)
- Task 6 produced `data/family-registry.json` with 958 families: 374 P2-used, 58 quarantined, 556 with empty usage (available for P3 partition).
- Task 7 reserves 120 candidate families for Frozen v3 evaluation. Task 8 will generate samples on these candidates, verify them, and either freeze 80-100 (if enough pass) or FIX_FIRST.

## Goal
Build two new files:
1. `scripts/reserve_frozen_v3_candidates.py` — CLI that samples 120 candidates from the test source pool, claims them in the registry, writes `data/frozen-eval/v3/candidates.json`
2. `tests/test_reserve_frozen_v3_candidates.py` — test suite

And update one existing file:
3. `data/family-registry.json` — claim `"frozen_v3_candidate"` tag for the 120 chosen families (in-place update via load → claim → save)

And produce one new data file:
4. `data/frozen-eval/v3/candidates.json` — the candidate list (commit this)

## Source Pool Definition (binding)
The "test source pool" for Frozen v3 candidates is the set of families in `data/family-registry.json` that satisfy ALL of:
- `source_split == "test"` (MBPP test split, task_ids 11-510)
- `"quarantine"` NOT in `usage`
- `"p2_train"` NOT in `usage` AND `"p2_validation"` NOT in `usage` AND `"p2_frozen_v2"` NOT in `usage`
- `"frozen_v3_candidate"` NOT in `usage` (idempotency — re-running the script should not fail if candidates already claimed)
- `"frozen_v3"` NOT in `usage` (none should exist yet at Task 7, but assert defensively)

The implementer must compute this set from the registry and verify its size is >= 120. If < 120, abort with exit code 1 and a clear error message ("FIX_FIRST: only N test-pool families available, need 120").

Expected: 494 test-split verified families, minus the subset quarantined from test split (count varies — the implementer must compute it). The result should be well above 120 based on Task 5 audit (58 total quarantined across all 3 splits; even if all 58 were from test split, 494-58=436 >> 120).

## Sampling Algorithm (binding)
1. **Stratify by `difficulty`**: read each candidate family's source Sample from `data/external/mbpp/verified/test.jsonl` to get its `difficulty` (0-4). Group families by difficulty bucket.
2. **Proportional allocation**: allocate the 120 slots proportionally to each bucket's size. Round DOWN to integer; distribute the remainder (120 - sum_of_floors) one slot at a time to the largest buckets.
3. **Within-bucket sampling**: use `random.Random(seed=42)` to shuffle each bucket and take the first N families.
4. **Combine and sort**: concatenate all bucket samples, then sort the final 120-family list by `family_id` ascending (for stable output).
5. **Determinism check**: re-running the script with the same registry state must produce the IDENTICAL 120-family list. Use a fixed seed and sorted iteration order in step 3 (sort bucket before shuffling).

## candidates.json Schema
```json
{
  "generated_at": "<iso8601 utc>",
  "generator": "reserve_frozen_v3_candidates.py",
  "schema_version": 1,
  "seed": 42,
  "source_pool_size": <int>,
  "source_pool_definition": "families with source_split=='test' AND not quarantine AND not p2_* AND not frozen_v3*",
  "candidate_count": 120,
  "stratification": {
    "method": "proportional_by_difficulty",
    "buckets": {
      "0": {"pool_size": <int>, "allocated": <int>},
      "1": {"pool_size": <int>, "allocated": <int>},
      "2": {"pool_size": <int>, "allocated": <int>},
      "3": {"pool_size": <int>, "allocated": <int>},
      "4": {"pool_size": <int>, "allocated": <int>}
    }
  },
  "candidates": [
    "mbpp_fam_<n>",
    ...
  ]
}
```

- `candidates` is a sorted list of 120 `family_id` strings.
- `source_pool_size` is the count of test-pool families BEFORE sampling.
- Each bucket's `pool_size` is the count of families at that difficulty in the source pool. `allocated` is how many were sampled from that bucket.
- Sum of `allocated` across buckets MUST equal 120.
- Sum of `pool_size` across buckets MUST equal `source_pool_size`.

## Hard Assertions (binding — abort with exit 1 if any fail)
1. `len(candidates) == 120`
2. All candidates are in `data/family-registry.json` AND have `source_split == "test"`
3. No candidate has `"quarantine"` in its `usage` (BEFORE claiming — after claiming, this should still hold)
4. No candidate has any P2 tag (`p2_train`/`p2_validation`/`p2_frozen_v2`)
5. No candidate has `"frozen_v3"` tag
6. All candidates are unique (no duplicates)
7. After claiming, re-loading the registry and asserting `families_with_usage("frozen_v3_candidate")` returns exactly these 120 families (sorted)
8. Pairwise disjoint assertion: `registry.assert_pairwise_disjoint(["p2_train", "p2_validation", "p2_frozen_v2", "frozen_v3_candidate", "quarantine"])` does NOT raise (no family has two of these tags, except potentially P2+quarantine overlap which is from Task 6 — those families are excluded from candidates by definition because they have quarantine tag). Actually wait — the `quarantine` families are excluded from candidates by source pool definition. So `frozen_v3_candidate` should be disjoint from ALL of `p2_*`, `quarantine`. Verify this with `assert_pairwise_disjoint`.

## CLI
```
python scripts/reserve_frozen_v3_candidates.py \
    --registry data/family-registry.json \
    --mbpp-verified-dir data/external/mbpp/verified \
    --output-candidates data/frozen-eval/v3/candidates.json \
    --output-registry data/family-registry.json \
    --seed 42 \
    --count 120
```

Note: `--output-registry` is the SAME file as `--registry` (in-place update). The script loads the registry, claims the candidates, then writes back to the same path.

Exit codes: 0 = success, 1 = invariant violation, insufficient pool, or I/O error.

## Idempotency
Re-running the script must be safe:
- If candidates have already been claimed (`"frozen_v3_candidate"` already in their usage), the script should detect this and either:
  - (Preferred) produce the IDENTICAL candidate list (same seed → same sampling) and re-claiming is a no-op (`claim()` is idempotent), OR
  - (Acceptable) detect existing candidates and skip the claim step, just rewrite `candidates.json`.
- The simplest approach: filter the source pool to exclude families that ALREADY have `"frozen_v3_candidate"` tag, then sample 120 from the remaining pool. BUT this would produce DIFFERENT candidates on re-run if 120 are already claimed (the pool would be smaller). The PREFERRED approach is to seed the RNG with a deterministic function of the full pool (e.g. sort all 494 test-pool families by family_id, then `random.Random(42).shuffle()` and take first 120). This way, re-running always produces the same 120.

## tests/test_reserve_frozen_v3_candidates.py

Use synthetic registries + synthetic verified JSONL (in `tmp_path`). Tests:

1. `test_source_pool_filters_correctly`: registry with test/validation/train families + some quarantined + some P2 → source pool excludes quarantined and P2, includes only test+empty-usage.
2. `test_candidate_count_120`: with a synthetic pool of 200 test families, sampling 120 → exactly 120 candidates.
3. `test_seed_determinism`: same pool, same seed → identical candidate list across two runs.
4. `test_stratification_proportional`: pool with 100 difficulty-0 + 50 difficulty-1 + 50 difficulty-2 (total 200) → 120 candidates allocated proportionally: 60 + 30 + 30. Verify the per-bucket counts in `stratification.buckets`.
5. `test_candidates_sorted`: `candidates` list is sorted ascending by family_id.
6. `test_no_candidate_in_quarantine`: no candidate has quarantine tag in the registry (after claiming).
7. `test_no_candidate_in_p2`: no candidate has any P2 tag.
8. `test_registry_claim_persists`: after running the script, re-load `data/family-registry.json` (or the synthetic equivalent) and verify `families_with_usage("frozen_v3_candidate")` returns exactly the 120 candidates.
9. `test_pairwise_disjoint_holds`: `registry.assert_pairwise_disjoint(["p2_train","p2_validation","p2_frozen_v2","frozen_v3_candidate","quarantine"])` does not raise after claiming.
10. `test_insufficient_pool_aborts`: pool with only 100 test families (need 120) → script exits with code 1, prints clear error, does NOT write a partial candidates.json.

## Constraints
- Do NOT modify any existing files under `src/` other than creating new files.
- Do NOT modify any existing files under `tests/`, `scripts/`, `reports/` (other than the new script and test).
- The script MUST update `data/family-registry.json` in place (load → claim → save with `sort_keys=True, indent=2`).
- Create the `data/frozen-eval/v3/` directory if it does not exist.
- Use only stdlib + project-local imports. `from src.family_registry import FamilyRegistry, FamilyEntry` is required (Task 6 API). Do NOT import `src.validators`.
- Single git commit at the end.
- All 10 tests must pass.

## Test Verification
`cd e:\agent\Qwen\qwen3-code-lab ; python -m pytest tests/test_reserve_frozen_v3_candidates.py -v`

All 10 tests must pass.

## Run the Script for Real
After tests pass, run for real on the actual data:
```
python scripts/reserve_frozen_v3_candidates.py \
    --registry data/family-registry.json \
    --mbpp-verified-dir data/external/mbpp/verified \
    --output-candidates data/frozen-eval/v3/candidates.json \
    --output-registry data/family-registry.json \
    --seed 42 \
    --count 120
```

The script must succeed (exit 0) and:
- Write `data/frozen-eval/v3/candidates.json` with 120 candidates
- Update `data/family-registry.json` to add `"frozen_v3_candidate"` tag to those 120 families
- The 120 candidates must all have `source_split == "test"`
- After the run, `total_new_available` in the registry should decrease by 120 (from 556 to 436)

## Report File
Write to: `.superpowers/sdd/task-7-report.md`
Include:
- Files created/modified (3 paths: script, test, candidates.json + 1 modified: registry)
- source_pool_size (count of test-pool families before sampling)
- candidate_count (should be 120)
- Per-difficulty-bucket allocation table (pool_size → allocated)
- Confirmation that all 8 hard assertions pass
- Confirmation that pairwise disjoint holds
- Confirmation that registry total_new_available decreased by 120
- The commit hash
- Test summary
- Any concerns

## Commit
- Stage: `scripts/reserve_frozen_v3_candidates.py`, `tests/test_reserve_frozen_v3_candidates.py`, `data/frozen-eval/v3/candidates.json`, `data/family-registry.json`
- Commit message: `feat(p3): reserve 120 frozen v3 candidate families`
- Single commit.

## Working Directory
e:\agent\Qwen\qwen3-code-lab
