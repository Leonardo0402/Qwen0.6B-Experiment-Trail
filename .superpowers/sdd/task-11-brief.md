# Task 11 Brief: Balanced Generalist Train Data (30/20/20/30)

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Tasks 1-10 complete)
- Plan: .superpowers/sdd/p3-plan.md (Global Constraint #11, #18; Amendment A7)
- Task 10 produced `data/p3-curriculum/canonical-pool.jsonl` with 782 samples:
  - code=281, boundary=125, static_repair=148, execution_repair=228
  - 408 families (out of 425 partition families)

## Goal

Build one new file:
1. `scripts/build_balanced_generalist.py` — orchestrator that sub-samples the canonical pool at 30/20/20/30

And produce output files in `data/p3-curriculum/balanced-generalist/`:
2. `train.jsonl` — sub-sampled train data (~626 samples)
3. `validation.jsonl` — P3 validation samples (90 samples)
4. `manifest.json` — statistics + ratios + SHA256
5. `families.json` — family list (train + validation)
6. `token_audit.json` — token count summary
7. `rejected.jsonl` — pool samples not selected (with reasons)

Plus tests:
8. `tests/test_build_balanced_generalist.py` — test suite

## Sub-sampling strategy (binding)

### Target ratios: 30/20/20/30 (Code/Boundary/Static/Exec), ±3pp

The binding constraint is **boundary** (125 samples available, target 20%).
- Max total = 125 / 0.20 = **625**
- code: 625 × 0.30 = 188 (have 281 → sample 188)
- boundary: 625 × 0.20 = 125 (have 125 → use ALL)
- static: 625 × 0.20 = 125 (have 148 → sample 125)
- exec: 625 × 0.30 = 188 (have 228 → sample 188)
- **Total: 626** (188+125+125+188)

Actual ratios: 30.0% / 20.0% / 20.0% / 30.0% — exactly on target ✓

### Sampling determinism
- Seed: 42 (consistent with all prior tasks)
- For each variant_type bucket: sort available sample_ids ascending, then
  `random.Random(42).sample(sorted_ids, target_count)`
- This ensures reproducibility regardless of dict insertion order

### Stratification note
The sub-sampling is by variant_type bucket (not by family). This means some
families might have multiple samples selected (e.g., a family with both code
and boundary variants will have both selected). This is acceptable — the
per-family cap (7) was already applied in Task 10, so no family is
over-represented.

## Validation data (binding)

Load from verified MBPP data for the 90 P3 validation families:
- 61 from `data/external/mbpp/verified/validation.jsonl` (validation split)
- 29 from `data/external/mbpp/verified/test.jsonl` (test split, supplemented to validation in Task 9)

Filter by `family_id` in `data/p3-curriculum/family-partition.json::p3_validation.family_ids`.

Each validation sample: set `variant_type = "code"`, `bug_type = None`.
(Validation samples are code_generation only — no variants for validation.)

## Output files

### train.jsonl
626 Sample objects, one per line, sorted by `sample_id` ascending.

### validation.jsonl
90 Sample objects, one per line, sorted by `sample_id` ascending.

### manifest.json
```json
{
  "schema_version": 1,
  "generated_at": "<ISO 8601 UTC>",
  "generator": "build_balanced_generalist.py",
  "candidate_type": "balanced_generalist",
  "seed": 42,
  "target_ratios": {"code": 0.30, "boundary": 0.20, "static_repair": 0.20, "execution_repair": 0.30},
  "actual_ratios": {"code": 0.300, "boundary": 0.200, "static_repair": 0.200, "execution_repair": 0.300},
  "ratio_tolerance_pp": 3,
  "ratio_within_tolerance": true,
  "train": {
    "count": 626,
    "variant_distribution": {"code": 188, "boundary": 125, "static_repair": 125, "execution_repair": 188},
    "family_count": "<count of unique families in train>",
    "sha256": "<SHA256 of train.jsonl>"
  },
  "validation": {
    "count": 90,
    "family_count": 90,
    "sha256": "<SHA256 of validation.jsonl>"
  },
  "families": {
    "train_family_count": "<count>",
    "validation_family_count": 90,
    "total_family_count": "<count>",
    "train_validation_disjoint": true
  },
  "pool_source": {
    "path": "data/p3-curriculum/canonical-pool.jsonl",
    "total_pool_samples": 782,
    "samples_selected": 626,
    "samples_rejected": 156
  }
}
```

### families.json
```json
{
  "schema_version": 1,
  "candidate_type": "balanced_generalist",
  "train_family_ids": ["<sorted list>"],
  "validation_family_ids": ["<sorted list of 90>"],
  "total_family_count": "<count>",
  "shared_with": "repair_specialist"
}
```

### token_audit.json
```json
{
  "schema_version": 1,
  "candidate_type": "balanced_generalist",
  "train": {
    "total_samples": 626,
    "total_tokens": "<sum of token counts>",
    "mean_tokens_per_sample": "<float>",
    "max_tokens": "<int>",
    "min_tokens": "<int>",
    "by_variant_type": {
      "code": {"count": 188, "total_tokens": "<int>", "mean": "<float>"},
      "boundary": {"count": 125, "total_tokens": "<int>", "mean": "<float>"},
      "static_repair": {"count": 125, "total_tokens": "<int>", "mean": "<float>"},
      "execution_repair": {"count": 188, "total_tokens": "<int>", "mean": "<float>"}
    }
  },
  "validation": {
    "total_samples": 90,
    "total_tokens": "<int>",
    "mean_tokens_per_sample": "<float>"
  }
}
```

Token count method: count whitespace-split tokens in
`instruction + " " + target_code + " " + public_tests + " " + hidden_tests`.
(Use `len(text.split())` as a simple token approximation.)

### rejected.jsonl
Pool samples NOT selected for train, one JSON object per line:
```json
{"sample_id": "<id>", "family_id": "<fid>", "variant_type": "<vt>", "rejection_reason": "ratio_balance_excess"}
```
Count: 782 - 626 = 156 rejected.

## Hard gates (binding — abort exit 1 if any fail)

1. Train count = 626 (±1 for rounding)
2. All 4 variant ratios within ±3pp of target (30/20/20/30)
3. Validation count = 90
4. Train ∩ Validation family_ids = ∅ (pairwise disjoint)
5. Train ∩ frozen_v3 family_ids = ∅
6. Validation ∩ frozen_v3 family_ids = ∅
7. All train samples have variant_type set (not None)
8. All train samples have verified=True
9. No duplicate sample_ids in train
10. No duplicate sample_ids in validation

## Existing infrastructure (use these)
1. `src/sample_pool.py::SamplePool` — use `SamplePool.from_jsonl(path)` to load the pool
2. `data/p3-curriculum/canonical-pool.jsonl` — the pool (read-only)
3. `data/p3-curriculum/family-partition.json` — partition family_ids
4. `data/external/mbpp/verified/{validation,test}.jsonl` — validation samples
5. `src/schemas.py::Sample` — use `Sample.from_json_line(line)` and `sample.to_json_line()`

## Tests (binding — minimum 8 tests)

1. `test_train_count` — train.jsonl has 626 samples
2. `test_validation_count` — validation.jsonl has 90 samples
3. `test_ratio_within_tolerance` — all 4 variant ratios within ±3pp of 30/20/20/30
4. `test_train_validation_disjoint` — no family in both train and validation
5. `test_train_frozen_v3_disjoint` — no frozen_v3 family in train
6. `test_all_variant_type_set` — no sample with variant_type=None in train
7. `test_no_duplicate_sample_ids` — no duplicate sample_ids in train or validation
8. `test_deterministic_sampling` — re-running produces same sample_ids
9. `test_rejected_count` — rejected.jsonl has 156 records
10. `test_manifest_consistency` — manifest counts match actual files

## Important notes
- Use `from __future__ import annotations` at top of all .py files
- The `data/p3-curriculum/balanced-generalist/` directory needs to be created
- Sample objects use pydantic — use `model_copy(update={...})` to set fields
- For SHA computation: read entire file as bytes, compute SHA256
- The train.jsonl is NOT immutable (unlike frozen v3). It can be rebuilt.
- Validation samples are code-only (no variants). This is intentional.
- The `shared_with: "repair_specialist"` field in families.json documents that
  Task 12 uses the same partition (Global Constraint #18).

## Commit message
`feat(p3): build balanced generalist train data (30/20/20/30, 626 train + 90 val)`

## Deviations / clarifications

1. **Actual yield 626 < 2300-3100 estimate**. Per Amendment A7, "404×7 is
   capacity estimate ONLY, actual yield from verified+deduped pool". The
   canonical pool has 782 samples; after sub-sampling to achieve 30/20/20/30
   ±3pp, train = 626. No duplication to pad (A7 binding).

2. **Validation is code-only**. The 90 validation families each have 1
   code_generation sample. No variants for validation — this is a sanity
   check set, not a ratio-matched eval set.

3. **Some families have multiple train samples**. A family with both code and
   boundary variants will have both selected (if both pass sub-sampling).
   This is acceptable — the per-family cap (7) was already applied in Task 10.

4. **boundary is the binding constraint**. With only 125 boundary samples
   available, the total is capped at 625-626. If more boundary samples were
   available (e.g., by generating boundary variants for P3 new train families),
   the total could be higher. This is documented for the Readiness Gate.
