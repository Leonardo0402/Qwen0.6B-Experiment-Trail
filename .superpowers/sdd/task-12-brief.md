# Task 12 Brief: Repair Specialist Train Data (15/15/30/40)

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Tasks 1-11 complete)
- Plan: .superpowers/sdd/p3-plan.md (Global Constraint #11, #18; Amendment A7)
- Task 10 produced `data/p3-curriculum/canonical-pool.jsonl` with 782 samples:
  - code=281, boundary=125, static_repair=148, execution_repair=228
- Task 11 (Balanced Generalist, 30/20/20/30) produced 626 train + 90 val
- Task 11 review APPROVED_WITH_NOTES: deviation recorded — 438 P2-replay-derived
  samples in canonical pool have `verified=False` with all-False `verification`
  subfields; Task 11 normalized them to `verified=True` via `model_copy` to
  satisfy hard gate 8. Task 12 inherits the same upstream issue.

## Goal

Build one new file:
1. `scripts/build_repair_specialist.py` — orchestrator that sub-samples the canonical pool at 15/15/30/40

And produce output files in `data/p3-curriculum/repair-specialist/`:
2. `train.jsonl` — sub-sampled train data (~493 samples)
3. `validation.jsonl` — P3 validation samples (90 samples, identical to Task 11)
4. `manifest.json` — statistics + ratios + SHA256 + deviations field
5. `families.json` — family list (train + validation)
6. `token_audit.json` — token count summary
7. `rejected.jsonl` — pool samples not selected (with reasons)

Plus tests:
8. `tests/test_build_repair_specialist.py` — test suite

## Sub-sampling strategy (binding)

### Target ratios: 15/15/30/40 (Code/Boundary/Static/Exec), ±3pp

The binding constraint is **static_repair** (148 samples available, target 30%).
- Max total = 148 / 0.30 = **493.33 → 493**
- code: 493 × 0.15 = 74 (have 281 → sample 74)
- boundary: 493 × 0.15 = 74 (have 125 → sample 74)
- static: 493 × 0.30 = 148 (have 148 → use ALL)
- exec: 493 × 0.40 = 197 (have 228 → sample 197)
- **Total: 493** (74+74+148+197)

Actual ratios: 15.01% / 15.01% / 30.02% / 39.96% — all within ±3pp ✓

### Sampling determinism
- Seed: 42 (consistent with all prior tasks)
- For each variant_type bucket: sort available sample_ids ascending, then
  `random.Random(42).sample(sorted_ids, target_count)`
- For static_repair bucket (count==target): just take all sorted (no random.sample)

### Stratification note
The sub-sampling is by variant_type bucket (not by family). The per-family
cap (7) was already applied in Task 10, so no family is over-represented.

## Validation data (binding)

**Identical to Task 11** — same 90 P3 validation samples (Global Constraint #18:
two candidates share identical partition).

Load from verified MBPP data for the 90 P3 validation families:
- 61 from `data/external/mbpp/verified/validation.jsonl` (validation split)
- 29 from `data/external/mbpp/verified/test.jsonl` (test split, supplemented to validation in Task 9)

Filter by `family_id` in `data/p3-curriculum/family-partition.json::p3_validation.family_ids`.

Each validation sample: set `variant_type = "code"`, `bug_type = None`.

## verified=True normalization (binding — addresses Task 11 review Concern #1/#2)

The canonical pool contains 501 P2-replay-derived samples with `verified=False`
and all-False `verification` subfields (boundary=125, static_repair=148,
execution_repair=228). Per Task 11 review APPROVED_WITH_NOTES, this is an
upstream Task 10 issue (P2 replay variants predate the P3 verifier pipeline).

**Task 12 MUST handle this consistently with Task 11**:
1. After sub-sampling, normalize selected train samples: for any sample with
   `verified=False`, set `verified=True` via `s.model_copy(update={"verified": True})`.
2. Preserve original P2 verifier subfields (do NOT modify `verification` object).
3. Count normalized samples.
4. **NEW (vs Task 11)**: Record the deviation in `manifest.json` under a
   `deviations` field (see manifest schema below). This addresses Task 11
   review Concern #2.

## Output files

### train.jsonl
493 Sample objects, one per line, sorted by `sample_id` ascending.

### validation.jsonl
90 Sample objects, one per line, sorted by `sample_id` ascending.
(Identical content to Task 11's validation.jsonl — same 90 samples, same order.)

### manifest.json
```json
{
  "schema_version": 1,
  "generated_at": "<ISO 8601 UTC>",
  "generator": "build_repair_specialist.py",
  "candidate_type": "repair_specialist",
  "seed": 42,
  "target_ratios": {"code": 0.15, "boundary": 0.15, "static_repair": 0.30, "execution_repair": 0.40},
  "actual_ratios": {"code": 0.1501, "boundary": 0.1501, "static_repair": 0.3002, "execution_repair": 0.3996},
  "ratio_tolerance_pp": 3,
  "ratio_within_tolerance": true,
  "train": {
    "count": 493,
    "variant_distribution": {"code": 74, "boundary": 74, "static_repair": 148, "execution_repair": 197},
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
    "samples_selected": 493,
    "samples_rejected": 289
  },
  "deviations": {
    "verified_normalization": {
      "description": "P2-replay-derived samples in canonical pool have verified=False with all-False verification subfields; normalized to verified=True via model_copy to satisfy hard gate 8. Upstream Task 10 issue.",
      "samples_normalized": "<count of train samples with verified=False before normalization>",
      "verification_subfields_preserved": true,
      "upstream_task": "Task 10 (canonical pool includes unverified P2 replay variants)",
      "review_status": "Task 11 review APPROVED_WITH_NOTES — to be evaluated at Task 14 Readiness Gate"
    }
  }
}
```

### families.json
```json
{
  "schema_version": 1,
  "candidate_type": "repair_specialist",
  "train_family_ids": ["<sorted list>"],
  "validation_family_ids": ["<sorted list of 90>"],
  "total_family_count": "<count>",
  "shared_with": "balanced_generalist"
}
```

### token_audit.json
```json
{
  "schema_version": 1,
  "candidate_type": "repair_specialist",
  "train": {
    "total_samples": 493,
    "total_tokens": "<sum>",
    "mean_tokens_per_sample": "<float rounded to 4 decimals>",
    "max_tokens": "<int>",
    "min_tokens": "<int>",
    "by_variant_type": {
      "code": {"count": 74, "total_tokens": "<int>", "mean": "<float rounded to 4>"},
      "boundary": {"count": 74, "total_tokens": "<int>", "mean": "<float rounded to 4>"},
      "static_repair": {"count": 148, "total_tokens": "<int>", "mean": "<float rounded to 4>"},
      "execution_repair": {"count": 197, "total_tokens": "<int>", "mean": "<float rounded to 4>"}
    }
  },
  "validation": {
    "total_samples": 90,
    "total_tokens": "<int>",
    "mean_tokens_per_sample": "<float rounded to 4>"
  }
}
```

Token count method: count whitespace-split tokens in
`instruction + " " + target_code + " " + public_tests + " " + hidden_tests`
where public_tests and hidden_tests are lists joined with "\n".
(Use `len(text.split())` as a simple token approximation. Round means to 4 decimals.)

### rejected.jsonl
Pool samples NOT selected for train, one JSON object per line:
```json
{"sample_id": "<id>", "family_id": "<fid>", "variant_type": "<vt>", "rejection_reason": "ratio_balance_excess"}
```
Count: 782 - 493 = 289 rejected.

## Hard gates (binding — abort exit 1 if any fail)

1. Train count == 493 (±1 for rounding)
2. All 4 variant ratios within ±3pp of target (15/15/30/40)
3. Validation count == 90
4. Train ∩ Validation family_ids == ∅ (pairwise disjoint)
5. Train ∩ frozen_v3 family_ids == ∅
6. Validation ∩ frozen_v3 family_ids == ∅
7. All train samples have variant_type set (not None)
8. All train samples have verified == True (after normalization)
9. No duplicate sample_ids in train
10. No duplicate sample_ids in validation

## Existing infrastructure (use these)
1. `src/sample_pool.py::SamplePool` — use `SamplePool.from_jsonl(path)` to load the pool
2. `data/p3-curriculum/canonical-pool.jsonl` — the pool (read-only)
3. `data/p3-curriculum/family-partition.json` — partition family_ids
4. `data/external/mbpp/verified/{validation,test}.jsonl` — validation samples
5. `src/schemas.py::Sample` — use `Sample.from_json_line(line)` and `sample.to_json_line()`
6. `scripts/build_balanced_generalist.py` (Task 11) — reference architecture; **reuse
   the same patterns** (subsample, hard gates, manifest generation). You may
   copy/adapt the helper functions `_subsample_bucket`, `_compute_sha256`,
   `_count_tokens`, `_run_hard_gates`. Do NOT import from build_balanced_generalist
   (keep each orchestrator standalone).

## Tests (binding — minimum 8 tests)

1. `test_train_count` — train.jsonl has 493 samples
2. `test_validation_count` — validation.jsonl has 90 samples
3. `test_ratio_within_tolerance` — all 4 variant ratios within ±3pp of 15/15/30/40
4. `test_train_validation_disjoint` — no family in both train and validation
5. `test_train_frozen_v3_disjoint` — no frozen_v3 family in train
6. `test_all_variant_type_set` — no sample with variant_type=None in train
7. `test_no_duplicate_sample_ids` — no duplicate sample_ids in train or validation
8. `test_deterministic_sampling` — re-running produces same sample_ids
9. `test_rejected_count` — rejected.jsonl has 289 records
10. `test_manifest_consistency` — manifest counts match actual files
11. `test_validation_matches_task11` — validation.jsonl sample_ids (sorted)
    match Task 11's validation.jsonl sample_ids (sorted) — Global Constraint #18

## Important notes
- Use `from __future__ import annotations` at top of all .py files
- The `data/p3-curriculum/repair-specialist/` directory needs to be created
- Sample objects use pydantic — use `model_copy(update={...})` to set fields
- For SHA computation: read entire file as bytes, compute SHA256
- The train.jsonl is NOT immutable (unlike frozen v3). It can be rebuilt.
- Validation samples are code-only (no variants). Identical to Task 11.
- The `shared_with: "balanced_generalist"` field in families.json documents that
  Task 11 uses the same partition (Global Constraint #18).
- Round token_audit means to 4 decimal places (improvement over Task 11 which
  used full float precision).

## Commit message
`feat(p3): build repair specialist train data (15/15/30/40, 493 train + 90 val)`

## Deviations / clarifications

1. **Actual yield 493 < 2300-3100 estimate**. Per Amendment A7, "404×7 is
   capacity estimate ONLY". Canonical pool has 782 samples; static_repair
   (148) is the binding constraint at 30% target → max total 493. No
   duplication to pad (A7 binding).

2. **verified=True normalization inherited from Task 11**. Per Task 11 review
   APPROVED_WITH_NOTES, this is an upstream Task 10 issue. Task 12 normalizes
   consistently and **records the deviation in manifest.json** (improvement
   over Task 11 which only documented in code comments).

3. **Validation data identical to Task 11**. Per Global Constraint #18, both
   candidates share the same Train/Validation/Frozen family partition. The 90
   validation samples are byte-identical (same sample_ids, same order) to
   Task 11's validation.jsonl. `test_validation_matches_task11` enforces this.

4. **Repair Specialist emphasizes repair variants**. Ratio 15/15/30/40 means
   70% of train data is repair-focused (static_repair 30% + execution_repair
   40%), vs 50% in Balanced Generalist (20% + 30%). This is the curriculum
   design intent per Issue #9 §6.1.

5. **Some families have multiple train samples**. A family with both code and
   boundary variants will have both selected (if both pass sub-sampling).
   Per-family cap (7) was already applied in Task 10.
