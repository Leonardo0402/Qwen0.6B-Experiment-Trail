# Task 10 Brief: Canonical Verified Sample Pool

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Tasks 1-9 complete)
- Plan: .superpowers/sdd/p3-plan.md (Global Constraint #10; Amendment A6)
- Task 9 produced `data/p3-curriculum/family-partition.json` with:
  - p3_train_new: 219 families (from test split, tagged `p3_train`)
  - p3_train_replay: 206 families (from p2_train, tagged `p3_train_replay`)

## Goal

Build two new files:
1. `src/sample_pool.py` — SamplePool class (load, dedup, index, cap, query)
2. `scripts/build_sample_pool.py` — orchestrator that builds the pool from sources

And produce one new data file:
3. `data/p3-curriculum/canonical-pool.jsonl` — the deduplicated, capped pool

Plus one manifest:
4. `data/p3-curriculum/canonical-pool-manifest.json` — pool statistics

Plus tests:
5. `tests/test_sample_pool.py` — unit tests for SamplePool
6. `tests/test_build_sample_pool.py` — integration tests for the orchestrator

## Pool sources (binding per Amendment A6 — NO direct Stage manifest concatenation)

### Source 1: P2 Replay Samples (from 206 replay families)
Load from these P2 stage train.jsonl files (DO NOT load stage3-repair-v3 — it's a remix):
- `data/p2-curriculum/stage1-code/train.jsonl` (84 samples)
- `data/p2-curriculum/stage2-boundary/train.jsonl` (280 samples)
- `data/p2-curriculum/stage3-repair/train.jsonl` (560 samples)

**Filter**: only include samples whose `family_id` is in the 206 replay families
(from `data/p3-curriculum/family-partition.json::p3_train_replay.family_ids`).
This EXCLUDES the 18 quarantined p2_train families.

### Source 2: P3 New Train Samples (from 219 train_new families)
Load from `data/external/mbpp/verified/test.jsonl` (365 samples total, but only 219
belong to train_new families).

**Filter**: only include samples whose `family_id` is in the 219 train_new families
(from `data/p3-curriculum/family-partition.json::p3_train_new.family_ids`).

These are raw code_generation samples with `variant_type=None` (needs normalization).

## variant_type Normalization (binding)

The P2 samples don't have `variant_type` set (it was added in Task 2). The pool
must normalize variant_type for EVERY sample based on:

| Condition | variant_type |
|---|---|
| `task_type == "code_generation"` AND `"boundary" in skill_tags` OR `sample_id` ends with `_boundary` | `"boundary"` |
| `task_type == "code_generation"` AND NOT boundary | `"code"` |
| `task_type == "static_repair"` | `"static_repair"` |
| `task_type == "execution_repair"` | `"execution_repair"` |

Also extract `bug_type` from `sample_id` using regex `.*_(sr|er)_(.+)$` → group 2.
If no match, `bug_type = None`.

For P3 new train samples (from verified MBPP): set `variant_type = "code"`,
`bug_type = None`, `source_split = "test"`.

Use `sample.model_copy(update={"variant_type": ..., "bug_type": ...})` to set fields.

## Pool construction pipeline (binding per Amendment A6)

```
1. Load all source samples (P2 stages + P3 verified)
2. Filter to partition families (206 replay + 219 new = 425 families)
3. Normalize variant_type + bug_type on every sample
4. Deduplicate by sample_id (first occurrence wins; log duplicates)
5. Index by (family_id, variant_type, bug_type)
6. Apply per-family contribution cap (default=7, configurable via --cap)
   - If a family has > cap samples, keep the first `cap` by sample_id ascending
   - Log capped families and excess counts
7. Write canonical-pool.jsonl (one Sample per line, sorted by sample_id)
8. Write canonical-pool-manifest.json (statistics)
```

## Per-family cap (binding)

Default cap = 7. This means at most 7 samples per family in the pool.
Rationale: 425 families × 7 = 2975 (within 2300-3100 capacity estimate).
Actual yield will be lower because not all families have 7 variants.

The cap is a MAX, not a MIN. Families with fewer samples are included as-is.
Families with more than `cap` samples have excess samples DROPPED (not duplicated).

## Output: data/p3-curriculum/canonical-pool.jsonl

One Sample JSON per line, sorted by `sample_id` ascending. Each sample is a
complete Sample object with `variant_type` and `bug_type` set.

## Output: data/p3-curriculum/canonical-pool-manifest.json

```json
{
  "schema_version": 1,
  "generated_at": "<ISO 8601 UTC>",
  "generator": "build_sample_pool.py",
  "sources": {
    "p2_stage1_code": {"path": "data/p2-curriculum/stage1-code/train.jsonl", "loaded": 84, "after_filter": "<count>"},
    "p2_stage2_boundary": {"path": "data/p2-curriculum/stage2-boundary/train.jsonl", "loaded": 280, "after_filter": "<count>"},
    "p2_stage3_repair": {"path": "data/p2-curriculum/stage3-repair/train.jsonl", "loaded": 560, "after_filter": "<count>"},
    "p3_verified_test": {"path": "data/external/mbpp/verified/test.jsonl", "loaded": 365, "after_filter": 219}
  },
  "total_loaded": "<sum of loaded>",
  "total_after_family_filter": "<sum of after_filter>",
  "total_after_dedup": "<count after removing duplicate sample_ids>",
  "duplicates_removed": "<count>",
  "total_after_cap": "<final count>",
  "families_capped": "<count of families that had samples dropped>",
  "samples_dropped_by_cap": "<count>",
  "per_family_cap": 7,
  "family_count": 425,
  "variant_distribution": {
    "code": "<count>",
    "boundary": "<count>",
    "static_repair": "<count>",
    "execution_repair": "<count>"
  },
  "bug_type_distribution": {"<bug_type>": "<count>", ...},
  "family_distribution": {
    "min_samples_per_family": "<int>",
    "max_samples_per_family": "<int>",
    "mean_samples_per_family": "<float>",
    "median_samples_per_family": "<float>"
  },
  "pool_sha256": "<SHA256 of canonical-pool.jsonl>"
}
```

## SamplePool class (src/sample_pool.py)

```python
class SamplePool:
    """Canonical verified sample pool with dedup, index, and per-family cap."""

    def __init__(self):
        self._samples: list[Sample] = []
        self._index: dict[str, list[int]] = {}  # sample_id -> [positions]
        self._family_index: dict[str, list[int]] = {}  # family_id -> [positions]

    def add(self, sample: Sample) -> bool:
        """Add a sample. Returns True if added, False if duplicate sample_id."""

    def dedup(self) -> int:
        """Remove duplicates by sample_id (keep first). Returns count removed."""

    def normalize_variant_type(self) -> int:
        """Set variant_type + bug_type on all samples. Returns count normalized."""

    def apply_family_cap(self, cap: int = 7) -> int:
        """Cap samples per family. Returns count dropped."""

    def filter_families(self, family_ids: set[str]) -> int:
        """Keep only samples whose family_id is in the set. Returns count kept."""

    def to_jsonl(self, path: Path) -> None:
        """Write samples sorted by sample_id ascending."""

    @classmethod
    def from_jsonl(cls, path: Path) -> "SamplePool":
        """Load from JSONL file."""

    def stats(self) -> dict:
        """Return statistics dict for manifest."""

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)
```

## Tests (binding — minimum 8 tests)

### tests/test_sample_pool.py (unit tests for SamplePool)
1. `test_add_and_dedup` — add samples with duplicate sample_id, dedup removes the right count
2. `test_normalize_variant_type_code` — code_generation without boundary → variant_type="code"
3. `test_normalize_variant_type_boundary` — code_generation with "boundary" in skill_tags → variant_type="boundary"
4. `test_normalize_variant_type_static_repair` — static_repair → variant_type="static_repair" + bug_type extracted
5. `test_normalize_variant_type_execution_repair` — execution_repair → variant_type="execution_repair" + bug_type extracted
6. `test_apply_family_cap` — family with 10 samples, cap=7, drops 3
7. `test_filter_families` — keeps only samples in the family set
8. `test_to_jsonl_sorted` — output is sorted by sample_id

### tests/test_build_sample_pool.py (integration tests)
9. `test_pool_loads_from_all_sources` — pool has samples from P2 stages + P3 verified
10. `test_pool_no_duplicate_sample_ids` — all sample_ids unique
11. `test_pool_variant_distribution` — all 4 variant types present (code/boundary/static/exec)
12. `test_pool_family_cap_enforced` — no family has > 7 samples
13. `test_pool_manifest_correct` — manifest counts match actual pool
14. `test_pool_only_partition_families` — no family outside the 425 partition families

## Hard gates (binding — abort with exit 1 if any fail)

1. Pool total >= 400 (minimum viable for training)
2. All variant_type values are in {code, boundary, static_repair, execution_repair}
3. No duplicate sample_ids
4. No family has > cap samples
5. All families in pool are in the 425 partition families
6. No quarantined family in pool

## Existing infrastructure (use these — do NOT reimplement)
1. `src/schemas.py::Sample` — use `Sample.from_json_line(line)` and `sample.to_json_line()` for I/O
2. `data/p3-curriculum/family-partition.json` — partition family lists (read-only)
3. P2 stage manifests in `data/p2-curriculum/stage*/manifest.json` (for SHA verification if needed)
4. `data/external/mbpp/verified/test.jsonl` — P3 verified samples

## Important notes
- Use `from __future__ import annotations` at top of all .py files
- The pool is the INPUT to Tasks 11/12. Do NOT sample at ratios here — just build the pool.
- The pool is NOT immutable (unlike frozen v3). It can be rebuilt if sources change.
- If actual yield < 2300, document it in the manifest. Do NOT duplicate samples to pad.
- The `canonical-pool.jsonl` file can be large (~1000+ lines). Use streaming I/O.
- For SHA computation: read the entire file as bytes, compute SHA256.

## Commit message
`feat(p3): build canonical verified sample pool (dedup + index + per-family cap)`

## Deviations / clarifications

1. **P3 new train samples are code-only**. The verified MBPP samples for the 219
   train_new families are raw code_generation (no boundary/repair variants).
   Variant generation for P3 new train is OUT OF SCOPE for Task 10. The pool
   will have code samples from P3 new train and all 4 variant types from P2 replay.
   If the variant distribution is skewed (too few static/exec), Tasks 11/12 will
   document the deviation and the Readiness Gate (Task 14) will assess sufficiency.

2. **stage3-repair-v3 is EXCLUDED**. It's a remix of stage1+2+3 samples and would
   introduce duplicates. The canonical pool uses the ORIGINAL stage outputs only.

3. **Duplicate sample_ids across stages**. Some samples might appear in multiple
   stages (e.g., a code sample in stage1 might also appear in stage3-repair's
   code_generation subset). Dedup by sample_id keeps the FIRST occurrence (stage1
   before stage2 before stage3) and logs the duplicates.

4. **Per-family cap is a MAX, not a MIN**. Families with 1-2 samples are included
   as-is. The cap only drops excess samples from families with > 7.
