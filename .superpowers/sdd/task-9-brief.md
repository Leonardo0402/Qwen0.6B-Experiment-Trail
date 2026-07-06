# Task 9 Brief: P3 Validation + Train Family Partition

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Tasks 1-8 complete, Task 8 review APPROVED)
- Plan file: .superpowers/sdd/p3-plan.md (Global Constraints #8, #9, #15 bind this task;
  Amendment A4 binds pairwise disjoint enforcement)
- Frozen v3 is IMMUTABLE: 100 families frozen, sha_lock locked (commit c2ac270).
  These families MUST NOT appear in any P3 Train or Validation split.

## Registry state at task start (verified 2026-07-04)

| Tag | Count | Source split breakdown |
|---|---|---|
| frozen_v3 | 100 | test (100) |
| p2_train | 224 | train (224) |
| p2_validation | 75 | validation (75) |
| p2_frozen_v2 | 75 | mixed |
| quarantine | 50 | test=17, validation=7, train=26 |
| available (empty usage) | 309 | test=248, validation=61 |

**IMPORTANT**: 26 of the 224 p2_train families are ALSO tagged quarantine. Per
Amendment A9, quarantine families are EXCLUDED from all P3 splits (including
replay). Therefore P3 Train replay = 224 - 26 = **198 families**.

## Goal

Build two new files:
1. `scripts/build_p3_partition.py` — partition orchestrator
2. `tests/test_build_p3_partition.py` — test suite

And produce one new data file (committed):
3. `data/p3-curriculum/family-partition.json` — the canonical partition record

Also modify:
4. `data/family-registry.json` — claim `p3_validation` / `p3_train` /
   `p3_train_replay` tags on the partitioned families

## Partition specification (binding)

### P3 Validation (target: ~90 families)
1. Take ALL 61 available (empty-usage) families from the `validation` source
   split. These are the "official validation split" families.
2. If 61 < 90, supplement from the `test` source split: sample
   `90 - 61 = 29` families from the 248 available test families, using
   `seed=42` for reproducibility.
3. Total validation = 61 + 29 = **90 families**.
4. All validation families MUST be:
   - Not in quarantine
   - Not in frozen_v3
   - Not in p2_frozen_v2
   - Not in p2_train / p2_validation (we're building NEW validation, not
     replaying P2 validation)

### P3 Train — NEW (target: >= 180 families)
1. From the remaining available test families (248 - 29 supplemented to
   validation = 219 families), take ALL of them.
2. 219 >= 180 ✓
3. All new-train families MUST be:
   - Not in quarantine
   - Not in frozen_v3
   - Not in p2_frozen_v2
   - Not in p2_train / p2_validation (new families only, no P2 overlap here;
     P2 overlap is handled by the replay set below)

### P3 Train — REPLAY (target: 198 families)
1. From the 224 `p2_train` families, EXCLUDE the 26 that are also tagged
   `quarantine`. Remaining = **198 families**.
2. These 198 families are the ONLY allowed historical intersection (per
   Amendment A4 / Global Constraint #9).
3. They keep their `p2_train` tag AND get a new `p3_train_replay` tag.

### Total P3 Train families
- New: 219
- Replay: 198
- **Total: 417** (>= 404 required ✓)

## Pairwise disjoint enforcement (binding — Amendment A4)

After partitioning, call `registry.assert_pairwise_disjoint()` on this
list of tags:

```python
usages = [
    "frozen_v3",
    "p3_validation",
    "p3_train",
    "p3_train_replay",
]
whitelist = [
    ("p3_train_replay", "p2_train"),  # the 198 replay families are intentionally reused
]
registry.assert_pairwise_disjoint(usages, whitelist=whitelist)
```

This checks ALL C(4,2) = 6 pairs:
- frozen_v3 ∩ p3_validation = ∅
- frozen_v3 ∩ p3_train = ∅
- frozen_v3 ∩ p3_train_replay = ∅
- p3_validation ∩ p3_train = ∅
- p3_validation ∩ p3_train_replay = ∅
- p3_train ∩ p3_train_replay = ∅

The whitelist allows the (p3_train_replay, p2_train) overlap — the 198
families intentionally reused from P2 train.

### Additional disjointness checks (binding)
Also assert (hard, with explicit error messages):
- `frozen_v3 ∩ quarantine = ∅` (frozen v3 must not contain quarantined families)
- `p3_validation ∩ quarantine = ∅`
- `p3_train ∩ quarantine = ∅`
- `p3_train_replay ∩ quarantine = ∅` (the 26 quarantined p2_train families
  were EXCLUDED from replay)
- `p3_validation ∩ p2_frozen_v2 = ∅`
- `p3_train ∩ p2_frozen_v2 = ∅`
- `p3_train_replay ∩ p2_frozen_v2 = ∅`
- `frozen_v3 ∩ p2_frozen_v2 = ∅`

## Output file: data/p3-curriculum/family-partition.json

```json
{
  "schema_version": 1,
  "generated_at": "<ISO 8601 UTC>",
  "generator": "build_p3_partition.py",
  "seed": 42,
  "p3_validation": {
    "count": 90,
    "from_validation_split": 61,
    "from_test_split_supplement": 29,
    "family_ids": ["<sorted list of 90 family_ids>"]
  },
  "p3_train_new": {
    "count": 219,
    "source": "test split (remaining after validation supplement)",
    "family_ids": ["<sorted list of 219 family_ids>"]
  },
  "p3_train_replay": {
    "count": 198,
    "source": "p2_train minus quarantine",
    "excluded_quarantine_count": 26,
    "family_ids": ["<sorted list of 198 family_ids>"]
  },
  "p3_train_total": 417,
  "frozen_v3_count": 100,
  "pairwise_disjoint": {
    "tags_checked": ["frozen_v3", "p3_validation", "p3_train", "p3_train_replay"],
    "pairs_checked": 6,
    "whitelist": [["p3_train_replay", "p2_train"]],
    "whitelist_intersection_count": 198,
    "result": "PASS"
  },
  "quarantine_exclusion": {
    "quarantine_total": 50,
    "excluded_from_validation": 7,
    "excluded_from_train_new": 17,
    "excluded_from_train_replay": 26,
    "excluded_from_frozen_v3": 0
  }
}
```

## Registry updates (binding)

For each family in the partition, claim the appropriate tag:
- 90 validation families → claim `p3_validation`
- 219 new train families → claim `p3_train`
- 198 replay train families → claim `p3_train_replay` (they already have `p2_train`)

Use `FamilyRegistry.claim(family_id, tag)`. Claiming is idempotent.

## Hard gates (binding — abort with exit 1 if any fail)

1. P3 Validation count >= 90 (if supplement insufficient, error)
2. P3 Train (new) count >= 180
3. P3 Train (replay) count == 198 (224 - 26 quarantine)
4. P3 Train total >= 404
5. `assert_pairwise_disjoint` passes (all 6 pairs, with whitelist)
6. All quarantine exclusion checks pass
7. No family appears in more than one P3 partition set (validation/train_new/train_replay)

## Existing infrastructure (use these — do NOT reimplement)

1. `src/family_registry.py::FamilyRegistry` — load/save/claim/unclaim/
   assert_pairwise_disjoint/families_with_usage. Already has all needed APIs.
2. `data/family-registry.json` — the registry file (load, modify, save)
3. Frozen v3 families list: `data/frozen-eval/v3/families.json` (read-only
   reference; the registry's `frozen_v3` tag is the source of truth)

## Sampling determinism (binding)

- Seed: 42 (consistent with Task 7 candidate reservation)
- For the 29 test-split families supplemented into validation: use
  `random.Random(42).sample(sorted_available_test_ids, 29)`.
  Sort the available test family_ids ascending BEFORE sampling to ensure
  deterministic output regardless of dict insertion order.
- The 219 remaining test families (for train_new) = all available test
  families minus the 29 sampled for validation. No additional sampling —
  take ALL remaining.

## Tests (binding — minimum 8 tests)

1. `test_partition_counts` — validation=90, train_new=219, train_replay=198, total=417
2. `test_pairwise_disjoint_passes` — assert_pairwise_disjoint with whitelist passes
3. `test_pairwise_disjoint_no_whitelist_fails` — without whitelist, the
   p3_train_replay ∩ p2_train overlap is detected (expected failure →
   confirms the check is working)
4. `test_quarantine_excluded_from_all_p3_splits` — no quarantine family in
   any P3 partition set
5. `test_frozen_v3_disjoint_from_p3` — no frozen_v3 family in train/validation
6. `test_p2_frozen_v2_disjoint_from_p3` — no p2_frozen_v2 family in P3 splits
7. `test_registry_claims_consistent` — every family_id in the partition has
   the correct tag claimed in the registry
8. `test_deterministic_sampling` — re-running with seed=42 produces the same
   29 supplemented validation family_ids
9. `test_json_output_schema` — family-partition.json has all required fields
10. `test_no_family_in_multiple_p3_sets` — no family_id appears in both
    p3_validation and p3_train (new or replay)

## Commit message

`feat(p3): partition P3 validation + train families (pairwise disjoint + P2 replay whitelist)`

## Deviations / clarifications

1. **P2 replay = 198, not 224**. The plan body said "P2 224 train families"
   but 26 of those 224 are quarantined (per Task 5-redo dedup audit). Per
   Amendment A9, quarantine families are excluded from ALL P3 splits. The
   brief documents this as the binding count. Total train = 219 + 198 = 417
   (still >= 404 ✓).

2. **Validation supplement from test split**. The official validation split
   only has 61 available families (< 90 target). The 29-family supplement
   comes from the test split (the same pool as frozen_v3, but frozen_v3
   families are already claimed and excluded from available). The
   supplement is deterministic (seed=42, sorted-then-sampled).

3. **No P2 validation replay**. The plan does NOT replay P2 validation
   families into P3 validation. P3 validation is entirely NEW (either from
   the official validation split or supplemented from test). This is
   consistent with Global Constraint #9 (P2 Train is the ONLY allowed
   historical intersection).

4. **`p3_train_replay` tag is ADDITIVE to `p2_train`**. The 198 replay
   families keep their `p2_train` tag AND get `p3_train_replay` added.
   This is so the whitelist check can verify the intentional overlap.
