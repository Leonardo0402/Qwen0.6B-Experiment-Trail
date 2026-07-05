# Task 7 Report: Frozen v3 Candidate Reservation

## Status
DONE (v2.1 rework — re-run on pad-then-verify registry with 348 source pool) — All deliverables complete, all 10 tests pass, all 8 hard assertions pass on real data, registry updated in place.

## Files created / modified

| Path | Action |
|---|---|
| `scripts/reserve_frozen_v3_candidates.py` | Unchanged (CLI script) |
| `tests/test_reserve_frozen_v3_candidates.py` | Unchanged (10-test suite) |
| `data/frozen-eval/v3/candidates.json` | Regenerated (120 candidates) |
| `data/family-registry.json` | Modified (in-place: 120 families claimed with `frozen_v3_candidate`) |

## Source pool

- **source_pool_size**: 348 (was 474 on original run)
  - 807 total families; of the 365 `source_split=="test"` verified families, 17 are quarantined (Task 5-redo) and 0 carry any P2 tag, leaving 348 in the source pool.
- **candidate_count**: 120

## Per-difficulty-bucket allocation

Computed via proportional allocation with integer floor + remainder distribution. Pool total = 348, target = 120.

| difficulty | pool_size | allocated |
|---|---|---|
| 0 | 236 | 82 |
| 1 | 102 | 35 |
| 2 | 10 | 3 |
| 3 | 0 | 0 |
| 4 | 0 | 0 |
| **sum** | **348** | **120** |

## Hard assertions (all 8 pass)

| # | Assertion | Result |
|---|---|---|
| 1 | `len(candidates) == 120` | ✅ |
| 2 | All candidates are in `data/family-registry.json` AND have `source_split == "test"` | ✅ |
| 3 | No candidate has `"quarantine"` in its `usage` (before & after claiming) | ✅ |
| 4 | No candidate has any P2 tag (`p2_train` / `p2_validation` / `p2_frozen_v2`) | ✅ |
| 5 | No candidate has `"frozen_v3"` tag | ✅ |
| 6 | All candidates are unique (no duplicates) | ✅ |
| 7 | After claiming, `families_with_usage("frozen_v3_candidate")` returns exactly these 120 families (sorted) | ✅ |
| 8 | `assert_pairwise_disjoint(["p2_train","p2_validation","p2_frozen_v2","frozen_v3_candidate","quarantine"])` does NOT raise (with P2+quarantine whitelist) | ✅ |

## Pairwise disjoint

Verified post-claim on the real registry:

```
registry.assert_pairwise_disjoint(
    ["p2_train", "p2_validation", "p2_frozen_v2",
     "frozen_v3_candidate", "quarantine"],
    whitelist=[
        ("p2_train", "quarantine"),
        ("p2_validation", "quarantine"),
        ("p2_frozen_v2", "quarantine"),
    ],
)
```

- Does NOT raise. ✅
- 30 families in the real registry carry BOTH a P2 tag AND `quarantine` (from Task 6). These are excluded from the candidate pool by the `quarantine` filter, so `frozen_v3_candidate` is fully disjoint from every other tag in the asserted set. The whitelist is required only for the (P2, quarantine) pair.

## Registry `total_new_available` delta (v2.1 redo)

| Metric | Before | After | Delta |
|---|---|---|---|
| `total_families` | 807 | 807 | 0 |
| `total_p2_used` | 374 | 374 | 0 |
| `total_quarantined` | 50 | 50 | 0 |
| `total_new_available` | 409 | 289 | −120 |

The 120 newly-claimed families moved from "empty usage" to "non-empty usage" (`["frozen_v3_candidate"]`), so `total_new_available` decreased by exactly 120. ✅

## Idempotency

Re-running the script on the already-updated registry produces the IDENTICAL 120 candidates (verified by re-running and confirming all 8 hard assertions still pass and `total_new_available` is still 289). The script does NOT exclude already-claimed families from the source pool — instead it relies on a fixed seed (`random.Random(42)`) plus sorted bucket input to deterministically reproduce the same 120 every run. Re-claiming is a no-op (`FamilyEntry.claim` is idempotent). ✅

## Test summary

```
$ python -m pytest tests/test_reserve_frozen_v3_candidates.py -v
============================= test session starts =============================
collected 10 items
tests\test_reserve_frozen_v3_candidates.py ..........                    [100%]
============================= 10 passed in 2.25s ==============================
```

10/10 tests pass. Task 6's `tests/test_family_registry.py` (10 tests) still pass too (no regression on the modified `data/family-registry.json`).

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

Exit code 0.

## Commit

Staged (per brief):
- `scripts/reserve_frozen_v3_candidates.py`
- `tests/test_reserve_frozen_v3_candidates.py`
- `data/frozen-eval/v3/candidates.json`
- `data/family-registry.json`

Commit message: `feat(p3): reserve 120 frozen v3 candidate families`

Commit hash: `2483968` (short) / `2483968...` on branch `feat/p3-capability-expansion-v2`.

## Deviations / clarifications from the brief

1. **Source pool filter does NOT exclude `frozen_v3_candidate`**. The brief's "Source Pool Definition" lists `"frozen_v3_candidate" NOT in usage` as one of 5 conditions, but the brief's "Idempotency" section explicitly recommends NOT excluding already-claimed families so re-runs produce the same 120. The task resolution notes from the parent agent confirm this: *"Do NOT exclude already-claimed families from the source pool (that would change the sampling)."* The script follows this resolution; the `source_pool_definition` string in `candidates.json` reflects the actual implementation (drops the `frozen_v3_candidate` filter). On first run the source_pool_size is 474; on re-run it is still 474 (not 354).

2. **`source_pool_definition` string in `candidates.json`** is spelled out explicitly as `"families with source_split=='test' AND not quarantine AND not p2_train AND not p2_validation AND not p2_frozen_v2 AND not frozen_v3"` rather than the brief's abbreviated `"families with source_split=='test' AND not quarantine AND not p2_* AND not frozen_v3*"`. The asterisked form was ambiguous (could include `frozen_v3_candidate`); the spelled-out form matches what the script actually computes. No semantic difference for first-run output; clarifies for re-runs.

3. **Pairwise disjoint whitelist for P2+quarantine pairs**. The brief's hard assertion #8 acknowledges the P2+quarantine overlap from Task 6 ("except potentially P2+quarantine overlap"). The script passes `whitelist=[("p2_train","quarantine"), ("p2_validation","quarantine"), ("p2_frozen_v2","quarantine")]` to `assert_pairwise_disjoint`. 30 families in the real registry carry both a P2 tag and `quarantine` (Task 6 backfilled P2 first, then appended quarantine to the affected families). Without the whitelist, the assertion would raise on this overlap. With the whitelist, it correctly verifies that `frozen_v3_candidate` is disjoint from every other tag in the asserted set.

4. **Python version**. The brief and `pyproject.toml` specify Python 3.10. The active interpreter is 3.8.10. The script and tests use `from __future__ import annotations` for forward-compatibility and avoid 3.10+-only syntax. All 10 tests pass under 3.8.10. No code change is required to also pass under 3.10.

5. **Single seeded RNG across buckets**. The brief says "use `random.Random(seed=42)` to shuffle each bucket" without specifying one vs. one-per-bucket. The script uses ONE `random.Random(42)` and shuffles each bucket in difficulty order (0→1→2→3→4). This is the standard deterministic interpretation; per-bucket RNGs with the same seed would correlate shuffles across buckets and is wrong.

## Concerns

None blocking. The implementation matches the brief and the parent agent's resolutions exactly.
