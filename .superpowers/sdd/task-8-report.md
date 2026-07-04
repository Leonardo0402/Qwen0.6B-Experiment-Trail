# Task 8 Report: Build Frozen v3 Samples with Verification and Freeze

## Status
DONE — All deliverables complete. 83 families qualified (>=80 threshold), freeze decision = `freeze_actual`, all 4 output files written, sha_lock verified consistent, registry updated, all 12 tests pass.

## Files created / modified

| Path | Action |
|---|---|
| `scripts/build_frozen_v3_samples.py` | Created (orchestrator, ~900 lines) |
| `tests/test_build_frozen_v3_samples.py` | Created (12-test suite) |
| `data/frozen-eval/v3/families.json` | Created (83 frozen family IDs) |
| `data/frozen-eval/v3/test_raw.jsonl` | Created (249 samples) |
| `data/frozen-eval/v3/manifest.json` | Created (with `immutability.sha_lock`) |
| `data/frozen-eval/v3/rejected.jsonl` | Created (239 records) |
| `data/family-registry.json` | Modified (83 families claimed `frozen_v3`, 37 unclaimed `frozen_v3_candidate`) |
| `src/family_registry.py` | Modified (added `unclaim` method on `FamilyEntry` and `FamilyRegistry`) |
| `src/training_data.py` | Modified (added `assert_not_frozen_v3` helper + `FROZEN_V3_DIR` constant) |

## Pipeline run

```
python scripts/build_frozen_v3_samples.py \
    --candidates data/frozen-eval/v3/candidates.json \
    --mbpp-verified-dir data/external/mbpp/verified \
    --output-dir data/frozen-eval/v3 \
    --registry data/family-registry.json \
    --seed 42 \
    --timeout 10.0
```

Exit code 0.

## Totals

| Metric | Value |
|---|---|
| Total candidates processed | 120 |
| Qualified families | 83 |
| Rejected families | 37 |
| Total frozen samples (test_raw.jsonl) | 249 |
| Rejected records (rejected.jsonl) | 239 |

## Freeze decision

`freeze_actual` — Froze all 83 qualified families (between `MIN_FROZEN=80` and `MAX_FROZEN=100`, so all qualified families were frozen with no surplus).

## variant_breakdown

| variant_type | count |
|---|---|
| code | 83 |
| boundary | 83 |
| static_repair | 0 |
| execution_repair | 0 |
| canary | 83 |
| **total** | **249** |

Each qualified family contributes exactly 3 samples (1 code + 1 boundary + 1 canary). The static_repair / execution_repair counts are 0 because `generate_all_variants` did not produce any failing-bug variants for these MBPP problems (`gen_rejected` reason `no_failing_bug_variants` — see "Concerns" below).

## Rejected records breakdown (rejected.jsonl — 239 records)

| count | rejection_reason | notes |
|---|---|---|
| 120 | `no_failing_bug_variants` | One per family (all 120 candidates). `gen_rejected` records from `generate_all_variants`; do NOT disqualify the family. |
| 45 | `hidden_padding_insufficient` | Could not pad hidden tests to count>=3 with boundary values. |
| 45 | `hidden_assertions_count_1_lt_3` | Final hidden_tests assert count < 3. |
| 25 | `boundary_variant_generation_failed` | Boundary variant generation failed (subset of the 37 rejected families). |
| 4 | `reference_verification_failed` | Reference sample (code or boundary) failed REAL pytest. |
| **239** | **TOTAL** | |

Note: A single rejected family typically contributes multiple records (e.g. one `no_failing_bug_variants` + one `hidden_padding_insufficient` + one `hidden_assertions_count_1_lt_3` per failing sample). The 120 `no_failing_bug_variants` records appear for ALL 120 families (qualified and rejected), because MBPP-style problems rarely produce failing-bug variants via the existing generator — these records are informational only.

## Hard gates (all pass on the 83 frozen families)

| # | Gate | Result |
|---|---|---|
| 1 | `public_tests` count >= 2 for code + boundary samples | ✅ |
| 2 | `hidden_tests` count >= 3 for code + boundary samples (after padding) | ✅ |
| 3 | Reference samples pass REAL pytest (`verify_sample`) | ✅ |
| 4 | `verify_broken_is_broken` (would apply to repair samples; none here) | N/A (no repair samples) |
| 5 | Real `execution_feedback` (would apply to execution_repair samples; none here) | N/A |
| 6 | Canary sample FAILS pytest (negative control) | ✅ (all 83 canaries verified=False) |

## sha_lock verification

```
$ python -c "from pathlib import Path; from scripts.build_frozen_v3_samples import verify_sha_lock; verify_sha_lock(Path('data/frozen-eval/v3')); print('sha_lock verification: OK')"
sha_lock verification: OK
```

The `immutability.sha_lock` stored in `manifest.json` is computed over the concatenation of `families.json` + `test_raw.jsonl` + `rejected.jsonl` (the 3 non-manifest files). Recomputed value matches the stored value. ✅

`sha_lock` value: `809811cd1b5e6e1d7f90f005a111f6798947bb7d03aeea510b977c77939ea8f4`

## Registry update

| Tag | Count | Notes |
|---|---|---|
| `frozen_v3_candidate` | 83 | Only the 83 frozen families retain this tag. |
| `frozen_v3` | 83 | Newly claimed on the 83 frozen families. |
| Reverted `frozen_v3_candidate` | 37 | The 37 rejected families had `frozen_v3_candidate` unclaimed (reverted to empty usage). |

Example:
- `mbpp_fam_101` (frozen): `usage = ["frozen_v3_candidate", "frozen_v3"]`
- `mbpp_fam_114` (rejected): `usage = []`

## Test summary

```
$ python -m pytest tests/test_build_frozen_v3_samples.py -v
============================= test session starts =============================
collected 12 items
tests\test_build_frozen_v3_samples.py ............                       [100%]
============================= 12 passed in 9.45s ==============================
```

12/12 named tests pass:

1. `test_pad_hidden_tests_success` ✅
2. `test_pad_hidden_tests_already_sufficient` ✅
3. `test_pad_hidden_tests_no_function` ✅
4. `test_post_process_variant_type_code` ✅
5. `test_post_process_variant_type_repair` ✅
6. `test_make_canary` ✅
7. `test_verify_family_qualified` ✅
8. `test_verify_family_rejected` ✅
9. `test_apply_freeze_decision_freeze_100` ✅
10. `test_apply_freeze_decision_fix_first` ✅ (uses monkeypatch to avoid slow pytest runs)
11. `test_sha_lock_consistency` ✅
12. `test_unclaim_idempotent` ✅

## Commit

Staged (per brief):
- `scripts/build_frozen_v3_samples.py`
- `tests/test_build_frozen_v3_samples.py`
- `data/frozen-eval/v3/families.json`
- `data/frozen-eval/v3/test_raw.jsonl`
- `data/frozen-eval/v3/manifest.json`
- `data/frozen-eval/v3/rejected.jsonl`
- `data/family-registry.json`
- `src/family_registry.py`
- `src/training_data.py`

Commit message: `feat(p3): build frozen v3 samples with verification and freeze`

Commit hash: see `git log -1` after commit.

## Deviations / clarifications from the brief

1. **Mixed-format public_tests normalization fix**. `generate_boundary_variant` (from Task 4) appends pytest test functions (with `from solution import` and `def test_*`) to the original bare-assert public tests, creating a mixed-format file. `src/sandbox.py::_normalize_test_code` returns such mixed-format code as-is when it detects `from solution` or `def test_`, which means the bare asserts at the top of the file run BEFORE the `from solution import` line and raise `NameError` during pytest collection. **All 38 first families were being REJECTED for this reason in the initial run.** Fix: added `_normalize_public_tests_for_pytest` to `scripts/build_frozen_v3_samples.py` that prepends `from solution import *` when bare asserts appear before the first `from solution` line. Integrated into `process_family` after the variant-type post-processing step (step 2b). This is a LOCAL normalization — `src/sandbox.py` is NOT modified (preserving the documented contract of `_normalize_test_code`). The fix is invisible to pure bare-assert samples (no-op) and pure pytest-format samples (no-op).

2. **`gen_rejected` records flow into `rejected.jsonl` even for qualified families**. Per the brief, `gen_rejected` records (e.g. `no_failing_bug_variants`) are added to `rejected_records` in `process_family` regardless of whether the family qualifies. These records are informational (samples that couldn't be generated, not samples that failed verification). They appear in `rejected.jsonl` for ALL 120 families. This is the documented behavior; no change.

3. **`sha_lock` computed over 3 non-manifest files**. The brief's chicken-and-egg resolution: `sha_lock` is computed over `families.json` + `test_raw.jsonl` + `rejected.jsonl` (the 3 non-manifest files), then stored in `manifest.json`'s `immutability.sha_lock` field. This is the resolution recommended in the brief's ambiguity section.

4. **Canary padding is skipped**. The brief says "skip padding for canary". `make_canary` is called AFTER padding; the canary's `hidden_tests` is whatever the source sample had (typically 1 assertion). This is intentional — the canary is a negative control and is expected to FAIL pytest regardless of hidden test count.

5. **`update_registry` keeps `frozen_v3_candidate` on frozen families**. Per the brief: "Frozen families: claim `frozen_v3` (keep `frozen_v3_candidate`)". So frozen families end up with BOTH tags. This is so the candidate tag remains as a historical record. Surplus and rejected families have `frozen_v3_candidate` unclaimed.

6. **Python version**. The brief and `pyproject.toml` specify Python 3.10. The active interpreter is 3.8.10. The script and tests use `from __future__ import annotations` for forward-compatibility. All 12 tests pass under 3.8.10.

7. **`unclaim` is a no-op if `family_id` not in registry**. `FamilyRegistry.unclaim` silently returns if the family_id is not present (idempotent). This is the documented behavior per the brief.

## Concerns

1. **Zero static_repair / execution_repair samples in the frozen set**. All 120 families produced `gen_rejected` records with reason `no_failing_bug_variants` — the existing `generate_all_variants` (from Task 4 / `scripts/build_p2_curriculum.py`) could not synthesize a failing broken-code variant for any of these MBPP problems. This means the frozen v3 eval set only covers `code_generation` task_type; it does NOT cover `static_repair` or `execution_repair` task_types. This is a known limitation of the generator, not a bug in Task 8. A future task may need to enhance the bug-injection generator to cover more MBPP problems.

2. **45 families rejected for `hidden_padding_insufficient`**. The boundary-value padding helper (`pad_hidden_tests`) failed to extend `hidden_tests` to count >= 3 for 45 of the 120 families. The primary cause is that `_extract_test_calls` could not parse the function call structure in the public tests (e.g. methods, calls with non-literal arguments). Combined with `hidden_assertions_count_1_lt_3`, these 45 records correspond to families where the original `hidden_tests` had only 1 assertion and padding failed. This is the dominant rejection reason (after `no_failing_bug_variants` which is informational).

3. **25 families rejected for `boundary_variant_generation_failed`**. `generate_boundary_variant` failed for 25 families. These overlap with the 37 rejected families; the boundary variant was either not generated or had a structural issue. These families also have `hidden_padding_insufficient` records.

4. **Real pytest execution time**. The full pipeline took ~10 minutes to process 120 families on Windows with 10s per-pytest timeout. This is acceptable for a one-shot freeze operation but would benefit from parallelization if the pipeline needs to be re-run frequently.

5. **`ruff: ruff not installed`** in some `verification_messages`. This is a pre-existing warning from `verify_sample` (ruff is not installed in the environment). It does NOT affect the verification outcome (`ruff_ok` is `False` but `is_accepted` only requires `syntax_ok AND pytest_ok AND NOT timeout`). No action needed.
