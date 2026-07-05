# Task 8 Brief: Build Frozen v3 Samples + Verify + Freeze

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Tasks 1-7 complete)
- Plan file: .superpowers/sdd/p3-plan.md (Global Constraints #6, #12 bind this task)
- Task 7 reserved 120 candidate families in `data/frozen-eval/v3/candidates.json` (all from MBPP test split, all claimed as `frozen_v3_candidate` in `data/family-registry.json`).
- This task generates samples on those 120 candidates, verifies them with REAL pytest, and freezes 80-100 qualified families as the immutable Frozen v3 evaluation set.

## Goal
Build two new files:
1. `scripts/build_frozen_v3_samples.py` — orchestrator that generates, verifies, and freezes Frozen v3 samples
2. `tests/test_build_frozen_v3_samples.py` — test suite

And produce four new data files (committed):
3. `data/frozen-eval/v3/families.json` — list of frozen family_ids (80-100)
4. `data/frozen-eval/v3/test_raw.jsonl` — frozen samples (one Sample per line)
5. `data/frozen-eval/v3/manifest.json` — manifest with SHA256, sample counts, freeze decision
6. `data/frozen-eval/v3/rejected.jsonl` — rejected samples with reasons (one JSON object per line)

Also modify:
7. `data/family-registry.json` — upgrade the 80-100 frozen families from `"frozen_v3_candidate"` to `"frozen_v3"` tag (claim, do not remove the candidate tag — both stay)

## Existing Infrastructure (use these — do NOT reimplement)
1. **`scripts/build_p2_curriculum.py::generate_all_variants(samples, *, timeout_s=10.0, seed=42)`** — orchestrator that takes a list of code_generation `Sample` objects and returns a 5-tuple `(code_gen, boundary, static_repair, execution_repair, rejected)`. Internally calls `inject_bugs`, `generate_boundary_variants`, `build_execution_repair`. Import it: `from scripts.build_p2_curriculum import generate_all_variants`.
2. **`scripts/generate_boundary_variants.py::generate_boundary_variant(sample, *, max_boundary_tests=8) -> Optional[Sample]`** — generates boundary tests and adds them to PUBLIC tests (NOT hidden). Import if you need to call it directly for hidden-padding.
3. **`src/validators.py::verify_sample(sample, *, run_ruff=True, run_mypy=False, pytest_timeout_s=10.0) -> SampleVerification`** — REAL verification with compile + pytest + ruff. Has `.is_accepted` property. Import it: `from src.validators import verify_sample, verify_broken_is_broken, compile_check`.
4. **`src/sandbox.py::run_pytest(code, tests, timeout_s) -> PytestResult`** — runs pytest in a subprocess sandbox.
5. **`src/schemas.py::Sample`** — pydantic BaseModel with fields including `variant_type`, `bug_type`, `source_split`, `execution_feedback`, `broken_code`. Use `Sample.from_json_line(line)` and `sample.to_json_line()` for serialization. To create a copy with modified fields: `sample.model_copy(update={...})`.

## Critical Issue: hidden_tests Must Be >= 3 (binding per Global Constraint #12)
The existing pipeline does NOT satisfy hidden>=3:
- MBPP source has 3 tests per task; `import_mbpp.py` splits as 2 public + 1 hidden.
- `generate_boundary_variants.py` adds boundary tests to PUBLIC only (does not touch hidden).
- Result: most samples have `hidden_count=1`, failing the hidden>=3 hard gate.

**Solution (binding)**: Implement a `pad_hidden_tests(sample, *, target_count=3) -> Sample` helper INSIDE `scripts/build_frozen_v3_samples.py` that:
1. Counts `assert ` occurrences in `sample.hidden_tests`.
2. If count >= target_count, return the sample unchanged.
3. If count < target_count:
   a. Parse `sample.target_code` with `ast.parse()` to extract function names and arg names.
   b. Generate boundary test cases for each function: empty input (if iterable), single-element input, boundary values (0, -1, max_int, None where applicable). Use the same heuristic as `scripts/generate_boundary_variants.py::_generate_boundary_tests` (you may import and call that internal function if accessible, or replicate the logic).
   c. Append the new tests to `sample.hidden_tests` (joined with `"\n\n"` separator) until `assert ` count >= target_count.
   d. Return the modified sample via `sample.model_copy(update={"hidden_tests": new_hidden})`.
4. Handle edge cases: if `ast.parse` fails (SyntaxError), skip padding and mark the sample for rejection with reason "hidden_padding_failed_syntax_error". If no functions found, skip padding and reject with reason "hidden_padding_failed_no_functions".
5. The padding helper must NOT modify `target_code`, `public_tests`, or `instruction`. Only `hidden_tests` is extended.

The padding must produce tests that PASS against `target_code` (otherwise the sample fails verification later and gets rejected — that's fine, it just means the padding logic needs to be smart). If a generated boundary test FAILS against `target_code`, discard that test and try the next one. If after exhausting candidates the hidden_count is still < 3, mark the sample for rejection with reason "hidden_padding_insufficient".

## Variant Type and Bug Type Post-Processing (binding per Global Constraint #19)
The existing scripts do NOT set `variant_type` or `bug_type` on the Sample objects. Task 8 must post-process every generated Sample to set these fields:

| Source | task_type | variant_type | bug_type |
|---|---|---|---|
| `code_gen` list | `code_generation` | `"code"` | `None` |
| `boundary` list | `code_generation` (from generate_boundary_variants) | `"boundary"` | `None` |
| `static_repair` list | `static_repair` | `"static_repair"` | extracted from `sample_id` suffix (e.g. `mbpp_42_sr_condition_error` → `"condition_error"`) |
| `execution_repair` list | `execution_repair` | `"execution_repair"` | extracted from `sample_id` suffix (e.g. `mbpp_42_er_off_by_one` → `"off_by_one"`) |

Bug type extraction: parse the `sample_id` field with regex `.*_(sr|er)_(.+)$` → group 2 is the bug_type. If the regex doesn't match, set `bug_type=None` and log a warning.

Set the fields via `sample.model_copy(update={"variant_type": ..., "bug_type": ...})`.

## Canary Sample (binding — one per family)
For each frozen family, generate ONE canary sample that MUST FAIL all tests. The canary is a negative-control sanity check:
- Copy the source code_generation sample.
- Replace `target_code` with a deliberately broken version: `def canary_always_fails():\n    raise AssertionError("canary")\n` (this is a stub that will fail any test calling the real function).
- Set `variant_type="canary"`, `bug_type=None`.
- Set `task_type="code_generation"` (canary is a code-gen variant, not a repair).
- Set `sample_id=f"{original_sample_id}_canary"`.
- Verify: `verify_sample(canary).is_accepted` must be `False` (the canary must fail). If it passes, this is a CRITICAL bug — the canary is not actually broken.

The canary is included in `test_raw.jsonl` so the eval harness can use it as a negative control.

## Verification Flow (binding — for each candidate family)
For each of the 120 candidates:

1. Load the source Sample from `data/external/mbpp/verified/test.jsonl` (filter by `family_id`).
2. Call `generate_all_variants([source_sample], timeout_s=10.0, seed=42)` to get `(code_gen, boundary, static_repair, execution_repair, rejected)`. Expect ~5 samples per family (1 code + 1 boundary + 1-2 static + 1-2 exec).
3. Post-process every generated Sample to set `variant_type` and `bug_type` per the table above.
4. Apply `pad_hidden_tests(sample, target_count=3)` to every sample (including the canary — but the canary's hidden tests don't matter since it must fail anyway; skip padding for canary).
5. Generate the canary sample (one per family).
6. Run verification:
   - **Reference (code_gen, boundary)**: `verify_sample(sample).is_accepted == True` (target_code passes public + hidden). Also assert `len(sample.public_tests.split("assert ")) >= 3` (i.e. public_count >= 2 — first split element is empty so count is asserts+1; assert at least 3 elements means >= 2 asserts) and `len(sample.hidden_tests.split("assert ")) >= 4` (hidden_count >= 3).
   - **Repair (static_repair, execution_repair)**: `verify_sample(sample).is_accepted == True` (target_code passes all tests). AND `verify_broken_is_broken(sample, pytest_timeout_s=10.0) == True` (broken_code fails at least one test). AND for execution_repair specifically: `sample.execution_feedback is not None and len(sample.execution_feedback) > 0` (real feedback present).
   - **Canary**: `verify_sample(canary).is_accepted == False` (must fail). The verification field should record `pytest_ok=False` (or `syntax_ok=False` — either is acceptable as long as `is_accepted` is False).
7. If ALL samples for a family pass verification → family is "qualified". Add all its samples (including canary) to `test_raw.jsonl`.
8. If ANY sample for a family fails verification → family is "rejected". Add the failing samples (with reason) to `rejected.jsonl`. The family does NOT contribute to `test_raw.jsonl`.

## Hard Gates (binding — abort with exit 1 if any fail)
Per Global Constraint #12, every sample in `test_raw.jsonl` MUST satisfy:
- `public_assertions_count >= 2` (count of `assert ` substrings in `public_tests`)
- `hidden_assertions_count >= 3` (count of `assert ` substrings in `hidden_tests`)

For repair samples specifically:
- `verify_broken_is_broken(sample) == True`
- `execution_feedback` is non-null and non-empty (for execution_repair only)
- The execution_feedback must come from REAL pytest failure output (compress_feedback in build_execution_repair.py guarantees this)

## Freeze Decision (binding)
After verification, count qualified families:

| Qualified Count | Action |
|---|---|
| >= 100 | Freeze the first 100 by family_id ascending (sorted). Mark them as `"frozen_v3"` in the registry. The remaining qualified families revert to available (their `frozen_v3_candidate` tag is REMOVED from registry — they go back to the pool for Task 9 partition). |
| 80-99 | Freeze all qualified families. Document the actual count in `manifest.json::freeze_decision_note`. Mark them as `"frozen_v3"`. |
| < 80 | **FIX FIRST** — abort. Do NOT write `families.json` / `test_raw.jsonl` / `manifest.json`. Write only `rejected.jsonl` with all rejection reasons. Exit code 1. The user must be informed to fix the pipeline before re-running. |

**Note on first-100 selection**: when qualified >= 100, sort all qualified family_ids ascending (lexicographic — `mbpp_fam_11`, `mbpp_fam_110`, ..., `mbpp_fam_99`). Take the first 100. The remaining qualified families have their `frozen_v3_candidate` tag removed (use a new helper `unclaim(family_id, tag)` — but the existing FamilyRegistry API does not have `unclaim`, so the implementer must add it OR directly manipulate the `usage` list). The simplest approach: load the registry, for each non-frozen qualified family, remove `"frozen_v3_candidate"` from `usage`. For each frozen family, add `"frozen_v3"` to `usage` (keep `"frozen_v3_candidate"` too — both tags stay).

**Add `unclaim` to FamilyRegistry** (small API extension): `def unclaim(self, family_id: str, tag: str) -> None: """Remove tag from family's usage list. No-op if tag not present."""`. Add a test for it in `tests/test_family_registry.py`? **No — do NOT modify `tests/test_family_registry.py`** (Task 6's tests). Instead, write the `unclaim` method in `src/family_registry.py` and add the test in `tests/test_build_frozen_v3_samples.py` (Task 8's test file). The Task 8 brief owns this extension.

## Output File Schemas

### data/frozen-eval/v3/families.json
```json
{
  "generated_at": "<iso8601 utc>",
  "schema_version": 1,
  "frozen_family_count": <int>,
  "freeze_decision": "freeze_100" | "freeze_actual",
  "freeze_decision_note": "<string>",
  "families": [
    "mbpp_fam_<n>",
    ...
  ]
}
```
- `families` is sorted ascending.
- `frozen_family_count` is between 80 and 100.

### data/frozen-eval/v3/test_raw.jsonl
One Sample JSON per line. Includes:
- All variant samples (code, boundary, static_repair, execution_repair) for each frozen family.
- One canary sample per frozen family.
- All samples have `variant_type` and `bug_type` set per the table above.
- All samples have `verified=True` with a real `Verification` object (except canary which has `verified=False` and a real failure verification).
- All samples have `hidden_tests` with >= 3 `assert ` substrings (except canary).
- All samples have `source_split="test"`.

### data/frozen-eval/v3/manifest.json
```json
{
  "generated_at": "<iso8601 utc>",
  "schema_version": 1,
  "frozen_version": "v3",
  "frozen_family_count": <int>,
  "total_sample_count": <int>,
  "variant_breakdown": {
    "code": <int>,
    "boundary": <int>,
    "static_repair": <int>,
    "execution_repair": <int>,
    "canary": <int>
  },
  "test_raw_sha256": "<sha256 of test_raw.jsonl bytes>",
  "families_sha256": "<sha256 of families.json bytes>",
  "rejected_count": <int>,
  "rejected_sha256": "<sha256 of rejected.jsonl bytes>",
  "freeze_decision": "freeze_100" | "freeze_actual",
  "freeze_decision_note": "<string>",
  "immutability": {
    "write_once": true,
    "any_change_requires": "v4",
    "sha_lock": "<sha256 of all 4 files concatenated, in order: families.json, test_raw.jsonl, manifest.json, rejected.jsonl>"
  }
}
```
- `test_raw_sha256` is the SHA256 of the raw bytes of `test_raw.jsonl` (not the parsed content).
- `sha_lock` is the SHA256 of the concatenation `families.json + test_raw.jsonl + manifest.json + rejected.jsonl` (in that order, raw bytes).
- After writing, the script MUST re-read all 4 files, recompute `sha_lock`, and assert it matches what was written. If mismatch, abort.

### data/frozen-eval/v3/rejected.jsonl
One JSON object per line:
```json
{"sample_id": "mbpp_XXX_...", "family_id": "mbpp_fam_XXX", "variant_type": "...", "rejection_reason": "...", "verification_messages": ["..."]}
```

## Training Code Hard-Block (binding per plan)
Add a path check to the training data collator that REFUSES to load samples from `data/frozen-eval/v3/`. Locate the file `src/training_data.py` (if it exists) or `src/dataset_builder.py` or wherever the training data loader is. Add a function:

```python
FROZEN_V3_DIR = "data/frozen-eval/v3"

def assert_not_frozen_v3(path: Path | str) -> None:
    """Refuse to load training data from the frozen v3 eval directory."""
    p = Path(path).resolve()
    frozen = (Path(_ROOT) / FROZEN_V3_DIR).resolve()
    if p == frozen or frozen in p.parents:
        raise ValueError(
            f"Training data loader refuses to read from frozen v3 eval directory: {p}. "
            f"This directory is write-once and reserved for evaluation only. "
            f"Use data/p3-curriculum/ for training data."
        )
```

Call this function at the top of any training-data-loading function in `src/`. If no such function exists, add it as a standalone utility and document where it should be called.

**Constraint**: this addition should be MINIMAL — do not refactor existing loading code. Just add the assertion helper and call it in 1-2 strategic places (the entry points where JSONL training data is loaded).

## tests/test_build_frozen_v3_samples.py

Use synthetic Samples built in-memory. Tests:

1. `test_pad_hidden_tests_passes_through_when_sufficient`: sample with hidden_count=3 → returned unchanged.
2. `test_pad_hidden_tests_extends_to_three`: sample with hidden_count=1 → returned sample has hidden_count >= 3 (and the new tests pass against target_code — verify with `verify_sample`).
3. `test_pad_hidden_tests_handles_syntax_error`: sample with malformed target_code → returns sample unchanged + rejection reason "hidden_padding_failed_syntax_error" recorded somewhere accessible (e.g. via a return tuple or a side-channel list).
4. `test_variant_type_post_processing`: a list of synthetic samples (one per task_type) → after post-processing, each has the correct `variant_type` and `bug_type` per the table.
5. `test_bug_type_extraction_from_sample_id`: sample with `sample_id="mbpp_42_sr_condition_error"` → `bug_type="condition_error"`. Sample with non-matching ID → `bug_type=None`.
6. `test_canary_sample_fails_verification`: synthetic canary → `verify_sample(canary).is_accepted == False`.
7. `test_qualified_family_passes_all_gates`: synthetic family with 5 valid samples → all pass verification, family is "qualified".
8. `test_rejected_family_recorded`: synthetic family with one failing sample → family is "rejected", failing sample appears in `rejected.jsonl` with reason.
9. `test_freeze_decision_100_or_more`: 110 qualified families → freeze first 100 by family_id, remaining 10 revert to available (frozen_v3_candidate tag removed).
10. `test_freeze_decision_less_than_80_aborts`: 70 qualified families → script exits with code 1, no `families.json` written, only `rejected.jsonl` written.
11. `test_manifest_sha_lock_consistent`: after running on synthetic data, re-read all 4 files, recompute `sha_lock`, assert it matches `manifest.json::immutability.sha_lock`.
12. `test_unclaim_removes_tag`: `registry.unclaim(family_id, "frozen_v3_candidate")` removes the tag from the family's usage list (no-op if not present).

(12 tests total — the brief lists 12, not 10.)

## Constraints
- Do NOT modify `src/validators.py` or `src/sandbox.py` (read-only).
- Do NOT modify existing `scripts/` files (read-only — you only IMPORT from them).
- Do NOT modify `tests/test_family_registry.py` (Task 6's tests) — add the `unclaim` test to `tests/test_build_frozen_v3_samples.py`.
- The script MAY add the `unclaim` method to `src/family_registry.py` (small API extension — Task 6 is complete but the API is allowed to grow).
- The script MAY add the `assert_not_frozen_v3` helper to `src/training_data.py` or another `src/` module. Identify the right location by inspecting where training data is loaded.
- Single git commit at the end.
- All 12 tests must pass: `python -m pytest tests/test_build_frozen_v3_samples.py -v` from project root, using Python 3.10.
- The real run on 120 candidates will be SLOW (each sample requires pytest execution). Expect ~5-15 minutes. Use a reasonable timeout (e.g. 30 minutes total).

## Test Verification
`cd e:\agent\Qwen\qwen3-code-lab ; python -m pytest tests/test_build_frozen_v3_samples.py -v`

## Run the Script for Real
After tests pass, run for real on the 120 candidates:
```
python scripts/build_frozen_v3_samples.py \
    --candidates data/frozen-eval/v3/candidates.json \
    --mbpp-verified-dir data/external/mbpp/verified \
    --output-dir data/frozen-eval/v3 \
    --registry data/family-registry.json \
    --seed 42 \
    --timeout 10.0
```

The script must:
- Succeed (exit 0) if qualified families >= 80. Write all 4 output files. Update the registry (upgrade frozen families from `frozen_v3_candidate` to also include `frozen_v3`; revert non-frozen qualified families by removing `frozen_v3_candidate`).
- Fail (exit 1) if qualified families < 80. Write only `rejected.jsonl`. Do NOT write `families.json` / `test_raw.jsonl` / `manifest.json`. Print "FIX_FIRST" to stderr.

## Report File
Write to: `.superpowers/sdd/task-8-report.md`
Include:
- Files created/modified (paths)
- Total candidates processed (120)
- Qualified family count
- Rejected family count (with breakdown by rejection reason)
- Freeze decision (freeze_100 or freeze_actual)
- Final frozen_family_count (80-100)
- variant_breakdown (code/boundary/static_repair/execution_repair/canary counts)
- Confirmation that sha_lock is consistent
- Confirmation that all hard gates pass
- The commit hash
- Test summary
- Any concerns

## Commit
- Stage: `scripts/build_frozen_v3_samples.py`, `tests/test_build_frozen_v3_samples.py`, `data/frozen-eval/v3/families.json`, `data/frozen-eval/v3/test_raw.jsonl`, `data/frozen-eval/v3/manifest.json`, `data/frozen-eval/v3/rejected.jsonl`, `data/family-registry.json`, `src/family_registry.py` (for unclaim), and the `src/` file modified for `assert_not_frozen_v3`.
- Commit message: `feat(p3): build frozen v3 samples with verification and freeze`
- Single commit.

## Working Directory
e:\agent\Qwen\qwen3-code-lab
