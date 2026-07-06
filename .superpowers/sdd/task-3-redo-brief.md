# Task 3-redo Brief: Extract pad_hidden_tests + Pad-then-Verify Refactor

## Context

- Project: `e:\agent\Qwen\qwen3-code-lab`
- Branch: `feat/p3-capability-expansion-v2` (HEAD=`0e60b3a`)
- Trigger: P3 plan v2.1 Amendment A2 — import verifier must HARD reject `hidden>=3`
  (was SOFT warning in commit `b46fcb7`). Raw MBPP has only 1 hidden assert per
  sample, so padding MUST happen before the hard check.
- Plan file: `.superpowers/sdd/p3-plan.md` (see "Binding Amendments v2.1" section)

## Goal

Three coordinated changes:

### Change 1: Create `src/hidden_test_padding.py`

Extract `pad_hidden_tests` and `_normalize_public_tests_for_pytest` from
`scripts/build_frozen_v3_samples.py` (lines 79-209) into a new shared module
`src/hidden_test_padding.py`.

The module must:
- Import the 5 helper functions from `scripts.generate_boundary_variants`:
  `_parse_function_info`, `_extract_test_calls`, `_ast_to_value`,
  `_classify_arg`, `_boundary_values_for_type`, `_execute_boundary_calls`
  (these remain in generate_boundary_variants.py — do NOT move them)
- Import `Sample` from `src.schemas`, `Verification` from `src.schemas`
- Import `ast` for SyntaxError handling
- Provide TWO public functions:
  1. `pad_hidden_tests(sample: Sample, *, target_count: int = 3) -> tuple[Sample, Optional[str]]`
     — exact same logic as build_frozen_v3_samples.py L79-170
  2. `normalize_public_tests_for_pytest(public_tests: str) -> str`
     — exact same logic as `_normalize_public_tests_for_pytest` L177-209
     (rename: drop leading underscore since it's now public API)
- Include the `TARGET_HIDDEN_COUNT = 3` constant
- Use the `_ROOT` project-root sys.path guard pattern (same as other src/ modules
  that need scripts/ access — see build_frozen_v3_samples.py L34-36)

### Change 2: Modify `scripts/verify_imported_mbpp.py` (pad-then-verify)

Current state (commit `e1466bf`):
- `check_sample()` does: hard public>=2 → SOFT warning hidden>=3 → repair checks → verify_sample
- `_MIN_HIDDEN_ASSERTS = 3` is a warning threshold, NOT a hard reject

Required changes:
1. Add import: `from src.hidden_test_padding import pad_hidden_tests`
2. In `check_sample()`, BEFORE the hidden assertion count check:
   - Call `pad_hidden_tests(sample, target_count=3)`
   - If `rejection_reason` is not None (padding failed), return HARD reject
     with the padding failure reason
   - If padding succeeded, replace `sample` with the padded sample for all
     subsequent checks (verify_sample runs on padded sample)
3. Change hidden>=3 check from SOFT warning to HARD reject:
   - Remove the `warning` soft-check path for hidden < 3
   - After padding, `n_hidden < 3` is a HARD reject (padding should have brought
     it to >=3; if still <3, reject with "hidden assertions insufficient after padding")
4. Update docstring: hidden>=3 is now HARD (after padding), not SOFT
5. Update the manifest `warnings` field: remove `low_hidden_count` warning
   (no longer applicable — samples that would have warned now either pass
   padding or get rejected). Replace with `padding_rejected_count` in the
   manifest summary if useful, or just remove the warnings dict entirely
   if no soft checks remain.

The `verify_split()` function must also be updated:
- Track `padding_rejected_count` (samples rejected because padding failed)
- Pass padded samples through to verification

### Change 3: Modify `scripts/build_frozen_v3_samples.py` (import from shared module)

Current state (commit `0e60b3a`):
- Defines `pad_hidden_tests` locally (L79-170)
- Defines `_normalize_public_tests_for_pytest` locally (L177-209)
- Imports 5 helpers from `scripts.generate_boundary_variants` (L39-46)

Required changes:
1. Remove the local `pad_hidden_tests` definition (L79-170)
2. Remove the local `_normalize_public_tests_for_pytest` definition (L177-209)
3. Remove the now-unused imports of the 5 helper functions from
   `scripts.generate_boundary_variants` (L39-46) IF they're not used elsewhere
   in the file. Check first — if any of the 5 are used directly in
   build_frozen_v3_samples.py (not just via pad_hidden_tests), keep those imports.
4. Add import: `from src.hidden_test_padding import pad_hidden_tests, normalize_public_tests_for_pytest`
5. Replace all calls to `_normalize_public_tests_for_pytest(...)` with
   `normalize_public_tests_for_pytest(...)` (drop underscore prefix)
6. The `TARGET_HIDDEN_COUNT` constant can stay local (or import from shared
   module — either is fine)

## Files to Modify

1. **CREATE** `src/hidden_test_padding.py` (~120 lines)
2. **MODIFY** `scripts/verify_imported_mbpp.py` (pad-then-verify flow + hard check)
3. **MODIFY** `scripts/build_frozen_v3_samples.py` (remove local defs, import from shared)

## Files to Update (tests)

4. **MODIFY** `tests/test_import_mbpp_p3.py` — update tests for pad-then-verify:
   - Tests that expected SOFT warning for hidden<3 → now expect HARD reject
   - Add test: sample with 1 hidden assert → padded → passes hard check
   - Add test: sample where padding fails → rejected with padding reason
   - Add test: padded sample's hidden_tests has >=3 asserts
5. **MODIFY** `tests/test_build_frozen_v3_samples.py` — update import paths:
   - If tests import `pad_hidden_tests` from `build_frozen_v3_samples`, update
     to import from `src.hidden_test_padding`
   - If tests reference `_normalize_public_tests_for_pytest`, update to
     `normalize_public_tests_for_pytest`
6. **CREATE** `tests/test_hidden_test_padding.py` — unit tests for the shared module:
   - `test_pad_hidden_tests_already_sufficient`: sample with >=3 asserts → no change
   - `test_pad_hidden_tests_pads_to_3`: sample with 1 assert → padded to >=3
   - `test_pad_hidden_tests_syntax_error`: bad target_code → rejection reason
   - `test_pad_hidden_tests_no_functions`: code without def → rejection reason
   - `test_pad_hidden_tests_insufficient`: no test calls extractable → rejection reason
   - `test_normalize_public_tests_pure_bare_assert`: no `from solution` → unchanged
   - `test_normalize_public_tests_mixed_format`: bare assert before `from solution` → prefixed

## Verification

Run ALL tests:
```bash
cd e:\agent\Qwen\qwen3-code-lab
python -m pytest tests/ -x -q --tb=short
```

Expected: all tests pass. If `test_import_mbpp_p3.py` tests that expected SOFT
warning now fail (because hidden>=3 is HARD), update them to expect HARD reject
or expect padding to succeed.

## Constraints

- Do NOT modify `scripts/generate_boundary_variants.py` (the 5 helpers stay there)
- Do NOT modify `scripts/import_mbpp.py` (it's already compliant — no pytest)
- Do NOT modify `src/validators.py` (frozen per earlier task constraints)
- Do NOT modify `src/schemas.py` (already has variant_type/bug_type/source_split)
- Do NOT run the actual import/verify pipeline (that's Task 4-redo, separate)
- Match existing code style (from __future__ import annotations, type hints, docstrings)

## Commit

Single commit with message:
```
refactor(p3): extract pad_hidden_tests to shared module + pad-then-verify in import verifier

- Create src/hidden_test_padding.py with pad_hidden_tests + normalize_public_tests_for_pytest
- verify_imported_mbpp.py: pad hidden tests BEFORE hard check (hidden>=3 now HARD)
- build_frozen_v3_samples.py: import from shared module (remove local defs)
- Supersedes b46fcb7 (soft warning); implements v2.1 Amendment A2

Per P3 plan v2.1 Amendment A2.
```
