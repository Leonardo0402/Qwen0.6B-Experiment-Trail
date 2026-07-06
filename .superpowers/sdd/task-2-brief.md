# Task 2 Brief: Extend Sample Schema

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Task 1 complete at 48614af)
- Plan file: .superpowers/sdd/p3-plan.md (Global Constraints apply)
- Schema file: src/schemas.py (pydantic BaseModel, use_enum_values=True)
- Existing tests: tests/test_schemas.py (uses _base_sample() and _repair_sample() helpers)

## Goal
Add 3 optional, backward-compatible fields to the `Sample` model in `src/schemas.py`:
- `variant_type: Optional[str] = None` — e.g. "boundary", "empty_input", "duplicates", None
- `bug_type: Optional[str] = None` — e.g. "condition_error", "off_by_one", "return_value_error", "index_error", "initialization_error", "aggregation_error", "branch_deletion", "type_error", None
- `source_split: Optional[str] = None` — e.g. "train", "test", "validation", None

## Why
- P3 needs to track boundary samples as `variant_type="boundary"` (NOT as a new task_type — Issue #9 user directive #3)
- P3 needs `bug_type` for per-bug-type success reporting (Issue #9 §8.3)
- P3 needs `source_split` to trace MBPP origin split for contamination tracking (user directive: manifest must record source_split)

## Constraints (binding)
1. **Backward compatible**: existing JSON without these 3 fields MUST load fine (fields default to None)
2. **No new task_type**: do NOT add "boundary" to TaskType enum. Boundary is expressed via `variant_type="boundary"` with `task_type="code_generation"`
3. **Optional fields**: all 3 fields default to None; no validation that they must be non-empty
4. **No enum restriction**: variant_type / bug_type / source_split are free-form strings (not Enums) — P3 may introduce new variant types later without schema migration. Do NOT add field_validator that restricts to a fixed set.
5. **Serialization**: `to_json_line()` must include these fields when set (pydantic model_dump includes all fields by default — verify). `from_json_line()` must handle missing fields (default None via pydantic).
6. **to_chatml**: unaffected — the 3 new fields are metadata, not chat content. Do NOT modify to_chatml.
7. **Verification model**: do NOT modify Verification class.
8. **Surgical change**: only add the 3 fields to Sample class. Do NOT refactor existing validators, do NOT change other fields, do NOT add new validators beyond what's specified.

## Implementation Steps
1. Read `src/schemas.py` to understand current structure (pydantic v2, ConfigDict, field_validator, model_validator, to_json_line, from_json_line)
2. Add 3 fields after `dataset_version` field (before the validators section):
   ```python
   variant_type: Optional[str] = None
   bug_type: Optional[str] = None
   source_split: Optional[str] = None
   ```
3. Verify `to_json_line()` includes them (it uses `model_dump(mode="json")` which includes all fields — should work without change)
4. Verify `from_json_line()` handles missing fields (pydantic defaults to None — should work without change)
5. Run existing tests/test_schemas.py to ensure backward compatibility (all must still pass)

## Tests (add to tests/test_schemas.py — DO NOT create a new test file)
Add a new test class `TestP3OptionalFields` with these tests:

1. `test_new_fields_default_none`: construct a Sample without variant_type/bug_type/source_split → fields are None
2. `test_new_fields_set`: construct a Sample with all 3 fields set → fields round-trip correctly
3. `test_backward_compat_existing_json`: load a JSON line (dict) that does NOT contain the 3 new fields → Sample constructs successfully with fields=None
4. `test_serialization_includes_new_fields`: set variant_type="boundary", serialize via to_json_line, parse JSON, assert "variant_type" key present with value "boundary"
5. `test_serialization_omits_none_fields_check`: set fields to None, serialize, parse — fields present with null value (pydantic default includes None fields). Just assert the keys exist with null values (do NOT use exclude_none — we want explicit null for schema clarity). If pydantic excludes None by default in model_dump, adjust to ensure fields are always present. Verify behavior and document in test.
6. `test_boundary_variant_is_not_new_task_type`: construct a Sample with task_type="code_generation" and variant_type="boundary" → loads fine, task_type is still "code_generation"
7. `test_round_trip_with_new_fields`: construct → to_json_line → from_json_line → assert all 3 fields match original
8. `test_bug_type_free_form`: set bug_type="some_new_bug_type_not_in_list" → loads fine (no enum restriction)

## Report File
Write your full report to: `.superpowers/sdd/task-2-report.md`
Return only: status (DONE/DONE_WITH_CONCERNS/BLOCKED/NEEDS_CONTEXT), commit hash, one-line test summary, concerns.

## Commit
- Stage: `src/schemas.py`, `tests/test_schemas.py`
- Commit message: `feat(p3): add variant_type/bug_type/source_split optional fields to Sample schema`
- Single commit.

## Working Directory
e:\agent\Qwen\qwen3-code-lab

## Test Verification
Run ALL schema tests (existing + new):
`cd e:\agent\Qwen\qwen3-code-lab ; python -m pytest tests/test_schemas.py -v`

Confirm: existing tests still pass (backward compat) AND new tests pass.

## Global Constraints (from .superpowers/sdd/p3-plan.md)
- Backward compatible (existing JSON without new fields loads fine)
- No new task_type (boundary is variant_type)
- Optional fields default to None
- Surgical change (only add fields, no refactor)
- Do NOT modify to_chatml (metadata only)
- Do NOT modify Verification class
