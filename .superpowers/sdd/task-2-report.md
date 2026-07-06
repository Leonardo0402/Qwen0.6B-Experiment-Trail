# Task 2 Report: Extend Sample Schema

## Status
DONE

## Commit
- Hash: `5b88a6e`
- Branch: `feat/p3-capability-expansion-v2`
- Parent: `48614af` (Task 1)
- Message: `feat(p3): add variant_type/bug_type/source_split optional fields to Sample schema`
- Files staged (only these two):
  - `src/schemas.py`
  - `tests/test_schemas.py`

## Test Summary
48/48 tests passed (`python -m pytest tests/test_schemas.py -v`)
- Existing tests: 40 (backward compat verified — all green, no regressions)
- New P3 tests: 8 (`TestP3OptionalFields`)

## Changes Made

### `src/schemas.py` (surgical — 3 new fields only)
Added 3 optional free-form `Optional[str] = None` fields to the `Sample` model, placed immediately after `dataset_version` and before the field-level validators section:

```python
generator: str
created_at: str
dataset_version: str
# P3 optional metadata fields (free-form strings, default None for
# backward compatibility with pre-P3 JSONL files).
variant_type: Optional[str] = None
bug_type: Optional[str] = None
source_split: Optional[str] = None
```

No other changes to `schemas.py`:
- `TaskType` enum untouched (no new task_type)
- Existing field validators (`difficulty_in_range`, `language_must_be_python`, `non_empty_string`) untouched
- `check_repair_fields` model_validator untouched
- `to_json_line` / `from_json_line` untouched (pydantic `model_dump(mode="json")` already includes all fields by default — including `None` as `null`)
- `to_chatml` untouched (these fields are metadata, not chat content)
- `Verification` class untouched

### `tests/test_schemas.py` (additive — new class only)
Appended a new test class `TestP3OptionalFields` (8 tests). Existing helper functions `_base_sample`, `_repair_sample`, `_exec_repair_sample`, `_verification` were reused unmodified (the `_base_sample` helper naturally accepts the new kwargs via its `**kwargs` passthrough, so no helper change was needed).

The 8 tests:
1. `test_new_fields_default_none` — Sample constructed without the 3 fields → fields are `None`
2. `test_new_fields_set` — all 3 fields set → round-trip via attribute access
3. `test_backward_compat_existing_json` — pre-P3 JSON line lacking the 3 keys → loads via `from_json_line` with `None` defaults
4. `test_serialization_includes_new_fields` — `to_json_line` JSON contains `variant_type: "boundary"`
5. `test_serialization_includes_none_as_null` — pydantic default: `None` fields appear as explicit `null` (verified behavior — `model_dump(mode="json")` does NOT omit None fields by default)
6. `test_boundary_variant_is_not_new_task_type` — `task_type="code_generation"` + `variant_type="boundary"` loads fine, task_type stays code_generation (boundary is variant, not task_type)
7. `test_round_trip_with_new_fields` — construct → `to_json_line` → `from_json_line` → all 3 fields match original, and the full Sample equality also holds
8. `test_bug_type_free_form` — arbitrary `bug_type="some_new_bug_type_not_in_list"` loads (no enum restriction, no field_validator)

## Constraint Compliance Checklist
- [x] Backward compatible: pre-P3 JSON without the 3 fields loads with `None` defaults (test #3)
- [x] No new task_type: boundary expressed as `variant_type="boundary"` with `task_type="code_generation"` (test #6)
- [x] Optional fields default `None` (test #1)
- [x] Free-form strings, no `field_validator` restricting to a fixed set (test #8)
- [x] `to_json_line()` includes the fields, including explicit `null` for `None` (tests #4 and #5)
- [x] `from_json_line()` handles missing fields (test #3)
- [x] `to_chatml` unmodified
- [x] `Verification` class unmodified
- [x] Surgical change: only 3 fields added to Sample; no refactor of existing validators
- [x] No new test file created; tests appended to existing `tests/test_schemas.py`
- [x] Existing code style matched (4-space indent, `_base_sample(**kwargs)` helper reuse, descriptive Chinese-aware English docstrings, comment headers with `# ---` separators)
- [x] No emojis in code or commit message
- [x] Single commit
- [x] Only `src/schemas.py` and `tests/test_schemas.py` staged

## Key Findings
- **pydantic serialization behavior verified**: `model_dump(mode="json")` includes `None`-valued fields as `null` by default. The `to_json_line()` method does NOT use `exclude_none`, so the 3 new fields always appear in the JSONL output. This satisfies constraint #5 of the brief without any code change to `to_json_line()`.
- **Helper compatibility**: `_base_sample(**kwargs)` already passes through arbitrary kwargs, so `_base_sample(variant_type="boundary", ...)` works without modifying the helper. This keeps the diff surgical.

## Concerns
None.
