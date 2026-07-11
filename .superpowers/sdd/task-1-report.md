# Task 1 Report: ProtocolBase ABC + ProtocolDiagnostics

## What was implemented

Created the foundational abstraction layer for P4.1b Protocol Ablation (Issue #29). T2-T4 (JSON/Tag/DSL protocols) will inherit from `ProtocolBase` and emit `ProtocolDiagnostics` per parse step.

- `ProtocolBase` — ABC defining the protocol contract: `name` (abstract property), `build_system_prompt` (abstract), `parse_output` (abstract), plus two static helpers `validate_action` and `is_valid_action_type`.
- `ProtocolDiagnostics` — Pydantic BaseModel with 10 fields. Each validity dimension (format_parse_ok / schema_valid / safety_valid / action_type_valid / arguments_valid) is computed independently, fixing the P4.1 bug where all 4 were set together.
- `validate_action` uses `TypeAdapter(Action).validate_python()` (not `Action.model_validate()` which fails on the Annotated union).
- `SentinelAction` imported under `TYPE_CHECKING` to avoid circular imports (only used in type hint of `parse_output`).
- `_ACTION_ADAPTER` is a module-level singleton; `_ALLOWED_ACTION_TYPES` is a `frozenset` for O(1) lookup.

## TDD Evidence

### RED (before implementation)
```
collected 0 items / 1 error
ERROR collecting tests/test_protocol_base.py
E   ModuleNotFoundError: No module named 'src.protocols'
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

### GREEN (after implementation)
```
collected 7 items

tests\test_protocol_base.py .......                                      [100%]

============================== 7 passed in 0.46s ==============================
```

7 tests, all passing:
1. `test_protocol_diagnostics_has_all_fields`
2. `test_protocol_diagnostics_failure_class_set`
3. `test_protocol_diagnostics_model_dump_works`
4. `test_validate_action_returns_none_for_invalid`
5. `test_validate_action_returns_none_for_empty`
6. `test_is_valid_action_type_recognizes_11_types`
7. `test_is_valid_action_type_rejects_unknown`

## Files changed

| File | Change | Lines |
|------|--------|-------|
| `src/protocols/__init__.py` | Created | +7 |
| `src/protocols/base.py` | Created | +82 |
| `tests/test_protocol_base.py` | Created | +72 |

Total: 3 files, 161 insertions, 0 deletions. No existing files modified.

## Commit

- SHA: `4a286ad`
- Subject: `feat(protocols): add ProtocolBase ABC and ProtocolDiagnostics (P4.1b T1)`
- Branch: `feat/p4-1b-protocol-ablation`

## Self-review findings

1. **Files match brief verbatim** — all 3 files written word-for-word from the task brief, including docstrings, comments, and the `_ALLOWED_ACTION_TYPES` frozenset contents.
2. **Surgical change** — only 3 new files added; no existing files touched (verified via `git show --stat HEAD`).
3. **Design decisions verified**:
   - `SentinelAction` is under `TYPE_CHECKING` block — circular import avoided.
   - `TypeAdapter(Action)` used, not `Action.model_validate()` — Annotated union compatibility maintained.
   - `ProtocolDiagnostics` is `BaseModel` (not dataclass) — `.model_dump()` works (test 3 confirms).
   - Module-level `_ACTION_ADAPTER` singleton — adapter not recreated per call.
4. **Test isolation** — ran `pytest tests/test_protocol_base.py` separately; 7/7 pass. No other test files were modified or impacted.
5. **Action union sanity** — verified `Action = Annotated[...]` exists at line 260 of `src/agent_actions.py` and `SentinelAction` at line 36 of `src/agent_model_provider.py`. Both imports resolve correctly.
6. **Unused import warning** — `pytest` is imported in the test file but not used directly. This is verbatim from the brief (likely included for future fixtures), so left as-is per surgical-changes principle.

## Concerns

None. Implementation is verbatim from the brief, tests pass, commit is clean. Ready for T2-T4 to inherit from `ProtocolBase`.
