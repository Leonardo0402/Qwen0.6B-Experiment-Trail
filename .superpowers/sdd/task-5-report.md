# Task 5 Report: ModelActionProvider Protocol Adapter

**Status:** DONE
**Commit:** `d92787c` on `feat/p4-1b-protocol-ablation` (parent: `53fae0d`)
**Date:** 2026-07-11

## What was implemented

Added a protocol adapter layer to `ModelActionProvider` so it can delegate prompt
building and output parsing to a `ProtocolBase` instance when `protocol=...` is
passed, while preserving the existing legacy JSON path when `protocol=None`
(backward-compatible).

### Changes to `src/agent_model_provider.py`
1. Added `TYPE_CHECKING` import for `ProtocolBase` and `ProtocolDiagnostics` (avoids
   circular import — `tag_protocol.py` imports `SentinelAction` from this module).
2. Added `protocol: "ProtocolBase | None" = None` parameter to `__init__`; stored
   as `self._protocol` and accompanied by `self._protocol_diagnostics = []`.
3. Added `_build_user_prompt(state)` — builds the user-facing state prompt for the
   protocol path (task_id, step, notes, hypothesis).
4. Split `next_action` into a dispatcher plus two private methods:
   - `_next_action_protocol` — delegates prompt + parsing to the protocol and
     records `ProtocolDiagnostics`.
   - `_next_action_legacy` — the original JSON path, with the **P4.1 fix**: the
     repair-success branch now `return action` instead of falling through to
     `SentinelAction`.
5. Updated `reset` to clear both `_diagnostics` and `_protocol_diagnostics`.
6. Updated `diagnostics` property to return `ProtocolDiagnostics` when protocol
   is set, else `ModelStepDiagnostics` (intentional type divergence).

### Changes to `tests/test_agent_model_provider.py`
Updated the 3 tests that use `ModelActionProvider.__new__()` to manually set
`provider._protocol = None` and `provider._protocol_diagnostics = []` after
`provider._diagnostics = []`:
- `test_model_provider_extracts_valid_json_mocked`
- `test_model_provider_records_diagnostics_on_invalid_json`
- `test_model_provider_repair_strips_fences_then_validates`

Without these, `next_action`'s `self._protocol` check raises `AttributeError`.

### New file `tests/test_agent_model_provider_protocol.py` (6 tests, verbatim from brief)
- `test_protocol_none_uses_legacy_path`
- `test_protocol_set_uses_protocol_path`
- `test_protocol_set_builds_system_prompt`
- `test_protocol_set_records_protocol_diagnostics`
- `test_reset_clears_protocol_diagnostics`
- `test_existing_tests_still_pass_with_protocol_none`

## TDD evidence

### RED (before implementation)
```
collected 6 items
tests\test_agent_model_provider_protocol.py FFFFFF [100%]
FAILED ... TypeError: ModelActionProvider.__init__() got an unexpected keyword argument 'protocol'
FAILED ... AttributeError: 'ModelActionProvider' object has no attribute '_protocol'
6 failed in 0.56s
```

### GREEN (after implementation)
```
collected 16 items
tests\test_agent_model_provider_protocol.py ...... [ 37%]
tests\test_agent_model_provider.py ..........    [100%]
16 passed in 0.55s
```

### Broader non-GPU suite
```
py -3.11 -m pytest tests/ -p no:warnings --tb=no -q -m "not gpu" --timeout=120 \
  --ignore=tests/test_data_pipeline.py --ignore=tests/test_p3_readiness_gate.py
```
Exit code 0. All dots (no `F`, no `E`). One collection warning unrelated to this
task (`TestObservation` has `__init__`, pre-existing in `src/agent_tools.py`).
One skipped test (pre-existing, unrelated).

## Files changed
- `src/agent_model_provider.py` (+47 / -4)
- `tests/test_agent_model_provider.py` (+6)
- `tests/test_agent_model_provider_protocol.py` (+108, new)

Total: 3 files changed, 161 insertions(+), 4 deletions(-).

## Self-review findings

- ✅ `TYPE_CHECKING` guard prevents the circular import (tag_protocol →
  agent_model_provider → protocols.base would be a cycle otherwise at type-check
  time, but `if TYPE_CHECKING:` blocks are not executed at runtime).
- ✅ The `diagnostics` property returns different concrete types depending on the
  protocol flag. This is intentional per the brief and tested by
  `test_protocol_none_uses_legacy_path` (asserts `"json_parse_ok" in d`) and
  `test_protocol_set_uses_protocol_path` (asserts `"format_parse_ok" in d` and
  `"json_parse_ok" not in d`).
- ✅ The P4.1 repair-path bug fix (`return action` after successful repair) is
  verified by the existing `test_model_provider_repair_strips_fences_then_validates`
  test, which now passes (previously the test was passing because the fence was
  stripped by `extract_json` directly, but the repair-path branch is now also
  correct).
- ✅ Backward compatibility: `ModelActionProvider(model_path="fake")` still works
  and `_protocol` defaults to `None`.
- ✅ Surgical changes: every changed line traces directly to the brief. No
  unrelated refactors, no comment additions to adjacent code.
- ✅ No imports were left unused. `Any` is still used in the typing import line
  (unchanged from original).

## Concerns

None. The implementation matches the brief verbatim, all tests pass (new, existing,
and broader non-GPU suite), and the commit message matches the brief exactly.

One minor observation (not a concern, just a note): the legacy `_next_action_legacy`
method sets `safety_valid = True` together with `schema_valid` (the original P4.1
behavior). The brief did not ask to fix this in the legacy path — the independent
dimension computation is a property of the *protocol* path only (via
`ProtocolDiagnostics`). This is consistent with the brief's intent.
