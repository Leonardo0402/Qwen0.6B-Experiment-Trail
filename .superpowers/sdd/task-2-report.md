# Task 2: JsonProtocol (Baseline) — Report

**Status:** DONE
**Commit SHA:** `6a2b99555d46f2f4d5d68ba7011dc59820cfecb1`
**Branch:** `feat/p4-1b-protocol-ablation`
**Base SHA:** `4a286ad`

## Summary

Implemented the JSON baseline protocol (`JsonProtocol`) for the P4.1b Protocol Ablation project. This task supersedes a previously BLOCKED dispatch — the brief was updated to fix a plan bug in the `parse_output` implementation.

## Files Committed

| File | Status | Lines Changed |
|------|--------|---------------|
| `src/protocols/json_protocol.py` | new file | +127 |
| `src/protocols/__init__.py` | modified | +2 / -1 |
| `tests/test_protocol_json.py` | new file | +132 |

Total: 3 files changed, 261 insertions(+), 1 deletion(-)

## Implementation Notes

### The Plan Bug (fixed in updated brief)

The previous brief's `parse_output` had a `try/except` block where the `except (json.JSONDecodeError, Exception)` clause set `diag.failure_class = "FORMAT_PARSE_FAIL"` and **returned early**, which meant the format-only repair path was unreachable for any input that failed `json.loads` outright. This caused `test_repair_fixes_trailing_comma` and `test_repair_path_returns_action` to fail (the JSON in those tests has a trailing comma that requires repair).

### The Fix (per updated brief)

The `parse_output` now uses a `data = None` sentinel pattern:

```python
data = None
try:
    data = json.loads(json_str)
    if isinstance(data, dict):
        action_type = data.get("action_type", "")
        diag.action_type_valid = self.is_valid_action_type(action_type)
except (json.JSONDecodeError, Exception):
    pass  # Fall through to repair path
```

- On `JSONDecodeError`, the code **does not return early** — it falls through to the repair path.
- After try, direct validation only runs if `data is not None`.
- Repair path then attempts `repair_json(json_str)` followed by `json.loads` + `validate_action`.
- The P4.1 repair-path bug fix (`return action, diag` after successful repair) is preserved.

### Failure Classification

Three failure classes are emitted at the end:
- `FORMAT_PARSE_FAIL` — when `data is None` (repair could not even parse)
- `UNKNOWN_ACTION_TYPE` — when `data` parsed but `action_type` is invalid
- `SCHEMA_VALIDATION_FAIL` — when `action_type` is valid but full schema validation fails

## Test Command & Output

```
$ py -3.11 -m pytest tests/test_protocol_json.py -v
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 11 items

tests\test_protocol_json.py ...........                                  [100%]

============================= 11 passed in 0.42s ==============================
```

**Result: 11/11 passed, no warnings.**

## Self-Review Checklist

- [x] Overwrote `src/protocols/json_protocol.py` with the corrected code from the updated brief.
- [x] `src/protocols/__init__.py` exports `JsonProtocol` (via `from src.protocols.json_protocol import JsonProtocol` and `__all__`).
- [x] P4.1 repair-path bug fix (`return action, diag` after successful repair) is present at line 116.
- [x] All 11 tests pass with pristine output (no warnings).
- [x] Committed with the exact message from the brief: `feat(protocols): add JsonProtocol baseline with independent diagnostics (P4.1b T2)`.
- [x] Staged exactly the 3 files specified: `src/protocols/json_protocol.py`, `src/protocols/__init__.py`, `tests/test_protocol_json.py`.

## Concerns

None. The implementation matches the corrected brief verbatim, all tests pass cleanly, and the commit contains only the three specified files.
