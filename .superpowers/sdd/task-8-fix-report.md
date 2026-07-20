# Task 8 Fix Report

## Finding Addressed
**F1 (Important):** Missing test for the `STOP_PROTOCOL_CHANGE` fallback verdict path (Rule 5).

## Fix Applied
Appended one new test, `test_verdict_stop_protocol_change_on_fallback`, to `tests/test_protocol_ablation.py` after the existing `test_verdict_is_valid_enum` test.

The test constructs results where:
- `avg_schema = {json: 0.4, tag: 0.45, dsl: 0.3}`
- Rule 4 passes (all `model_load_ok=True`, no high crash)
- Rule 2 passes (not all rates < 0.30)
- Rule 1 passes (no alternative >30% better than JSON)
- Rule 3 fails (JSON rate 0.4 is not the max 0.45)
- Fallback returns `STOP_PROTOCOL_CHANGE`

## Verification
- Ran: `py -3.11 -m pytest tests/test_protocol_ablation.py -v`
- Result: **14/14 tests pass** (13 existing + 1 new)

## Commit
- SHA: `7a3bd66fdd9d850cfb1169698f986fbb2f908264`
- Message: `test(p4-1b): add STOP_PROTOCOL_CHANGE fallback verdict test (T8 review fix)`
- Branch: `feat/p4-1b-protocol-ablation`
- Files staged: `tests/test_protocol_ablation.py` only
