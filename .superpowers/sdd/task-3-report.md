# Task 3: TagProtocol — Report

**Status:** DONE
**Commit SHA(s) produced:** `a8a7b410a637534c41559c595910a7498c184ad6`
**Branch:** `feat/p4-1b-protocol-ablation` (verified, base SHA `6a2b995`)

---

## Summary

Implemented `TagProtocol` (XML Tag action protocol) per the UPDATED Task 3 brief. The brief's two plan bugs (unknown-key detection no-op, missing finish defaults) were fixed in the brief's `parse_output` code. I transcribed the corrected code verbatim, then made one additional minimal fix (see "Deviation from brief" below) so all 10 tests pass.

---

## Files

| File | Status | Lines |
|------|--------|-------|
| `src/protocols/tag_protocol.py` | new file (created in this commit) | 252 |
| `tests/test_protocol_tag.py` | new file (already on disk from prior dispatch, committed unchanged) | 110 |
| `src/protocols/__init__.py` | modified (adds `TagProtocol` import/export, already on disk from prior dispatch) | 6 |

Only these 3 files were staged and committed. No other modified/untracked files were included.

---

## Test output (verbatim)

Command: `py -3.11 -m pytest tests/test_protocol_tag.py -v`

```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 10 items

tests\test_protocol_tag.py ..........                                    [100%]

============================= 10 passed in 0.49s ==============================
```

All 10 tests pass:
- `test_valid_tag_action_parses`
- `test_think_block_ignored`
- `test_content_subtag_handled`
- `test_unclosed_tag_fails`
- `test_unknown_key_fails` (now passes — unknown keys hard-fail with `SCHEMA_VALIDATION_FAIL`)
- `test_unknown_action_type_fails`
- `test_forbidden_path_fails`
- `test_empty_output_fails`
- `test_build_system_prompt_contains_format_instructions`
- `test_minimal_tag_parses_with_defaults` (now passes — finish defaults filled in)

---

## Deviation from brief (one minimal fix)

The updated brief's `parse_output` fills finish defaults as Python booleans:

```python
if action_type == "finish":
    arguments.setdefault("success_criterion", "test_pass")
    arguments.setdefault("tests_passed", False)          # Python bool
    arguments.setdefault("identification_verified", False)  # Python bool
```

However, the next line calls `self._coerce_arguments(action_type, arguments)`, which unconditionally calls `.lower()` on `tests_passed` and `identification_verified`:

```python
for bool_key in ("tests_passed", "identification_verified"):
    if bool_key in coerced:
        coerced[bool_key] = coerced[bool_key].lower() in ("true", "yes", "1")
```

Calling `False.lower()` raises `AttributeError: 'bool' object has no attribute 'lower'`, which causes `test_minimal_tag_parses_with_defaults` to fail (9/10 pass, 1 fail).

**Fix applied (minimal, preserves brief intent):** Changed the two boolean defaults from Python `False` to the string `"false"`. This is consistent with how all other tag values flow through the parser (every tag value is a string until coerced). The coercion step then converts `"false".lower() in ("true", "yes", "1")` → `False`, yielding the same intended boolean value.

```python
if action_type == "finish":
    arguments.setdefault("success_criterion", "test_pass")
    arguments.setdefault("tests_passed", "false")          # string, coerced to False
    arguments.setdefault("identification_verified", "false")  # string, coerced to False
```

This is the only deviation from the brief's verbatim code. The unknown-key detection logic (`has_unknown_key` tracking with `SCHEMA_VALIDATION_FAIL` return) was transcribed exactly as the updated brief specifies.

---

## Self-review checklist

- [x] Did I read the UPDATED brief before implementing? **YES** — confirmed both plan bugs were fixed in the brief.
- [x] Did I verify the branch is `feat/p4-1b-protocol-ablation`? **YES**.
- [x] Did I overwrite the stale `tag_protocol.py` with the corrected code? **YES**.
- [x] Are all 10 tests passing? **YES** — 10 passed in 0.49s.
- [x] Did I stage only the 3 files specified in the brief? **YES** — `src/protocols/tag_protocol.py`, `src/protocols/__init__.py`, `tests/test_protocol_tag.py`.
- [x] Did I commit with the exact message from the brief? **YES** — `feat(protocols): add TagProtocol with content subtag support (P4.1b T3)`.
- [x] Did I avoid touching `tests/test_protocol_tag.py` and `src/protocols/__init__.py` (already correct)? **YES** — neither file was modified; only staged and committed as-is.
- [x] Is the deviation from the brief documented? **YES** — see "Deviation from brief" above.

---

## Concerns

1. **Brief bug not fully fixed:** The updated brief fixed the unknown-key detection and added finish defaults, but the finish defaults use Python `bool` (`False`) which is incompatible with the existing `_coerce_arguments` method (expects strings). I made a minimal one-word fix (`False` → `"false"`) to resolve this. The brief author may want to update the brief to reflect this, so future dispatches transcribe working code verbatim.

2. **`_BASE_FIELD_MAP` is defined but unused:** Lines 48-52 define `_BASE_FIELD_MAP` mapping `action_id`/`reason`/`expected` to their ActionBase field names, but `parse_output` does not use it — it hard-codes the same mapping inline (lines 188-191). This is pre-existing in the brief's code and not causing test failures, so I left it untouched per the "surgical changes" rule. Mentioning it here for visibility.

---

## Status: DONE

**Return message:** DONE — `a8a7b410a637534c41559c595910a7498c184ad6` — 10/10 tests pass — one minimal fix to brief's finish defaults (bool `False` → string `"false"`) so they survive `_coerce_arguments`.
