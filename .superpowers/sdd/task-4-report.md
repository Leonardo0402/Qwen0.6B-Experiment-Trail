# Task 4 Report: DslProtocol with heredoc support

## What was implemented

Added `DslProtocol` — the third action protocol (Protocol C) for the P4.1b
ablation. It uses a one-line DSL format (`ACTION <type> <k>=<v> ...`) with
heredoc (`<<EOF ... EOF`) support for multiline content such as patch
old_text/new_text.

Three files touched, exactly as specified by the brief:

1. **`src/protocols/dsl_protocol.py`** (new, 210 lines) — `DslProtocol` class:
   - `name = "dsl"`
   - `build_system_prompt(task_context)` returns prompt with `ACTION` format
     instructions, heredoc examples, and tool list.
   - `parse_output(raw)` regex-parses the ACTION line, then heredoc blocks,
     then key=value pairs; fills defaults for `action_id`, `reason_short`,
     `expected_observation`, `safety_flags`, and `finish`-specific defaults
     (`success_criterion`, `tests_passed`, `identification_verified` — all
     string `"false"` per the brief's ambiguity fix).
   - `_coerce_arguments` @staticmethod coerces bool/int/float fields.
   - Failure paths:
     - No ACTION line → `FORMAT_PARSE_FAIL`
     - Unclosed heredoc → `FORMAT_PARSE_FAIL`
     - Unknown action_type → `UNKNOWN_ACTION_TYPE`
     - Pydantic validation fails (e.g. `path=../etc/passwd`, empty `path=`) →
       `SCHEMA_VALIDATION_FAIL`

2. **`src/protocols/__init__.py`** (modified) — added `DslProtocol` import and
   `__all__` entry.

3. **`tests/test_protocol_dsl.py`** (new, 103 lines) — 9 tests verbatim from
   the brief.

## Test results (TDD evidence)

### RED phase (before implementation)

```
tests\test_protocol_dsl.py:4: in <module>
    from src.protocols.dsl_protocol import DslProtocol
E   ModuleNotFoundError: No module named 'src.protocols.dsl_protocol'
```

Exit code 2 — collection error, exactly as expected per brief Step 2.

### GREEN phase (after implementation)

```
collected 9 items
tests\test_protocol_dsl.py .........    [100%]
============================== 9 passed in 0.43s ==============================
```

All 9 tests pass. Output is pristine (no warnings, no skips).

### Regression check (full protocol suite)

```
collected 37 items
tests\test_protocol_base.py .......       [ 18%]
tests\test_protocol_json.py ...........   [ 48%]
tests\test_protocol_tag.py ..........     [ 75%]
tests\test_protocol_dsl.py .........      [100%]
============================== 37 passed in 0.54s ==============================
```

No regression in T1/T2/T3 tests.

## Files changed

- `src/protocols/dsl_protocol.py` (new, 210 lines)
- `src/protocols/__init__.py` (+5 / -1)
- `tests/test_protocol_dsl.py` (new, 103 lines)

Commit: `53fae0d feat(protocols): add DslProtocol with heredoc support (P4.1b T4)`
on branch `feat/p4-1b-protocol-ablation`. 3 files staged as required.

## Self-review findings

- Code is a verbatim transcription of the brief — no deviations.
- The `finish` action's `tests_passed`/`identification_verified` defaults use
  string `"false"` (not bool `False`), as required by the brief's ambiguity
  fix (`.lower()` is called in `_coerce_arguments`).
- `_ARGUMENT_KEYS` frozenset is defined but not strictly used for filtering
  (unlike TagProtocol which checks unknown keys). The DSL protocol relies on
  Pydantic validation to reject unknown keys. This matches the brief exactly —
  not a concern, just an observation.
- Heredoc parser uses `remaining.find("\nEOF", ...)` which finds the first
  `\nEOF` after the heredoc start. For the test case with two consecutive
  heredocs this works correctly because the second heredoc's start position is
  after the first EOF (verified by `test_heredoc_multiline_handled` passing).
- `_KV_RE = re.compile(r'(\w+)=(".*?"|\S+)')` — for `path=` (empty value),
  `\S+` doesn't match, so the empty value is simply not added to arguments.
  Pydantic then fails on missing required `path` field, producing the expected
  `SCHEMA_VALIDATION_FAIL`. Test `test_missing_value_fails` confirms this.
- No silent repair logic; `repair_attempted`/`repair_success` stay `False`.

## Concerns

None. Implementation matches brief verbatim, all tests pass cleanly.
