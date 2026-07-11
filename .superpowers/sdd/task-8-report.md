# Task 8 Report: Verdict Decision Logic

## 1. Files Modified

- `scripts/run_protocol_ablation.py`
  - Added `_ALLOWED_VERDICTS` set (after `generate_report()`, before `main()`)
  - Added `compute_verdict(results)` function (after `_ALLOWED_VERDICTS`, before `main()`)
  - Updated `main()` with Step 6 verdict block (after Step 5 generate report, before final `print(f"\nDone. Reports in {_REPORT_DIR}")` line)
- `tests/test_protocol_ablation.py`
  - Appended 7 new verdict tests at end of file (after existing 6 tests)

Commit SHA: `095c822bca6c3905a4d6419136dec870b2431971`
Branch: `feat/p4-1b-protocol-ablation`

## 2. Test Results

### Step 2: Verify new test fails (before implementation)

```
============================= test session starts =============================
platform win32 -- Python 3.11.11, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 1 item

tests\test_protocol_ablation.py F                                        [100%]

================================== FAILURES ===================================
________________ test_verdict_keep_action_json_when_json_best _________________

    def test_verdict_keep_action_json_when_json_best():
>       from scripts.run_protocol_ablation import compute_verdict
E       ImportError: cannot import name 'compute_verdict' from 'scripts.run_protocol_ablation' (E:\agent\Qwen\qwen3-code-lab\scripts\run_protocol_ablation.py)

tests\test_protocol_ablation.py:229: ImportError
=========================== short test summary info ===========================
FAILED tests/test_protocol_ablation.py::test_verdict_keep_action_json_when_json_best - ImportError: cannot import name 'compute_verdict' from 'scripts.run_protocol_ablation' (E:\agent\Qwen\qwen3-code-lab\scripts\run_protocol_ablation.py)
============================== 1 failed in 0.57s ===============================
```

✅ Expected failure with `ImportError: cannot import name 'compute_verdict'`.

### Step 4: All ablation tests (13/13 pass)

```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 13 items

tests\test_protocol_ablation.py .............                            [100%]

=================================== PASSES ====================================
___________________ test_run_combination_with_mock_protocol ___________________
---------------------------- Captured stdout call -----------------------------
[mock/mock-base] 2/2 task_002
=========================== short test summary info ===========================
PASSED tests/test_protocol_ablation.py::test_baseline_lock_records_all_fields
PASSED tests/test_protocol_ablation.py::test_aggregate_metrics_computes_all_fields
PASSED tests/test_protocol_ablation.py::test_classify_failures_returns_taxonomy
PASSED tests/test_protocol_ablation.py::test_run_combination_with_mock_protocol
PASSED tests/test_protocol_ablation.py::test_generate_report_contains_all_protocols
PASSED tests/test_protocol_ablation.py::test_generate_report_has_markdown_table
PASSED tests/test_protocol_ablation.py::test_verdict_keep_action_json_when_json_best
PASSED tests/test_protocol_ablation.py::test_verdict_try_tag_when_tag_significantly_better
PASSED tests/test_protocol_ablation.py::test_verdict_try_dsl_when_dsl_significantly_better
PASSED tests/test_protocol_ablation.py::test_verdict_fix_prompt_when_all_below_30pct
PASSED tests/test_protocol_ablation.py::test_verdict_fix_evaluator_when_model_load_fails
PASSED tests/test_protocol_ablation.py::test_verdict_fix_evaluator_when_high_crash
PASSED tests/test_protocol_ablation.py::test_verdict_is_valid_enum
============================= 13 passed in 0.60s ==============================
```

✅ All 13 ablation tests pass (6 existing + 7 new from T8).

### Step 5: Full protocol test suite (66/66 pass across 7 files)

```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 66 items

tests\test_protocol_base.py .......                                      [ 10%]
tests\test_protocol_json.py ...........                                  [ 27%]
tests\test_protocol_tag.py ..........                                    [ 42%]
tests\test_protocol_dsl.py .........                                     [ 56%]
tests\test_protocol_ablation.py .............                            [ 75%]
tests\test_agent_model_provider_protocol.py ......                       [ 84%]
tests\test_agent_model_provider.py ..........                            [100%]

=================================== PASSES ====================================
___________________ test_run_combination_with_mock_protocol ___________________
---------------------------- Captured stdout call -----------------------------
[mock/mock-base] 2/2 task_002
=========================== short test summary info ===========================
PASSED tests/test_protocol_base.py::test_protocol_diagnostics_has_all_fields
PASSED tests/test_protocol_base.py::test_protocol_diagnostics_failure_class_set
PASSED tests/test_protocol_base.py::test_protocol_diagnostics_model_dump_works
PASSED tests/test_protocol_base.py::test_validate_action_returns_none_for_invalid
PASSED tests/test_protocol_base.py::test_validate_action_returns_none_for_empty
PASSED tests/test_protocol_base.py::test_is_valid_action_type_recognizes_11_types
PASSED tests/test_protocol_base.py::test_is_valid_action_type_rejects_unknown
PASSED tests/test_protocol_json.py::test_valid_action_parses
PASSED tests/test_protocol_json.py::test_fenced_json_parses
PASSED tests/test_protocol_json.py::test_malformed_json_fails
PASSED tests/test_protocol_json.py::test_unknown_action_type_fails
PASSED tests/test_protocol_json.py::test_forbidden_path_fails
PASSED tests/test_protocol_json.py::test_missing_required_field_fails
PASSED tests/test_protocol_json.py::test_repair_fixes_trailing_comma
PASSED tests/test_protocol_json.py::test_repair_does_not_change_semantics
PASSED tests/test_protocol_json.py::test_empty_output_fails
PASSED tests/test_protocol_json.py::test_repair_path_returns_action
PASSED tests/test_protocol_json.py::test_build_system_prompt_contains_format_instructions
PASSED tests/test_protocol_tag.py::test_valid_tag_action_parses
PASSED tests/test_protocol_tag.py::test_think_block_ignored
PASSED tests/test_protocol_tag.py::test_content_subtag_handled
PASSED tests/test_protocol_tag.py::test_unclosed_tag_fails
PASSED tests/test_protocol_tag.py::test_unknown_key_fails
PASSED tests/test_protocol_tag.py::test_unknown_action_type_fails
PASSED tests/test_protocol_tag.py::test_forbidden_path_fails
PASSED tests/test_protocol_tag.py::test_empty_output_fails
PASSED tests/test_protocol_tag.py::test_build_system_prompt_contains_format_instructions
PASSED tests/test_protocol_tag.py::test_minimal_tag_parses_with_defaults
PASSED tests/test_protocol_dsl.py::test_valid_dsl_action_parses
PASSED tests/test_protocol_dsl.py::test_heredoc_multiline_handled
PASSED tests/test_protocol_dsl.py::test_malformed_dsl_fails
PASSED tests/test_protocol_dsl.py::test_missing_value_fails
PASSED tests/test_protocol_dsl.py::test_unknown_action_type_fails
PASSED tests/test_protocol_dsl.py::test_forbidden_path_fails
PASSED tests/test_protocol_dsl.py::test_empty_output_fails
PASSED tests/test_protocol_dsl.py::test_build_system_prompt_contains_format_instructions
PASSED tests/test_protocol_dsl.py::test_minimal_dsl_parses_with_defaults
PASSED tests/test_protocol_ablation.py::test_baseline_lock_records_all_fields
PASSED tests/test_protocol_ablation.py::test_aggregate_metrics_computes_all_fields
PASSED tests/test_protocol_ablation.py::test_classify_failures_returns_taxonomy
PASSED tests/test_protocol_ablation.py::test_run_combination_with_mock_protocol
PASSED tests/test_protocol_ablation.py::test_generate_report_contains_all_protocols
PASSED tests/test_protocol_ablation.py::test_generate_report_has_markdown_table
PASSED tests/test_protocol_ablation.py::test_verdict_keep_action_json_when_json_best
PASSED tests/test_protocol_ablation.py::test_verdict_try_tag_when_tag_significantly_better
PASSED tests/test_protocol_ablation.py::test_verdict_try_dsl_when_dsl_significantly_better
PASSED tests/test_protocol_ablation.py::test_verdict_fix_prompt_when_all_below_30pct
PASSED tests/test_protocol_ablation.py::test_verdict_fix_evaluator_when_model_load_fails
PASSED tests/test_protocol_ablation.py::test_verdict_fix_evaluator_when_high_crash
PASSED tests/test_protocol_ablation.py::test_verdict_is_valid_enum
PASSED tests/test_agent_model_provider_protocol.py::test_protocol_none_uses_legacy_path
PASSED tests/test_agent_model_provider_protocol.py::test_protocol_set_uses_protocol_path
PASSED tests/test_agent_model_provider_protocol.py::test_protocol_set_builds_system_prompt
PASSED tests/test_agent_model_provider_protocol.py::test_protocol_set_records_protocol_diagnostics
PASSED tests/test_agent_model_provider_protocol.py::test_reset_clears_protocol_diagnostics
PASSED tests/test_agent_model_provider_protocol.py::test_existing_tests_still_pass_with_protocol_none
PASSED tests/test_agent_model_provider.py::test_build_prompt_produces_nonempty_string
PASSED tests/test_agent_model_provider.py::test_extract_json_finds_first_json_block
PASSED tests/test_agent_model_provider.py::test_extract_json_returns_none_on_no_json
PASSED tests/test_agent_model_provider.py::test_repair_json_strips_markdown_fences
PASSED tests/test_agent_model_provider.py::test_repair_json_removes_trailing_commas
PASSED tests/test_agent_model_provider.py::test_repair_json_does_not_choose_action_type
PASSED tests/test_agent_model_provider.py::test_sentinel_action_marks_invalid
PASSED tests/test_agent_model_provider.py::test_model_provider_extracts_valid_json_mocked
PASSED tests/test_agent_model_provider.py::test_model_provider_records_diagnostics_on_invalid_json
PASSED tests/test_agent_model_provider.py::test_model_provider_repair_strips_fences_then_validates
============================= 66 passed in 0.72s ==============================
```

✅ All 66 tests pass across all 7 protocol test files. No regressions.

### Commit

```
[feat/p4-1b-protocol-ablation 095c822] feat(p4-1b): add verdict decision logic (P4.1b T8)
 2 files changed, 210 insertions(+)
```

Commit message (multi-line, preserved via `git commit -F`):
```
feat(p4-1b): add verdict decision logic (P4.1b T8)

Verdict rules:
- Alternative protocol >30% better than JSON → TRY_TAG/TRY_DSL
- All protocols <30% schema_valid → FIX_PROMPT_FIRST
- JSON highest → KEEP_ACTION_JSON
- Fallback → STOP_PROTOCOL_CHANGE
```

## 3. Deviations from the Brief

None. All code (function bodies, test bodies, type annotations, main() Step 6 block, commit message) was added verbatim from the brief. The brief specified PowerShell-incompatible heredoc syntax for the commit; used `git commit -F <file>` with a temporary file containing the exact verbatim commit message, then deleted the temp file. Commit message content is byte-identical to the brief.

## 4. Self-Review Notes

- **Additive only:** No existing functions or tests were modified. Only additions: `_ALLOWED_VERDICTS`, `compute_verdict()`, Step 6 block in `main()`, and 7 new tests appended to end of test file.
- **Placement verified:** `_ALLOWED_VERDICTS` and `compute_verdict()` are placed AFTER `generate_report()` (which ends with `return "\n".join(lines)` at line 356) and BEFORE `main()` (now starting at line 429).
- **Step 6 placement verified:** Step 6 block in `main()` is placed AFTER the Step 5 generate-report block and BEFORE the final `print(f"\nDone. Reports in {_REPORT_DIR}")` line.
- **Imports already present:** `json` is already imported at module top (used by other steps in `main()`), so the Step 6 `json.dumps` call works without additional imports.
- **Type annotation note:** The brief's `compute_verdict` uses `dict[str, float]` annotations for `proto_rates`/`proto_safety` even though they hold lists during accumulation. This is verbatim from the brief and does not affect runtime behavior (Python does not enforce annotations). Not raised as a deviation since the brief explicitly specified the verbatim code.
- **TDD discipline:** Step 2 confirmed the test failed with the expected `ImportError` before any implementation was added.
- **Test count math:** 13 ablation tests = 4 (T6: baseline_lock, aggregate_metrics, classify_failures, run_combination) + 2 (T7: generate_report × 2) + 7 (T8: verdict × 7). Matches brief's expectation.
- **Full suite math:** 66 tests = 7 (base) + 11 (json) + 10 (tag) + 9 (dsl) + 13 (ablation) + 6 (provider_protocol) + 10 (provider) = 66. All pass.
- **Git hygiene:** Only `scripts/run_protocol_ablation.py` and `tests/test_protocol_ablation.py` were staged and committed. The pre-existing uncommitted changes to `.superpowers/sdd/*` and `data/p3-limited/*` were left untouched.
