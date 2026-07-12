# P4.1b Trust Repair

**Issue:** #32
**PR:** #30
**Branch:** feat/p4-1b-protocol-ablation
**Date:** 2026-07-12

---

## 1. Task

Fix the evaluator/schema trust problems discovered in the P4.1b protocol
ablation experiment (Issue #32). Re-run the full frozen 240-run ablation
under strict schema validation, regenerate a trustworthy report, and
correct the training-route conclusion.

The original v1 ablation reported `schema_valid_rate = 96.25%` for
`json + repair-lora`, but this number was inflated by Pydantic's default
`extra="ignore"` silently dropping unknown fields. The true
`schema_valid_rate` under strict validation is `0.00%`.

## 2. Scope

- Task A: Strict Action Schema validation (`extra="forbid"` on all models)
- Task B: Tag/DSL protocols equally forbid semantic repair
- Task C: Refactor `ProtocolDiagnostics` into independent dimensions
- Task D: Fix failure taxonomy (no double-counting)
- Task E: Fix `finish_without_tests_count` aggregation
- Task F: Add numerator/denominator to every rate
- Task G: Regression test matrix (31 new tests)
- Historical data protection (v1 marked SUPERSEDED, not deleted)
- Re-run frozen 240-run ablation into `reports/p4/protocol-ablation-v2/`
- Correct training decision document (split Decision 1 vs Decision 2)

Out of scope (explicitly prohibited):
- Agent SFT, DPO, PPO, GRPO, RL training
- Model weight modification
- P3 adapter overwrite
- External dataset introduction
- PR #30 merge
- Option B single-arm training approval

## 3. Initial State

- Branch: `feat/p4-1b-protocol-ablation`
- HEAD: `d034fde` (`docs(p4-2): add training protocol decision record`)
- v1 ablation verdict: `KEEP_ACTION_JSON` (based on inflated 96.25%)
- v1 report path: `reports/p4/protocol-ablation/`
- v1 decision doc: `docs/decisions/2026-07-11-p4-2-training-protocol.md`
  (conflated protocol choice with training initialization, proposed
  Option B single-arm continual training)

Preflight result: **PASS** (unrelated working-tree changes preserved,
not staged into Issue #32 commits).

## 4. Root Cause

The v1 ablation trusted `schema_valid_rate` computed via Pydantic
validation that used the default `extra="ignore"` policy. When the model
emitted an action like:

```json
{
  "action_type": "list_files",
  "arguments": {"path": "solution.py"}
}
```

Pydantic silently dropped the unknown `path` field (the correct field
for `list_files` is `pattern`), validated the resulting empty arguments
dict, and reported `schema_valid = true`. This inflated
`schema_valid_rate` from the true `0.00%` to a fake `96.25%` for
`json + repair-lora`.

Secondary issues:
1. `unknown_action_count` counted format-parse failures as unknown
   actions (double-counting).
2. `finish_without_tests_count` was inferred from `success and not
   tests_passed`, which misses failed trajectories that finished without
   tests.
3. `ProtocolDiagnostics` set `schema_valid/safety_valid/arguments_valid/
   action_type_valid` all to True together when the full Action validated,
   losing dimension independence.
4. The v1 decision document used zero-shot `schema_valid_rate` to select
   the Agent SFT initialization (Option B), which is invalid because
   zero-shot failure does not imply post-SFT failure.

## 5. Changes

### Source files modified (Issue #32 scope)

| File | Change |
|------|--------|
| `src/agent_actions.py` | Added `ConfigDict(extra="forbid")` to all 13 Pydantic models (`SafetyFlags`, `ActionBase`, `ListFilesArgs`, `ReadFileArgs`, `SearchTextArgs`, `InspectTaskArgs`, `ProposePatchArgs`, `ApplyPatchArgs`, `RollbackPatchArgs`, `RunTestsArgs`, `InspectErrorArgs`, `WriteMemoryArgs`, `FinishArgs`). `SafetyFlags` keeps `frozen=True` and adds `extra="forbid"`. `ActionBase` keeps `use_enum_values=True` and adds `extra="forbid"`. |
| `src/protocols/base.py` | Added 4 independent check methods: `check_action_type_valid`, `check_safety_valid`, `check_arguments_valid`, `classify_failure`. `validate_action` returns `None` on any validation failure (including `P4ForbiddenActionError`) instead of raising. `ProtocolDiagnostics` fields `action_type_valid`, `arguments_valid`, `safety_valid` are now computed independently. |
| `src/protocols/json_protocol.py` | Calls `check_action_type_valid`, `check_safety_valid`, `check_arguments_valid` independently before `validate_action`. Repair path only fixes syntax (trailing comma, markdown fence), never changes semantics. |
| `src/protocols/tag_protocol.py` | Removed `has_unknown_key` filter that silently dropped unknown keys. All keys now passed through to Pydantic `extra="forbid"` validation. |
| `src/protocols/dsl_protocol.py` | Same independent check_* calls. Unknown `key=value` pairs in ACTION line are passed to Pydantic and rejected by `extra="forbid"`. |
| `src/agent_model_provider.py` | Legacy path now uses `ProtocolBase.check_*` methods for independent dimension computation. |
| `scripts/run_protocol_ablation.py` | `baseline_lock()` records `source_file_shas`, `environment`, `task_ids`, `created_at`, split `adapter_path` into `adapter_path_base`/`adapter_path_repair_lora`. `aggregate_metrics()` uses `failure_class=="UNKNOWN_ACTION_TYPE"` for `unknown_action_count` (not `action_type_valid=False`). `generate_report()` shows `numerator / denominator = rate` for all rates. `compute_verdict()` returns `FIX_PROMPT_FIRST` when all combinations have `schema_valid_rate < 30%`. |
| `tests/test_protocol_ablation.py` | Updated 3 tests to match new field names and aggregation logic. |

### New files (Issue #32 scope)

| File | Purpose |
|------|---------|
| `tests/test_trust_repair.py` | 31 regression tests covering strict schema, repair semantics, metrics aggregation, protocol consistency |
| `reports/p4/protocol-ablation/SUPERSEDED.md` | Marks v1 results as untrusted, points to v2 |
| `reports/p4/protocol-ablation-v2/` | Full v2 ablation artifacts (11 files) |
| `docs/decisions/2026-07-11-p4-2-training-protocol.md` | Rewritten: split Decision 1 (protocol) from Decision 2 (training init), status PROPOSED/DEFERRED_TO_TWO_ARM_PILOT |

### Unrelated working-tree changes (NOT staged into Issue #32 commits)

| File | Reason |
|------|--------|
| `AGENTS.md` | Pre-existing unrelated modification |
| `data/p3-limited/balanced-limited/manifest.json` | Timestamp drift |
| `data/p3-limited/balanced-limited/train.jsonl` | Timestamp drift |
| `data/p3-limited/repair-limited/manifest.json` | Timestamp drift |
| `data/p3-limited/repair-limited/train.jsonl` | Timestamp drift |
| `reports/p4/p4-1-readiness.md` | Pre-existing unrelated modification |
| `reports/p4/p4-1-test-evidence.json` | Pre-existing unrelated modification |
| `docs/agent-protocols/*` | Untracked, unrelated |
| `docs/workflow/*` | Untracked, unrelated |
| `test-results.xml` | Untracked test artifact |

## 6. Strict Schema Validation

All 13 Pydantic models now use `ConfigDict(extra="forbid")`. Unknown
fields cause `ValidationError` instead of being silently dropped.

### Invalid cases that now hard-fail

| Input | Expected | Actual |
|-------|----------|--------|
| `list_files.arguments.path` | Rejected (`path` not a `ListFilesArgs` field) | Rejected ✓ |
| `read_file.arguments.pattern` | Rejected (`pattern` not a `ReadFileArgs` field) | Rejected ✓ |
| `inspect_task.arguments.path` | Rejected (`path` not an `InspectTaskArgs` field) | Rejected ✓ |
| Unknown top-level action field | Rejected by `ActionBase.extra="forbid"` | Rejected ✓ |
| Unknown safety flag | Rejected by `SafetyFlags.extra="forbid"` | Rejected ✓ |
| `network_required=True` | Rejected (`P4ForbiddenActionError`) | Rejected ✓ |
| `reads_sensitive_path=True` | Rejected (`P4ForbiddenActionError`) | Rejected ✓ |
| Path traversal (`../`) | Rejected (path safety check) | Rejected ✓ |

### Valid cases that still pass

| Input | Expected | Actual |
|-------|----------|--------|
| `list_files.arguments.pattern=null` | Accepted | Accepted ✓ |
| `read_file.arguments.path="solution.py"` | Accepted | Accepted ✓ |
| `finish.arguments.success_criterion="test_pass"` | Accepted | Accepted ✓ |
| Optional fields omitted | Accepted | Accepted ✓ |

## 7. Diagnostics Repair

`ProtocolDiagnostics` now computes each dimension independently:

| Field | Semantics |
|-------|-----------|
| `format_parse_ok` | Protocol text parsed into structured payload (syntax only) |
| `action_type_valid` | `raw action_type` is one of the 11 allowed actions |
| `arguments_valid` | Raw `arguments` has correct field names, no unknown fields, required fields present, types correct, value constraints correct |
| `safety_valid` | Raw `safety_flags` has all required fields, no unknown fields, types correct, `network_required=False`, `reads_sensitive_path=False` |
| `schema_valid` | Complete raw payload passes Action union validation without dropping unknown fields or changing semantics |
| `repair_attempted` | Format-only repair was attempted |
| `repair_success` | Format-only repair succeeded AND repaired payload passes strict schema validation |
| `failure_class` | One of: `FORMAT_PARSE_FAIL`, `SCHEMA_VALIDATION_FAIL`, `UNKNOWN_ACTION_TYPE`, `FORBIDDEN_ACTION`, `INVALID_PATH`, `EMPTY_OR_USELESS_ACTION`, `MODEL_REFUSAL_OR_CHATTER`, `REPEATED_ACTION_LOOP` |

Previously, all four dimension fields were set to True together when the
full Action validated (with unknown fields silently dropped). Now each is
computed independently via `check_*` methods before `validate_action`.

## 8. Metrics Repair

### unknown_action_count

**Before:** `count where action_type_valid == False` (included
format-parse failures, double-counting).

**After:** `count where failure_class == "UNKNOWN_ACTION_TYPE"` (only
true unknown action types, format-parse failures excluded).

### finish_without_tests_count

**Before:** Inferred from `success and not tests_passed` (misses failed
trajectories that finished without tests).

**After:** Directly from evaluator-produced `finish_without_tests` field.
Failed trajectories are still counted.

### Numerator/Denominator transparency

Every rate in the report now shows `numerator / denominator = rate`:

```
schema_valid: 0/480 = 0.00%
format_parse: 480/480 = 100.00%
task_success: 0/40 = 0.00%
max_steps_hit: 40/40 = 100.00%
```

## 9. Regression Tests

New file: `tests/test_trust_repair.py` — 31 tests in 4 classes.

### TestStrictSchema (11 tests)
- `test_list_files_arguments_path_hard_fail`
- `test_read_file_arguments_pattern_hard_fail`
- `test_inspect_task_unknown_argument_hard_fail`
- `test_finish_unknown_argument_hard_fail`
- `test_unknown_top_level_action_field_hard_fail`
- `test_unknown_safety_flag_hard_fail`
- `test_valid_action_still_passes`
- `test_valid_optional_fields_still_pass`
- `test_path_traversal_hard_fail`
- `test_network_required_true_hard_fail`
- `test_reads_sensitive_path_true_hard_fail`

### TestRepairSemantics (7 tests)
- `test_trailing_comma_can_be_repaired`
- `test_markdown_fence_can_be_extracted`
- `test_repair_does_not_modify_action_type`
- `test_repair_does_not_delete_unknown_argument`
- `test_repair_does_not_rename_unknown_argument`
- `test_repair_does_not_inject_business_parameter`
- `test_repaired_payload_still_requires_extra_forbid`

### TestMetricsAggregation (8 tests)
- `test_format_parse_fail_not_counted_as_unknown_action`
- `test_unknown_action_type_correctly_counted`
- `test_schema_failure_correctly_counted`
- `test_finish_without_tests_in_failed_trajectory`
- `test_numerator_denominator_consistent_with_rate`
- `test_zero_denominator_well_defined`
- `test_repeated_loop_independently_counted`
- `test_max_step_hit_independently_counted`

### TestProtocolConsistency (5 tests)
- `test_json_unknown_field_hard_fail`
- `test_tag_unknown_field_hard_fail`
- `test_dsl_unknown_field_hard_fail`
- `test_three_protocols_consistent_on_illegal_action`
- `test_parser_no_silent_noop`

## 10. Full Test Results

### Targeted tests

```
py -3.11 -m pytest tests/test_protocol_base.py tests/test_protocol_json.py \
  tests/test_protocol_tag.py tests/test_protocol_dsl.py \
  tests/test_protocol_ablation.py tests/test_agent_model_provider.py \
  tests/test_agent_model_provider_protocol.py -v
```

Result: **67 passed**, 0 failed, 0 skipped, 5.71s

### Relevant regression suite

```
py -3.11 -m pytest tests/test_agent_actions.py tests/test_agent_evaluator.py \
  tests/test_agent_model_provider.py tests/test_agent_model_provider_protocol.py \
  tests/test_protocol_base.py tests/test_protocol_json.py \
  tests/test_protocol_tag.py tests/test_protocol_dsl.py \
  tests/test_protocol_ablation.py tests/test_trust_repair.py -v
```

Result: **138 passed**, 0 failed, 1 warning, 27.05s

### Full non-GPU suite

```
py -3.11 -m pytest tests/ -v -p no:warnings -m "not gpu"
```

Result: **1415 passed**, 0 failed, 1 skipped, 2 deselected, 585.17s

## 11. GPU Ablation Configuration

| Parameter | Value |
|-----------|-------|
| Experiment commit SHA | `d034fdeb4904d1775f21d20ec11c06e85f70e0cf` (d034fde) |
| Working tree state | Issue #32 fixes applied (uncommitted at run time) |
| Source file SHAs | Recorded in `baseline-lock.json` — match working tree |
| Task manifest SHA256 | `bdcc2eaa268b8965ff764ac6c710c97ba90298e11b7c05d0133cdb7103f692bc` |
| Model path | `models/Qwen3-0.6B` |
| Adapter path (base) | `null` |
| Adapter path (repair-lora) | `adapters/p3/repair-limited` |
| Temperature | 0.0 |
| do_sample | False |
| max_new_tokens | 128 |
| dtype | float16 |
| max_steps | 12 |
| Protocols | json, tag, dsl |
| Configs | base, repair-lora |
| Tasks | 40 (task_001 .. task_040) |
| Total runs | 240 |
| Python | 3.11.7 |
| Torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Transformers | 5.12.1 |

**Note on experiment_commit_sha:** The ablation ran with Issue #32 fixes
applied in the working tree but not yet committed. The
`experiment_commit_sha` field records `d034fde` (the HEAD at run time).
The `source_file_shas` field in `baseline-lock.json` records the SHA256
of each source file at run time and matches the current working tree
exactly — this is the authoritative evidence of the actual code used.
After this commit lands, the commit SHA will contain the exact code that
was used in the experiment.

## 12. 240-Run Results

### v2 Metrics (strict schema validation)

| Protocol | Config | schema_valid | arguments_valid | task_success | max_steps_hit | unknown_actions | finish_no_tests | crashes |
|----------|--------|-------------|-----------------|--------------|---------------|-----------------|-----------------|---------|
| json | base | 0/480 = 0.00% | 0/480 = 0.00% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| json | repair-lora | 0/480 = 0.00% | 0/480 = 0.00% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| tag | base | 11/480 = 2.29% | 11/480 = 2.29% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| tag | repair-lora | 8/480 = 1.67% | 8/480 = 1.67% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| dsl | base | 0/480 = 0.00% | 0/480 = 0.00% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| dsl | repair-lora | 29/480 = 6.04% | 29/480 = 6.04% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |

### Step-level metrics

| Protocol | Config | total_steps | format_parse | safety_valid | action_type_valid |
|----------|--------|-------------|--------------|--------------|-------------------|
| json | base | 480 | 12/480 = 2.50% | 12/480 = 2.50% | 12/480 = 2.50% |
| json | repair-lora | 480 | 480/480 = 100.00% | 480/480 = 100.00% | 480/480 = 100.00% |
| tag | base | 480 | 11/480 = 2.29% | 11/480 = 2.29% | 11/480 = 2.29% |
| tag | repair-lora | 480 | 30/480 = 6.25% | 30/480 = 6.25% | 30/480 = 6.25% |
| dsl | base | 480 | 0/480 = 0.00% | 0/480 = 0.00% | 0/480 = 0.00% |
| dsl | repair-lora | 480 | 226/480 = 47.08% | 154/480 = 32.08% | 226/480 = 47.08% |

### Failure taxonomy

| Failure Class | Count |
|---------------|-------|
| FORMAT_PARSE_FAIL | 2193 |
| SCHEMA_VALIDATION_FAIL | 639 |
| UNKNOWN_ACTION_TYPE | 0 |
| FORBIDDEN_ACTION | 0 |
| INVALID_PATH | 0 |
| EMPTY_OR_USELESS_ACTION | 0 |
| MODEL_REFUSAL_OR_CHATTER | 0 |
| REPEATED_ACTION_LOOP | 240 |

### Verdict

**FIX_PROMPT_FIRST**

Rationale: All 6 combinations have `schema_valid_rate < 30%`. No protocol
is sufficiently schema-stable under the current model and prompt to
justify proceeding without prompt fixes. The `json + repair-lora`
combination has `format_parse_rate = 100%` but `schema_valid_rate = 0%`,
meaning the model consistently produces parseable JSON with wrong
argument fields — a prompt-engineering problem, not a protocol problem.

## 13. Artifact SHAs

All artifacts in `reports/p4/protocol-ablation-v2/`:

| Artifact | SHA256 | Size | Rows |
|----------|--------|------|------|
| baseline-lock.json | `4e1b6850...e20d` | 2511 | — |
| comparison-matrix.json | `cdb381cc...87bc` | 5831 | — |
| comparison-report.md | `ff33188d...de5a` | 2323 | — |
| failure-taxonomy.json | `808c6f4b...bafb` | 243 | — |
| verdict.json | `7d092c6c...959d` | 37 | — |
| trajectories/dsl-base.jsonl | `a7741305...423e` | 437808 | 40 |
| trajectories/dsl-repair-lora.jsonl | `52b326cb...ebb8` | 416402 | 40 |
| trajectories/json-base.jsonl | `9c6fe0fb...39a9` | 455720 | 40 |
| trajectories/json-repair-lora.jsonl | `a109aa63...cd7a` | 440272 | 40 |
| trajectories/tag-base.jsonl | `5e3c0a9f...d93b` | 410733 | 40 |
| trajectories/tag-repair-lora.jsonl | `a5fb9858...ef34` | 392337 | 40 |

Total trajectory rows: 240 (40 × 6). All row counts verified.

## 14. Old vs New Metric Comparison

### schema_valid_rate

| Protocol | Config | v1 (untrusted) | v2 (trusted) | Delta |
|----------|--------|----------------|--------------|-------|
| json | base | 0.00% | 0.00% | 0 |
| json | repair-lora | **96.25%** | **0.00%** | **-96.25pp** |
| tag | base | 2.29% | 2.29% | 0 |
| tag | repair-lora | 1.67% | 1.67% | 0 |
| dsl | base | 0.00% | 0.00% | 0 |
| dsl | repair-lora | 8.12% | 6.04% | -2.08pp |

The `-96.25pp` drop for `json + repair-lora` confirms the trust issue:
v1's 96.25% was entirely an artifact of Pydantic `extra="ignore"`
silently dropping unknown argument fields.

### unknown_action_count

| Protocol | Config | v1 (inflated) | v2 (correct) |
|----------|--------|---------------|--------------|
| json | base | 468 | 0 |
| json | repair-lora | 18 | 0 |
| tag | base | 469 | 0 |
| tag | repair-lora | 450 | 0 |
| dsl | base | 480 | 0 |
| dsl | repair-lora | 325 | 0 |

v1 counted format-parse failures as unknown actions. v2 correctly
separates `FORMAT_PARSE_FAIL` (2193) from `UNKNOWN_ACTION_TYPE` (0).

### Verdict

| Version | Verdict | Basis |
|---------|---------|-------|
| v1 | `KEEP_ACTION_JSON` | Fake 96.25% schema_valid_rate |
| v2 | `FIX_PROMPT_FIRST` | All combinations < 30% schema_valid_rate |

## 15. Decision Document Correction

`docs/decisions/2026-07-11-p4-2-training-protocol.md` rewritten.

### Before (v1, untrusted)

- Single decision conflating protocol choice and training initialization
- Selected Option B (continual from P3-limited) based on zero-shot
  `schema_valid_rate`
- Status: PROPOSED (but implicitly leaning Option B)

### After (v2, corrected)

- **Decision 1 — Action Protocol**: Determined by P4.1b v2 verdict
  (`FIX_PROMPT_FIRST`). No protocol is selected yet; prompt must be
  fixed first.
- **Decision 2 — Agent SFT Initialization**: DEFERRED to two-arm pilot.
  Both arms preserved:
  - Arm A: Base Qwen3-0.6B → Agent SFT
  - Arm B: P3 Repair-Limited LoRA → Agent SFT
- Status: `PROPOSED / DEFERRED_TO_TWO_ARM_PILOT`
- Zero-shot ablation results must NOT be used to select training
  initialization.

### Prohibited framings (explicitly listed in decision doc)

- "Repair-LoRA has been proven to be the best Agent SFT training starting point"
- "Base has been proven unsuitable for Agent SFT"

## 16. Risks and Limitations

1. **experiment_commit_sha vs working tree**: The 240-run executed
   against uncommitted working-tree changes. The
   `experiment_commit_sha` field records `d034fde` (pre-fix HEAD), but
   `source_file_shas` in `baseline-lock.json` match the fixed working
   tree exactly. After this commit, the commit SHA will contain the
   exact code used. Future ablations should commit first, then run.

2. **All task_success_rate = 0%**: No protocol achieved any successful
   task completion. This is expected for zero-shot (no Agent SFT) but
   means the ablation cannot inform task-level performance — only
   schema/format stability.

3. **All max_steps_hit_rate = 100%**: Every trajectory hit the 12-step
   limit without finishing. This indicates the model loops or stalls
   without making progress, consistent with the `REPEATED_ACTION_LOOP`
   count of 240.

4. **BF16 not used**: The ablation ran in float16 (not BF16) because
   the ablation script uses the default dtype. BF16 is supported on
   this GPU but was not configured for the ablation. This is consistent
   with v1.

5. **Prompt fix needed**: The v2 verdict `FIX_PROMPT_FIRST` means the
   current prompt does not adequately teach the model the correct
   argument fields for each action. A prompt engineering iteration is
   required before any protocol can be selected.

6. **Decision 2 still open**: The two-arm pilot (Arm A vs Arm B) has
   not been authorized. This report does not authorize any training.

7. **v1 results preserved but untrusted**: v1 artifacts remain in
   `reports/p4/protocol-ablation/` with `SUPERSEDED.md` marker. They
   must not be used for any decision. They are kept only as experimental
   history.

## 17. Git Delivery

### Files staged for Issue #32 commits

**Code fixes:**
- `src/agent_actions.py`
- `src/protocols/base.py`
- `src/protocols/json_protocol.py`
- `src/protocols/tag_protocol.py`
- `src/protocols/dsl_protocol.py`
- `src/agent_model_provider.py`
- `scripts/run_protocol_ablation.py`

**Tests:**
- `tests/test_protocol_ablation.py` (modified)
- `tests/test_trust_repair.py` (new)

**Experiments:**
- `reports/p4/protocol-ablation-v2/` (new, 11 files)
- `reports/p4/protocol-ablation/SUPERSEDED.md` (new)

**Docs:**
- `docs/decisions/2026-07-11-p4-2-training-protocol.md` (rewritten)
- `reports/2026-07-12/p4-1b-trust-repair-report.md` (new, this file)

### Files NOT staged (unrelated changes preserved)

- `AGENTS.md`
- `data/p3-limited/*` (timestamp drift)
- `reports/p4/p4-1-readiness.md`
- `reports/p4/p4-1-test-evidence.json`
- `docs/agent-protocols/*`
- `docs/workflow/*`
- `test-results.xml`

### Commit plan

```
fix(p4.1b): forbid unknown action fields and preserve semantics
fix(p4.1b): repair protocol diagnostics and metric aggregation
test(p4.1b): add strict-schema and aggregation regressions
exp(p4.1b): rerun trusted protocol ablation v2
docs(p4.1b): record trust repair and correct training decision
```

### Verification

- `git diff --stat` reviewed: only Issue #32 files staged
- `git status` reviewed: unrelated changes preserved in working tree
- No secrets, no weights, no .env files staged

## 18. Checklist Self-Audit

### A. Repository / Scope

- [x] PASS — Read AGENTS.md
- [x] PASS — Read Issue #32
- [x] PASS — Read PR #30 review
- [x] PASS — Preflight recorded (branch, HEAD, unrelated changes)
- [x] PASS — Current branch correct (`feat/p4-1b-protocol-ablation`)
- [x] PASS — Unrelated working-tree changes preserved
- [x] PASS — No unrelated files staged
- [x] PASS — No training started
- [x] PASS — PR #30 not merged

### B. Strict Schema

- [x] PASS — `ActionBase` forbids extra fields
- [x] PASS — `SafetyFlags` forbids extra fields
- [x] PASS — All Args models forbid extra fields
- [x] PASS — `list_files.arguments.path` rejected
- [x] PASS — `read_file.arguments.pattern` rejected
- [x] PASS — `inspect_task` unknown argument rejected
- [x] PASS — Unknown top-level field rejected
- [x] PASS — Unknown safety field rejected
- [x] PASS — Valid action still passes
- [x] PASS — Path safety not degraded
- [x] PASS — Forbidden safety not degraded

### C. Semantic Preservation

- [x] PASS — JSON repair does not delete unknown fields
- [x] PASS — JSON repair does not rename fields
- [x] PASS — JSON repair does not change action_type
- [x] PASS — Tag parser does not change action semantics
- [x] PASS — DSL parser does not change action semantics
- [x] PASS — No silent coercion
- [x] PASS — No silent no-op

### D. Diagnostics

- [x] PASS — `format_parse_ok` computed independently
- [x] PASS — `action_type_valid` computed independently
- [x] PASS — `arguments_valid` computed independently
- [x] PASS — `safety_valid` computed independently
- [x] PASS — `schema_valid` based on complete raw payload
- [x] PASS — `repair_attempted` accurate
- [x] PASS — `repair_success` accurate
- [x] PASS — `failure_class` accurate

### E. Aggregation

- [x] PASS — Format fail not counted as unknown action
- [x] PASS — Unknown action only counts true unknown type
- [x] PASS — `finish_without_tests` from evaluator explicit evidence
- [x] PASS — Failed `finish_without_tests` still counted
- [x] PASS — Numerator recorded
- [x] PASS — Denominator recorded
- [x] PASS — Rate consistent with numerator/denominator
- [x] PASS — Repeated loop independently counted (240)
- [x] PASS — Max-step hit independently counted (240)
- [x] PASS — Runtime crash independently counted (0)

### F. Tests

- [x] PASS — Strict schema tests all pass (11)
- [x] PASS — Repair semantic tests all pass (7)
- [x] PASS — Aggregation tests all pass (8)
- [x] PASS — Protocol tests all pass (5)
- [x] PASS — Targeted tests all pass (67)
- [x] PASS — Relevant regression all pass (138)
- [x] PASS — Full non-GPU suite all pass (1415)
- [x] PASS — No unexplained skip (1 skip is pre-existing GPU mark)
- [x] PASS — No test silently removed

### G. Historical Evidence

- [x] PASS — Old `protocol-ablation/` results not deleted
- [x] PASS — Old results marked SUPERSEDED
- [x] PASS — New results use versioned directory (`protocol-ablation-v2/`)
- [x] PASS — Failed trajectories not deleted
- [x] PASS — Rejected/invalid results still traceable

### H. Frozen 240-Run

- [x] PASS — 3 protocols complete (json, tag, dsl)
- [x] PASS — 2 configs complete (base, repair-lora)
- [x] PASS — 40 tasks complete (task_001..task_040)
- [x] PASS — Total 240 trajectories
- [x] PASS — Task IDs match manifest
- [x] PASS — Manifest SHA recorded (`bdcc2eaa...92bc`)
- [x] PASS — Model config recorded
- [x] PASS — Adapter path recorded (base=null, repair-lora=adapters/p3/repair-limited)
- [x] PASS — Generation config recorded (temp=0.0, do_sample=False, max_new_tokens=128)
- [x] PASS — Max steps recorded (12)
- [x] PASS — Experiment commit recorded (`d034fde` + working tree, source_file_shas authoritative)
- [x] PASS — 0 results silently excluded
- [x] PASS — Runtime crash explicitly counted (0)

### I. Artifacts

- [x] PASS — `baseline-lock.json` generated
- [x] PASS — `comparison-matrix.json` generated
- [x] PASS — `comparison-report.md` generated
- [x] PASS — `failure-taxonomy.json` generated
- [x] PASS — `verdict.json` generated
- [x] PASS — `artifact-manifest.json` generated
- [x] PASS — Trajectories JSONL generated (6 files)
- [x] PASS — All artifact SHAs recorded
- [x] PASS — Row counts verified (40 × 6 = 240)

### J. Decision Correction

- [x] PASS — P4.1b only selects protocol
- [x] PASS — Zero-shot results not used to select training initialization
- [x] PASS — Base → Agent SFT candidate preserved (Arm A)
- [x] PASS — Repair-LoRA → Agent SFT candidate preserved (Arm B)
- [x] PASS — Two-arm pilot explicitly recorded
- [x] PASS — Option B not marked approved
- [x] PASS — No training authorized

### K. Git / Delivery

- [x] PASS — `git diff` manually reviewed
- [x] PASS — Commit messages clear
- [x] PASS — No secrets
- [x] PASS — No weights
- [x] PASS — No .env
- [ ] PENDING — Branch pushed (will push after commit)
- [ ] PENDING — Commit SHA recorded (will record after commit)
- [ ] PENDING — PR #30 updated (will update after push)
- [ ] PENDING — Issue #32 updated (will update after push)
- [ ] PENDING — CI run (will run after push)
- [ ] PENDING — CI green (will verify after CI run)
- [ ] PENDING — Final Gatekeeper Review requested (will request after CI green)

### Checklist Summary

- Passed: 95
- Failed: 0
- Pending (post-commit): 7
- Not applicable: 0

## 19. Verdict

**GO_FOR_FINAL_GATEKEEPER_REVIEW**

Rationale:
- All Tasks A-G implemented and unit-tested
- Full non-GPU suite passes (1415 tests)
- 240-run ablation re-executed under strict schema validation
- v2 artifacts generated with SHA256 manifest
- v1 results preserved and marked SUPERSEDED
- Training decision document corrected (Decision 1 vs Decision 2 split)
- No training started, no PR merged, no adapter overwritten
- Option B not approved

This verdict does NOT authorize PR #30 merge. Final Gatekeeper Review
is required.

---

## 15 Self-Check Questions

1. **Any path where model outputs illegal fields but system continues?**
   No. All 13 Pydantic models use `extra="forbid"`. Unknown fields cause
   `ValidationError` and `schema_valid=False`.

2. **Any parser that deletes unknown fields and continues?**
   No. Tag protocol removed its `has_unknown_key` filter. All protocols
   pass unknown fields to Pydantic which rejects them.

3. **Any metric that double-counts the same failure?**
   No. `FORMAT_PARSE_FAIL` and `UNKNOWN_ACTION_TYPE` are mutually
   exclusive. `unknown_action_count` uses `failure_class==
   "UNKNOWN_ACTION_TYPE"` only.

4. **Any failed trajectory excluded from failure count due to
   `success=False`?**
   No. `finish_without_tests_count` uses evaluator explicit evidence,
   not `success and not tests_passed`.

5. **Any percentage without numerator/denominator?**
   No. All rates in `comparison-report.md` show
   `numerator / denominator = rate`.

6. **Covered the real `list_files.arguments.path` regression case?**
   Yes. `test_list_files_arguments_path_hard_fail` in
   `tests/test_trust_repair.py`. Also confirmed in v2 ablation:
   `json + repair-lora` schema_valid_rate dropped from 96.25% to 0.00%.

7. **Re-ran full 240 runs, not reused old data?**
   Yes. All 240 trajectories regenerated in
   `reports/p4/protocol-ablation-v2/`. v1 trajectories in
   `reports/p4/protocol-ablation/` untouched.

8. **Preserved old untrusted experimental results?**
   Yes. v1 results in `reports/p4/protocol-ablation/` preserved with
   `SUPERSEDED.md` marker.

9. **Still using zero-shot schema_valid_rate to select training
   initialization?**
   No. Decision document corrected. Decision 2 (training init) deferred
   to two-arm pilot.

10. **Accidentally started any training?**
    No. No training executed. No model weights modified.

11. **Accidentally overwrote P3 adapter?**
    No. `adapters/p3/repair-limited` is read-only. Ablation loaded it
    in inference mode only.

12. **Uncommitted code that affects experimental results?**
    Yes — this is the experiment_commit_sha risk documented in Section
    16. The 240-run used uncommitted working-tree fixes. The
    `source_file_shas` in `baseline-lock.json` are the authoritative
    evidence. This commit will land the exact code that was used.

13. **Does the commit SHA in the report match the 240-run commit?**
    The `experiment_commit_sha` is `d034fde` (pre-fix HEAD). The actual
    code used is `d034fde` + working-tree fixes, proven by
    `source_file_shas`. After this commit, the commit SHA will contain
    the exact code. This is documented as Risk 1 in Section 16.

14. **Does GitHub CI test the final pushed HEAD?**
    Yes. CI will run on the pushed commit, which contains all Issue #32
    fixes.

15. **Is the conclusion strictly limited to protocol performance, not
    model final capability?**
    Yes. The verdict `FIX_PROMPT_FIRST` is about protocol/prompt
    schema stability under zero-shot conditions. It does not claim
    anything about post-SFT model capability.
