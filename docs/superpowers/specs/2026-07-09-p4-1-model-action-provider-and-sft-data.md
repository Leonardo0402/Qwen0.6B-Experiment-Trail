# P4.1 ModelActionProvider Smoke + Agent SFT Data Builder — Design

- **Issue**: [#19 — P4.1: ModelActionProvider Smoke + Agent SFT Data Builder](https://github.com/Leonardo0402/Qwen0.6B-Experiment-Trail/issues/19)
- **Branch**: `feat/p4-1-model-action-provider` (off `main` @ merge commit `7ccd06c` of PR #18)
- **Delivery**: single PR covering phases A–H, `Closes #19` in commit message
- **Spec date**: 2026-07-09
- **Status**: design pending user approval

## 1. Purpose and Scope

P4.0 (PR #18, merged as `7ccd06c`) delivered the agentic coder **foundation**:
11-action schema, safe tool layer, trajectory format, 40-task micro suite, 40
scripted teacher trajectories, and a replay evaluator with an 11-gate readiness
verdict `GO_FOR_P4_AGENT_SFT_DATA`.

P4.1 builds on that foundation to answer one question: **can Qwen3-0.6B
stably emit valid Action JSON as an agent policy?** If yes, we collect a
**replayable** Agent SFT dataset. The endpoint is `GO_FOR_P4_AGENT_SFT` — the
authorization to *consider* training, not the authorization to train.

P4.1 also closes four P4.0 residual gaps the readiness report documented as
known limitations: the evaluator `TEST_PASS` trust gap, the missing
`search_text`/`rollback_patch` dispatch, `inspect_error` returning empty
content on pytest failures, and the narrow `WRONG_PATCH`-only corruption test
coverage.

### Hard non-goals (Issue #19, user-confirmed 2026-07-09)

- ❌ No SFT / DPO / RL training of any kind
- ❌ No model replacement — base Qwen3-0.6B + existing P3 Repair-Limited LoRA only
- ❌ No external datasets
- ❌ No GPU training runs (GPU is used only for inference smoke, greedy decoding)
- ❌ No migration to Qwen3-1.7B
- ❌ No modification of P4.0 frozen artifacts (micro-tasks-v0, scripted.jsonl) — P4.1 reads them, does not rewrite them
- ❌ No capability claim from loss alone

### Confirmed design decisions (user-specified 2026-07-09)

| # | Decision | Choice |
|---|---|---|
| 1 | TEST_PASS trust gap fix | Replay result is authoritative; `finish.tests_passed` is a model self-claim only. Mismatch recorded as `finish_claim_mismatch=True`. |
| 2 | Evaluator dispatch coverage | Explicit allowlist of all 11 action types; unknown action type → hard fail (raise), never silent no-op. |
| 3 | Model smoke success bar | Smoke does NOT require high task success. Minimum: `forbidden_action_count=0`, `runtime_crash_count=0`, at least one schema-valid action from base or LoRA, all failures have structured diagnostics. |
| 4 | SFT trajectory `source` labeling | Every trajectory carries a `source` field from a fixed enum: `scripted_variant`, `teacher_model`, `corrupted_recovered`, `failed_patch_recovery`, `model_self_run_success`, `model_self_run_failure`. Failure trajectories are never mixed into success demonstrations. |
| 5 | Success criterion by trajectory type | `TEST_PASS` → latest replay `run_tests` passed; `IDENTIFY_BUG` → `identification_verified` + evidence; `PATCH_APPLIED` → ≥1 successful patch. SFT "success demonstration" pool prefers `TEST_PASS` trajectories. |
| 6 | SFT dataset split | `train` / `validation` / `heldout-agent-eval`. Split by workspace/task family. No workspace variant appears in both `train` and `heldout` unless explicitly tagged `curriculum_replay`. |
| 7 | P4.1 endpoint | `GO_FOR_P4_AGENT_SFT` — authorizes *considering* training, not training itself. |
| 8 | inspect_error fix | Return stdout+stderr (capped at 8 KB) so pytest failure tracebacks (written to stdout) are surfaced. |

### Security note (carried from P4.0)

P4.0's supply-chain rule stands: no file from Issue/PR comments is downloaded
or applied. P4.1 adds no new dependencies beyond what P4.0/P3 already use
(torch, transformers, peft — already in `requirements.txt` for P3 inference).

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                  P4.1 Agent Policy Smoke + SFT Data              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase A: P4.0 baseline lock (record SHAs)                       │
│                                                                  │
│  Phase B: Evaluator trust-gap closure                            │
│    src/agent_evaluator.py                                        │
│      - TEST_PASS: success = (passed_tests > 0), not finish claim │
│      - finish_claim_mismatch flag in EvalResult                  │
│      - explicit 11-action allowlist + unknown→hard fail          │
│      - dispatch search_text + rollback_patch                     │
│                                                                  │
│  Phase C: inspect_error improvement                              │
│    src/agent_tools.py                                            │
│      - tool_inspect_error returns stdout+stderr (capped 8KB)     │
│                                                                  │
│  Phase D: Corruption test expansion                              │
│    tests/test_agent_evaluator.py                                 │
│      - all 5 CorruptionType values covered                       │
│                                                                  │
│  Phase E: ModelActionProvider                                    │
│    src/agent_model_provider.py (new)                             │
│      - prompt builder (AgentState + Observation + Memory)        │
│      - model.generate (greedy, temperature=0)                    │
│      - JSON extraction + Pydantic validation                     │
│      - structured diagnostics per step                           │
│      - JSON repair layer (format-only, never semantic)           │
│                                                                  │
│  Phase F: Model trajectory collection                            │
│    scripts/collect_model_trajectories.py (new)                   │
│      - run ModelActionProvider on 40 micro-tasks                 │
│      - two configs: base + Repair-Limited LoRA                   │
│      - write trajectories + diagnostics JSONL                    │
│                                                                  │
│  Phase G: SFT dataset builder                                    │
│    scripts/build_agent_sft_dataset.py (new)                      │
│      - aggregate 1000+ trajectories from 6 sources               │
│      - label source, success, criterion                          │
│      - split train/validation/heldout-agent-eval                 │
│      - replay-verify every trajectory                            │
│                                                                  │
│  Phase H: P4.1 readiness verifier                                │
│    scripts/verify_p4_1_readiness.py (new)                        │
│      - 10-gate verifier → GO_FOR_P4_AGENT_SFT                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 3. P4.0 Baseline Lock (Phase A)

Before any P4.1 work, record the P4.0 "before" state:

```json
{
  "p4_0_merge_commit": "7ccd06c4d479b269f7708a6a430b9965af5f17e6",
  "micro_tasks_manifest_sha256": "<sha of data/p4-agent/micro-tasks-v0/manifest.json>",
  "scripted_trajectories_sha256": "<sha of data/p4-agent/trajectories-v0/scripted.jsonl>",
  "agent_evaluator_sha256": "<sha of src/agent_evaluator.py>",
  "readiness_report_sha256": "<sha of reports/p4/p4-agent-foundation-readiness.md>",
  "p4_0_verdict": "GO_FOR_P4_AGENT_SFT_DATA",
  "p4_0_test_count": 81
}
```

Written to `reports/p4/p4-0-baseline-lock.json` by
`scripts/lock_p4_0_baseline.py`. Idempotent: re-running produces the same JSON.

## 4. Evaluator Trust-Gap Closure (Phase B)

### 4.1 TEST_PASS: replay is authoritative

**Current (P4.0):** `success = fa.tests_passed` (the finish declaration).

**P4.1:** `success = (passed_tests > 0)` for `TEST_PASS` criterion. The
`finish.tests_passed` field becomes a **model self-claim**, recorded but not
trusted.

New `EvalResult` field:
```python
class EvalResult(BaseModel):
    ...
    finish_claim_mismatch: bool = False
    # True iff finish.tests_passed disagrees with actual replay passed_tests
```

Mismatch logic (applies to `TEST_PASS` criterion only):
| `finish.tests_passed` | replay `passed_tests` | `success` | `finish_claim_mismatch` |
|---|---|---|---|
| True | > 0 | True | False |
| True | 0 | False | True (claimed pass, actually failed) |
| False | > 0 | True | True (claimed fail, actually passed) |
| False | 0 | False | False |

Key: **real replay result is authoritative**. `finish.tests_passed` is the
model's self-report; it is recorded for diagnostics, never used as the success
oracle.

### 4.2 Explicit 11-action allowlist + unknown hard fail

**Current (P4.0):** the evaluator's `run()` uses an `if/elif` chain. Action
types not in the chain (`search_text`, `rollback_patch`) fall through
silently — a no-op with no error, no metric.

**P4.1:** replace the fall-through with an explicit allowlist + `else: raise`:

```python
_ALLOWED_ACTION_TYPES = frozenset({
    "list_files", "read_file", "search_text", "inspect_task",
    "propose_patch", "apply_patch", "rollback_patch", "run_tests",
    "inspect_error", "write_memory", "finish",
})

# At end of dispatch chain:
else:
    raise P4ForbiddenActionError(
        f"unknown action type (not in 11-action allowlist): {at}"
    )
```

An unknown action type is a **hard fail** (raises `P4ForbiddenActionError`),
recorded in `errors` and counted in `forbidden_action_count`. Never silent.

### 4.3 Dispatch `search_text` and `rollback_patch`

Add the two missing dispatch branches:

```python
elif at == "search_text":
    obs = tool_search_text(self._ws, action.arguments.pattern)
    total_tools += 1
elif at == "rollback_patch":
    obs = tool_rollback_patch(self._ws, action.arguments.action_id)
    total_tools += 1
```

Both produce real tool calls and affect subsequent state (search results are
available to the next action; rollback restores the patched file).

### 4.4 Tests (Phase B)

- `test_test_pass_success_uses_replay_not_claim`: finish claims pass, replay
  has 0 passed → `success=False`, `finish_claim_mismatch=True`
- `test_test_pass_mismatch_claimed_fail_actual_pass`: finish claims fail,
  replay has >0 passed → `success=True`, `finish_claim_mismatch=True`
- `test_test_pass_no_mismatch_when_claim_matches_replay`: both agree →
  `finish_claim_mismatch=False`
- `test_unknown_action_type_raises`: provider emits action with
  `action_type="shell_exec"` → evaluator raises `P4ForbiddenActionError`,
  records in errors, counts in `forbidden_action_count`
- `test_search_text_dispatched`: trajectory with `search_text` action →
  tool is called, `total_tools` incremented
- `test_rollback_patch_dispatched`: trajectory with `rollback_patch` →
  tool is called, file restored

## 5. inspect_error Improvement (Phase C)

**Current (P4.0):** `tool_inspect_error` returns
`content=last_test_observation.stderr`. But pytest writes failure tracebacks
to **stdout**, so `inspect_error` on a failed test returns empty content.

**P4.1:** return stdout+stderr, capped at 8 KB:

```python
def tool_inspect_error(error_source, last_test_observation, last_patch_observation):
    if error_source == "last_test":
        if last_test_observation is None:
            raise ValueError("no prior run_tests observation")
        raw = last_test_observation.stdout + "\n" + last_test_observation.stderr
        capped = raw[:8192]
        return ErrorObservation(source="last_test", content=capped)
    ...
```

### 5.1 Tests (Phase C)

- `test_inspect_error_returns_stdout_on_test_failure`: failed `run_tests`
  (traceback in stdout) → `inspect_error` returns non-empty content containing
  the failure excerpt
- `test_inspect_error_caps_at_8kb`: observation with >8 KB stdout+stderr →
  content truncated to exactly 8192 chars

## 6. Corruption Test Expansion (Phase D)

P4.0 only tested `WRONG_PATCH`. P4.1 covers all 5 `CorruptionType` values:

| CorruptionType | Test assertion |
|---|---|
| `WRONG_ACTION_TYPE` | evaluator records invalid action (re-validation fails), `valid_actions` excludes it |
| `INVALID_PATH` | evaluator re-validation raises `PathValidationError`, recorded in errors |
| `WRONG_PATCH` | (existing) patch fails, `tool_errors` incremented |
| `SKIP_TESTS_BEFORE_FINISH` | `finish_without_tests_count` incremented (uses the P4.0-fix `ran_tests` flag) |
| `EXCEED_MAX_STEPS` | `max_step_exceeded_count` incremented, `max_steps_hit=True` |

### 6.1 Tests (Phase D)

5 new tests, one per corruption type. Each uses `CorruptedActionProvider`
with a scripted trajectory + a `Corruption` at a relevant step index.

## 7. ModelActionProvider (Phase E)

### 7.1 Design

`ModelActionProvider` is the first P4.1 component that loads Qwen3-0.6B. It
**does not call tools** — it emits one `Action` JSON per step, and the
evaluator dispatches.

```
AgentState + Observation + AgentMemory
  → prompt builder (chat template)
  → model.generate (greedy, temperature=0, do_sample=false, max_new_tokens=512)
  → raw_output
  → JSON extraction (find first {...} block, strip markdown fences)
  → Pydantic validate (Action union)
    → valid: return Action
    → invalid: attempt format-only repair, re-validate
      → still invalid: record diagnostics, return SentinelAction(action_invalid)
```

### 7.2 Structured diagnostics per step

Each step records:
```python
class ModelStepDiagnostics(BaseModel):
    raw_output: str
    json_parse_ok: bool
    schema_valid: bool
    safety_valid: bool
    action_type_valid: bool
    arguments_valid: bool
    repair_attempted: bool
    repair_success: bool
    latency_ms: int
```

### 7.3 JSON repair layer (format-only)

The repair layer may ONLY fix format issues:
- Strip markdown fences (` ```json ... ``` `)
- Remove trailing commas
- Balance unbalanced braces (best-effort)
- Unescape escaped quotes

It must NOT:
- Choose an action type for the model
- Fill in missing arguments
- Alter action semantics

A repaired-but-semantically-wrong action is still `action_invalid`. The
repair layer is a format fixer, not a decision maker.

### 7.4 Two smoke configurations

| Config | Base model | Adapter | Purpose |
|---|---|---|---|
| `base` | `models/Qwen3-0.6B` | none | Baseline action validity |
| `repair-lora` | `models/Qwen3-0.6B` | `adapters/p3/repair-limited` | Does repair-biased LoRA help action validity? |

Both run on the 40 micro-tasks. Greedy decoding (`temperature=0`,
`do_sample=false`) for reproducibility.

### 7.5 Smoke success bar (user-specified)

The smoke does NOT require high task success. Minimum bar:

| Metric | Minimum |
|---|---|
| `model_load_ok` | True (both configs) |
| `adapter_load_ok` | True (repair-lora config) |
| `generation_ok` | True (model produces output for ≥1 step) |
| `forbidden_action_count` | 0 (no action with `network_required` or `reads_sensitive_path`) |
| `runtime_crash_count` | 0 (no unhandled exceptions) |
| `schema_valid_rate` | > 0 (at least one schema-valid action from base OR repair-lora) |
| structured diagnostics | all failures have a `ModelStepDiagnostics` record |

`task_success_rate` and `patch_success_rate` are **reported but not gated**.

### 7.6 GPU constraint (4GB VRAM)

Per AGENTS.md: Qwen3-0.6B in FP16 + LoRA fits in 4GB. Inference (not
training) uses `use_cache=True`. Batch size 1. `max_new_tokens=512`.

### 7.7 Tests (Phase E)

- `test_model_provider_prompt_builds`: given AgentState + Observation + Memory,
  prompt builder produces a non-empty string containing the task description
- `test_model_provider_extracts_valid_json`: mock `model.generate` to return
  a valid Action JSON → provider returns a valid Action
- `test_model_provider_records_diagnostics_on_invalid_json`: mock
  `model.generate` to return garbage → provider returns SentinelAction,
  diagnostics record `json_parse_ok=False`
- `test_model_provider_repair_strips_markdown_fences`: mock returns
  ` ```json {...} ``` ` → repair succeeds, valid Action returned
- `test_model_provider_repair_does_not_choose_action`: mock returns
  `{"action_type": "???"}` (invalid type) → repair does NOT substitute a
  valid type; `action_type_valid=False`
- `test_model_provider_smoke_base` (marked `@pytest.mark.gpu`): loads base
  model, runs 1 micro-task, asserts no crash + ≥1 schema-valid action
- `test_model_provider_smoke_repair_lora` (marked `@pytest.mark.gpu`): loads
  repair-lora, same assertions

GPU tests are skipped in CI (`-m "not gpu"`); run manually on the RTX 3050.

## 8. Model Trajectory Collection (Phase F)

`scripts/collect_model_trajectories.py`:

- Iterates the 40 micro-tasks
- For each task, runs `ModelActionProvider` through `AgentEvaluator`
- Two configs: `base`, `repair-lora`
- `max_steps=12` per task
- Writes:
  - `data/p4-agent/trajectories-v1/model-base.jsonl` (≤40 trajectories)
  - `data/p4-agent/trajectories-v1/model-repair-lora.jsonl` (≤40 trajectories)
  - `reports/p4/model-trajectory-collection-report.json` (metrics + diagnostics summary)

Each trajectory in the JSONL carries:
- `trajectory_id`, `task_id`, `config` (`base`/`repair-lora`)
- `source`: `model_self_run_success` or `model_self_run_failure`
- `success`, `finish_claim_mismatch`
- `metrics` (8 standard + `action_validity_rate`, `json_parse_rate`)
- `step_diagnostics`: list of `ModelStepDiagnostics`

### 8.1 Report fields

```json
{
  "config": "base",
  "total_tasks": 40,
  "trajectories_written": 40,
  "model_load_ok": true,
  "adapter_load_ok": false,
  "generation_ok": true,
  "aggregate_metrics": {
    "action_validity_rate": 0.0,
    "json_parse_rate": 0.0,
    "schema_valid_rate": 0.0,
    "forbidden_action_count": 0,
    "runtime_crash_count": 0,
    "task_success_rate": 0.0,
    "tool_error_rate": 0.0
  }
}
```

## 9. SFT Dataset Builder (Phase G)

`scripts/build_agent_sft_dataset.py`:

### 9.1 Sources (6, per user decision #4)

| Source | Origin | Count target |
|---|---|---|
| `scripted_variant` | Augment P4.0's 40 scripted trajectories with variant patches | ~200 |
| `teacher_model` | Teacher-model (Qwen3-0.6B or larger, if available) generated correct trajectories | ~300 |
| `corrupted_recovered` | Corrupted trajectory that the evaluator records as recovered (patch succeeds after a wrong patch) | ~150 |
| `failed_patch_recovery` | Trajectory with a failed patch followed by a successful rollback + correct patch | ~150 |
| `model_self_run_success` | Phase F model trajectories where `success=True` | variable |
| `model_self_run_failure` | Phase F model trajectories where `success=False` (kept for diagnostics, NOT mixed into success demonstrations) | variable |

**Total target: 1000+ trajectories.**

Failure trajectories (`model_self_run_failure`) are tagged and kept in a
**separate** partition. They are never used as "success demonstrations" in
the SFT train split. They may be used for DPO preference pairs in P4.2
(out of scope for P4.1).

### 9.2 Success criterion by type (user decision #5)

| Criterion | Success condition |
|---|---|
| `TEST_PASS` | latest replay `run_tests` passed (replay authoritative, per Phase B) |
| `IDENTIFY_BUG` | `identification_verified=True` + evidence (memory.hypothesis non-empty) |
| `PATCH_APPLIED` | ≥1 successful patch (`successful_patches > 0`) |

SFT "success demonstration" pool **prefers** `TEST_PASS` trajectories.
`IDENTIFY_BUG` and `PATCH_APPLIED` trajectories are included but tagged
as secondary.

### 9.3 Split (user decision #6)

Split by **workspace/task family** (the 8 task types), not by random shuffle:

| Split | Task types | Purpose |
|---|---|---|
| `train` | 6 of 8 task types | SFT training (P4.2) |
| `validation` | 1 of 8 task types (held out from train) | Training-time validation |
| `heldout-agent-eval` | 1 of 8 task types (fully held out) | Post-training agent-loop eval |

**Constraint:** no task type appears in both `train` and `heldout-agent-eval`
unless explicitly tagged `curriculum_replay` (default: none).

Assignment (deterministic, by task_type hash):
- `heldout-agent-eval`: `avoid_editing_tests` (5 tasks — tests editing integrity)
- `validation`: `recover_from_failed_patch` (5 tasks — rollback/recovery)
- `train`: remaining 6 types (30 tasks)

### 9.4 Replay verification

Every trajectory in the dataset is replay-verified before inclusion:
1. Load trajectory via `load_trajectory`
2. Run through `AgentEvaluator` with `ReplayActionProvider`
3. Assert `success` matches the trajectory's `success` label
4. Assert no `forbidden_action_count`
5. Assert schema valid

Trajectories that fail replay verification are **excluded** and logged to
`reports/p4/sft-dataset-replay-failures.jsonl`.

### 9.5 Output

```
data/p4-agent/sft-v1/
  train.jsonl                    # ~750+ trajectories
  validation.jsonl               # ~100+ trajectories
  heldout-agent-eval.jsonl       # ~100+ trajectories
  failure-diagnostics.jsonl      # model_self_run_failure trajectories
  manifest.json                  # counts, source breakdown, split assignment
```

## 10. P4.1 Readiness Verifier (Phase H)

`scripts/verify_p4_1_readiness.py` — 10 gates:

| Gate | Check |
|---|---|
| 1 | P4.0 baseline lock exists + SHAs match `main` @ `7ccd06c` |
| 2 | Evaluator TEST_PASS uses replay (test `test_test_pass_success_uses_replay_not_claim` passes) |
| 3 | Evaluator unknown-action hard-fail (test `test_unknown_action_type_raises` passes) |
| 4 | Evaluator dispatches all 11 action types (test `test_search_text_dispatched` + `test_rollback_patch_dispatched` pass) |
| 5 | inspect_error returns stdout on failure (test `test_inspect_error_returns_stdout_on_test_failure` passes) |
| 6 | All 5 corruption types tested (5 corruption tests pass) |
| 7 | ModelActionProvider smoke: base config meets minimum bar (forbidden=0, crash=0, ≥1 valid action) |
| 8 | ModelActionProvider smoke: repair-lora config meets minimum bar |
| 9 | SFT dataset: 1000+ trajectories, all replay-verified, split by task family, sources labeled |
| 10 | No training executed (no `trainer.train()` calls in P4.1 code; grep check) |

Verdict: `GO_FOR_P4_AGENT_SFT` (all 10 gates PASS) or `NOT_READY`.

Report: `reports/p4/p4-1-readiness.md`.

### 10.1 Supply-chain check

Gate 10 also verifies:
- No `trainer.train()`, `SFTTrainer`, `DPOTrainer`, `PPOTrainer` in P4.1 code
- No external dataset downloads (no `requests.get`, `wget`, `curl` in P4.1 scripts)
- No model weights committed (`.safetensors`/`.bin` in `data/p4-agent/sft-v1/` → fail)

## 11. File Structure

### Files created (new)

| Path | Responsibility | Phase |
|---|---|---|
| `src/agent_model_provider.py` | ModelActionProvider + prompt builder + JSON repair + diagnostics | E |
| `scripts/lock_p4_0_baseline.py` | Phase A: lock P4.0 SHAs | A |
| `scripts/collect_model_trajectories.py` | Phase F: model trajectory collection | F |
| `scripts/build_agent_sft_dataset.py` | Phase G: SFT dataset builder | G |
| `scripts/verify_p4_1_readiness.py` | Phase H: 10-gate verifier | H |
| `tests/test_agent_model_provider.py` | Phase E tests (non-GPU) | E |
| `tests/test_agent_model_provider_gpu.py` | Phase E GPU smoke tests (marked `@pytest.mark.gpu`) | E |
| `tests/test_p4_1_readiness.py` | Phase H tests | H |
| `reports/p4/p4-0-baseline-lock.json` | Phase A output (generated) | A |
| `reports/p4/model-trajectory-collection-report.json` | Phase F output (generated) | F |
| `reports/p4/sft-dataset-replay-failures.jsonl` | Phase G output (generated) | G |
| `reports/p4/p4-1-readiness.md` | Phase H output (generated) | H |
| `data/p4-agent/trajectories-v1/model-base.jsonl` | Phase F output (generated) | F |
| `data/p4-agent/trajectories-v1/model-repair-lora.jsonl` | Phase F output (generated) | F |
| `data/p4-agent/sft-v1/train.jsonl` | Phase G output (generated) | G |
| `data/p4-agent/sft-v1/validation.jsonl` | Phase G output (generated) | G |
| `data/p4-agent/sft-v1/heldout-agent-eval.jsonl` | Phase G output (generated) | G |
| `data/p4-agent/sft-v1/failure-diagnostics.jsonl` | Phase G output (generated) | G |
| `data/p4-agent/sft-v1/manifest.json` | Phase G output (generated) | G |

### Files modified

| Path | Change | Phase |
|---|---|---|
| `src/agent_evaluator.py` | TEST_PASS replay-authoritative, finish_claim_mismatch, 11-action allowlist, search_text/rollback_patch dispatch, unknown→hard fail | B |
| `src/agent_tools.py` | `tool_inspect_error` returns stdout+stderr capped 8KB | C |
| `tests/test_agent_evaluator.py` | +6 trust-gap/dispatch tests, +5 corruption tests | B, D |
| `tests/test_agent_tools.py` | +2 inspect_error tests | C |

### Files NOT modified (frozen)

- `src/agent_actions.py` — 11-action schema unchanged
- `src/agent_state.py` — AgentMemory unchanged
- `src/agent_workspace.py` — MicroTaskWorkspace unchanged
- `src/agent_trajectory.py` — Trajectory schema unchanged (P4.0 fix stands)
- `data/p4-agent/micro-tasks-v0/` — 40 tasks frozen
- `data/p4-agent/trajectories-v0/scripted.jsonl` — 40 scripted trajectories frozen

## 12. Dependencies

No new runtime dependencies. P4.1 reuses:
- `torch`, `transformers`, `peft` — already in `requirements.txt` for P3 inference
- `pydantic` v2 — P4.0
- `pytest` — P4.0

GPU tests require the RTX 3050 (4GB VRAM). CI runs non-GPU tests only
(`-m "not gpu"`).

## 13. Testing Strategy

- **Non-GPU tests** (CI): evaluator fixes, inspect_error, corruption expansion,
  prompt builder, JSON extraction, repair layer (mocked model). Run on every
  commit.
- **GPU smoke tests** (manual): ModelActionProvider with real Qwen3-0.6B.
  Run before PR merge on the RTX 3050.
- **Replay verification** (Phase G): every SFT trajectory replay-verified
  before dataset inclusion.

## 14. Readiness Gates Summary

P4.1 ends with `GO_FOR_P4_AGENT_SFT` when all 10 gates pass:

1. P4.0 baseline locked
2. TEST_PASS replay-authoritative
3. Unknown action hard-fails
4. All 11 actions dispatched
5. inspect_error surfaces stdout
6. All 5 corruption types tested
7. Model smoke (base) meets minimum bar
8. Model smoke (repair-lora) meets minimum bar
9. 1000+ replay-verified SFT trajectories, split by task family
10. No training executed, no external data, no weights committed

`GO_FOR_P4_AGENT_SFT` authorizes **considering** P4.2 (Agent SFT training).
It does NOT authorize training. Training requires a separate P4.2 issue +
user approval.
