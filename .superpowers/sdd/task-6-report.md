# Task 6 Report: Baseline Lock + Smoke Run Script

**Status:** DONE
**Branch:** `feat/p4-1b-protocol-ablation`
**Commit SHA:** `9ebfdd01483890148ee6b3bbf19f9f2abf7c59f4`
**Base SHA:** `d92787c` (T5 commit)

## 1. Files Created/Modified

### Created
- `scripts/run_protocol_ablation.py` (361 lines) — orchestration script with `baseline_lock()`, `run_combination()`, `aggregate_metrics()`, `classify_failures()`, `RecordingProvider`, and `main()`. Transcribed verbatim from the brief.
- `tests/test_protocol_ablation.py` (190 lines) — 4 integration tests. Transcribed verbatim from the brief, with `test_run_combination_with_mock_protocol` taken from the PROACTIVE FIX section (not the plan's original version).

### Modified
- None outside the two files above. Only `scripts/run_protocol_ablation.py` and `tests/test_protocol_ablation.py` were staged in the commit (verified via `git show --stat HEAD`).

## 2. Test Results

### Step 2 — Pre-implementation (RED)

```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 4 items

tests\test_protocol_ablation.py FFFF                                     [100%]

================================== FAILURES ===================================
____________________ test_baseline_lock_records_all_fields ____________________

    def test_baseline_lock_records_all_fields():
>       from scripts.run_protocol_ablation import baseline_lock
E       ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'

tests\test_protocol_ablation.py:48: ModuleNotFoundError
_________________ test_aggregate_metrics_computes_all_fields __________________

    def test_aggregate_metrics_computes_all_fields():
>       from scripts.run_protocol_ablation import aggregate_metrics
E       ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'

tests\test_protocol_ablation.py:62: ModuleNotFoundError
___________________ test_classify_failures_returns_taxonomy ___________________

    def test_classify_failures_returns_taxonomy():
>       from scripts.run_protocol_ablation import classify_failures
E       ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'

tests\test_protocol_ablation.py:101: ModuleNotFoundError
___________________ test_run_combination_with_mock_protocol ___________________

    def test_run_combination_with_mock_protocol():
        """Test that run_combination works with a mock protocol (no model loading)."""
>       from scripts.run_protocol_ablation import run_combination, _TASKS_DIR
E       ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'

tests\test_protocol_ablation.py:126: ModuleNotFoundError
=========================== short test summary info ===========================
FAILED tests/test_protocol_ablation.py::test_baseline_lock_records_all_fields - ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'
FAILED tests/test_protocol_ablation.py::test_aggregate_metrics_computes_all_fields - ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'
FAILED tests/test_protocol_ablation.py::test_classify_failures_returns_taxonomy - ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'
FAILED tests/test_protocol_ablation.py::test_run_combination_with_mock_protocol - ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'
============================== 4 failed in 0.69s ==============================
```

Matches brief expectation: FAIL with `ModuleNotFoundError: No module named 'scripts.run_protocol_ablation'`.

### Step 4 — Post-implementation (GREEN)

```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 4 items

tests\test_protocol_ablation.py ....                                     [100%]

============================== 4 passed in 0.55s ==============================
```

All 4 tests pass:
- `test_baseline_lock_records_all_fields` — verifies baseline_lock returns commit_sha, manifest sha256, model/adapter paths, generation_config (temperature=0.0, do_sample=False), and total_tasks=40.
- `test_aggregate_metrics_computes_all_fields` — verifies all 12+ required metric fields present, with correct values (format_parse_rate=1.0, schema_valid_rate=0.5, unknown_action_count=1, finish_claim_mismatch_count=1).
- `test_classify_failures_returns_taxonomy` — verifies failure taxonomy classification including REPEATED_ACTION_LOOP trajectory-level detection.
- `test_run_combination_with_mock_protocol` — verifies run_combination works end-to-end with a mocked ModelActionProvider, writes 2 trajectories (one per task), and returns metrics. Uses the corrected version from the PROACTIVE FIX section.

## 3. Deviations from the Brief

**None.** Both files were transcribed verbatim from the brief. The corrected `test_run_combination_with_mock_protocol` (from the PROACTIVE FIX section) was used instead of the plan's original version, exactly as the brief instructs.

The brief's plan code included a `time` import in `scripts/run_protocol_ablation.py` that is not used anywhere in the file. This was kept verbatim per the brief's "EXACT code to write" instruction — no surgical cleanup was applied since the brief explicitly states this is transcription, not design.

## 4. Self-Review Notes

- **Proactive fix applied correctly:** The mock's `reset()` is a real closure that recreates `actions_state["iter"] = iter(["list_files", "finish"])` and clears `provider.diagnostics = []`. This ensures each of the 2 tasks gets a fresh 2-action sequence, so `trajectories_written == 2` as asserted.
- **Commit hygiene:** `git add` was scoped to ONLY `scripts/run_protocol_ablation.py tests/test_protocol_ablation.py`. Other modified files in the working tree (progress.md, task briefs, p3-limited manifests, untracked review diffs) were left unstaged. Verified via `git show --stat HEAD` — commit touches exactly 2 files, 551 insertions.
- **Test isolation:** The mock test patches `scripts.run_protocol_ablation.ModelActionProvider` so no actual model is loaded; the test runs in ~0.5s total.
- **Interface alignment:** The implementation consumes `JsonProtocol`/`TagProtocol`/`DslProtocol` from `src.protocols`, `ModelActionProvider`/`SentinelAction` from `src.agent_model_provider`, `AgentEvaluator`/`ActionProvider`/`AgentState` from `src.agent_evaluator`, and `MicroTaskWorkspace` from `src.agent_workspace` — all interfaces the brief confirmed as verified by the controller.
- **CRLF warnings:** Git emitted `LF will be replaced by CRLF` warnings on staging (Windows repo). This is the repo's normal line-ending handling and does not affect file content or test execution.
- **Brief's `total_tasks == 40` assertion:** Verified — `baseline_lock()` hardcodes `"total_tasks": 40`, and the manifest at `data/p4-agent/micro-tasks-v0/manifest.json` contains 40 task directories (task_001 through task_040).

## 5. Commit Details

```
commit 9ebfdd01483890148ee6b3bbf19f9f2abf7c59f4
Author: Claude Code <noreply@anthropic.com>
Date:   Sat Jul 11 21:07:54 2026 +0800

    feat(p4-1b): add baseline lock and smoke run script (P4.1b T6)

 scripts/run_protocol_ablation.py | 361 +++++++++++++++++++++++++++++++++++++++
 tests/test_protocol_ablation.py  | 190 +++++++++++++++++++++
 2 files changed, 551 insertions(+)
```
