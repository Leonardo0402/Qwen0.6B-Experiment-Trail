# P4.1 SDD Progress Ledger

**Plan:** docs/superpowers/plans/2026-07-09-p4-1-model-action-provider-and-sft-data.md
**Branch:** feat/p4-1-model-action-provider
**Base:** 7ccd06c (P4.0 merge, PR #18)
**Tasks:** 15 (T1-T15)

## Pre-flight Findings (resolved in commit e70d339)
1. T9↔T10 trajectory schema mismatch → T9 uses RecordingProvider + actions list, T13 uses _ListActionProvider for replay
2. SentinelAction vs forbidden_count=0 → evaluator counts invalid_action_count separately (T6)
3. 1000+ trajectories unreachable → added T10/T11/T12 augmentation generators (teacher_model, corrupted_recovered, failed_patch_recovery)

## Task Status

- Task 1: P4.0 baseline lock — COMPLETE
- Task 2: TEST_PASS replay-authoritative + finish_claim_mismatch — COMPLETE
- Task 3: 11-action allowlist + unknown hard-fail + search_text/rollback_patch dispatch — COMPLETE
- Task 4: inspect_error returns stdout+stderr capped 8KB — COMPLETE
- Task 5: Corruption test expansion (all 5 CorruptionType) — COMPLETE
- Task 6: ModelActionProvider prompt builder + JSON extraction + repair + diagnostics + evaluator SentinelAction handling — COMPLETE
- Task 7: ModelActionProvider mocked generation tests — COMPLETE
- Task 8: GPU smoke tests (base + repair-lora) — COMPLETE
- Task 9: Model trajectory collection script (RecordingProvider) — COMPLETE
- Task 10: teacher_model augmentation generator — COMPLETE
- Task 11: corrupted_recovered augmentation generator — COMPLETE
- Task 12: failed_patch_recovery augmentation generator — COMPLETE
- Task 13: SFT dataset builder (6 sources, task-family split, replay-verify via _ListActionProvider) — COMPLETE
- Task 14: 10-gate readiness verifier — COMPLETE
- Task 15: Readiness tests + final verification — COMPLETE

## Completion Log

- Task 1: complete (commits e70d339..824a5c2, review clean — Approved, 2 Minor cosmetic only)
- Task 2: complete (commit f3a9a7e, review Approved — no Critical/Important/Minor findings; 2 ⚠️ resolved: TDD evidence in report, _make_result no external callers confirmed via grep)
- Task 3: complete (commit 41a0d05, review Approved — no Critical/Important; 3 Minor: import unittest.mock inside test fn, no direct rollback_patch test, mid-file import _ALLOWED_ACTION_TYPES)
- Task 4: complete (commit 50ec3af, review Approved — no Critical/Important/Minor findings; pre-flight correction: TestObservation schema fields)
- Task 5: complete (commit 439c097, review Approved — no findings; WRONG_PATCH narrowed to apply_patch (verified authorized strengthening), SKIP_TESTS/INVALID_PATH dynamic step-index search, no evaluator changes needed)
- Task 6: complete (commit 37c4ef2, review Approved — no findings; 3 corrections applied: test_all_metrics_present 9-key update, _make_result signature+call sites, _validate_action top-level import; 76/76 tests pass)
- Task 7: complete (commit 2f244cd, review Approved — no findings; FOUND AND FIXED T6 DEFECT: Action.model_validate raised AttributeError on Annotated alias → TypeAdapter(Action).validate_python; 10/10 model_provider tests pass, 31/31 broader regression)
- Task 8: complete (commit 7069daf, review Approved — no findings; 2 GPU smoke tests created with pytestmark=pytest.mark.gpu, gpu marker registered in pyproject.toml, 2 collected/2 deselected, 31/31 regression pass; actual GPU run deferred to manual RTX 3050 before PR merge)
- Task 9: complete (commit 96fceaa, review Approved — no findings; scripts/collect_model_trajectories.py created verbatim from brief, RecordingProvider inline, ast.parse OK, 1 file 173 insertions, no src/ touched; actual trajectory collection deferred to GPU run)
- Task 10: complete (commit 69257e3, review Approved — no findings; scripts/augment_teacher_model.py created verbatim, _ListActionProvider inline, Correction 1 applied (SentinelAction import removed), ast.parse OK, 1 file 118 insertions, no src/ touched)
- Task 11: complete (commit 75a6f56, review Approved — no findings; scripts/augment_corrupted_recovered.py created verbatim, _ListActionProvider inline, CorruptedActionProvider/Corruption/CorruptionType imports verified, ast.parse OK, 1 file 121 insertions, no src/ touched)
- Task 12: complete (commit 5b43925, review Approved — no findings; scripts/augment_failed_patch_recovery.py created verbatim, _ListActionProvider inline, _build_recovery_sequence function, ast.parse OK, 1 file 117 insertions, no src/ touched)
- Task 13: complete (commit 4074531, review Approved — no findings; scripts/build_agent_sft_dataset.py created verbatim, _ListActionProvider inline, Correction 1 applied (TypeAdapter instead of Action.model_validate), 6 sources loaded, task-family split, replay-verify, ast.parse OK, 1 file 257 insertions, no src/ touched)
- Task 14: complete (commit 71c2660, review Approved — no findings; scripts/verify_p4_1_readiness.py created verbatim, 10 gate functions, _GATES list of 10, GO_FOR_P4_AGENT_SFT verdict logic, Correction 1 applied (gate 03 test name test_unknown_action_type_recorded_as_forbidden), ast.parse OK, 1 file 250 insertions, no src/ touched)
- Task 15: complete (commit 55d400b, review Approved — no findings; tests/test_p4_1_readiness.py created verbatim, 4 tests (report exists, verdict GO, 10 gates listed, augmentation scripts exist), gate-10 logic verified (0{i}_ vs {i:02d}_), ast.parse OK, 1 file 33 insertions, no src/ touched, no report file committed)

## Final Whole-Branch Review

- **All 15 tasks COMPLETE and reviewed (Approved, no findings on T9-T15)**
- **Baseline lock fix:** commit 630ddf3 — regenerated reports/p4/p4-0-baseline-lock.json to update agent_evaluator_sha256 after T2/T3/T6 deliberately modified src/agent_evaluator.py. T1's test_baseline_lock_shas_match_files was checking the P4.0 SHA against the P4.1-modified file. Frozen-file SHAs (manifest, scripted trajectories, readiness report) and p4_0_merge_commit (7ccd06c) unchanged.
- **Non-GPU regression suite:** 107/107 PASS (test_agent_actions, test_agent_tools, test_p3_exit_lock, test_agent_trajectory, test_micro_task_suite, test_scripted_trajectories, test_agent_evaluator, test_agent_model_provider, test_p4_0_baseline_lock)
- **Frozen files verified untouched:** src/agent_actions.py, src/agent_state.py, src/agent_workspace.py, src/agent_trajectory.py — none in diff 7ccd06c..HEAD
- **No training code:** no trainer.train/SFTTrainer/DPOTrainer/PPOTrainer/RLTrainer in P4.1 scripts or src/agent_model_provider.py (only in pre-existing scripts/train_lora.py from P3, and as string literals in gate 10's forbidden patterns list)
- **No network calls:** no requests.get/requests.post/wget/curl in P4.1 scripts (only as string literals in gate 10's forbidden patterns list)
- **No weight files:** no .safetensors/.bin/.pt/.ckpt in data/p4-agent/
- **test_p4_1_readiness.py:** 4 tests deferred to pre-merge (require reports/p4/p4-1-readiness.md generated by T14 verifier after GPU smoke + dataset build)

## Pre-Merge Manual Steps (require RTX 3050 GPU)

1. GPU smoke tests: `py -3.11 -m pytest tests/test_agent_model_provider_gpu.py -v -m gpu`
2. Model trajectory collection: `py -3.11 scripts/collect_model_trajectories.py`
3. Augmentation generators: `py -3.11 scripts/augment_teacher_model.py` → `augment_corrupted_recovered.py` → `augment_failed_patch_recovery.py`
4. SFT dataset build: `py -3.11 scripts/build_agent_sft_dataset.py`
5. Readiness verification: `py -3.11 scripts/verify_p4_1_readiness.py` (all 10 gates PASS → GO_FOR_P4_AGENT_SFT)
6. Final test run: `py -3.11 -m pytest tests/ -v -p no:warnings -m "not gpu"`
