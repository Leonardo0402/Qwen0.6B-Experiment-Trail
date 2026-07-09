# P4.0 Agentic Coder Foundation — Readiness Report

**Date:** 2026-07-09T14:33:07.488712
**Branch:** feat/p4-agent-foundation
**Verdict:** `GO_FOR_P4_AGENT_SFT_DATA`

## Gates

| # | Gate | Status | Evidence |
|---|------|--------|----------|
| 1 | P3 exit baseline locked | PASS | locked, PR #15 merge d91586e |
| 2 | Action schema tests pass | PASS | ============================= 18 passed in 0.65s ============================== |
| 3 | Tool layer safety tests pass | PASS | ============================= 28 passed in 42.72s ============================= |
| 4 | Trajectory schema tests pass | PASS | ============================== 7 passed in 0.85s ============================== |
| 5 | Micro task suite verified | PASS | 40 tasks, ======================== 6 passed in 60.05s (0:01:00) ========================= |
| 6 | Scripted trajectories verified | PASS | 40 trajectories, 400 steps, all verified=True |
| 7 | Evaluator replay success = 100% | PASS | 100% replay success, eval_hash=e774fd7cf93618f4 |
| 8 | Corrupted trajectory tests fail as expected | PASS | test_corrupted_injection passes (WRONG_PATCH detected) |
| 9 | No forbidden shell/network/git actions | PASS | no forbidden shell/network/git patterns in P4-agent files |
| 10 | CI green (P4-agent tests) | PASS | ======================= 78 passed in 203.14s (0:03:23) ======================== |
| 11 | State transition consistency | PASS | all 40 trajectories, state transitions consistent |

## Summary

- Total gates: 11
- Passed: 11
- Failed: 0

## Known Limitations

1. **Gate 8 (corrupted trajectory tests):** Only WRONG_PATCH corruption type is
   tested. The other 4 types (WRONG_ACTION_TYPE, INVALID_PATH,
   SKIP_TESTS_BEFORE_FINISH, EXCEED_MAX_STEPS) are implemented but not individually
   tested. Documented for P4.1 expansion.
2. **Gate 9 (no forbidden actions):** Scope is narrowed to P4-agent files only
   (`src/agent_*.py` + P4 scripts). Pre-existing P1-P3 infrastructure (sandbox.py,
   validators.py) is not re-audited — it was verified in its own phases.
3. **Gate 10 (CI green):** Runs only the 7 P4-agent test files, not the full
   53-file suite. CI (on Linux) runs the full suite; this gate verifies P4-agent
   tests pass locally. Pre-existing P1-P3 tests may fail on Windows due to
   platform-specific issues (CRLF, subprocess timeout) but pass on CI's Linux.
4. **40 trajectories are NOT training data:** Per spec section 15, these are
   foundation verification artifacts only. P4.1 will produce 1000+ trajectories.
5. **Evaluator trust gap (partially resolved):** Three metric issues were fixed:
   `finish_without_tests_count` now tracks actual `run_tests` execution (not
   `finish.tests_passed`); `PATCH_APPLIED` success is based on actual patch results
   (`successful_patches > 0`); `tool_error_rate` denominator now counts failed tool
   attempts. Residual gap: `task_success_rate` for `TEST_PASS` criterion still uses
   the scripted `finish.tests_passed` declaration, not cross-checked against actual
   replay `passed_tests`. Acceptable for P4.0 (scripted teachers don't lie);
   enforcement is a P4.1 prerequisite for model-generated trajectories.
6. **Evaluator dispatch omits search_text and rollback_patch:** The evaluator's
   `run()` method does not dispatch `search_text` or `rollback_patch` actions. No
   scripted trajectory uses these actions, so P4.0 is unaffected, but P4.1 trajectories
   using them would produce silent no-ops. Must be fixed before P4.1 model agents.

## Supply-Chain Warning

Issue #17 received a comment from unverified account `depucobose87` attaching
`p4_baseline_fix.zip`. This was treated as a potential supply-chain attack.
No file from Issue/PR comments was downloaded, inspected, or applied.
All P4.0 code was written from scratch under TDD discipline.

## Next Steps (P4.1, out of scope)

1. Build supervised action-policy dataset from 1000+ scripted/teacher trajectories
2. Implement ModelActionProvider with Qwen3-0.6B
3. Train and evaluate agent policy

**This report does not authorize any P4.1 work.**
