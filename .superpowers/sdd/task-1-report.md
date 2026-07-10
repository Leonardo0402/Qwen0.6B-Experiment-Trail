# Task 1 Report: Phase A — P4.0 Baseline Lock

**Status:** DONE
**Branch:** feat/p4-1-model-action-provider
**Commit:** 824a5c2
**Date:** 2026-07-09

## 1. Goal

Create a baseline lock script `scripts/lock_p4_0_baseline.py` that records the
SHA256 of four P4.0 artifacts plus three P4.0 metadata constants into
`reports/p4/p4-0-baseline-lock.json`, with a test file
`tests/test_p4_0_baseline_lock.py` enforcing the lock's contract.

This lock is consumed by T14 (readiness verifier) gate 1 to detect drift of
P4.0 artifacts during P4.1 development. All code was taken verbatim from the
task brief; the brief supplied verified values to use as-is.

## 2. What Was Implemented

Two source files (verbatim from the brief) plus one generated JSON artifact:

### 2.1 `scripts/lock_p4_0_baseline.py`
Idempotent script. Reads four P4.0 artifacts, computes SHA256 of each, and
writes a 7-field JSON lock to `reports/p4/p4-0-baseline-lock.json`. Hardcodes
the three verified P4.0 metadata constants:

- `_P4_0_MERGE_COMMIT = "7ccd06c4d479b269f7708a6a430b9965af5f17e6"`
- `_P4_0_VERDICT = "GO_FOR_P4_AGENT_SFT_DATA"`
- `_P4_0_TEST_COUNT = 81`

### 2.2 `tests/test_p4_0_baseline_lock.py`
Five tests enforcing the lock contract:
1. `test_baseline_lock_exists` — lock file present
2. `test_baseline_lock_has_required_fields` — all 7 required keys present
3. `test_baseline_lock_shas_match_files` — re-computes SHA256 of the 4 source
   files and asserts equality with the locked values (real semantic check)
4. `test_baseline_lock_p4_0_merge_commit_is_7ccd06c` — merge commit prefix
5. `test_baseline_lock_verdict_is_go` — verdict string + test count

### 2.3 `reports/p4/p4-0-baseline-lock.json` (generated)
```json
{
  "p4_0_merge_commit": "7ccd06c4d479b269f7708a6a430b9965af5f17e6",
  "micro_tasks_manifest_sha256": "bdcc2eaa268b8965ff764ac6c710c97ba90298e11b7c05d0133cdb7103f692bc",
  "scripted_trajectories_sha256": "50485b6df9a6a7eaf6fd2d00b718c32cb52545b402379cc964cd66ee350ec18f",
  "agent_evaluator_sha256": "9ee5742d535bed14ae7434417ecffa3436e975b96fd48687497f1a010877fe05",
  "readiness_report_sha256": "e4500cdc46132776cc3730363a671197cff7ca2f7b0ebed4384023337ce5cd11",
  "p4_0_verdict": "GO_FOR_P4_AGENT_SFT_DATA",
  "p4_0_test_count": 81
}
```

Locked artifacts (consumes):
- `data/p4-agent/micro-tasks-v0/manifest.json`
- `data/p4-agent/trajectories-v0/scripted.jsonl`
- `src/agent_evaluator.py`
- `reports/p4/p4-agent-foundation-readiness.md`

## 3. TDD Evidence

### 3.1 RED — test run before implementation

Command: `py -3.11 -m pytest tests/test_p4_0_baseline_lock.py -v -p no:warnings`

Result: 5 failed in 1.22s. All five tests failed for the expected reason —
the lock file did not exist yet. The first test failed with the exact
assertion message mandated by the brief; the remaining four failed with
`FileNotFoundError` because they attempt to read the lock before asserting.

Relevant output:
```
tests\test_p4_0_baseline_lock.py FFFFF                                   [100%]
FAILED tests/test_p4_0_baseline_lock.py::test_baseline_lock_exists - AssertionError: p4-0-baseline-lock.json not found
FAILED tests/test_p4_0_baseline_lock.py::test_baseline_lock_has_required_fields - FileNotFoundError: ... p4-0-baseline-lock.json
FAILED tests/test_p4_0_baseline_lock.py::test_baseline_lock_shas_match_files - FileNotFoundError: ... p4-0-baseline-lock.json
FAILED tests/test_p4_0_baseline_lock.py::test_baseline_lock_p4_0_merge_commit_is_7ccd06c - FileNotFoundError: ... p4-0-baseline-lock.json
FAILED tests/test_p4_0_baseline_lock.py::test_baseline_lock_verdict_is_go - FileNotFoundError: ... p4-0-baseline-lock.json
```

Why expected: the lock script had not yet been written, so the JSON lock file
was absent. This confirms the tests genuinely depend on the script's output
rather than testing pre-existing state.

### 3.2 GREEN — test run after implementation

Command (script): `py -3.11 scripts/lock_p4_0_baseline.py`
Output: `wrote E:\agent\Qwen\qwen3-code-lab\reports\p4\p4-0-baseline-lock.json`

Command (tests): `py -3.11 -m pytest tests/test_p4_0_baseline_lock.py -v -p no:warnings`

Result: 5 passed in 0.39s. Output pristine — no warnings, no errors.

```
tests\test_p4_0_baseline_lock.py .....                                   [100%]
============================== 5 passed in 0.39s ==============================
```

## 4. Files Changed

Created (committed in `824a5c2`):
- `scripts/lock_p4_0_baseline.py` (45 lines)
- `tests/test_p4_0_baseline_lock.py` (57 lines)
- `reports/p4/p4-0-baseline-lock.json` (9 lines)

Total: 3 files changed, 111 insertions(+).

No pre-existing files were modified. The parent agent's working-tree changes
(`.superpowers/sdd/progress.md`, `.superpowers/sdd/task-1-brief.md`) and other
untracked scratch files were intentionally left unstaged.

## 5. Commit

```
824a5c2 feat(p4-1): Phase A — P4.0 baseline lock
```

Commit message matches the brief verbatim, including the em dash (U+2014),
consistent with existing repo history (e.g. commit `d91586e`).

## 6. Self-Review

**Completeness**
- [x] Test file written verbatim from brief (5 tests).
- [x] Lock script written verbatim from brief.
- [x] Lock file generated with all 7 required fields.
- [x] All 5 tests pass; output pristine.
- [x] Committed with the exact brief message.
- [x] RED and GREEN evidence captured.

**Quality**
- [x] Code is verbatim from the brief — no deviations, no over-engineering.
- [x] Lock file is valid JSON, 2-space indent.
- [x] All four SHA256 values are 64-char hex strings.
- [x] Script is idempotent (deterministic read-then-write; re-running yields
      byte-identical JSON modulo line-ending normalization).
- [x] The SHA-matching test re-computes hashes from the source files on disk,
      so it is a real semantic check, not a tautology.

**Discipline (TDD + surgical)**
- [x] Test written first; watched it fail before any implementation.
- [x] Each failure was for the expected reason (missing file, not a typo).
- [x] Only the three brief-specified files were created.
- [x] No pre-existing code, comments, or formatting touched.
- [x] No emojis in code, commit message, or lock file.
- [x] No staging of unrelated parent changes.

**Testing**
- [x] 5/5 passing.
- [x] Output pristine (`-p no:warnings` per brief).
- [x] Tests cover existence, schema, SHA integrity, merge-commit identity,
      and verdict/test-count constants.

## 7. Concerns

None blocking. Three informational notes:

1. **Verdict/test_count are recorded constants, not derived.** Per the brief,
   `p4_0_verdict` and `p4_0_test_count` are hardcoded in the script rather
   than parsed from the readiness report. This is by design — they are
   human-readable annotations of the P4.0 state. The report itself is locked
   via `readiness_report_sha256`, so any drift in the report will be caught
   by T14's SHA gate; the verdict/test_count fields are not the primary
   integrity signal.

2. **Overwrote a stale `task-1-report.md`.** The pre-existing file at this
   path was a leftover report from a prior P3 cycle (Issue #9, branch
   `feat/p3-capability-expansion-v2`). The parent task description explicitly
   instructed writing the new report to this exact path, so it was
   overwritten. The `.superpowers/sdd/` directory is the per-plan scratch
   area, so this is expected behavior.

3. **Windows line-ending warning.** Git emitted `LF will be replaced by CRLF`
   warnings on staging. This is normal on Windows and does not affect
   correctness: the script and the test both read the same bytes from disk
   when computing/comparing SHA256, so line-ending normalization cannot
   cause a spurious mismatch.

## 8. Environment

- Python: 3.11.7 (invoked as `py -3.11`)
- pytest: 9.1.1
- Branch: `feat/p4-1-model-action-provider`
- P4.0 merge commit confirmed in `git log`: `7ccd06c feat(p4-agent): P4.0 Agentic Coder Foundation (#18)`
