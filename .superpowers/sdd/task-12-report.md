# Task 12 Report — Phase G: failed_patch_recovery augmentation generator

**Status:** DONE

**Commit SHA:** `5b43925b2a2cc67f54c9e13555b233dabfc60473`

**BASE:** `75a6f56` (T11 commit) → HEAD: `5b43925b2a2cc67f54c9e13555b233dabfc60473`

## 1. What was implemented
Created `scripts/augment_failed_patch_recovery.py` — a Phase G augmentation generator that, for each scripted trajectory, finds every `apply_patch` step, truncates after it (simulating patch failure), then appends a recovery sequence (`rollback_patch` → `propose_patch` → `apply_patch` → `run_tests` → `finish`) drawn from the original trajectory's later steps. Each recovery sequence is replayed through `AgentEvaluator` driven by an inline `_ListActionProvider` (same pattern as T10/T11). Successful runs are written to `data/p4-agent/trajectories-v1/failed-patch-recovery.jsonl` (generated at runtime, not committed). The file content was copied verbatim from the brief — no modifications, no extra error handling, no refactors.

## 2. Syntax verification
Command: `py -3.11 -c "import ast; ast.parse(open('scripts/augment_failed_patch_recovery.py').read()); print('OK')"`

Output:
```
OK
```

Exit code 0. Script was not executed (per brief: only `ast.parse` in this environment; actual run happens before PR merge).

## 3. Diffstat
Command: `git diff --stat 75a6f56 HEAD`

Output:
```
 scripts/augment_failed_patch_recovery.py | 117 +++++++++++++++++++++++++++++++
 1 file changed, 117 insertions(+)
```

## 4. Self-review findings
- **Surgical changes:** Exactly one new file created. No files in `src/` touched; `src/agent_trajectory.py` (FROZEN) untouched. Diff contains only the new script (117 insertions, 0 deletions).
- **Staging discipline:** Only `scripts/augment_failed_patch_recovery.py` was staged. The working tree had many dirty `.superpowers/sdd/` files (briefs, reports, review packages, progress.md) — none were staged or committed.
- **Verbatim copy:** File content matches the brief byte-for-byte (module docstring, imports order, `_ListActionProvider` definition, `_build_recovery_sequence`, `main`, `if __name__ == "__main__"`). No emojis, no incidental docstrings, no extra error handling.
- **Dependencies respected:** Imports only `load_trajectories`, `AgentEvaluator`/`ActionProvider`/`AgentState`, `Action`, `MicroTaskWorkspace` — all verified in the brief's pre-flight section. No `SentinelAction` import.
- **Commit message:** Exactly `feat(p4-1): Phase G — failed_patch_recovery augmentation generator` as specified.
- **Environment:** All Python invocations used `py -3.11`.

## 5. Concerns
None. Script content is verbatim from the brief; syntax verified; staging was surgical; commit message exact.

## 6. One-line test summary
`ast.parse` on the new script returns `OK` (exit 0); runtime execution deferred to PR-merge environment per brief.
