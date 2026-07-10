# Task 9 Report — Phase F: Model trajectory collection script (RecordingProvider + action-list JSONL)

**Status:** DONE

**Commit SHA:** `96fceaa5ca36ad5001f836279b7fc15081a72312`

**BASE:** `7069daf` (T8 commit) → HEAD: `96fceaa5ca36ad5001f836279b7fc15081a72312`

## 1. What was implemented

Created `scripts/collect_model_trajectories.py` (173 lines) verbatim from the Task 9 brief. The script contains:

- A `RecordingProvider(ActionProvider)` wrapper class that captures each action returned by an inner provider (dict via `action.model_dump()` for normal actions, `{"__sentinel__": True, ...}` for `SentinelAction`s) for later replay.
- Two configs (`base`, `repair-lora`) of `ModelActionProvider` from `src/agent_model_provider.py`.
- `_run_config()` that loads the model once, iterates over all 40 micro-tasks (loaded from `data/p4-agent/micro-tasks-v0/manifest.json`), runs `AgentEvaluator` (max_steps=12), records per-task trajectories with an `actions` list field (NOT P4.0 `Trajectory` schema — per design note, since `src/agent_trajectory.py` is FROZEN).
- `main()` that writes JSONL files to `data/p4-agent/trajectories-v1/model-{base,repair-lora}.jsonl` and a summary report to `reports/p4/model-trajectory-collection-report.json` (these outputs are generated at runtime on GPU — NOT committed).

The script is GPU-only and not executed in this environment; only syntax was verified.

## 2. Syntax verification

```
$ py -3.11 -c "import ast; ast.parse(open('scripts/collect_model_trajectories.py').read()); print('OK')"
OK
```

Exit code 0. No syntax errors.

## 3. Diffstat

```
$ git diff --stat 7069daf HEAD
 scripts/collect_model_trajectories.py | 173 ++++++++++++++++++++++++++++++++++
 1 file changed, 173 insertions(+)
```

Only one file added. No files in `src/` touched.

## 4. Self-review findings

- **Surgical changes:** Only `scripts/collect_model_trajectories.py` was created — exactly one file, matching the brief's Global Constraint #1. No `src/` files modified (verified via `git diff --stat`).
- **FROZEN file untouched:** `src/agent_trajectory.py` was not modified.
- **RecordingProvider location:** Defined INSIDE the script file (not in a separate module), matching Global Constraint #4.
- **Verbatim copy:** The script content matches the brief byte-for-byte (including docstring, comments, `_CONFIGS`, `MAX_STEPS = 12`, `P4_ALLOW_NETWORK` env default). No emojis, no incidental docstrings, no refactors added.
- **Staging discipline:** Only `scripts/collect_model_trajectories.py` was staged. The pre-existing modified/untracked `.superpowers/sdd/` files (progress.md, task-*-brief.md, task-*-report.md, review-package.md, etc.) were explicitly NOT staged. The commit contains exactly 1 file changed, 173 insertions.
- **Commit message:** Exactly as specified in the brief: `feat(p4-1): Phase F — model trajectory collection script (RecordingProvider + action-list JSONL)`.
- **No script execution:** Per Global Constraint #3, the script was not run (no GPU in this environment). Only `ast.parse` was used for verification.

## 5. Concerns

None. The script content is the verbatim brief text; runtime execution is deferred to the GPU environment before PR merge, as the brief explicitly states.

## 6. One-line test summary

`py -3.11 -c "import ast; ast.parse(open('scripts/collect_model_trajectories.py').read()); print('OK')"` → `OK` (syntax verified; runtime GPU execution deferred to pre-PR-merge window).
