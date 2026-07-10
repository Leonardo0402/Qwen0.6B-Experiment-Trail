# Task 13 Report — Phase G: SFT dataset builder (6 sources, list-based replay, task-family split)

**Status:** DONE

**Commit SHA:** `40745310b79e4d61dfbf3a99ebe37735f6227730`

**BASE:** `5b43925` (T12 commit) → HEAD: `40745310b79e4d61dfbf3a99ebe37735f6227730`

## 1. What was implemented
Created `scripts/build_agent_sft_dataset.py` — the Phase G SFT dataset builder. The script aggregates trajectories from 6 sources (scripted_variant, model_self_run_success/failure from T9, teacher_model from T10, corrupted_recovered from T11, failed_patch_recovery from T12), labels each with its task_type, splits them by task family (train / validation=`recover_from_failed_patch` / heldout-agent-eval=`avoid_editing_tests` / failure-diagnostics for `model_self_run_failure`), and replay-verifies every trajectory via `_ListActionProvider` before inclusion. Outputs (generated at runtime, NOT committed) go to `data/p4-agent/sft-v1/{train,validation,heldout-agent-eval,failure-diagnostics}.jsonl`, `data/p4-agent/sft-v1/manifest.json`, and `reports/p4/sft-dataset-replay-failures.jsonl`.

`_ListActionProvider` is defined inline in the script (same pattern as T10/T11/T12). `_ACTION_ADAPTER = TypeAdapter(Action)` is used to reconstruct Action objects from P4.1 JSONL action dicts; SentinelActions are reconstructed via the `__sentinel__` marker.

## 2. Syntax verification
Command: `py -3.11 -c "import ast; ast.parse(open('scripts/build_agent_sft_dataset.py').read()); print('OK')"`

Output:
```
OK
```

## 3. Diffstat
Command: `git diff --stat 5b43925 HEAD`

Output:
```
 scripts/build_agent_sft_dataset.py | 257 +++++++++++++++++++++++++++++++++++++
 1 file changed, 257 insertions(+)
```

`git log -1 --format='%H%n%s'`:
```
40745310b79e4d61dfbf3a99ebe37735f6227730
feat(p4-1): Phase G — SFT dataset builder (6 sources, list-based replay, task-family split)
```

## 4. Corrections applied
Confirmed: Pre-flight Correction 1 was ALREADY applied in the brief code and preserved verbatim in the committed file. Specifically:
- `from pydantic import TypeAdapter` is present in the imports (after `from pathlib import Path`).
- Module-level `_ACTION_ADAPTER = TypeAdapter(Action)` is defined after the `_ListActionProvider` class, before the path constants.
- In `_reconstruct_actions`, the line is `actions.append(_ACTION_ADAPTER.validate_python(d))` — NOT `Action.model_validate(d)`. This avoids the `AttributeError` that would occur because `Action` is an `Annotated[Union[...], Field(discriminator="action_type")]` alias, not a Pydantic BaseModel class.

## 5. Self-review findings
- **Surgical changes:** Only ONE new file was created (`scripts/build_agent_sft_dataset.py`, 257 lines). No existing files were modified.
- **Frozen files respected:** `src/agent_trajectory.py` and all other `src/` files were NOT touched.
- **Staging discipline:** `git add scripts/build_agent_sft_dataset.py` was the ONLY staged path. The pre-existing modified/untracked `.superpowers/sdd/` files (briefs, reports, review packages, `progress.md`, `task_brief.py`, etc.) were left unstaged and are NOT part of the commit. Verified via `git status --short` before committing (only `A  scripts/build_agent_sft_dataset.py` was in the index).
- **Verbatim brief code:** The file content matches the brief exactly, including the inline `_ListActionProvider` class, the `__sentinel__` marker handling, the split logic (HELDOUT_TYPE / VALIDATION_TYPE / train), and the failure-diagnostics routing for `model_self_run_failure`.
- **Commit message:** Matches the brief exactly: `feat(p4-1): Phase G — SFT dataset builder (6 sources, list-based replay, task-family split)`.
- **No run attempted:** Per task instructions, only `ast.parse` syntax verification was performed; the script was NOT executed (it depends on T9/T10/T11/T12 outputs that are generated at runtime and not committed).

## 6. Concerns
None. The script is syntax-valid, follows the T10/T11/T12 inline-`_ListActionProvider` pattern, and the pre-flight Correction 1 (TypeAdapter) is correctly applied.

## 7. One-line test summary
`ast.parse` on `scripts/build_agent_sft_dataset.py` returned `OK`; commit `4074531` adds only that file (+257 lines) on top of T12 base `5b43925`.
