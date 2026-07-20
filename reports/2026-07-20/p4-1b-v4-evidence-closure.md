# P4.1b v4 Evidence Closure

> **Status**: This document closes the v4 evidence chain for PR #30.
> It does NOT authorize training, does NOT merge the PR, and does NOT
> close Issue #32. It prepares the evidence for independent review only.

## 1. Scope

This closure report covers:

- Confirming the exact CI failure that blocked PR #30 on commit `71389e0`.
- Establishing a clear test-layering model so CPU CI does not depend on
  ~1.5GB local model / adapter weights.
- Adding CPU-only evidence tests that validate the committed v4 artifacts.
- Correcting factual / causal statements in `v4-vs-v3-analysis.md`.
- Updating the PR #30 description and Issue #32 to v4 authority.
- Verifying the final GitHub Actions CI run is green on the new HEAD.

This closure does NOT modify:

- `src/` (any protocol / action / evaluator / model provider code)
- `scripts/run_protocol_ablation.py` (experiment harness)
- `data/p4-agent/` (task definitions)
- `models/` or `adapters/` (weights)
- Any v4 core artifact (trajectories, comparison-matrix, comparison-report,
  failure-taxonomy, verdict, baseline-lock, artifact-manifest)
- The Action Schema, the Protocol parser implementation, the Prompt
  implementation, the Evaluator implementation, or the Tool Layer

The v4 experiment commit `a5577ba081c806aaf00627262a36cbb1c38e72df`
remains the authoritative source of the v4 evidence. The closure work
only adds tests, fixes CI configuration, and corrects post-hoc analysis
text. It does not rerun the 240-run ablation and does not change any
result.

## 2. Initial PR and CI State

- Branch: `feat/p4-1b-protocol-ablation`
- Initial HEAD (before closure work): `71389e0f38d0db73339d7b6176b8f76e75fa1997`
- v4 experiment commit: `a5577ba081c806aaf00627262a36cbb1c38e72df`
- Latest failed CI run (on `71389e0`):
  - Workflow run ID: `29736679048`
  - Workflow name: `CI (CPU tests)`
  - Event: `pull_request`
  - Conclusion: `failure`
  - Job: `Non-GPU unit tests` (databaseId `88333541515`)
  - Failed step: `Run all tests`
  - Original pytest command: `python -m pytest tests/ -m "not gpu" -v --tb=short --timeout=120 -p no:warnings`

Unrelated working-tree modifications present at preflight (preserved, NOT
part of this closure's commits):

- `AGENTS.md`
- `data/p3-limited/balanced-limited/manifest.json`
- `data/p3-limited/balanced-limited/train.jsonl`
- `data/p3-limited/repair-limited/manifest.json`
- `data/p3-limited/repair-limited/train.jsonl`
- `reports/p4/p4-1-readiness.md`
- `reports/p4/p4-1-test-evidence.json`

Untracked (also preserved, NOT part of this closure's commits):

- `docs/agent-protocols/{plan,spec-doc,spec-process}-document-review-gate/`
- `docs/workflow/spec-plan-review-session-prompts.md`
- `docs/workflow/spec-plan-review-workflow.md`
- `reports/p4/protocol-ablation-v4-smoke/`
- `test-results.xml`, `test-results-full.xml`

## 3. Exact CI Failure

CI step output recorded:

```
3 failed, 1469 passed, 4 skipped, 2 deselected in 206.00s
FAILED tests/test_trust_repair.py::TestModelAdapterFingerprint::test_model_fingerprint_real_model
    - assert False is True
FAILED tests/test_trust_repair.py::TestModelAdapterFingerprint::test_adapter_fingerprint_real_adapter
    - assert 'adapter_model.safetensors' in {README.md, adapter_config.json, ...}
FAILED tests/test_trust_repair.py::TestModelAdapterFingerprint::test_baseline_lock_includes_fingerprints
    - assert False is True
```

## 4. Root Cause

The 3 failing tests directly read local model / adapter directories that
do not exist in the CPU CI environment:

- `test_model_fingerprint_real_model` calls `_model_fingerprint()` which
  reads `models/Qwen3-0.6B/`. On CI this directory is absent, so
  `fp["exists"]` is `False` and the assertion `assert fp["exists"] is True`
  fails.
- `test_adapter_fingerprint_real_adapter` calls
  `_adapter_fingerprint("adapters/p3/repair-limited")`. On CI this
  directory is absent, so the returned `files` dict is empty (or does not
  contain `adapter_model.safetensors`), and the assertion fails.
- `test_baseline_lock_includes_fingerprints` calls `baseline_lock()` which
  reads both local paths; the resulting `exists` fields are `False`, so
  `assert lock["model_fingerprint"]["exists"] is True` fails.

A fourth test, `test_post_experiment_fingerprint_check_passes`, also
reads the local paths but happened to "pass" on CI because
`_assert_fingerprint_unchanged` compares two empty `aggregate_sha256`
strings (both `""`), which are equal. That pass is semantically
misleading: the test's intent is to verify real artifact pre/post
equality, not equality of two absences.

The CPU CI runner has no GPU, no 1.5GB model weights, and no adapter
weights. The original `pytest -m "not gpu"` filter did not separate
these local-artifact tests from true CPU tests.

## 5. Test Classification

After the closure work, tests are classified into three layers:

| Layer | Marker | Runs in CPU CI | Runs locally (RTX 3050) | Purpose |
|-------|--------|----------------|-------------------------|---------|
| Unit tests | (none) | yes | yes | Pure logic, schema, parser, metrics |
| Local artifact integration tests | `@pytest.mark.local_artifacts` | no | yes | Reads real `models/Qwen3-0.6B` / `adapters/p3/repair-limited` |
| GPU tests | `@pytest.mark.gpu` | no | yes | Requires physical GPU |

Marker registration: `pyproject.toml` `[tool.pytest.ini_options]` markers
list now includes both `gpu` and `local_artifacts`.

Tests marked `local_artifacts` in `tests/test_trust_repair.py`
(`TestModelAdapterFingerprint` class):

- `test_model_fingerprint_real_model`
- `test_adapter_fingerprint_real_adapter`
- `test_baseline_lock_includes_fingerprints`
- `test_post_experiment_fingerprint_check_passes`

The other 10 tests in `TestModelAdapterFingerprint` (fingerprint algorithm
tests using `tmp_path`, `_assert_fingerprint_unchanged` mock tests,
`_adapter_fingerprint(None)` test) remain in the CPU CI unit-test layer.
They verify the fingerprint algorithm itself and do not need real
artifacts.

## 6. Committed Evidence Tests

New file: `tests/test_p4_1b_v4_evidence.py` (CPU-only, no
`local_artifacts` marker).

This file reads ONLY committed JSON / JSONL files under
`reports/p4/protocol-ablation-v4/`. It does not touch the local model or
adapter directories. It enforces:

- **Section 3.1 — baseline-lock basic fields**: experiment_commit_sha,
  git_worktree_clean_for_experiment, report_dir_name, total_tasks=40,
  total_combinations=6, total_runs=240, max_steps=12, protocols, configs,
  task_ids count/ uniqueness/ pattern, micro_task_manifest_sha256.
- **Section 3.2 — model_fingerprint structure**: exists, file_count,
  aggregate_sha, presence of config.json / model.safetensors /
  tokenizer.json, positive model.safetensors size, every file sha is
  64-hex.
- **Section 3.3 — adapter_fingerprint structure**: repair_lora exists,
  aggregate_sha, adapter_config.json / adapter_model.safetensors present,
  positive size, no `checkpoint-*` subdirs, base adapter fingerprint is
  absent.
- **Section 3.4 — pre/post equality**: model_fingerprint pre ==
  model_fingerprint_post, adapter_fingerprint_repair_lora pre ==
  adapter_fingerprint_repair_lora_post, fingerprint_check_passed is true,
  post files dict equals pre files dict.
- **Section 3.5 — artifact-manifest**: report_dir, artifact_count == 11,
  no self-reference, no `v4-vs-v3-analysis.md` in manifest, 6 trajectory
  JSONL files, each row_count == 40, total rows == 240, all SHAs are
  64-hex, all experiment_commit fields match `a5577ba...`, all 11
  required core artifacts are present.
- **Section 3.6 — comparison-matrix + verdict**: 6 combinations, all
  protocols/configs covered, 40 trajectories per combination, 480 steps
  per combination, 0 runtime crashes, 0 task successes, 40/40
  max_steps_hit, model_load_ok true for all, verdict is
  `FIX_PROMPT_FIRST`.
- **Section 3.7 — v4-vs-v3-analysis.md presence**: the post-hoc analysis
  document exists and the v4 directory contains exactly 13 files
  (7 in root + 6 trajectories).

Path validation uses `relative_path.replace("\\", "/")` so the tests
work on both POSIX CI and Windows local machines.

Total: 52 test cases in this file, all CPU-only.

## 7. Local Artifact Tests

The 4 tests marked `@pytest.mark.local_artifacts` in
`tests/test_trust_repair.py` were executed locally on the RTX 3050
machine where `models/Qwen3-0.6B/` and `adapters/p3/repair-limited/`
exist.

Local command:

```
py -3.11 -m pytest tests/test_trust_repair.py -m "local_artifacts" -v --tb=short -p no:warnings
```

Result: **4 passed, 108 deselected in 15.29s**.

This confirms that the v4 evidence's pre/post fingerprint equality holds
against the real model and adapter on disk, not just against the
committed JSON snapshots.

## 8. CI-Equivalent Results

Local CI-equivalent command (closure-scope subset, see note below):

```
py -3.11 -m pytest tests/test_p4_1b_v4_evidence.py tests/test_trust_repair.py tests/test_protocol_json.py tests/test_protocol_tag.py tests/test_protocol_dsl.py -m "not local_artifacts" --tb=short -p no:warnings
```

Result: **196 passed, 4 deselected in 19.75s**.

This run includes:

- `tests/test_p4_1b_v4_evidence.py` — 52 new CPU-only v4 evidence tests.
- `tests/test_trust_repair.py` — 108 trust repair CPU tests (4
  `local_artifacts` tests deselected).
- `tests/test_protocol_{json,tag,dsl}.py` — protocol parser tests.

### 8.1 Note on `tests/test_data_pipeline.py` (pre-existing Windows issue)

A full `tests/` run on Windows locally surfaces a pre-existing failure
in `tests/test_data_pipeline.py::test_build_dataset_creates_split_files`
caused by `shutil.rmtree(tmpdir, ignore_errors=True)` timing out during
sandbox cleanup in `src/sandbox.py:401`. This failure is:

- **Unrelated to this closure**: the closure commit does not modify
  `src/sandbox.py`, `src/validators.py`, or `tests/test_data_pipeline.py`.
- **Windows-specific**: the GitHub Actions CPU CI (Ubuntu) log for HEAD
  `71389e0` shows `1469 passed` including this test.
- **Pre-existing**: present on HEAD `71389e0` before any closure changes.

The closure therefore relies on GitHub Actions (Ubuntu) for the full
`tests/` suite result, while the local CI-equivalent verification above
covers all closure-scope tests.

## 9. GitHub CI Result

After the closure commit was pushed, the GitHub Actions workflow
`CI (CPU tests)` ran on HEAD `ae8791c`:

```
python -m pytest tests/ -m "not gpu and not local_artifacts" -v --tb=short --timeout=120 -p no:warnings
```

Final CI result (verified):

- Workflow run ID: `29742946676` (pull_request event)
- Push run ID: `29742944683` (push event, also success)
- Job ID: `88353776234`
- Job name: `Non-GPU unit tests`
- Tested SHA: `ae8791c3e4d93a37c89a8414da9c17bfaf271cd6`
- Workflow status: `completed`
- Workflow conclusion: `success`
- Job conclusion: `success`
- Started: `2026-07-20T12:37:54Z`
- Completed: `2026-07-20T12:42:33Z`
- Runtime: 262.39s (4m 22s)
- Test result: **1520 passed, 4 skipped, 6 deselected**
  - 6 deselected = 4 `local_artifacts` + 2 `gpu`
  - 4 skipped = pre-existing skips
  - 0 failed

URL: https://github.com/Leonardo0402/Qwen0.6B-Experiment-Trail/actions/runs/29742946676/job/88353776234

## 10. v4 Evidence Summary

| Field | Value |
|-------|-------|
| v4 experiment commit | `a5577ba081c806aaf00627262a36cbb1c38e72df` |
| Working tree clean at experiment time | `true` |
| Micro-task manifest SHA256 | `bdcc2eaa268b8965ff764ac6c710c97ba90298e11b7c05d0133cdb7103f692bc` |
| Tasks | 40 |
| Protocols | 3 (json / tag / dsl) |
| Configs | 2 (base / repair-lora) |
| Combinations | 6 |
| Trajectories | 240 (40 per combination) |
| Steps per combination | 480 |
| Total steps | 2880 |
| Runtime crashes | 0 |
| Model load OK | true for all 6 combinations |
| Max-steps-hit | 40/40 for all 6 combinations |
| Task success | 0/40 for all 6 combinations |
| Model fingerprint (pre) | `5b935e4a160c011081dcf16bdb2c84bf1184caefa114ee4657d21c417fcb963f` |
| Model fingerprint (post) | `5b935e4a160c011081dcf16bdb2c84bf1184caefa114ee4657d21c417fcb963f` |
| Adapter fingerprint (pre) | `5a3602be1890eefe297cedec04fd4217757ff4f5cfeae18eec1168a281345434` |
| Adapter fingerprint (post) | `5a3602be1890eefe297cedec04fd4217757ff4f5cfeae18eec1168a281345434` |
| `fingerprint_check_passed` | `true` |
| FORMAT_PARSE_FAIL | 744 |
| SCHEMA_VALIDATION_FAIL | 2081 |
| REPEATED_ACTION_LOOP | 240 |
| Verdict | `FIX_PROMPT_FIRST` |
| Core experiment artifacts | 11 (listed in `artifact-manifest.json`) |
| Manifest self-reference | absent |
| Post-hoc analysis document | `v4-vs-v3-analysis.md` (not in manifest) |
| Total files in v4 directory | 13 |

## 11. v4 Artifact Count Explanation

The `reports/p4/protocol-ablation-v4/` directory contains 13 files. The
`artifact-manifest.json` reports `artifact_count: 11` because the manifest
is self-excluded and only lists core experiment artifacts:

- **11 core experiment artifacts** (in manifest): `baseline-lock.json`,
  `comparison-matrix.json`, `comparison-report.md`,
  `failure-taxonomy.json`, `verdict.json`, and 6 trajectory JSONL files
  under `trajectories/`.
- **1 manifest file**: `artifact-manifest.json` (self-excluded from its
  own listing).
- **1 post-hoc analysis document**: `v4-vs-v3-analysis.md` (not a core
  experiment artifact; not in the manifest).

`tests/test_p4_1b_v4_evidence.py::TestArtifactManifest::test_manifest_does_not_reference_itself`
and `::test_v4_directory_total_file_count_is_13` enforce this contract.

## 12. v4-vs-v3 Analysis Corrections

The previous draft of `v4-vs-v3-analysis.md` had four factual / causal
issues that have been corrected in this closure:

1. **Causality wording**: The previous draft attributed the v4 format
   improvement solely to `raw_decode()`. Corrected: v3 → v4 changed 4
   variables simultaneously (token budget, JSON extraction, default
   injection, protocol prompt formatting); `raw_decode()` only affects
   JSON, yet Tag/DSL format rates also shifted. The v3/v4 comparison is
   now explicitly described as a state comparison, not a single-variable
   causal ablation.
2. **`search_text` schema**: The previous draft listed `search_text` as
   requiring `query + path`. Corrected to its actual schema
   `search_text(query: str, file_glob: str | None = None, max_results: int = 20)`.
   `path` is not part of `SearchTextArgs`.
3. **Artifact count wording**: The previous draft did not clearly
   distinguish the 11 core artifacts from the 13 directory files.
   Corrected: §5 of the analysis document now states explicitly that
   `artifact_count` is 11, the directory contains 13 files, and the two
   numbers refer to different concepts.
4. **v5 status**: The previous draft's v5 section recommended silent
   defaults such as `missing identification_verified → default false`.
   Corrected: §4 of the analysis document now explicitly states v5 is
   PROPOSED / NOT AUTHORIZED / OUTSIDE PR #30, does NOT recommend silent
   defaults, and proposes instead a runtime-authoritative finish
   direction as a future P4.1c candidate.

## 13. v5 Status

- v5 is **PROPOSED** in §4 of `v4-vs-v3-analysis.md` for planning
  purposes only.
- v5 is **NOT AUTHORIZED** by this PR or by Issue #32.
- v5 is **OUTSIDE PR #30 scope**. No v5 code, no v5 prompt change, no v5
  Action Schema change, no v5 `finish` default injection is included in
  this PR.
- Starting v5 requires explicit user authorization in a separate issue.

## 14. Git Delivery

- Branch: `feat/p4-1b-protocol-ablation`
- Initial HEAD (before closure): `71389e0f38d0db73339d7b6176b8f76e75fa1997`
- Closure commit: see final HEAD recorded in PR #30 description.
- Files changed in closure commit:
  - `pyproject.toml` (registered `local_artifacts` marker)
  - `.github/workflows/ci-tests.yml` (CI command excludes `local_artifacts`)
  - `tests/test_trust_repair.py` (4 tests marked `local_artifacts`)
  - `tests/test_p4_1b_v4_evidence.py` (new CPU-only evidence tests, 52 cases)
  - `reports/p4/protocol-ablation-v4/v4-vs-v3-analysis.md` (corrected)
  - `reports/2026-07-20/p4-1b-v4-evidence-closure.md` (this file, new)
- Unrelated working-tree modifications (AGENTS.md, data/p3-limited/*,
  reports/p4/p4-1-*, docs/agent-protocols/*, docs/workflow/*,
  protocol-ablation-v4-smoke/, test-results*.xml) are NOT included in
  the closure commit. They are preserved as-is.
- No force push. No `git reset --hard`. No `git clean -fd`.

## 15. Self-Check Checklist

See the final reply comment in PR #30 for the completed checklist. Key
state:

- v4 core artifacts: untouched.
- v4 experiment commit: `a5577ba...` (unchanged).
- Local artifact integration tests: PASS locally (4/4 on RTX 3050).
- CPU evidence tests: PASS (52/52, runs in CI).
- Trust repair CPU tests: PASS (108/108, runs in CI).
- CI-equivalent local run: PASS.
- v4 analysis document: corrected.
- PR #30 description: updated to v4 authority.
- Issue #32: new v4 closure comment added; not closed.
- PR #30: not merged.
- Training authorized: NO.
- v5 implemented: NO.

## 16. Verdict

```
GO_FOR_INDEPENDENT_REVIEW
```

Conditions met:

- Real CI root cause confirmed from logs (3 tests requiring local
  model/adapter weights).
- Test layering established (`local_artifacts` marker registered, 4 tests
  marked).
- Local artifact integration tests pass on the RTX 3050 machine.
- CPU evidence tests pass against committed v4 JSON artifacts.
- CI-equivalent local suite passes (closure-scope subset, 196 passed).
- v4 analysis document corrected (causality, search_text schema, artifact
  count, v5 status).
- PR #30 description updated to v4 (no v2 / f58dfdf references remain).
- Issue #32 has a new v4 closure comment; not closed.
- No experiment code modified; no v5 work; no training; no merge.
- Final GitHub CI verified green on HEAD `ae8791c`:
  1520 passed, 4 skipped, 6 deselected, 0 failed.

Next step:

```
Independent review model: read-only review of v4 evidence
→ Final Gatekeeper Review
→ Decide whether to merge PR #30
```

This `GO_FOR_INDEPENDENT_REVIEW` does NOT authorize merging PR #30.
