# Task 4 Report: Run Import + Verify + Source Audit

## Status: DONE_WITH_CONCERNS

The audit script + tests + report are complete and committed, but the
verifier's `hidden >= 3` hard gate rejected 100% of standard MBPP samples.
This makes the current P3 plan INFEASIBLE until the hidden-tests shortage is
addressed (likely via the Frozen v3 hidden-test augmentation step already
specified in Task 8 of the P3 plan).

## Work Completed

1. Wrote `scripts/generate_source_audit.py` -- reads per-split manifests +
   normalised JSONL, produces `reports/p3/mbpp-source-audit.json`.
2. Wrote `tests/test_generate_source_audit.py` -- 31 tests with synthetic
   manifests (no network).  All 31 pass on Python 3.10.
3. Imported all 3 MBPP splits from HuggingFace (`google-research-datasets/mbpp`,
   revision=`main`) via `scripts/import_mbpp.py`.  Network access required the
   user's proxy at `127.0.0.1:7897` (HTTP_PROXY / HTTPS_PROXY env vars).
4. Verified all 3 splits via `scripts/verify_imported_mbpp.py` (real
   `src.validators.verify_sample`).  All 964 samples were rejected by the
   `hidden_tests >= 3` hard gate (see Finding F-1 below).
5. Generated the audit report with the real (non-synthetic) manifests.

## Actual Split Counts (from real manifests)

| Split      | Imported | Verified | Rejected | task_id range | Fingerprint        |
|------------|----------|----------|----------|---------------|--------------------|
| train      | 374      | 0        | 374      | 601 -- 974    | 5cdc16311fd8e220   |
| test       | 500      | 0        | 500      | 11 -- 510     | 44206dd27b0bf01d   |
| validation | 90       | 0        | 90       | 511 -- 600    | 8f9deacfc72ae133   |
| **Total**  | **964**  | **0**    | **964**  | --            | --                 |

Cross-split task_id overlap: **empty for all three pairs** (train_test,
train_validation, test_validation).  MBPP splits are disjoint by
construction; the audit confirms this empirically.

Missing / duplicate task_ids: **none** in any split.

## Conclusion

**INFEASIBLE** (test verified_count=0 < 240 threshold).

`new_families_available = test_verified + validation_verified = 0 + 0 = 0`.

Per the P3 plan, this result means: **stop and escalate** before proceeding
to Tasks 5-14.

## Findings

### F-1 (CRITICAL): MBPP standard tests fail the `hidden >= 3` hard gate

The verifier (`scripts/verify_imported_mbpp.py`) enforces these per-sample
hard checks (defined in Task 3, mirror of P3 plan constraint #12):

- `public_tests.count("assert ") >= 2`
- `hidden_tests.count("assert ") >= 3`

MBPP ships ~3 test cases per task in `test_list`.  The importer
(`scripts/import_mbpp.py:split_mbpp_tests`) splits them as `min(2, n)` public
+ remainder hidden, so the standard case yields 2 public + 1 hidden.  The
`hidden >= 3` check then rejects every sample.

**Rejection reason histogram** (from the verifier stdout):

| Split      | Reason                              | Count |
|------------|-------------------------------------|-------|
| train      | `hidden assertions 1 < 3`          | 374   |
| test       | `hidden assertions 1 < 3`          | 499   |
| test       | `hidden assertions 2 < 3`          | 1     |
| validation | `hidden assertions 1 < 3`          | 90    |

No sample reached the real `verify_sample` call (compile + pytest + ruff)
because the assertion-count hard check runs first.

### F-2 (Expected): Cross-split task_id overlap is empty

Confirms MBPP's documented disjoint splits: train [601-974], validation
[511-600], test [11-510].  No remediation needed for the dedup audit
(Task 5).

### F-3 (Process): Python 3.8 is incompatible with `datasets>=3.x`

`scripts/import_mbpp.py` requires `datasets` which raised
`TypeError: must be called with a dataclass type or instance` on Python 3.8
(the system's default `python`).  Switched to the project's pinned
`Python 3.10.9` (`pyproject.toml:requires-python = ">=3.10,<3.11"`).
The script + tests were validated against this interpreter.  All 143
schema / import / audit tests pass.

## Network / Timeout Issues

- **Network**: HuggingFace download succeeded on the first try once the
  user's proxy was set via `$env:HTTP_PROXY` / `$env:HTTPS_PROXY`.
  No retries needed.
- **Timeouts**: zero.  Verification of all 964 samples completed in
  under 90 seconds because the assertion-count hard check short-circuits
  before the real `pytest` call -- so the slow path (compile + pytest +
  ruff per sample) was never reached.  If the hidden-test shortage is
  fixed (e.g. by augmenting tests), verification is expected to take
  15-45 minutes per the original brief.

## Files Touched

- New: `scripts/generate_source_audit.py`
- New: `tests/test_generate_source_audit.py`
- New: `reports/p3/mbpp-source-audit.json`
- New: `data/external/mbpp/manifest.train.json`
- New: `data/external/mbpp/manifest.test.json`
- New: `data/external/mbpp/manifest.validation.json`
- New: `data/external/mbpp/manifest.index.json`
- New: `data/external/mbpp/normalized/test.jsonl`
- New: `data/external/mbpp/normalized/validation.jsonl`
- Modified: `data/external/mbpp/normalized/train.jsonl` (re-imported in P3
  format: `verified=false`, all-false Verification, `source_split="train"`)
- New: `data/external/mbpp/rejected/train.jsonl`
- New: `data/external/mbpp/rejected/test.jsonl`
- New: `data/external/mbpp/rejected/validation.jsonl`
- Untouched: `scripts/import_mbpp.py`, `scripts/verify_imported_mbpp.py`,
  `src/` files (per Task 3 constraints)
- Untouched: old `data/external/mbpp/manifest.json` (P2 artifact, kept as-is)

## Test Summary

```
$ python -m pytest tests/test_generate_source_audit.py tests/test_import_mbpp.py \
                   tests/test_import_mbpp_p3.py tests/test_schemas.py -v
============================= 143 passed in 7.81s ==============================
```

All 31 new audit-script tests pass; no regressions in importer / verifier /
schema tests.

## Recommendation for Next Step

Per the P3 plan, **stop and escalate** because the audit conclusion is
INFEASIBLE.  Before Task 5 (cross-split dedup), one of these remediations
is required:

1. **Augment hidden tests** (matches the P3 plan's Task 8 spec: "generate
   boundary tests if insufficient, verify with Reference Code").  Generate
   additional hidden tests from the reference code so each sample has
   `public >= 2 AND hidden >= 3`, then re-run `verify_imported_mbpp.py`.
2. **Loosen the hard gate** for the import-time verifier and defer the
   `hidden >= 3` enforcement to the Frozen v3 build step (Task 8).  This
   keeps the imported sample pool intact but moves the strict gate
   downstream.

Either path is a Task 3 / P3-plan revision, NOT a Task 4 deliverable, so
Task 4 itself is complete.
