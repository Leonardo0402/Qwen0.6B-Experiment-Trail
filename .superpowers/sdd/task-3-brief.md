# Task 3 Brief: Fix import_mbpp.py + New verify_imported_mbpp.py

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Task 1 @ 48614af, Task 2 @ 5b88a6e complete)
- Plan file: .superpowers/sdd/p3-plan.md (Global Constraints apply)
- Schema: src/schemas.py now has `variant_type`, `bug_type`, `source_split` Optional[str] fields (Task 2)
- Existing importer: scripts/import_mbpp.py (currently writes verified=true preset — BUG)
- Existing validator: src/validators.py has `verify_sample()` and `verify_broken_is_broken()` (real execution)

## Goal (two parts)

### Part A: Fix import_mbpp.py
1. **Remove verified=true preset**: imported samples default to `verified=False` with `Verification(syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False)` — no validation claims until real verification runs.
2. **Per-split manifest**: change from single `manifest.json` (overwritten) to:
   - `manifest.{split}.json` — per-split manifest (one file per split, never overwritten by other splits)
   - `manifest.index.json` — index of all imported splits (updated/merged on each import)
3. **Pin HF source revision**: pass `revision` param to `datasets.load_dataset()` if available; record `source_revision` and `dataset_fingerprint` in manifest.
4. **Benchmark contamination flag**: add `benchmark_contaminated` and `standard_mbpp_test_claims_disallowed` fields to manifest (both `true` for test split, `false` for train/validation splits).
5. **Fill source_split field**: set `source_split` on each Sample (the new optional field from Task 2) to the split name being imported.
6. **Normalized subdirectory**: keep `normalized/{split}.jsonl` (already correct).
7. **No pytest in importer**: importer only downloads + normalizes. Do NOT call verify_sample or pytest.

### Part B: New scripts/verify_imported_mbpp.py
A standalone verifier that reads normalized JSONL, runs real validation, writes verified/rejected output.

1. **Input**: `data/external/mbpp/normalized/{split}.jsonl`
2. **Process per sample**:
   - Call `src.validators.verify_sample(sample)` (real compile + pytest + ruff)
   - Hard check: count public assertions >= 2 (count `assert ` occurrences in public_tests)
   - Hard check: count hidden assertions >= 3 (count `assert ` occurrences in hidden_tests)
   - For repair samples (task_type in static_repair/execution_repair): call `verify_broken_is_broken(sample)` and confirm broken_code fails at least one test
   - For execution_repair: confirm `execution_feedback` is non-empty and appears to be real failure output (contains "Error", "assert", "Traceback", "FAILED", or similar failure marker — use a reasonable heuristic, do NOT require exact match)
3. **Output**:
   - `data/external/mbpp/verified/{split}.jsonl` — samples that pass ALL checks, with `verified=True` and `verification` updated to real results
   - `data/external/mbpp/rejected/{split}.jsonl` — samples that fail any check, with `verified=False`, `verification` real results, and `rejection_reason` field added
   - Update `manifest.{split}.json` with: `verified_sha256`, `verified_count`, `rejected_count`, `rejected_sha256`, `verified_at`
4. **CLI**: `python scripts/verify_imported_mbpp.py --split test --output-dir data/external/mbpp`
5. **Exit codes**: 0 success, 1 all rejected, 2 partial rejection (still 0, but report counts)

## Constraints (binding)

1. **Import + verify split**: importer does NOT run pytest. Verifier does NOT download. Clean separation.
2. **Default verified=false in importer**: `_VERIFIED_VER` constant must be removed or changed to all-false.
3. **Per-split manifest non-overwrite**: running `--split test` then `--split validation` must NOT overwrite each other's manifests. Only `manifest.index.json` is shared and merged.
4. **HF revision pin**: use `revision="main"` explicitly (or a specific commit hash if discoverable). Record whatever `datasets` reports as fingerprint. If revision/fingerprint unavailable, record null and note in manifest — do NOT fabricate.
5. **Benchmark contamination**: only test split gets `benchmark_contaminated=true` (because P3 will use it for training). train/validation splits get `false`.
6. **source_split field**: every imported Sample gets `source_split=<split_name>`.
7. **Hard assertion counts**: verifier counts `assert ` literal occurrences (case-sensitive) in public_tests and hidden_tests strings. public < 2 → reject. hidden < 3 → reject. (Count via `public_tests.count("assert ")` — simple and deterministic.)
8. **verify_broken_is_broken**: for repair samples, must call this and require True. If False, reject with reason "broken_code passes all tests".
9. **execution_feedback**: for execution_repair samples, must be non-empty and contain a failure marker. Heuristic: any of ["Error", "assert", "Traceback", "FAILED", "Exception", "fail"] (case-insensitive). If none present, reject with reason "execution_feedback lacks failure marker".
10. **Surgical change to importer**: keep conversion helpers (extract_skill_tags, infer_difficulty, split_mbpp_tests, mbpp_record_to_sample) — only modify verification preset, manifest structure, and add source_split field.
11. **Do NOT modify src/validators.py or src/schemas.py** — they are correct as-is.

## Manifest Schema (per-split: manifest.{split}.json)
```json
{
  "source": "google-research-datasets/mbpp",
  "source_revision": "<commit_hash_or_main>",
  "dataset_fingerprint": "<hf_fingerprint_or_null>",
  "split": "test",
  "sample_count": 500,
  "normalized_sha256": "<sha256_of_normalized_jsonl>",
  "normalized_file": "normalized/test.jsonl",
  "license": "Apache-2.0",
  "imported_at": "<iso8601>",
  "benchmark_contaminated": true,
  "standard_mbpp_test_claims_disallowed": true,
  "verified_sha256": "<filled_by_verifier>",
  "verified_count": "<filled_by_verifier>",
  "rejected_count": "<filled_by_verifier>",
  "rejected_sha256": "<filled_by_verifier>",
  "verified_at": "<filled_by_verifier>"
}
```

## Manifest Index (manifest.index.json)
```json
{
  "source": "google-research-datasets/mbpp",
  "splits": ["train", "test", "validation"],
  "updated_at": "<iso8601>",
  "splits_detail": {
    "train": {"sample_count": 374, "normalized_sha256": "...", "manifest": "manifest.train.json"},
    "test": {"sample_count": 500, "normalized_sha256": "...", "manifest": "manifest.test.json"},
    "validation": {"sample_count": 90, "normalized_sha256": "...", "manifest": "manifest.validation.json"}
  }
}
```

## Tests (tests/test_import_mbpp_p3.py)

### Importer tests (can mock datasets to avoid network)
1. `test_imported_samples_default_verified_false`: imported Sample has verified=False, verification all False
2. `test_per_split_manifest_no_overwrite`: simulate importing train then test — both manifest.train.json and manifest.test.json exist with correct content, manifest.index.json lists both
3. `test_benchmark_contamination_flag`: test split manifest has benchmark_contaminated=true, train split has false
4. `test_source_split_field_set`: imported Sample has source_split equal to split name
5. `test_no_pytest_in_importer`: verify importer does not call verify_sample or run_pytest (can check by mocking or by code inspection — prefer a unit test that mocks datasets and checks output has verified=false)

### Verifier tests (use synthetic samples, no network)
6. `test_verified_sample_passes_all_checks`: sample with valid code, public>=2, hidden>=3 → written to verified/{split}.jsonl with verified=true
7. `test_rejected_low_public_count`: sample with only 1 assert in public → rejected with reason
8. `test_rejected_low_hidden_count`: sample with only 2 asserts in hidden → rejected with reason
9. `test_repair_sample_verify_broken_is_broken`: static_repair sample where broken_code passes → rejected
10. `test_execution_repair_feedback_check`: execution_repair with empty/non-failure execution_feedback → rejected
11. `test_manifest_updated_with_verified_fields`: after verification, manifest.{split}.json has verified_sha256, verified_count, rejected_count filled

## Report File
Write your full report to: `.superpowers/sdd/task-3-report.md`
Return only: status (DONE/DONE_WITH_CONCERNS/BLOCKED/NEEDS_CONTEXT), commit hash, one-line test summary, concerns.

## Commit
- Stage: `scripts/import_mbpp.py`, `scripts/verify_imported_mbpp.py`, `tests/test_import_mbpp_p3.py`
- Commit message: `fix(p3): split import/verify pipeline, per-split manifest, real verification`
- Single commit.

## Working Directory
e:\agent\Qwen\qwen3-code-lab

## Test Verification
`cd e:\agent\Qwen\qwen3-code-lab ; python -m pytest tests/test_import_mbpp_p3.py -v`

All tests must pass. For tests that need to mock `datasets.load_dataset`, use `unittest.mock.patch` — do NOT hit network in tests.

## Implementation Hints
- For HF revision: `datasets.load_dataset(repo, split=split, revision="main")` — "main" is always valid. For fingerprint, after loading, check `ds._info` or `ds.info` for fingerprint attribute; if not found, record null.
- For assertion counting: `public_tests.count("assert ")` — simple and works for MBPP-style tests.
- For verifier CLI: mirror import_mbpp.py's argparse structure.
- Mock datasets in tests by patching `scripts.import_mbpp._load_dataset` to return a synthetic list of dicts.
- For verifier tests, use the existing `_base_sample()` helper pattern from tests/test_schemas.py (or create minimal synthetic Samples inline).

## Global Constraints (from .superpowers/sdd/p3-plan.md)
- import + verify split: importer only normalizes (verified=false); verify_imported_mbpp.py runs real validation
- per-split manifest: manifest.{split}.json + manifest.index.json
- HF pin source revision + fingerprint + normalized SHA + verified SHA
- Benchmark contamination: test split → benchmark_contaminated=true
- Hard checks: public>=2, hidden>=3; verify_broken_is_broken; real execution_feedback
- Do NOT modify src/validators.py or src/schemas.py
