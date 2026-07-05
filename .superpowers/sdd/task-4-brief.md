# Task 4 Brief: Run Import + Verify + Source Audit

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Tasks 1-3 complete)
- Importer: scripts/import_mbpp.py (fixed in Task 3, writes verified=false)
- Verifier: scripts/verify_imported_mbpp.py (new in Task 3, runs real validation)
- Plan file: .superpowers/sdd/p3-plan.md

## Goal
1. Run import_mbpp.py for all 3 splits (test, validation, train-reimport)
2. Run verify_imported_mbpp.py for all 3 splits (real pytest validation)
3. Write scripts/generate_source_audit.py to produce reports/p3/mbpp-source-audit.json
4. Generate the audit report

## Execution Steps

### Step 1: Import all 3 splits
Run from `e:\agent\Qwen\qwen3-code-lab`:
```
python scripts/import_mbpp.py --split test --output-dir data/external/mbpp
python scripts/import_mbpp.py --split validation --output-dir data/external/mbpp
python scripts/import_mbpp.py --split train --output-dir data/external/mbpp
```

Expected outputs per split:
- `data/external/mbpp/normalized/{split}.jsonl`
- `data/external/mbpp/manifest.{split}.json`
- `data/external/mbpp/manifest.index.json` (merged)

**Network note**: If HuggingFace download fails (proxy/network issue), check if `HF_ENDPOINT` or `HTTP_PROXY`/`HTTPS_PROXY` env vars need setting. The user's proxy is at 127.0.0.1:7897. If import still fails after trying proxy, record as BLOCKED with the error message — do NOT fabricate data.

### Step 2: Verify all 3 splits
```
python scripts/verify_imported_mbpp.py --split test --output-dir data/external/mbpp
python scripts/verify_imported_mbpp.py --split validation --output-dir data/external/mbpp
python scripts/verify_imported_mbpp.py --split train --output-dir data/external/mbpp
```

Expected outputs per split:
- `data/external/mbpp/verified/{split}.jsonl`
- `data/external/mbpp/rejected/{split}.jsonl`
- Updated `manifest.{split}.json` (verified_sha256, verified_count, rejected_count, etc.)

**Performance note**: This runs real pytest on ~964 samples total (374 train + ~500 test + ~90 validation). Each sample runs compile + public pytest + hidden pytest + ruff. This may take 15-45 minutes. Use non-blocking execution if available. If a sample times out, it goes to rejected/ with timeout reason — that's expected behavior.

### Step 3: Write scripts/generate_source_audit.py
A script that reads all manifest files and generates `reports/p3/mbpp-source-audit.json`:

```json
{
  "generated_at": "<iso8601>",
  "source": "google-research-datasets/mbpp",
  "source_revision": "<from manifests>",
  "splits": {
    "train": {
      "sample_count": <int>,
      "task_id_range": {"min": <int>, "max": <int>},
      "missing_task_ids": [<int>, ...],
      "duplicate_task_ids": [<int>, ...],
      "normalized_sha256": "<hex>",
      "verified_sha256": "<hex>",
      "verified_count": <int>,
      "rejected_count": <int>,
      "benchmark_contaminated": false,
      "dataset_fingerprint": "<str_or_null>"
    },
    "test": { ... same structure ... },
    "validation": { ... same structure ... }
  },
  "cross_split_task_id_overlap": {
    "train_test": [<int>, ...],
    "train_validation": [<int>, ...],
    "test_validation": [<int>, ...]
  },
  "total_samples": <int>,
  "total_verified": <int>,
  "total_rejected": <int>,
  "new_families_available": <int>,
  "conclusion": "LIKELY_FEASIBLE" | "INFEASIBLE",
  "feasibility_notes": "<str>"
}
```

**Conclusion logic**:
- `LIKELY_FEASIBLE` if: test_split verified_count >= 240 (180 train + 60 val minimum) AND validation_split verified_count >= 0
- `INFEASIBLE` if: test_split verified_count < 240
- `new_families_available` = test verified_count + validation verified_count (these are the new families not used in P2)

**Task ID extraction**: parse from sample_id field (format: `mbpp_{task_id}`). Collect all task_ids per split, compute min/max/range, find missing (gaps in range) and duplicate (same task_id appearing twice).

**Cross-split overlap**: compute set intersections of task_ids across all 3 splits. Should be empty (MBPP splits are disjoint by construction), but verify and report any overlaps.

### Step 4: Generate audit report
```
python scripts/generate_source_audit.py --output-dir data/external/mbpp --report-dir reports/p3
```

## Tests (tests/test_generate_source_audit.py)
- Test with synthetic manifest files (create temp dir with mock manifest.{split}.json files)
- Test: correct extraction of sample_count, task_id_range, missing/duplicate IDs
- Test: cross_split_task_id_overlap computation (empty for disjoint, non-empty for overlapping)
- Test: conclusion logic (LIKELY_FEASIBLE when test>=240, INFEASIBLE when test<240)
- Test: new_families_available computation
- Do NOT hit network or run real import/verify in tests

## Constraints
- Do NOT modify scripts/import_mbpp.py or scripts/verify_imported_mbpp.py (Task 3 complete)
- Do NOT modify src/ files
- If import fails due to network: record error, still write the audit script and tests, return BLOCKED
- If verify is extremely slow (>60 min): you may verify just test + validation splits first, defer train re-verify (train was already imported in P2, though with old manifest format)
- The audit report must reflect ACTUAL data, not estimates

## Report File
Write to: `.superpowers/sdd/task-4-report.md`
Include: actual split counts, task_id ranges, verified/rejected counts per split, conclusion, any network/timeout issues.

Return: status, commit hash, one-line test summary, concerns.

## Commit
- Stage: `scripts/generate_source_audit.py`, `tests/test_generate_source_audit.py`, `reports/p3/mbpp-source-audit.json`, `data/external/mbpp/normalized/*.jsonl`, `data/external/mbpp/manifest.*.json`, `data/external/mbpp/verified/*.jsonl`, `data/external/mbpp/rejected/*.jsonl`
- Commit message: `feat(p3): import + verify MBPP splits, generate source audit report`
- Single commit.
- Note: data/external/mbpp/ files may be large — check .gitignore. If normalized/verified JSONL files are git-ignored, only commit scripts + tests + report + manifests.

## Working Directory
e:\agent\Qwen\qwen3-code-lab
