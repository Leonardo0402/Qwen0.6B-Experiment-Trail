# Task 1 Report: Lock Historical Baselines for P3 Comparison

**Status:** DONE
**Branch:** feat/p3-capability-expansion-v2
**Issue:** #9
**Date:** 2026-07-04

## 1. Goal

Create `reports/p3/p3-baseline-lock.json` recording immutable historical
baselines for 3 P2 models so that P3 candidates can be compared against
them under identical configuration:

1. Base Qwen3-0.6B (foundation, no adapter)
2. Stage3-Independent (P2 repair baseline, independent from Base)
3. Stage3-v3-Antiforget (P2 balanced candidate, continual from Stage2-v2)

Plus a test file `tests/test_p3_baseline_lock.py` enforcing the lock's
contract.

## 2. Discovery Findings

### 2.1 Adapter directories (final, after verifying against training configs)

| Model | adapter_path | Training config |
| --- | --- | --- |
| Base Qwen3-0.6B | `models/Qwen3-0.6B` | (none) |
| Stage3-Independent | `adapters/p2/independent/stage3-repair-v2` | `configs/curriculum/p2-stage3-repair-independent.yaml` |
| Stage3-v3-Antiforget | `adapters/p2/continual/stage3-repair-v3` | `configs/curriculum/p2-stage3-repair-v3-antiforget.yaml` |

Cross-validation: the `output_dir` field of each YAML config matches the
adapter path recorded above, so the (adapter, training config) pairings
are correct.

### 2.2 Frozen-eval-v2 manifest

- File: `data/p2-curriculum/frozen-eval-v2/manifest.json`
- `test_sha256`: `748ea3a467876ef62c17d65006b4950477956a76a29379c0ad2ff0c71d897df3`
- 576 test samples / 75 families; train/validation slots = 0 (eval-only)
- Same value is also recorded in `reports/p2/router-policy-v1.json::selection_dataset_sha256`.

### 2.3 Base model

- Directory exists: `models/Qwen3-0.6B/config.json` (726 bytes).
- Per the brief, weight_sha256 uses the sentinel `BASE_MODEL_NO_ADAPTER`
  (the base model weights are ~600 MB-1 GB and must NOT be hashed).
- config_sha256 = SHA256 of `models/Qwen3-0.6B/config.json`.
- training_config_sha256 = `BASE_MODEL_NO_TRAINING_CONFIG`.

### 2.4 Historical metrics source

Per-model metrics for the 3 baseline models were extracted from
`reports/p2/router-analysis.json::comparison_table`. This is the
canonical per-model metrics table produced by P2 router analysis.

Two complementary overall_pass values were recorded per model:

- `overall_pass` (from router-analysis.json) — computed on the
  router-policy-v1 eval subset (30 families / 234 samples), which is
  the family-disjoint held-out subset used for router selection.
- `full576_overall_pass` (from full576-paired-stats.json) — the rate
  on the full 576-sample frozen-eval-v2 set, derived from
  `pair_comparisons[*].sample_compare.rate_a` / `rate_b`.

The metrics fields recorded per model:
`overall_pass`, `codegen_pass1`, `static_repair`, `execution_repair`,
`family_pass`, plus `full576_overall_pass` and source_file pointers.

## 3. SHA256 Values (freshly computed with Python hashlib, 8192-byte chunks)

| File | SHA256 | Size (bytes) |
| --- | --- | --- |
| `models/Qwen3-0.6B/config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` | 726 |
| `adapters/p2/independent/stage3-repair-v2/adapter_model.safetensors` | `65e5c1f0aacff3d7e3203d90326ce6441823c3c3537fdaf0438ab12a5a8ea41a` | 40,422,168 |
| `adapters/p2/independent/stage3-repair-v2/adapter_config.json` | `6a7ca1fcf35fb647dff89b119b79c94d613b909e88da75147739e0a5c59e4653` | 1,144 |
| `configs/curriculum/p2-stage3-repair-independent.yaml` | `74e84c3e66a3385132395eef62bc096f61ec45984ea0f0f98227b160d10d3dc3` | 1,161 |
| `adapters/p2/continual/stage3-repair-v3/adapter_model.safetensors` | `768bc7b6de538678097eaeed3d729733cb2297ab395380f593d0d9816c4a4491` | 40,422,168 |
| `adapters/p2/continual/stage3-repair-v3/adapter_config.json` | `c50954b45bcf7f0fbebe21c23fdbd3399c18dc60ab0aee28ee406801c667bdfa` | 1,144 |
| `configs/curriculum/p2-stage3-repair-v3-antiforget.yaml` | `f6adbdf3f5c47e80b7390b22e50f75a915b70fa41465c80b6e169b927e9aaf31` | 1,927 |

### 3.1 Cross-check vs `reports/p2/adapter-evidence.json`

The 4 adapter-related SHA values computed freshly above exactly match
the values already recorded in `reports/p2/adapter-evidence.json`
(`independent-stage3` and `stage3-v3-antiforget` blocks), confirming the
adapter artifacts on disk are the same ones P2 used for evaluation.

The training_config_sha256 values computed freshly above DIFFER from
the values in `adapter-evidence.json`:
- Independent: fresh=`74e84c3e...`, evidence=`2f7800f4...`
- Antiforget: fresh=`f6adbdf3...`, evidence=`e604412b...`

This is expected: the YAML files were likely edited after training
(comments / formatting), and the brief mandates computing the SHA of the
file as it currently exists on disk. The fresh values are the canonical
ones for this lock. No action needed, but flagged here for transparency.

## 4. Extracted Metrics

Per-model metrics on the router-policy-v1 eval subset (234 samples /
30 families, family-disjoint from selection) from `router-analysis.json`:

| Model | overall_pass | codegen | static_repair | execution_repair | family_pass |
| --- | --- | --- | --- | --- | --- |
| Base Qwen3-0.6B | 0.36324786324786323 | 0.15517241379310345 | 0.4659090909090909 | 0.3977272727272727 | 0.0 |
| Stage3-Independent | 0.49572649572649574 | 0.15517241379310345 | 0.6136363636363636 | 0.6022727272727273 | 0.03333333333333333 |
| Stage3-v3-Antiforget | 0.4658119658119658 | 0.1724137931034483 | 0.5568181818181818 | 0.5681818181818182 | 0.03333333333333333 |

Full-576 overall pass rate (from `full576-paired-stats.json`):

| Model | full576_overall_pass |
| --- | --- |
| Base Qwen3-0.6B | 0.3611111111111111 (~208/576) |
| Stage3-Independent | 0.484375 (~279/576) |
| Stage3-v3-Antiforget | 0.4444444444444444 (~256/576) |

## 5. Files Created

- `reports/p3/p3-baseline-lock.json` — the lock (valid JSON, 2-space indent,
  3 model records with all required fields).
- `tests/test_p3_baseline_lock.py` — 26 test cases across 7 test classes.

## 6. Test Results

```
$ python -m pytest tests/test_p3_baseline_lock.py -v
============================= test session starts =============================
platform win32 -- Python 3.8.10, pytest-8.3.5, pluggy-1.5.0
collected 26 items

tests\test_p3_baseline_lock.py ..........................                [100%]

============================= 26 passed in 0.12s ==============================
```

Test coverage:
- Top-level fields (issue=9, branch, purpose, manifest path/sha, created_at)
- 3 models present, no duplicates, expected names
- All required per-model fields present and non-empty
- adapter_path is relative and resolves
- weight_sha256 = 64-char hex OR `BASE_MODEL_NO_ADAPTER` (Base only)
- config_sha256 = 64-char hex
- training_config_sha256 = 64-char hex OR `BASE_MODEL_NO_TRAINING_CONFIG` (Base only)
- historical_eval_set_sha256 = 64-char hex and matches manifest.test_sha256
- created_at valid ISO 8601 (top-level + per-model)
- historical_held_out_metrics non-empty dict with source_file existing and overall_pass in [0,1]

## 7. Constraints Compliance

- [x] Read-only task: no training/evaluation run.
- [x] No modifications outside `reports/p3/` and `tests/`.
- [x] Base model weights NOT hashed; sentinel used.
- [x] Valid JSON, 2-space indent.
- [x] SHA256 computed with Python hashlib, 8192-byte chunks.
- [x] No emojis in code, commit message, or lock file.

## 8. Concerns

None blocking. Two informational notes:

1. **Training config SHA divergence from adapter-evidence.json** —
   discussed in section 3.1. The fresh SHA of the YAML as it exists on
   disk is used (per the brief), and differs from the SHA recorded at
   training time in `adapter-evidence.json` (likely due to post-training
   YAML edits). This is documented, not a defect.
2. **Metrics granularity** — per-model metrics on the full 576-sample
   frozen-eval-v2 set are only available as overall_pass in
   `full576-paired-stats.json` (per-pair `rate_a`/`rate_b`); per-task-type
   breakdown on the full 576 set is NOT separately recorded in any P2
   report. The per-task-type metrics in this lock come from the
   router-policy-v1 eval subset (234 samples). This is cited per model
   in `historical_held_out_metrics.source_file`.

## 9. Commit

Staged files:
- `reports/p3/p3-baseline-lock.json`
- `tests/test_p3_baseline_lock.py`

Commit message: `feat(p3): lock historical baselines for P3 comparison`
