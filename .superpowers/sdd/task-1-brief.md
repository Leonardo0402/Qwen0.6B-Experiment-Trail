# Task 1 Brief: Lock Historical Baseline

## Context
- Project: e:\agent\Qwen\qwen3-code-lab (Qwen3-0.6B code training lab)
- Branch: feat/p3-capability-expansion-v2 (just created from main @ 42a489c)
- This is the FIRST task of P3 Capability Expansion v2 (Issue #9)
- Per user directive: baseline lock must be created BEFORE any data build or training config change
- Plan file: .superpowers/sdd/p3-plan.md (Global Constraints apply)

## Goal
Create `reports/p3/p3-baseline-lock.json` recording immutable historical baselines for 3 models:
1. Base Qwen3-0.6B (the foundation model, no adapter)
2. Stage3-Independent (P2 repair baseline)
3. Stage3-v3-Antiforget (P2 balanced candidate)

## Required Fields (per model)
- `model_name`: human-readable name (e.g. "Base Qwen3-0.6B", "Stage3-Independent", "Stage3-v3-Antiforget")
- `adapter_path`: relative path from repo root (e.g. "models/Qwen3-0.6B" for Base, "adapters/code-lora-v3-stage3-independent" for adapter)
- `weight_sha256`: SHA256 of adapter weights file (adapter_model.safetensors or similar). For Base model, use the config SHA256 or a sentinel like "BASE_MODEL_NO_ADAPTER" if no adapter weights exist locally (DO NOT compute SHA of the full base model — it's too large; record the model dir path and note "base model, no adapter weights")
- `config_sha256`: SHA256 of adapter_config.json (for adapters) or model config.json (for Base)
- `training_config_sha256`: SHA256 of the training config YAML used to produce this adapter (look in configs/ directory). For Base, use "BASE_MODEL_NO_TRAINING_CONFIG"
- `historical_eval_set_sha256`: SHA256 of frozen-eval-v2 test_raw.jsonl (from data/p2-curriculum/frozen-eval-v2/manifest.json field `test_sha256`)
- `historical_held_out_metrics`: dict of metrics from reports/p2/ for this model on frozen-eval-v2 (e.g. overall_pass, codegen_pass1, static_repair, execution_repair). Look in reports/p2/*.json for per-model eval results. If exact metrics not found, record what's available and note the source file.
- `created_at`: ISO 8601 timestamp

## Top-level fields
- `issue`: 9
- `branch`: "feat/p3-capability-expansion-v2"
- `purpose`: "Lock P2 historical baselines for P3 same-config comparison. Must not be modified after P3 training starts."
- `frozen_eval_v2_manifest`: path to the v2 manifest
- `frozen_eval_v2_sha256`: from v2 manifest
- `created_at`: ISO 8601 timestamp
- `models`: list of 3 model records above

## Discovery Steps (implementer should do)
1. Find adapter directories: `adapters/` likely has code-lora-v3-* subdirs. Use Glob to find.
2. For Base model: path is likely `models/Qwen3-0.6B` (check if exists; if not, note path from configs).
3. Find training configs: `configs/` directory, look for stage3-independent and stage3-v3-antiforget configs.
4. Read data/p2-curriculum/frozen-eval-v2/manifest.json to get test_sha256.
5. Find historical metrics: Glob reports/p2/*.json, read files to extract per-model metrics. Key files likely: full576-comparison.json, router-analysis.json, or per-model eval JSONs.
6. Compute SHA256 of adapter_model.safetensors (or .bin) and adapter_config.json for each adapter.
7. Compute SHA256 of training config YAMLs.

## Tests (tests/test_p3_baseline_lock.py)
- Test file loads reports/p3/p3-baseline-lock.json
- Test: 3 models present with correct names
- Test: each model has all required fields (non-empty)
- Test: weight_sha256 is 64-char hex string OR "BASE_MODEL_NO_ADAPTER" sentinel
- Test: config_sha256 is 64-char hex string
- Test: historical_eval_set_sha256 matches data/p2-curriculum/frozen-eval-v2/manifest.json::test_sha256
- Test: created_at is valid ISO 8601

## Constraints
- DO NOT modify any existing files outside reports/p3/ and tests/
- DO NOT compute SHA of base model weights (too large, ~600MB-1GB). Use sentinel "BASE_MODEL_NO_ADAPTER" for weight_sha256 of Base, and use config.json SHA for config_sha256.
- DO NOT run training or evaluation. This is a read-only lock task.
- File must be valid JSON (pretty-printed, 2-space indent).
- Use Python's hashlib for SHA256 computation.
- For adapter weight files: compute SHA256 by reading in chunks (8192 bytes) to handle large files.

## Report File
Write your full report to: .superpowers/sdd/task-1-report.md
Return only: status (DONE/DONE_WITH_CONCERNS/BLOCKED/NEEDS_CONTEXT), commit hash, one-line test summary, concerns.

## Commit
- Stage: reports/p3/p3-baseline-lock.json, tests/test_p3_baseline_lock.py
- Commit message: "feat(p3): lock historical baselines for P3 comparison"
- Single commit.

## Working Directory
e:\agent\Qwen\qwen3-code-lab

## Git Branch
feat/p3-capability-expansion-v2 (already checked out)
