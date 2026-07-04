# Task 13 Brief: Training Config + 3-Tier Checkpoint Evaluator

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Tasks 1-12 complete)
- Plan: .superpowers/sdd/p3-plan.md (Global Constraint #2, #3, #13, #14; Amendment A11 — 3-tier checkpoint)
- Task 11 produced `data/p3-curriculum/balanced-generalist/` (626 train + 90 val)
- Task 12 produced `data/p3-curriculum/repair-specialist/` (493 train + 90 val)
- Both candidates share the SAME 90 validation samples (Global Constraint #18, verified byte-identical)
- Hardware: RTX 3050 4GB VRAM (Global Constraint #2)
- Model: Qwen3-0.6B
- Existing P2 configs in `configs/curriculum/p2-stage3-repair-v3-antiforget.yaml` (reference for schema style)
- Existing `src/metrics.py` with `EvalOutcome`, `summarize()`, `pass_at_1()`, `repair_success_rate()`, `regression_rate()`

## Goal

Build TWO config files and ONE evaluator module + tests:

1. `configs/p3/balanced-generalist.yaml` — Training config for Balanced Generalist candidate
2. `configs/p3/repair-specialist.yaml` — Training config for Repair Specialist candidate
3. `src/p3_checkpoint_evaluator.py` — 3-Tier Checkpoint Evaluator module
4. `tests/test_p3_checkpoint_evaluator.py` — Test suite (minimum 10 tests)

**NO actual training is launched in this task.** This task only builds the config + evaluator infrastructure. Training launch is deferred until after Task 14 Readiness Gate passes.

## Part A: Config files

### A.1 configs/p3/balanced-generalist.yaml

Reference `configs/curriculum/p2-stage3-repair-v3-antiforget.yaml` for schema style. Key differences from P2:

```yaml
# P3 Balanced Generalist — Issue #9 §6.1
# Train: 626 samples (30/20/20/30 code/boundary/static/exec)
# Validation: 90 samples (code-only, shared with Repair Specialist)
training_mode: continual
model_name_or_path: models/Qwen3-0.6B
initial_adapter: adapters/p2/continual/stage3-repair-v3-antiforget  # Continue from P2 final

train_file: data/p3-curriculum/balanced-generalist/train.jsonl
eval_file: data/p3-curriculum/balanced-generalist/validation.jsonl
output_dir: adapters/p3/balanced-generalist
dataset_manifest: data/p3-curriculum/balanced-generalist/manifest.json
frozen_eval_file: data/frozen-eval/v3/test_raw.jsonl
frozen_eval_manifest: data/frozen-eval/v3/manifest.json

# RTX 3050 4GB constraints
max_seq_length: 384
num_train_epochs: 3
learning_rate: 0.00002  # 2e-5 (between P2 v2 3e-5 and v3-antiforget 1e-5)
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.0
max_grad_norm: 0.3

per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8

# BF16 with explicit FP16 fallback (Amendment A11 — no silent fallback)
bf16:
  enabled: true
  runtime_check: true  # Call torch.cuda.is_bf16_supported(); if False, fall back to fp16 and log warning
fp16:
  enabled: false  # Only used if bf16 unsupported
gradient_checkpointing: true
use_cache: false

# Tier 1 monitoring (Amendment A11 — every 25-50 steps)
logging_steps: 50  # Log train_loss, eval_loss, lr, gpu_mem, nan/inf check every 50 steps
save_steps: 100
eval_steps: 50  # Tier 1 eval (loss only, not full validation)

save_total_limit: 5

assistant_only_loss: true
truncation_policy: preserve_assistant

# LoRA config (Issue #9 §6.1 — rank=16, alpha=32, dropout=0.05, 7 modules)
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

seed: 42
dataloader_num_workers: 0
report_to: none

# 3-Tier Checkpoint Evaluator (Amendment A11)
checkpoint_evaluator:
  tier1:
    interval_steps: 50
    metrics: [train_loss, eval_loss, lr, gpu_mem_mb, nan_inf_detected]
  tier2:
    interval_epoch_fraction: 0.25  # Every 0.25 epoch
    probe_size: 75  # 60-90 family-stratified probe (use 75 as midpoint)
    probe_stratify_by: variant_type
    probe_seed: 42
    composite_score: true
  tier3:
    interval_epochs: 1  # Every 1 epoch
    full_validation: true
    composite_score: true

# Composite Score formula (Issue #9 §6.5 — Balanced Generalist)
# Match training ratio 30/20/20/30
composite_score:
  code_generation_pass_at_1: 0.30
  boundary_pass_at_1: 0.20  # variant_type="boundary", task_type="code_generation"
  static_repair_success: 0.20
  execution_repair_success: 0.30
  # Hard constraint: code_generation drop vs P2 final <= 3.0 percentage points
  hard_constraint:
    code_generation_drop_vs_p2_final_max_pct: 3.0

# Early stopping (Issue #9 §6.6)
early_stopping:
  enabled: true
  # Probe signal triggers, full validation confirms
  probe_patience: 4  # 4 consecutive probe drops (1.0 epoch)
  probe_min_delta: 0.005
  full_validation_confirm: true  # Must confirm with Tier 3 full validation
  divergence_nan_inf: true  # Immediate stop on NaN/Inf
  max_epochs: 3  # Hard cap

# Best checkpoint selection (Amendment A11 — by full Validation Composite only)
best_checkpoint:
  selection_metric: full_validation_composite
  never_use: [frozen_v3, probe]  # Best checkpoint NEVER selected by frozen v3 or probe
```

### A.2 configs/p3/repair-specialist.yaml

Identical structure to balanced-generalist.yaml EXCEPT:
- `train_file: data/p3-curriculum/repair-specialist/train.jsonl`
- `eval_file: data/p3-curriculum/repair-specialist/validation.jsonl`
- `output_dir: adapters/p3/repair-specialist`
- `dataset_manifest: data/p3-curriculum/repair-specialist/manifest.json`
- `composite_score` weights match 15/15/30/40:
  - `code_generation_pass_at_1: 0.15`
  - `boundary_pass_at_1: 0.15`
  - `static_repair_success: 0.30`
  - `execution_repair_success: 0.40`
- All other fields identical (LoRA, batch, lr, tiers, early stopping, best checkpoint policy)

## Part B: src/p3_checkpoint_evaluator.py

Module implementing the 3-Tier Checkpoint Evaluator. Key classes/functions:

### B.1 `CompositeScore` (dataclass)
```python
@dataclass
class CompositeScore:
    """Composite score for a checkpoint, weighted by candidate type."""
    code_generation_pass_at_1: float  # [0, 1]
    boundary_pass_at_1: float  # [0, 1] — variant_type="boundary"
    static_repair_success: float  # [0, 1]
    execution_repair_success: float  # [0, 1]

    def compute(self, weights: dict[str, float]) -> float:
        """Weighted sum. weights keys: code_generation_pass_at_1,
        boundary_pass_at_1, static_repair_success, execution_repair_success.
        Returns float in [0, 1]."""
```

### B.2 `ProbeResult` (dataclass)
```python
@dataclass
class ProbeResult:
    """Result of a Tier 2 probe evaluation."""
    step: int
    epoch: float
    probe_sample_ids: list[str]
    composite_score: CompositeScore
    composite_value: float
    metrics: dict  # raw metrics dict from src.metrics.summarize
```

### B.3 `FullValidationResult` (dataclass)
```python
@dataclass
class FullValidationResult:
    """Result of a Tier 3 full validation."""
    step: int
    epoch: int
    composite_score: CompositeScore
    composite_value: float
    metrics: dict  # raw metrics dict
    hard_constraint_pass: bool
    hard_constraint_violations: list[str]
```

### B.4 `CheckpointEvaluator` class
```python
class CheckpointEvaluator:
    """3-Tier Checkpoint Evaluator per Amendment A11.

    Tier 1 (every 25-50 steps): train_loss, eval_loss, lr, gpu_mem, nan/inf
    Tier 2 (every 0.25 epoch): 60-90 family-stratified probe + Composite
    Tier 3 (every 1 epoch): full validation + Composite

    Early stop: probe signal triggers, full validation confirms.
    Best checkpoint: by full Validation Composite only (never frozen v3, never probe).
    """

    def __init__(self, config: dict, total_train_samples: int):
        """config is the loaded YAML as a dict."""
        ...

    def should_run_tier1(self, step: int) -> bool:
        """True if step is a multiple of tier1.interval_steps."""

    def should_run_tier2(self, epoch: float) -> bool:
        """True if epoch crosses a 0.25 boundary (0.25, 0.5, 0.75, 1.0, ...)."""

    def should_run_tier3(self, epoch: int) -> bool:
        """True if epoch is an integer >= 1."""

    def select_probe_samples(self, validation_samples: list, seed: int = 42) -> list:
        """Select 60-90 family-stratified probe samples.
        Stratify by variant_type. Fixed seed for reproducibility."""

    def compute_composite(self, metrics: dict, weights: dict) -> CompositeScore:
        """Build CompositeScore from src.metrics.summarize() output.
        NOTE: boundary_pass_at_1 requires filtering by variant_type='boundary'
        in the eval outcomes — the evaluator must track variant_type per sample."""

    def check_hard_constraint(self, metrics: dict, baseline: dict) -> tuple[bool, list[str]]:
        """Check code_generation_drop_vs_p2_final_max_pct.
        baseline is from reports/p3/p3-baseline-lock.json."""

    def update_best_checkpoint(self, full_result: FullValidationResult) -> bool:
        """Returns True if this full validation is the new best.
        Best = highest full_validation_composite. Never uses probe or frozen_v3."""

    def check_early_stop(self, probe_history: list[ProbeResult],
                         full_history: list[FullValidationResult],
                         nan_inf_detected: bool) -> tuple[bool, str]:
        """Returns (should_stop, reason).
        Triggers:
        1. NaN/Inf detected → immediate stop
        2. probe_patience consecutive probe drops → trigger, but require full_validation_confirm
        3. max_epochs reached → hard cap
        """
```

### B.5 BF16 runtime check function
```python
def check_bf16_support() -> tuple[bool, str]:
    """Check if BF16 is supported on the current GPU.
    Returns (supported, message).
    Uses torch.cuda.is_bf16_supported().
    If unsupported, returns (False, 'BF16 not supported, falling back to FP16').
    NEVER silently falls back — always logs/returns the message."""
```

### B.6 Composite Score computation details

The evaluator must handle the fact that `src/metrics.py::summarize()` does NOT
break down by `variant_type`. The evaluator must:
1. Track `variant_type` per validation sample (from the Sample schema)
2. Compute `pass_at_1` separately for:
   - `variant_type="code"` samples → `code_generation_pass_at_1`
   - `variant_type="boundary"` samples → `boundary_pass_at_1`
3. Compute `repair_success_rate` for:
   - `variant_type="static_repair"` → `static_repair_success`
   - `variant_type="execution_repair"` → `execution_repair_success`
4. If a bucket has 0 samples, the rate is 0.0 (not vacuous true)

**IMPORTANT**: The 90 P3 validation samples are code-only (no variants). So during
Tier 3 full validation, only `code_generation_pass_at_1` will be non-zero. The
other 3 components will be 0.0. This means the Composite Score on validation
will be dominated by `code_generation_pass_at_1 × weight`.

For Tier 2 probe, the probe samples ARE stratified by variant_type (selected
from the training data, not validation), so all 4 components can be non-zero.
This is the design intent: probe measures training-time capability, full
validation measures held-out generalization.

### B.7 Probe sample selection

`select_probe_samples` must:
1. Take the TRAINING samples (not validation) as input
2. Stratify by `variant_type` (4 buckets)
3. For each bucket, sample `min(bucket_size, probe_size // 4)` samples
4. If `probe_size=75`, each bucket gets ~18-19 samples
5. Use `random.Random(42)` for determinism
6. Return the selected samples (75 total, ±1 for rounding)

## Part C: Tests (minimum 10 tests)

Create `tests/test_p3_checkpoint_evaluator.py`:

1. `test_config_schema_balanced` — balanced-generalist.yaml has all required fields
2. `test_config_schema_repair` — repair-specialist.yaml has all required fields
3. `test_config_ratio_weights_balanced` — composite_score weights sum to 1.0 and match 30/20/20/30
4. `test_config_ratio_weights_repair` — composite_score weights sum to 1.0 and match 15/15/30/40
5. `test_bf16_check_returns_bool_and_message` — check_bf16_support returns (bool, str)
6. `test_composite_score_compute` — CompositeScore.compute() returns weighted sum in [0, 1]
7. `test_tier_scheduling` — should_run_tier1/2/3 return correct booleans for given steps/epochs
8. `test_probe_sample_selection` — select_probe_samples returns 75±1 samples, stratified by variant_type
9. `test_best_checkpoint_only_uses_full_validation` — update_best_checkpoint ignores probe results
10. `test_early_stop_nan_inf` — NaN/Inf triggers immediate stop
11. `test_early_stop_probe_patience` — 4 consecutive probe drops triggers stop (with full validation confirm)
12. `test_hard_constraint_check` — check_hard_constraint correctly detects code_generation drop > 3pp

## Hard gates (binding — abort exit 1 if any fail)

1. Both YAML configs exist and parse without error
2. Both configs have `lora.rank=16, lora.alpha=32, lora.dropout=0.05, lora.target_modules` with 7 modules
3. Both configs have `bf16.runtime_check: true`
4. Both configs have `checkpoint_evaluator` with tier1/tier2/tier3 sections
5. Both configs have `composite_score` with 4 weights summing to 1.0
6. Both configs have `early_stopping.enabled: true` with `full_validation_confirm: true`
7. Both configs have `best_checkpoint.selection_metric: full_validation_composite`
8. Both configs have `best_checkpoint.never_use: [frozen_v3, probe]`
9. Balanced weights match 30/20/20/30 (±0.01)
10. Repair weights match 15/15/30/40 (±0.01)
11. `src/p3_checkpoint_evaluator.py` imports without error
12. All tests pass

## Existing infrastructure (use these)
1. `src/metrics.py` — EvalOutcome, summarize(), pass_at_1(), repair_success_rate()
2. `src/schemas.py` — Sample class (variant_type field)
3. `configs/curriculum/p2-stage3-repair-v3-antiforget.yaml` — reference config schema
4. `reports/p3/p3-baseline-lock.json` — P2 baseline metrics (for hard_constraint check)

## Important notes
- Use `from __future__ import annotations` at top of all .py files
- The `configs/p3/` directory needs to be created
- DO NOT launch any training in this task. Only build config + evaluator + tests.
- DO NOT modify `src/metrics.py`, `src/schemas.py`, or any Task 1-12 files.
- Python 3.8.10 is the active interpreter.
- The evaluator is a MODULE (importable), not a script. No `if __name__ == "__main__"` block needed.
- Use `pyyaml` for YAML parsing (already a dependency per P2 configs).
- For `check_bf16_support`: import torch lazily inside the function (so the module imports even without torch).
- The probe samples come from TRAINING data (not validation) — this is intentional per the design.
- All Composite Score weights must sum to 1.0 (within 0.01 tolerance for float).

## Commit message
`feat(p3): training config + 3-tier checkpoint evaluator (2 candidates, 12 tests)`

## Deviations / clarifications

1. **No actual training launched**. This task only builds infrastructure.
   Training is deferred until Task 14 Readiness Gate passes (GO verdict).

2. **Validation is code-only**. The 90 P3 validation samples have
   `variant_type="code"` only. So Tier 3 full validation Composite will be
   dominated by `code_generation_pass_at_1 × weight`. The other 3 components
   will be 0.0 (no boundary/static/exec samples in validation). This is the
   design intent — validation measures held-out generalization on code gen,
   while probe measures training-time capability across all 4 variant types.

3. **Probe uses training samples**. Tier 2 probe samples come from the
   training set (not validation), stratified by variant_type. This is the
   design intent — probe measures training-time capability.

4. **best_checkpoint never uses frozen_v3 or probe**. Per Amendment A11,
   best checkpoint is selected ONLY by full Validation Composite. frozen_v3
   is a separate eval (held out, used for contamination checks, not for
   checkpoint selection).

5. **early_stop requires full_validation_confirm**. A probe signal (4
   consecutive drops) triggers a CHECK, but the stop only happens if the
   next Tier 3 full validation confirms the degradation. This prevents
   premature stopping from probe noise.

6. **Composite weights match training ratios**. Balanced: 30/20/20/30.
   Repair: 15/15/30/40. This aligns the optimization objective with the
   curriculum design intent.
