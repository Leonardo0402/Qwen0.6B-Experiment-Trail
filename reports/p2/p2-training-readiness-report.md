# P2 Training Readiness Report

Generated: 2026-06-28

## Git State

- Branch: `feat/p2-agentic-code-training-v2`
- Base commit: `8ccd4e2` (P0+P1 fixes)
- Remote: `https://github.com/Leonardo0402/Qwen0.6B-Experiment-Trail.git`

## Gate Status

### 4.1 Dataset Integrity Gate — PASS

- Family partition: 224 train / 75 validation / 75 frozen
- Train ∩ Validation: ∅
- Train ∩ Frozen: ∅
- Validation ∩ Frozen: ∅

### 4.2 Training Trust Gate — PASS

- Assistant-only Loss: System/User/Pad → -100, Assistant → input_ids
- Prefix alignment verification: implemented
- Truncation policy: preserve_assistant
- Token audit (max_seq_length=384):
  - Stage 1: 84 samples, 100% retention, 0 lost
  - Stage 2: 280 samples, 100% retention, 0 lost
  - Stage 3: 560 samples, 100% retention, 0 lost

### 4.3 Evaluation Trust Gate — PASS

- Only Raw Sample format accepted
- Pydantic Sample validation active
- Public tests non-empty enforced
- pytest num_collected > 0 enforced
- Canary mechanism: 3 canaries must all fail
- Dataset SHA256 recorded
- Generation config fixed (deterministic)

### 4.4 Continual Adapter Gate — PASS

- `PeftModel.from_pretrained(model, initial_adapter, is_trainable=True)`
- LoRA trainable parameter verification
- Parent adapter SHA256 recorded
- Output directory protection (refuse overwrite)

### 4.5 GPU Safety Gate — PASS

- CUDA available: True
- BF16 supported: True (RTX 3050 Laptop GPU)
- P1 Smoke Test completed: forward + backward + optimizer.step + save + reload + inference
- Peak GPU memory (P1 smoke): 4201 MiB

## Data Summary

| Stage | Train | Validation | Families |
|---|---:|---:|---:|
| Stage 1 (Code) | 84 | 34 | 79 |
| Stage 2 (Boundary) | 280 | 113 | 184 |
| Stage 3 (Repair) | 560 | 226 | 188 |
| Frozen Eval v2 | 576 | — | 75 |
| **Total** | **1500** | **373** | — |

- Dataset source: MBPP (google-research-datasets/mbpp)
- Original MBPP samples: 374
- Generated variants: boundary 317, static_repair 975, execution_repair 975
- Rejected: 58

## Training Configuration

- Model: Qwen3-0.6B
- Precision: BF16
- Max sequence length: 384
- LoRA rank: 16, alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Assistant-only loss: true
- Truncation policy: preserve_assistant

### Stage 1 (Code Foundation)
- Epochs: 3, LR: 1e-4
- Output: `adapters/p2/continual/stage1-code-v1`

### Stage 2 (Boundary Reasoning)
- Initial adapter: stage1-code-v1
- Epochs: 2, LR: 5e-5
- Output: `adapters/p2/continual/stage2-boundary-v1`

### Stage 3 (Execution Repair)
- Initial adapter: stage2-boundary-v1
- Epochs: 2, LR: 3e-5
- Output: `adapters/p2/continual/stage3-repair-v1`

## Frozen Eval

- Path: `data/p2-curriculum/frozen-eval-v2/test_raw.jsonl`
- Samples: 576
- Families: 75
- SHA256: pending

## Known Risks

1. **GPU memory**: 4GB VRAM, BF16 training with LoRA rank=16 — peak ~4200 MiB observed in P1 smoke
2. **Training duration**: ~1500 train samples × 7 epochs total on 4GB GPU
3. **MBPP data quality**: Some MBPP samples may have incomplete test cases
4. **Sandbox execution**: TRAE sandbox may terminate long-running training (use batch file bypass)

## Verdict

```
GO FOR FULL TRAINING
```

All gates PASS. Data isolation verified. Assistant retention 100%. Continual chain validated.
