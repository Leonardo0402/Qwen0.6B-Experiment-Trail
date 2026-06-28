# P0 Trusted Evaluation — Model Comparison Report

Generated: 2026-06-27T13:54:59.923408+00:00

## Evaluation Trustworthiness

| Check | Status |
|---|---|
| Dataset SHA256 | `bb5d505dc85e110a...` |
| All 4 runs same SHA256 | YES |
| Sample count | 36 |
| Task types | {'code_generation': 6, 'static_repair': 15, 'execution_repair': 15} |
| Unique families | 6 |
| Schema validation | {'validated_samples': 36, 'failed_samples': 0} |
| Canary passed | True (all 3 canaries failed as expected) |
| Generation config | identical across all 4 runs |
| sample_id='?' present | NO (all use validated Sample objects) |

## Canary Details

| Canary | Code | Expected | Actual | num_collected | public_passed |
|---|---|---|---|---|---|
| hello_world | `print("Hello, World!")` | fail | fail | 1 | False |
| pass_stmt | `pass` | fail | fail | 1 | False |
| return_none | `def solution(*args, **kwargs):\n    retu` | fail | fail | 1 | False |

## Generation Config (fixed, identical for all models)

```json
{
  "enable_thinking": false,
  "do_sample": false,
  "num_beams": 1,
  "max_new_tokens": 384,
  "temperature": null,
  "top_p": null,
  "repetition_penalty": 1.1,
  "pad_token_id": "<eos>"
}
```

## Metrics Summary

| Metric | Baseline (Qwen3-0.6B) | v3-easy (Easy LoRA) | v3-boundary-v2 (Boundary LoRA) | v3-repair (Repair LoRA) |
|---|---|---|---|---|
| Pass@1 | 0.833 | 0.833 | 0.833 | 0.833 |
| Syntax Rate | 1.000 | 1.000 | 1.000 | 1.000 |
| Hidden Pass Rate | 0.750 | 0.806 | 0.750 | 0.750 |
| Format Compliance | 1.000 | 1.000 | 1.000 | 1.000 |
| Timeout Rate | 0.000 | 0.000 | 0.000 | 0.000 |
| Repair Success | 0.700 | 0.800 | 0.733 | 0.733 |
| Regression Rate | 0.300 | 0.200 | 0.200 | 0.200 |

## Sample Counts

| Category | Baseline (Qwen3-0.6B) | v3-easy (Easy LoRA) | v3-boundary-v2 (Boundary LoRA) | v3-repair (Repair LoRA) |
|---|---|---|---|---|
| Total | 36 | 36 | 36 | 36 |
| Generation | 6 | 6 | 6 | 6 |
| Repair | 30 | 30 | 30 | 30 |

## Deltas vs Baseline

| Metric | v3-easy (Easy LoRA) | v3-boundary-v2 (Boundary LoRA) | v3-repair (Repair LoRA) |
|---|---|---|---|
| Pass@1 | 0.000 | 0.000 | 0.000 |
| Syntax Rate | 0.000 | 0.000 | 0.000 |
| Hidden Pass Rate | +0.056 | 0.000 | 0.000 |
| Format Compliance | 0.000 | 0.000 | 0.000 |
| Timeout Rate | 0.000 | 0.000 | 0.000 |
| Repair Success | +0.100 | +0.033 | +0.033 |
| Regression Rate | -0.100 | -0.100 | -0.100 |

## Key Findings

### 1. Evaluator Trustworthiness Restored

Previous evaluation reported **100% Pass@1 for all 4 models** (including Baseline), 
which was demonstrably false — the old evaluator treated empty test suites as 
`passed=True`. The fixed evaluator now shows real differentiation:

- **Baseline Pass@1 = 0.833** (not 1.0)
- **Baseline Repair Success = 0.700**
- **Baseline Regression Rate = 0.300**

### 2. Canary Mechanism Working

All 3 canary codes (`print('Hello, World!')`, `pass`, `return None`) were 
correctly judged as FAILING against real test samples. The old evaluator would 
have passed them due to the empty-test bug.

### 3. Model Differentiation

- **v3-easy** shows the best Repair Success (0.800) and lowest 
  Regression (0.200) — a genuine improvement over Baseline.
- **v3-boundary-v2** and **v3-repair** show Repair Success of 
  0.733 and 
  0.733 respectively, 
  with Regression = 0.200.
- **Pass@1 is identical (0.833)** across all 4 models, suggesting the 6 
  code_generation samples are too easy to differentiate. The repair tasks 
  (30 samples) show the real differences.

### 4. Test Set Limitations

- Only **6 unique family_ids** and **36 samples** — too small to draw 
  strong conclusions about model capability differences.
- Pass@1 only covers 6 code_generation samples; all 4 models solve 5/6.
- The real differentiation is in Repair Success and Regression Rate.
- **P1 will expand the frozen eval set** using existing untrained families.

## Compatibility Verification

- Baseline vs v3-easy: **COMPATIBLE**
- Baseline vs v3-boundary-v2: **COMPATIBLE**
- Baseline vs v3-repair: **COMPATIBLE**

All comparisons compatible: **True**

## Files

```
evaluations/fixed-p0/
├── baseline.json          # Baseline (Qwen3-0.6B, no adapter)
├── v3-easy.json           # code-lora-v3-easy adapter
├── v3-boundary-v2.json    # code-lora-v3-boundary-v2 adapter
├── v3-repair.json         # code-lora-v3-repair adapter
├── canary-report.json     # Canary test results
└── comparison.md          # This report
```

