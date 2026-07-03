# P2 Final Comparison Report

Generated: 2026-07-03T02:47:58.118022+00:00

## Evaluation Setup

- Dataset: data/p2-curriculum/frozen-eval-v2/test_raw.jsonl
- Dataset SHA256: ?...
- Samples: 120 (stratified: 40 code_generation + 40 static_repair + 40 execution_repair)
- Families: 40
- Canary: All canaries failed as expected

## Overall Metrics

| Model | Pass@1 | Syntax | Repair | Hidden | Format | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| Stage1-Code | 0.0% | 99.2% | 0.0% | 0.0% | 99.2% | 0.0% |
| Stage2-Boundary | 0.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| Stage3-Repair | 0.0% | 99.2% | 0.0% | 0.0% | 99.2% | 0.0% |

## Per-Task-Type Breakdown

### code_generation

| Model | Total | Passed | Rate |
|---|---:|---:|---:|
| Base | 40 | 0 | 0.0% |
| Stage1-Code | 40 | 0 | 0.0% |
| Stage2-Boundary | 40 | 0 | 0.0% |
| Stage3-Repair | 40 | 0 | 0.0% |

### static_repair

| Model | Total | Passed | Rate |
|---|---:|---:|---:|
| Base | 40 | 0 | 0.0% |
| Stage1-Code | 40 | 0 | 0.0% |
| Stage2-Boundary | 40 | 0 | 0.0% |
| Stage3-Repair | 40 | 0 | 0.0% |

### execution_repair

| Model | Total | Passed | Rate |
|---|---:|---:|---:|
| Base | 40 | 0 | 0.0% |
| Stage1-Code | 40 | 0 | 0.0% |
| Stage2-Boundary | 40 | 0 | 0.0% |
| Stage3-Repair | 40 | 0 | 0.0% |

## Family-Level Pass

| Model | Families Passed | Total Families | Rate |
|---|---:|---:|---:|
| Base | 0 | 40 | 0.0% |
| Stage1-Code | 0 | 40 | 0.0% |
| Stage2-Boundary | 0 | 40 | 0.0% |
| Stage3-Repair | 0 | 40 | 0.0% |

## Adapter Evidence

| Stage | SHA256 (first 32) | Parent SHA256 | Training Mode |
|---|---|---|---|
| stage1-code | `b4909abac0c6dcf11d497151e2a99a51` | `None` | independent |
| stage2-boundary | `00bd35d43864522ed7ff79a374af4399` | `83c6cb1001ec4b18bec2d374135603d4` | continual |
| stage3-repair | `bfe6b9c3525ab30279a7474bf3ef8de7` | `c90058a5fa1fa78abf18555c2c91c4fb` | continual |

- All adapter SHA256 different: TRUE
- Parent chain verified: TRUE (parent_adapter_sha256 matches parent's adapter_config.json SHA256)

## Verdict Analysis

### Engineering准线 (all PASS)

- Pytest: PASS (all tests green after fixes)
- Canary: PASS (all canaries failed)
- Assistant retention: 100% PASS
- Train/Val/Frozen zero leakage: PASS
- No OOM: PASS (peak ~1371 MiB)
- Adapter save/reload: PASS
- SHA256 different between stages: PASS

### Capability准线

- Code Generation Pass@1 improvement: 0% (0% → 0%)
- Boundary Success improvement: 0% (0% → 0%)
- Execution Repair Success: 0% (target ≥ 40%)
- Family-level Pass: 0% (target ≥ 10% improvement)

### Root Cause Analysis

The 0% Pass@1 across all models (including base) is due to:

1. **Function name mismatch**: MBPP instructions describe the task in natural language
   but do NOT include the expected function signature. The 0.6B model cannot infer
   exact function names (e.g., instruction says 'find max of nth column' but tests
   expect `max_of_nth()`, model generates `max_of_nth_column()`).

2. **Small model capacity**: Qwen3-0.6B has limited code generation capability on
   raw MBPP without function signatures or few-shot examples.

3. **Training data format**: The instruction → target_code mapping doesn't include
   function signatures in instructions, so the model learns the same pattern.

### Verdict: FIX FIRST

The engineering infrastructure is complete and trustworthy, but the capability
准线 is not met. All models (including base) score 0% Pass@1.

## Recommended Next Steps

1. **Include function signatures in instructions**: Modify the data factory to
   extract function name from target_code and append to instruction.
2. **Add few-shot examples**: Include 1-2 examples in the prompt.
3. **Increase training data**: 84 samples (Stage 1) is too small; expand to 500+.
4. **Consider instruction tuning**: Format as chat with system prompt containing
   coding guidelines.
