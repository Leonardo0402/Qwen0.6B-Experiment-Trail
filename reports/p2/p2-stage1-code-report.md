# P2 Stage 1 Code Foundation Training Report

Generated: 2026-07-03T06:57:35.038775+00:00

## Training Configuration

- Model: Qwen3-0.6B
- Training mode: independent
- Initial adapter: None (from base)
- Max sequence length: 384
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable params: 10,092,544 / 606,142,464 (1.67%)
- Assistant-only loss: True
- Truncation policy: preserve_assistant

## Training Results

- Started: 2026-07-03T03:31:42.727092+00:00
- Finished: 2026-07-03T03:34:55.740020+00:00
- Duration: 3m 7.4s
- Peak GPU memory: 1348.5 MiB
- Train data SHA256: 78c2cce6b5d8cc7baf5579600e478bb7...
- Eval data SHA256: 462c1f700b21123cf19e61cf96f3e8d6...

## Token Audit

| Metric | Value |
|---|---:|
| Total samples | 84 |
| Truncated | 1 |
| Assistant intact | 84 |
| Assistant partial | 0 |
| Assistant lost | 0 |
| Target too long | 0 |
| **Assistant retention rate** | **100.0%** |

## Continual Chain

- Parent adapter: None (Stage 1 trains from base)
