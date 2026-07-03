# P2 Stage 1 Code Foundation Training Report

Generated: 2026-07-03T02:47:58.074400+00:00

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

- Started: 2026-07-02T15:41:32.453676+00:00
- Finished: 2026-07-02T15:44:50.608497+00:00
- Duration: 3m 12.7s
- Peak GPU memory: 1342.6 MiB
- Train data SHA256: e26407bee34cfbaffbd5858df48389eb...
- Eval data SHA256: 813a6a221aadb21534f4b2dc6759e365...

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
