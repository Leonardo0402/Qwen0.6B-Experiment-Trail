# P2 Stage 3 Execution Repair Training Report

Generated: 2026-07-03T02:47:58.100008+00:00

## Training Configuration

- Model: Qwen3-0.6B
- Training mode: continual
- Initial adapter: adapters/p2/continual/stage2-boundary-v1
- Max sequence length: 384
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable params: 10,092,544 / 606,142,464 (1.67%)
- Assistant-only loss: True
- Truncation policy: preserve_assistant

## Training Results

- Started: 2026-07-02T15:55:11.649121+00:00
- Finished: 2026-07-02T16:14:40.067786+00:00
- Duration: 19m 22.8s
- Peak GPU memory: 1370.8 MiB
- Train data SHA256: 51d20ed10e7b00b2de4f1de2fb191254...
- Eval data SHA256: 6285d9b329e2ac65077fd1684ddbd2b2...

## Token Audit

| Metric | Value |
|---|---:|
| Total samples | 560 |
| Truncated | 125 |
| Assistant intact | 560 |
| Assistant partial | 0 |
| Assistant lost | 0 |
| Target too long | 0 |
| **Assistant retention rate** | **100.0%** |

## Continual Chain

- Parent adapter SHA256: c90058a5fa1fa78abf18555c2c91c4fbf196feb7a6878b47dc0572a3988a173e
- Parent adapter: adapters/p2/continual/stage2-boundary-v1
