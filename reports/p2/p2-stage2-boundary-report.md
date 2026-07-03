# P2 Stage 2 Boundary Reasoning Training Report

Generated: 2026-07-03T06:57:35.039916+00:00

## Training Configuration

- Model: Qwen3-0.6B
- Training mode: continual
- Initial adapter: adapters/p2/continual/stage1-code-v2
- Max sequence length: 384
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable params: 10,092,544 / 606,142,464 (1.67%)
- Assistant-only loss: True
- Truncation policy: preserve_assistant

## Training Results

- Started: 2026-07-03T03:35:19.301257+00:00
- Finished: 2026-07-03T03:43:52.119771+00:00
- Duration: 8m 27.5s
- Peak GPU memory: 1353.2 MiB
- Train data SHA256: 956b6f789909fb4022a6361d9c817384...
- Eval data SHA256: 0cd916b9c8d958ca639c18831dbf503d...

## Token Audit

| Metric | Value |
|---|---:|
| Total samples | 280 |
| Truncated | 7 |
| Assistant intact | 280 |
| Assistant partial | 0 |
| Assistant lost | 0 |
| Target too long | 0 |
| **Assistant retention rate** | **100.0%** |

## Continual Chain

- Parent adapter SHA256: 6655eade4d74a7ce33db8873405e845ee1a1519562e4727b0c11928d343c4e5a
- Parent adapter: adapters/p2/continual/stage1-code-v2
