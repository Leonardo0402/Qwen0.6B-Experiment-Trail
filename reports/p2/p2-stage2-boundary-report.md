# P2 Stage 2 Boundary Reasoning Training Report

Generated: 2026-07-03T02:47:58.087423+00:00

## Training Configuration

- Model: Qwen3-0.6B
- Training mode: continual
- Initial adapter: adapters/p2/continual/stage1-code-v1
- Max sequence length: 384
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable params: 10,092,544 / 606,142,464 (1.67%)
- Assistant-only loss: True
- Truncation policy: preserve_assistant

## Training Results

- Started: 2026-07-02T15:46:55.912705+00:00
- Finished: 2026-07-02T15:55:01.390596+00:00
- Duration: 7m 59.8s
- Peak GPU memory: 1346.7 MiB
- Train data SHA256: 83b9da9acf7dadd607e6881a82a97b72...
- Eval data SHA256: 898fd3a715b3ff7a43560eac5c3bcfb2...

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

- Parent adapter SHA256: 83c6cb1001ec4b18bec2d374135603d4ca14b5895df6ab933b42450b827e5888
- Parent adapter: adapters/p2/continual/stage1-code-v1
