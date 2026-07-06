# P3 GPU Smoke Report — Issue #12 P7

**Date:** 2026-07-05
**Verdict:** PASS (both candidates)

---

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Total VRAM | 4095 MB (4 GB) |
| Compute Capability | 8.6 (Ampere) |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| transformers | 5.12.1 |
| peft | 0.19.1 |
| Python | 3.11 |
| OS | Windows |
| BF16 Supported | Yes |
| Actual Precision | Model load: fp16, Training: bf16 (mixed) |

---

## Balanced Generalist (independent)

**Config:** `configs/p3/balanced-generalist-pilot.yaml`
**Adapter:** `adapters/p3/balanced-generalist-pilot/`

| Metric | Value |
|---|---|
| training_mode | independent |
| initial_adapter | null |
| max_steps | 3 |
| Trainable params | 10,092,544 (10.1 M) |
| Total params | 606,142,464 (606.1 M) |
| Train samples | 622 |
| Train duration | 64.21 s (3 steps) |
| Step time (avg) | ~21.4 s/step |
| Peak VRAM | 1350.86 MiB |
| NaN/Inf | None detected |
| OOM | No |
| Token audit | total=622, truncated=104, assistant_intact=622 |
| Checkpoints | checkpoint-2, checkpoint-3 |
| Save/Reload | OK (LoRA param mean matched within 1e-5) |
| Adapter SHA256 | 68c732a7e0499637cebbca0da09701c95af64caf07970766d79ef8a8fe644836 |

**Inference output:**
```
Prompt: def add(a, b):
    return
Generated:  a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a,
```

---

## Repair Specialist (independent)

**Config:** `configs/p3/repair-specialist-pilot.yaml`
**Adapter:** `adapters/p3/repair-specialist-pilot/`

| Metric | Value |
|---|---|
| training_mode | independent |
| initial_adapter | null |
| max_steps | 3 |
| Trainable params | 10,092,544 (10.1 M) |
| Total params | 606,142,464 (606.1 M) |
| Train samples | 490 |
| Train duration | 70.95 s (3 steps) |
| Step time (avg) | ~23.7 s/step |
| Peak VRAM | 1342.05 MiB |
| NaN/Inf | None detected |
| OOM | No |
| Token audit | total=490, truncated=110, assistant_intact=490 |
| Checkpoints | checkpoint-2, checkpoint-3 |
| Save/Reload | OK (LoRA param mean matched within 1e-5) |
| Adapter SHA256 | 88fd1c3bd81292bb2a8026f47dccc98307c297868a9c3e8e98f52b4df5a7b07c |

**Inference output:**
```
Prompt: def add(a, b):
    return
Generated:  a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a,
```

---

## P7 Checklist

| Requirement | Balanced | Repair |
|---|---|---|
| 2-5 optimizer steps | 3 steps | 3 steps |
| forward | OK | OK |
| backward | OK | OK |
| optimizer.step | OK | OK |
| eval loss | OK (eval_steps=2) | OK (eval_steps=2) |
| checkpoint save | OK (checkpoint-2, checkpoint-3) | OK (checkpoint-2, checkpoint-3) |
| adapter reload | OK (param mean matched) | OK (param mean matched) |
| inference | OK (generated valid code) | OK (generated valid code) |
| GPU model | RTX 3050 Laptop GPU | RTX 3050 Laptop GPU |
| CUDA / PyTorch / driver | 12.4 / 2.6.0+cu124 | 12.4 / 2.6.0+cu124 |
| BF16 support | Yes | Yes |
| Actual precision | fp16 load + bf16 train | fp16 load + bf16 train |
| Peak VRAM | 1350.86 MiB (33% of 4GB) | 1342.05 MiB (33% of 4GB) |
| Step time | ~21.4 s/step | ~23.7 s/step |
| NaN/Inf | None | None |
| OOM | No | No |
| Save/reload hash | 68c732a7... | 88fd1c3b... |
| Inference output | "a + b" (valid) | "a + b" (valid) |

---

## Conclusion

Both candidates passed GPU smoke on RTX 3050 4GB:

- **Independent mode** confirmed (no P2 adapter loaded, `initial_adapter: null`).
- **3 optimizer steps** completed for each candidate.
- **Forward / backward / optimizer.step / eval loss / checkpoint save / adapter reload / inference** all verified.
- **Peak VRAM** ~1.35 GB (33% of 4 GB) — well within RTX 3050 capacity.
- **No NaN/Inf, no OOM.**
- **BF16 training** enabled and supported.
- **Inference** produces valid code completion (`a + b` for `def add(a, b): return`).

**Verdict: PASS** — GPU smoke requirements met for both candidates.

Next: Phase 8/9 Pilot (max_steps=50, max 0.25 epoch) — requires user authorization under GO_FOR_P3_PILOT_ONLY.
