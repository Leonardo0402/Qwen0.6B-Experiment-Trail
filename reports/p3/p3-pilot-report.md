# P3 Independent Engineering Smoke Report — Issue #14 P0.2 Reclassification

**Date:** 2026-07-05 (originally run as "Pilot", reclassified per Issue #14 P0.2)
**Verdict:** PASS (engineering smoke, NOT compliant pilot)
**Capability claims allowed:** NO (engineering smoke, not formal training)

> **Reclassification Notice (Issue #14 P0.2):**
> The two 50-step runs originally labeled "Pilot" exceed the 0.25-epoch
> limit and are reclassified as **Independent Engineering Smoke**.
> Compliant pilot max_steps (Issue #14 P0.3):
> - Balanced: ceil(622/8)=78 steps/epoch, pilot_max=min(50, floor(0.25×78))=19
> - Repair: ceil(490/8)=62 steps/epoch, pilot_max=min(50, floor(0.25×62))=15
> The 50-step runs (~0.64 and ~0.81 epoch respectively) do NOT satisfy
> the 0.25-epoch pilot contract. They remain valid as engineering smoke
> evidence only.

---

## Engineering Smoke Context

Per Issue #12 P8 (original) / Issue #14 P0.2 (reclassified):
- 50 optimizer steps per candidate (exceeds 0.25 epoch)
- GO_FOR_P3_PILOT_ONLY verdict (capacity < 2300 but >= 400)
- Results must NOT be reported as formal capability
- No Frozen v4 formal evaluation

---

## Balanced Generalist Engineering Smoke

**Config:** `configs/p3/balanced-generalist-pilot.yaml` (max_steps=50, independent)
**Adapter:** `adapters/p3/balanced-generalist-pilot/`

| Metric | Value |
|---|---|
| training_mode | independent |
| initial_adapter | null |
| max_steps | 50 |
| Learning rate | 5e-5 |
| Trainable params | 10,092,544 (10.1 M) |
| Total params | 606,142,464 (606.1 M) |
| Train samples | 622 |
| Train duration | 837.08 s (50 steps) |
| Step time (avg) | ~16.7 s/step |
| Peak VRAM | 1350.86 MiB (33% of 4GB) |
| NaN/Inf | None detected |
| OOM | No |
| Token audit | total=622, truncated=104, assistant_intact=622 |
| Save/Reload | OK |
| Adapter SHA256 | fd5254cf108bf38d1a0bbaec55fc39a7bc099346794a948aea1d198c55e79905 |

**Inference output (50 steps):**
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

**P8 Pilot verification:**
- Data loading: OK (622 samples)
- assistant-only loss: OK (assistant_intact=622)
- Loss finite: OK (no NaN/Inf, exit code 0)
- 3-tier evaluator callback: attached (pilot_mode=True, Tier 2/3 deferred)
- Checkpoint save: OK
- Reload: OK
- Early-stop signal: N/A (50 steps completed without divergence)
- VRAM stability: OK (1350 MiB, 33% of 4GB)

---

## Repair Specialist Engineering Smoke

**Config:** `configs/p3/repair-specialist-pilot.yaml` (max_steps=50, independent)
**Adapter:** `adapters/p3/repair-specialist-pilot/`

| Metric | Value |
|---|---|
| training_mode | independent |
| initial_adapter | null |
| max_steps | 50 |
| Learning rate | 3e-5 |
| Trainable params | 10,092,544 (10.1 M) |
| Total params | 606,142,464 (606.1 M) |
| Train samples | 490 |
| Train duration | 983.42 s (50 steps) |
| Step time (avg) | ~19.7 s/step |
| Peak VRAM | 1342.05 MiB (33% of 4GB) |
| NaN/Inf | None detected |
| OOM | No |
| Token audit | total=490, truncated=110, assistant_intact=490 |
| Save/Reload | OK |
| Adapter SHA256 | 2810cf548651cd9786e7464623ce1fc8e59c176002199af1363e1ae3295dcf30 |

**Inference output (50 steps):**
```
Prompt: def add(a, b):
    return
Generated:  a + b

def main():
    a = int(input("Enter first number: "))
    b = int(input("Enter second number: "))
    print(add
```

**P8 Pilot verification:**
- Data loading: OK (490 samples)
- assistant-only loss: OK (assistant_intact=490)
- Loss finite: OK (no NaN/Inf, exit code 0)
- 3-tier evaluator callback: attached (pilot_mode=True, Tier 2/3 deferred)
- Checkpoint save: OK
- Reload: OK
- Early-stop signal: N/A (50 steps completed without divergence)
- VRAM stability: OK (1342 MiB, 33% of 4GB)

---

## Pilot Summary

| Candidate | Steps | Duration | Peak VRAM | NaN/Inf | OOM | Save/Reload | Inference |
|---|---|---|---|---|---|---|---|
| Balanced | 50 | 837s | 1350 MiB | None | No | OK | Valid code |
| Repair | 50 | 983s | 1342 MiB | None | No | OK | Valid code |

**Capability claims allowed:** NO

Both pilots completed 50 optimizer steps in independent mode without issues:
- No NaN/Inf, no OOM
- Peak VRAM ~1.35 GB (33% of 4 GB RTX 3050)
- Checkpoint save/reload verified
- Inference produces valid code completions
- Training pipeline (data loading, assistant-only loss, 3-tier evaluator callback) functional

**Pilot verdict:** PASS — training pipeline verified, ready for formal training authorization (separate user approval required).

---

## Issue #12 Final Output Format

```markdown
## Verdict

GO_FOR_P3_PILOT_ONLY

## Data

| Candidate | Samples | Families | Code | Boundary | Static | Exec | Ratio OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced | 622 | 312 | 30% | 20% | 20% | 30% | Yes |
| Repair | 490 | 245 | 15% | 15% | 30% | 40% | Yes |

## Eval Coverage

| Dataset | Families | Code | Boundary | Static | Exec | Canary |
|---|---:|---:|---:|---:|---:|---:|
| Frozen v4 | 100 | 25-30% | 15-20% | 25-30% | 25-30% | 5 |

## GPU Smoke

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- precision: fp16 load + bf16 train
- peak VRAM: 1350 MiB (Balanced) / 1342 MiB (Repair)
- steps: 3 (smoke) + 50 (pilot)
- save/reload: OK (both candidates)
- verdict: PASS

## Pilot

- Balanced: PASS (50 steps, 837s, independent)
- Repair: PASS (50 steps, 983s, independent)
- capability claims allowed: NO

## Git Delivery

- branch: feat/p3-boundary-repair-pipeline-v3
- commits: 3c746eb, 24f6842, 4568cdb, (pilot commit)
- PR: #13
- CI: green (run 28742119618)
```
