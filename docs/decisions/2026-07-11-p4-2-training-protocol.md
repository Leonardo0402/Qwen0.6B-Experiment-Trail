# Decision Record: P4.2 Agent SFT Training Protocol

**Date:** 2026-07-11 (corrected 2026-07-12 per Issue #32)
**Status:** PROPOSED / DEFERRED_TO_TWO_ARM_PILOT
**Issue:** #27
**Depends on:** #29 (P4.1b Protocol Ablation), #32 (Trust Repair)

## Context

P4.1 delivered 1350 SFT trajectories with GO_FOR_P4_AGENT_SFT verdict. P4.1b
ablation (Issue #29) was intended to produce a verdict constraining the
training protocol choice. However, Issue #32 (Trust Repair) discovered that
the original ablation results were **not trustworthy** due to silent field
dropping (Pydantic `extra="ignore"`).

This decision record is corrected to clearly separate two independent
decisions that were previously conflated:

1. **Decision 1 — Action Protocol**: Which protocol format to use (JSON/Tag/DSL)
2. **Decision 2 — Agent SFT Initialization**: Which base/adapter to start from

P4.1b ablation **only** informs Decision 1. Decision 2 must be resolved by a
matched two-arm Agent SFT pilot, not by zero-shot protocol ablation results.

## Decision 1 — Action Protocol (informed by P4.1b)

### Evidence Base (P4.1b v2 Ablation, 240 runs)

Full ablation report: `reports/p4/protocol-ablation-v2/comparison-report.md`

**IMPORTANT**: The v1 ablation results (`reports/p4/protocol-ablation/`) are
SUPERSEDED and must not be used. See
`reports/p4/protocol-ablation/SUPERSEDED.md` for details.

The v2 ablation uses strict schema validation (`extra="forbid"` on all
Pydantic models) and independent diagnostic dimensions. Results are
recorded in the v2 comparison report after the 240-run is re-executed.

### Protocol Selection

The protocol verdict from v2 ablation determines which of JSON/Tag/DSL
is the most schema-stable protocol **under the current model, current
prompt, and current absence of Agent SFT**.

**Accurate framing of the verdict:**

> In the current model, current prompt, and current no-Agent-SFT condition,
> Action JSON is the three candidate protocols with the highest format and
> schema stability.

**Prohibited framings:**

- "Repair-LoRA has been proven to be the best Agent SFT training starting point"
- "Base has been proven unsuitable for Agent SFT"

### Protocol Decision Status

Determined by P4.1b v2 verdict. See `reports/p4/protocol-ablation-v2/verdict.json`.

## Decision 2 — Agent SFT Initialization (DEFERRED)

### Why Zero-Shot Ablation Cannot Decide This

The original decision record used zero-shot `schema_valid_rate` to select
the training initialization (Option B: P3-limited). This is invalid because:

1. **Zero-shot failure does not imply post-SFT failure**: A base model that
   produces 0% schema_valid zero-shot may learn protocol compliance during
   Agent SFT. The ablation cannot predict post-training behavior.
2. **Zero-shot success does not imply best training start**: A repair-LoRA
   that achieves high schema_valid zero-shot may plateau or overfit during
   continual training, while a fresh base may generalize better.
3. **Selection bias**: Using the same data (repair-LoRA) that was trained
   on JSON format to validate JSON format is circular reasoning.

### Two Candidate Arms

Both arms must be preserved for a matched comparison:

**Arm A: Base → Agent SFT**
- Load: `models/Qwen3-0.6B` (no adapter)
- Train data: P4.1 Agent SFT trajectories (1350 samples)
- Protocol: Decision 1 verdict
- Rationale: Clean baseline, no protocol-bound pretraining bias

**Arm B: P3 Repair-Limited LoRA → Agent SFT**
- Load: `adapters/p3/repair-limited` (frozen parent)
- Train data: P4.1 Agent SFT trajectories (1350 samples)
- Protocol: Decision 1 verdict
- Rationale: Continual from repair curriculum, may benefit from code repair priors

### Required Evidence

The final selection between Arm A and Arm B must be based on:

1. Matched two-arm Agent SFT pilot (same data, same protocol, same hyperparameters)
2. Post-training evaluation on frozen held-out set
3. Composite score comparison (schema_valid_rate, task_success_rate, safety_valid_rate)
4. Statistical significance assessment

### Initialization Decision Status

**DEFERRED_TO_TWO_ARM_PILOT** — not authorized until:
- P4.2a two-arm readiness gate passes
- User explicitly approves the two-arm pilot
- TRAINING_AUTHORIZATION_GATE returns AUTHORIZED_FOR_DECLARED_PILOT_ONLY

## Training Configuration (proposal, subject to P4.2a readiness gate)

- Protocol: Decision 1 verdict
- Train data: P4.1 Agent SFT trajectories (1350 samples)
- Precision: float16 (BF16 if `torch.cuda.is_bf16_supported()` returns True)
- LoRA: rank TBD in P4.2a (must fit 4GB VRAM)
- Max seq len: TBD in P4.2a (must fit 4GB VRAM)
- Batch/accumulation: TBD in P4.2a
- Output paths:
  - Arm A: `adapters/p4/agent-sft-arm-a-v0` (versioned, empty)
  - Arm B: `adapters/p4/agent-sft-arm-b-v0` (versioned, empty)
- Determinism: seed recorded

## Safety and Resource Constraints

- All training must fit in RTX 3050 Laptop 4GB VRAM.
- No overwrite of existing adapters (append-only rule).
- No training execution without explicit user approval (TRAINING_AUTHORIZATION_GATE).
- No silent CPU/precision/config fallback.
- Parent adapter `adapters/p3/repair-limited` is locked (read-only).

## Required Approvals Before P4.2a

This decision record is **PROPOSED / DEFERRED_TO_TWO_ARM_PILOT**, not authorized.
The following are required before P4.2a training readiness:

1. User explicitly approves the two-arm pilot design.
2. P4.2a training readiness gate passes (separate issue).
3. TRAINING_AUTHORIZATION_GATE returns AUTHORIZED_FOR_DECLARED_PILOT_ONLY.
4. Decision 1 (protocol) is confirmed from v2 ablation verdict.

## PR Relationship

- Refs #27 (this decision record)
- Refs #29 (ablation evidence base, v1 SUPERSEDED)
- Refs #32 (Trust Repair, v2 ablation evidence base)
- Refs #19 (P4.1 SFT data)
