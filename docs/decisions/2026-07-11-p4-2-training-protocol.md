# Decision Record: P4.2 Agent SFT Training Protocol

**Date:** 2026-07-11
**Status:** PROPOSED (awaiting user approval)
**Issue:** #27
**Depends on:** #29 (P4.1b Protocol Ablation, verdict produced)

## Context

P4.1 delivered 1350 SFT trajectories with GO_FOR_P4_AGENT_SFT verdict. P4.1b
ablation (Issue #29) produced a verdict (`KEEP_ACTION_JSON`) that constrains
the training protocol choice. This decision record selects the training
protocol for P4.2 Agent SFT.

## Evidence Base (P4.1b Ablation, 240 runs)

Full ablation report: `reports/p4/protocol-ablation/comparison-report.md`

| Protocol | Config | schema_valid_rate | task_success_rate |
|----------|--------|-------------------|-------------------|
| json     | repair-lora | 96.25% | 0.00% |
| dsl      | repair-lora | 8.12%  | 0.00% |
| tag      | repair-lora | 1.67%  | 0.00% |
| json     | base        | 0.00%  | 0.00% |
| tag      | base        | 2.29%  | 0.00% |
| dsl      | base        | 0.00%  | 0.00% |

Key findings:

1. **base model (no LoRA) is 0% schema_valid across all 3 protocols** —
   independent training from base must learn protocol format from scratch
   and carries high risk of failure.
2. **P3 repair-lora + JSON is the only usable combination (96.25%)** —
   the repair LoRA already provides JSON protocol compliance.
3. **tag/dsl protocols fail even with repair-lora (1.67% / 8.12%)** —
   LoRA training is protocol-bound and does not transfer.
4. **task_success_rate is 0% for all combinations** — schema compliance
   is necessary but not sufficient; P4.2 must improve task-level success.

## Options Considered

| Option | Load | Train on | Pros | Cons | Ablation support |
|--------|------|----------|------|------|------------------|
| A. Continual (P2 stage3) | P2 stage3 adapter | MBPP + Agent SFT | Extends existing curriculum | P2 stage3 JSON compatibility unverified | None — P2 stage3 not in ablation |
| B. Continual (P3-limited) | P3-limited adapter | Agent SFT only | Proven 96.25% JSON schema_valid | P3 was limited experiment | **Direct evidence: only working config** |
| C. Independent (base) | Qwen3-0.6B base | Agent SFT only | Clean baseline | 0% schema_valid from base | **Directly contradicted by ablation** |
| D. Multi-task (all) | P2 stage3 | P2 + P3 + Agent SFT | Maximal retention | Highest VRAM risk, out of P4.2 scope | None |

## Selected Protocol (PROPOSED)

**Option B: Continual (P3-limited → Agent SFT)**

### Rationale

- Ablation directly proves P3-limited + JSON is the only configuration that
  achieves usable schema validity (96.25%).
- Option A (P2 stage3) lacks ablation evidence and requires additional
  verification before commitment.
- Option C (Independent) is directly contradicted by the 0% base-model
  schema_valid_rate across all protocols.
- Option D (Multi-task) exceeds the P4.2 single-purpose Agent SFT scope and
  carries the highest VRAM risk on RTX 3050 4GB.

### Training Configuration (proposal, subject to P4.2a readiness gate)

- Base: `adapters/p3/repair-limited` (frozen, parent artifact locked)
- Train data: P4.1 Agent SFT trajectories (1350 samples)
- Protocol: JSON (frozen, per ablation verdict)
- Precision: float16
- LoRA: rank TBD in P4.2a
- Max seq len: TBD in P4.2a (must fit 4GB VRAM)
- Batch/accumulation: TBD in P4.2a
- Output path: `adapters/p4/agent-sft-v0` (versioned, empty)
- Determinism: seed recorded

### Reversibility

- Parent adapter `adapters/p3/repair-limited` is locked (read-only).
- New adapter is written to a new versioned path.
- Rollback = delete `adapters/p4/agent-sft-v0`.

## Safety and Resource Constraints

- All training must fit in RTX 3050 Laptop 4GB VRAM.
- No overwrite of existing adapters (append-only rule).
- No training execution without explicit user approval (TRAINING_AUTHORIZATION_GATE).
- No silent CPU/precision/config fallback.

## Required Approvals Before P4.2a

This decision record is **PROPOSED**, not authorized. The following are
required before P4.2a training readiness:

1. **User explicitly approves Option B** (or selects another option with
   justification that overrides the ablation evidence).
2. P4.2a training readiness gate passes (separate issue).
3. TRAINING_AUTHORIZATION_GATE returns AUTHORIZED_FOR_DECLARED_PILOT_ONLY.

## PR Relationship

- Refs #27 (this decision record)
- Refs #29 (ablation evidence base)
- Refs #19 (P4.1 SFT data)
