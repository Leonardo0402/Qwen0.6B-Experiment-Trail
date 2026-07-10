# AGENTS.md

This file provides context and operating rules for AI agents (Claude / Codex / etc.)
working on the `qwen3-code-lab` repository.

## Repository Purpose

Experimental training platform for `Qwen3-0.6B` on a single RTX 3050 Laptop GPU
(4 GB VRAM). The goal is to strengthen Python **code generation / static bug
repair / execution-feedback repair** via Continual Curriculum LoRA on the MBPP
dataset. All work must remain reproducible on the reference hardware.

## Hard Constraints (do not violate)

1. **Hardware ceiling** — every training run must fit in 4 GB VRAM
   (BF16 + LoRA rank 16 + gradient checkpointing). Validate
   `torch.cuda.is_bf16_supported()` before training.
2. **Data isolation** — train / validation / frozen-eval families are
   three-way disjoint (`family-partition.json`). Never write frozen-eval
   family ids into a training config or train file.
3. **Assistant-only loss** — System / User / Pad tokens masked to `-100`.
   Assistant target answers must be 100 % preserved under truncation
   (left-truncate the prompt, never the assistant answer).
4. **Continual chain integrity** — Stage N+1 loads Stage N adapter with
   `PeftModel.from_pretrained(model, initial_adapter, is_trainable=True)`.
   Each stage's adapter SHA256 must differ from its parent.
5. **Frozen eval immutability** — `data/p2-curriculum/frozen-eval-v2/` is
   the canonical evaluation set. Never overwrite `test_raw.jsonl` or its
   manifest. Formal evaluation must use this file exclusively.
6. **No external datasets** — MBPP only. Do not pull HumanEval / LeetCode /
   MBPP-sanitized into training data.
7. **Full training requires explicit user instruction** — only smoke tests
   may be auto-executed.
8. **Adapters are append-only** — never overwrite an existing adapter
   directory; create a new `-v{N}` suffix instead. v1 adapters are backed
   up under `adapters/p2/continual-backup-v1/`.

## Repository Layout

```
src/                  reusable modules (schemas, sandbox, collator, ...)
scripts/              CLI entry points (training, eval, audit, reports)
tests/                pytest suite (must stay green)
configs/curriculum/   per-stage training YAMLs
data/p2-curriculum/
  stage1-code/        easy curriculum (84 train)
  stage2-boundary/    boundary reasoning (280 train)
  stage3-repair/      execution repair (560 train)
  frozen-eval-v2/     FROZEN, immutable evaluation set (576 samples)
  family-partition.json
adapters/p2/
  continual/          v2 continual chain (stage1 / stage2 / stage3)
  continual-backup-v1/v1 legacy adapters, read-only
  independent/        independent-mode control adapters
evaluations/p2/       per-model JSON outcomes + comparison.json
reports/p2/           Markdown reports + audit JSONs
```

## Operating Rules for Agents

### Before writing code

- Read the relevant `src/` module and its `tests/` counterpart first.
- Match existing style; do not reformat untouched code.
- State assumptions explicitly when ambiguous; ask rather than guess.
- Prefer editing existing files over creating new ones.

### When running training

- Always launch long-running training via `Start-Process` + `cmd.exe` to
  bypass the TRAE sandbox (which otherwise terminates long Python runs).
- Record in `metrics.json`: parent adapter SHA256 (weight + config),
  trainable param count, LoRA parameter norms, peak GPU MiB, full loss
  curve, train/eval data SHA256.
- Smoke test before full training.

### When evaluating

- Use `data/p2-curriculum/frozen-eval-v2/test_raw.jsonl` only.
- The 120-sample stratified subset is fixed in
  `data/p2-curriculum/frozen-eval-v2/stratified-120/` with its own
  manifest and SHA256. Do not re-sample per run.
- Canary samples must all fail; if any canary passes, abort and investigate.
- `_normalize_test_code` in `src/sandbox.py` must wrap bare-assert MBPP
  tests before pytest collection.

### When committing

- Never `git add -A` / `git add .` — stage files explicitly.
- `adapter_model.safetensors` and other large blobs are gitignored; keep
  them on disk only.
- Commit messages: `type(scope): subject` followed by a body explaining
  the why. Reference the issue number (`Closes #N`) when applicable.
- Never force-push to `main` or to a shared branch.

### When opening a PR

- PR title summarizes the change; body lists what / why / how-to-verify.
- All pytest tests must pass before requesting review.
- Include before/after metrics tables when capability is affected.

## Post-Task Review Gate

**After completing any task, check the current project state against the key milestones in `../skills/qwen-experiment-trail-review-gate/关键节点.md`.**

Procedure:
1. Identify the current project phase (P4.1, P4.1b, P4.2, P4.3, etc.)
2. Run through the relevant milestone checklist from `关键节点.md`
3. If any checkbox is unchecked, determine whether it blocks the current phase
4. If the current phase is complete, determine whether a PR should be opened
5. Check the "任何阶段的阻断性审查" checklist regardless of phase

**PR readiness check:**
- [ ] All relevant milestone checkboxes are checked
- [ ] `pytest tests/` is green
- [ ] All modified files are explicitly staged (no `git add -A`)
- [ ] Commit messages follow `type(scope): subject` convention
- [ ] If the work closes an issue, `Closes #N` is in the PR body
- [ ] `../skills/qwen-experiment-trail-review-gate/SKILL.md` review rules have been applied

## Reference Documents

- Spec: `../docx/Qwen3-0.6B_Code_Training_Development_Spec.md`
- Implementation plan: `../docx/Qwen3-0.6B_Implementation_Plan.md`
- P2 readiness report: `reports/p2/p2-training-readiness-report.md`
- P2 final comparison: `reports/p2/p2-final-comparison-report.md`
- Review gate skill: `../skills/qwen-experiment-trail-review-gate/SKILL.md`
- Key milestones: `../skills/qwen-experiment-trail-review-gate/关键节点.md`

## Verification Checklist (run before any merge)

- [ ] `pytest tests/` — all green
- [ ] Canary samples all fail
- [ ] Train / Validation / Frozen family sets are three-way disjoint
- [ ] Assistant retention rate = 100 %
- [ ] Adapter SHA256 differs between stages; parent chain verified
- [ ] No OOM; peak GPU MiB recorded
- [ ] `dataset-audit.json` totals match `84 + 280 + 560 = 924` train
      (frozen-eval-v2 NOT counted as train)
- [ ] `parent_adapter_weight_sha256` references the parent's
      `adapter_model.safetensors`, NOT its `adapter_config.json`
