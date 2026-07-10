# Qwen0.6B-Experiment-Trail Roadmap after P4.1 Local Completion Claim

> Date: 2026-07-10  
> Project role: Technical Review / Experimental Route / GitHub Gatekeeping  
> Current status: **P4.1 local completion claimed, remote verification pending**  
> Current model position: **Constrained Local Repair Agent**, not Builder

---

## 1. Current Verification Verdict

### Verdict

```text
FIX FIRST — REMOTE EVIDENCE REQUIRED
```

The reported P4.1 result is technically plausible, but it is not yet a verified project milestone.

At the time of verification:

- remote `main` still points to P4.0 merge commit `7ccd06c`;
- Issue #19 remains open;
- no P4.1 PR is visible;
- the reported P4.1 files and reports are not available on remote `main`;
- therefore code changes, tests, manifests, dataset counts, replay evidence, and readiness gates cannot yet be independently audited.

The endpoint should currently be described as:

```text
LOCAL_CANDIDATE_GO_FOR_P4_AGENT_SFT
```

not:

```text
VERIFIED_GO_FOR_P4_AGENT_SFT
```

---

## 2. Blocking Consistency Checks

### 2.1 Dataset count mismatch

Reported:

```text
total trajectories = 1315
train = 920
validation = 130
heldout-agent-eval = 220
```

But:

```text
920 + 130 + 220 = 1270
1315 - 1270 = 45
```

The missing 45 trajectories must be explicitly accounted for.

Acceptable explanations include:

- quarantined;
- rejected;
- audit-only;
- unsplit model-smoke trajectories;
- test-only;
- duplicated and excluded;
- invalid/replay-failed.

However, the manifest must record them. A dataset cannot pass readiness while 45 trajectories have no declared lifecycle.

Required invariant:

```python
total_generated == accepted + rejected + quarantined
accepted == train + validation + heldout_agent_eval
```

---

### 2.2 Model smoke sample size

Reported:

```text
base = 5 trajectories
repair-lora = 5 trajectories
```

This may be acceptable as a GPU loading/generation smoke, but it does not by itself satisfy a 40-task comparison.

The evidence must distinguish:

```text
GPU/model smoke:
- 5 tasks per configuration

Full ModelActionProvider evaluation:
- 40 tasks per configuration
```

If only 5 + 5 were run, the gate must be labelled:

```text
LIMITED_MODEL_SMOKE_PASS
```

It must not claim that the full 40-task comparison was completed.

---

### 2.3 Smoke success metrics are incomplete

The following are insufficient by themselves:

```text
model_load_ok = true
runtime_crash_count = 0
```

Required per configuration:

```text
model_load_ok
adapter_load_ok
generation_ok
json_parse_rate
schema_valid_rate
safety_valid_rate
action_type_valid_rate
arguments_valid_rate
forbidden_action_count
tool_dispatch_ok
max_step_stop_ok
runtime_crash_count
```

All failures must remain in the report as structured diagnostics.

---

### 2.4 Teacher trajectory provenance

The 1000 `teacher_model` trajectories require:

```text
source label
generator identity
generator version/model identifier
prompt/template version
generation configuration
seed where applicable
generation timestamp
raw artifact SHA256
accepted/rejected counts
replay result
task/workspace family
```

A `teacher_model` label alone is not provenance.

The project must also state whether a remote teacher API was used. Synthetic generation is not automatically an external dataset, but its source must be explicitly authorized and auditable.

---

### 2.5 Scripted success must not be hard-coded

The statement:

```text
scripted trajectories success=True
```

requires careful review.

Allowed:

```text
success is recomputed from replay evidence and then serialized as true
```

Not allowed:

```text
success is assigned true because the trajectory is scripted
```

Required rule:

```text
trajectory source does not determine success
replay/test evidence determines success
```

---

### 2.6 CRLF/LF normalization requirements

The line-ending fix is plausible and likely necessary on Windows, but it must preserve patch semantics.

Required tests:

1. LF file + LF patch.
2. CRLF file + LF patch.
3. LF file + CRLF patch.
4. CRLF file + CRLF patch.
5. `old_text` absent after normalization.
6. `old_text` non-unique after normalization.
7. SHA mismatch still hard-fails before patching.
8. `new_text` adopts the target file line-ending convention.
9. No mixed line endings are introduced.
10. `propose_patch` and `apply_patch` calculate identical `after_sha256`.
11. Rollback restores byte-identical original content.
12. Binary and non-UTF-8 rejection remains unchanged.

Normalization must not turn two distinct matches into an unnoticed ambiguous match.

---

### 2.7 Readiness verifier independence

Removing the verifier itself from its own file list may be correct if it avoids circular/self-referential validation.

But it requires an independent test that confirms:

```text
all required gates are registered
all expected artifacts are checked
gate count cannot silently decrease
unknown/missing gates fail
the verifier cannot certify itself only through its own output
```

Recommended invariant:

```python
actual_gate_ids == EXPECTED_GATE_IDS
```

Do not accept only:

```python
len(gates) == 10
```

because a required gate could be replaced by an irrelevant passing gate.

---

### 2.8 Test evidence requirements

“Exit 0” is not enough for final delivery.

The final report should record:

```text
exact command
test count
pass count
fail count
skip count
warning count
runtime
environment
commit SHA
CI run/check status
```

Local tests and GitHub CI must be reported separately.

---

## 3. Immediate Delivery Roadmap

### Stage P4.1-Final — Evidence Closure and Merge

Objective:

```text
Turn the local candidate result into an auditable, remote, reproducible P4.1 result.
```

Required work:

1. Resolve the 45-trajectory count discrepancy.
2. Clarify 5+5 limited smoke versus 40-task evaluation.
3. Record full model-action diagnostics.
4. Confirm teacher trajectory provenance.
5. Confirm success is replay-derived.
6. Add CRLF/LF regression tests.
7. Add independent readiness-verifier coverage tests.
8. Generate final report and manifests.
9. Commit and push to a P4.1 branch.
10. Open PR referencing Issue #19.
11. Run CI.
12. Complete independent review.
13. Merge only after all blockers are closed.
14. Close Issue #19 only when all acceptance criteria are satisfied.

Endpoint:

```text
VERIFIED_GO_FOR_P4_AGENT_SFT
```

This still does not authorize training.

---

## 4. Project Phase Roadmap

### P4.1b — Protocol & Harness Ablation

Timing:

```text
After verified P4.1
Before any Agent SFT training
```

Objective:

Compare:

- Action JSON;
- compact tag protocol;
- one-line action DSL.

Keep:

- existing 11-action Safe Tool Layer;
- no unrestricted shell;
- same micro-task families;
- same generation settings;
- no training.

Endpoint:

```text
KEEP_ACTION_JSON
TRY_TAG_PROTOCOL_FOR_P4_2
TRY_DSL_FOR_P4_2
FIX_PROMPT_FIRST
FIX_EVALUATOR_FIRST
```

This phase determines the protocol used as the P4.2 training target.

---

### P4.2a — Agent SFT Training Readiness

Objective:

Validate that the selected dataset and protocol are trainable and safe on RTX 3050 4GB.

Required gates:

```text
dataset integrity
family-disjoint split
source distribution
assistant-only label mask
token retention
target truncation audit
chat template audit
GPU forward/backward/optimizer smoke
save/reload smoke
output path protection
baseline model/adapters locked
```

No full training until explicit user approval.

Endpoint:

```text
GO_FOR_P4_AGENT_SFT_PILOT
FIX_FIRST
STOP
```

---

### P4.2b — Controlled Agent SFT Pilot

Objective:

Run a small, controlled SFT pilot after explicit authorization.

Candidates should be limited and predeclared, for example:

```text
Base Qwen3-0.6B + Agent SFT
Repair-Limited LoRA + Agent SFT
```

Required controls:

- identical training sample families;
- identical protocol;
- identical evaluation set;
- versioned output directories;
- no overwrite;
- no capability claim from training loss;
- checkpoint selection independent of heldout-agent-eval.

Endpoint:

```text
GO_FOR_FULL_AGENT_SFT
GO_WITH_CAUTION
STOP_AGENT_SFT
```

---

### P4.2c — Full Agent SFT

Only if the pilot passes.

Objective:

Train the selected candidate under the verified protocol and dataset contract.

Required evidence:

- training trust gate;
- GPU safety gate;
- exact dataset SHA;
- exact adapter/base SHA;
- checkpoint history;
- save/reload;
- no parent adapter mutation;
- final report.

---

### P4.3 — Frozen Agent Loop Evaluation

Objective:

Determine whether the trained model actually repairs code through a tool loop.

Compare:

```text
Base Qwen3-0.6B
Repair-Limited LoRA
Agent-SFT candidate
```

Use heldout workspace families never used for training or checkpoint selection.

Primary metrics:

```text
task_success_rate
test-pass success
patch_success_rate
regression_rate
rollback_recovery_rate
action_validity_rate
forbidden_action_count
tool_error_rate
steps_to_success
finish_claim_mismatch
```

Endpoint:

```text
AGENT_CAPABILITY_SIGNAL
NO_AGENT_CAPABILITY_SIGNAL
PARTIAL_REPAIR_SIGNAL
```

Do not use “Builder” claims.

---

### P4.4 — Harness Optimization and Failure Recovery

Objective:

Improve system-level performance without immediately retraining.

Potential experiments:

- observation compression;
- error excerpt selection;
- memory summarization;
- repeated-action detection;
- dynamic max-step policy;
- deterministic recovery templates;
- rollback policy;
- patch-size limits;
- task-type-specific tool allowlists.

This phase should separate:

```text
model improvement
from
harness improvement
```

---

### P5 — Small Real-Repository Repair Benchmark

Only after P4.3 shows a reliable signal.

Objective:

Move from toy workspaces to small, frozen, local Python repositories with bounded defects.

Restrictions:

- small repositories;
- one or few-file repairs;
- no network;
- no Git push;
- controlled commands only;
- frozen tests;
- family/project-level train/eval isolation.

This is the first phase where the project can test whether the Repair Agent transfers beyond synthetic micro-tasks.

---

## 5. Mandatory Review Checkpoints

The technical gatekeeper should review only at these key points:

### Review A — P4.1 Pre-Merge

Check:

- Issue #19 acceptance criteria;
- dataset arithmetic;
- replay truth;
- model smoke scope;
- CRLF fix;
- readiness gates;
- CI.

### Review B — P4.1b Protocol Decision

Check:

- fair protocol comparison;
- same tasks/config;
- parser safety;
- no semantic repair;
- selected protocol justification.

### Review C — P4.2 Pre-Training Authorization

This is the most important gate.

Check:

- dataset integrity;
- assistant-only loss;
- token audit;
- GPU smoke;
- output paths;
- expected duration;
- explicit user approval.

### Review D — P4.2 Post-Training Evidence

Check:

- real execution;
- adapter hashes;
- parent immutability;
- no silent fallback;
- checkpoint selection;
- no unsupported capability claim.

### Review E — P4.3 Capability Claim

Check:

- frozen heldout families;
- same evaluator/config;
- statistical uncertainty;
- failure taxonomy;
- Builder/Repair wording.

Routine implementation PRs may be handled first by DS-Pro. Escalate only when a checkpoint above is reached or a blocking trust/safety issue appears.

---

## 6. Rules to Add to the Project Skill

1. **Remote-evidence hierarchy**

```text
local claim < committed artifact < pushed branch < PR CI < merged evidence
```

A local `exit 0` cannot produce a verified project verdict.

2. **Dataset accounting invariant**

```text
generated = accepted + rejected + quarantined
accepted = train + validation + heldout
```

No unexplained remainder is allowed.

3. **No source-based success**

Never mark a trajectory successful because it is scripted, teacher-generated, or expected to pass.

4. **Limited smoke wording**

A subset run must be labelled `LIMITED_SMOKE`. It cannot satisfy a full-suite acceptance criterion.

5. **Teacher provenance**

Every teacher-generated trajectory set must record generator, prompt version, generation config, source SHA, replay result, and acceptance status.

6. **Readiness gate identity**

Validate exact gate IDs, not only the number of passing gates.

7. **Line-ending normalization**

Normalization must preserve uniqueness checks, SHA checks, target line-ending convention, propose/apply equivalence, and byte-identical rollback.

8. **No silent exclusions**

Rejected, quarantined, skipped, or failed trajectories must remain counted in manifests and reports.

9. **Capability vocabulary**

Use:

```text
repair signal
agent-action validity
tool-loop success
heldout repair success
```

Do not use:

```text
general Builder
full coding agent
production-ready
```

without new evidence.

10. **Review cadence**

Escalate to the main technical gatekeeper only for:

- phase readiness;
- pre-training authorization;
- post-training evidence;
- capability claims;
- trust/safety blockers.
