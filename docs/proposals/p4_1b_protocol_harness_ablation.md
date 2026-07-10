# P4.1b: Protocol & Harness Ablation for Tiny Repair Agent

> Project: `Qwen0.6B-Experiment-Trail`  
> Proposed phase: **P4.1b — after P4.1, before P4.2 training**  
> Status: Proposal / future issue seed  
> Core direction: **Constrained Local Repair Agent**, not Builder  
> Purpose: Decide whether Qwen3-0.6B should keep using Action JSON, switch to a lower-entropy protocol, or require prompt/data repair before Agent SFT.

---

## 1. Decision

This direction is valuable, but it should **not** be inserted into P4.1.

P4.1 already has a clear scope:

```text
ModelActionProvider Smoke + Agent SFT Data Builder
```

P4.1 must first validate whether Qwen3-0.6B can emit legal Action JSON and whether the current evaluator/data-builder pipeline is trustworthy.

The new direction should become:

```text
P4.1b: Protocol & Harness Ablation for Tiny Repair Agent
```

It should run **after P4.1 completes** and **before P4.2 Agent SFT training**.

Reason:

```text
Before training the model on Agent trajectories, we should determine which action protocol and harness structure is easiest for a 0.6B model to follow safely and consistently.
```

---

## 2. Background

The current project conclusion is:

```text
Qwen3-0.6B is not suitable as a general Builder.
It is more promising as a Constrained Local Repair Agent for small Python tasks.
```

This means the model should not be expected to perform large-scale architecture design, multi-file planning, or unrestricted shell-based coding.

Instead, it should operate inside a strict loop:

```text
observe
→ choose one safe action
→ receive tool result
→ patch narrowly
→ run tests
→ inspect failure
→ rollback or continue
→ finish only when replay/test evidence supports success
```

The proposed P4.1b experiment asks:

```text
Given this constrained repair-agent role, what is the best action protocol and harness design for Qwen3-0.6B?
```

---

## 3. Why This Direction Is Valuable

The idea has real value because small models often fail not only from weak reasoning, but from excessive interface complexity.

For Qwen3-0.6B, the following factors can dominate performance:

- action format entropy;
- prompt length;
- number of available tools;
- observation length;
- whether actions are recoverable;
- whether invalid outputs hard-fail or silently no-op;
- whether the model is asked to generate nested JSON, tag blocks, or a compact DSL;
- whether the harness protects the model from excessive freedom.

This phase does not ask:

```text
Can 0.6B become a full coding agent?
```

It asks the sharper and more useful question:

```text
Under which protocol and harness constraints can 0.6B reliably behave as a local repair worker?
```

That question directly supports future Agent SFT.

---

## 4. Relationship to Existing Phases

### P4.0

P4.0 built the foundation:

- Action Schema;
- Safe Tool Layer;
- Micro Task Suite;
- Trajectory Schema;
- Evaluator;
- Readiness Gate.

P4.1b must reuse these artifacts and must not replace them with an unrestricted framework.

### P4.1

P4.1 remains the current mainline phase.

P4.1 validates:

- `ModelActionProvider`;
- Action JSON parsing;
- schema-valid actions;
- safety-valid actions;
- 1000+ replayable Agent SFT trajectories;
- no training;
- no external data;
- no model replacement.

P4.1b should not block the creation of the P4.1 spec/plan.

### P4.1b

P4.1b compares protocols and harness variants.

It should be run only after P4.1 establishes a trustworthy baseline.

### P4.2

P4.2 is future Agent SFT training.

P4.2 should not start until P4.1b answers whether the training target should remain Action JSON or use a lower-entropy protocol.

---

## 5. Core Hypothesis

### Primary hypothesis

```text
For Qwen3-0.6B, a lower-entropy action protocol may improve parse validity, schema validity, safety validity, and loop stability compared with full Action JSON.
```

### Secondary hypothesis

```text
The existing 11-action Safe Tool Layer is safer and more useful than a generic read_file/write_file/run_terminal harness.
```

### Negative hypothesis

```text
If protocol changes do not improve action validity or safety, keep Action JSON and improve prompt/data quality instead.
```

---

## 6. Candidate Protocols

### Protocol A — Existing Action JSON Baseline

Example:

```json
{
  "action_type": "read_file",
  "path": "solution.py"
}
```

Purpose:

```text
Baseline protocol from P4.1.
```

Pros:

- strict schema validation;
- easy Pydantic integration;
- strong auditability;
- clean dataset format.

Cons:

- small models may emit malformed JSON;
- nested fields may increase format errors;
- repair layer must avoid changing semantics.

---

### Protocol B — Compact XML / Tag Protocol

Example:

```text
<think>
Need to inspect the failing file before patching.
</think>
<action>
tool: read_file
path: solution.py
</action>
```

Purpose:

```text
Test whether tag-based action output is easier for Qwen3-0.6B than strict JSON.
```

Pros:

- lower formatting burden;
- regex parser can be simple;
- easier to separate reasoning and action;
- may be more robust under small-model generation.

Cons:

- less naturally schema-typed;
- parser ambiguity risk;
- must prevent free-form action arguments;
- may encourage verbose `<think>` output.

---

### Protocol C — One-Line Action DSL

Example:

```text
ACTION read_file path=solution.py
```

Another example:

```text
ACTION apply_patch patch_id=patch_001
```

Purpose:

```text
Minimize action entropy as much as possible.
```

Pros:

- easiest to parse;
- compact;
- good for small models;
- can hard-fail unknown keys.

Cons:

- less expressive;
- escaping multiline content is harder;
- may require patch content to be stored separately by `propose_patch`.

---

## 7. Harness Principle

P4.1b should **not** introduce unrestricted terminal execution.

Do not give the model:

```text
run_terminal(command)
```

Instead, retain the existing constrained tool/action layer:

```text
list_files
read_file
search_text
inspect_task
propose_patch
apply_patch
rollback_patch
run_tests
inspect_error
write_memory
finish
```

Reason:

```text
The goal is not to maximize freedom.
The goal is to maximize safe repair-loop reliability.
```

The harness should remain:

- path-restricted;
- patch-oriented;
- test-driven;
- rollback-capable;
- replayable;
- deterministic where possible;
- auditable through trajectory logs.

---

## 8. Scope

P4.1b includes:

1. Define protocol variants.
2. Implement deterministic parsers for each protocol.
3. Reuse the same 40 micro-tasks from P4.0/P4.1.
4. Reuse the same safe tool layer.
5. Run ModelActionProvider smoke under each protocol.
6. Compare action format stability and loop safety.
7. Produce a protocol comparison report.
8. Decide whether P4.2 should train on JSON, tag protocol, DSL, or defer training.

---

## 9. Out of Scope

P4.1b must not include:

- SFT training;
- DPO / PPO / GRPO / RL training;
- model replacement;
- external datasets;
- unrestricted shell;
- network-enabled tools;
- Git push tools;
- large framework migration;
- LangChain / AutoGen integration;
- broad Builder capability claims.

---

## 10. Metrics

P4.1b should compare protocols using the same tasks and equivalent prompts.

Required metrics:

```text
parse_success_rate
schema_valid_rate
safety_valid_rate
action_type_valid_rate
arguments_valid_rate
forbidden_action_count
unknown_action_count
tool_error_rate
finish_without_tests_count
finish_claim_mismatch_count
max_steps_hit_rate
runtime_crash_count
task_success_rate
patch_success_rate
rollback_success_rate
```

Important rule:

```text
Task success is not the primary metric.
Protocol reliability and safety are the primary metrics.
```

Low task success is acceptable if action validity and safety are measurable.

---

## 11. Required Failure Taxonomy

Every failed trajectory should be classified.

Suggested failure classes:

```text
FORMAT_PARSE_FAIL
SCHEMA_VALIDATION_FAIL
UNKNOWN_ACTION_TYPE
FORBIDDEN_ACTION
INVALID_PATH
EMPTY_OR_USELESS_ACTION
TOOL_ERROR
PATCH_APPLY_FAIL
TEST_FAIL
FINISH_WITHOUT_TESTS
FINISH_CLAIM_MISMATCH
MAX_STEPS_EXCEEDED
MODEL_REFUSAL_OR_CHATTER
OBSERVATION_OVERFLOW
REPEATED_ACTION_LOOP
```

---

## 12. Acceptance Criteria

P4.1b is complete only if:

1. Existing Action JSON remains the baseline.
2. At least two alternative protocols are specified and tested.
3. Each protocol has a deterministic parser.
4. Unknown or malformed actions hard-fail.
5. No silent no-op exists.
6. All protocol runs use the same task set.
7. All protocol runs record reproducible generation settings.
8. All tool calls are replayable.
9. Safety violations are counted.
10. A protocol comparison report is produced.
11. Final verdict is one of:

```text
KEEP_ACTION_JSON
TRY_TAG_PROTOCOL_FOR_P4_2
TRY_DSL_FOR_P4_2
FIX_PROMPT_FIRST
FIX_EVALUATOR_FIRST
STOP_PROTOCOL_CHANGE
```

---

## 13. Suggested 8-Task SDD Plan

### T1 — Baseline Lock

Record:

- P4.1 endpoint commit;
- micro-task manifest SHA;
- evaluator SHA;
- model config;
- adapter config if used;
- generation config.

### T2 — Protocol Specification

Write formal grammar for:

- Action JSON;
- tag protocol;
- one-line DSL.

### T3 — Parser Unit Tests

For each protocol:

- valid action parses;
- malformed action fails;
- unknown action fails;
- forbidden path fails;
- multiline content behavior is defined.

### T4 — Prompt Templates

Create minimal prompt templates for each protocol.

Each prompt must expose the same tool semantics.

### T5 — ModelActionProvider Protocol Adapter

Add a protocol abstraction around model output parsing.

The model still emits one action per step.

### T6 — Smoke Run on 40 Micro-Tasks

Run each protocol on the same tasks with deterministic generation.

No training.

### T7 — Protocol Comparison Report

Produce metrics, failure taxonomy, examples, and risk assessment.

### T8 — Gatekeeper Decision

Decide whether P4.2 should use JSON, tag protocol, DSL, or delay training.

---

## 14. PR Relationship

Recommended issue title:

```text
P4.1b: Protocol & Harness Ablation for Tiny Repair Agent
```

Recommended PR title:

```text
exp(p4.1b): compare action protocols for tiny repair agent
```

Recommended relationship:

```text
Refs #19
```

Do not use `Closes #19` unless this phase is explicitly merged into the same acceptance criteria as P4.1.

If P4.1 already closes #19, this phase should have a separate issue.

---

## 15. Gatekeeper Review Checklist

A PR for this phase should be blocked if it:

- starts training;
- changes model family;
- introduces external datasets;
- gives the model unrestricted shell;
- bypasses the safe tool layer;
- accepts malformed actions silently;
- allows unknown actions to no-op;
- treats model self-declared success as real success;
- lacks replay evidence;
- lacks parser tests;
- lacks failure taxonomy;
- compares protocols using different task sets or generation settings.

---

## 16. Final Recommendation

This direction is worth keeping.

But it should be positioned as:

```text
A protocol/harness ablation before Agent SFT training.
```

Not as:

```text
A replacement for P4.1.
```

The correct sequencing is:

```text
P4.1  → prove current JSON/action pipeline and build replayable SFT data
P4.1b → compare JSON vs tag vs DSL protocol reliability
P4.2  → train only after deciding the best protocol target
P4.3  → evaluate real agent loop behavior
```

This protects the project from premature training and prevents the model from being optimized toward a protocol that may be unnecessarily difficult for a 0.6B model.
