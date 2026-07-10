# Qwen0.6B-Experiment-Trail — Trae Multi-Session Workflow

## Recommended Session Topology

Use three permanent roles. A fourth session is optional only for major milestone review.

| Session | Default model | Primary skill | Writes code? | Purpose |
|---|---|---|---:|---|
| A. Decision | Strong reasoning model / DS | `technical-decision-brief` + `brainstorming` | No | Translate architecture choices and produce an approved decision record |
| B. Development | GLM 5.2 | `using-superpowers`, `writing-plans`, `test-driven-development`, execution skills | Yes | Implement only the approved decision/spec/plan |
| C. Verification & Gate | DS in a fresh session | `independent-verification-gate`, then `qwen-experiment-trail-review-gate` | Read-only first | Recompute evidence, review PR, decide merge/Issue/phase |
| D. External milestone review | GPT / another independent model | project gate | No | Optional for training authorization, capability claims, or disputed blockers |

## Why Not Put Everything in One Session

A single session accumulates three biases:

1. **Design anchoring** — the model prefers its first brainstorming idea.
2. **Implementation ownership** — the model explains away defects in code it wrote.
3. **Completion bias** — after a long task, the model tends to accept its own tests and report.

Fresh sessions reduce these biases even when the underlying model is the same.

## Why Not Create Too Many Sessions

More than three routine sessions creates synchronization cost and conflicting edits. Use artifacts and commit SHAs as the communication layer, not copied chat history.

## Artifact Handoff Contract

### Decision Session produces

```text
docs/superpowers/decisions/YYYY-MM-DD-<slug>.md
```

Status must remain `PROPOSED` until the user approves it.

### Development Session consumes

- exact Issue;
- approved decision record;
- spec;
- plan;
- declared base commit.

It produces:

- implementation and tests;
- report;
- commit SHA;
- pushed branch / PR.

### Verification Session consumes

- exact base/head commits;
- Issue;
- decision/spec/plan;
- PR diff;
- reports/manifests;
- local and CI evidence.

It produces:

- traceability matrix;
- independent test evidence;
- blockers;
- merge/Issue/phase verdict.

## Shared Working Tree Rules for Trae

Because Trae sessions do not branch conversation state, use Git as the state boundary.

1. Only the development session edits production files during implementation.
2. Decision session is read-only.
3. Verification session is read-only until a separate remediation task is authorized.
4. Every handoff states exact `HEAD` SHA and branch.
5. A verdict applies only to the exact reviewed SHA.
6. Any later edit invalidates the previous verification.
7. Never let two sessions edit the working tree concurrently.
8. Prefer pushed branches and PR diffs for final review; do not review a moving uncommitted workspace.

## Recommended Flow with Superpowers

```text
/using-superpowers
→ /brainstorming
→ technical-decision-brief (fresh Decision session when material choices exist)
→ user approves decision record
→ /writing-plans
→ plan review by project gate
→ /executing-plans or subagent-driven development
→ /test-driven-development per task
→ /verification-before-completion
→ fresh Independent Verification session
→ PR creation
→ fresh PR Merge Review
→ merge / fix / next Issue
```

## When a Separate Decision Session Is Required

Open it when any answer is `yes`:

- Does the choice affect architecture, public interfaces, schemas, data formats, evaluator truth, safety, training, or long-term maintenance?
- Are two or more options plausible?
- Is the choice hard to reverse?
- Does the user not understand the technical consequences?
- Is the development model strongly recommending its own preferred option?

Do not open it for trivial naming, formatting, or obviously local implementation details.

## When a Separate Verification Session Is Required

Always use it before:

- closing an Issue;
- merging a non-trivial PR;
- claiming an experiment milestone;
- authorizing training;
- claiming model improvement or agent capability;
- creating the next phase Issue.

For small documentation-only edits, the same review session may be sufficient.

## Model Allocation Recommendation

### GLM 5.2

Best used for:

- sustained implementation;
- local repository navigation;
- TDD loops;
- plan execution;
- focused bug fixing.

### DS

Best used for:

- architecture challenge;
- decision comparison;
- independent regression review;
- Issue/PR acceptance mapping;
- next-step design.

Do not rely on model identity alone. Role isolation, fresh context, exact commit targeting, and executable evidence matter more.

## Minimal Routine

For ordinary implementation work:

```text
Decision session only when needed
Development session always
Verification/Gate session always before merge
```

This is the lowest-overhead configuration that still provides meaningful independence.
