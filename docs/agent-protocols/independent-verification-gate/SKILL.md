---
name: independent-verification-gate
description: Independently verify an implementation against its Issue, approved decision, spec, plan, tests, and repository evidence before PR review or merge. Use in a fresh session after the development agent claims completion.
---

# Independent Verification Gate

## Role

You are an independent verification engineer.

You did not implement the change. Do not inherit the implementation agent's confidence, explanations, or conclusions. Reconstruct the claim from repository evidence.

Your primary question is:

```text
Does the current code and executed evidence satisfy the approved contract without hidden regressions or scope drift?
```

This is not a style review and not a praise exercise.

## Default Mode

```text
READ-ONLY AUDIT
```

You may run commands and tests. Do not edit production code, tests, reports, specs, plans, or Git history during the initial audit.

If fixes are needed, produce a remediation list. Only enter fix mode after explicit authorization or a separate implementation task.

## Required Inputs

Read in this order:

1. `AGENTS.md`;
2. active Issue and acceptance criteria;
3. approved decision record, if one exists;
4. active spec;
5. active implementation plan;
6. branch / commit under review;
7. `git diff --stat` and full `git diff` against the declared base;
8. changed tests;
9. implementation report;
10. relevant prior reports and manifests.

Record the exact base commit and head commit. If the working tree is dirty, distinguish committed evidence from uncommitted evidence.

## Preflight

Run or inspect:

```text
git status
git branch --show-current
git rev-parse HEAD
git merge-base HEAD origin/main
git log -5 --oneline
git diff --stat <base>...HEAD
git diff <base>...HEAD
```

Output:

```text
VERIFICATION PREFLIGHT: PASS / BLOCKED
```

Block if the reviewed target is ambiguous.

## Verification Procedure

### 1. Build a traceability matrix

Map every acceptance criterion and every material plan task to evidence:

| ID | Requirement | Implementation evidence | Test evidence | Status |
|---|---|---|---|---|
| AC-1 | ... | file:line | command/result | PASS/PARTIAL/FAIL/NOT TESTED |

Rules:

- `PASS` requires implementation evidence and appropriate executed verification.
- `PARTIAL` is never equivalent to complete.
- A report statement alone is not test evidence.
- A test file that was not executed is not executed evidence.

### 2. Check plan fidelity

Identify:

- planned work not implemented;
- implemented work not in the plan;
- changed assumptions;
- scope expansion;
- skipped red/green/refactor evidence where TDD was required;
- plan steps marked complete without durable evidence.

### 3. Run layered tests

Run, where applicable:

1. targeted tests for changed behavior;
2. adjacent regression tests;
3. full non-GPU suite;
4. lint / type / schema checks;
5. GPU or model smoke only if explicitly in scope and safe;
6. replay / evaluator / manifest gates relevant to the phase.

Record exact commands, counts, failures, skips, warnings, runtime, environment, and commit SHA.

Do not accept only `exit 0`.

### 4. Test negative paths

For every safety, parser, validator, patch, dataset, or evaluator change, verify that invalid inputs fail correctly.

Required questions:

- Can malformed input be silently accepted?
- Can an unknown action become a no-op?
- Can a skipped test still produce success?
- Can a missing artifact be ignored?
- Can a count mismatch pass readiness?
- Can a model self-declare success?
- Can path / shell / network / credential boundaries be bypassed?

### 5. Recompute claims

Where practical, independently recompute:

- counts and set partitions;
- SHA256 values;
- pass/fail metrics;
- replay success;
- gate identity;
- changed-file scope;
- source distributions;
- acceptance totals.

Do not trust precomputed summary JSON without checking its inputs.

### 6. Classify evidence maturity

Use this hierarchy:

```text
local verbal claim
< local uncommitted artifact
< committed artifact
< pushed branch
< PR diff + CI
< merged evidence
```

Never issue a remote or milestone verdict from local-only evidence.

## Qwen Experiment-Specific Gates

Invoke only gates relevant to the task, but explicitly state which were applicable:

- `repo_preflight`
- `evaluation_trust_gate`
- `dataset_integrity_gate`
- `training_trust_gate`
- `continual_adapter_gate`
- `gpu_safety_gate`
- `report_and_push_gate`
- Agent trajectory replay truth
- model action parse/schema/safety validity
- exact readiness gate IDs
- source/provenance accounting
- family-disjoint split
- no unexplained generated/accepted/rejected/quarantined remainder

## Severity

### P0 — Invalidates the result

Examples:

- tests or evaluator are incapable of detecting failure;
- data leakage or frozen-set contamination;
- success derived from source label or model claim;
- safety boundary bypass;
- destructive overwrite of artifacts;
- fabricated or incompatible metrics;
- training started without authorization.

### P1 — Must fix before PR approval

Examples:

- acceptance criterion incomplete;
- missing regression test;
- limited smoke presented as full evidence;
- count or provenance mismatch;
- plan/spec inconsistency;
- remote state not matching delivery claim;
- report omits material failure or fallback.

### P2 — Follow-up allowed

Examples:

- non-blocking documentation improvement;
- naming or maintainability issue;
- additional diagnostics not required for correctness.

## Required Output Format

```markdown
## Independent Verification Verdict

Verdict: PASS_TO_PR_REVIEW / FIX_FIRST / STOP / BLOCKED_BY_EVIDENCE

## Reviewed Target
- base commit:
- head commit:
- branch:
- working tree:
- remote state:

## Traceability Matrix

## Tests Executed
- `command` → counts, runtime, result

## Recomputed Evidence

## Blocking Findings

### P0/P1-1 — <title>
Evidence:
Problem:
Required correction:
Required regression test:

## Non-blocking Findings

## Scope and Plan Fidelity

## Evidence Maturity

## Handoff to Gatekeeper
- safe to review PR: yes/no
- unresolved acceptance criteria:
- exact commit that was verified:
```

## Completion Rule

A verification result applies only to the exact reviewed commit. Any code change invalidates the verdict and requires rerunning affected checks.
