# SUPERSEDED — P4.1b Protocol Ablation v1

**Status**: SUPERSEDED by `reports/p4/protocol-ablation-v2/`
**Date**: 2026-07-12
**Issue**: #32

## Reason for Supersession

The results in this directory are **not trustworthy** and must not be used
for P4.2 training initialization decisions.

### Root Cause

All Pydantic models (ActionBase, SafetyFlags, and every Args model) used
the default `extra="ignore"`, which caused unknown fields in model output
to be **silently dropped** before validation. For example:

```json
{
  "action_type": "list_files",
  "arguments": {
    "path": "solution.py"
  }
}
```

The `path` field is not a valid argument for `list_files` (the correct
field is `pattern`). With `extra="ignore"`, Pydantic silently dropped
`path`, and the action was recorded as:

```
schema_valid = true
arguments_valid = true
```

This **inflated** `schema_valid_rate` to a reported 96.25% that did not
reflect true schema compliance.

### Additional Issues Fixed in v2

1. `unknown_action_count` counted FORMAT_PARSE_FAIL steps as unknown
   actions (double-counting bug).
2. `finish_without_tests_count` used `success and not tests_passed`,
   which missed failed trajectories that finished without tests.
3. ProtocolDiagnostics set all four validity dimensions
   (schema_valid/safety_valid/action_type_valid/arguments_valid) to True
   together, making it impossible to identify which dimension failed.
4. Reports showed only percentages without numerator/denominator,
   preventing traceability.

## What Is Preserved

The following artifacts are preserved as experimental history:
- `trajectories/*.jsonl` — original 240 trajectories
- `baseline-lock.json` — original baseline lock
- `comparison-matrix.json` — original metrics (untrusted)
- `failure-taxonomy.json` — original taxonomy (untrusted)
- `comparison-report.md` — original report (untrusted)
- `verdict.json` — original verdict (KEEP_ACTION_JSON, untrusted)

## Correct Usage

- **Do NOT** use any metrics from this directory for training decisions.
- **Do NOT** cite the 96.25% schema_valid_rate as evidence.
- **Do NOT** use the verdict (KEEP_ACTION_JSON) as final.
- **DO** refer to `reports/p4/protocol-ablation-v3/` for the final
  reproducible committed-state results.
- **DO** refer to `reports/2026-07-12/p4-1b-trust-repair-final-report.md`
  for the full final trust repair report.

Note: v2 (`reports/p4/protocol-ablation-v2/`) is also superseded for
reproducibility reasons — see v2's `SUPERSEDED_FOR_REPRODUCIBILITY.md`.
v3 is the only final candidate.

## Related

- Issue #32: Trust Repair acceptance criteria
- PR #30: Updated with v2 results
- Commit: see `experiment_commit_sha` in v2 baseline-lock.json
