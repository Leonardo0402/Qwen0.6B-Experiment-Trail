# SUPERSEDED FOR REPRODUCIBILITY — P4.1b Protocol Ablation v2

**Status**: SUPERSEDED for reproducibility by `reports/p4/protocol-ablation-v3/`
**Date**: 2026-07-12
**Issue**: #32

## Reason for Supersession

The v2 strict-validation metrics may be informative — the Pydantic
`extra="forbid"` fix, independent dimension checks, and numerator/
denominator reporting are all correct improvements over v1.

**However**, the v2 experiment was run from an uncommitted working tree.

### Root Cause

The v2 `baseline-lock.json` records:

```json
{
  "experiment_commit_sha": "d034fde"
}
```

But the actual experiment used `d034fde` + uncommitted working-tree
fixes that were not committed until later. This means:

```powershell
git checkout d034fde
py -3.11 scripts/run_protocol_ablation.py
```

**cannot reproduce** the v2 results. The `source_file_shas` in the v2
baseline lock are correct as supplementary evidence, but they do not
substitute for a real, checkout-able experiment commit.

### Issue #32 Acceptance Criterion

> The 240-run must be rerun from a recorded committed state.

v2 does not meet this criterion.

## What Is Preserved

The following artifacts are preserved as experimental history:

- `trajectories/*.jsonl` — v2 trajectories (strict validation)
- `baseline-lock.json` — v2 baseline lock (commit SHA is real but
  does not include the experiment code)
- `comparison-matrix.json` — v2 metrics (informative but not final)
- `failure-taxonomy.json` — v2 taxonomy (informative but not final)
- `comparison-report.md` — v2 report (informative but not final)
- `verdict.json` — v2 verdict (FIX_PROMPT_FIRST, informative but
  not final)

## Correct Usage

- **Do NOT** treat v2 as the final reproducible P4.1b result.
- **Do NOT** cite v2 metrics as final without also citing v3.
- **DO** refer to `reports/p4/protocol-ablation-v3/` for the final
  reproducible committed-state results.
- **DO** refer to `reports/2026-07-12/p4-1b-trust-repair-final-report.md`
  for the full final trust repair report.
- **MAY** use v2 metrics as supplementary evidence alongside v3, with
  explicit notation that v2 lacks committed-state provenance.

## Additional Context

v2 also predates the Issue #32 Final Trust Repair semantic fixes:

1. **Semantic default injection**: v2 Tag/DSL parsers still injected
   `success_criterion`, `tests_passed`, and `identification_verified`
   defaults for `finish` actions via `arguments.setdefault(...)`.
2. **Silent boolean coercion**: v2 Tag/DSL parsers mapped invalid
   boolean values (e.g. `tests_passed=banana`) to `False` instead
   of hard-failing.
3. **Silent numeric pass-through**: v2 Tag/DSL parsers caught
   `ValueError` from `int()`/`float()` and passed the original
   string through instead of hard-failing.

These were fixed in commit `6da4c0e` (the v3 experiment commit).

## Related

- Issue #32: Trust Repair acceptance criteria (Blocker A — reproducibility)
- PR #30: Updated with v3 results
- v1: `reports/p4/protocol-ablation/SUPERSEDED.md` (TRUST INVALID)
- v3: `reports/p4/protocol-ablation-v3/` (final committed-state rerun)
