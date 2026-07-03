# Task 2: Extend compute_router_analysis.py with P3 Decision Gate

## Location
- File to modify: `scripts/compute_router_analysis.py`
- New test file: `tests/test_router_gate.py`
- Working directory: `e:\agent\Qwen\qwen3-code-lab`
- Current git branch: `feat/p2.2-ci-router-validation`
- BASE commit (before your work): `deb8db9` (the Task 1 commit)

## Background

`scripts/compute_router_analysis.py` already computes four routing strategies across 5 P2 models and produces `reports/p2/router-analysis.json` + `reports/p2/router-analysis.md`. The four strategies are:

1. **Best Single Model** — overall pass rate is highest among the 5 models.
2. **Oracle Router** — any model passes → router passes (upper bound, not deployable).
3. **Metadata Router** — routes by `task_type` to the best model per task_type.
4. **Deployable Deterministic Router** — routes by deployment-observable signals (broken_code presence, execution_feedback presence) inferred from task_type. Same routing map as Metadata but described via observable signals.

The script already computes:
- `best_single_key`, `best_single_rate`, `best_single` stats dict (incl. `overall_pass`)
- `oracle_router` stats dict (incl. `overall_pass`, `lift_vs_best_single`)
- `deployable_router` stats dict (incl. `overall_pass`, `lift_vs_best_single`)
- `deployable_pass: dict[str, bool]` — per-sample pass/fail under the Deployable Router
- `pass_map` for each model (line 261): `{sid: _passed(by_model[key][sid]) for sid in common}`

What's MISSING is the P3 Decision Gate: a structured verdict (GO / NO-GO / SIGNAL) with statistical evidence (McNemar exact test + paired bootstrap 95% CI) comparing the Deployable Router against the Best Single model.

## The P3 Gate Criteria (from Issue #6)

- **GO**: Deployable Router lift over Best Single ≥ 5pp AND (bootstrap 95% CI lower bound > 0 OR McNemar p < 0.05)
- **NO-GO**: Oracle Router lift over Best Single < 5pp (even the upper bound shows no meaningful routing gain) OR Deployable Router shows no significant improvement
- **SIGNAL**: Oracle lift ≥ 5pp (meaningful routing potential exists) BUT Deployable Router lift < 5pp OR not statistically significant (Deployable cannot capture the potential with observable signals only)

Threshold: 5pp = 0.05 (absolute percentage-point difference in overall pass rate).

## Required Implementation

### 1. Add two helper functions (inline, near the top of the file after the existing helpers)

Copy the same implementations already used in `scripts/compute_paired_stats.py` (lines 69-108 of that file). Do NOT import from compute_paired_stats — duplicate the small functions inline to keep compute_router_analysis.py self-contained.

```python
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via binomial.
    b = #samples where A passed, B failed. c = #samples where A failed, B passed.
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k_min = min(b, c)
    p_one_tail = sum(comb(n, k) for k in range(k_min + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * p_one_tail)


def paired_bootstrap_ci(
    pass_a: list[bool], pass_b: list[bool], n_boot: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """Paired bootstrap 95% CI for the difference in pass rate (b - a)."""
    import random
    rng = random.Random(seed)
    n = len(pass_a)
    if n == 0:
        return (0.0, 0.0)
    diffs = [(1 if b else 0) - (1 if a else 0) for a, b in zip(pass_a, pass_b)]
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return (lo, hi)
```

### 2. Add a `apply_decision_gate(...)` function

Place it after the helper functions, before `main()`. This function MUST be pure (no I/O, no global state) so it is unit-testable.

```python
def apply_decision_gate(
    *,
    deployable_lift: float,
    oracle_lift: float,
    deployable_mcnemar_p: float,
    deployable_ci_lo: float,
    deployable_ci_hi: float,
    deployable_b: int,
    deployable_c: int,
    n_common: int,
) -> dict:
    """Apply the P3 Decision Gate criteria.

    Returns dict with:
      - verdict: "GO" | "NO-GO" | "SIGNAL"
      - criteria: dict of the raw thresholds and booleans used
      - reason: human-readable explanation string
    """
    LIFT_THRESHOLD = 0.05  # 5 percentage points

    oracle_meaningful = oracle_lift >= LIFT_THRESHOLD
    deployable_meaningful = deployable_lift >= LIFT_THRESHOLD
    ci_significant = (deployable_ci_lo > 0.0) or (deployable_mcnemar_p < 0.05)
    deployable_significant = deployable_meaningful and ci_significant

    if not oracle_meaningful:
        verdict = "NO-GO"
        reason = (
            f"Oracle Router lift ({oracle_lift*100:.1f}pp) < 5pp threshold — "
            "even the upper bound shows no meaningful routing gain."
        )
    elif deployable_significant:
        verdict = "GO"
        reason = (
            f"Deployable Router lift ({deployable_lift*100:.1f}pp) >= 5pp "
            f"with statistical significance (McNemar p={deployable_mcnemar_p:.4f}, "
            f"95% CI=[{deployable_ci_lo:+.4f}, {deployable_ci_hi:+.4f}])."
        )
    elif oracle_meaningful and not deployable_significant:
        verdict = "SIGNAL"
        reason = (
            f"Oracle lift ({oracle_lift*100:.1f}pp) >= 5pp (routing potential exists), "
            f"but Deployable Router lift ({deployable_lift*100:.1f}pp) or significance "
            f"(McNemar p={deployable_mcnemar_p:.4f}, CI=[{deployable_ci_lo:+.4f}, {deployable_ci_hi:+.4f}]) "
            "does not meet the GO threshold — observable signals alone cannot capture the potential."
        )
    else:
        verdict = "NO-GO"
        reason = (
            f"Deployable Router lift ({deployable_lift*100:.1f}pp) shows no significant "
            f"improvement (McNemar p={deployable_mcnemar_p:.4f}, CI=[{deployable_ci_lo:+.4f}, {deployable_ci_hi:+.4f}])."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "criteria": {
            "lift_threshold_pp": LIFT_THRESHOLD * 100,
            "oracle_lift": oracle_lift,
            "oracle_meaningful": oracle_meaningful,
            "deployable_lift": deployable_lift,
            "deployable_meaningful": deployable_meaningful,
            "deployable_mcnemar_p": deployable_mcnemar_p,
            "deployable_ci_95": [deployable_ci_lo, deployable_ci_hi],
            "deployable_ci_significant": ci_significant,
            "deployable_significant": deployable_significant,
            "deployable_mcnemar_b": deployable_b,
            "deployable_mcnemar_c": deployable_c,
            "n_common": n_common,
        },
    }
```

### 3. In `main()`, compute the gate after the deployable router section

After the existing `deployable_router = {...}` block (currently around line 377) and BEFORE the "Comparison table" section, add:

```python
# ------------------------------------------------------------------
# 5. P3 Decision Gate — Deployable Router vs Best Single
# ------------------------------------------------------------------
best_single_pass_list = [pass_map_best_single[sid] for sid in common]
deployable_pass_list = [deployable_pass[sid] for sid in common]

# McNemar counts: b = best_single passes but deployable fails;
#                  c = best_single fails but deployable passes.
gate_b = sum(1 for s in common if best_single_pass_list[i] and not deployable_pass_list[i] for i, s in enumerate(common))
gate_c = sum(1 for s in common if not best_single_pass_list[i] and deployable_pass_list[i] for i, s in enumerate(common))
# NOTE: the above enumeration is awkward — use a cleaner loop:
gate_b = 0
gate_c = 0
for sid in common:
    bs_pass = best_single_pass[<index>]  # you need to align indices
    dep_pass = deployable_pass[sid]
    # ... count b and c
```

**IMPORTANT**: Do NOT copy the awkward enumeration above verbatim. Write clean code that:
1. Builds `best_single_pass_map = {sid: _passed(by_model[best_single_key][sid]) for sid in common}` (or reuse the existing `pass_map` from line 261 — but that variable is local to the per-model loop, so you may need to re-derive it)
2. Iterates `common` once to compute `b_count` (best_single passed, deployable failed) and `c_count` (best_single failed, deployable passed)
3. Builds `pass_a` and `pass_b` lists for the bootstrap CI
4. Calls `mcnemar_exact(b_count, c_count)` and `paired_bootstrap_ci(pass_a, pass_b)`
5. Calls `apply_decision_gate(...)` with all the arguments
6. Adds the result to the `result` dict as `result["decision_gate"] = gate_result`

**Re-deriving best_single_pass_map**: The existing loop at line 261 computes `pass_map` per model but it's local. The cleanest approach is to compute the best_single pass map explicitly:
```python
best_single_pass_map = {sid: _passed(by_model[best_single_key][sid]) for sid in common}
```
This is consistent with how `deployable_pass` is built.

### 4. Add `decision_gate` to the JSON result

In the `result = {...}` dict assembly, add:
```python
"decision_gate": gate_result,
```

### 5. Add a "P3 Decision Gate" section to the markdown report

After the existing "Methodology Notes" section, add:

```python
md.append("## P3 Decision Gate")
md.append("")
md.append(f"**Verdict: {gate_result['verdict']}**")
md.append("")
md.append(f"{gate_result['reason']}")
md.append("")
md.append("### Gate Criteria")
md.append("")
md.append("| Criterion | Value | Threshold | Met |")
md.append("|-----------|-------|-----------|-----|")
c = gate_result["criteria"]
md.append(f"| Oracle lift vs Best Single | {c['oracle_lift']*100:.1f}pp | >= 5.0pp | {'YES' if c['oracle_meaningful'] else 'NO'} |")
md.append(f"| Deployable lift vs Best Single | {c['deployable_lift']*100:.1f}pp | >= 5.0pp | {'YES' if c['deployable_meaningful'] else 'NO'} |")
md.append(f"| Deployable McNemar p (2-sided) | {c['deployable_mcnemar_p']:.4f} | < 0.05 | {'YES' if c['deployable_mcnemar_p'] < 0.05 else 'NO'} |")
md.append(f"| Deployable 95% CI | [{c['deployable_ci_95'][0]:+.4f}, {c['deployable_ci_95'][1]:+.4f}] | lower > 0 | {'YES' if c['deployable_ci_95'][0] > 0 else 'NO'} |")
md.append(f"| Deployable b/c (McNemar) | {c['deployable_mcnemar_b']}/{c['deployable_mcnemar_c']} | — | — |")
md.append(f"| Common samples | {c['n_common']} | — | — |")
md.append("")
```

### 6. Create test file `tests/test_router_gate.py`

Test the `apply_decision_gate` function with these cases (use pytest, no mocks):

```python
"""Tests for the P3 Decision Gate logic in compute_router_analysis.apply_decision_gate."""
import pytest
from scripts.compute_router_analysis import apply_decision_gate


class TestDecisionGate:
    def _gate(self, **overrides):
        defaults = dict(
            deployable_lift=0.07,      # 7pp
            oracle_lift=0.10,          # 10pp
            deployable_mcnemar_p=0.01, # significant
            deployable_ci_lo=0.03,     # CI doesn't cross 0
            deployable_ci_hi=0.11,
            deployable_b=5,
            deployable_c=20,
            n_common=576,
        )
        defaults.update(overrides)
        return apply_decision_gate(**defaults)

    def test_GO_when_lift_significant_and_ci_positive(self):
        g = self._gate()
        assert g["verdict"] == "GO"

    def test_GO_when_lift_significant_and_mcnemar_significant_even_if_ci_crosses_zero(self):
        g = self._gate(deployable_ci_lo=-0.01, deployable_ci_hi=0.15, deployable_mcnemar_p=0.03)
        assert g["verdict"] == "GO"

    def test_NO_GO_when_oracle_lift_below_threshold(self):
        g = self._gate(oracle_lift=0.04)
        assert g["verdict"] == "NO-GO"
        assert "Oracle Router lift" in g["reason"]

    def test_SIGNAL_when_oracle_meaningful_but_deployable_lift_below_threshold(self):
        g = self._gate(oracle_lift=0.08, deployable_lift=0.03)
        assert g["verdict"] == "SIGNAL"

    def test_SIGNAL_when_oracle_meaningful_but_deployable_not_significant(self):
        g = self._gate(oracle_lift=0.08, deployable_lift=0.06, deployable_mcnemar_p=0.20, deployable_ci_lo=-0.02)
        assert g["verdict"] == "SIGNAL"

    def test_NO_GO_when_deployable_no_significant_improvement_and_oracle_also_low(self):
        g = self._gate(oracle_lift=0.04, deployable_lift=0.02, deployable_mcnemar_p=0.50, deployable_ci_lo=-0.03)
        assert g["verdict"] == "NO-GO"

    def test_GO_at_exact_5pp_threshold(self):
        # >= 5pp is the threshold (inclusive)
        g = self._gate(deployable_lift=0.05, oracle_lift=0.06, deployable_mcnemar_p=0.04, deployable_ci_lo=0.01)
        assert g["verdict"] == "GO"

    def test_criteria_dict_contains_all_fields(self):
        g = self._gate()
        c = g["criteria"]
        for key in ("lift_threshold_pp", "oracle_lift", "oracle_meaningful",
                    "deployable_lift", "deployable_meaningful", "deployable_mcnemar_p",
                    "deployable_ci_95", "deployable_ci_significant",
                    "deployable_significant", "deployable_mcnemar_b",
                    "deployable_mcnemar_c", "n_common"):
            assert key in c, f"missing key: {key}"

    def test_reason_string_contains_numbers(self):
        g = self._gate()
        assert "7.0pp" in g["reason"] or "10.0pp" in g["reason"]
```

Also test the helper functions:

```python
class TestMcNemarExact:
    def test_no_discordant_pairs_returns_1(self):
        from scripts.compute_router_analysis import mcnemar_exact
        assert mcnemar_exact(0, 0) == 1.0

    def test_symmetric_distribution_returns_high_p(self):
        from scripts.compute_router_analysis import mcnemar_exact
        # 10 vs 10 — no asymmetry
        p = mcnemar_exact(10, 10)
        assert p > 0.5

    def test_extreme_asymmetry_returns_low_p(self):
        from scripts.compute_router_analysis import mcnemar_exact
        # 0 vs 20 — extreme asymmetry
        p = mcnemar_exact(0, 20)
        assert p < 0.001


class TestPairedBootstrapCI:
    def test_identical_pass_lists_give_ci_centered_at_zero(self):
        from scripts.compute_router_analysis import paired_bootstrap_ci
        passes = [True] * 100 + [False] * 100
        lo, hi = paired_bootstrap_ci(passes, passes, n_boot=1000, seed=42)
        assert -0.01 < lo < 0.01
        assert -0.01 < hi < 0.01

    def test_all_b_passes_all_a_fails_gives_positive_ci(self):
        from scripts.compute_router_analysis import paired_bootstrap_ci
        a = [False] * 100
        b = [True] * 100
        lo, hi = paired_bootstrap_ci(a, b, n_boot=1000, seed=42)
        assert lo > 0.5
        assert hi > 0.5
```

## Verification

1. Run the new test file:
   ```
   D:\Anaconda\envs\qwen3-code-lab\python.exe -m pytest tests/test_router_gate.py -v
   ```
   All tests must pass.

2. Verify the script still imports correctly (eval files don't exist yet, so it will skip):
   ```
   D:\Anaconda\envs\qwen3-code-lab\python.exe -c "from scripts.compute_router_analysis import apply_decision_gate, mcnemar_exact, paired_bootstrap_ci; print('imports OK')"
   ```

3. Verify the existing CI tests still pass (don't break anything):
   ```
   D:\Anaconda\envs\qwen3-code-lab\python.exe -m pytest tests/test_p2_evidence_hardening.py tests/test_metrics.py tests/test_schemas.py tests/test_validators.py -v --tb=short
   ```

## Explicitly NOT in scope

- Do NOT modify the existing router computation logic (Best Single, Oracle, Metadata, Deployable sections).
- Do NOT change the output file paths (`router-analysis.json`, `router-analysis.md`).
- Do NOT modify `compute_paired_stats.py` (the inline duplication of McNemar/bootstrap is intentional — keep compute_router_analysis.py self-contained).
- Do NOT add the gate logic to `compute_paired_stats.py` or any other file.
- Do NOT change the MODELS list or any existing function signatures.
- Do NOT reformat existing code.

## Commit message

```
feat(router): add P3 Decision Gate with McNemar + bootstrap CI

Extends compute_router_analysis.py with:
- mcnemar_exact() and paired_bootstrap_ci() inline helpers
- apply_decision_gate() pure function returning GO/NO-GO/SIGNAL verdict
- decision_gate section in router-analysis.json output
- P3 Decision Gate section in router-analysis.md output
- tests/test_router_gate.py with 12 test cases

Gate criteria:
- GO: deployable_lift >= 5pp AND (CI_lo > 0 OR McNemar p < 0.05)
- NO-GO: oracle_lift < 5pp OR deployable no significant improvement
- SIGNAL: oracle_lift >= 5pp BUT deployable not significant
```

## Self-review checklist

- [ ] `apply_decision_gate` is a pure function (no I/O, no global state)
- [ ] McNemar and bootstrap CI functions match the implementations in compute_paired_stats.py
- [ ] `decision_gate` key added to the `result` dict in `main()`
- [ ] "P3 Decision Gate" section added to the markdown output
- [ ] All 12+ tests in `tests/test_router_gate.py` pass
- [ ] Existing CI tests (evidence_hardening, metrics, schemas, validators) still pass
- [ ] No existing router computation logic was modified
- [ ] No reformatting of existing code
