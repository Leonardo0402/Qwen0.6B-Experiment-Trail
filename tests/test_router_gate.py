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
