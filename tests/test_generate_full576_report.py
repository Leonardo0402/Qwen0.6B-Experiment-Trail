"""Tests for scripts/generate_full576_report.py."""
import json

import pytest

from scripts.generate_full576_report import MODELS, generate_report, main


def _minimal_comparison() -> dict:
    """Build a minimal valid full576-comparison.json structure."""
    out = {}
    for key, label in MODELS:
        out[key] = {
            "label": label,
            "metrics": {
                "pass_at_1": 0.25,
                "syntax_rate": 0.80,
                "repair_success_rate": 0.10,
                "hidden_pass_rate": 0.20,
                "format_compliance_rate": 0.95,
                "timeout_rate": 0.05,
            },
            "per_task_type": {
                "code_generation": {"total": 140, "passed": 35, "syntax_ok": 112, "format_ok": 133},
                "static_repair": {"total": 218, "passed": 21, "syntax_ok": 174, "format_ok": 207},
                "execution_repair": {"total": 218, "passed": 22, "syntax_ok": 174, "format_ok": 207},
            },
            "family_pass_count": 5,
            "family_total": 75,
        }
    return out


def _minimal_paired_stats() -> dict:
    return {
        "models": [k for k, _ in MODELS],
        "common_sample_count": 576,
        "pair_comparisons": [
            {
                "pair": ["full576-base", "full576-stage2-boundary"],
                "sample_compare": {
                    "n_compared": 576,
                    "win": 10,
                    "loss": 5,
                    "unchanged": 561,
                    "rate_a": 0.25,
                    "rate_b": 0.30,
                    "delta": 0.05,
                    "mcnemar_b": 5,
                    "mcnemar_c": 10,
                    "mcnemar_p_two_sided": 0.30,
                    "bootstrap_95ci": [0.005, 0.095],
                },
                "family_compare": {
                    "n_families": 75,
                    "families_gained": [],
                    "families_lost": [],
                    "net_gain": 0,
                    "pass_a": 5,
                    "pass_b": 6,
                },
            }
        ],
        "per_model_bug_type_repair": {
            "full576-base": {
                "off_by_one": {"total": 30, "passed": 3, "pass_rate": 0.1},
                "return_value_error": {"total": 0, "passed": 0, "pass_rate": 0.0},
            },
        },
    }


def _minimal_router_analysis() -> dict:
    return {
        "models_loaded": [k for k, _ in MODELS],
        "models_skipped": [],
        "common_sample_count": 576,
        "best_single": {
            "model": "Base",
            "model_key": "full576-base",
            "overall_pass": 0.25,
            "family_pass": 0.067,
            "per_task_type": {},
            "n_samples": 576,
        },
        "oracle_router": {
            "overall_pass": 0.40,
            "family_pass": 0.10,
            "lift_vs_best_single": 0.15,
        },
        "metadata_router": {
            "overall_pass": 0.30,
            "family_pass": 0.08,
            "routing_map": {},
            "lift_vs_best_single": 0.05,
        },
        "deployable_router": {
            "overall_pass": 0.28,
            "family_pass": 0.07,
            "routing_map": {},
            "lift_vs_best_single": 0.03,
        },
        "decision_gate": {
            "verdict": "SIGNAL",
            "reason": "Oracle lift >= 5pp but deployable not significant.",
            "criteria": {
                "lift_threshold_pp": 5.0,
                "oracle_lift": 0.15,
                "oracle_meaningful": True,
                "deployable_lift": 0.03,
                "deployable_meaningful": False,
                "deployable_mcnemar_p": 0.20,
                "deployable_ci_95": [-0.02, 0.08],
                "deployable_ci_significant": False,
                "deployable_significant": False,
                "deployable_mcnemar_b": 5,
                "deployable_mcnemar_c": 10,
                "n_common": 576,
            },
        },
        "comparison_table": [
            {
                "name": "Base",
                "type": "single_model",
                "overall_pass": 0.25,
                "family_pass": 0.067,
                "code_generation_pass": 0.25,
                "static_repair_pass": 0.10,
                "execution_repair_pass": 0.10,
                "lift_vs_best_single": 0.0,
            }
        ],
    }


def test_generates_report_with_mock_data(tmp_path, monkeypatch):
    """Create temp JSON files with minimal valid data, run main, verify output exists."""
    (tmp_path / "full576-comparison.json").write_text(
        json.dumps(_minimal_comparison()), encoding="utf-8"
    )
    (tmp_path / "full576-paired-stats.json").write_text(
        json.dumps(_minimal_paired_stats()), encoding="utf-8"
    )
    (tmp_path / "router-analysis.json").write_text(
        json.dumps(_minimal_router_analysis()), encoding="utf-8"
    )

    import scripts.generate_full576_report as mod
    monkeypatch.setattr(mod, "COMPARISON_PATH", tmp_path / "full576-comparison.json")
    monkeypatch.setattr(mod, "PAIRED_STATS_PATH", tmp_path / "full576-paired-stats.json")
    monkeypatch.setattr(mod, "ROUTER_ANALYSIS_PATH", tmp_path / "router-analysis.json")
    out_path = tmp_path / "p2-full576-comparison-report.md"
    monkeypatch.setattr(mod, "OUTPUT_PATH", out_path)

    rc = main()
    assert rc == 0
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")

    expected_sections = [
        "# P2 Full-576 Comparison Report",
        "## Evaluation Setup",
        "## Overall Metrics",
        "## Per-Task-Type Breakdown",
        "## Family-Level Pass",
        "## Paired Statistics Summary",
        "## Bug-Type Repair Success Rate",
        "## Router Feasibility Summary",
        "## P3 Decision Gate",
    ]
    for section in expected_sections:
        assert section in content, f"missing section header: {section}"


def test_missing_input_returns_error(tmp_path, monkeypatch):
    """Verify main() returns 1 when input files are missing."""
    import scripts.generate_full576_report as mod
    monkeypatch.setattr(mod, "COMPARISON_PATH", tmp_path / "nope-1.json")
    monkeypatch.setattr(mod, "PAIRED_STATS_PATH", tmp_path / "nope-2.json")
    monkeypatch.setattr(mod, "ROUTER_ANALYSIS_PATH", tmp_path / "nope-3.json")
    monkeypatch.setattr(mod, "OUTPUT_PATH", tmp_path / "out.md")

    rc = main()
    assert rc == 1
    assert not (tmp_path / "out.md").exists()


def test_percentage_formatting():
    """Verify pass rates are formatted as percentages with 1 decimal place."""
    md = generate_report(
        _minimal_comparison(), _minimal_paired_stats(), _minimal_router_analysis()
    )
    # pass_at_1=0.25 -> "25.0%"
    assert "25.0%" in md
    # syntax_rate=0.80 -> "80.0%"
    assert "80.0%" in md
    # timeout_rate=0.05 -> "5.0%"
    assert "5.0%" in md


def test_decision_gate_verdict_displayed():
    """Verify the P3 Decision Gate verdict is present in the output."""
    md = generate_report(
        _minimal_comparison(), _minimal_paired_stats(), _minimal_router_analysis()
    )
    assert "**Verdict: SIGNAL**" in md
    # The reason should also be present
    assert "Oracle lift >= 5pp but deployable not significant." in md
    # Gate criteria table should be present
    assert "### Gate Criteria" in md
    assert "Oracle lift" in md
    assert "Deployable lift" in md
    assert "McNemar p" in md
    assert "95% CI" in md


def test_all_five_models_in_table():
    """Verify all 5 model labels appear in the overall metrics table."""
    md = generate_report(
        _minimal_comparison(), _minimal_paired_stats(), _minimal_router_analysis()
    )
    for _, label in MODELS:
        assert label in md, f"missing model label in table: {label}"
