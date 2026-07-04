"""Tests for scripts/compute_router_analysis.py — Task 2 holdout policy application.

Verifies that the router analysis script:
1. Loads the frozen router policy from reports/p2/router-policy-v1.json.
2. Validates that the loaded evals match the policy's selection_dataset_sha256.
3. Restricts all router evaluation to the held-out eval_sample_ids.
4. Uses the FROZEN Best Single / Metadata / Deployable choices from the policy.
5. Recomputes Oracle Router on the eval subset (upper bound, not "trained").

Uses synthetic in-memory fixtures — does NOT touch real eval files.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from scripts.compute_router_analysis import (
    MODELS,
    TASK_TYPES,
    _filter_to_eval_subset,
    _load_policy,
    _passed,
    _validate_policy_alignment,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

# Use the first two MODELS keys as our synthetic "model_a" / "model_b".
_MODEL_A = MODELS[0][0]  # "full576-base"
_MODEL_B = MODELS[1][0]  # "full576-stage2-boundary"

_DATASET_SHA = "abc123def456"  # synthetic SHA for fixture alignment

# 4 selection families + 4 eval families = 8 total. 2 samples each = 16 samples.
_SELECTION_FAMILIES = [f"fam_sel_{i}" for i in range(4)]
_EVAL_FAMILIES = [f"fam_eval_{i}" for i in range(4)]
_ALL_FAMILIES = _SELECTION_FAMILIES + _EVAL_FAMILIES


def _make_synthetic_policy() -> dict[str, Any]:
    """Build a synthetic policy artifact.

    4 selection families / 4 eval families, 2 samples per family (16 total).
    best_single = _MODEL_A; metadata and deployable maps route static/exec
    repair to _MODEL_B.
    """
    selection_sample_ids: list[str] = []
    eval_sample_ids: list[str] = []
    for fam_id in _ALL_FAMILIES:
        for s_idx in range(2):
            sample_id = f"{fam_id}_s{s_idx}"
            if fam_id in _SELECTION_FAMILIES:
                selection_sample_ids.append(sample_id)
            else:
                eval_sample_ids.append(sample_id)
    return {
        "policy_version": "v1",
        "created_at": "2026-07-04T00:00:00Z",
        "selection_dataset_sha256": _DATASET_SHA,
        "selection_family_count": len(_SELECTION_FAMILIES),
        "eval_family_count": len(_EVAL_FAMILIES),
        "selection_families": list(_SELECTION_FAMILIES),
        "eval_families": list(_EVAL_FAMILIES),
        "selection_sample_ids": selection_sample_ids,
        "eval_sample_ids": eval_sample_ids,
        "models": [_MODEL_A, _MODEL_B],
        "best_single_model": _MODEL_A,
        "metadata_router_mapping": {
            "code_generation": _MODEL_A,
            "static_repair": _MODEL_B,
            "execution_repair": _MODEL_B,
        },
        "deployable_router_mapping": {
            "code_generation": _MODEL_A,
            "static_repair": _MODEL_B,
            "execution_repair": _MODEL_B,
        },
        "selection_metrics": {
            "best_single_pass_rate": 0.75,
            "metadata_router_pass_rate": 0.625,
            "deployable_router_pass_rate": 0.625,
            "oracle_router_pass_rate": 0.875,
        },
    }


def _task_type_for(fam_idx: int, s_idx: int) -> str:
    """Cycle through TASK_TYPES so all three appear in selection AND eval subsets."""
    return TASK_TYPES[(fam_idx + s_idx) % 3]


def _make_synthetic_docs(policy: dict) -> dict[str, dict]:
    """Build synthetic eval docs that align with the policy.

    Each doc carries dataset_sha256 matching the policy, and an `outcomes`
    array covering ALL sample IDs (selection + eval). _MODEL_A passes ~75%,
    _MODEL_B passes ~25% — this makes _MODEL_A the legitimate Best Single
    on the selection subset.
    """

    def _det_pass(key: str, threshold: int) -> bool:
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return (h % 100) >= threshold

    all_sample_ids = list(policy["selection_sample_ids"]) + list(
        policy["eval_sample_ids"]
    )
    # Build sample_id -> (family_id, task_type) lookup.
    sid_to_fam: dict[str, str] = {}
    sid_to_tt: dict[str, str] = {}
    for fam_idx, fam_id in enumerate(_ALL_FAMILIES):
        for s_idx in range(2):
            sid = f"{fam_id}_s{s_idx}"
            sid_to_fam[sid] = fam_id
            sid_to_tt[sid] = _task_type_for(fam_idx, s_idx)

    docs: dict[str, dict] = {}
    for model_key in policy["models"]:
        outcomes = []
        for sid in all_sample_ids:
            # _MODEL_A is stronger (~75% pass); _MODEL_B weaker (~25%).
            threshold = 25 if model_key == _MODEL_A else 75
            public_passed = _det_pass(f"{model_key}|{sid}", threshold)
            hidden_passed = _det_pass(f"{model_key}_hidden|{sid}", threshold)
            outcomes.append({
                "sample_id": sid,
                "family_id": sid_to_fam[sid],
                "task_type": sid_to_tt[sid],
                "public_passed": public_passed,
                "hidden_passed": hidden_passed,
            })
        docs[model_key] = {
            "dataset_sha256": _DATASET_SHA,
            "outcomes": outcomes,
        }
    return docs


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_policy() -> dict:
    return _make_synthetic_policy()


@pytest.fixture
def synthetic_docs(synthetic_policy) -> dict[str, dict]:
    return _make_synthetic_docs(synthetic_policy)


# ---------------------------------------------------------------------------
# TestLoadPolicy
# ---------------------------------------------------------------------------

class TestLoadPolicy:
    def test_loads_policy_when_file_exists(self, tmp_path, monkeypatch, synthetic_policy):
        policy_path = tmp_path / "policy.json"
        policy_path.write_text(
            json.dumps(synthetic_policy, indent=2), encoding="utf-8"
        )
        monkeypatch.setattr(
            "scripts.compute_router_analysis._POLICY_PATH", policy_path
        )
        result = _load_policy()
        assert result == synthetic_policy

    def test_raises_system_exit_when_file_missing(self, tmp_path, monkeypatch):
        missing = tmp_path / "does_not_exist.json"
        monkeypatch.setattr(
            "scripts.compute_router_analysis._POLICY_PATH", missing
        )
        with pytest.raises(SystemExit) as exc_info:
            _load_policy()
        assert "router policy not found" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# TestValidatePolicyAlignment
# ---------------------------------------------------------------------------

class TestValidatePolicyAlignment:
    def _common(self, docs) -> list[str]:
        sid_sets = []
        for d in docs.values():
            sid_sets.append({o["sample_id"] for o in d["outcomes"]})
        return sorted(set.intersection(*sid_sets))

    def test_passes_when_everything_aligned(self, synthetic_policy, synthetic_docs):
        common = self._common(synthetic_docs)
        # Should not raise.
        _validate_policy_alignment(synthetic_policy, synthetic_docs, common)

    def test_fails_on_dataset_sha_mismatch(self, synthetic_policy, synthetic_docs):
        first_key = next(iter(synthetic_docs))
        synthetic_docs[first_key]["dataset_sha256"] = "tampered"
        common = self._common(synthetic_docs)
        with pytest.raises(SystemExit) as exc_info:
            _validate_policy_alignment(synthetic_policy, synthetic_docs, common)
        assert "dataset_sha256 mismatch" in str(exc_info.value)

    def test_fails_on_selection_eval_sample_overlap(
        self, synthetic_policy, synthetic_docs
    ):
        synthetic_policy["selection_sample_ids"].append(
            synthetic_policy["eval_sample_ids"][0]
        )
        common = self._common(synthetic_docs)
        with pytest.raises(SystemExit) as exc_info:
            _validate_policy_alignment(synthetic_policy, synthetic_docs, common)
        assert "overlap" in str(exc_info.value).lower()

    def test_fails_on_selection_eval_family_overlap(
        self, synthetic_policy, synthetic_docs
    ):
        synthetic_policy["selection_families"].append(
            synthetic_policy["eval_families"][0]
        )
        common = self._common(synthetic_docs)
        with pytest.raises(SystemExit) as exc_info:
            _validate_policy_alignment(synthetic_policy, synthetic_docs, common)
        assert "overlap" in str(exc_info.value).lower()

    def test_fails_when_eval_sample_missing_from_models(
        self, synthetic_policy, synthetic_docs
    ):
        target_sid = synthetic_policy["eval_sample_ids"][0]
        for d in synthetic_docs.values():
            d["outcomes"] = [
                o for o in d["outcomes"] if o["sample_id"] != target_sid
            ]
        common = self._common(synthetic_docs)
        with pytest.raises(SystemExit) as exc_info:
            _validate_policy_alignment(synthetic_policy, synthetic_docs, common)
        assert "not present in all loaded" in str(exc_info.value)


# ---------------------------------------------------------------------------
# TestFilterToEvalSubset
# ---------------------------------------------------------------------------

class TestFilterToEvalSubset:
    def test_returns_only_eval_samples(self, synthetic_policy):
        all_ids = (
            list(synthetic_policy["selection_sample_ids"])
            + list(synthetic_policy["eval_sample_ids"])
        )
        common = sorted(all_ids)
        result = _filter_to_eval_subset({}, common, synthetic_policy)
        assert result == sorted(synthetic_policy["eval_sample_ids"])

    def test_excludes_selection_samples(self, synthetic_policy):
        all_ids = (
            list(synthetic_policy["selection_sample_ids"])
            + list(synthetic_policy["eval_sample_ids"])
        )
        common = sorted(all_ids)
        result = _filter_to_eval_subset({}, common, synthetic_policy)
        for sid in synthetic_policy["selection_sample_ids"]:
            assert sid not in result

    def test_returns_empty_when_no_overlap(self, synthetic_policy):
        # common = only selection_sample_ids -> no overlap with eval_sample_ids.
        common = sorted(synthetic_policy["selection_sample_ids"])
        result = _filter_to_eval_subset({}, common, synthetic_policy)
        assert result == []


# ---------------------------------------------------------------------------
# TestEndToEndOnEvalSubset
# ---------------------------------------------------------------------------

class TestEndToEndOnEvalSubset:
    """Run main() with synthetic policy + docs patched in."""

    def _setup(
        self, monkeypatch, tmp_path, synthetic_policy, synthetic_docs
    ):
        # Write policy file.
        policy_path = tmp_path / "policy.json"
        policy_path.write_text(
            json.dumps(synthetic_policy, indent=2), encoding="utf-8"
        )
        monkeypatch.setattr(
            "scripts.compute_router_analysis._POLICY_PATH", policy_path
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "scripts.compute_router_analysis._OUT_DIR", out_dir
        )

        # Patch _load to return synthetic docs (None for models not in policy).
        def _fake_load(name: str):
            return synthetic_docs.get(name)

        monkeypatch.setattr(
            "scripts.compute_router_analysis._load", _fake_load
        )
        return out_dir

    def test_best_single_uses_frozen_choice(
        self, monkeypatch, tmp_path, synthetic_policy, synthetic_docs
    ):
        from scripts.compute_router_analysis import main

        out_dir = self._setup(monkeypatch, tmp_path, synthetic_policy, synthetic_docs)
        rc = main()
        assert rc == 0
        result = json.loads(
            (out_dir / "router-analysis.json").read_text(encoding="utf-8")
        )
        assert result["best_single"]["model_key"] == synthetic_policy["best_single_model"]
        assert result["best_single"]["source"] == "frozen_policy_v1"

    def test_router_metrics_use_eval_subset_only(
        self, monkeypatch, tmp_path, synthetic_policy, synthetic_docs
    ):
        from scripts.compute_router_analysis import main

        out_dir = self._setup(monkeypatch, tmp_path, synthetic_policy, synthetic_docs)
        rc = main()
        assert rc == 0
        result = json.loads(
            (out_dir / "router-analysis.json").read_text(encoding="utf-8")
        )
        n_eval = len(synthetic_policy["eval_sample_ids"])
        assert result["common_sample_count"] == n_eval
        assert result["eval_subset_size"] == n_eval
        assert result["deployable_router"]["source"] == "frozen_policy_v1"
        assert result["metadata_router"]["source"] == "frozen_policy_v1"
        for note in result["notes"]:
            assert "train-on-test" not in note.lower()

    def test_oracle_recomputed_and_gate_uses_eval_subset(
        self, monkeypatch, tmp_path, synthetic_policy, synthetic_docs
    ):
        from scripts.compute_router_analysis import main

        out_dir = self._setup(monkeypatch, tmp_path, synthetic_policy, synthetic_docs)
        rc = main()
        assert rc == 0
        result = json.loads(
            (out_dir / "router-analysis.json").read_text(encoding="utf-8")
        )
        n_eval = len(synthetic_policy["eval_sample_ids"])
        # Oracle is recomputed on eval subset (not frozen), so its n_samples
        # matches the eval subset size and it has no "source" field.
        assert result["oracle_router"]["n_samples"] == n_eval
        assert "source" not in result["oracle_router"]
        # Decision gate n_common also matches the eval subset size.
        assert result["decision_gate"]["criteria"]["n_common"] == n_eval
