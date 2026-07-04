"""Tests for scripts/split_router_selection.py — holdout router policy generation.

Verifies that router policy (Best Single, Metadata Router, Deployable Router) is
determined on a held-out selection subset of families, disjoint from the eval
subset used to apply the frozen rules. This eliminates the train-on-test bias
identified in PR #7 post-merge review.
"""
from __future__ import annotations

import hashlib

import pytest

from scripts.compute_router_analysis import MODELS, TASK_TYPES
from scripts.split_router_selection import compute_policy, compute_split


# ---------------------------------------------------------------------------
# Synthetic fixture: 75 families, 8 samples each, all 5 models.
# Pass/fail outcomes are deterministic (md5-based) so router mappings are stable.
# ---------------------------------------------------------------------------

def _det_hash(s: str) -> int:
    """Deterministic hash (stable across runs, unlike built-in hash())."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _make_outcomes(samples, model_key):
    """Generate deterministic outcomes for a list of (sample_id, family_id, task_type)."""
    model_keys = [k for k, _ in MODELS]
    model_idx = model_keys.index(model_key)
    outcomes = []
    for sample_id, family_id, task_type in samples:
        h = _det_hash(f"{model_key}|{sample_id}")
        tt_idx = TASK_TYPES.index(task_type)
        # Each model is a "specialist" for the task_type whose index matches its own.
        # This forces non-trivial router mappings (different model per category).
        if model_idx == tt_idx:
            pass_threshold = 25  # ~75% pass
        else:
            pass_threshold = 55  # ~45% pass
        public_passed = (h % 100) >= pass_threshold
        hidden_passed = (h % 100) >= (pass_threshold + 5)
        outcomes.append({
            "sample_id": sample_id,
            "family_id": family_id,
            "task_type": task_type,
            "public_passed": public_passed,
            "hidden_passed": hidden_passed,
        })
    return outcomes


def _make_synthetic_docs():
    """Create synthetic eval docs for 5 models, 75 families, 8 samples each."""
    families = [f"mbpp_fam_{i:03d}" for i in range(75)]
    samples = []  # (sample_id, family_id, task_type)
    for fam_idx, fam_id in enumerate(families):
        for s_idx in range(8):
            sample_id = f"mbpp_{fam_idx:03d}_{s_idx}"
            task_type = TASK_TYPES[s_idx % 3]
            samples.append((sample_id, fam_id, task_type))
    docs = {}
    for model_key, _label in MODELS:
        docs[model_key] = {"outcomes": _make_outcomes(samples, model_key)}
    return docs


@pytest.fixture(scope="module")
def policy():
    """Generate a synthetic policy artifact for testing."""
    docs = _make_synthetic_docs()
    return compute_policy(docs, dataset_sha256="0" * 64)


# ---------------------------------------------------------------------------
# Required top-level fields in the policy artifact (schema from task brief).
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "policy_version",
    "created_at",
    "selection_dataset_sha256",
    "selection_family_count",
    "eval_family_count",
    "selection_families",
    "eval_families",
    "selection_sample_ids",
    "eval_sample_ids",
    "models",
    "best_single_model",
    "metadata_router_mapping",
    "deployable_router_mapping",
    "selection_metrics",
}


class TestFamilyPartition:
    """Tests 1-4: deterministic family split with isolation."""

    def test_family_partition_disjoint(self, policy):
        # Test 1: selection_families ∩ eval_families == empty set
        sel = set(policy["selection_families"])
        evl = set(policy["eval_families"])
        assert sel.isdisjoint(evl), "selection and eval families must not overlap"

    def test_family_partition_covers_all(self, policy):
        # Test 2: selection_families ∪ eval_families == all 75 families
        sel = set(policy["selection_families"])
        evl = set(policy["eval_families"])
        union = sel | evl
        assert len(union) == 75, f"expected 75 families, got {len(union)}"
        assert len(sel) + len(evl) == 75, "partition must cover all families exactly once"

    def test_deterministic_seed(self, policy):
        # Test 3: running split twice produces identical selection/eval family lists
        families = [f"mbpp_fam_{i:03d}" for i in range(75)]
        sel1, evl1 = compute_split(families, seed=42)
        sel2, evl2 = compute_split(families, seed=42)
        assert sel1 == sel2, "split must be deterministic given seed=42"
        assert evl1 == evl2, "split must be deterministic given seed=42"
        # And the policy's families must match a fresh split
        assert policy["selection_families"] == sel1
        assert policy["eval_families"] == evl1

    def test_split_ratio_60_40(self, policy):
        # Test 4: len(selection_families) == 45, len(eval_families) == 30
        assert policy["selection_family_count"] == 45
        assert policy["eval_family_count"] == 30
        assert len(policy["selection_families"]) == 45
        assert len(policy["eval_families"]) == 30


class TestPolicyArtifact:
    """Tests 5-8: policy artifact schema and field validity."""

    def test_policy_artifact_contains_required_fields(self, policy):
        # Test 5: JSON has all required top-level keys from the schema
        missing = REQUIRED_FIELDS - set(policy.keys())
        assert not missing, f"policy missing required fields: {missing}"

    def test_best_single_is_one_of_models(self, policy):
        # Test 6: best_single_model in MODELS list
        assert policy["best_single_model"] in policy["models"], (
            f"best_single_model {policy['best_single_model']!r} not in models list"
        )

    def test_router_mappings_cover_all_task_types(self, policy):
        # Test 7: both router mappings have exactly 3 keys: code_generation, static_repair, execution_repair
        expected_keys = {"code_generation", "static_repair", "execution_repair"}
        for field in ("metadata_router_mapping", "deployable_router_mapping"):
            mapping = policy[field]
            assert set(mapping.keys()) == expected_keys, (
                f"{field} has keys {set(mapping.keys())}, expected {expected_keys}"
            )

    def test_router_mapping_values_are_valid_models(self, policy):
        # Test 8: every value in *_router_mapping is in MODELS list
        valid_models = set(policy["models"])
        for field in ("metadata_router_mapping", "deployable_router_mapping"):
            mapping = policy[field]
            for category, model_key in mapping.items():
                assert model_key in valid_models, (
                    f"{field}[{category!r}] = {model_key!r} not in models list"
                )


class TestSampleIsolation:
    """Tests 9-10: sample_ids belong to the correct family subset."""

    def test_selection_sample_ids_belong_to_selection_families(self, policy):
        # Test 9: every sample_id in selection_sample_ids has family_id in selection_families
        sel_families = set(policy["selection_families"])
        # Build sample_id -> family_id map from any model doc in the synthetic docs
        docs = _make_synthetic_docs()
        sid_to_family = {}
        for o in docs[policy["models"][0]]["outcomes"]:
            sid_to_family[o["sample_id"]] = o["family_id"]
        for sid in policy["selection_sample_ids"]:
            fam = sid_to_family.get(sid)
            assert fam is not None, f"sample {sid!r} not found in any model doc"
            assert fam in sel_families, (
                f"sample {sid!r} has family {fam!r} not in selection_families"
            )

    def test_eval_sample_ids_belong_to_eval_families(self, policy):
        # Test 10: every sample_id in eval_sample_ids has family_id in eval_families
        eval_families = set(policy["eval_families"])
        docs = _make_synthetic_docs()
        sid_to_family = {}
        for o in docs[policy["models"][0]]["outcomes"]:
            sid_to_family[o["sample_id"]] = o["family_id"]
        for sid in policy["eval_sample_ids"]:
            fam = sid_to_family.get(sid)
            assert fam is not None, f"sample {sid!r} not found in any model doc"
            assert fam in eval_families, (
                f"sample {sid!r} has family {fam!r} not in eval_families"
            )
