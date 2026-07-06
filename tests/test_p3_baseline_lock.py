"""tests/test_p3_baseline_lock.py -- P3 Task 1 baseline lock guards.

These tests enforce the contract of ``reports/p3/p3-baseline-lock.json``:

1. The lock file exists and is valid JSON.
2. Top-level fields (issue, branch, purpose, manifest path, sha, created_at)
   are present and well-formed.
3. Exactly 3 historical baseline models are recorded with the expected names.
4. Each model record has all required fields, non-empty.
5. weight_sha256 is a 64-char hex string OR the ``BASE_MODEL_NO_ADAPTER``
   sentinel (only allowed for the Base model).
6. config_sha256 is a 64-char hex string.
7. training_config_sha256 is a 64-char hex string OR the
   ``BASE_MODEL_NO_TRAINING_CONFIG`` sentinel (only allowed for the Base
   model).
8. historical_eval_set_sha256 matches the value recorded in
   ``data/p2-curriculum/frozen-eval-v2/manifest.json::test_sha256``.
9. created_at fields are valid ISO 8601 timestamps.

This is a read-only lock: tests do NOT compute SHAs (that is the job of the
script that produces the lock). They only verify the recorded values.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


LOCK_FILE = _ROOT / "reports" / "p3" / "p3-baseline-lock.json"
FROZEN_EVAL_V2_MANIFEST = _ROOT / "data" / "p2-curriculum" / "frozen-eval-v2" / "manifest.json"

EXPECTED_MODEL_NAMES = (
    "Base Qwen3-0.6B",
    "Stage3-Independent",
    "Stage3-v3-Antiforget",
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
BASE_WEIGHT_SENTINEL = "BASE_MODEL_NO_ADAPTER"
BASE_TRAIN_SENTINEL = "BASE_MODEL_NO_TRAINING_CONFIG"


@pytest.fixture(scope="module")
def lock_data():
    assert LOCK_FILE.exists(), (
        f"Baseline lock file not found at {LOCK_FILE}. "
        f"Run the P3 baseline lock task first."
    )
    return json.load(LOCK_FILE.open(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Top-level fields
# ---------------------------------------------------------------------------

class TestTopLevelFields:
    def test_issue_is_9(self, lock_data):
        assert lock_data["issue"] == 9

    def test_branch(self, lock_data):
        assert lock_data["branch"] == "feat/p3-capability-expansion-v2"

    def test_purpose_non_empty(self, lock_data):
        assert isinstance(lock_data["purpose"], str)
        assert lock_data["purpose"].strip()

    def test_frozen_eval_v2_manifest_path(self, lock_data):
        p = lock_data["frozen_eval_v2_manifest"]
        assert isinstance(p, str) and p.endswith("manifest.json")
        # Path is relative to repo root and must resolve.
        assert (_ROOT / p).exists(), f"manifest not found at {p}"

    def test_frozen_eval_v2_sha256_format(self, lock_data):
        sha = lock_data["frozen_eval_v2_sha256"]
        assert HEX64.match(sha), f"not a 64-char hex SHA: {sha}"

    def test_created_at_iso8601(self, lock_data):
        ts = lock_data["created_at"]
        datetime.fromisoformat(ts)  # raises ValueError if invalid


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------

class TestModelsList:
    def test_models_key_present(self, lock_data):
        assert "models" in lock_data

    def test_exactly_three_models(self, lock_data):
        assert len(lock_data["models"]) == 3

    def test_expected_model_names(self, lock_data):
        names = [m["model_name"] for m in lock_data["models"]]
        assert sorted(names) == sorted(EXPECTED_MODEL_NAMES)

    def test_no_duplicate_names(self, lock_data):
        names = [m["model_name"] for m in lock_data["models"]]
        assert len(set(names)) == 3


# ---------------------------------------------------------------------------
# Per-model required fields
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = (
    "model_name",
    "adapter_path",
    "weight_sha256",
    "config_sha256",
    "training_config_sha256",
    "historical_eval_set_sha256",
    "historical_held_out_metrics",
    "created_at",
)


class TestModelFields:
    def test_each_model_has_all_required_fields(self, lock_data):
        for m in lock_data["models"]:
            missing = [f for f in REQUIRED_FIELDS if f not in m]
            assert not missing, (
                f"{m.get('model_name', '?')} missing fields: {missing}"
            )

    def test_required_fields_non_empty(self, lock_data):
        for m in lock_data["models"]:
            for f in REQUIRED_FIELDS:
                v = m[f]
                if isinstance(v, str):
                    assert v.strip(), (
                        f"{m['model_name']}.{f} is empty"
                    )
                else:
                    assert v, f"{m['model_name']}.{f} is empty"

    def test_adapter_path_is_relative(self, lock_data):
        for m in lock_data["models"]:
            p = m["adapter_path"]
            # Use forward-slash form for portability; reject absolute paths.
            assert not Path(p).is_absolute(), (
                f"{m['model_name']}.adapter_path must be relative, got {p}"
            )
            # Skip existence check in CI environments without model files
            if not (_ROOT / p).exists():
                pytest.skip(
                    f"{m['model_name']}.adapter_path not available in CI: {p}"
                )


# ---------------------------------------------------------------------------
# SHA format and sentinel rules
# ---------------------------------------------------------------------------

class TestShaFormat:
    def test_weight_sha256_format_or_sentinel(self, lock_data):
        for m in lock_data["models"]:
            sha = m["weight_sha256"]
            is_hex = bool(HEX64.match(sha))
            is_sentinel = sha == BASE_WEIGHT_SENTINEL
            assert is_hex or is_sentinel, (
                f"{m['model_name']}.weight_sha256 must be 64-char hex or "
                f"'{BASE_WEIGHT_SENTINEL}', got {sha}"
            )

    def test_weight_sentinel_only_for_base(self, lock_data):
        for m in lock_data["models"]:
            if m["weight_sha256"] == BASE_WEIGHT_SENTINEL:
                assert m["model_name"] == "Base Qwen3-0.6B", (
                    f"{m['model_name']} uses weight sentinel but is not Base"
                )
            else:
                assert m["model_name"] != "Base Qwen3-0.6B", (
                    f"Base model must use weight sentinel, got {m['weight_sha256']}"
                )

    def test_config_sha256_is_hex64(self, lock_data):
        for m in lock_data["models"]:
            sha = m["config_sha256"]
            assert HEX64.match(sha), (
                f"{m['model_name']}.config_sha256 must be 64-char hex, got {sha}"
            )

    def test_training_config_sha256_format_or_sentinel(self, lock_data):
        for m in lock_data["models"]:
            sha = m["training_config_sha256"]
            is_hex = bool(HEX64.match(sha))
            is_sentinel = sha == BASE_TRAIN_SENTINEL
            assert is_hex or is_sentinel, (
                f"{m['model_name']}.training_config_sha256 must be 64-char "
                f"hex or '{BASE_TRAIN_SENTINEL}', got {sha}"
            )

    def test_training_sentinel_only_for_base(self, lock_data):
        for m in lock_data["models"]:
            if m["training_config_sha256"] == BASE_TRAIN_SENTINEL:
                assert m["model_name"] == "Base Qwen3-0.6B", (
                    f"{m['model_name']} uses training sentinel but is not Base"
                )
            else:
                assert m["model_name"] != "Base Qwen3-0.6B", (
                    f"Base model must use training sentinel"
                )

    def test_historical_eval_set_sha256_is_hex64(self, lock_data):
        for m in lock_data["models"]:
            sha = m["historical_eval_set_sha256"]
            assert HEX64.match(sha), (
                f"{m['model_name']}.historical_eval_set_sha256 must be "
                f"64-char hex, got {sha}"
            )


# ---------------------------------------------------------------------------
# Cross-check with frozen-eval-v2 manifest
# ---------------------------------------------------------------------------

class TestFrozenEvalV2Consistency:
    def test_top_level_sha_matches_manifest(self, lock_data):
        manifest = json.load(FROZEN_EVAL_V2_MANIFEST.open(encoding="utf-8"))
        expected = manifest["test_sha256"]
        assert lock_data["frozen_eval_v2_sha256"] == expected, (
            f"top-level frozen_eval_v2_sha256={lock_data['frozen_eval_v2_sha256']} "
            f"!= manifest test_sha256={expected}"
        )

    def test_each_model_eval_set_sha_matches_manifest(self, lock_data):
        manifest = json.load(FROZEN_EVAL_V2_MANIFEST.open(encoding="utf-8"))
        expected = manifest["test_sha256"]
        for m in lock_data["models"]:
            assert m["historical_eval_set_sha256"] == expected, (
                f"{m['model_name']}.historical_eval_set_sha256 "
                f"!= manifest test_sha256"
            )


# ---------------------------------------------------------------------------
# ISO 8601 created_at
# ---------------------------------------------------------------------------

class TestCreatedAt:
    def test_top_level_created_at(self, lock_data):
        datetime.fromisoformat(lock_data["created_at"])

    def test_each_model_created_at(self, lock_data):
        for m in lock_data["models"]:
            datetime.fromisoformat(m["created_at"])


# ---------------------------------------------------------------------------
# historical_held_out_metrics shape
# ---------------------------------------------------------------------------

class TestHistoricalMetrics:
    def test_metrics_dict_present(self, lock_data):
        for m in lock_data["models"]:
            d = m["historical_held_out_metrics"]
            assert isinstance(d, dict), (
                f"{m['model_name']}.historical_held_out_metrics must be a dict"
            )
            assert d, (
                f"{m['model_name']}.historical_held_out_metrics is empty"
            )

    def test_metrics_have_source_file(self, lock_data):
        for m in lock_data["models"]:
            src = m["historical_held_out_metrics"].get("source_file")
            assert src, f"{m['model_name']} missing source_file in metrics"
            assert (_ROOT / src).exists(), (
                f"{m['model_name']}.source_file does not exist: {src}"
            )

    def test_overall_pass_present(self, lock_data):
        for m in lock_data["models"]:
            v = m["historical_held_out_metrics"].get("overall_pass")
            assert isinstance(v, (int, float)), (
                f"{m['model_name']}.overall_pass missing or not numeric"
            )
            assert 0.0 <= v <= 1.0, (
                f"{m['model_name']}.overall_pass out of [0,1]: {v}"
            )
