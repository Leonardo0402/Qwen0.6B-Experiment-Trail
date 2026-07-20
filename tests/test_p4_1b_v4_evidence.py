# tests/test_p4_1b_v4_evidence.py
"""P4.1b v4 Evidence Closure — CPU-only validation of committed v4 artifacts.

This test file reads ONLY committed JSON / JSONL files under
``reports/p4/protocol-ablation-v4/``. It does NOT touch:
- ``models/Qwen3-0.6B`` (~1.5GB; absent from CI)
- ``adapters/p3/repair-limited`` (absent from CI)
- any GPU / model / adapter loading code

It enforces the v4 evidence contract documented in
``reports/2026-07-20/p4-1b-v4-evidence-closure.md`` so that CPU CI can
independently verify the integrity of the committed v4 artifacts without
requiring the experimental hardware.

Scope:
- Section 3.1 baseline-lock.json basic fields
- Section 3.2 model_fingerprint structure
- Section 3.3 adapter_fingerprint structure
- Section 3.4 pre/post fingerprint equality
- Section 3.5 artifact-manifest.json structure
- Section 3.6 comparison-matrix.json + verdict.json

These tests MUST run in CPU CI (no local_artifacts marker).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_V4_DIR = _ROOT / "reports" / "p4" / "protocol-ablation-v4"

_BASELINE_LOCK_PATH = _V4_DIR / "baseline-lock.json"
_ARTIFACT_MANIFEST_PATH = _V4_DIR / "artifact-manifest.json"
_COMPARISON_MATRIX_PATH = _V4_DIR / "comparison-matrix.json"
_VERDICT_PATH = _V4_DIR / "verdict.json"

_EXPECTED_EXPERIMENT_COMMIT = "a5577ba081c806aaf00627262a36cbb1c38e72df"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def baseline_lock() -> dict:
    return json.loads(_BASELINE_LOCK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact_manifest() -> dict:
    return json.loads(_ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def comparison_matrix() -> list[dict]:
    return json.loads(_COMPARISON_MATRIX_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def verdict() -> dict:
    return json.loads(_VERDICT_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Section 3.1 — baseline-lock.json basic fields
# --------------------------------------------------------------------------- #
class TestBaselineLockBasic:
    """Validate baseline-lock.json top-level fields."""

    def test_experiment_commit_sha_matches_v4_run(self, baseline_lock):
        assert baseline_lock["experiment_commit_sha"] == _EXPECTED_EXPERIMENT_COMMIT

    def test_git_worktree_clean_for_experiment_is_true(self, baseline_lock):
        assert baseline_lock["git_worktree_clean_for_experiment"] is True

    def test_report_dir_name_is_protocol_ablation_v4(self, baseline_lock):
        assert baseline_lock["report_dir_name"] == "protocol-ablation-v4"

    def test_total_tasks_is_40(self, baseline_lock):
        assert baseline_lock["total_tasks"] == 40

    def test_total_combinations_is_6(self, baseline_lock):
        assert baseline_lock["total_combinations"] == 6

    def test_total_runs_is_240(self, baseline_lock):
        assert baseline_lock["total_runs"] == 240

    def test_max_steps_is_12(self, baseline_lock):
        assert baseline_lock["max_steps"] == 12

    def test_protocols_are_json_tag_dsl(self, baseline_lock):
        assert baseline_lock["protocols"] == ["json", "tag", "dsl"]

    def test_configs_are_base_and_repair_lora(self, baseline_lock):
        assert baseline_lock["configs"] == ["base", "repair-lora"]

    def test_task_ids_count_is_40_and_unique(self, baseline_lock):
        task_ids = baseline_lock["task_ids"]
        assert len(task_ids) == 40
        assert len(set(task_ids)) == 40

    def test_task_ids_match_expected_pattern(self, baseline_lock):
        """All task IDs should follow task_NNN pattern."""
        for tid in baseline_lock["task_ids"]:
            assert re.match(r"^task_\d{3}$", tid), f"unexpected task id: {tid}"

    def test_micro_task_manifest_sha_is_64_hex(self, baseline_lock):
        sha = baseline_lock["micro_task_manifest_sha256"]
        assert _HEX64_RE.match(sha), f"not 64-hex: {sha}"


# --------------------------------------------------------------------------- #
# Section 3.2 — model_fingerprint structure
# --------------------------------------------------------------------------- #
class TestModelFingerprint:
    """Validate model_fingerprint structure in baseline-lock.json."""

    def test_exists_is_true(self, baseline_lock):
        assert baseline_lock["model_fingerprint"]["exists"] is True

    def test_file_count_positive(self, baseline_lock):
        assert baseline_lock["model_fingerprint"]["file_count"] > 0

    def test_aggregate_sha_is_64_hex(self, baseline_lock):
        sha = baseline_lock["model_fingerprint"]["aggregate_sha256"]
        assert _HEX64_RE.match(sha), f"not 64-hex: {sha}"

    def test_includes_config_json(self, baseline_lock):
        assert "config.json" in baseline_lock["model_fingerprint"]["files"]

    def test_includes_model_safetensors(self, baseline_lock):
        assert "model.safetensors" in baseline_lock["model_fingerprint"]["files"]

    def test_includes_tokenizer_json(self, baseline_lock):
        assert "tokenizer.json" in baseline_lock["model_fingerprint"]["files"]

    def test_model_safetensors_size_positive(self, baseline_lock):
        files = baseline_lock["model_fingerprint"]["files"]
        assert files["model.safetensors"]["size"] > 0

    def test_every_file_sha_is_64_hex(self, baseline_lock):
        files = baseline_lock["model_fingerprint"]["files"]
        for name, info in files.items():
            assert _HEX64_RE.match(info["sha256"]), \
                f"file {name} sha256 not 64-hex: {info['sha256']}"


# --------------------------------------------------------------------------- #
# Section 3.3 — adapter_fingerprint structure
# --------------------------------------------------------------------------- #
class TestAdapterFingerprint:
    """Validate adapter_fingerprint fields in baseline-lock.json."""

    def test_repair_lora_exists_is_true(self, baseline_lock):
        assert baseline_lock["adapter_fingerprint_repair_lora"]["exists"] is True

    def test_repair_lora_aggregate_sha_is_64_hex(self, baseline_lock):
        sha = baseline_lock["adapter_fingerprint_repair_lora"]["aggregate_sha256"]
        assert _HEX64_RE.match(sha), f"not 64-hex: {sha}"

    def test_repair_lora_includes_adapter_config_json(self, baseline_lock):
        files = baseline_lock["adapter_fingerprint_repair_lora"]["files"]
        assert "adapter_config.json" in files

    def test_repair_lora_includes_adapter_model_safetensors(self, baseline_lock):
        files = baseline_lock["adapter_fingerprint_repair_lora"]["files"]
        assert "adapter_model.safetensors" in files

    def test_repair_lora_adapter_model_size_positive(self, baseline_lock):
        files = baseline_lock["adapter_fingerprint_repair_lora"]["files"]
        assert files["adapter_model.safetensors"]["size"] > 0

    def test_repair_lora_has_no_checkpoint_subdirs(self, baseline_lock):
        files = baseline_lock["adapter_fingerprint_repair_lora"]["files"]
        for name in files:
            assert not name.startswith("checkpoint-"), \
                f"checkpoint subdir leaked into fingerprint: {name}"

    def test_base_adapter_fingerprint_does_not_exist(self, baseline_lock):
        """Base config has no adapter; fingerprint should reflect absence."""
        assert baseline_lock["adapter_fingerprint_base"]["exists"] is False
        assert baseline_lock["adapter_fingerprint_base"]["file_count"] == 0
        assert baseline_lock["adapter_fingerprint_base"]["aggregate_sha256"] == ""


# --------------------------------------------------------------------------- #
# Section 3.4 — pre/post fingerprint equality
# --------------------------------------------------------------------------- #
class TestPrePostFingerprintEquality:
    """Verify model and adapter were unchanged during the v4 experiment."""

    def test_model_fingerprint_pre_post_equal(self, baseline_lock):
        pre = baseline_lock["model_fingerprint"]["aggregate_sha256"]
        post = baseline_lock["model_fingerprint_post"]["aggregate_sha256"]
        assert pre == post, \
            f"model fingerprint changed during experiment: pre={pre} post={post}"

    def test_adapter_fingerprint_pre_post_equal(self, baseline_lock):
        pre = baseline_lock["adapter_fingerprint_repair_lora"]["aggregate_sha256"]
        post = baseline_lock["adapter_fingerprint_repair_lora_post"]["aggregate_sha256"]
        assert pre == post, \
            f"adapter fingerprint changed during experiment: pre={pre} post={post}"

    def test_fingerprint_check_passed_is_true(self, baseline_lock):
        assert baseline_lock["fingerprint_check_passed"] is True

    def test_model_post_files_match_pre_files(self, baseline_lock):
        """Defensive: post fingerprint files dict should equal pre."""
        pre_files = baseline_lock["model_fingerprint"]["files"]
        post_files = baseline_lock["model_fingerprint_post"]["files"]
        assert pre_files == post_files

    def test_adapter_post_files_match_pre_files(self, baseline_lock):
        pre_files = baseline_lock["adapter_fingerprint_repair_lora"]["files"]
        post_files = baseline_lock["adapter_fingerprint_repair_lora_post"]["files"]
        assert pre_files == post_files


# --------------------------------------------------------------------------- #
# Section 3.5 — artifact-manifest.json structure
# --------------------------------------------------------------------------- #
class TestArtifactManifest:
    """Validate the v4 artifact manifest."""

    def test_report_dir_is_protocol_ablation_v4(self, artifact_manifest):
        assert artifact_manifest["report_dir"] == "protocol-ablation-v4"

    def test_artifact_count_is_11(self, artifact_manifest):
        assert artifact_manifest["artifact_count"] == 11
        assert len(artifact_manifest["artifacts"]) == 11

    def test_manifest_does_not_reference_itself(self, artifact_manifest):
        """artifact-manifest.json must not list itself as an artifact."""
        for art in artifact_manifest["artifacts"]:
            normalized = art["relative_path"].replace("\\", "/")
            assert normalized != "artifact-manifest.json", \
                "manifest references itself"
            assert normalized != "v4-vs-v3-analysis.md", \
                "v4-vs-v3-analysis.md is not a core artifact (it is a " \
                "post-hoc analysis document and must not be in the manifest)"

    def test_contains_six_trajectory_jsonl_files(self, artifact_manifest):
        traj_paths = [
            art["relative_path"].replace("\\", "/")
            for art in artifact_manifest["artifacts"]
            if "trajectories/" in art["relative_path"].replace("\\", "/")
        ]
        assert len(traj_paths) == 6, f"expected 6 trajectories, got {traj_paths}"

        expected = {
            "trajectories/dsl-base.jsonl",
            "trajectories/dsl-repair-lora.jsonl",
            "trajectories/json-base.jsonl",
            "trajectories/json-repair-lora.jsonl",
            "trajectories/tag-base.jsonl",
            "trajectories/tag-repair-lora.jsonl",
        }
        assert set(traj_paths) == expected

    def test_each_trajectory_has_40_rows(self, artifact_manifest):
        for art in artifact_manifest["artifacts"]:
            normalized = art["relative_path"].replace("\\", "/")
            if "trajectories/" in normalized:
                assert art["row_count"] == 40, \
                    f"{normalized} row_count={art.get('row_count')}"

    def test_total_trajectory_rows_is_240(self, artifact_manifest):
        total = sum(
            art.get("row_count", 0)
            for art in artifact_manifest["artifacts"]
            if "trajectories/" in art["relative_path"].replace("\\", "/")
        )
        assert total == 240

    def test_all_shas_are_64_hex(self, artifact_manifest):
        for art in artifact_manifest["artifacts"]:
            assert _HEX64_RE.match(art["sha256"]), \
                f"{art['relative_path']} sha256 not 64-hex: {art['sha256']}"

    def test_all_artifacts_share_v4_experiment_commit(self, artifact_manifest):
        for art in artifact_manifest["artifacts"]:
            assert art["experiment_commit"] == _EXPECTED_EXPERIMENT_COMMIT, \
                f"{art['relative_path']} experiment_commit={art['experiment_commit']}"

    def test_manifest_includes_required_core_files(self, artifact_manifest):
        """11 core artifacts must include the expected set."""
        paths = {
            art["relative_path"].replace("\\", "/")
            for art in artifact_manifest["artifacts"]
        }
        required = {
            "baseline-lock.json",
            "comparison-matrix.json",
            "comparison-report.md",
            "failure-taxonomy.json",
            "verdict.json",
            "trajectories/dsl-base.jsonl",
            "trajectories/dsl-repair-lora.jsonl",
            "trajectories/json-base.jsonl",
            "trajectories/json-repair-lora.jsonl",
            "trajectories/tag-base.jsonl",
            "trajectories/tag-repair-lora.jsonl",
        }
        assert required.issubset(paths), \
            f"missing required artifacts: {required - paths}"


# --------------------------------------------------------------------------- #
# Section 3.6 — comparison-matrix.json + verdict.json
# --------------------------------------------------------------------------- #
class TestComparisonMatrixAndVerdict:
    """Validate the comparison matrix and verdict."""

    def test_matrix_has_six_combinations(self, comparison_matrix):
        assert len(comparison_matrix) == 6

    def test_matrix_combinations_cover_all_protocols_and_configs(
        self, comparison_matrix
    ):
        combos = {(c["protocol"], c["config"]) for c in comparison_matrix}
        expected = {
            ("json", "base"), ("json", "repair-lora"),
            ("tag", "base"), ("tag", "repair-lora"),
            ("dsl", "base"), ("dsl", "repair-lora"),
        }
        assert combos == expected

    def test_each_combination_has_40_trajectories(self, comparison_matrix):
        for c in comparison_matrix:
            assert c["trajectories_written"] == 40, \
                f"{c['protocol']}-{c['config']} trajectories={c['trajectories_written']}"

    def test_each_combination_has_480_steps(self, comparison_matrix):
        for c in comparison_matrix:
            assert c["metrics"]["total_steps"] == 480, \
                f"{c['protocol']}-{c['config']} total_steps={c['metrics']['total_steps']}"

    def test_each_combination_runtime_crash_zero(self, comparison_matrix):
        for c in comparison_matrix:
            assert c["metrics"]["runtime_crash_count"] == 0, \
                f"{c['protocol']}-{c['config']} runtime_crash_count={c['metrics']['runtime_crash_count']}"

    def test_each_combination_task_success_zero(self, comparison_matrix):
        for c in comparison_matrix:
            assert c["metrics"]["successful_trajectories"] == 0, \
                f"{c['protocol']}-{c['config']} successful_trajectories={c['metrics']['successful_trajectories']}"

    def test_each_combination_max_steps_hit_40_of_40(self, comparison_matrix):
        for c in comparison_matrix:
            assert c["metrics"]["max_steps_hit_count"] == 40, \
                f"{c['protocol']}-{c['config']} max_steps_hit_count={c['metrics']['max_steps_hit_count']}"

    def test_each_combination_model_load_ok(self, comparison_matrix):
        """All six combinations must have loaded the model successfully."""
        for c in comparison_matrix:
            assert c["model_load_ok"] is True, \
                f"{c['protocol']}-{c['config']} model_load_ok={c['model_load_ok']}"
            assert c["metrics"]["model_load_ok"] is True

    def test_verdict_is_fix_prompt_first(self, verdict):
        assert verdict["verdict"] == "FIX_PROMPT_FIRST"


# --------------------------------------------------------------------------- #
# Section 3.7 — v4-vs-v3-analysis.md presence (post-hoc analysis document)
# --------------------------------------------------------------------------- #
class TestV4VsV3AnalysisDocument:
    """Verify the post-hoc analysis document exists alongside the v4 artifacts.

    This file is NOT a core experiment artifact and is not listed in
    artifact-manifest.json. It is a separate analysis document.
    """

    def test_v4_vs_v3_analysis_md_exists(self):
        path = _V4_DIR / "v4-vs-v3-analysis.md"
        assert path.exists(), f"missing {path}"
        assert path.stat().st_size > 0

    def test_v4_directory_total_file_count_is_13(self):
        """11 core artifacts + artifact-manifest.json + v4-vs-v3-analysis.md = 13."""
        all_files = sorted(p.name for p in _V4_DIR.iterdir() if p.is_file())
        traj_files = sorted(
            p.name for p in (_V4_DIR / "trajectories").iterdir() if p.is_file()
        )
        # 7 files in v4 root: artifact-manifest.json, baseline-lock.json,
        # comparison-matrix.json, comparison-report.md, failure-taxonomy.json,
        # v4-vs-v3-analysis.md, verdict.json
        assert len(all_files) == 7, \
            f"expected 7 files in v4 root, got {len(all_files)}: {all_files}"
        # 6 trajectory files
        assert len(traj_files) == 6, \
            f"expected 6 trajectory files, got {len(traj_files)}: {traj_files}"
        # Total = 13
        assert len(all_files) + len(traj_files) == 13
