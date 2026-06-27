"""Tests for scripts/download_model.py

Covers:
- verify_artifacts: PRESENT/MISSING status for each required artifact
- build_manifest: required keys, ISO-parseable timestamp, null-safe commit hash
- write_manifest: file creation and JSON round-trip
- should_skip_download: skip/no-skip decision without performing any download

No network calls, no model files required.  All tests use tmp_path fixtures.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.download_model import (
    build_manifest,
    should_skip_download,
    verify_artifacts,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_model_dir(
    base: Path,
    *,
    config: bool = True,
    tokenizer: bool = True,
    safetensors: bool = True,
) -> Path:
    """Populate *base* to look like a partial or complete model directory."""
    if config:
        (base / "config.json").write_text("{}", encoding="utf-8")
    if tokenizer:
        (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    if safetensors:
        (base / "model.safetensors").write_bytes(b"\x00" * 16)
    return base


# ---------------------------------------------------------------------------
# verify_artifacts
# ---------------------------------------------------------------------------

class TestVerifyArtifacts:
    def test_all_present(self, tmp_path: Path) -> None:
        _make_model_dir(tmp_path)
        statuses = verify_artifacts(tmp_path)
        assert statuses["config.json"] == "PRESENT"
        assert statuses["tokenizer"] == "PRESENT"
        assert statuses["safetensors"] == "PRESENT"

    def test_missing_config(self, tmp_path: Path) -> None:
        _make_model_dir(tmp_path, config=False)
        statuses = verify_artifacts(tmp_path)
        assert statuses["config.json"] == "MISSING"
        assert statuses["tokenizer"] == "PRESENT"
        assert statuses["safetensors"] == "PRESENT"

    def test_missing_tokenizer(self, tmp_path: Path) -> None:
        _make_model_dir(tmp_path, tokenizer=False)
        statuses = verify_artifacts(tmp_path)
        assert statuses["config.json"] == "PRESENT"
        assert statuses["tokenizer"] == "MISSING"
        assert statuses["safetensors"] == "PRESENT"

    def test_missing_safetensors(self, tmp_path: Path) -> None:
        _make_model_dir(tmp_path, safetensors=False)
        statuses = verify_artifacts(tmp_path)
        assert statuses["config.json"] == "PRESENT"
        assert statuses["tokenizer"] == "PRESENT"
        assert statuses["safetensors"] == "MISSING"

    def test_tokenizer_config_json_satisfies_tokenizer_check(self, tmp_path: Path) -> None:
        """tokenizer_config.json (without tokenizer.json) should count as PRESENT."""
        _make_model_dir(tmp_path, tokenizer=False)
        (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        statuses = verify_artifacts(tmp_path)
        assert statuses["tokenizer"] == "PRESENT"

    def test_both_tokenizer_files_ok(self, tmp_path: Path) -> None:
        """Having both tokenizer.json and tokenizer_config.json should still be PRESENT."""
        _make_model_dir(tmp_path)
        (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        statuses = verify_artifacts(tmp_path)
        assert statuses["tokenizer"] == "PRESENT"

    def test_all_missing_empty_dir(self, tmp_path: Path) -> None:
        statuses = verify_artifacts(tmp_path)
        assert all(s == "MISSING" for s in statuses.values())

    def test_returns_dict_with_all_keys(self, tmp_path: Path) -> None:
        statuses = verify_artifacts(tmp_path)
        for key in ("config.json", "tokenizer", "safetensors"):
            assert key in statuses

    def test_values_are_present_or_missing(self, tmp_path: Path) -> None:
        _make_model_dir(tmp_path)
        statuses = verify_artifacts(tmp_path)
        for val in statuses.values():
            assert val in ("PRESENT", "MISSING")


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def _local_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "model"
        d.mkdir()
        return d

    def test_required_keys_present(self, tmp_path: Path) -> None:
        m = build_manifest("Qwen/Qwen3-0.6B", self._local_dir(tmp_path), "abc123", _now_iso(), [])
        for key in ("repo_id", "local_dir", "commit_hash", "download_date", "files"):
            assert key in m, f"Missing key: {key}"

    def test_repo_id_preserved(self, tmp_path: Path) -> None:
        m = build_manifest("Qwen/Qwen3-0.6B", self._local_dir(tmp_path), None, _now_iso(), [])
        assert m["repo_id"] == "Qwen/Qwen3-0.6B"

    def test_local_dir_is_absolute_string(self, tmp_path: Path) -> None:
        local_dir = self._local_dir(tmp_path)
        m = build_manifest("Qwen/Qwen3-0.6B", local_dir, None, _now_iso(), [])
        assert isinstance(m["local_dir"], str)
        # Must be an absolute path string
        assert Path(m["local_dir"]).is_absolute()

    def test_commit_hash_null(self, tmp_path: Path) -> None:
        m = build_manifest("Qwen/Qwen3-0.6B", self._local_dir(tmp_path), None, _now_iso(), [])
        assert m["commit_hash"] is None

    def test_commit_hash_preserved(self, tmp_path: Path) -> None:
        m = build_manifest("Qwen/Qwen3-0.6B", self._local_dir(tmp_path), "deadbeef", _now_iso(), [])
        assert m["commit_hash"] == "deadbeef"

    def test_download_date_is_iso_parseable(self, tmp_path: Path) -> None:
        ts = _now_iso()
        m = build_manifest("Qwen/Qwen3-0.6B", self._local_dir(tmp_path), None, ts, [])
        parsed = datetime.fromisoformat(m["download_date"])  # must not raise
        assert parsed is not None

    def test_files_list_preserved(self, tmp_path: Path) -> None:
        files = ["config.json", "model.safetensors", "tokenizer.json"]
        m = build_manifest("Qwen/Qwen3-0.6B", self._local_dir(tmp_path), None, _now_iso(), files)
        assert m["files"] == files

    def test_empty_files_list(self, tmp_path: Path) -> None:
        m = build_manifest("Qwen/Qwen3-0.6B", self._local_dir(tmp_path), None, _now_iso(), [])
        assert m["files"] == []


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_creates_file_with_parents(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "manifest.json"
        write_manifest({"key": "value"}, path)
        assert path.exists()

    def test_json_is_parseable(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        data = {"repo_id": "Qwen/Qwen3-0.6B", "files": ["a.txt", "b.txt"]}
        write_manifest(data, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["repo_id"] == "Qwen/Qwen3-0.6B"
        assert loaded["files"] == ["a.txt", "b.txt"]

    def test_null_commit_serialised_as_null(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        data = {"commit_hash": None}
        write_manifest(data, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["commit_hash"] is None


# ---------------------------------------------------------------------------
# should_skip_download
# ---------------------------------------------------------------------------

class TestShouldSkipDownload:
    def test_skip_when_present_and_no_force(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert should_skip_download(tmp_path, force=False) is True

    def test_no_skip_when_force_even_if_present(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert should_skip_download(tmp_path, force=True) is False

    def test_no_skip_when_dir_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent_model"
        assert should_skip_download(missing, force=False) is False

    def test_no_skip_when_dir_exists_but_no_config(self, tmp_path: Path) -> None:
        # tmp_path itself exists, but has no config.json
        assert should_skip_download(tmp_path, force=False) is False

    def test_force_on_missing_dir_still_no_skip(self, tmp_path: Path) -> None:
        missing = tmp_path / "also_missing"
        assert should_skip_download(missing, force=True) is False
