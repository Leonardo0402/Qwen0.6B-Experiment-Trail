"""Phase A: P3 exit baseline lock tests."""
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_LOCK_PATH = _ROOT / "reports/p4/p3-exit-baseline-lock.json"


def test_lock_file_exists():
    assert _LOCK_PATH.exists(), "Run scripts/lock_p3_exit_baseline.py first"


def test_lock_schema_version():
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1


def test_lock_verdict():
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    assert data["p3_terminal_verdict"] == "MBPP_FAMILY_OR_VARIANT_LIMIT"


def test_lock_pr_15_sha():
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    assert data["pr_15"]["merge_commit_sha"] == "d91586e0d31214f4ed3edbdce524e6b0e8067070"


def test_lock_adapters_match_manifests():
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    for cand in ("balanced_limited", "repair_limited"):
        fs_name = cand.replace("_", "-")
        manifest = json.loads((_ROOT / f"data/p3-limited/{fs_name}/manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((_ROOT / f"adapters/p3/{fs_name}/metrics.json").read_text(encoding="utf-8"))
        assert data["adapters"][cand]["train_hash_in_metrics"] == manifest["train_sha256"]
        assert metrics["train_hash"] == manifest["train_sha256"]


def test_lock_warning_count():
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    assert len(data["warnings"]) >= 3
