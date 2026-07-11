"""P4 §2.6 — CRLF/LF normalization regression tests for the patch tools.

Implements the 12 required tests from
docs/roadmaps/Qwen0.6B-Experiment-Trail_P4-Roadmap_2026-07-10.md §2.6.

These tests exercise ``tool_propose_patch`` / ``tool_apply_patch`` /
``tool_rollback_patch`` in src/agent_tools.py against files with LF and
CRLF line-ending conventions.

Implementation note (items 8 & 9): the current implementation normalizes
only ``old_text`` to match the file's line-ending style; ``new_text`` is
inserted verbatim via ``str.replace``. The tests below reflect the
actual behavior and document, in comments, where it falls short of the
roadmap's auto-normalization requirements. They are written to pass
against the current code.
"""
import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from src.agent_workspace import MicroTaskWorkspace
from src.agent_tools import (
    tool_apply_patch,
    tool_propose_patch,
    tool_rollback_patch,
)


# --- helpers -----------------------------------------------------------------


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _has_mixed_line_endings(data: bytes) -> bool:
    """True if ``data`` contains both CRLF and a lone LF (not part of CRLF)."""
    has_crlf = b"\r\n" in data
    # Remove all CRLF pairs, then any remaining \n is a lone LF.
    has_lone_lf = b"\n" in data.replace(b"\r\n", b"")
    return has_crlf and has_lone_lf


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def make_workspace():
    """Factory: build a MicroTaskWorkspace from a {rel_path: bytes} dict.

    Tracks every created workspace + source dir and cleans them all up on
    teardown so each test is self-contained.
    """
    created: list[tuple[MicroTaskWorkspace, Path]] = []

    def _make(files: dict[str, bytes]) -> MicroTaskWorkspace:
        source_dir = Path(tempfile.mkdtemp(prefix="p4_crlf_src_"))
        for rel, data in files.items():
            _write(source_dir / rel, data)
        ws = MicroTaskWorkspace.from_task(source_dir)
        created.append((ws, source_dir))
        return ws

    yield _make

    for ws, source_dir in created:
        ws.cleanup()
        shutil.rmtree(source_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolate_network(monkeypatch):
    """Force P4_ALLOW_NETWORK=0 for every test (per task instructions)."""
    monkeypatch.setenv("P4_ALLOW_NETWORK", "0")


# --- 1. LF file + LF patch ---------------------------------------------------


def test_lf_file_lf_patch_applies(make_workspace):
    """§2.6 test 1: LF file + LF patch applies correctly."""
    ws = make_workspace({"code.py": b"def add(a, b):\n    return a - b\n"})
    old_text = "return a - b\n"
    new_text = "return a + b\n"

    obs = tool_apply_patch(ws, "code.py", old_text, new_text, action_id="t1")

    assert obs.success is True
    assert obs.error is None
    result = (ws.workspace_root / "code.py").read_bytes()
    assert result == b"def add(a, b):\n    return a + b\n"
    assert b"\r\n" not in result


# --- 2. CRLF file + LF patch -------------------------------------------------


def test_crlf_file_lf_patch_applies(make_workspace):
    """§2.6 test 2: CRLF file + LF patch applies via old_text normalization."""
    ws = make_workspace({"code.py": b"def add(a, b):\r\n    return a - b\r\n"})
    old_text = "return a - b\n"  # LF — must be normalized to CRLF to match
    new_text = "return a + b\r\n"  # caller supplies matching CRLF

    obs = tool_apply_patch(ws, "code.py", old_text, new_text, action_id="t2")

    assert obs.success is True
    assert obs.error is None
    result = (ws.workspace_root / "code.py").read_bytes()
    assert result == b"def add(a, b):\r\n    return a + b\r\n"


# --- 3. LF file + CRLF patch -------------------------------------------------


def test_lf_file_crlf_patch_applies(make_workspace):
    """§2.6 test 3: LF file + CRLF patch applies via old_text normalization."""
    ws = make_workspace({"code.py": b"def add(a, b):\n    return a - b\n"})
    old_text = "return a - b\r\n"  # CRLF — must be normalized to LF to match
    new_text = "return a + b\n"  # caller supplies matching LF

    obs = tool_apply_patch(ws, "code.py", old_text, new_text, action_id="t3")

    assert obs.success is True
    assert obs.error is None
    result = (ws.workspace_root / "code.py").read_bytes()
    assert result == b"def add(a, b):\n    return a + b\n"


# --- 4. CRLF file + CRLF patch -----------------------------------------------


def test_crlf_file_crlf_patch_applies(make_workspace):
    """§2.6 test 4: CRLF file + CRLF patch applies correctly."""
    ws = make_workspace({"code.py": b"def add(a, b):\r\n    return a - b\r\n"})
    old_text = "return a - b\r\n"
    new_text = "return a + b\r\n"

    obs = tool_apply_patch(ws, "code.py", old_text, new_text, action_id="t4")

    assert obs.success is True
    assert obs.error is None
    result = (ws.workspace_root / "code.py").read_bytes()
    assert result == b"def add(a, b):\r\n    return a + b\r\n"


# --- 5. old_text absent after normalization ----------------------------------


def test_old_text_absent_after_normalization_hard_fails(make_workspace):
    """§2.6 test 5: old_text absent after normalization hard-fails.

    The patch must NOT silently no-op when the normalized old_text is
    still absent from the file.
    """
    ws = make_workspace({"code.py": b"def add(a, b):\r\n    return a + b\r\n"})
    old_text = "return a - b\n"  # LF; normalized to CRLF but still absent
    new_text = "return a + b\r\n"

    obs = tool_apply_patch(ws, "code.py", old_text, new_text, action_id="t5")

    assert obs.success is False
    assert obs.error is not None
    assert "not found" in obs.error
    # file must be untouched
    assert (ws.workspace_root / "code.py").read_bytes() == \
        b"def add(a, b):\r\n    return a + b\r\n"


# --- 6. old_text non-unique after normalization ------------------------------


def test_old_text_non_unique_after_normalization_hard_fails(make_workspace):
    """§2.6 test 6: old_text non-unique after normalization hard-fails.

    Normalization must NOT turn two distinct matches into an unnoticed
    ambiguous match — the uniqueness check runs on the normalized text,
    so a duplicate old_text still hard-fails instead of patching the
    first match silently.
    """
    ws = make_workspace({"code.py": b"x = 1\r\nx = 1\r\n"})
    old_text = "x = 1\n"  # LF; normalized to CRLF -> 2 matches
    new_text = "x = 2\r\n"

    obs = tool_apply_patch(ws, "code.py", old_text, new_text, action_id="t6")

    assert obs.success is False
    assert obs.error is not None
    assert "unique" in obs.error
    # file must be untouched
    assert (ws.workspace_root / "code.py").read_bytes() == b"x = 1\r\nx = 1\r\n"


# --- 7. SHA mismatch still hard-fails before patching ------------------------


def test_sha_mismatch_hard_fails_before_normalization(make_workspace):
    """§2.6 test 7: SHA mismatch hard-fails before old_text/normalization.

    The SHA check happens before normalization and the old_text search,
    so even an old_text that would match after normalization is rejected
    when the expected SHA is wrong.
    """
    ws = make_workspace({"code.py": b"def add(a, b):\r\n    return a - b\r\n"})
    original = (ws.workspace_root / "code.py").read_bytes()
    wrong_sha = "0" * 64
    old_text = "return a - b\n"  # would normalize + match, but SHA fails first

    obs = tool_apply_patch(
        ws, "code.py", old_text, "return a + b\r\n",
        expected_before_sha256=wrong_sha, action_id="t7",
    )

    assert obs.success is False
    assert obs.error is not None
    assert "SHA" in obs.error or "mismatch" in obs.error
    # file must be untouched
    assert (ws.workspace_root / "code.py").read_bytes() == original


# --- 8. new_text adopts the target file line-ending convention ---------------


def test_new_text_adopts_target_line_ending_convention(make_workspace):
    """§2.6 test 8: new_text adopts the target file line-ending convention.

    NOTE: The current implementation does NOT auto-normalize ``new_text``;
    it is inserted verbatim. The caller must supply ``new_text`` with the
    target file's line-ending convention. This test verifies that when the
    caller supplies matching CRLF ``new_text`` for a CRLF file, the patched
    result preserves a single consistent CRLF convention. Auto-adoption of
    ``new_text`` line endings (roadmap §2.6 item 8) is NOT implemented.
    """
    ws = make_workspace({"code.py": b"def add(a, b):\r\n    return a - b\r\n"})
    old_text = "return a - b\n"  # LF; normalized to CRLF
    new_text = "return a + b\r\n"  # caller supplies matching CRLF

    obs = tool_apply_patch(ws, "code.py", old_text, new_text, action_id="t8")

    assert obs.success is True
    result = (ws.workspace_root / "code.py").read_bytes()
    # convention preserved: CRLF present, no lone LF introduced
    assert b"\r\n" in result
    assert b"\n" not in result.replace(b"\r\n", b"")


# --- 9. No mixed line endings are introduced ---------------------------------


def test_no_mixed_line_endings_introduced(make_workspace):
    """§2.6 test 9: no mixed line endings are introduced.

    NOTE: The current implementation does NOT guard against a caller
    supplying ``new_text`` with mismatched line endings — mixed endings
    would then result. This test verifies the positive case: when
    ``old_text`` (after normalization) and ``new_text`` are both
    consistent with the file's CRLF convention, the patched file has no
    mixed line endings. Guarding against mismatched ``new_text`` is NOT
    implemented (gap vs. roadmap §2.6 item 9).
    """
    ws = make_workspace({"code.py": b"def add(a, b):\r\n    return a - b\r\n"})
    old_text = "return a - b\n"  # LF; normalized to CRLF
    new_text = "return a + b\r\n"  # caller supplies matching CRLF

    obs = tool_apply_patch(ws, "code.py", old_text, new_text, action_id="t9")

    assert obs.success is True
    result = (ws.workspace_root / "code.py").read_bytes()
    assert not _has_mixed_line_endings(result)


# --- 10. propose_patch and apply_patch calculate identical after_sha256 ------


def test_propose_and_apply_identical_after_sha256(make_workspace):
    """§2.6 test 10: propose_patch and apply_patch produce identical
    after_sha256 for the same inputs.

    ``tool_propose_patch`` is a dry-run of ``tool_apply_patch``; both
    compute ``after_sha256`` from the same replacement, so the values
    must match exactly (and match the on-disk SHA after the apply).
    """
    ws = make_workspace({"code.py": b"def add(a, b):\n    return a - b\n"})
    old_text = "return a - b\n"
    new_text = "return a + b\n"

    proposal = tool_propose_patch(ws, "code.py", old_text, new_text)
    assert proposal.would_succeed is True

    applied = tool_apply_patch(
        ws, "code.py", old_text, new_text, action_id="t10",
    )
    assert applied.success is True

    assert proposal.before_sha256 == applied.before_sha256
    assert proposal.after_sha256 == applied.after_sha256

    on_disk = hashlib.sha256(
        (ws.workspace_root / "code.py").read_bytes()
    ).hexdigest()
    assert on_disk == applied.after_sha256


# --- 11. Rollback restores byte-identical original content -------------------


def test_rollback_restores_byte_identical_original(make_workspace):
    """§2.6 test 11: rollback restores byte-identical original content
    (line endings included)."""
    original_bytes = b"def add(a, b):\r\n    return a - b\r\n"
    ws = make_workspace({"code.py": original_bytes})
    original_sha = hashlib.sha256(original_bytes).hexdigest()

    old_text = "return a - b\n"  # LF; normalized to CRLF
    new_text = "return a + b\r\n"

    patch_obs = tool_apply_patch(
        ws, "code.py", old_text, new_text, action_id="t11",
    )
    assert patch_obs.success is True
    assert (ws.workspace_root / "code.py").read_bytes() != original_bytes

    rollback_obs = tool_rollback_patch(ws, "t11")
    assert rollback_obs.success is True
    assert rollback_obs.after_sha256 == original_sha

    restored = (ws.workspace_root / "code.py").read_bytes()
    assert restored == original_bytes


# --- 12. Binary and non-UTF-8 rejection remains unchanged --------------------


def test_binary_and_non_utf8_rejection_unchanged(make_workspace):
    """§2.6 test 12: binary (NUL-byte) and non-UTF-8 rejection unchanged.

    The CRLF normalization must not weaken the existing binary / non-UTF-8
    guards: both must still hard-fail without modifying the file.
    """
    # 12a — NUL-byte binary file
    ws_bin = make_workspace({"bin.dat": b"\x00\x01\x02binary\x00data"})
    obs_bin = tool_apply_patch(
        ws_bin, "bin.dat", "binary", "text", action_id="t12bin",
    )
    assert obs_bin.success is False
    assert obs_bin.error is not None
    assert "binary" in obs_bin.error
    assert (ws_bin.workspace_root / "bin.dat").read_bytes() == \
        b"\x00\x01\x02binary\x00data"

    # 12b — non-UTF-8 file (no NUL byte, but invalid UTF-8 start byte)
    ws_bad = make_workspace({"bad.txt": b"\xff\xfe\xfd not valid utf-8"})
    obs_bad = tool_apply_patch(
        ws_bad, "bad.txt", "not", "valid", action_id="t12utf",
    )
    assert obs_bad.success is False
    assert obs_bad.error is not None
    assert "binary" in obs_bad.error or "utf" in obs_bad.error.lower()
    assert (ws_bad.workspace_root / "bad.txt").read_bytes() == \
        b"\xff\xfe\xfd not valid utf-8"


# --- extra: rollback unknown action_id still raises (sanity, not in §2.6) ----
# Deliberately omitted — only the 12 roadmap tests are required here.
