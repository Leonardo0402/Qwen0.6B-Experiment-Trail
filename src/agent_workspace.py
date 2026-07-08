"""P4.0 Phase C — MicroTaskWorkspace.

Manages the persistent task directory (read-only source) and a runtime
temp copy where agent tools operate. Path validation is delegated to
``src.agent_actions.validate_path`` so the same safety rules apply
across the workspace and the action schema.

See docs/superpowers/specs/2026-07-08-p4-agent-foundation-design.md §5.
"""
from __future__ import annotations

import fnmatch
import hashlib
import shutil
import tempfile
from pathlib import Path

from src.agent_actions import PathValidationError, validate_path


_IGNORE_DIRS: set[str] = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}

_IGNORE_PATTERNS: list[str] = [
    "*.pyc",
    "*.pyo",
    ".env",
    ".env.local",
    "credentials.json",
    "*.key",
    "*.pem",
    "adapter_model.safetensors",
    "*.bin",
    "*.pt",
    "*.ckpt",
]

_SHA_CHUNK = 64 * 1024


class MicroTaskWorkspace:
    """Persistent task directory + runtime temp copy.

    ``persistent_dir`` is the original task directory (read-only
    reference). ``workspace_root`` is a mutable temp copy where agent
    tools read files, apply patches, and run tests.
    """

    def __init__(self, persistent_dir: Path, workspace_root: Path):
        self.persistent_dir = persistent_dir
        self.workspace_root = workspace_root

    @classmethod
    def from_task(cls, task_dir: Path) -> "MicroTaskWorkspace":
        """Copy task_dir to a fresh tempfile.mkdtemp() and return workspace."""
        workspace_root = Path(tempfile.mkdtemp(prefix="p4_ws_"))
        shutil.copytree(task_dir, workspace_root, dirs_exist_ok=True)
        return cls(persistent_dir=Path(task_dir), workspace_root=workspace_root)

    def resolve_path(self, rel_path: str) -> Path:
        """Validate and resolve a workspace-relative path.

        Delegates to ``validate_path`` so traversal / absolute / secret
        basename violations raise ``PathValidationError``. Returns an
        absolute Path inside ``workspace_root``.
        """
        validate_path(rel_path)
        return self.workspace_root / rel_path

    def list_files(self, pattern: str | None = None) -> list[str]:
        """List workspace files, excluding ignore-listed dirs and patterns.

        Returns a sorted list of workspace-relative paths using forward
        slashes. If ``pattern`` is given, further filters via
        ``fnmatch.fnmatch`` on the relative path.
        """
        results: list[str] = []
        for path in self.workspace_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.workspace_root)
            if any(part in _IGNORE_DIRS for part in rel.parts):
                continue
            if any(fnmatch.fnmatch(path.name, pat) for pat in _IGNORE_PATTERNS):
                continue
            rel_str = rel.as_posix()
            if pattern is not None and not fnmatch.fnmatch(rel_str, pattern):
                continue
            results.append(rel_str)
        results.sort()
        return results

    def file_sha256(self, rel_path: str) -> str:
        """Compute the SHA-256 hex digest of a workspace file."""
        path = self.resolve_path(rel_path)
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(_SHA_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def cleanup(self) -> None:
        """Remove the temp workspace directory.

        Not in the spec interface but needed for test hygiene. Uses
        ``ignore_errors=True`` so it is safe to call twice.
        """
        shutil.rmtree(self.workspace_root, ignore_errors=True)
