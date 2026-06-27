"""
scripts/download_model.py -- Download Qwen3-0.6B from HuggingFace Hub.

Downloads the model into models/Qwen3-0.6B and writes a manifest to
data/manifests/model_manifest.json.

Usage:
    python scripts/download_model.py [--repo-id REPO_ID] [--local-dir LOCAL_DIR] [--force]

Exit codes:
    0  success
    1  failure

Windows DLL / import-order note
---------------------------------
Heavy library imports are performed eagerly at module-load time, before any
CUDA initialisation, to avoid the Windows access-violation crash (0xC0000005)
that occurs when certain C-extension libraries are imported after torch.cuda.
This script does NOT use torch or datasets at all; only huggingface_hub is
needed.  An OSError [WinError 126] (missing DLL) is caught here so the module
can still be imported and the error reported gracefully.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Eager imports -- catch missing-DLL OSError on Windows.
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import snapshot_download as _snapshot_download
    _HF_AVAILABLE = True
    _HF_ERROR: Optional[str] = None
except (ImportError, OSError) as _exc:
    _snapshot_download = None  # type: ignore[assignment]
    _HF_AVAILABLE = False
    _HF_ERROR = f"{type(_exc).__name__}: {_exc}"

# ---------------------------------------------------------------------------
# Project root (one level up from scripts/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Artifact verification
# ---------------------------------------------------------------------------

#: Checks performed against the local model directory.
REQUIRED_ARTIFACTS: Dict[str, object] = {
    "config.json": lambda d: (d / "config.json").exists(),
    "tokenizer": lambda d: (
        (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists()
    ),
    "safetensors": lambda d: bool(list(d.glob("*.safetensors"))),
}


def verify_artifacts(local_dir: Path) -> Dict[str, str]:
    """Return PRESENT/MISSING status for each required model artifact.

    Parameters
    ----------
    local_dir:
        Path to the local model directory.

    Returns
    -------
    dict mapping artifact name -> "PRESENT" or "MISSING".
    """
    return {
        name: "PRESENT" if check(local_dir) else "MISSING"  # type: ignore[operator]
        for name, check in REQUIRED_ARTIFACTS.items()
    }


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def build_manifest(
    repo_id: str,
    local_dir: Path,
    commit_hash: Optional[str],
    download_date: str,
    files: List[str],
) -> dict:
    """Build the manifest dict (does not write it to disk).

    Parameters
    ----------
    repo_id:
        HuggingFace repo identifier, e.g. "Qwen/Qwen3-0.6B".
    local_dir:
        Absolute local path where the model was saved.
    commit_hash:
        Resolved HF commit hash, or None when unavailable.
    download_date:
        ISO-8601 timestamp string (UTC).
    files:
        Sorted list of relative file paths present in *local_dir*.
    """
    return {
        "repo_id": repo_id,
        "local_dir": str(local_dir.resolve()),
        "commit_hash": commit_hash,
        "download_date": download_date,
        "files": files,
    }


def write_manifest(manifest: dict, manifest_path: Path) -> None:
    """Write *manifest* as formatted JSON to *manifest_path*.

    Creates parent directories as needed.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Download-decision helper
# ---------------------------------------------------------------------------

def should_skip_download(local_dir: Path, force: bool) -> bool:
    """Return True when the model appears present and *force* is False.

    "Present" means the directory exists AND contains config.json.
    """
    if force:
        return False
    return local_dir.exists() and (local_dir / "config.json").exists()


# ---------------------------------------------------------------------------
# Internal helpers (thin wrappers around HF APIs -- not tested directly)
# ---------------------------------------------------------------------------

def _resolve_commit_hash(repo_id: str) -> Optional[str]:
    """Try to resolve the latest commit hash for *repo_id*; return None on failure."""
    try:
        from huggingface_hub import model_info as _model_info
        info = _model_info(repo_id)
        return getattr(info, "sha", None)
    except Exception:  # noqa: BLE001
        return None


def _list_model_files(local_dir: Path) -> List[str]:
    """Return a sorted list of relative file paths under *local_dir*."""
    if not local_dir.exists():
        return []
    return sorted(
        str(p.relative_to(local_dir)).replace("\\", "/")
        for p in local_dir.rglob("*")
        if p.is_file()
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Download the model, verify artifacts, write manifest.

    Returns 0 on success, 1 on failure.
    """
    # Ensure stdout can handle non-ASCII (model paths, etc.) without crashing on
    # gbk/cp936 consoles that are common when using conda run on Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="Download Qwen3-0.6B from HuggingFace Hub."
    )
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace repo ID (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--local-dir",
        default=str(_PROJECT_ROOT / "models" / "Qwen3-0.6B"),
        help="Local directory to download into (default: models/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the model is already present",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    manifest_path = _PROJECT_ROOT / "data" / "manifests" / "model_manifest.json"

    if not _HF_AVAILABLE:
        print(f"ERROR: huggingface_hub not available: {_HF_ERROR}")
        return 1

    # ------------------------------------------------------------------
    # Download or skip
    # ------------------------------------------------------------------
    if should_skip_download(local_dir, args.force):
        print(f"Model already present at {local_dir} -- skipping download.")
        print("  (Use --force to re-download.)")
    else:
        print(f"Downloading {args.repo_id} -> {local_dir} ...")
        try:
            _snapshot_download(  # type: ignore[misc]
                repo_id=args.repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print("Download complete.")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR during download: {exc}")
            return 1

    # ------------------------------------------------------------------
    # Verify artifacts
    # ------------------------------------------------------------------
    print("\nVerifying artifacts:")
    statuses = verify_artifacts(local_dir)
    all_ok = True
    for name, status in statuses.items():
        print(f"  {name}: {status}")
        if status == "MISSING":
            all_ok = False

    if not all_ok:
        print("\nWARNING: some artifacts are MISSING -- the download may be incomplete.")

    # ------------------------------------------------------------------
    # Build and write manifest
    # ------------------------------------------------------------------
    commit_hash = _resolve_commit_hash(args.repo_id)
    download_date = datetime.now(timezone.utc).isoformat()
    files = _list_model_files(local_dir)
    manifest = build_manifest(args.repo_id, local_dir, commit_hash, download_date, files)
    write_manifest(manifest, manifest_path)

    print(f"\nManifest written to: {manifest_path}")
    print(f"  repo_id:       {manifest['repo_id']}")
    print(f"  local_dir:     {manifest['local_dir']}")
    print(f"  commit_hash:   {manifest['commit_hash']}")
    print(f"  download_date: {manifest['download_date']}")
    print(f"  files:         {len(files)} file(s)")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
