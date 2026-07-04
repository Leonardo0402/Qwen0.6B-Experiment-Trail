"""
scripts/verify_imported_mbpp.py -- Standalone verifier for imported MBPP samples.

Reads ``data/external/mbpp/normalized/<split>.jsonl`` (produced by
:mod:`scripts.import_mbpp`), runs the REAL verification pipeline on each
Sample, and writes:

    <output-dir>/verified/<split>.jsonl
    <output-dir>/rejected/<split>.jsonl

The verifier also updates ``<output-dir>/manifest.<split>.json`` with the
verified/rejected counts and SHA-256 digests.

Per the P3 task brief, the verifier performs these checks per sample:

1. Hard check: ``public_tests.count("assert ") >= 2``
2. Hard check: ``hidden_tests.count("assert ") >= 3``
3. For ``static_repair`` / ``execution_repair`` samples: call
   :func:`src.validators.verify_broken_is_broken` and require True
   (broken_code must fail at least one test).
4. For ``execution_repair`` samples: ``execution_feedback`` must be
   non-empty AND contain a failure marker (case-insensitive:
   any of ``Error``, ``assert``, ``Traceback``, ``FAILED``,
   ``Exception``, ``fail``).
5. Real verification: call :func:`src.validators.verify_sample` and
   require ``is_accepted=True`` (syntax_ok AND pytest_ok AND NOT timeout).

Samples that pass ALL checks go to ``verified/<split>.jsonl`` with
``verified=True`` and the real ``verification`` results.  Samples that
fail ANY check go to ``rejected/<split>.jsonl`` with ``verified=False``
and a ``rejection_reason`` field.

The verifier does NOT download datasets -- that is the importer's job.

Usage
-----
    python scripts/verify_imported_mbpp.py [--split test] [--output-dir DIR]

Exit codes
----------
    0   success (all samples verified, or partial rejection -- counts reported)
    1   error (input missing / no samples / all samples rejected)
    2   partial rejection (some verified, some rejected -- counts reported)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root import guard (so the script works from any cwd)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample  # noqa: E402
from src.validators import verify_broken_is_broken, verify_sample  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum assertion counts (P3 task brief constraints #7).
_MIN_PUBLIC_ASSERTS = 2
_MIN_HIDDEN_ASSERTS = 3

# Failure markers used to validate execution_feedback for execution_repair
# samples (case-insensitive substring match).  Per task brief constraint #9.
_FAILURE_MARKERS = (
    "error",
    "assert",
    "traceback",
    "failed",
    "exception",
    "fail",
)

# Task type strings (mirror src.schemas.TaskType enum values, kept as plain
# strings to avoid importing the enum for a trivial comparison).
_TASK_STATIC_REPAIR = "static_repair"
_TASK_EXECUTION_REPAIR = "execution_repair"


# ---------------------------------------------------------------------------
# Hard-check helpers (pure, importable, testable)
# ---------------------------------------------------------------------------

def count_asserts(text: str) -> int:
    """Count literal ``assert `` occurrences in *text* (case-sensitive).

    Per task brief constraint #7: ``public_tests.count("assert ")`` is the
    canonical counting method.  The trailing space avoids matching
    ``assertion`` / ``asserted``.
    """
    return (text or "").count("assert ")


def has_failure_marker(feedback: str) -> bool:
    """Return True if *feedback* contains a recognised failure marker.

    Case-insensitive substring match against :data:`_FAILURE_MARKERS`.
    Per task brief constraint #9.
    """
    if not feedback:
        return False
    lower = feedback.lower()
    return any(marker in lower for marker in _FAILURE_MARKERS)


def check_sample(sample: Sample) -> tuple[bool, str]:
    """Run all hard checks + real verification on *sample*.

    Returns
    -------
    (passed, reason)
        *passed* is True when the sample passes ALL checks.  *reason* is
        the empty string on success, or a short human-readable reason on
        failure.
    """
    # ------------------------------------------------------------------
    # Hard check 1: public assertions >= 2
    # ------------------------------------------------------------------
    n_public = count_asserts(sample.public_tests)
    if n_public < _MIN_PUBLIC_ASSERTS:
        return False, (
            f"public assertions {n_public} < {_MIN_PUBLIC_ASSERTS}"
        )

    # ------------------------------------------------------------------
    # Hard check 2: hidden assertions >= 3
    # ------------------------------------------------------------------
    n_hidden = count_asserts(sample.hidden_tests)
    if n_hidden < _MIN_HIDDEN_ASSERTS:
        return False, (
            f"hidden assertions {n_hidden} < {_MIN_HIDDEN_ASSERTS}"
        )

    # ------------------------------------------------------------------
    # Hard check 3 (repair only): verify_broken_is_broken
    # ------------------------------------------------------------------
    if sample.task_type in (_TASK_STATIC_REPAIR, _TASK_EXECUTION_REPAIR):
        broken = (sample.broken_code or "").strip()
        if not broken:
            # Schema should have rejected this earlier; defensive.
            return False, "repair sample missing broken_code"
        try:
            is_broken = verify_broken_is_broken(sample)
        except ValueError as exc:
            return False, f"verify_broken_is_broken raised: {exc}"
        if not is_broken:
            return False, "broken_code passes all tests"

    # ------------------------------------------------------------------
    # Hard check 4 (execution_repair only): execution_feedback failure marker
    # ------------------------------------------------------------------
    if sample.task_type == _TASK_EXECUTION_REPAIR:
        feedback = sample.execution_feedback or ""
        if not feedback.strip():
            return False, "execution_feedback is empty"
        if not has_failure_marker(feedback):
            return False, "execution_feedback lacks failure marker"

    # ------------------------------------------------------------------
    # Real verification: compile + pytest + ruff
    # ------------------------------------------------------------------
    sv = verify_sample(sample)
    # ALWAYS update the sample's verification with the real results so the
    # verified/rejected JSONL files reflect what was actually run.
    sample.verification = sv.verification
    if not sv.is_accepted:
        reason = "; ".join(sv.messages[:3]) if sv.messages else (
            "verification failed (syntax/pytest/timeout)"
        )
        return False, reason

    return True, ""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_normalized_jsonl(path: Path) -> list[Sample]:
    """Load Sample objects from a normalised JSONL file (blank lines skipped)."""
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(Sample.from_json_line(line))
    return samples


def write_verified_jsonl(samples: list[Sample], path: Path) -> None:
    """Write verified samples (verified=True) as one JSONL line each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(s.to_json_line() + "\n")


def write_rejected_jsonl(
    rejected: list[tuple[Sample, str]], path: Path
) -> None:
    """Write rejected samples to JSONL with a ``rejection_reason`` field.

    The Sample's own ``verified`` field is forced to False and
    ``verification`` is whatever was actually run (all-false when the
    sample failed a hard check before verification; real results when
    verification ran but failed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for sample, reason in rejected:
            record = json.loads(sample.to_json_line())
            record["verified"] = False
            record["rejection_reason"] = reason
            fh.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def update_manifest_with_verified(
    manifest_path: Path,
    *,
    verified_sha256: str,
    verified_count: int,
    rejected_count: int,
    rejected_sha256: str,
    verified_at: str,
) -> dict:
    """Update an existing per-split manifest with verifier-filled fields.

    Reads the manifest at *manifest_path*, fills in the verifier fields,
    writes it back, and returns the updated dict.
    """
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["verified_sha256"] = verified_sha256
    manifest["verified_count"] = verified_count
    manifest["rejected_count"] = rejected_count
    manifest["rejected_sha256"] = rejected_sha256
    manifest["verified_at"] = verified_at
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def verify_split(
    samples: list[Sample],
) -> tuple[list[Sample], list[tuple[Sample, str]]]:
    """Verify all *samples*; return (verified, rejected) lists.

    Verified samples have ``verified=True`` and ``verification`` updated to
    the real results from :func:`verify_sample`.  Rejected samples keep
    ``verified=False`` (will be re-stated in the rejection writer) and are
    paired with a short reason string.
    """
    verified: list[Sample] = []
    rejected: list[tuple[Sample, str]] = []
    for sample in samples:
        passed, reason = check_sample(sample)
        if passed:
            sample.verified = True
            # verification was already updated inside check_sample.
            verified.append(sample)
        else:
            sample.verified = False
            rejected.append((sample, reason))
    return verified, rejected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Verify imported MBPP samples (reads normalized/<split>.jsonl, "
            "writes verified/<split>.jsonl + rejected/<split>.jsonl)."
        ),
    )
    p.add_argument(
        "--output-dir", default="data/external/mbpp",
        help="Root output directory (default: data/external/mbpp).",
    )
    p.add_argument(
        "--split", default="test",
        help="Dataset split to verify (default: test).",
    )
    return p


def main() -> int:
    """CLI entry point.  Returns 0/1/2 (see module docstring)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    out_dir = Path(args.output_dir)
    split = args.split

    normalized_path = out_dir / "normalized" / f"{split}.jsonl"
    if not normalized_path.exists():
        print(
            f"ERROR: normalized input not found: {normalized_path}",
            file=sys.stderr,
        )
        print(
            "  Run scripts/import_mbpp.py first to produce the normalised "
            "JSONL for this split.",
            file=sys.stderr,
        )
        return 1

    samples = load_normalized_jsonl(normalized_path)
    if not samples:
        print(f"ERROR: no samples in {normalized_path}", file=sys.stderr)
        return 1

    print(
        f"verify_imported_mbpp: verifying {len(samples)} samples "
        f"from {normalized_path}"
    )

    verified, rejected = verify_split(samples)

    verified_path = out_dir / "verified" / f"{split}.jsonl"
    rejected_path = out_dir / "rejected" / f"{split}.jsonl"

    if verified:
        write_verified_jsonl(verified, verified_path)
    if rejected:
        write_rejected_jsonl(rejected, rejected_path)

    verified_sha = compute_sha256(verified_path) if verified else ""
    rejected_sha = compute_sha256(rejected_path) if rejected else ""

    verified_at = datetime.now(timezone.utc).isoformat()
    manifest_path = out_dir / f"manifest.{split}.json"
    if manifest_path.exists():
        update_manifest_with_verified(
            manifest_path,
            verified_sha256=verified_sha,
            verified_count=len(verified),
            rejected_count=len(rejected),
            rejected_sha256=rejected_sha,
            verified_at=verified_at,
        )
    else:
        print(
            f"WARNING: manifest {manifest_path} not found; skipped manifest "
            "update.",
            file=sys.stderr,
        )

    print(
        f"  n_in={len(samples)}  verified={len(verified)}  "
        f"rejected={len(rejected)}"
    )
    if verified:
        print(f"  verified -> {verified_path}")
        print(f"    sha256: {verified_sha[:16]}...")
    if rejected:
        print(f"  rejected -> {rejected_path}")
        print(f"    sha256: {rejected_sha[:16]}...")
        # Top rejection reasons
        hist: dict[str, int] = {}
        for _, reason in rejected:
            key = reason[:60]
            hist[key] = hist.get(key, 0) + 1
        print("  Rejection reasons:")
        for reason, count in sorted(hist.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    [{count}] {reason}")

    # Exit code: 1 = all rejected, 2 = partial rejection, 0 = all verified.
    if not verified:
        return 1
    if rejected:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
