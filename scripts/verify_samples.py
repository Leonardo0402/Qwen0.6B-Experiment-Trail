"""
scripts/verify_samples.py -- Batch sample verifier.

Reads a JSONL of Sample objects, runs validators.verify_sample on each,
and writes accepted samples (is_accepted=True) to --verified-out and
rejected samples to --rejected-out with a reason field appended.

Usage
-----
    python scripts/verify_samples.py
        [--in PATH]
        [--verified-out PATH]
        [--rejected-out PATH]
        [--timeout S]
        [--no-ruff]

Exit codes
----------
    0   success
    1   error (I/O, no samples, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample  # noqa: E402
from src.validators import verify_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class VerifyResult:
    """Summary of a batch verification run."""

    accepted: list[Sample] = field(default_factory=list)
    rejected: list[tuple[Sample, str]] = field(default_factory=list)

    @property
    def n_in(self) -> int:
        return self.n_verified + self.n_rejected

    @property
    def n_verified(self) -> int:
        return len(self.accepted)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    @property
    def reasons(self) -> dict[str, int]:
        """Histogram of rejection reason prefixes."""
        hist: dict[str, int] = {}
        for _, reason in self.rejected:
            key = reason[:60] if reason else "unknown"
            hist[key] = hist.get(key, 0) + 1
        return hist


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def verify_jsonl_samples(
    samples: list[Sample],
    *,
    run_ruff: bool = True,
    timeout_s: float = 10.0,
) -> VerifyResult:
    """Verify each sample; return a VerifyResult with accepted / rejected lists.

    Accepted samples have their ``verified`` and ``verification`` fields
    updated in-place (since Sample is not frozen).

    Parameters
    ----------
    samples:
        Input samples (any task_type).
    run_ruff:
        Whether to run ruff advisory lint.
    timeout_s:
        Per-pytest-invocation timeout.

    Returns
    -------
    VerifyResult
    """
    result = VerifyResult()
    for sample in samples:
        sv = verify_sample(sample, run_ruff=run_ruff, pytest_timeout_s=timeout_s)
        if sv.is_accepted:
            sample.verified = True
            sample.verification = sv.verification
            result.accepted.append(sample)
        else:
            reason = "; ".join(sv.messages[:3]) if sv.messages else "unknown"
            result.rejected.append((sample, reason))
    return result


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(Sample.from_json_line(line))
    return samples


def _write_jsonl_samples(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(s.to_json_line() + "\n")


def _write_rejected_jsonl(
    rejected: list[tuple[Sample, str]], path: Path
) -> None:
    """Write rejected samples to JSONL, injecting a '_reject_reason' field."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for sample, reason in rejected:
            record = json.loads(sample.to_json_line())
            record["_reject_reason"] = reason
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify a JSONL of samples; split into accepted and rejected.",
    )
    p.add_argument("--in", dest="in_path",
                   default="data/generated/mutations.jsonl",
                   help="Input JSONL of samples to verify.")
    p.add_argument("--verified-out",
                   default="data/verified/samples.jsonl",
                   help="Output JSONL for accepted (verified) samples.")
    p.add_argument("--rejected-out",
                   default="data/rejected/samples.jsonl",
                   help="Output JSONL for rejected samples.")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="Per-pytest timeout in seconds.")
    p.add_argument("--no-ruff", action="store_true",
                   help="Skip ruff advisory lint.")
    return p


def main() -> int:
    """CLI entry point."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    in_path = Path(args.in_path)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    samples = _load_jsonl(in_path)
    if not samples:
        print(f"ERROR: no samples in {in_path}", file=sys.stderr)
        return 1

    print(f"verify_samples: {len(samples)} samples from {in_path}")

    run_ruff = not args.no_ruff
    vr = verify_jsonl_samples(samples, run_ruff=run_ruff, timeout_s=args.timeout)

    print(
        f"  n_in={vr.n_in}  verified={vr.n_verified}  rejected={vr.n_rejected}"
    )

    if vr.rejected:
        print("  Rejection reasons:")
        for reason, count in sorted(
            vr.reasons.items(), key=lambda kv: -kv[1]
        )[:10]:
            print(f"    [{count}] {reason}")

    if vr.accepted:
        _write_jsonl_samples(vr.accepted, Path(args.verified_out))
        print(f"  wrote {vr.n_verified} verified -> {args.verified_out}")

    if vr.rejected:
        _write_rejected_jsonl(vr.rejected, Path(args.rejected_out))
        print(f"  wrote {vr.n_rejected} rejected -> {args.rejected_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
