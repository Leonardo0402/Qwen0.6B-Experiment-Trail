"""
scripts/generate_reference.py -- Validate task-bank references.

Reads unverified code_generation samples (output of generate_tasks.py),
runs validators.verify_sample on each, and emits only those whose reference
implementation self-verifies (is_accepted=True).

Families whose reference fails are logged and dropped.

Usage
-----
    python scripts/generate_reference.py [--in PATH] [--out PATH] [--timeout S]

Exit codes
----------
    0   success (at least one sample accepted)
    1   error or all samples rejected
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample  # noqa: E402
from src.validators import verify_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def validate_references(
    samples: list[Sample],
    *,
    run_ruff: bool = True,
    pytest_timeout_s: float = 10.0,
) -> tuple[list[Sample], list[tuple[Sample, list[str]]]]:
    """Verify each sample's reference implementation.

    Parameters
    ----------
    samples:
        code_generation Samples with unverified target_code.
    run_ruff:
        Whether to run ruff (advisory; does not gate acceptance).
    pytest_timeout_s:
        Per-pytest-invocation timeout in seconds.

    Returns
    -------
    (accepted, rejected)
        ``accepted`` — Samples whose is_accepted is True, with
        verified/verification fields updated in-place.
        ``rejected`` — List of (sample, messages) for dropped samples.
    """
    accepted: list[Sample] = []
    rejected: list[tuple[Sample, list[str]]] = []

    for sample in samples:
        result = verify_sample(
            sample,
            run_ruff=run_ruff,
            run_mypy=False,
            pytest_timeout_s=pytest_timeout_s,
        )
        if result.is_accepted:
            sample.verified = True
            sample.verification = result.verification
            accepted.append(sample)
        else:
            rejected.append((sample, result.messages))

    return accepted, rejected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate reference implementations; drop failing families.",
    )
    p.add_argument("--in", dest="in_path",
                   default="data/generated/tasks.jsonl",
                   help="Input JSONL of unverified code_generation samples.")
    p.add_argument("--out",
                   default="data/generated/reference.jsonl",
                   help="Output JSONL of verified samples.")
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
    out_path = Path(args.out)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    # Load
    samples: list[Sample] = []
    with in_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(Sample.from_json_line(line))

    print(f"generate_reference: loaded {len(samples)} samples from {in_path}")

    # Validate
    run_ruff = not args.no_ruff
    accepted, rejected = validate_references(
        samples,
        run_ruff=run_ruff,
        pytest_timeout_s=args.timeout,
    )

    print(f"  accepted: {len(accepted)}  rejected: {len(rejected)}")
    for sample, msgs in rejected:
        reason = "; ".join(msgs[:3]) if msgs else "unknown"
        print(f"  REJECTED {sample.sample_id}: {reason[:120]}")

    if not accepted:
        print("ERROR: no samples accepted -- check task family implementations.", file=sys.stderr)
        return 1

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in accepted:
            fh.write(s.to_json_line() + "\n")

    print(f"  wrote {len(accepted)} verified samples -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
