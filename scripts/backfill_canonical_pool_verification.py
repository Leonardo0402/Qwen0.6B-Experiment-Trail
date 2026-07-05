"""scripts/backfill_canonical_pool_verification.py -- Issue #10 Fix 1.

Backfills the real ``verification`` subfields for the 501 P2-replay-derived
samples in ``data/p3-curriculum/canonical-pool.jsonl`` that shipped with the
placeholder ``Verification(syntax_ok=False, pytest_ok=False, ruff_ok=False,
timeout=False)`` and ``verified=False``.

Flow per sample:

  1. If ``verified=True`` AND any verification subfield is True: skip
     (already genuinely verified -- the 281 mbpp-importer code samples).
  2. Otherwise: run ``pad_hidden_tests(sample)`` to extend hidden_tests with
     boundary-condition asserts (per A2 design), then
     ``verify_sample(sample_padded, run_ruff=True, pytest_timeout_s=10.0)``.
  3. Update via
     ``model_copy(update={"verified": sv.is_accepted,
                          "verification": sv.verification,
                          "hidden_tests": padded_hidden})``.

Output:

  - Overwrites ``data/p3-curriculum/canonical-pool.jsonl`` (byte-stable:
    UTF-8, ensure_ascii=False, one JSON per line).
  - Writes ``data/p3-curriculum/canonical-pool-backfill-manifest.json`` with
    before/after counts, per-variant_type breakdown, and timestamp.
  - Writes ``data/p3-curriculum/canonical-pool.jsonl.pre-backfill.bak`` on
    first run (preserves the pre-backfill state for traceability; not
    overwritten on subsequent runs).

Usage
-----
    python scripts/backfill_canonical_pool_verification.py

Exit codes
----------
    0   success
    1   I/O or schema error
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root import guard (so the script works from any cwd)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hidden_test_padding import pad_hidden_tests  # noqa: E402
from src.schemas import Sample  # noqa: E402
from src.validators import verify_sample  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
GENERATOR_NAME: str = "backfill_canonical_pool_verification.py"

POOL_PATH = _ROOT / "data" / "p3-curriculum" / "canonical-pool.jsonl"
POOL_BACKUP_PATH = (
    _ROOT / "data" / "p3-curriculum" / "canonical-pool.jsonl.pre-backfill.bak"
)
MANIFEST_PATH = (
    _ROOT / "data" / "p3-curriculum" / "canonical-pool-backfill-manifest.json"
)

PYTEST_TIMEOUT_S: float = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_samples(path: Path) -> list:
    """Stream a JSONL file into a list of Sample objects (preserves order)."""
    samples: list = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(Sample.from_json_line(line))
    return samples


def _has_real_verification(sample: Sample) -> bool:
    """True iff *sample* has at least one True verification subfield.

    Used to detect samples that were genuinely verified (skip) vs samples
    that still carry the False-preset placeholder (backfill).
    """
    v = sample.verification
    return bool(v.syntax_ok or v.pytest_ok or v.ruff_ok or v.timeout)


def _write_jsonl(samples: list, path: Path) -> None:
    """Write samples to *path* as JSONL (UTF-8, ensure_ascii=False, \\n)."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(s.to_json_line())
            fh.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point. Returns 0 on success, 1 on error."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    if not POOL_PATH.exists():
        print(f"ERROR: canonical pool not found: {POOL_PATH}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 1. Create pre-backfill backup (only on first run)
    # ------------------------------------------------------------------
    if not POOL_BACKUP_PATH.exists():
        shutil.copy2(POOL_PATH, POOL_BACKUP_PATH)
        print(f"Created pre-backfill backup: {POOL_BACKUP_PATH.name}")
    else:
        print(f"Pre-backfill backup already exists: {POOL_BACKUP_PATH.name}")

    # ------------------------------------------------------------------
    # 2. Load samples
    # ------------------------------------------------------------------
    samples = _load_samples(POOL_PATH)
    before_count = len(samples)
    print(f"Loaded canonical pool: {before_count} samples")

    # ------------------------------------------------------------------
    # 3. Backfill each sample
    # ------------------------------------------------------------------
    backfilled_sample_ids: list = []
    skipped_count = 0
    by_vt_stats: dict = {}

    def _ensure_vt(vt: str) -> dict:
        if vt not in by_vt_stats:
            by_vt_stats[vt] = {
                "total": 0,
                "backfilled": 0,
                "skipped_already_verified": 0,
                "verified_true_after": 0,
                "verified_false_after": 0,
            }
        return by_vt_stats[vt]

    t0 = time.perf_counter()
    new_samples: list = []
    for i, s in enumerate(samples, start=1):
        vt = s.variant_type or "unknown"
        stats = _ensure_vt(vt)
        stats["total"] += 1

        if s.verified and _has_real_verification(s):
            # Already genuinely verified -- skip
            stats["skipped_already_verified"] += 1
            skipped_count += 1
            new_samples.append(s)
            if i % 100 == 0 or i == before_count:
                print(f"  [{i}/{before_count}] skipped {i} samples "
                      f"(already verified)")
            continue

        # Backfill: pad_hidden_tests then verify_sample
        try:
            padded_sample, _reason = pad_hidden_tests(s)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [{i}/{before_count}] pad_hidden_tests raised for "
                  f"{s.sample_id}: {exc!r}", file=sys.stderr)
            padded_sample = s  # fall back to original (no padding)

        try:
            sv = verify_sample(
                padded_sample,
                run_ruff=True,
                pytest_timeout_s=PYTEST_TIMEOUT_S,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [{i}/{before_count}] verify_sample raised for "
                  f"{s.sample_id}: {exc!r}", file=sys.stderr)
            # On unexpected verifier error, mark as not verified
            new_samples.append(s)
            stats["backfilled"] += 1
            stats["verified_false_after"] += 1
            backfilled_sample_ids.append(s.sample_id)
            continue

        new_verified: bool = sv.is_accepted
        new_samples.append(
            padded_sample.model_copy(update={
                "verified": new_verified,
                "verification": sv.verification,
            })
        )
        stats["backfilled"] += 1
        backfilled_sample_ids.append(s.sample_id)
        if new_verified:
            stats["verified_true_after"] += 1
        else:
            stats["verified_false_after"] += 1

        if i % 25 == 0 or i == before_count:
            elapsed = time.perf_counter() - t0
            print(f"  [{i}/{before_count}] backfilled {i} samples "
                  f"({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - t0
    print(f"\nBackfill complete: {len(backfilled_sample_ids)} backfilled, "
          f"{skipped_count} skipped ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # 4. Write updated canonical pool (byte-stable)
    # ------------------------------------------------------------------
    _write_jsonl(new_samples, POOL_PATH)
    print(f"Wrote updated canonical pool: {POOL_PATH}")

    # ------------------------------------------------------------------
    # 5. Compute manifest statistics
    # ------------------------------------------------------------------
    after_count = len(new_samples)
    backfilled_count = len(backfilled_sample_ids)
    verified_true_after = sum(
        1 for s in new_samples if s.verified
    )
    verified_false_after = after_count - verified_true_after
    dropped_count = verified_false_after

    # Fill in verified_true_after / verified_false_after for skipped buckets
    # (skipped samples retain their verified=True state).
    for vt, stats in by_vt_stats.items():
        # Recompute from new_samples for consistency
        vt_samples = [s for s in new_samples if (s.variant_type or "unknown") == vt]
        stats["verified_true_after"] = sum(1 for s in vt_samples if s.verified)
        stats["verified_false_after"] = sum(1 for s in vt_samples if not s.verified)

    timestamp = datetime.now(timezone.utc).isoformat()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp,
        "timestamp": timestamp,
        "generator": GENERATOR_NAME,
        "pool_path": str(POOL_PATH.relative_to(_ROOT)).replace("\\", "/"),
        "backup_path": str(POOL_BACKUP_PATH.relative_to(_ROOT)).replace("\\", "/"),
        "before_count": before_count,
        "after_count": after_count,
        "backfilled_count": backfilled_count,
        "skipped_count": skipped_count,
        "dropped_count": dropped_count,
        "verified_true_after": verified_true_after,
        "verified_false_after": verified_false_after,
        "by_variant_type": by_vt_stats,
        "backfilled_sample_ids": backfilled_sample_ids,
        "pytest_timeout_s": PYTEST_TIMEOUT_S,
        "run_ruff": True,
    }

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote backfill manifest: {MANIFEST_PATH}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\nBackfill summary:")
    print(f"  before_count:         {before_count}")
    print(f"  after_count:         {after_count}")
    print(f"  backfilled_count:    {backfilled_count}")
    print(f"  skipped_count:       {skipped_count}")
    print(f"  dropped_count:       {dropped_count} "
          f"(verified=False after backfill)")
    print(f"  verified_true_after: {verified_true_after}")
    print(f"  verified_false_after:{verified_false_after}")
    print(f"  by_variant_type:")
    for vt in sorted(by_vt_stats.keys()):
        s = by_vt_stats[vt]
        print(f"    {vt:20s}: total={s['total']:4d} "
              f"backfilled={s['backfilled']:4d} "
              f"skipped={s['skipped_already_verified']:4d} "
              f"verified_true={s['verified_true_after']:4d} "
              f"verified_false={s['verified_false_after']:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
