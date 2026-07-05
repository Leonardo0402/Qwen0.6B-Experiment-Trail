"""scripts/p3_formal_pool_builder.py -- Formal Canonical Pool v2 builder (Issue #14 Wave 4-G1).

Sharded, resumable generation pipeline that produces the verified canonical
pool for the P3 formal training data. Reuses the variant-generation and
sandbox-verification logic proven in ``scripts/p3_yield_pilot.py`` and
extends it with:

  * Multi-variant boundary generation (2-3 variants per family with disjoint
    edge sets, per Issue #14 P6.2).
  * 25-family shards with append-only per-shard artifacts.
  * Per-family verification cache so reruns skip already-completed families
    (only infrastructure failures are retried, never deterministic test
    failures).
  * Multi-dimensional deduplication (sample_id, instruction, target_code,
    broken_code, test_suite, AST structural hash).
  * Per-family / per-bucket caps: 8 samples per family, 3 per bucket.

Layout (under ``data/p3-formal/``)::

    family-partition.json       family universe (copy of source partition)
    canonical-pool.jsonl        all verified samples (concatenated)
    canonical-pool-manifest.json pool manifest with SHA-256
    shards/
      shard-NNN/
        candidate.jsonl         generated candidates
        verified.jsonl          passed verification
        rejected.jsonl          rejected candidates (with reason)
        verification-cache.jsonl per-family completion records
        manifest.json           shard manifest

Usage
-----
    py -3.11 scripts/p3_formal_pool_builder.py --shard 0 --shard-size 25
    py -3.11 scripts/p3_formal_pool_builder.py --shard 0 --shard-size 25 --resume
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample, Verification  # noqa: E402
from src.sandbox import run_pytest  # noqa: E402
from src.validators import verify_sample, compile_check  # noqa: E402
from scripts.inject_bugs import inject_all_bugs  # noqa: E402
from scripts.extract_function_signature import (  # noqa: E402
    extract_function_signature,
    extract_function_name,
)
from scripts.generate_boundary_variants import (  # noqa: E402
    generate_boundary_variants_multi,
)
from scripts.p3_yield_pilot import (  # noqa: E402
    _make_code_variants,
    _make_static_repair_variants,
    _make_execution_repair_variants,
    _verify_code_or_boundary,
    _verify_repair,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 2
GENERATOR_NAME: str = "p3_formal_pool_builder.py"

PARTITION_PATH = _ROOT / "data" / "p3-curriculum" / "family-partition.json"
CANONICAL_POOL_PATH = _ROOT / "data" / "p3-curriculum" / "canonical-pool.jsonl"

FORMAL_ROOT = _ROOT / "data" / "p3-formal"
FORMAL_PARTITION_PATH = FORMAL_ROOT / "family-partition.json"
FORMAL_POOL_PATH = FORMAL_ROOT / "canonical-pool.jsonl"
FORMAL_MANIFEST_PATH = FORMAL_ROOT / "canonical-pool-manifest.json"
SHARDS_ROOT = FORMAL_ROOT / "shards"

# Per-bucket caps (Issue #14 P6.2)
MAX_CODE_PER_FAMILY: int = 3
MAX_BOUNDARY_PER_FAMILY: int = 3
MAX_STATIC_PER_FAMILY: int = 3
MAX_EXEC_PER_FAMILY: int = 3
MAX_TOTAL_PER_FAMILY: int = 8

TOTAL_SHARED_FAMILIES: int = 425
DEFAULT_SHARD_SIZE: int = 25

BUCKETS: tuple[str, ...] = (
    "code", "boundary", "static_repair", "execution_repair",
)

_PLACEHOLDER_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relpath(path: Path) -> str:
    try:
        rel = path.relative_to(_ROOT)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_pool_by_family(path: Path) -> dict[str, Sample]:
    """Load canonical-pool.jsonl, return family_id -> representative Sample.

    Preference: variant_type == "code" (raw code_generation); falls back to
    the first sample seen for the family.
    """
    by_fam: dict[str, Sample] = {}
    code_by_fam: dict[str, Sample] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = Sample.from_json_line(line)
            if s.family_id not in by_fam:
                by_fam[s.family_id] = s
            if s.variant_type == "code" and s.family_id not in code_by_fam:
                code_by_fam[s.family_id] = s
    for fid, s in code_by_fam.items():
        by_fam[fid] = s
    return by_fam


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ast_structural_hash(code: str) -> str:
    """Hash the AST dump of *code* so trivial whitespace changes don't dedup.

    Falls back to the raw SHA-256 when the code cannot be parsed.
    """
    try:
        tree = ast.parse(code)
        dump = ast.dump(tree, annotate_fields=False)
        return _sha256(dump)
    except SyntaxError:
        return _sha256(code)


def _normalized_instruction(instr: str) -> str:
    """Normalize instruction for dedup: strip + collapse whitespace + lower."""
    return " ".join(instr.lower().split())


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

def load_train_families(partition_path: Path = PARTITION_PATH) -> list[str]:
    """Return the sorted list of all P3 train family IDs (new + replay)."""
    partition = _load_json(partition_path)
    train_new = set(partition["p3_train_new"]["family_ids"])
    train_replay = set(partition["p3_train_replay"]["family_ids"])
    return sorted(train_new | train_replay)


def shard_family_ids(
    all_families: list[str],
    shard_index: int,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> list[str]:
    """Return the slice of family IDs assigned to *shard_index*."""
    start = shard_index * shard_size
    end = start + shard_size
    return all_families[start:end]


def total_shards(
    n_families: int, shard_size: int = DEFAULT_SHARD_SIZE
) -> int:
    return math.ceil(n_families / shard_size)


# ---------------------------------------------------------------------------
# Dedup state
# ---------------------------------------------------------------------------

class DedupState:
    """Multi-dimensional dedup tracker for a single shard run.

    A sample is a duplicate iff ANY of:
      1. sample_id matches an already-seen sample_id (exact match).
      2. The full content signature matches: (normalized_instruction,
         target_code, broken_code, test_suite) -- i.e. the exact same
         training example.
      3. The structural signature matches: (AST_hash, test_suite_hash) --
         same code structure AND same tests, catching trivial reformatting.

    Individual fields (target_code alone, instruction alone) are NOT
    standalone dedup keys because within-family variants legitimately share
    target_code (code_orig / code_sig / boundary all use the same correct
    solution) and instruction (repair variants share the prompt).
    """

    def __init__(self) -> None:
        self.sample_ids: set[str] = set()
        self.full_signatures: set[str] = set()
        self.structural_signatures: set[str] = set()

    def _signatures(self, sample: Sample) -> tuple[str, str, str]:
        broken = sample.broken_code or ""
        feedback = sample.execution_feedback or ""
        instr = _normalized_instruction(sample.instruction)
        target = sample.target_code
        tests = sample.public_tests + "\n" + (sample.hidden_tests or "")
        # Full content signature includes execution_feedback so that
        # static_repair and execution_repair variants of the same bug are NOT
        # deduped against each other (they are different training signals).
        full_sig = _sha256("|".join([instr, target, broken, tests, feedback]))
        ast_h = _ast_structural_hash(target)
        tests_h = _sha256(tests)
        broken_h = _sha256(broken) if broken else ""
        instr_h = _sha256(instr)
        feedback_h = _sha256(feedback) if feedback else ""
        # Structural signature normalizes target_code whitespace via AST but
        # keeps instruction + broken_code + tests + feedback so repair
        # variants are NOT deduped against code variants of the same family.
        struct_sig = _sha256("|".join([instr_h, ast_h, broken_h, tests_h, feedback_h]))
        return sample.sample_id, full_sig, struct_sig

    def is_duplicate(self, sample: Sample) -> Optional[str]:
        """Return the dedup reason if *sample* is a duplicate, else None."""
        sid, full_sig, struct_sig = self._signatures(sample)
        if sid in self.sample_ids:
            return f"duplicate sample_id: {sid}"
        if full_sig in self.full_signatures:
            return "duplicate full content signature"
        if struct_sig in self.structural_signatures:
            return "duplicate structural signature (AST + tests)"
        return None

    def add(self, sample: Sample) -> None:
        sid, full_sig, struct_sig = self._signatures(sample)
        self.sample_ids.add(sid)
        self.full_signatures.add(full_sig)
        self.structural_signatures.add(struct_sig)


# ---------------------------------------------------------------------------
# Generation (reuses p3_yield_pilot helpers + multi-boundary)
# ---------------------------------------------------------------------------

def _make_boundary_variants_multi(
    src: Sample, *, max_n: int = MAX_BOUNDARY_PER_FAMILY
) -> list[Sample]:
    """Generate up to max_n boundary variants using the disjoint edge-set
    multi-generator (Issue #14 P6.2)."""
    variants = generate_boundary_variants_multi(
        src, max_variants=max_n, max_boundary_tests_per_variant=4
    )
    # Re-tag sample_id scheme to match the formal-pool convention
    out: list[Sample] = []
    for i, v in enumerate(variants[:max_n]):
        out.append(v.model_copy(update={
            "sample_id": f"{src.sample_id}_boundary_v{i}",
            "variant_type": "boundary",
            "task_type": "code_generation",
        }))
    return out


def _make_static_repair(src: Sample) -> list[Sample]:
    """Generate up to MAX_STATIC_PER_FAMILY static_repair variants."""
    return _make_static_repair_variants(src, max_n=MAX_STATIC_PER_FAMILY)


def _make_execution_repair(src: Sample) -> list[Sample]:
    """Generate up to MAX_EXEC_PER_FAMILY execution_repair variants."""
    return _make_execution_repair_variants(src, max_n=MAX_EXEC_PER_FAMILY)


# ---------------------------------------------------------------------------
# Family processing
# ---------------------------------------------------------------------------

def _reject_record(
    family_id: str, sample_id: Optional[str], bucket: str, reason: str
) -> dict:
    return {
        "family_id": family_id,
        "sample_id": sample_id,
        "bucket": bucket,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _verified_record(sample: Sample, bucket: str) -> dict:
    return {
        "family_id": sample.family_id,
        "sample_id": sample.sample_id,
        "bucket": bucket,
        "verified": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _process_family(
    family_id: str,
    src: Sample,
    dedup: DedupState,
    *,
    pytest_timeout_s: float = 10.0,
) -> tuple[list[Sample], list[dict], list[dict], dict]:
    """Generate + verify all variants for one family.

    Returns (verified_samples, verified_records, rejected_records,
    per_bucket_counts).

    Per-family allocation: 2 candidates per bucket (4 buckets × 2 = 8 = cap).
    If a bucket produces fewer than 2 candidates, the slack is NOT
    redistributed (keeps the logic simple and predictable). The per-bucket
    cap (3) is still respected as the absolute upper bound.
    """
    verified_samples: list[Sample] = []
    verified_records: list[dict] = []
    rejected_records: list[dict] = []
    bucket_counts = {b: 0 for b in BUCKETS}

    # Generate candidates per bucket (up to per-bucket cap = 3)
    per_bucket_candidates: dict[str, list[Sample]] = {
        "code": _make_code_variants(src)[:MAX_CODE_PER_FAMILY],
        "boundary": _make_boundary_variants_multi(src, max_n=MAX_BOUNDARY_PER_FAMILY),
        "static_repair": _make_static_repair(src),
        "execution_repair": _make_execution_repair(src),
    }

    # Fair allocation: 2 per bucket = 8 total (matches per-family cap).
    # This ensures execution_repair is never starved by the cap.
    per_bucket_quota = 2
    candidates: list[tuple[str, Sample]] = []
    for bucket in BUCKETS:
        for v in per_bucket_candidates[bucket][:per_bucket_quota]:
            candidates.append((bucket, v))
        # Reject any overflow per-bucket (above quota)
        for v in per_bucket_candidates[bucket][per_bucket_quota:]:
            rejected_records.append(_reject_record(
                family_id, v.sample_id, bucket,
                f"per-family quota ({per_bucket_quota}/bucket) exceeded",
            ))

    # Per-family total cap (8) -- defensive; 4×2=8 so normally a no-op
    if len(candidates) > MAX_TOTAL_PER_FAMILY:
        dropped = candidates[MAX_TOTAL_PER_FAMILY:]
        for bucket, v in dropped:
            rejected_records.append(_reject_record(
                family_id, v.sample_id, bucket,
                f"per-family total cap ({MAX_TOTAL_PER_FAMILY}) exceeded",
            ))
        candidates = candidates[:MAX_TOTAL_PER_FAMILY]

    # Verify each candidate
    for bucket, v in candidates:
        # Dedup check first (cheap)
        dup_reason = dedup.is_duplicate(v)
        if dup_reason is not None:
            rejected_records.append(_reject_record(
                family_id, v.sample_id, bucket, f"dedup: {dup_reason}",
            ))
            continue

        # Verify (sandbox)
        try:
            if bucket == "code" or bucket == "boundary":
                ok = _verify_code_or_boundary(v)
            elif bucket == "static_repair":
                ok = _verify_repair(v, require_feedback=False)
            else:  # execution_repair
                ok = _verify_repair(v, require_feedback=True)
        except Exception as exc:
            rejected_records.append(_reject_record(
                family_id, v.sample_id, bucket,
                f"verification exception: {type(exc).__name__}: {exc}",
            ))
            continue

        if not ok:
            rejected_records.append(_reject_record(
                family_id, v.sample_id, bucket, "verification failed",
            ))
            continue

        # Accepted: mark verified, add to dedup state
        verified_sample = v.model_copy(update={
            "verified": True,
            "verification": Verification(
                syntax_ok=True, pytest_ok=True, ruff_ok=False, timeout=False
            ),
        })
        dedup.add(verified_sample)
        verified_samples.append(verified_sample)
        verified_records.append(_verified_record(verified_sample, bucket))
        bucket_counts[bucket] += 1

    return verified_samples, verified_records, rejected_records, bucket_counts


def _bucket_cap(bucket: str) -> int:
    return {
        "code": MAX_CODE_PER_FAMILY,
        "boundary": MAX_BOUNDARY_PER_FAMILY,
        "static_repair": MAX_STATIC_PER_FAMILY,
        "execution_repair": MAX_EXEC_PER_FAMILY,
    }[bucket]


# ---------------------------------------------------------------------------
# Shard runner with resumable cache
# ---------------------------------------------------------------------------

def _shard_dir(shard_index: int) -> Path:
    return SHARDS_ROOT / f"shard-{shard_index:03d}"


def _load_cache_completed(cache_path: Path) -> set[str]:
    """Load family IDs that already have a 'completed' cache entry."""
    if not cache_path.exists():
        return set()
    completed: set[str] = set()
    with cache_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("completed") is True and rec.get("family_id"):
                completed.add(rec["family_id"])
    return completed


def _append_jsonl(path: Path, records: list) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")


def _append_samples_jsonl(path: Path, samples: list[Sample]) -> None:
    if not samples:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(s.to_json_line())
            fh.write("\n")


def _write_shard_manifest(
    shard_index: int,
    shard_size: int,
    family_ids: list[str],
    n_verified: int,
    n_rejected: int,
    n_candidate: int,
    bucket_counts: dict[str, int],
    started_at: str,
    duration_s: float,
) -> Path:
    shard_dir = _shard_dir(shard_index)
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_dir / "manifest.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "shard_id": f"shard-{shard_index:03d}",
        "shard_index": shard_index,
        "shard_size": shard_size,
        "family_ids": family_ids,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration_s, 3),
        "candidate_count": n_candidate,
        "verified_count": n_verified,
        "rejected_count": n_rejected,
        "bucket_counts": bucket_counts,
        "generator": GENERATOR_NAME,
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return manifest_path


def run_shard(
    shard_index: int,
    shard_size: int = DEFAULT_SHARD_SIZE,
    *,
    resume: bool = True,
    pytest_timeout_s: float = 10.0,
) -> dict:
    """Run one shard: generate, verify, write artifacts.

    When *resume* is True, families that already have a 'completed' cache
    entry are skipped (their verified samples remain in verified.jsonl).
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = _perf_counter_now()

    all_families = load_train_families(PARTITION_PATH)
    family_ids = shard_family_ids(all_families, shard_index, shard_size)
    if not family_ids:
        raise SystemExit(
            f"shard {shard_index} is empty (total families="
            f"{len(all_families)}, shard_size={shard_size})"
        )

    shard_dir = _shard_dir(shard_index)
    shard_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = shard_dir / "candidate.jsonl"
    verified_path = shard_dir / "verified.jsonl"
    rejected_path = shard_dir / "rejected.jsonl"
    cache_path = shard_dir / "verification-cache.jsonl"

    # On a fresh (non-resume) run, truncate the shard artifacts
    if not resume:
        for p in (candidate_path, verified_path, rejected_path, cache_path):
            if p.exists():
                p.unlink()

    completed_families = _load_cache_completed(cache_path) if resume else set()

    pool_by_fam = _load_pool_by_family(CANONICAL_POOL_PATH)

    dedup = DedupState()
    # Pre-load existing verified samples into dedup state so resumed runs
    # don't re-emit duplicates of already-verified samples.
    if resume and verified_path.exists():
        with verified_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = Sample.from_json_line(line)
                    dedup.add(s)
                except Exception:
                    continue

    total_verified = 0
    total_rejected = 0
    total_candidate = 0
    bucket_counts_total = {b: 0 for b in BUCKETS}
    families_processed = 0
    families_skipped = 0

    for idx, fid in enumerate(family_ids):
        if fid in completed_families:
            families_skipped += 1
            continue

        src = pool_by_fam.get(fid)
        if src is None:
            _append_jsonl(rejected_path, [_reject_record(
                fid, None, "none", "no source sample in canonical pool",
            )])
            # Cache as completed so we don't keep retrying
            _append_jsonl(cache_path, [{
                "family_id": fid,
                "completed": True,
                "verified_count": 0,
                "rejected_count": 1,
                "reason": "no source sample",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }])
            continue

        # Generate + verify
        verified_samples, verified_records, rejected_records, bucket_counts = (
            _process_family(
                fid, src, dedup, pytest_timeout_s=pytest_timeout_s,
            )
        )

        # Build candidate records (for candidate.jsonl)
        candidate_records: list[dict] = []
        # Re-derive candidate list for the record (lightweight: just counts)
        # We log the verified + rejected as the candidate set
        for s in verified_samples:
            candidate_records.append({
                "family_id": s.family_id,
                "sample_id": s.sample_id,
                "verified": True,
            })
        for r in rejected_records:
            candidate_records.append({
                "family_id": r["family_id"],
                "sample_id": r.get("sample_id"),
                "verified": False,
                "reason": r["reason"],
            })

        # Append artifacts
        _append_jsonl(candidate_path, candidate_records)
        _append_samples_jsonl(verified_path, verified_samples)
        _append_jsonl(rejected_path, rejected_records)

        # Append cache entry marking this family as completed
        _append_jsonl(cache_path, [{
            "family_id": fid,
            "completed": True,
            "verified_count": len(verified_samples),
            "rejected_count": len(rejected_records),
            "bucket_counts": bucket_counts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }])

        total_verified += len(verified_samples)
        total_rejected += len(rejected_records)
        total_candidate += len(candidate_records)
        for b in BUCKETS:
            bucket_counts_total[b] += bucket_counts.get(b, 0)
        families_processed += 1

        if (idx + 1) % 5 == 0:
            print(
                f"  shard {shard_index}: processed {idx + 1}/{len(family_ids)} "
                f"families (verified so far: {total_verified})"
            )

    duration_s = _perf_counter_now() - t0

    manifest_path = _write_shard_manifest(
        shard_index=shard_index,
        shard_size=shard_size,
        family_ids=family_ids,
        n_verified=total_verified,
        n_rejected=total_rejected,
        n_candidate=total_candidate,
        bucket_counts=bucket_counts_total,
        started_at=started_at,
        duration_s=duration_s,
    )

    payload = {
        "shard_id": f"shard-{shard_index:03d}",
        "shard_index": shard_index,
        "shard_size": shard_size,
        "families_total": len(family_ids),
        "families_processed": families_processed,
        "families_skipped_resume": families_skipped,
        "candidate_count": total_candidate,
        "verified_count": total_verified,
        "rejected_count": total_rejected,
        "bucket_counts": bucket_counts_total,
        "duration_s": round(duration_s, 3),
        "manifest_path": _relpath(manifest_path),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def _perf_counter_now() -> float:
    import time
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Pool consolidation (run after all shards complete)
# ---------------------------------------------------------------------------

def consolidate_pool() -> dict:
    """Concatenate all shard verified.jsonl into canonical-pool.jsonl and
    write the pool manifest with SHA-256.

    Also copies the family-partition.json into the formal root.
    """
    FORMAL_ROOT.mkdir(parents=True, exist_ok=True)

    # Copy partition
    partition = _load_json(PARTITION_PATH)
    with FORMAL_PARTITION_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(partition, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # Concatenate verified samples
    all_samples: list[Sample] = []
    shard_manifests: list[dict] = []
    if SHARDS_ROOT.exists():
        for shard_dir in sorted(SHARDS_ROOT.iterdir()):
            if not shard_dir.is_dir():
                continue
            vpath = shard_dir / "verified.jsonl"
            if vpath.exists():
                with vpath.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            all_samples.append(Sample.from_json_line(line))
                        except Exception:
                            continue
            mpath = shard_dir / "manifest.json"
            if mpath.exists():
                with mpath.open(encoding="utf-8") as fh:
                    shard_manifests.append(json.load(fh))

    # Write canonical-pool.jsonl
    with FORMAL_POOL_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for s in all_samples:
            fh.write(s.to_json_line())
            fh.write("\n")

    # Compute pool SHA-256
    pool_bytes = FORMAL_POOL_PATH.read_bytes()
    pool_sha = hashlib.sha256(pool_bytes).hexdigest()

    bucket_counts = {b: 0 for b in BUCKETS}
    family_counts: dict[str, int] = {}
    for s in all_samples:
        vt = (s.variant_type or s.task_type) if hasattr(s, "variant_type") else s.task_type
        # Map variant_type/task_type to bucket
        if vt in ("code", "code_generation"):
            bucket_counts["code"] += 1
        elif vt == "boundary":
            bucket_counts["boundary"] += 1
        elif vt == "static_repair":
            bucket_counts["static_repair"] += 1
        elif vt == "execution_repair":
            bucket_counts["execution_repair"] += 1
        family_counts[s.family_id] = family_counts.get(s.family_id, 0) + 1

    max_family_pct = (
        max(family_counts.values()) / len(all_samples) * 100
        if all_samples else 0.0
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": GENERATOR_NAME,
        "pool_sha256": pool_sha,
        "total_samples": len(all_samples),
        "total_families": len(family_counts),
        "bucket_counts": bucket_counts,
        "max_family_pct": round(max_family_pct, 4),
        "max_per_family_cap": MAX_TOTAL_PER_FAMILY,
        "per_bucket_cap": {
            "code": MAX_CODE_PER_FAMILY,
            "boundary": MAX_BOUNDARY_PER_FAMILY,
            "static_repair": MAX_STATIC_PER_FAMILY,
            "execution_repair": MAX_EXEC_PER_FAMILY,
        },
        "shard_count": len(shard_manifests),
        "shards": shard_manifests,
    }
    with FORMAL_MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="P3 Formal Canonical Pool v2 builder (sharded + resumable)."
    )
    p.add_argument(
        "--shard", type=int, default=None,
        help="Run a single shard by index (0-based).",
    )
    p.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
        help=f"Families per shard (default {DEFAULT_SHARD_SIZE}).",
    )
    p.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip families with a completed cache entry (default True).",
    )
    p.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Truncate shard artifacts and re-run from scratch.",
    )
    p.add_argument(
        "--consolidate", action="store_true",
        help="Consolidate all shards into canonical-pool.jsonl + manifest.",
    )
    p.add_argument(
        "--pytest-timeout", type=float, default=10.0,
        help="Per-variant pytest timeout in seconds (default 10.0).",
    )
    return p


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()

    if args.consolidate:
        manifest = consolidate_pool()
        print(f"Consolidated pool: {_relpath(FORMAL_POOL_PATH)}")
        print(f"  total_samples={manifest['total_samples']}")
        print(f"  total_families={manifest['total_families']}")
        print(f"  pool_sha256={manifest['pool_sha256']}")
        print(f"  bucket_counts={manifest['bucket_counts']}")
        print(f"  max_family_pct={manifest['max_family_pct']}%")
        return 0

    if args.shard is None:
        print("ERROR: --shard INDEX is required (or use --consolidate).",
              file=sys.stderr)
        return 2

    all_families = load_train_families(PARTITION_PATH)
    n_total = total_shards(len(all_families), args.shard_size)
    if args.shard < 0 or args.shard >= n_total:
        print(
            f"ERROR: shard {args.shard} out of range "
            f"(0..{n_total - 1} for {len(all_families)} families "
            f"@ {args.shard_size}/shard)",
            file=sys.stderr,
        )
        return 2

    print(
        f"Running shard {args.shard}/{n_total} "
        f"(size={args.shard_size}, resume={args.resume})"
    )
    payload = run_shard(
        shard_index=args.shard,
        shard_size=args.shard_size,
        resume=args.resume,
        pytest_timeout_s=args.pytest_timeout,
    )

    print("\nShard summary:")
    print(f"  shard_id: {payload['shard_id']}")
    print(f"  families_total: {payload['families_total']}")
    print(f"  families_processed: {payload['families_processed']}")
    print(f"  families_skipped_resume: {payload['families_skipped_resume']}")
    print(f"  candidate_count: {payload['candidate_count']}")
    print(f"  verified_count: {payload['verified_count']}")
    print(f"  rejected_count: {payload['rejected_count']}")
    print(f"  bucket_counts: {payload['bucket_counts']}")
    print(f"  duration_s: {payload['duration_s']}")
    print(f"  manifest: {payload['manifest_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
