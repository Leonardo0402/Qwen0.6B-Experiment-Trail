"""src/sample_pool.py -- Canonical verified sample pool.

Defines:
  - SamplePool: in-memory container for Sample objects with dedup, family
    filtering, variant_type normalisation, and per-family cap.

The pool is the INPUT to downstream P3 curriculum builders (Tasks 11/12).
It is NOT immutable -- it can be rebuilt from sources at any time.

Per .superpowers/sdd/task-10-brief.md:
  - variant_type is normalised from task_type + skill_tags + sample_id
  - bug_type is extracted from sample_id via ``.*_(sr|er)_(.+)$``
  - per-family cap default=7 (drops excess samples by sample_id ascending)
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, Optional

from src.schemas import Sample


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_VARIANT_TYPES: frozenset[str] = frozenset(
    {"code", "boundary", "static_repair", "execution_repair"}
)

# Regex extracting bug_type from sample_id: group 2 captures the bug_type.
# Examples:
#   mbpp_693_sr_return_value_error -> ("sr", "return_value_error")
#   mbpp_955_er_off_by_one          -> ("er", "off_by_one")
#   mbpp_917                        -> no match
_BUG_TYPE_RE: re.Pattern[str] = re.compile(r".*_(sr|er)_(.+)$")


# ---------------------------------------------------------------------------
# SamplePool
# ---------------------------------------------------------------------------

class SamplePool:
    """Canonical verified sample pool with dedup, index, and per-family cap."""

    def __init__(self) -> None:
        self._samples: list[Sample] = []
        # sample_id -> list of positions (kept for diagnostics; dedup leaves 1)
        self._index: dict[str, list[int]] = {}
        # family_id -> list of positions
        self._family_index: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, sample: Sample) -> bool:
        """Add a sample. Returns True if added, False if duplicate sample_id."""
        sid = sample.sample_id
        if sid in self._index:
            # Duplicate: do not add, but record would-be position for diagnostics
            return False
        pos = len(self._samples)
        self._samples.append(sample)
        self._index.setdefault(sid, []).append(pos)
        self._family_index.setdefault(sample.family_id, []).append(pos)
        return True

    def dedup(self) -> int:
        """Remove duplicates by sample_id (keep first). Returns count removed.

        Because :meth:`add` already rejects duplicates, this method is
        effectively a no-op when samples are added via ``add``. It is
        provided for the case where the internal list is rebuilt (e.g.
        after :meth:`from_jsonl` loads samples bypassing ``add``).
        """
        seen: set[str] = set()
        kept: list[Sample] = []
        removed = 0
        for s in self._samples:
            if s.sample_id in seen:
                removed += 1
                continue
            seen.add(s.sample_id)
            kept.append(s)
        if removed:
            self._samples = kept
            self._reindex()
        return removed

    def normalize_variant_type(self) -> int:
        """Set variant_type + bug_type on all samples. Returns count normalised.

        Per task-10-brief.md (binding):
          - task_type == "code_generation" AND ("boundary" in skill_tags
            OR sample_id ends with "_boundary") -> "boundary"
          - task_type == "code_generation" AND NOT boundary -> "code"
          - task_type == "static_repair"    -> "static_repair"
          - task_type == "execution_repair" -> "execution_repair"
        bug_type extracted via ``.*_(sr|er)_(.+)$`` -> group(2); else None.
        """
        normalised = 0
        new_samples: list[Sample] = []
        for s in self._samples:
            new_vt = self._compute_variant_type(s)
            new_bt = self._extract_bug_type(s.sample_id)
            if s.variant_type != new_vt or s.bug_type != new_bt:
                new_samples.append(
                    s.model_copy(update={"variant_type": new_vt, "bug_type": new_bt})
                )
                normalised += 1
            else:
                new_samples.append(s)
        self._samples = new_samples
        # No need to rebuild _index / _family_index: positions unchanged.
        return normalised

    def apply_family_cap(self, cap: int = 7) -> int:
        """Cap samples per family. Returns count dropped.

        If a family has > ``cap`` samples: keep first ``cap`` by sample_id
        ascending, drop the rest.
        """
        if cap < 1:
            raise ValueError(f"cap must be >= 1, got {cap}")
        # Group positions by family
        by_family: dict[str, list[int]] = defaultdict(list)
        for pos, s in enumerate(self._samples):
            by_family[s.family_id].append(pos)

        drop_positions: set[int] = set()
        dropped_total = 0
        for fid, positions in by_family.items():
            if len(positions) <= cap:
                continue
            # Sort the samples in this family by sample_id ascending, keep
            # the first `cap`, drop the rest.
            family_samples = sorted(
                positions, key=lambda p: self._samples[p].sample_id
            )
            to_drop = family_samples[cap:]
            drop_positions.update(to_drop)
            dropped_total += len(to_drop)

        if not dropped_total:
            return 0

        # Rebuild samples list, skipping dropped positions
        new_samples = [
            s for pos, s in enumerate(self._samples) if pos not in drop_positions
        ]
        self._samples = new_samples
        self._reindex()
        return dropped_total

    def filter_families(self, family_ids: set[str]) -> int:
        """Keep only samples whose family_id is in the set. Returns count kept."""
        allowed = set(family_ids)
        new_samples = [s for s in self._samples if s.family_id in allowed]
        kept = len(new_samples)
        if kept != len(self._samples):
            self._samples = new_samples
            self._reindex()
        return kept

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_jsonl(self, path: Path) -> None:
        """Write samples sorted by sample_id ascending (one Sample per line)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self._samples, key=lambda s: s.sample_id)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for s in ordered:
                fh.write(s.to_json_line())
                fh.write("\n")

    @classmethod
    def from_jsonl(cls, path: Path) -> "SamplePool":
        """Load from JSONL file."""
        pool = cls()
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sample = Sample.from_json_line(line)
                # Use direct append + index update to bypass dedup check
                # (canonical pool file is already deduplicated).
                pos = len(pool._samples)
                pool._samples.append(sample)
                pool._index.setdefault(sample.sample_id, []).append(pos)
                pool._family_index.setdefault(sample.family_id, []).append(pos)
        return pool

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return statistics dict for the manifest."""
        variant_counter: Counter[str] = Counter()
        bug_counter: Counter[Optional[str]] = Counter()
        family_counter: Counter[str] = Counter()
        for s in self._samples:
            variant_counter[s.variant_type or "unknown"] += 1
            bug_counter[s.bug_type] += 1
            family_counter[s.family_id] += 1

        family_counts = sorted(family_counter.values())
        n_fam = len(family_counts)
        if n_fam:
            min_per = family_counts[0]
            max_per = family_counts[-1]
            mean_per = sum(family_counts) / n_fam
            # Median (float): middle value if odd, average of two middle if even.
            if n_fam % 2 == 1:
                median_per = float(family_counts[n_fam // 2])
            else:
                median_per = (
                    family_counts[n_fam // 2 - 1] + family_counts[n_fam // 2]
                ) / 2.0
        else:
            min_per = 0
            max_per = 0
            mean_per = 0.0
            median_per = 0.0

        # bug_type_distribution: only non-None bug_types are reported by name;
        # None bug_type is reported under the key "none" for visibility.
        bug_distribution: dict[str, int] = {}
        for bt, cnt in bug_counter.items():
            key = bt if bt is not None else "none"
            bug_distribution[key] = bug_distribution.get(key, 0) + cnt

        return {
            "total_samples": len(self._samples),
            "family_count": n_fam,
            "variant_distribution": {
                v: variant_counter.get(v, 0)
                for v in ("code", "boundary", "static_repair", "execution_repair")
                if variant_counter.get(v, 0) > 0
            },
            "bug_type_distribution": bug_distribution,
            "family_distribution": {
                "min_samples_per_family": min_per,
                "max_samples_per_family": max_per,
                "mean_samples_per_family": mean_per,
                "median_samples_per_family": median_per,
            },
        }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reindex(self) -> None:
        """Rebuild _index and _family_index from _samples (post-mutation)."""
        self._index = {}
        self._family_index = {}
        for pos, s in enumerate(self._samples):
            self._index.setdefault(s.sample_id, []).append(pos)
            self._family_index.setdefault(s.family_id, []).append(pos)

    @staticmethod
    def _compute_variant_type(sample: Sample) -> str:
        """Compute variant_type for a sample per the brief's binding rules."""
        tt = sample.task_type
        sid = sample.sample_id
        tags = set(sample.skill_tags or [])
        if tt == "code_generation":
            if "boundary" in tags or sid.endswith("_boundary"):
                return "boundary"
            return "code"
        if tt == "static_repair":
            return "static_repair"
        if tt == "execution_repair":
            return "execution_repair"
        # Defensive fallback: unknown task_type -> "code"
        return "code"

    @staticmethod
    def _extract_bug_type(sample_id: str) -> Optional[str]:
        """Extract bug_type from sample_id using ``.*_(sr|er)_(.+)$``.

        Returns group(2) (the bug_type) on match, else None.
        """
        m = _BUG_TYPE_RE.match(sample_id)
        if m:
            return m.group(2)
        return None

    # ------------------------------------------------------------------
    # SHA helper (used by build_sample_pool orchestrator)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """Read entire file as bytes, normalize CRLF to LF, compute SHA256 hex digest.

        CRLF→LF normalization ensures cross-platform SHA consistency (Windows
        checkout with ``core.autocrlf=true`` produces CRLF; Linux CI produces LF).
        """
        h = hashlib.sha256()
        with path.open("rb") as fh:
            data = fh.read()
        # Normalize CRLF to LF for cross-platform consistency
        data = data.replace(b"\r\n", b"\n")
        h.update(data)
        return h.hexdigest()
