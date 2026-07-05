"""src/family_registry.py -- Canonical Family Registry for the P3 partition.

Defines:
  - FamilyEntry: per-family dataclass (source, usage tags, sample_ids).
  - FamilyRegistry: container with load/save + claim/disjoint helpers.

The registry is the single source of truth for family-level state that
Tasks 7/8/9 use to claim families for frozen_v3_candidate / frozen_v3 /
p3_train / p3_validation and assert pairwise disjointness.

Per P3 plan Global Constraints #8, #9, #15: families used in the P3
partition must be pairwise disjoint (not just 3-way empty), with the
only allowed overlap being P2 Train replay (whitelisted).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
GENERATOR_NAME: str = "build_family_registry.py"

# Tags that mark a family as part of the P2 partition (any of these
# counts toward ``total_p2_used``).
P2_USAGE_TAGS: frozenset[str] = frozenset(
    {"p2_train", "p2_validation", "p2_frozen_v2"}
)

# P3 future tags (NOT applied by this builder; Tasks 7-9 will claim them).
P3_USAGE_TAGS: frozenset[str] = frozenset(
    {"frozen_v3_candidate", "frozen_v3", "p3_train", "p3_validation",
     "p3_train_replay"}
)

# Subset of P3 tags that are mutually exclusive with ``quarantine``.
P3_EXCLUSIVE_TAGS: frozenset[str] = frozenset(
    {"frozen_v3", "p3_train", "p3_validation"}
)


# ---------------------------------------------------------------------------
# FamilyEntry
# ---------------------------------------------------------------------------

@dataclass
class FamilyEntry:
    """One family's registry record."""

    family_id: str
    source_task_id: str
    source_split: str
    usage: list[str] = field(default_factory=list)
    first_commit: str = "unknown"
    dataset_version: str = "mbpp-v1"
    sample_ids: list[str] = field(default_factory=list)

    def is_used(self) -> bool:
        """True iff the usage list is non-empty."""
        return len(self.usage) > 0

    def has_usage(self, tag: str) -> bool:
        """True iff *tag* is in this family's usage list."""
        return tag in self.usage

    def claim(self, tag: str) -> None:
        """Add *tag* to usage. Idempotent: re-claiming an existing tag is
        a no-op (the tag is not duplicated)."""
        if tag not in self.usage:
            self.usage.append(tag)

    def unclaim(self, tag: str) -> None:
        """Remove *tag* from usage. No-op if *tag* is not present."""
        if tag in self.usage:
            self.usage.remove(tag)

    def to_dict(self) -> dict:
        """Serialise to the per-family dict shape used in family-registry.json."""
        return {
            "source_task_id": self.source_task_id,
            "source_split": self.source_split,
            "usage": list(self.usage),
            "first_commit": self.first_commit,
            "dataset_version": self.dataset_version,
            "sample_ids": list(self.sample_ids),
        }

    @classmethod
    def from_dict(cls, family_id: str, data: dict) -> "FamilyEntry":
        """Deserialise from a per-family dict (the reverse of :meth:`to_dict`)."""
        return cls(
            family_id=family_id,
            source_task_id=data["source_task_id"],
            source_split=data["source_split"],
            usage=list(data.get("usage", [])),
            first_commit=data.get("first_commit", "unknown"),
            dataset_version=data.get("dataset_version", "mbpp-v1"),
            sample_ids=list(data.get("sample_ids", [])),
        )


# ---------------------------------------------------------------------------
# FamilyRegistry
# ---------------------------------------------------------------------------

@dataclass
class FamilyRegistry:
    """Container for all family entries plus derived count fields."""

    families: dict[str, FamilyEntry] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: Path | str) -> "FamilyRegistry":
        """Load a registry from ``data/family-registry.json``.

        Only the ``families`` dict is loaded into the in-memory object;
        the top-level count fields are derived and recomputed on save.
        """
        p = Path(path)
        with p.open(encoding="utf-8") as fh:
            data = json.load(fh)
        reg = cls()
        for fam_id, entry_data in data.get("families", {}).items():
            reg.families[fam_id] = FamilyEntry.from_dict(fam_id, entry_data)
        return reg

    def to_path(self, path: Path | str) -> None:
        """Write the registry to ``data/family-registry.json``.

        Pretty-printed with ``indent=2`` and ``sort_keys=True`` for
        stable diffs. Ends with a trailing newline. The four count
        fields are recomputed from ``self.families`` so the file always
        reflects the current in-memory state.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": GENERATOR_NAME,
            "schema_version": SCHEMA_VERSION,
            "total_families": len(self.families),
            "total_p2_used": self._count_p2_used(),
            "total_quarantined": self._count_quarantined(),
            "total_new_available": self._count_new_available(),
            "families": {
                fid: entry.to_dict()
                for fid, entry in self.families.items()
            },
        }
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, family_id: str) -> Optional[FamilyEntry]:
        """Return the entry for *family_id*, or ``None`` if absent."""
        return self.families.get(family_id)

    def is_used(self, family_id: str) -> bool:
        """True iff *family_id* exists and has a non-empty usage list.

        Raises ``KeyError`` if *family_id* is not in the registry.
        """
        entry = self.families.get(family_id)
        if entry is None:
            raise KeyError(family_id)
        return entry.is_used()

    def families_with_usage(self, tag: str) -> list[str]:
        """Return a sorted list of family_ids whose usage contains *tag*."""
        return sorted(
            fid for fid, e in self.families.items() if tag in e.usage
        )

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def claim(self, family_id: str, tag: str) -> None:
        """Claim *tag* for *family_id*.

        Raises ``KeyError`` if *family_id* is not in the registry.
        Claiming is idempotent (delegates to :meth:`FamilyEntry.claim`).
        """
        entry = self.families.get(family_id)
        if entry is None:
            raise KeyError(family_id)
        entry.claim(tag)

    def unclaim(self, family_id: str, tag: str) -> None:
        """Remove *tag* from *family_id*'s usage list.

        No-op if *tag* is not present in the family's usage. No-op if
        *family_id* is not in the registry (silent — useful for
        idempotent re-runs of cleanup paths).
        """
        entry = self.families.get(family_id)
        if entry is None:
            return
        entry.unclaim(tag)

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def assert_pairwise_disjoint(
        self,
        usages: list[str],
        whitelist: list[tuple[str, str]] | None = None,
    ) -> None:
        """Assert that the family sets for each pair of tags in *usages*
        are pairwise disjoint, EXCEPT for pairs listed in *whitelist*.

        The whitelist is a list of ``(tag_a, tag_b)`` tuples; the pair
        ``(A, B)`` is equivalent to ``(B, A)`` for the skip check.

        Raises ``AssertionError`` with a message listing the violating
        family_ids and the offending pair.
        """
        wl = whitelist or []
        wl_normalized: set[tuple[str, str]] = set()
        for a, b in wl:
            wl_normalized.add((a, b))
            wl_normalized.add((b, a))

        sets = {tag: set(self.families_with_usage(tag)) for tag in usages}
        for i, a in enumerate(usages):
            for b in usages[i + 1:]:
                if (a, b) in wl_normalized:
                    continue
                overlap = sets[a] & sets[b]
                if overlap:
                    raise AssertionError(
                        f"pairwise disjoint violation: tags {a!r} and "
                        f"{b!r} share {len(overlap)} family_ids: "
                        f"{sorted(overlap)}"
                    )

    # ------------------------------------------------------------------
    # Internal count helpers (also used by the builder)
    # ------------------------------------------------------------------

    def _count_p2_used(self) -> int:
        """Count of families whose usage contains ANY P2 tag."""
        return sum(
            1 for e in self.families.values()
            if any(t in P2_USAGE_TAGS for t in e.usage)
        )

    def _count_quarantined(self) -> int:
        """Count of families whose usage contains ``quarantine``."""
        return sum(
            1 for e in self.families.values() if "quarantine" in e.usage
        )

    def _count_new_available(self) -> int:
        """Count of families whose usage is exactly empty."""
        return sum(
            1 for e in self.families.values() if len(e.usage) == 0
        )
