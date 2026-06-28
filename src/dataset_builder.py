"""Dataset building utilities for the Qwen3-0.6B Code Recovery Lab.

Spec §11.3: deduplication, LEAK-PROOF family split, and ChatML output.

ANTI-LEAKAGE GUARANTEE
-----------------------
split_by_family partitions samples by family_id, never by individual sample.
Every sample that shares a family_id ends up in exactly ONE of
train / val / test / heldout.  The family-id sets of train, val, and test are
pairwise disjoint by construction: the algorithm sorts all active family_ids,
shuffles them with a fixed RNG seed, then slices the resulting list at
cumulative-fraction boundaries.  Because a list slice can never overlap with
another slice into the same list, the disjointness guarantee is algebraic and
does not depend on heuristics.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.schemas import Sample, to_chatml
from src import curriculum


# ---------------------------------------------------------------------------
# DatasetSplit dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DatasetSplit:
    """Container for the four sample partitions produced by split_by_family.

    Attributes
    ----------
    train, val, test, heldout:
        Lists of Sample objects assigned to each partition.
    """

    train: list[Sample]
    val: list[Sample]
    test: list[Sample]
    heldout: list[Sample]

    def family_ids(self, split: str) -> set[str]:
        """Return the set of family_ids in the named split.

        Parameters
        ----------
        split:
            One of "train", "val", "test", "heldout".

        Raises ValueError for unknown split names.
        """
        parts: dict[str, list[Sample]] = {
            "train":   self.train,
            "val":     self.val,
            "test":    self.test,
            "heldout": self.heldout,
        }
        if split not in parts:
            raise ValueError(
                f"Unknown split {split!r}. Valid splits: {list(parts)}"
            )
        return {s.family_id for s in parts[split]}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _content_hash(sample: Sample) -> str:
    """Compute a stable SHA-256 hash of a sample's semantically-identifying fields.

    Fields hashed: instruction, target_code, broken_code, execution_feedback,
    task_type.  sample_id is intentionally excluded so that two samples with
    identical content but different IDs are correctly identified as duplicates.

    Empty-string optional fields are normalised to None before hashing
    (``"" or None`` -> None) so that a sample carrying broken_code="" and an
    otherwise-identical sample carrying broken_code=None are treated as the
    same content.
    """
    key = json.dumps(
        {
            "instruction": sample.instruction,
            "target_code": sample.target_code,
            "broken_code": (sample.broken_code or None),
            "execution_feedback": (sample.execution_feedback or None),
            "task_type":   sample.task_type,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def dedup(samples: list[Sample]) -> list[Sample]:
    """Remove duplicate samples, keeping the first occurrence of each.

    Two samples are considered duplicates when they share the same content
    hash (instruction + target_code + broken_code + task_type).
    sample_id alone does NOT determine uniqueness.

    Stable insertion order is preserved; first-seen wins.

    Parameters
    ----------
    samples:
        Input list, potentially containing duplicates.

    Returns
    -------
    list[Sample]
        De-duplicated list in original order.
    """
    seen: set[str] = set()
    result: list[Sample] = []
    for s in samples:
        h = _content_hash(s)
        if h not in seen:
            seen.add(h)
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Family split
# ---------------------------------------------------------------------------

def split_by_family(
    samples: list[Sample],
    *,
    train: float = 0.70,
    val: float = 0.10,
    test: float = 0.20,
    heldout_family_ids: Optional[set[str]] = None,
    seed: int = 42,
) -> DatasetSplit:
    """Split samples into train / val / test / heldout by family_id.

    ANTI-LEAKAGE: every sample of a given family_id goes ENTIRELY into
    exactly one split.  The family-id sets of train, val, and test are
    pairwise disjoint by construction (see module docstring).

    Algorithm
    ---------
    1. Separate heldout families upfront.
    2. Collect unique active family_ids; sort for stability, then shuffle
       deterministically with *seed*.
    3. Partition the shuffled list by cumulative fraction::

           train_end = round(n * train)
           val_end   = round(n * (train + val))

       families[0 : train_end]    → train
       families[train_end : val_end] → val
       families[val_end :]         → test

    4. Assign each remaining sample to the split of its family.

    Small-pool caveat
    -----------------
    Because split boundaries are derived from ``round(n * fraction)``, a small
    active-family count can round a split's share down to zero.  With the
    default val=0.10, at least 10 active families are required to guarantee a
    non-empty val split (e.g. n=8 yields zero val families).  When a split ends
    up empty while there are enough families to have populated it, a
    ``warnings.warn`` is emitted naming that split.  The disjointness algorithm
    (non-overlapping slices) is unaffected.

    Parameters
    ----------
    samples:
        Full pool of samples.
    train, val, test:
        Target fractions by family count.  Must sum to ~1.0 (±1e-6).
    heldout_family_ids:
        If provided, families listed here are excluded from train/val/test
        entirely and returned as the heldout partition.
    seed:
        RNG seed for deterministic family shuffle.

    Returns
    -------
    DatasetSplit
        Container with train, val, test, heldout sample lists.

    Raises
    ------
    ValueError
        If train + val + test does not sum to ~1.0.
    """
    total_frac = train + val + test
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"train + val + test must sum to 1.0, got {total_frac:.10f}"
        )

    heldout_set: set[str] = heldout_family_ids if heldout_family_ids is not None else set()

    # Separate heldout samples first.
    heldout_samples = [s for s in samples if s.family_id in heldout_set]
    active_samples  = [s for s in samples if s.family_id not in heldout_set]

    # Collect unique active family_ids; sort then shuffle for determinism.
    family_ids: list[str] = sorted({s.family_id for s in active_samples})
    rng = random.Random(seed)
    rng.shuffle(family_ids)

    n = len(family_ids)
    train_end = round(n * train)
    val_end   = round(n * (train + val))

    # Non-overlapping slices → disjoint sets guaranteed.
    train_families: set[str] = set(family_ids[:train_end])
    val_families:   set[str] = set(family_ids[train_end:val_end])
    test_families:  set[str] = set(family_ids[val_end:])

    # Warn when a split that was *supposed* to receive families (positive
    # fraction) ends up empty purely because of small-n rounding, while there
    # are enough active families that it could have been populated. This does
    # NOT alter the disjoint slicing above; it only surfaces a quality risk.
    positive_splits = [
        ("train", train, train_families),
        ("val",   val,   val_families),
        ("test",  test,  test_families),
    ]
    num_positive = sum(1 for _, frac, _ in positive_splits if frac > 0.0)
    if n >= num_positive:
        for split_name, frac, fam_set in positive_splits:
            if frac > 0.0 and not fam_set:
                warnings.warn(
                    f"split_by_family: {split_name!r} split is EMPTY after "
                    f"rounding (active families n={n}, {split_name} fraction="
                    f"{frac}). Increase the pool or adjust fractions; "
                    f"val=0.10 needs >=10 active families for a non-empty val.",
                    stacklevel=2,
                )

    train_samples = [s for s in active_samples if s.family_id in train_families]
    val_samples   = [s for s in active_samples if s.family_id in val_families]
    test_samples  = [s for s in active_samples if s.family_id in test_families]

    return DatasetSplit(
        train=train_samples,
        val=val_samples,
        test=test_samples,
        heldout=heldout_samples,
    )


# ---------------------------------------------------------------------------
# ChatML conversion
# ---------------------------------------------------------------------------

def to_chatml_records(samples: list[Sample]) -> list[dict]:
    """Convert a list of Sample objects to ChatML-formatted dicts.

    Each dict has the structure produced by schemas.to_chatml:
    {"messages": [system_msg, user_msg, assistant_msg]}.

    Parameters
    ----------
    samples:
        Any list of Sample objects.

    Returns
    -------
    list[dict]
        One ChatML dict per sample, in the same order.
    """
    return [to_chatml(s) for s in samples]


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: "str | Path") -> None:
    """Write *records* to *path* as a JSONL file (one JSON object per line).

    Creates parent directories if they do not exist.

    Parameters
    ----------
    records:
        List of JSON-serialisable dicts.
    path:
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" forces LF line endings on all platforms so JSONL files stay
    # portable for Linux consumers (Windows would otherwise emit CRLF).
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def dataset_hash(records: list[dict]) -> str:
    """Compute a SHA-256 hash over *records*, order-independent.

    Records are serialised with sorted keys, then the serialised strings are
    sorted lexicographically before hashing.  This makes the hash stable
    regardless of the order in which records appear in the list.

    Parameters
    ----------
    records:
        List of dicts to hash.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    serialized = sorted(
        json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for r in records
    )
    combined = "\n".join(serialized)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Split writer
# ---------------------------------------------------------------------------

def write_split(
    split: DatasetSplit,
    out_dir: "str | Path",
    *,
    seed: int = 42,
) -> dict:
    """Write ChatML JSONL files for a DatasetSplit and return a manifest dict.

    Files written
    -------------
    - train.jsonl
    - validation.jsonl
    - test.jsonl
    - heldout.jsonl  (only when split.heldout is non-empty)

    Manifest keys
    -------------
    "train", "validation", "test", "heldout"
        Each a sub-dict with:
        - "sample_count"   : int
        - "difficulty_mix" : dict[int, int]  (from curriculum.mix_report)
        - "family_count"   : int
    "dataset_hash"
        SHA-256 hex digest of all train + val + test ChatML records combined.
    "seed"
        The seed parameter passed to this function.
    "created_at"
        ISO-8601 UTC timestamp of when this function ran.

    Parameters
    ----------
    split:
        DatasetSplit produced by split_by_family.
    out_dir:
        Directory to write files into (created if it does not exist).
    seed:
        Seed to record in the manifest (for provenance).

    Returns
    -------
    dict
        Manifest as described above.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map split attribute names to output file stem names.
    named_parts: list[tuple[str, list[Sample]]] = [
        ("train",      split.train),
        ("validation", split.val),
        ("test",       split.test),
    ]

    all_records: list[dict] = []
    manifest: dict = {}

    for split_name, samp_list in named_parts:
        records = to_chatml_records(samp_list)
        write_jsonl(records, out_dir / f"{split_name}.jsonl")
        all_records.extend(records)

        manifest[split_name] = {
            "sample_count":   len(samp_list),
            "difficulty_mix": curriculum.mix_report(samp_list),
            "family_count":   len({s.family_id for s in samp_list}),
        }

    # Write raw Sample JSONL for test split (needed for evaluation)
    # ChatML format strips away test code, so we need raw samples for evaluate_model.py
    if split.test:
        raw_test_records = [s.model_dump() for s in split.test]
        write_jsonl(raw_test_records, out_dir / "test_raw.jsonl")

    # Heldout: always in manifest; file written only if non-empty.
    manifest["heldout"] = {
        "sample_count":   len(split.heldout),
        "difficulty_mix": curriculum.mix_report(split.heldout),
        "family_count":   len({s.family_id for s in split.heldout}),
    }
    if split.heldout:
        heldout_records = to_chatml_records(split.heldout)
        write_jsonl(heldout_records, out_dir / "heldout.jsonl")

    manifest["dataset_hash"] = dataset_hash(all_records)
    manifest["seed"]         = seed
    manifest["created_at"]   = datetime.now(tz=timezone.utc).isoformat()

    return manifest
