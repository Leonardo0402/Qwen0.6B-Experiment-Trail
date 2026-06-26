"""Curriculum design for the Qwen3-0.6B Code Recovery Lab.

Spec §9: difficulty levels, staged mix ratios, and stage-mix sampling.

Difficulty levels used in this phase (0..3):
  0 — Syntax / local errors
  1 — Single-function implementation
  2 — Boundary conditions & data structures
  3 — Execution-feedback repair

Three training stages with target difficulty-mix fractions (fractions per
stage sum to 1.0):
  "easy"     : {0: 0.70, 1: 0.30}
  "boundary" : {0: 0.20, 1: 0.30, 2: 0.50}
  "repair"   : {0: 0.10, 1: 0.20, 2: 0.25, 3: 0.45}

Older difficulty levels are NOT discarded between stages (replay).

Under-fill behaviour (build_stage_mix)
---------------------------------------
When the pool has fewer samples of a given difficulty than the target count,
all available samples of that difficulty are taken and the shortfall is
recorded.  The function never crashes and never duplicates samples to pad.
The shortfall for the most recent call is accessible via get_last_shortfalls().
"""
from __future__ import annotations

import random
from enum import Enum

from src.schemas import Sample


# ---------------------------------------------------------------------------
# Difficulty level descriptions
# ---------------------------------------------------------------------------

_LEVEL_DESCRIPTIONS: dict[int, str] = {
    0: "Syntax / local errors",
    1: "Single-function implementation",
    2: "Boundary conditions & data structures",
    3: "Execution-feedback repair",
}


def level_description(level: int) -> str:
    """Return a short human description for difficulty level 0..3.

    Raises ValueError for any out-of-range level.
    """
    if level not in _LEVEL_DESCRIPTIONS:
        raise ValueError(
            f"difficulty level must be 0..3, got {level!r}"
        )
    return _LEVEL_DESCRIPTIONS[level]


# ---------------------------------------------------------------------------
# Training stages and mix ratios
# ---------------------------------------------------------------------------

class Stage(str, Enum):
    easy = "easy"
    boundary = "boundary"
    repair = "repair"


# Per-stage target difficulty distribution fractions.
# Each stage's fractions must sum to ~1.0; validated at import time.
STAGE_MIX: dict[str, dict[int, float]] = {
    Stage.easy.value:     {0: 0.70, 1: 0.30},
    Stage.boundary.value: {0: 0.20, 1: 0.30, 2: 0.50},
    Stage.repair.value:   {0: 0.10, 1: 0.20, 2: 0.25, 3: 0.45},
}

_FRAC_TOL = 1e-6
for _stage_name, _fracs in STAGE_MIX.items():
    _total = sum(_fracs.values())
    if abs(_total - 1.0) > _FRAC_TOL:
        raise ValueError(
            f"STAGE_MIX[{_stage_name!r}] fractions sum to {_total}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# Shortfall tracking (module-level; updated on each build_stage_mix call)
# ---------------------------------------------------------------------------

_last_shortfalls: dict[int, int] = {}


def get_last_shortfalls() -> dict[int, int]:
    """Return shortfalls from the most recent build_stage_mix call.

    Keys are difficulty levels; values are the number of missing samples
    (target_count - available_count) for each under-filled difficulty.
    Returns an empty dict when no shortfall occurred.

    Thread-safety: ``_last_shortfalls`` is module-global mutable state.
    Concurrent build_stage_mix calls race on it; call from a single thread
    (or guard externally) if you rely on the shortfall report.
    """
    return dict(_last_shortfalls)


# ---------------------------------------------------------------------------
# Stage-mix sampling
# ---------------------------------------------------------------------------

def build_stage_mix(
    samples: list[Sample],
    stage: str,
    total: int,
    *,
    seed: int = 42,
) -> list[Sample]:
    """Sample from *samples* to approximate the difficulty mix for *stage*.

    Per-difficulty target counts are computed as::

        target_count = round(total * fraction)

    Sampling is WITHOUT replacement within each difficulty bucket.

    Approximate total
    -----------------
    *total* is a TARGET, not a guarantee.  Because each per-difficulty count
    is independently rounded (``round(total * fraction)``), the assembled
    length can differ from *total* by a small amount (e.g. boundary stage with
    total=3 yields 4; repair stage with total=10 yields 9).  When the pool is
    sufficient, the length equals the sum of the rounded per-difficulty
    targets.

    Under-fill behaviour
    --------------------
    If the pool has fewer samples of some difficulty than target_count, all
    available samples of that difficulty are taken.  The shortfall is recorded
    and accessible via :func:`get_last_shortfalls`.  The function never
    duplicates samples and never crashes.

    The returned list is deterministically shuffled given the same seed and
    pool order.  Reproducibility is guaranteed when seed and pool are
    identical across calls.

    Parameters
    ----------
    samples:
        Pool of candidate samples to draw from.
    stage:
        One of "easy", "boundary", "repair" (keys of STAGE_MIX).
    total:
        Approximate target size of the assembled list (see "Approximate
        total" above; per-level rounding may shift the exact count).
    seed:
        RNG seed for determinism.

    Returns
    -------
    list[Sample]
        Assembled list, shuffled deterministically.
    """
    global _last_shortfalls

    if stage not in STAGE_MIX:
        raise ValueError(
            f"Unknown stage {stage!r}. Valid stages: {list(STAGE_MIX)}"
        )

    mix = STAGE_MIX[stage]
    rng = random.Random(seed)

    # Group pool samples by difficulty.
    pool_by_diff: dict[int, list[Sample]] = {}
    for s in samples:
        pool_by_diff.setdefault(s.difficulty, []).append(s)

    shortfalls: dict[int, int] = {}
    selected: list[Sample] = []

    for diff, fraction in mix.items():
        target_count = round(total * fraction)
        available = pool_by_diff.get(diff, [])

        if len(available) >= target_count:
            # Normal case: sample without replacement.
            chosen = rng.sample(available, target_count)
        else:
            # Under-fill: take all available, record shortfall.
            # In this branch len(available) < target_count, so the shortfall
            # is always positive — no guard needed.
            shortfalls[diff] = target_count - len(available)
            chosen = list(available)   # copy; do not mutate pool
            rng.shuffle(chosen)         # still advance rng for determinism

        selected.extend(chosen)

    _last_shortfalls = shortfalls

    # Final deterministic shuffle of the assembled list.
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Mix report
# ---------------------------------------------------------------------------

def mix_report(samples: list[Sample]) -> dict[int, int]:
    """Return a count of samples per difficulty level.

    Only levels actually present in *samples* appear as keys
    (levels with zero samples are omitted).

    Parameters
    ----------
    samples:
        Any list of Sample objects.

    Returns
    -------
    dict[int, int]
        Mapping of difficulty -> count.
    """
    report: dict[int, int] = {}
    for s in samples:
        report[s.difficulty] = report.get(s.difficulty, 0) + 1
    return report
