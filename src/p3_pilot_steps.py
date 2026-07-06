"""src/p3_pilot_steps.py — Compliant pilot step calculation (Issue #14 P0.3).

Provides the canonical formula for computing the maximum optimizer steps
allowed in a compliant 0.25-epoch Pilot, per Issue #14 P0.3:

    steps_per_epoch = ceil(num_train_samples / effective_batch_size)
    pilot_max_steps = min(50, floor(0.25 * steps_per_epoch))

where:
    effective_batch_size = per_device_train_batch_size
                           × gradient_accumulation_steps
                           × world_size
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PilotStepCalc:
    """Result of pilot step calculation."""
    steps_per_epoch: int
    pilot_max_steps: int
    effective_batch_size: int
    num_train_samples: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    world_size: int

    def as_dict(self) -> dict:
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "pilot_max_steps": self.pilot_max_steps,
            "effective_batch_size": self.effective_batch_size,
            "num_train_samples": self.num_train_samples,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "world_size": self.world_size,
        }


def compute_pilot_max_steps(
    num_train_samples: int,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    world_size: int = 1,
    max_pilot_steps_cap: int = 50,
    epoch_fraction: float = 0.25,
) -> PilotStepCalc:
    """Compute compliant pilot max_steps per Issue #14 P0.3.

    Args:
        num_train_samples: Total training samples in the dataset.
        per_device_train_batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        world_size: Number of processes (data-parallel world size).
        max_pilot_steps_cap: Hard cap on pilot steps (default 50).
        epoch_fraction: Fraction of epoch allowed for pilot (default 0.25).

    Returns:
        PilotStepCalc with steps_per_epoch, pilot_max_steps, and inputs.

    Raises:
        ValueError: If any input is non-positive or batch params are invalid.
    """
    if num_train_samples <= 0:
        raise ValueError(f"num_train_samples must be > 0, got {num_train_samples}")
    if per_device_train_batch_size <= 0:
        raise ValueError(f"per_device_train_batch_size must be > 0, got {per_device_train_batch_size}")
    if gradient_accumulation_steps <= 0:
        raise ValueError(f"gradient_accumulation_steps must be > 0, got {gradient_accumulation_steps}")
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if max_pilot_steps_cap <= 0:
        raise ValueError(f"max_pilot_steps_cap must be > 0, got {max_pilot_steps_cap}")
    if not (0 < epoch_fraction <= 1.0):
        raise ValueError(f"epoch_fraction must be in (0, 1], got {epoch_fraction}")

    effective_batch_size = (
        per_device_train_batch_size
        * gradient_accumulation_steps
        * world_size
    )
    steps_per_epoch = math.ceil(num_train_samples / effective_batch_size)
    pilot_max_steps = min(max_pilot_steps_cap, int(math.floor(epoch_fraction * steps_per_epoch)))
    # Ensure at least 1 step for small datasets
    if pilot_max_steps < 1:
        pilot_max_steps = 1

    return PilotStepCalc(
        steps_per_epoch=steps_per_epoch,
        pilot_max_steps=pilot_max_steps,
        effective_batch_size=effective_batch_size,
        num_train_samples=num_train_samples,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
    )
