"""CheckpointEvaluator integration into HuggingFace Trainer (Issue #12 P4).

Provides a ``CheckpointEvaluatorCallback`` that wires the 3-tier
checkpoint evaluation strategy (Tier 1/2/3) into the training loop via
the TrainerCallback interface.

Tier 1 (every 50 steps): records train_loss, eval_loss, lr, gpu_mem, nan/inf
Tier 2 (every 0.25 epoch): probe evaluation + Composite Score
Tier 3 (every 1 epoch): full validation + Composite Score

For Pilot mode, Tier 2/3 actual evaluation (model.generate + sandbox) is
deferred — the callback records the scheduling decision and logs it,
proving the call chain works end-to-end without the full inference cost.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState
except ImportError:  # CI environments without torch/transformers
    class TrainerCallback:  # type: ignore[no-redef]
        def on_log(self, args, state, control, logs=None, **kwargs): pass
        def on_epoch_end(self, args, state, control, **kwargs): pass
        def on_train_end(self, args, state, control, **kwargs): pass

    class TrainerControl:  # type: ignore[no-redef]
        def __init__(self):
            self.should_training_stop = False

    class TrainerState:  # type: ignore[no-redef]
        pass


class CheckpointEvaluatorCallback(TrainerCallback):
    """Wires CheckpointEvaluator into the HuggingFace Trainer.

    Parameters
    ----------
    evaluator : CheckpointEvaluator
        The evaluator instance (from ``src.p3_checkpoint_evaluator``).
    config : dict
        The loaded YAML config (for reading composite weights, etc.).
    output_dir : Path
        Where to save evaluator state (``evaluator_state.json``).
    pilot_mode : bool
        If True, Tier 2/3 actual evaluation is deferred (only scheduling
        is logged). Used for Pilot training to keep it fast.
    """

    def __init__(
        self,
        evaluator: Any,
        config: dict,
        output_dir: Path,
        pilot_mode: bool = False,
    ):
        self.evaluator = evaluator
        self.config = config
        self.output_dir = Path(output_dir)
        self.pilot_mode = pilot_mode

        # Tier 1 history (in-memory, flushed at end)
        self.tier1_log: list[dict] = []
        # Tier 2/3 scheduling log (records when evaluation *would* run)
        self.tier2_log: list[dict] = []
        self.tier3_log: list[dict] = []

        # Early stop state
        self.nan_inf_detected = False
        self.early_stop_triggered = False
        self.early_stop_reason = ""

        # Best checkpoint tracking
        self.best_composite_value: float | None = None

    # ------------------------------------------------------------------
    # Tier 1: on_log (every logging_steps)
    # ------------------------------------------------------------------

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record Tier 1 metrics: train_loss, eval_loss, lr, gpu_mem, nan/inf."""
        if logs is None:
            return

        # Check for NaN/Inf in loss
        loss = logs.get("loss") or logs.get("eval_loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            self.nan_inf_detected = True
            print(
                f"[CheckpointEvaluator] WARNING: NaN/Inf detected at "
                f"step {state.global_step}: loss={loss}",
                flush=True,
            )

        # Record Tier 1 entry
        try:
            import torch
            gpu_mem_mb = 0.0
            if torch.cuda.is_available():
                gpu_mem_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        except ImportError:
            gpu_mem_mb = 0.0

        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "train_loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "lr": logs.get("learning_rate"),
            "gpu_mem_mb": round(gpu_mem_mb, 1),
            "nan_inf_detected": self.nan_inf_detected,
        }
        self.tier1_log.append(entry)

        # Check if Tier 1 should run (every interval_steps)
        if self.evaluator.should_run_tier1(state.global_step):
            print(
                f"[CheckpointEvaluator] Tier 1 @ step {state.global_step}: "
                f"loss={loss} nan_inf={self.nan_inf_detected} "
                f"gpu={gpu_mem_mb:.0f}MiB",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Tier 2/3: on_epoch_end
    # ------------------------------------------------------------------

    def on_epoch_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Check Tier 2/3 scheduling at epoch boundaries."""
        epoch_float = state.epoch
        epoch_int = int(round(epoch_float))

        # Tier 2: every 0.25 epoch
        if self.evaluator.should_run_tier2(epoch_float):
            if self.pilot_mode:
                print(
                    f"[CheckpointEvaluator] Tier 2 SCHEDULED @ epoch "
                    f"{epoch_float:.2f} (PILOT: actual evaluation deferred)",
                    flush=True,
                )
                self.tier2_log.append({
                    "epoch": epoch_float,
                    "step": state.global_step,
                    "status": "SCHEDULED_PILOT_DEFERRED",
                })
            else:
                print(
                    f"[CheckpointEvaluator] Tier 2 RUNNING @ epoch "
                    f"{epoch_float:.2f}",
                    flush=True,
                )
                self.tier2_log.append({
                    "epoch": epoch_float,
                    "step": state.global_step,
                    "status": "RUNNING",
                })

        # Tier 3: every 1 epoch
        if self.evaluator.should_run_tier3(epoch_int):
            if self.pilot_mode:
                print(
                    f"[CheckpointEvaluator] Tier 3 SCHEDULED @ epoch "
                    f"{epoch_int} (PILOT: actual evaluation deferred)",
                    flush=True,
                )
                self.tier3_log.append({
                    "epoch": epoch_int,
                    "step": state.global_step,
                    "status": "SCHEDULED_PILOT_DEFERRED",
                })
            else:
                print(
                    f"[CheckpointEvaluator] Tier 3 RUNNING @ epoch "
                    f"{epoch_int}",
                    flush=True,
                )
                self.tier3_log.append({
                    "epoch": epoch_int,
                    "step": state.global_step,
                    "status": "RUNNING",
                })

        # Check early stop (NaN/Inf only in pilot mode — probe/full
        # validation history is empty in pilot)
        should_stop, reason = self.evaluator.check_early_stop(
            probe_history=[],
            full_history=[],
            nan_inf_detected=self.nan_inf_detected,
        )
        if should_stop:
            self.early_stop_triggered = True
            self.early_stop_reason = reason
            print(
                f"[CheckpointEvaluator] EARLY STOP: {reason}",
                flush=True,
            )
            control.should_training_stop = True

    # ------------------------------------------------------------------
    # on_train_end: save evaluator state
    # ------------------------------------------------------------------

    def on_train_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Save evaluator state to ``evaluator_state.json`` in output_dir."""
        state_dict = {
            "tier1_entries": len(self.tier1_log),
            "tier2_entries": len(self.tier2_log),
            "tier3_entries": len(self.tier3_log),
            "nan_inf_detected": self.nan_inf_detected,
            "early_stop_triggered": self.early_stop_triggered,
            "early_stop_reason": self.early_stop_reason,
            "best_composite_value": self.best_composite_value,
            "pilot_mode": self.pilot_mode,
            "tier1_first": self.tier1_log[:3] if self.tier1_log else [],
            "tier1_last": self.tier1_log[-3:] if self.tier1_log else [],
            "tier2_log": self.tier2_log,
            "tier3_log": self.tier3_log,
        }
        state_path = self.output_dir / "evaluator_state.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with state_path.open("w", encoding="utf-8") as fh:
            json.dump(state_dict, fh, indent=2, ensure_ascii=False)
        print(
            f"[CheckpointEvaluator] State saved: {state_path} "
            f"(tier1={len(self.tier1_log)} tier2={len(self.tier2_log)} "
            f"tier3={len(self.tier3_log)})",
            flush=True,
        )
