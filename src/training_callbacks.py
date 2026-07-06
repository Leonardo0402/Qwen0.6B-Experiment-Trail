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

Issue #14 Wave 3-D: in non-pilot mode with model + validation_samples
provided, the callback executes real Tier 2 probes (via ``on_step_end``
at 0.25 epoch boundaries) and Tier 3 full validation (via ``on_epoch_end``
at integer epoch boundaries), integrates ``EarlyStoppingManager`` for
persistent early-stop state, and records ``CheckpointEvidence`` on save.
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
        def on_step_end(self, args, state, control, **kwargs): pass
        def on_save(self, args, state, control, **kwargs): pass
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
    model : Any, optional
        The model to evaluate (required for real Tier 2/3 execution).
    tokenizer : Any, optional
        The tokenizer for chat template application.
    validation_samples : list, optional
        Validation v2 Sample list (180 samples) for Tier 2/3 evaluation.
    reports_dir : Path, optional
        Where to save probe/fullval reports and checkpoint evidence.
        Defaults to ``output_dir / "reports"``.
    config_path : Path, optional
        Path to the training config YAML (for checkpoint evidence SHA).
    train_file : Path, optional
        Path to the training data file (for checkpoint evidence SHA).
    validation_file : Path, optional
        Path to the validation data file (for checkpoint evidence SHA).
    baseline : dict, optional
        Historical baseline metrics for hard-constraint check.
    pytest_timeout_s : float
        Per-sample pytest timeout for Tier 2/3 evaluation (default 10.0).
    """

    def __init__(
        self,
        evaluator: Any,
        config: dict,
        output_dir: Path,
        pilot_mode: bool = False,
        *,
        model: Any = None,
        tokenizer: Any = None,
        validation_samples: Optional[list] = None,
        reports_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
        train_file: Optional[Path] = None,
        validation_file: Optional[Path] = None,
        baseline: Optional[dict] = None,
        pytest_timeout_s: float = 10.0,
    ):
        self.evaluator = evaluator
        self.config = config
        self.output_dir = Path(output_dir)
        self.pilot_mode = pilot_mode

        # Real-evaluation inputs (None in pilot/test mode)
        self.model = model
        self.tokenizer = tokenizer
        self.validation_samples = validation_samples
        self.reports_dir = (
            Path(reports_dir) if reports_dir else self.output_dir / "reports"
        )
        self.config_path = Path(config_path) if config_path else None
        self.train_file = Path(train_file) if train_file else None
        self.validation_file = Path(validation_file) if validation_file else None
        self.baseline = baseline or {}
        self.pytest_timeout_s = pytest_timeout_s

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

        # Track which 0.25 boundaries have been probed (avoid duplicates
        # between on_step_end and on_epoch_end).
        self._probed_boundaries: set[float] = set()

        # Real evaluation components (initialized lazily when model is
        # available and not in pilot mode).
        self._tier2_probe: Any = None
        self._tier3_validator: Any = None
        self._early_stopping_mgr: Any = None
        self._checkpoint_evidence: Any = None

        if (
            not pilot_mode
            and model is not None
            and validation_samples is not None
            and "checkpoint_evaluator" in config
        ):
            from src.p3_tier_evaluator import (
                CheckpointEvidence,
                EarlyStoppingManager,
                Tier2Probe,
                Tier3FullValidation,
            )

            self._tier2_probe = Tier2Probe(
                config, validation_samples, self.reports_dir,
            )
            self._tier3_validator = Tier3FullValidation(
                config, validation_samples, self.reports_dir,
                baseline=self.baseline,
            )
            self._early_stopping_mgr = EarlyStoppingManager(config)
            self._checkpoint_evidence = CheckpointEvidence(self.reports_dir)

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
    # Tier 2: on_step_end (detect 0.25 epoch boundaries)
    # ------------------------------------------------------------------

    def on_step_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Detect 0.25/0.50/0.75 epoch boundaries and trigger Tier 2 probe.

        In pilot mode or when no model/validation_samples are configured,
        this is a no-op (scheduling is logged via ``on_epoch_end``).
        """
        if self.pilot_mode:
            return
        if self._tier2_probe is None:
            return  # No real evaluation configured

        epoch = state.epoch
        tier2_cfg = self.config.get("checkpoint_evaluator", {}).get("tier2", {})
        fraction = tier2_cfg.get("interval_epoch_fraction", 0.25)

        # Find the nearest boundary at or below current epoch
        if epoch <= 0:
            return
        n_boundaries = int(epoch // fraction)
        current_boundary = round(n_boundaries * fraction, 6)

        # Skip if boundary is zero or already probed
        if current_boundary <= 0:
            return
        if current_boundary in self._probed_boundaries:
            return

        self._probed_boundaries.add(current_boundary)
        self._run_tier2_probe(state.global_step, current_boundary)

        # Check early stop after probe
        self._check_early_stop_and_signal(control)

        # NaN/Inf immediate stop
        if self.nan_inf_detected and self._early_stopping_mgr is not None:
            stop, reason = self._early_stopping_mgr.check_nan_inf(True)
            if stop:
                self.early_stop_triggered = True
                self.early_stop_reason = reason
                control.should_training_stop = True

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
        """Check Tier 2/3 scheduling at epoch boundaries.

        Tier 2 (every 0.25 epoch): probe evaluation + Composite Score.
        Tier 3 (every 1 epoch): full validation + Composite Score.
        """
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
            elif self._tier2_probe is not None and epoch_float not in self._probed_boundaries:
                # Real evaluation: run Tier 2 probe
                self._probed_boundaries.add(epoch_float)
                self._run_tier2_probe(state.global_step, epoch_float)
            else:
                # No model configured (test mode) or already probed:
                # log RUNNING for backward compatibility.
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
            elif self._tier3_validator is not None:
                # Real evaluation: run Tier 3 full validation
                self._run_tier3_validation(state.global_step, epoch_int)
            else:
                # No model configured (test mode): log RUNNING
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

        # Early stop check
        self._check_early_stop_and_signal(control)

    # ------------------------------------------------------------------
    # on_save: record checkpoint evidence
    # ------------------------------------------------------------------

    def on_save(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Record checkpoint evidence when a checkpoint is saved."""
        if self._checkpoint_evidence is None:
            return

        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_path.exists():
            return

        # Collect latest metrics for evidence
        metrics: dict[str, Any] = {}
        if self.tier2_log:
            last_tier2 = self.tier2_log[-1]
            if "composite_value" in last_tier2:
                metrics["tier2_composite"] = last_tier2["composite_value"]
        if self.tier3_log:
            last_tier3 = self.tier3_log[-1]
            if "composite_value" in last_tier3:
                metrics["tier3_composite"] = last_tier3["composite_value"]
        metrics["nan_inf_detected"] = self.nan_inf_detected

        from src.p3_tier_evaluator import PROBE_GENERATION_CONFIG

        try:
            self._checkpoint_evidence.record(
                checkpoint_path=checkpoint_path,
                config_path=self.config_path or Path(""),
                train_file=self.train_file or Path(""),
                validation_file=self.validation_file or Path(""),
                generation_config=PROBE_GENERATION_CONFIG,
                metrics=metrics,
            )
        except Exception as exc:
            print(
                f"[CheckpointEvaluator] WARNING: evidence record failed: {exc}",
                flush=True,
            )

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
        state_dict: dict[str, Any] = {
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

        # Include EarlyStoppingManager state if available
        if self._early_stopping_mgr is not None:
            state_dict["early_stopping_state"] = self._early_stopping_mgr.to_dict()

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

    # ------------------------------------------------------------------
    # Internal: run Tier 2 probe
    # ------------------------------------------------------------------

    def _run_tier2_probe(self, step: int, epoch: float) -> None:
        """Execute Tier 2 probe and record results."""
        from src.p3_tier_evaluator import CompositeCoverageError

        log_entry: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "status": "RUNNING",
            "probe_sha": self._tier2_probe.probe_sha,
        }

        try:
            result = self._tier2_probe.run(
                self.model, self.tokenizer, step, epoch,
                pytest_timeout_s=self.pytest_timeout_s,
            )
            log_entry["status"] = "COMPLETED"
            log_entry["composite_value"] = result.composite_value
            log_entry["probe_sample_count"] = len(result.probe_sample_ids)

            # Record in early stopping manager
            self._early_stopping_mgr.record_probe(result, result.metrics)

            print(
                f"[CheckpointEvaluator] Tier 2 COMPLETED @ epoch {epoch:.2f}: "
                f"composite={result.composite_value:.4f}",
                flush=True,
            )
        except CompositeCoverageError as exc:
            log_entry["status"] = "FAILED_BUCKET_MISSING"
            log_entry["error"] = str(exc)
            self._early_stopping_mgr.record_bucket_missing(exc)
            print(
                f"[CheckpointEvaluator] Tier 2 FAILED (bucket missing) @ "
                f"epoch {epoch:.2f}: {exc}",
                flush=True,
            )
        except Exception as exc:
            log_entry["status"] = "FAILED"
            log_entry["error"] = str(exc)
            print(
                f"[CheckpointEvaluator] Tier 2 FAILED @ epoch {epoch:.2f}: "
                f"{exc}",
                flush=True,
            )

        self.tier2_log.append(log_entry)

    # ------------------------------------------------------------------
    # Internal: run Tier 3 full validation
    # ------------------------------------------------------------------

    def _run_tier3_validation(self, step: int, epoch: int) -> None:
        """Execute Tier 3 full validation and record results."""
        from src.p3_tier_evaluator import CompositeCoverageError

        log_entry: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "status": "RUNNING",
        }

        try:
            result = self._tier3_validator.run(
                self.model, self.tokenizer, step, epoch,
                pytest_timeout_s=self.pytest_timeout_s,
            )
            log_entry["status"] = "COMPLETED"
            log_entry["composite_value"] = result.composite_value
            log_entry["hard_constraint_pass"] = result.hard_constraint_pass
            log_entry["hard_constraint_violations"] = result.hard_constraint_violations

            # Record in early stopping manager
            self._early_stopping_mgr.record_full(result, result.metrics)

            # Update best checkpoint (only by full validation composite)
            if (
                self.best_composite_value is None
                or result.composite_value > self.best_composite_value
            ):
                self.best_composite_value = result.composite_value
                log_entry["is_best"] = True

            print(
                f"[CheckpointEvaluator] Tier 3 COMPLETED @ epoch {epoch}: "
                f"composite={result.composite_value:.4f} "
                f"hard_pass={result.hard_constraint_pass}",
                flush=True,
            )
        except CompositeCoverageError as exc:
            log_entry["status"] = "FAILED_BUCKET_MISSING"
            log_entry["error"] = str(exc)
            self._early_stopping_mgr.record_bucket_missing(exc)
            print(
                f"[CheckpointEvaluator] Tier 3 FAILED (bucket missing) @ "
                f"epoch {epoch}: {exc}",
                flush=True,
            )
        except Exception as exc:
            log_entry["status"] = "FAILED"
            log_entry["error"] = str(exc)
            print(
                f"[CheckpointEvaluator] Tier 3 FAILED @ epoch {epoch}: {exc}",
                flush=True,
            )

        self.tier3_log.append(log_entry)

    # ------------------------------------------------------------------
    # Internal: check early stop and signal control
    # ------------------------------------------------------------------

    def _check_early_stop_and_signal(self, control: TrainerControl) -> None:
        """Check early stop conditions and set control.should_training_stop."""
        if self._early_stopping_mgr is not None:
            # Real mode: use EarlyStoppingManager
            if self.nan_inf_detected:
                stop, reason = self._early_stopping_mgr.check_nan_inf(True)
            else:
                stop, reason = self._early_stopping_mgr.should_stop()
        else:
            # Pilot/test mode: use evaluator.check_early_stop (legacy)
            stop, reason = self.evaluator.check_early_stop(
                probe_history=[],
                full_history=[],
                nan_inf_detected=self.nan_inf_detected,
            )

        if stop:
            self.early_stop_triggered = True
            self.early_stop_reason = reason
            print(
                f"[CheckpointEvaluator] EARLY STOP: {reason}",
                flush=True,
            )
            control.should_training_stop = True
