"""P3 3-Tier Checkpoint Evaluator (Amendment A11).

Implements the three-tier checkpoint evaluation strategy for P3 training:

- Tier 1 (every 25-50 steps): train_loss, eval_loss, lr, gpu_mem, nan/inf check
- Tier 2 (every 0.25 epoch): 60-90 family-stratified probe + Composite Score
- Tier 3 (every 1 epoch): full validation + Composite Score

Early stop: probe signal triggers, full validation confirms.
Best checkpoint: by full Validation Composite only (never frozen v3, never probe).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.metrics import (
    METRICS_SCHEMA_VERSION,
    EvalOutcome,
    normalize_baseline_key,
    pass_at_1,
    repair_success_rate,
)


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------


@dataclass
class CompositeScore:
    """Composite score components for a checkpoint, each in [0, 1].

    The four components correspond to the four variant_type buckets used in
    P3 training data. ``compute(weights)`` returns the weighted sum in [0, 1].
    """

    code_generation_pass_at_1: float
    boundary_pass_at_1: float  # variant_type="boundary", task_type="code_generation"
    static_repair_success: float
    execution_repair_success: float

    def compute(self, weights: dict[str, float]) -> float:
        """Weighted sum of the four components.

        Parameters
        ----------
        weights : dict
            Keys: ``code_generation_pass_at_1``, ``boundary_pass_at_1``,
            ``static_repair_success``, ``execution_repair_success``.

        Returns
        -------
        float
            Weighted sum in [0, 1].
        """
        return (
            self.code_generation_pass_at_1 * weights["code_generation_pass_at_1"]
            + self.boundary_pass_at_1 * weights["boundary_pass_at_1"]
            + self.static_repair_success * weights["static_repair_success"]
            + self.execution_repair_success * weights["execution_repair_success"]
        )


# ---------------------------------------------------------------------------
# Probe result (Tier 2)
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Result of a Tier 2 probe evaluation.

    The probe samples come from the TRAINING set (not validation), stratified
    by ``variant_type``. All four composite components can be non-zero.
    """

    step: int
    epoch: float
    probe_sample_ids: list[str]
    composite_score: CompositeScore
    composite_value: float
    metrics: dict  # raw metrics dict from src.metrics.summarize


# ---------------------------------------------------------------------------
# Full validation result (Tier 3)
# ---------------------------------------------------------------------------


@dataclass
class FullValidationResult:
    """Result of a Tier 3 full validation.

    The 90 P3 validation samples are code-only, so only
    ``composite_score.code_generation_pass_at_1`` will be non-zero on
    validation. The other three components will be 0.0.
    """

    step: int
    epoch: int
    composite_score: CompositeScore
    composite_value: float
    metrics: dict  # raw metrics dict from src.metrics.summarize
    hard_constraint_pass: bool
    hard_constraint_violations: list[str]


# ---------------------------------------------------------------------------
# Checkpoint evaluator
# ---------------------------------------------------------------------------


class CheckpointEvaluator:
    """3-Tier Checkpoint Evaluator per Amendment A11.

    Tier 1 (every 25-50 steps): train_loss, eval_loss, lr, gpu_mem, nan/inf
    Tier 2 (every 0.25 epoch): 60-90 family-stratified probe + Composite
    Tier 3 (every 1 epoch): full validation + Composite

    Early stop: probe signal triggers, full validation confirms.
    Best checkpoint: by full Validation Composite only (never frozen v3,
    never probe).
    """

    def __init__(self, config: dict, total_train_samples: int):
        """config is the loaded YAML as a dict."""
        self.config = config
        self.total_train_samples = total_train_samples
        # Best checkpoint tracking — only updated via update_best_checkpoint
        self._best_composite_value: float | None = None
        self._best_full_result: FullValidationResult | None = None
        # Cached config shortcuts
        self._tier1 = config["checkpoint_evaluator"]["tier1"]
        self._tier2 = config["checkpoint_evaluator"]["tier2"]
        self._tier3 = config["checkpoint_evaluator"]["tier3"]
        self._early_stopping = config["early_stopping"]
        self._composite_weights = {
            k: v
            for k, v in config["composite_score"].items()
            if not isinstance(v, dict)
        }
        self._hard_constraint = config["composite_score"]["hard_constraint"]

    # ------------------------------------------------------------------
    # Tier scheduling
    # ------------------------------------------------------------------

    def should_run_tier1(self, step: int) -> bool:
        """True if step is a multiple of tier1.interval_steps (and step > 0)."""
        interval = self._tier1["interval_steps"]
        return step > 0 and step % interval == 0

    def should_run_tier2(self, epoch: float) -> bool:
        """True if epoch crosses a 0.25 boundary (0.25, 0.5, 0.75, 1.0, ...).

        Uses ``epoch % fraction < epsilon`` to detect boundaries. The epoch=0
        case is excluded (no probe before training starts).
        """
        fraction = self._tier2["interval_epoch_fraction"]
        epsilon = 1e-9
        if epoch <= 0:
            return False
        # Detect if epoch is at a multiple of `fraction`
        return abs(epoch % fraction) < epsilon

    def should_run_tier3(self, epoch: int) -> bool:
        """True if epoch is an integer >= 1."""
        if isinstance(epoch, bool):
            return False
        if not isinstance(epoch, int):
            return False
        return epoch >= 1

    # ------------------------------------------------------------------
    # Probe sample selection (Tier 2)
    # ------------------------------------------------------------------

    def select_probe_samples(self, training_samples: list, seed: int = 42) -> list:
        """Select 60-90 family-stratified probe samples from TRAINING data.

        Stratifies by ``variant_type`` into 4 buckets, then samples
        ``probe_size // 4`` per bucket (remainder distributed to first buckets).
        Uses ``random.Random(seed)`` after sorting each bucket by sample_id
        ascending for reproducibility.

        Returns
        -------
        list
            Selected Sample objects (75 +/- 1 total).
        """
        probe_size = self._tier2["probe_size"]
        stratify_by = self._tier2["probe_stratify_by"]

        # Bucket samples by variant_type
        buckets: dict[str, list] = {}
        for sample in training_samples:
            key = getattr(sample, stratify_by, None)
            if key is None:
                continue
            buckets.setdefault(key, []).append(sample)

        # Sort each bucket by sample_id ascending (for determinism)
        for key in buckets:
            buckets[key].sort(key=lambda s: s.sample_id)

        # Distribute probe_size across buckets
        # base = probe_size // 4, remainder = probe_size % 4
        # First `remainder` buckets get base+1, rest get base
        bucket_keys = sorted(buckets.keys())
        n_buckets = len(bucket_keys)
        if n_buckets == 0:
            return []

        base = probe_size // 4
        remainder = probe_size % 4

        rng = random.Random(seed)
        selected: list = []

        for i, key in enumerate(bucket_keys):
            # Only consider the first 4 buckets (the 4 variant_types)
            if i >= 4:
                break
            target = base + (1 if i < remainder else 0)
            pool = buckets[key]
            target = min(target, len(pool))
            if target <= 0:
                continue
            # Sample indices from the sorted pool using the seeded RNG
            indices = sorted(rng.sample(range(len(pool)), target))
            selected.extend(pool[idx] for idx in indices)

        return selected

    # ------------------------------------------------------------------
    # Composite score computation
    # ------------------------------------------------------------------

    def compute_composite(
        self,
        metrics_by_variant: dict[str, list[EvalOutcome]],
        weights: dict[str, float],
    ) -> CompositeScore:
        """Build a CompositeScore from per-variant-type EvalOutcome lists.

        Parameters
        ----------
        metrics_by_variant : dict
            Keys: ``"code"``, ``"boundary"``, ``"static_repair"``,
            ``"execution_repair"``. Values: lists of EvalOutcome for that
            variant_type bucket.
        weights : dict
            Composite weights (accepted for API symmetry; the actual
            weighting is applied via ``CompositeScore.compute(weights)``).

        Returns
        -------
        CompositeScore
        """
        code_outcomes = metrics_by_variant.get("code", [])
        boundary_outcomes = metrics_by_variant.get("boundary", [])
        static_outcomes = metrics_by_variant.get("static_repair", [])
        exec_outcomes = metrics_by_variant.get("execution_repair", [])

        # Empty bucket -> rate 0.0 (pass_at_1 / repair_success_rate already
        # return 0.0 on empty lists, but we make the intent explicit).
        code_rate = pass_at_1(code_outcomes) if code_outcomes else 0.0
        boundary_rate = pass_at_1(boundary_outcomes) if boundary_outcomes else 0.0
        static_rate = repair_success_rate(static_outcomes) if static_outcomes else 0.0
        exec_rate = repair_success_rate(exec_outcomes) if exec_outcomes else 0.0

        return CompositeScore(
            code_generation_pass_at_1=code_rate,
            boundary_pass_at_1=boundary_rate,
            static_repair_success=static_rate,
            execution_repair_success=exec_rate,
        )

    # ------------------------------------------------------------------
    # Hard constraint check
    # ------------------------------------------------------------------

    def check_hard_constraint(
        self, metrics: dict, baseline: dict
    ) -> tuple[bool, list[str]]:
        """Check code_generation_drop_vs_p2_final_max_pct.

        Baseline keys are normalized via ``normalize_baseline_key`` so legacy
        ``codegen_pass1`` is compared against ``pass_at_1`` under a single
        canonical key (Issue #10 Fix 4).

        Parameters
        ----------
        metrics : dict
            Output of ``src.metrics.summarize()`` (must contain ``pass_at_1``).
        baseline : dict
            ``historical_held_out_metrics`` from
            ``reports/p3/p3-baseline-lock.json`` (Stage3-v3-Antiforget entry,
            may contain legacy ``codegen_pass1`` and optional
            ``schema_version``).

        Returns
        -------
        tuple
            (passed, violations) — ``passed`` is True when no hard violation
            (drop_pct) occurs. ``violations`` may still contain schema_version
            mismatch warnings (which do NOT cause FAIL by themselves).
        """
        max_drop_pct = self._hard_constraint[
            "code_generation_drop_vs_p2_final_max_pct"
        ]
        normalized_baseline = normalize_baseline_key(baseline)
        p2_codegen = normalized_baseline.get("pass_at_1", 0.0)
        current_codegen = metrics.get("pass_at_1", 0.0)
        drop_pct = (p2_codegen - current_codegen) * 100.0

        hard_violations: list[str] = []
        if drop_pct > max_drop_pct:
            hard_violations.append(
                f"code_generation_pass_at_1 dropped {drop_pct:.2f}pp vs P2 "
                f"final ({p2_codegen:.4f} -> {current_codegen:.4f}), max "
                f"allowed {max_drop_pct}pp"
            )

        # Schema version mismatch check (warning only — does NOT FAIL).
        # Issue #10 Fix 4: recorded to violations but never causes FAIL.
        warnings: list[str] = []
        baseline_schema = baseline.get("schema_version")
        if baseline_schema is not None and baseline_schema != METRICS_SCHEMA_VERSION:
            warnings.append(
                f"schema_version mismatch: baseline={baseline_schema} vs "
                f"metrics={METRICS_SCHEMA_VERSION} (warning only, not FAIL)"
            )

        all_violations = hard_violations + warnings
        return (len(hard_violations) == 0, all_violations)

    # ------------------------------------------------------------------
    # Best checkpoint selection
    # ------------------------------------------------------------------

    def update_best_checkpoint(self, full_result: FullValidationResult) -> bool:
        """Returns True if this full validation is the new best.

        Best = highest ``composite_value``. Never uses probe or frozen_v3
        (enforced by accepting only FullValidationResult).
        """
        if (
            self._best_composite_value is None
            or full_result.composite_value > self._best_composite_value
        ):
            self._best_composite_value = full_result.composite_value
            self._best_full_result = full_result
            return True
        return False

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def check_early_stop(
        self,
        probe_history: list[ProbeResult],
        full_history: list[FullValidationResult],
        nan_inf_detected: bool,
    ) -> tuple[bool, str]:
        """Returns (should_stop, reason).

        Triggers (checked in order):
        1. NaN/Inf detected -> immediate stop
        2. probe_patience consecutive probe drops -> trigger, but require
           full_validation_confirm
        3. max_epochs reached -> hard cap
        """
        # Trigger 1: NaN/Inf -> immediate stop
        if nan_inf_detected:
            return (True, "nan_or_inf_detected: immediate stop on divergence")

        probe_patience = self._early_stopping["probe_patience"]
        probe_min_delta = self._early_stopping["probe_min_delta"]
        full_validation_confirm = self._early_stopping["full_validation_confirm"]

        # Trigger 2: probe_patience consecutive probe drops
        # We interpret "probe_patience consecutive drops" as probe_patience
        # probes where each is strictly lower than the previous (by more
        # than probe_min_delta). This requires at least probe_patience probes.
        # When full_validation_confirm is True, a stop is only confirmed when
        # the full validation history ALSO shows a drop (latest < previous -
        # probe_min_delta). Issue #10 Fix 3.
        if len(probe_history) >= probe_patience and probe_patience >= 2:
            recent = probe_history[-probe_patience:]
            all_dropping = all(
                recent[i].composite_value
                < recent[i - 1].composite_value - probe_min_delta
                for i in range(1, len(recent))
            )
            if all_dropping:
                if full_validation_confirm:
                    # Require confirming full validation drops to stop
                    if len(full_history) < 2:
                        return (
                            False,
                            "probe drops detected, awaiting 2 full "
                            "validations to confirm",
                        )
                    if (
                        full_history[-1].composite_value
                        < full_history[-2].composite_value - probe_min_delta
                    ):
                        return (
                            True,
                            "probe_patience consecutive drops confirmed by "
                            "full validation drop",
                        )
                    else:
                        return (
                            False,
                            "probe drops detected but full validation did "
                            "not confirm drop (latest vs prev)",
                        )
                else:
                    return (
                        True,
                        "probe_patience consecutive drops",
                    )

        # Trigger 3: max_epochs reached
        max_epochs = self._early_stopping["max_epochs"]
        if full_history and full_history[-1].epoch >= max_epochs:
            return (
                True,
                f"max_epochs reached: {full_history[-1].epoch} >= {max_epochs}",
            )

        return (False, "no early stop trigger")


# ---------------------------------------------------------------------------
# BF16 runtime check
# ---------------------------------------------------------------------------


def check_bf16_support() -> tuple[bool, str]:
    """Check if BF16 is supported on the current GPU.

    Returns
    -------
    tuple
        (supported, message). On unsupported hardware, returns
        ``(False, "BF16 not supported, falling back to FP16")``.
        NEVER silently falls back — always returns a message.

    Notes
    -----
    Torch is imported lazily inside this function so that the module can be
    imported in environments without torch installed.
    """
    try:
        import torch  # noqa: WPS433 — lazy import is intentional
    except ImportError:
        return (
            False,
            "torch not installed; BF16 support cannot be verified, falling "
            "back to FP16",
        )

    if not torch.cuda.is_available():
        return (
            False,
            "BF16 not supported, falling back to FP16",
        )

    if torch.cuda.is_bf16_supported():
        return (True, "BF16 supported")

    return (False, "BF16 not supported, falling back to FP16")
