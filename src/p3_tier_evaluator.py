"""P3 3-Tier Evaluator (Issue #14 Wave 3-D).

Real execution of Tier 2 (probe) and Tier 3 (full validation) evaluations.
Replaces the SCHEDULED_PILOT_DEFERRED stub with actual model.generate +
sandbox evaluation against Validation v2.

Probe samples are drawn from Validation v2 (NOT the training set), stratified
by ``variant_type`` into the four required buckets (code, boundary,
static_repair, execution_repair). Sample IDs and generation config are
SHA-locked for reproducibility.

Classes
-------
Tier2Probe
    60-90 family-stratified probe from Validation v2.
Tier3FullValidation
    All 180 Validation v2 samples.
EarlyStoppingManager
    Persistent early-stop state machine (pending → confirmed).
CheckpointEvidence
    Checkpoint provenance (weight/config/data SHA).

Functions
---------
select_probe_samples
    Family-stratified probe sampling from Validation v2.
extract_code_block
    Extract triple-backtick python code block from generated text.
build_eval_messages
    Build ChatML messages for a Sample.
evaluate_single_sample
    Generate + sandbox for one sample.
compute_probe_sha
    SHA256 of probe sample IDs + generation config.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.metrics import EvalOutcome, summarize
from src.p3_checkpoint_evaluator import (
    CompositeCoverageError,
    CompositeScore,
    FullValidationResult,
    ProbeResult,
)
from src.sandbox import check_code_safety, run_pytest
from src.schemas import Sample


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "你是一个严谨的 Python 代码助手。"
    "根据任务（及真实执行反馈）输出正确代码。"
    "除非用户要求解释，否则只输出完整代码。"
)

# Fixed, deterministic generation config for all Tier 2/3 evaluations.
# Issue #14 Wave 3-D: temperature=0, max_new_tokens=384, do_sample=False.
PROBE_GENERATION_CONFIG: dict[str, Any] = {
    "temperature": 0,
    "max_new_tokens": 384,
    "do_sample": False,
    "num_beams": 1,
}

# The four required variant_type buckets. All must be non-empty for
# compute_composite to succeed (CompositeCoverageError otherwise).
_REQUIRED_VARIANT_BUCKETS = (
    "code",
    "boundary",
    "static_repair",
    "execution_repair",
)


# ---------------------------------------------------------------------------
# Code extraction (mirrors scripts/evaluate_model.py)
# ---------------------------------------------------------------------------


def extract_code_block(text: str) -> Optional[str]:
    """Extract code from the first triple-backtick python block.

    Falls back to returning the stripped text if no block is found.
    """
    match = re.search(r"```python\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Message building (matches schemas.to_chatml + evaluate_model)
# ---------------------------------------------------------------------------


def build_eval_messages(sample: Sample) -> list[dict]:
    """Build ChatML messages for evaluation, matching training format.

    Returns ``[system, user]`` (no assistant — generation prompt is added
    by ``tokenizer.apply_chat_template``).
    """
    parts: list[str] = [sample.instruction]

    tt = sample.task_type
    if tt in ("static_repair", "execution_repair"):
        parts.append(f"```python\n{sample.broken_code}\n```")
    if tt == "execution_repair":
        parts.append(sample.execution_feedback or "")
    if tt == "code_generation":
        parts.append("请输出完整代码")
    else:
        parts.append("请输出修复后的完整代码")

    user_content = "\n\n".join(parts)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Probe sample selection from Validation v2
# ---------------------------------------------------------------------------


def select_probe_samples(
    validation_samples: list[Sample],
    probe_size: int = 75,
    seed: int = 42,
) -> list[Sample]:
    """Select 60-90 family-stratified probe samples from Validation v2.

    Stratifies by ``variant_type`` into 4 buckets (code, boundary,
    static_repair, execution_repair), then samples
    ``probe_size // 4`` per bucket (remainder distributed to first buckets).
    Uses ``random.Random(seed)`` after sorting each bucket by ``sample_id``
    ascending for reproducibility.

    Parameters
    ----------
    validation_samples : list[Sample]
        The full Validation v2 sample list (180 samples, 45 per bucket).
    probe_size : int
        Target probe size (60-90; default 75).
    seed : int
        RNG seed for reproducibility (default 42).

    Returns
    -------
    list[Sample]
        Selected Sample objects, stratified across all 4 variant types.
    """
    # Bucket by variant_type
    buckets: dict[str, list[Sample]] = {}
    for sample in validation_samples:
        key = sample.variant_type
        if key is None:
            continue
        buckets.setdefault(key, []).append(sample)

    # Sort each bucket by sample_id ascending (for determinism)
    for key in buckets:
        buckets[key].sort(key=lambda s: s.sample_id)

    bucket_keys = sorted(buckets.keys())
    n_buckets = len(bucket_keys)
    if n_buckets == 0:
        return []

    base = probe_size // 4
    remainder = probe_size % 4

    rng = random.Random(seed)
    selected: list[Sample] = []

    for i, key in enumerate(bucket_keys):
        if i >= 4:
            break
        target = base + (1 if i < remainder else 0)
        pool = buckets[key]
        target = min(target, len(pool))
        if target <= 0:
            continue
        indices = sorted(rng.sample(range(len(pool)), target))
        selected.extend(pool[idx] for idx in indices)

    return selected


def compute_probe_sha(
    sample_ids: list[str],
    generation_config: dict[str, Any],
) -> str:
    """SHA256 of sorted probe sample IDs + generation config.

    This locks the probe composition and generation parameters for
    reproducibility auditing. The same probe_size + seed + validation set
    always produces the same SHA.
    """
    payload = {
        "sample_ids": sorted(sample_ids),
        "generation_config": generation_config,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Single-sample evaluation (generate + sandbox)
# ---------------------------------------------------------------------------


def evaluate_single_sample(
    model: Any,
    tokenizer: Any,
    sample: Sample,
    *,
    max_new_tokens: int = 384,
    pytest_timeout_s: float = 10.0,
) -> tuple[EvalOutcome, dict]:
    """Evaluate a single sample: generate code, run sandbox tests.

    Returns ``(EvalOutcome, detail_dict)``. The detail dict captures
    per-sample signals for JSONL persistence.

    Mirrors ``scripts/evaluate_model._evaluate_one`` but lives in ``src/``
    so the training callback can call it without importing a script.
    """
    import torch  # lazy import for test compatibility

    detail: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "family_id": sample.family_id,
        "variant_type": sample.variant_type,
        "task_type": sample.task_type,
        "difficulty": sample.difficulty,
        "generated": "",
        "extracted_code": None,
        "format_ok": False,
        "syntax_ok": False,
        "public_tests_collected": 0,
        "public_passed": False,
        "hidden_tests_present": bool(sample.hidden_tests.strip()),
        "hidden_tests_collected": 0,
        "hidden_passed": False,
        "timed_out": False,
        "error": None,
    }

    repair_succeeded: Optional[bool] = None
    broke_other_tests: Optional[bool] = None
    is_repair = sample.task_type in ("static_repair", "execution_repair")

    try:
        # Build prompt
        messages = build_eval_messages(sample)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,  # ignored when do_sample=False
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
            )

        new_ids = output_ids[0][input_len:]
        completion = tokenizer.decode(new_ids, skip_special_tokens=True)
        detail["generated"] = completion

        del inputs, output_ids, new_ids

        # Extract code block
        extracted = extract_code_block(completion)
        detail["extracted_code"] = extracted
        detail["format_ok"] = extracted is not None

        if extracted is None:
            extracted = completion.strip()

        # Safety check (non-blocking)
        safety_warnings = check_code_safety(extracted)
        if safety_warnings:
            detail["safety_warnings"] = safety_warnings

        # Syntax check
        try:
            ast.parse(extracted)
            detail["syntax_ok"] = True
        except SyntaxError:
            detail["syntax_ok"] = False

        # Run pytest — public tests
        pub_result = run_pytest(
            extracted, sample.public_tests, timeout_s=pytest_timeout_s
        )
        detail["timed_out"] = pub_result.timed_out
        detail["public_tests_collected"] = pub_result.num_collected
        detail["public_passed"] = pub_result.passed

        # Run pytest — hidden tests
        if sample.hidden_tests.strip():
            hid_result = run_pytest(
                extracted, sample.hidden_tests, timeout_s=pytest_timeout_s
            )
            detail["hidden_passed"] = hid_result.passed
            detail["hidden_tests_collected"] = hid_result.num_collected
            detail["timed_out"] = detail["timed_out"] or hid_result.timed_out
        else:
            detail["hidden_passed"] = False
            detail["hidden_tests_collected"] = 0

        # Repair-specific metrics
        if is_repair:
            repair_succeeded = (
                detail["syntax_ok"]
                and detail["public_passed"]
                and detail["hidden_passed"]
            )
            if sample.target_code.strip():
                orig_pub = run_pytest(
                    sample.target_code, sample.public_tests,
                    timeout_s=pytest_timeout_s,
                )
                if orig_pub.passed and not detail["public_passed"]:
                    broke_other_tests = True
                else:
                    broke_other_tests = False

    except Exception as exc:
        detail["error"] = str(exc)

    outcome = EvalOutcome(
        task_type=sample.task_type,
        syntax_ok=detail["syntax_ok"],
        public_passed=detail["public_passed"],
        public_tests_collected=detail["public_tests_collected"],
        hidden_passed=detail["hidden_passed"],
        hidden_tests_present=detail["hidden_tests_present"],
        hidden_tests_collected=detail["hidden_tests_collected"],
        format_ok=detail["format_ok"],
        timed_out=detail["timed_out"],
        is_repair=is_repair,
        repair_succeeded=repair_succeeded,
        broke_other_tests=broke_other_tests,
    )

    return outcome, detail


# ---------------------------------------------------------------------------
# Tier 2: Probe evaluation
# ---------------------------------------------------------------------------


class Tier2Probe:
    """Manages Tier 2 probe evaluation against Validation v2.

    Selects a fixed, family-stratified probe (60-90 samples) from the
    Validation v2 set, runs generate + sandbox for each, and computes the
    five-component Composite Score.

    Parameters
    ----------
    config : dict
        The loaded YAML config (must contain ``checkpoint_evaluator.tier2``
        and ``composite_score`` blocks).
    validation_samples : list[Sample]
        Full Validation v2 sample list (180 samples).
    output_dir : Path
        Where to save per-sample JSONL and probe reports.
    """

    def __init__(
        self,
        config: dict,
        validation_samples: list[Sample],
        output_dir: Path,
    ):
        self.config = config
        self.validation_samples = validation_samples
        self.output_dir = Path(output_dir)

        tier2_cfg = config["checkpoint_evaluator"]["tier2"]
        self.probe_size: int = tier2_cfg["probe_size"]
        self.probe_seed: int = tier2_cfg.get("probe_seed", 42)

        # Composite weights (filter out nested dicts like hard_constraint)
        self._composite_weights: dict[str, float] = {
            k: v
            for k, v in config["composite_score"].items()
            if not isinstance(v, dict)
        }

        # Select probe samples (deterministic)
        self.probe_samples: list[Sample] = select_probe_samples(
            validation_samples, probe_size=self.probe_size, seed=self.probe_seed
        )
        self.probe_sample_ids: list[str] = [
            s.sample_id for s in self.probe_samples
        ]
        self.probe_sha: str = compute_probe_sha(
            self.probe_sample_ids, PROBE_GENERATION_CONFIG
        )

    def run(
        self,
        model: Any,
        tokenizer: Any,
        step: int,
        epoch: float,
        *,
        pytest_timeout_s: float = 10.0,
    ) -> ProbeResult:
        """Execute the probe: generate + sandbox for each probe sample.

        Returns a ``ProbeResult`` with the Composite Score.
        Raises ``CompositeCoverageError`` if a required bucket is missing.
        """
        import torch  # lazy import

        # Switch to eval mode for generation
        was_training = model.training
        model.eval()

        try:
            per_sample_details: list[dict] = []
            outcomes_by_variant: dict[str, list[EvalOutcome]] = {
                v: [] for v in _REQUIRED_VARIANT_BUCKETS
            }

            for sample in self.probe_samples:
                outcome, detail = evaluate_single_sample(
                    model, tokenizer, sample,
                    max_new_tokens=PROBE_GENERATION_CONFIG["max_new_tokens"],
                    pytest_timeout_s=pytest_timeout_s,
                )
                per_sample_details.append(detail)

                bucket = sample.variant_type
                if bucket in outcomes_by_variant:
                    outcomes_by_variant[bucket].append(outcome)

            # Compute composite (raises CompositeCoverageError if bucket missing)
            composite_score = self._compute_composite(outcomes_by_variant)
            composite_value = composite_score.compute(self._composite_weights)

            # Summarize metrics over all outcomes
            all_outcomes = []
            for v in _REQUIRED_VARIANT_BUCKETS:
                all_outcomes.extend(outcomes_by_variant[v])
            metrics = summarize(all_outcomes)

            result = ProbeResult(
                step=step,
                epoch=epoch,
                probe_sample_ids=list(self.probe_sample_ids),
                composite_score=composite_score,
                composite_value=composite_value,
                metrics=metrics,
            )

            # Persist per-sample JSONL + probe report
            self._save_results(result, per_sample_details)

            return result
        finally:
            if was_training:
                model.train()

    def _compute_composite(
        self, outcomes_by_variant: dict[str, list[EvalOutcome]]
    ) -> CompositeScore:
        """Build CompositeScore from per-variant outcomes.

        Raises ``CompositeCoverageError`` if any required bucket is empty.
        """
        missing_or_empty = [
            bucket
            for bucket in _REQUIRED_VARIANT_BUCKETS
            if not outcomes_by_variant.get(bucket)
        ]
        if missing_or_empty:
            raise CompositeCoverageError(
                "Tier2Probe hard fail: missing or empty variant buckets: "
                f"{missing_or_empty}. Required buckets = "
                f"{_REQUIRED_VARIANT_BUCKETS}. Readiness="
                f"{CompositeCoverageError.READINESS}."
            )

        from src.metrics import hidden_pass_rate, pass_at_1, repair_success_rate

        code_outcomes = outcomes_by_variant["code"]
        boundary_outcomes = outcomes_by_variant["boundary"]
        static_outcomes = outcomes_by_variant["static_repair"]
        exec_outcomes = outcomes_by_variant["execution_repair"]

        all_outcomes = (
            code_outcomes + boundary_outcomes
            + static_outcomes + exec_outcomes
        )

        return CompositeScore(
            code_generation_pass_at_1=pass_at_1(code_outcomes),
            boundary_pass_at_1=pass_at_1(boundary_outcomes),
            static_repair_success=repair_success_rate(static_outcomes),
            execution_repair_success=repair_success_rate(exec_outcomes),
            hidden_pass_rate=hidden_pass_rate(all_outcomes),
        )

    def _save_results(
        self, result: ProbeResult, details: list[dict]
    ) -> None:
        """Save per-sample JSONL and probe report to output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Per-sample JSONL
        samples_path = (
            self.output_dir
            / f"probe_step{result.step}_samples.jsonl"
        )
        with samples_path.open("w", encoding="utf-8") as fh:
            for d in details:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")

        # Probe report
        report = {
            "step": result.step,
            "epoch": result.epoch,
            "probe_sha": self.probe_sha,
            "probe_sample_ids": result.probe_sample_ids,
            "generation_config": PROBE_GENERATION_CONFIG,
            "composite_value": result.composite_value,
            "composite_score": {
                "code_generation_pass_at_1": result.composite_score.code_generation_pass_at_1,
                "boundary_pass_at_1": result.composite_score.boundary_pass_at_1,
                "static_repair_success": result.composite_score.static_repair_success,
                "execution_repair_success": result.composite_score.execution_repair_success,
                "hidden_pass_rate": result.composite_score.hidden_pass_rate,
            },
            "metrics": result.metrics,
            "n_samples": len(details),
        }
        report_path = (
            self.output_dir
            / f"probe_step{result.step}_report.json"
        )
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tier 3: Full validation
# ---------------------------------------------------------------------------


class Tier3FullValidation:
    """Manages Tier 3 full validation against all 180 Validation v2 samples.

    Runs generate + sandbox for every sample, computes the five-component
    Composite Score, and checks hard constraints.

    Parameters
    ----------
    config : dict
        The loaded YAML config.
    validation_samples : list[Sample]
        Full Validation v2 sample list (180 samples).
    output_dir : Path
        Where to save per-sample / per-family / per-bucket metrics.
    baseline : dict, optional
        Historical baseline for hard-constraint check (P2 final metrics).
    """

    def __init__(
        self,
        config: dict,
        validation_samples: list[Sample],
        output_dir: Path,
        baseline: Optional[dict] = None,
    ):
        self.config = config
        self.validation_samples = validation_samples
        self.output_dir = Path(output_dir)
        self.baseline = baseline or {}

        self._composite_weights: dict[str, float] = {
            k: v
            for k, v in config["composite_score"].items()
            if not isinstance(v, dict)
        }
        self._hard_constraint = config["composite_score"].get(
            "hard_constraint", {}
        )

    def run(
        self,
        model: Any,
        tokenizer: Any,
        step: int,
        epoch: int,
        *,
        pytest_timeout_s: float = 10.0,
    ) -> FullValidationResult:
        """Execute full validation: generate + sandbox for all 180 samples.

        Returns a ``FullValidationResult``. Raises ``CompositeCoverageError``
        if a required bucket is missing.
        """
        import torch  # lazy import

        was_training = model.training
        model.eval()

        try:
            per_sample_details: list[dict] = []
            outcomes_by_variant: dict[str, list[EvalOutcome]] = {
                v: [] for v in _REQUIRED_VARIANT_BUCKETS
            }

            for sample in self.validation_samples:
                outcome, detail = evaluate_single_sample(
                    model, tokenizer, sample,
                    max_new_tokens=PROBE_GENERATION_CONFIG["max_new_tokens"],
                    pytest_timeout_s=pytest_timeout_s,
                )
                per_sample_details.append(detail)

                bucket = sample.variant_type
                if bucket in outcomes_by_variant:
                    outcomes_by_variant[bucket].append(outcome)

            # Compute composite (raises CompositeCoverageError if bucket missing)
            composite_score = self._compute_composite(outcomes_by_variant)
            composite_value = composite_score.compute(self._composite_weights)

            all_outcomes = []
            for v in _REQUIRED_VARIANT_BUCKETS:
                all_outcomes.extend(outcomes_by_variant[v])
            metrics = summarize(all_outcomes)

            # Hard constraint check
            hard_pass, hard_violations = self._check_hard_constraint(metrics)

            result = FullValidationResult(
                step=step,
                epoch=epoch,
                composite_score=composite_score,
                composite_value=composite_value,
                metrics=metrics,
                hard_constraint_pass=hard_pass,
                hard_constraint_violations=hard_violations,
            )

            self._save_results(result, per_sample_details, outcomes_by_variant)

            return result
        finally:
            if was_training:
                model.train()

    def _compute_composite(
        self, outcomes_by_variant: dict[str, list[EvalOutcome]]
    ) -> CompositeScore:
        """Build CompositeScore. Raises CompositeCoverageError if bucket empty."""
        missing_or_empty = [
            bucket
            for bucket in _REQUIRED_VARIANT_BUCKETS
            if not outcomes_by_variant.get(bucket)
        ]
        if missing_or_empty:
            raise CompositeCoverageError(
                "Tier3FullValidation hard fail: missing or empty variant "
                f"buckets: {missing_or_empty}. Required buckets = "
                f"{_REQUIRED_VARIANT_BUCKETS}. Readiness="
                f"{CompositeCoverageError.READINESS}."
            )

        from src.metrics import hidden_pass_rate, pass_at_1, repair_success_rate

        code_outcomes = outcomes_by_variant["code"]
        boundary_outcomes = outcomes_by_variant["boundary"]
        static_outcomes = outcomes_by_variant["static_repair"]
        exec_outcomes = outcomes_by_variant["execution_repair"]

        all_outcomes = (
            code_outcomes + boundary_outcomes
            + static_outcomes + exec_outcomes
        )

        return CompositeScore(
            code_generation_pass_at_1=pass_at_1(code_outcomes),
            boundary_pass_at_1=pass_at_1(boundary_outcomes),
            static_repair_success=repair_success_rate(static_outcomes),
            execution_repair_success=repair_success_rate(exec_outcomes),
            hidden_pass_rate=hidden_pass_rate(all_outcomes),
        )

    def _check_hard_constraint(
        self, metrics: dict
    ) -> tuple[bool, list[str]]:
        """Check code_generation drop vs P2 final baseline."""
        from src.metrics import normalize_baseline_key

        max_drop_pct = self._hard_constraint.get(
            "code_generation_drop_vs_p2_final_max_pct", 3.0
        )
        normalized_baseline = normalize_baseline_key(self.baseline)
        p2_codegen = normalized_baseline.get("pass_at_1", 0.0)
        current_codegen = metrics.get("pass_at_1", 0.0)
        drop_pct = (p2_codegen - current_codegen) * 100.0

        violations: list[str] = []
        if drop_pct > max_drop_pct:
            violations.append(
                f"code_generation_pass_at_1 dropped {drop_pct:.2f}pp vs P2 "
                f"final ({p2_codegen:.4f} -> {current_codegen:.4f}), max "
                f"allowed {max_drop_pct}pp"
            )
        return (len(violations) == 0, violations)

    def _save_results(
        self,
        result: FullValidationResult,
        details: list[dict],
        outcomes_by_variant: dict[str, list[EvalOutcome]],
    ) -> None:
        """Save per-sample / per-bucket / full report to output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Per-sample JSONL
        samples_path = (
            self.output_dir
            / f"fullval_epoch{result.epoch}_samples.jsonl"
        )
        with samples_path.open("w", encoding="utf-8") as fh:
            for d in details:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")

        # Per-bucket summary
        from src.metrics import (
            hidden_pass_rate, pass_at_1, repair_success_rate,
        )

        per_bucket: dict[str, dict] = {}
        for bucket, outcomes in outcomes_by_variant.items():
            if not outcomes:
                per_bucket[bucket] = {"n": 0}
                continue
            per_bucket[bucket] = {
                "n": len(outcomes),
                "pass_at_1": pass_at_1(outcomes),
                "repair_success_rate": repair_success_rate(outcomes),
                "hidden_pass_rate": hidden_pass_rate(outcomes),
            }

        # Full report
        report = {
            "step": result.step,
            "epoch": result.epoch,
            "composite_value": result.composite_value,
            "composite_score": {
                "code_generation_pass_at_1": result.composite_score.code_generation_pass_at_1,
                "boundary_pass_at_1": result.composite_score.boundary_pass_at_1,
                "static_repair_success": result.composite_score.static_repair_success,
                "execution_repair_success": result.composite_score.execution_repair_success,
                "hidden_pass_rate": result.composite_score.hidden_pass_rate,
            },
            "metrics": result.metrics,
            "hard_constraint_pass": result.hard_constraint_pass,
            "hard_constraint_violations": result.hard_constraint_violations,
            "per_bucket": per_bucket,
            "n_samples": len(details),
            "generation_config": PROBE_GENERATION_CONFIG,
        }
        report_path = (
            self.output_dir
            / f"fullval_epoch{result.epoch}_report.json"
        )
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Early stopping state machine
# ---------------------------------------------------------------------------


class EarlyStoppingManager:
    """Persistent early-stop state machine for Tier 2/3 evaluations.

    Rules (checked in order):
    1. NaN/Inf detected → immediate confirmed stop.
    2. Required bucket missing (CompositeCoverageError) → immediate stop.
    3. Syntax drop > 5pp vs previous measurement → immediate stop.
    4. Timeout > 8% → immediate stop.
    5. probe_patience consecutive probes with no improvement → pending stop.
    6. Pending stop + next full validation with no improvement → confirmed.
    7. max_epochs reached → confirmed stop.

    State is persistable via ``to_dict()`` / ``from_dict()``.
    """

    def __init__(self, config: dict):
        es_cfg = config.get("early_stopping", {})
        self.probe_patience: int = es_cfg.get("probe_patience", 4)
        self.probe_min_delta: float = es_cfg.get("probe_min_delta", 0.005)
        self.max_epochs: int = es_cfg.get("max_epochs", 3)

        # Immediate-stop thresholds (Issue #14 Wave 3-D spec)
        self.syntax_drop_max_pct: float = 0.05  # 5pp
        self.timeout_max_rate: float = 0.08     # 8%

        # History
        self.probe_history: list[ProbeResult] = []
        self.full_history: list[FullValidationResult] = []

        # Best tracking
        self._best_probe: Optional[float] = None
        self._best_full: Optional[float] = None
        self._consecutive_no_improve: int = 0
        self._prev_syntax: Optional[float] = None

        # Stop state
        self.pending_stop: bool = False
        self.confirmed_stop: bool = False
        self.stop_reason: str = ""

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_probe(
        self,
        probe_result: ProbeResult,
        metrics: dict,
    ) -> tuple[bool, str]:
        """Record a probe result. Returns (should_stop, reason)."""
        self.probe_history.append(probe_result)

        # Immediate stops
        stop, reason = self._check_immediate_stop(metrics)
        if stop:
            self.confirmed_stop = True
            self.stop_reason = reason
            return (True, reason)

        # Track improvement
        if self._best_probe is None:
            self._best_probe = probe_result.composite_value
            self._consecutive_no_improve = 0
        elif probe_result.composite_value > self._best_probe + self.probe_min_delta:
            self._best_probe = probe_result.composite_value
            self._consecutive_no_improve = 0
        else:
            self._consecutive_no_improve += 1

        # Pending stop check
        if (
            self._consecutive_no_improve >= self.probe_patience
            and not self.pending_stop
        ):
            self.pending_stop = True
            self.stop_reason = (
                f"pending_stop: {self.probe_patience} consecutive probes "
                f"without improvement (min_delta={self.probe_min_delta})"
            )

        return (self.confirmed_stop, self.stop_reason)

    def record_full(
        self,
        full_result: FullValidationResult,
        metrics: dict,
    ) -> tuple[bool, str]:
        """Record a full validation result. Returns (should_stop, reason)."""
        self.full_history.append(full_result)

        # Immediate stops
        stop, reason = self._check_immediate_stop(metrics)
        if stop:
            self.confirmed_stop = True
            self.stop_reason = reason
            return (True, reason)

        # Update best full
        if self._best_full is None or full_result.composite_value > self._best_full:
            self._best_full = full_result.composite_value

        # Confirm pending stop
        if self.pending_stop and not self.confirmed_stop:
            if len(self.full_history) >= 2:
                prev = self.full_history[-2].composite_value
                curr = full_result.composite_value
                if curr <= prev + self.probe_min_delta:
                    self.confirmed_stop = True
                    self.stop_reason = (
                        "confirmed_stop: probe pending + full validation "
                        "no improvement"
                    )
                    return (True, self.stop_reason)
                else:
                    # Full validation improved vs previous full → clear pending
                    self.pending_stop = False
                    self.stop_reason = ""
            else:
                # Only 1 full validation: compare against best probe.
                # If full composite exceeds best probe by min_delta, the
                # model is still improving → clear pending.
                if (
                    self._best_probe is not None
                    and full_result.composite_value
                    > self._best_probe + self.probe_min_delta
                ):
                    self.pending_stop = False
                    self.stop_reason = ""

        # Max epochs check
        if full_result.epoch >= self.max_epochs:
            self.confirmed_stop = True
            self.stop_reason = (
                f"max_epochs reached: {full_result.epoch} >= {self.max_epochs}"
            )
            return (True, self.stop_reason)

        return (self.confirmed_stop, self.stop_reason)

    def check_nan_inf(self, nan_inf_detected: bool) -> tuple[bool, str]:
        """Immediate stop on NaN/Inf."""
        if nan_inf_detected:
            self.confirmed_stop = True
            self.stop_reason = (
                "nan_or_inf_detected: immediate stop on divergence"
            )
            return (True, self.stop_reason)
        return (self.confirmed_stop, self.stop_reason)

    def record_bucket_missing(self, bucket_error: CompositeCoverageError) -> None:
        """Immediate stop on required bucket missing."""
        self.confirmed_stop = True
        self.stop_reason = (
            f"required_bucket_missing: {bucket_error}. Readiness="
            f"{CompositeCoverageError.READINESS}"
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def should_stop(self) -> tuple[bool, str]:
        """Returns (confirmed_stop, reason)."""
        return (self.confirmed_stop, self.stop_reason)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_immediate_stop(self, metrics: dict) -> tuple[bool, str]:
        """Check syntax drop and timeout rate for immediate stop."""
        if not metrics:
            return (False, "")

        # Timeout > 8%
        timeout = metrics.get("timeout_rate", 0.0)
        if timeout > self.timeout_max_rate:
            return (
                True,
                f"timeout_rate {timeout:.2%} > {self.timeout_max_rate:.0%} "
                f"threshold: immediate stop",
            )

        # Syntax drop > 5pp vs previous measurement
        current_syntax = metrics.get("syntax_rate", 1.0)
        if self._prev_syntax is not None:
            drop = self._prev_syntax - current_syntax
            if drop > self.syntax_drop_max_pct:
                return (
                    True,
                    f"syntax_rate dropped {drop*100:.2f}pp > "
                    f"{self.syntax_drop_max_pct*100:.0f}pp threshold: "
                    f"immediate stop",
                )
        self._prev_syntax = current_syntax

        return (False, "")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize state for persistence (evaluator_state.json)."""
        return {
            "probe_patience": self.probe_patience,
            "probe_min_delta": self.probe_min_delta,
            "max_epochs": self.max_epochs,
            "syntax_drop_max_pct": self.syntax_drop_max_pct,
            "timeout_max_rate": self.timeout_max_rate,
            "probe_history_count": len(self.probe_history),
            "full_history_count": len(self.full_history),
            "best_probe": self._best_probe,
            "best_full": self._best_full,
            "consecutive_no_improve": self._consecutive_no_improve,
            "prev_syntax": self._prev_syntax,
            "pending_stop": self.pending_stop,
            "confirmed_stop": self.confirmed_stop,
            "stop_reason": self.stop_reason,
            "probe_composite_values": [
                p.composite_value for p in self.probe_history
            ],
            "full_composite_values": [
                f.composite_value for f in self.full_history
            ],
        }

    @classmethod
    def from_dict(cls, config: dict, d: dict) -> "EarlyStoppingManager":
        """Restore state from a persisted dict."""
        mgr = cls(config)
        mgr._best_probe = d.get("best_probe")
        mgr._best_full = d.get("best_full")
        mgr._consecutive_no_improve = d.get("consecutive_no_improve", 0)
        mgr._prev_syntax = d.get("prev_syntax")
        mgr.pending_stop = d.get("pending_stop", False)
        mgr.confirmed_stop = d.get("confirmed_stop", False)
        mgr.stop_reason = d.get("stop_reason", "")
        # Histories are not fully restored (ProbeResult/FullValidationResult
        # are dataclasses; we restore composite values for reference but
        # the actual objects would need re-construction).
        return mgr


# ---------------------------------------------------------------------------
# Checkpoint evidence
# ---------------------------------------------------------------------------


class CheckpointEvidence:
    """Records checkpoint provenance (weight/config/data SHA).

    Saves one JSON line per checkpoint to
    ``reports/p3/checkpoint-evidence.jsonl``.
    """

    def __init__(self, reports_dir: Path):
        self.reports_dir = Path(reports_dir)
        self.evidence_path = self.reports_dir / "checkpoint-evidence.jsonl"

    def record(
        self,
        checkpoint_path: Path,
        config_path: Path,
        train_file: Path,
        validation_file: Path,
        generation_config: dict[str, Any],
        metrics: dict,
        created_at: Optional[str] = None,
    ) -> dict:
        """Record evidence for one checkpoint. Returns the evidence dict."""
        checkpoint_path = Path(checkpoint_path)
        config_path = Path(config_path)
        train_file = Path(train_file)
        validation_file = Path(validation_file)

        evidence = {
            "checkpoint_path": str(checkpoint_path),
            "weight_sha": self._sha256_file(
                checkpoint_path / "adapter_model.safetensors",
                normalize_crlf=True,
            ),
            "config_sha": self._sha256_file(
                config_path, normalize_crlf=True,
            ),
            "train_data_sha": self._sha256_file(
                train_file, normalize_crlf=False,
            ),
            "validation_sha": self._sha256_file(
                validation_file, normalize_crlf=False,
            ),
            "generation_config_sha": self._sha256_json(generation_config),
            "metrics_sha": self._sha256_json(metrics),
            "created_at": created_at
            or datetime.now(timezone.utc).isoformat(),
        }

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        with self.evidence_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(evidence, ensure_ascii=False) + "\n")

        return evidence

    def _sha256_file(
        self, path: Path, normalize_crlf: bool = False
    ) -> str:
        """SHA256 of file content. CRLF→LF normalization if requested."""
        if not path.exists():
            return ""
        data = path.read_bytes()
        if normalize_crlf:
            data = data.replace(b"\r\n", b"\n")
        return hashlib.sha256(data).hexdigest()

    def _sha256_json(self, obj: Any) -> str:
        """SHA256 of a JSON-serialized object (sorted keys)."""
        raw = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
