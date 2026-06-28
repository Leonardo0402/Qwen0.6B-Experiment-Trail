"""
scripts/evaluate_model.py -- Trusted evaluation of a model (+ optional LoRA).

P0 fixes:
  - Only accepts raw Sample format (test_raw.jsonl), rejects ChatML.
  - Validates every record via Pydantic Sample.model_validate().
  - Enforces non-empty public_tests; empty tests => immediate abort.
  - Tracks num_collected from pytest; zero-collected => failure.
  - Canary mechanism: 3 wrong codes must all fail before real eval.
  - Records dataset_sha256, generation_config, canary results.
  - No more sample_id="?" — uses validated Sample objects.

Usage
-----
    python scripts/evaluate_model.py
        --model models/Qwen3-0.6B
        --output evaluations/fixed-p0/baseline.json

    # Default dataset: data/frozen-eval/v1/test_raw.jsonl (if exists),
    # fallback to data/splits/test_raw.jsonl

    # With LoRA adapter
    python scripts/evaluate_model.py
        --model models/Qwen3-0.6B
        --adapter adapters/code-lora-v3-easy
        --output evaluations/fixed-p0/v3-easy.json

    # Explicit dataset
    python scripts/evaluate_model.py
        --model models/Qwen3-0.6B
        --dataset data/splits/test_raw.jsonl
        --output evaluations/fixed-p0/baseline.json

    # Skip canary (dev only)
    python scripts/evaluate_model.py ... --skip-canary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import EvalOutcome, summarize  # noqa: E402
from src.sandbox import run_pytest, check_code_safety  # noqa: E402
from src.schemas import Sample  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "你是一个严谨的 Python 代码助手。"
    "根据任务（及真实执行反馈）输出正确代码。"
    "除非用户要求解释，否则只输出完整代码。"
)

# Fixed generation config for all evaluations — deterministic
_GENERATION_CONFIG = {
    "enable_thinking": False,
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 384,
    "temperature": None,  # not used in deterministic mode
    "top_p": None,        # not used in deterministic mode
    "repetition_penalty": 1.1,
    "pad_token_id": "<eos>",
}

# Canary codes — all must FAIL
_CANARY_CODES = {
    "hello_world": 'print("Hello, World!")',
    "pass_stmt": "pass",
    "return_none": "def solution(*args, **kwargs):\n    return None",
}


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def _dataset_sha256(path: Path) -> str:
    """Compute SHA-256 of the raw file content."""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _load_and_validate_samples(path: Path) -> list[Sample]:
    """Load JSONL and validate every record as a Pydantic Sample.

    Rejects:
      - ChatML format (records with only "messages" key)
      - Malformed JSON
      - Pydantic validation failures
      - Empty public_tests

    Aborts on first error — no skipping.
    """
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"ERROR: Malformed JSON at line {line_num}: {exc}",
                    file=sys.stderr,
                )
                raise

            # Detect ChatML format
            if "messages" in record and "sample_id" not in record:
                print(
                    "ERROR: Evaluation dataset is ChatML training format.\n"
                    "Use the raw Sample dataset, such as test_raw.jsonl.",
                    file=sys.stderr,
                )
                raise ValueError(
                    "Evaluation dataset is ChatML training format. "
                    "Use the raw Sample dataset, such as test_raw.jsonl."
                )

            # Pydantic validation
            try:
                sample = Sample.model_validate(record)
            except Exception as exc:
                sid = record.get("sample_id", "<unknown>")
                print(
                    f"ERROR: Pydantic validation failed at line {line_num}, "
                    f"sample_id={sid}: {exc}",
                    file=sys.stderr,
                )
                raise

            # Enforce non-empty public tests
            if not sample.public_tests.strip():
                print(
                    f"ERROR: EMPTY_PUBLIC_TESTS: {sample.sample_id}",
                    file=sys.stderr,
                )
                raise ValueError(f"EMPTY_PUBLIC_TESTS: {sample.sample_id}")

            samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Canary mechanism
# ---------------------------------------------------------------------------

def run_canary(samples: list[Sample], pytest_timeout_s: float = 10.0) -> dict:
    """Run 3 canary codes against real test samples. All must fail.

    Returns a dict with:
      - passed: bool (True iff all canaries failed)
      - cases: list of per-canary details
    """
    # Pick 3 samples with clear function constraints and tests
    canary_samples = []
    for s in samples:
        if s.task_type == "code_generation" and len(canary_samples) < 3:
            canary_samples.append(s)
    if len(canary_samples) < 3:
        # Fall back to any samples
        canary_samples = samples[:3]

    cases = []
    all_failed = True

    for name, code in _CANARY_CODES.items():
        sample = canary_samples[len(cases) % len(canary_samples)]
        pub_result = run_pytest(code, sample.public_tests, timeout_s=pytest_timeout_s)

        case = {
            "canary_name": name,
            "canary_code": code,
            "test_sample_id": sample.sample_id,
            "family_id": sample.family_id,
            "public_passed": pub_result.passed,
            "num_collected": pub_result.num_collected,
            "num_passed": pub_result.num_passed,
            "num_failed": pub_result.num_failed,
            "timed_out": pub_result.timed_out,
            "expected": "fail",
            "actual": "pass" if pub_result.passed else "fail",
        }
        cases.append(case)

        if pub_result.passed:
            all_failed = False
            print(
                f"CANARY FAILURE: '{name}' was judged as PASS — "
                "evaluation harness is not trustworthy",
                file=sys.stderr,
            )
        elif pub_result.num_collected == 0:
            all_failed = False
            print(
                f"CANARY FAILURE: '{name}' collected 0 tests — "
                "evaluation harness is not trustworthy",
                file=sys.stderr,
            )

    return {"passed": all_failed, "cases": cases}


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def _extract_code_block(text: str) -> Optional[str]:
    """Extract code from the first triple-backtick python block."""
    match = re.search(r"```python\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Prompt building (matches schemas.to_chatml)
# ---------------------------------------------------------------------------

def _build_eval_messages(sample: Sample) -> list[dict]:
    """Build ChatML messages for evaluation, matching training format."""
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
# Per-sample evaluation
# ---------------------------------------------------------------------------

def _evaluate_one(
    model: Any,
    tokenizer: Any,
    sample: Sample,
    *,
    max_new_tokens: int,
    pytest_timeout_s: float,
) -> tuple[EvalOutcome, dict]:
    """Evaluate a single sample and return (EvalOutcome, detail_dict)."""
    import torch

    detail = {
        "sample_id": sample.sample_id,
        "family_id": sample.family_id,
        "task_type": sample.task_type,
        "difficulty": sample.difficulty,
        "generated": "",
        "extracted_code": None,
        "format_ok": False,
        "syntax_ok": False,
        "public_tests_present": True,  # enforced by validation
        "public_tests_collected": 0,
        "public_passed": False,
        "hidden_tests_present": bool(sample.hidden_tests.strip()),
        "hidden_tests_collected": 0,
        "hidden_passed": False,
        "timed_out": False,
        "error": None,
    }

    repair_succeeded = None
    broke_other_tests = None
    is_repair = sample.task_type in ("static_repair", "execution_repair")

    try:
        # Build prompt
        messages = _build_eval_messages(sample)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,  # ignored when do_sample=False
                top_p=1.0,        # ignored when do_sample=False
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                repetition_penalty=1.1,
            )

        new_ids = output_ids[0][input_len:]
        completion = tokenizer.decode(new_ids, skip_special_tokens=True)
        detail["generated"] = completion

        del inputs, output_ids, new_ids

        # Extract code block
        extracted = _extract_code_block(completion)
        detail["extracted_code"] = extracted
        detail["format_ok"] = extracted is not None

        if extracted is None:
            extracted = completion.strip()

        # Safety check
        safety_warnings = check_code_safety(extracted)
        if safety_warnings:
            detail["safety_warnings"] = safety_warnings

        # Syntax check
        try:
            import ast as _ast
            _ast.parse(extracted)
            detail["syntax_ok"] = True
        except SyntaxError:
            detail["syntax_ok"] = False

        # Run pytest — public tests (guaranteed non-empty by validation)
        pub_result = run_pytest(extracted, sample.public_tests, timeout_s=pytest_timeout_s)
        detail["timed_out"] = pub_result.timed_out
        detail["public_tests_collected"] = pub_result.num_collected
        detail["public_passed"] = pub_result.passed
        detail["public_detail"] = {
            "passed": pub_result.passed,
            "num_collected": pub_result.num_collected,
            "num_passed": pub_result.num_passed,
            "num_failed": pub_result.num_failed,
            "timed_out": pub_result.timed_out,
        }

        # Run pytest — hidden tests
        if sample.hidden_tests.strip():
            hid_result = run_pytest(extracted, sample.hidden_tests, timeout_s=pytest_timeout_s)
            detail["hidden_passed"] = hid_result.passed
            detail["hidden_tests_collected"] = hid_result.num_collected
            detail["timed_out"] = detail["timed_out"] or hid_result.timed_out
            detail["hidden_detail"] = {
                "passed": hid_result.passed,
                "num_collected": hid_result.num_collected,
                "num_passed": hid_result.num_passed,
                "num_failed": hid_result.num_failed,
                "timed_out": hid_result.timed_out,
            }
        else:
            detail["hidden_passed"] = False  # NOT vacuously True
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
        traceback.print_exc()

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
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    dataset_path: str,
    output_path: str,
    *,
    adapter: Optional[str] = None,
    max_new_tokens: int = 384,
    pytest_timeout_s: float = 10.0,
    skip_canary: bool = False,
) -> dict:
    """Run evaluation and return a results dict (also saved to *output_path*)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # ------------------------------------------------------------------
    # 1. Validate dataset
    # ------------------------------------------------------------------
    frozen_eval_path = _ROOT / "data" / "frozen-eval" / "v1" / "test_raw.jsonl"
    is_frozen = dataset_path.resolve() == frozen_eval_path.resolve()

    print(f"Validating dataset: {dataset_path}")
    if is_frozen:
        print("  [Frozen Eval v1]")
    dataset_sha = _dataset_sha256(dataset_path)
    samples = _load_and_validate_samples(dataset_path)
    print(f"  Loaded {len(samples)} validated samples")
    print(f"  Dataset SHA256: {dataset_sha[:16]}...")

    # Count task types and families
    task_type_counts: dict[str, int] = {}
    family_ids: set[str] = set()
    for s in samples:
        task_type_counts[s.task_type] = task_type_counts.get(s.task_type, 0) + 1
        family_ids.add(s.family_id)
    print(f"  Task types: {task_type_counts}")
    print(f"  Unique families: {len(family_ids)}")

    schema_validation = {
        "validated_samples": len(samples),
        "failed_samples": 0,
    }

    # ------------------------------------------------------------------
    # 2. Run canary
    # ------------------------------------------------------------------
    canary_result: dict
    if skip_canary:
        print("WARNING: --skip-canary used (dev mode). Canary not run.")
        canary_result = {"passed": None, "skipped": True, "cases": []}
    else:
        print("\n=== Running Canary Tests ===")
        canary_result = run_canary(samples, pytest_timeout_s=pytest_timeout_s)
        if not canary_result["passed"]:
            print(
                "CANARY FAILURE: evaluation harness is not trustworthy. "
                "Aborting.",
                file=sys.stderr,
            )
            raise RuntimeError(
                "CANARY FAILURE: evaluation harness is not trustworthy"
            )
        print("  All canaries failed as expected. Harness is trustworthy.\n")

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )

    if adapter:
        print(f"Loading LoRA adapter from {adapter} ...")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.config.use_cache = False
    model.eval()

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    outcomes: list[EvalOutcome] = []
    details: list[dict] = []

    for i, sample in enumerate(samples):
        print(
            f"\r[{i + 1}/{len(samples)}] {sample.sample_id}",
            end="", flush=True,
        )

        outcome, detail = _evaluate_one(
            model, tokenizer, sample,
            max_new_tokens=max_new_tokens,
            pytest_timeout_s=pytest_timeout_s,
        )
        outcomes.append(outcome)
        details.append(detail)

    print()  # newline after progress

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    metrics = summarize(outcomes)

    # ------------------------------------------------------------------
    # 6. Assemble results
    # ------------------------------------------------------------------
    results = {
        "model_path": str(model_path.resolve()),
        "adapter": adapter,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha,
        "generation_config": _GENERATION_CONFIG,
        "canary": canary_result,
        "schema_validation": schema_validation,
        "sample_count": len(samples),
        "task_type_counts": task_type_counts,
        "family_count": len(family_ids),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "outcomes": details,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    print(f"\nResults saved to: {output_path}")
    print(f"\n=== Metrics ===")
    print(f"  Pass@1:              {metrics['pass_at_1']:.3f}")
    print(f"  Syntax rate:         {metrics['syntax_rate']:.3f}")
    print(f"  Hidden pass rate:    {metrics['hidden_pass_rate']:.3f}")
    print(f"  Format compliance:   {metrics['format_compliance_rate']:.3f}")
    print(f"  Timeout rate:        {metrics['timeout_rate']:.3f}")
    print(f"  Repair success:      {metrics['repair_success_rate']:.3f}")
    print(f"  Regression rate:     {metrics['regression_rate']:.3f}")
    print(f"  Total samples:       {int(metrics['n_total'])}")
    print(f"  Generation:          {int(metrics['n_generation'])}")
    print(f"  Repair:              {int(metrics['n_repair'])}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_dataset() -> str:
    """Return the default evaluation dataset path.

    Prefers the frozen eval set if it exists, otherwise falls back to the
    legacy splits test set.
    """
    frozen = _ROOT / "data" / "frozen-eval" / "v1" / "test_raw.jsonl"
    legacy = _ROOT / "data" / "splits" / "test_raw.jsonl"
    if frozen.exists():
        return str(frozen)
    if legacy.exists():
        return str(legacy)
    # If neither exists, return the frozen path so argparse will show
    # a meaningful error when the file is missing.
    return str(frozen)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a model (+ optional LoRA) on a test set.",
    )
    p.add_argument("--model", required=True, help="Path to model directory.")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter directory.")
    p.add_argument(
        "--dataset",
        default=_default_dataset(),
        help="Path to test_raw.jsonl file. "
             "Default: data/frozen-eval/v1/test_raw.jsonl (if exists), "
             "else data/splits/test_raw.jsonl.",
    )
    p.add_argument("--output", required=True, help="Path to output JSON file.")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--pytest-timeout", type=float, default=10.0)
    p.add_argument(
        "--run-canary", action="store_true", default=True,
        help="Run canary tests (default: always run).",
    )
    p.add_argument(
        "--skip-canary", action="store_true", default=False,
        help="Skip canary tests (dev only; recorded in output).",
    )
    return p


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    args = _build_parser().parse_args()

    try:
        evaluate_model(
            model_path=args.model,
            dataset_path=args.dataset,
            output_path=args.output,
            adapter=args.adapter,
            max_new_tokens=args.max_new_tokens,
            pytest_timeout_s=args.pytest_timeout,
            skip_canary=args.skip_canary,
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
