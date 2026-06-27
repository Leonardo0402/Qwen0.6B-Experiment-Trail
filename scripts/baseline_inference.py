"""
scripts/baseline_inference.py -- Smoke-test baseline inference for Qwen3-0.6B.

Loads the model from a local directory, runs a Chinese coding instruction
prompt, measures peak GPU memory and generation time, and optionally writes
a JSON result to evaluations/baselines/baseline_smoke.json.

Usage:
    python scripts/baseline_inference.py [--model PATH] [--max-new-tokens N] [--prompt TEXT]

Exit codes:
    0  success
    1  failure (model missing, load error, inference error, etc.)

Windows DLL / import-order note
---------------------------------
torch and transformers are NOT imported at module level.  This script is
collected by pytest alongside other test modules that import ``datasets`` --
and importing torch (which initialises CUDA DLL state) BEFORE datasets'
C-extension DLLs are loaded triggers a Windows access-violation (0xC0000005).
By deferring the torch/transformers imports until main() -- after the model
path check -- we avoid the DLL ordering issue entirely.

Eager module-level imports are therefore limited to pure-stdlib and
huggingface_hub (which has no CUDA dependency).  OSError [WinError 126]
(missing DLL) is caught in each deferred import block.

trust_remote_code note
-----------------------
trust_remote_code is NOT set.  Qwen3 is natively supported in transformers
5.x and does not require remote code execution.  If a future model variant
does require it, add trust_remote_code=True to the from_pretrained calls and
document the change here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root and defaults
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = str(_PROJECT_ROOT / "models" / "Qwen3-0.6B")

# Default Chinese instruction (raw UTF-8). stdout is reconfigured to UTF-8 in
# main(), so printing this is safe on gbk/cp936 consoles.
# "请用 Python 实现一个函数，返回列表中的最大值，只输出代码。"
DEFAULT_INSTRUCTION = (
    "请用 Python 实现一个函数，"
    "返回列表中的最大值，只输出代码。"
)


# ---------------------------------------------------------------------------
# Pure helpers -- no torch/transformers required; safe to import in tests
# ---------------------------------------------------------------------------

def check_model_exists(model_path: Path) -> bool:
    """Return True when *model_path* looks like a valid local model directory.

    "Valid" means the directory exists and contains config.json.
    """
    return (model_path / "config.json").exists()


def bytes_to_gb(n: int) -> float:
    """Convert bytes to gigabytes, rounded to 3 decimal places."""
    return round(n / (1024 ** 3), 3)


def build_prompt(tokenizer: Any, instruction: str) -> str:
    """Build a chat-template prompt string for *instruction*.

    Wraps the instruction in a user-role message and calls
    ``tokenizer.apply_chat_template`` (transformers 5.x API) to produce the
    final prompt string, including the assistant-turn header that signals to
    the model where to start generating.

    Parameters
    ----------
    tokenizer:
        A loaded HuggingFace tokenizer that supports ``apply_chat_template``.
    instruction:
        The natural-language instruction to pass as the user message.

    Returns
    -------
    A non-empty string ready to be tokenized and fed to the model.
    """
    messages = [{"role": "user", "content": instruction}]
    prompt: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# ---------------------------------------------------------------------------
# Model-loading helper (thin + mockable; imports deferred to call time)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str) -> Tuple[Any, Any]:
    """Load the tokenizer and causal-LM model from *model_path*.

    Imports transformers and torch locally so that this module can be safely
    collected by pytest even in environments where torch is not yet loaded.

    Uses float16 precision and device_map="cuda" to fit within the
    RTX 3050 4 GB VRAM budget.  trust_remote_code is NOT set -- Qwen3 is
    natively supported in transformers 5.x.

    Returns
    -------
    (model, tokenizer) tuple.

    Raises
    ------
    RuntimeError if transformers or torch are unavailable.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"transformers not available: {exc}") from exc
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"torch not available: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference helper (thin + mockable; imports deferred to call time)
# ---------------------------------------------------------------------------

def run_inference(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[str, float, float]:
    """Run a single greedy-decode forward pass and report metrics.

    Peak GPU memory is measured via torch.cuda.reset_peak_memory_stats() /
    torch.cuda.max_memory_allocated() so the measurement is scoped to this
    generation call only.

    Parameters
    ----------
    model:
        Loaded AutoModelForCausalLM (on CUDA).
    tokenizer:
        Matching AutoTokenizer.
    prompt:
        Full chat-template prompt string (already includes system/user turns).
    max_new_tokens:
        Maximum number of new tokens to generate.

    Returns
    -------
    (completion_text, peak_vram_gb, generation_time_s)
    """
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"torch not available: {exc}") from exc

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    t1 = time.perf_counter()
    peak_bytes = torch.cuda.max_memory_allocated()

    # Decode only the newly generated tokens (skip the prompt).
    new_ids = output_ids[0][input_len:]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)

    return completion, bytes_to_gb(peak_bytes), t1 - t0


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def write_results(data: dict, output_path: Path) -> None:
    """Write *data* as formatted JSON to *output_path*, creating parents."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Load model, run inference, report results.

    Returns 0 on success, 1 on any failure.

    Library import ordering
    -----------------------
    torch and transformers are imported INSIDE this function, AFTER the model
    path guard, so that:
    (a) pytest can collect this module without triggering CUDA DLL loads, and
    (b) we fail fast on a missing model before spending time on lib checks.
    """
    # Make stdout robust to non-ASCII (Chinese model output) on gbk/cp936
    # consoles produced by conda run on Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="Baseline inference smoke-test for Qwen3-0.6B."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to local model directory (default: models/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_INSTRUCTION,
        help="Instruction text to send as the user message",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()

    # ------------------------------------------------------------------
    # Pre-flight: model directory (fast check; no heavy imports needed)
    # ------------------------------------------------------------------
    if not check_model_exists(model_path):
        print(
            f"ERROR: model not found at {model_path}\n"
            "Run scripts/download_model.py first to download the model."
        )
        return 1

    # ------------------------------------------------------------------
    # Import heavy libraries (deferred to here to avoid Windows DLL ordering
    # issues when this module is collected alongside datasets-using test files)
    # ------------------------------------------------------------------
    try:
        from transformers import AutoModelForCausalLM as _ACM, AutoTokenizer as _AT  # noqa: F401
    except (ImportError, OSError) as exc:
        print(f"ERROR: transformers not available: {exc}")
        return 1
    try:
        import torch as _torch  # noqa: F401
    except (ImportError, OSError) as exc:
        print(f"ERROR: torch not available: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Load model and tokenizer
    # ------------------------------------------------------------------
    print(f"Loading model from {model_path} ...")
    t_load_start = time.perf_counter()
    try:
        model, tokenizer = load_model_and_tokenizer(str(model_path))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR loading model: {exc}")
        return 1
    t_load_end = time.perf_counter()
    load_time_s = t_load_end - t_load_start
    print(f"Model loaded in {load_time_s:.2f}s")

    # ------------------------------------------------------------------
    # Build prompt and run inference
    # ------------------------------------------------------------------
    prompt = build_prompt(tokenizer, args.prompt)
    # Print only the first 120 chars of the prompt so large system prompts
    # don't flood the terminal; use repr to keep output ASCII-safe.
    prompt_preview = repr(prompt[:120])
    print(f"\nPrompt preview: {prompt_preview}")
    print(f"\nRunning inference (max_new_tokens={args.max_new_tokens}, do_sample=False) ...")

    try:
        completion, peak_vram_gb, gen_time_s = run_inference(
            model, tokenizer, prompt, args.max_new_tokens
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR during inference: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Report results
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Completion:")
    print(completion)
    print("=" * 60)
    print(f"Peak VRAM:       {peak_vram_gb:.3f} GB")
    print(f"Load time:       {load_time_s:.2f}s")
    print(f"Generation time: {gen_time_s:.2f}s")

    # ------------------------------------------------------------------
    # Write JSON result
    # ------------------------------------------------------------------
    results = {
        "model_path": str(model_path),
        "prompt": args.prompt,
        "completion": completion,
        "peak_vram_gb": peak_vram_gb,
        "load_time_s": round(load_time_s, 3),
        "gen_time_s": round(gen_time_s, 3),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    output_path = _PROJECT_ROOT / "evaluations" / "baselines" / "baseline_smoke.json"
    write_results(results, output_path)
    print(f"\nResults written to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
