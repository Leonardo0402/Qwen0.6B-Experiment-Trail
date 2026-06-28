"""
scripts/chat.py -- Interactive local chat with Qwen3-0.6B (+ optional LoRA).

Loads a base model and optionally a LoRA adapter, then enters an interactive
REPL loop.  The chat template is applied automatically.

Usage
-----
    python scripts/chat.py
        --model models/Qwen3-0.6B
        --adapter adapters/code-lora-v3-repair

    # Baseline (no adapter)
    python scripts/chat.py --model models/Qwen3-0.6B

Commands in the REPL
--------------------
    /quit, /exit     Exit the chat.
    /clear           Clear conversation history.
    /system <text>   Set a new system prompt.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DEFAULT_SYSTEM = (
    "你是一个严谨的 Python 代码助手。"
    "根据任务（及真实执行反馈）输出正确代码。"
    "除非用户要求解释，否则只输出完整代码。"
)


def _load_model(model_path: str, adapter: Optional[str] = None) -> tuple[Any, Any]:
    """Load tokenizer and model, optionally with LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path} (fp16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter:
        print(f"Loading LoRA adapter from {adapter} ...")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.eval()
    print("Model ready.")
    return model, tokenizer


def chat_loop(model: Any, tokenizer: Any, system_prompt: str) -> None:
    """Run the interactive chat REPL."""
    import torch

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    print("\n=== Qwen3-0.6B Code Chat ===")
    print("Type /quit to exit, /clear to reset history, /system <text> to change system prompt.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("[History cleared]")
            continue
        if user_input.lower().startswith("/system "):
            system_prompt = user_input[8:].strip()
            messages[0] = {"role": "system", "content": system_prompt}
            print(f"[System prompt updated]")
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Generate
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][input_len:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)

        print(f"Assistant: {response}")
        print()

        # Add assistant message to history
        messages.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive chat with Qwen3-0.6B (+ optional LoRA).",
    )
    p.add_argument("--model", required=True, help="Path to model directory.")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter directory.")
    return p


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    if args.adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            print(f"ERROR: adapter not found: {adapter_path}", file=sys.stderr)
            return 1

    try:
        model, tokenizer = _load_model(str(model_path), adapter=args.adapter)
    except Exception:
        traceback.print_exc()
        return 1

    chat_loop(model, tokenizer, DEFAULT_SYSTEM)
    return 0


if __name__ == "__main__":
    sys.exit(main())