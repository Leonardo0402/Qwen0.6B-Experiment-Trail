"""
scripts/smoke_test_p1.py -- P1 minimal 1-batch forward/backward smoke test.

Verifies the Assistant-only Loss pipeline end-to-end with the real Qwen3-0.6B:
  1. Load tokenizer + model
  2. Load 2 samples from data/curriculum/easy/train.jsonl (Raw Sample format)
  3. Build AssistantOnlyDataset + AssistantOnlyCollator
  4. Apply LoRA (independent mode, rank=32)
  5. 1-batch forward + backward
  6. Verify: loss finite, labels mask correct, supervised tokens > 0

Runs on CPU if CUDA unavailable (smoke test only verifies pipeline, not perf).
Does NOT start full training.

Usage:
    python scripts/smoke_test_p1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from src.schemas import Sample
from src.training_data import (
    AssistantOnlyCollator,
    AssistantOnlyDataset,
    build_assistant_only_features,
)


def main() -> int:
    print("=" * 60)
    print("P1 Smoke Test: Assistant-only Loss 1-batch forward/backward")
    print("=" * 60)

    model_path = _ROOT / "models" / "Qwen3-0.6B"
    train_file = _ROOT / "data" / "curriculum" / "easy" / "train.jsonl"

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1
    if not train_file.exists():
        print(f"ERROR: train file not found: {train_file}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {model_path}")
    print(f"Data:   {train_file}")

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"  pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

    # ------------------------------------------------------------------
    # 2. Load 2 samples from curriculum easy train
    # ------------------------------------------------------------------
    print("\n[2/6] Loading 2 samples from curriculum easy train...")
    records: list[dict] = []
    with train_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= 2:
                break
    assert len(records) >= 2, f"need >=2 samples, got {len(records)}"

    # Validate via Pydantic
    for i, rec in enumerate(records):
        s = Sample.model_validate(rec)
        print(f"  sample {i}: id={s.sample_id}, family={s.family_id}, "
              f"task_type={s.task_type}, difficulty={s.difficulty}")

    # ------------------------------------------------------------------
    # 3. Build Assistant-only features + verify label mask
    # ------------------------------------------------------------------
    print("\n[3/6] Building Assistant-only features (verifying label mask)...")
    max_seq_length = 512
    dataset = AssistantOnlyDataset(records, tokenizer, max_seq_length, "preserve_assistant")
    collator = AssistantOnlyCollator(tokenizer=tokenizer)

    for i in range(len(records)):
        feat = dataset[i]
        prompt_len = feat["prompt_len"]
        assistant_len = feat["assistant_len"]
        full_len = feat["full_len"]
        status = feat["assistant_status"]
        labels = feat["labels"]

        # Verify prompt tokens are all -100
        prompt_masked = all(l == -100 for l in labels[:prompt_len])
        # Verify assistant has non-(-100) tokens
        assistant_supervised = sum(1 for l in labels[prompt_len:] if l != -100)
        # Verify total supervised
        total_supervised = sum(1 for l in labels if l != -100)

        print(f"  sample {i}: prompt={prompt_len}, assistant={assistant_len}, "
              f"full={full_len}, status={status}")
        print(f"    prompt all -100: {prompt_masked}")
        print(f"    assistant supervised tokens: {assistant_supervised}")
        print(f"    total supervised tokens: {total_supervised}")

        assert prompt_masked, f"sample {i}: prompt tokens not all -100"
        assert assistant_supervised > 0, f"sample {i}: assistant has no supervised tokens"
        assert status == "intact", f"sample {i}: assistant_status={status} (expected intact)"

    # ------------------------------------------------------------------
    # 4. Load model + apply LoRA (independent mode)
    # ------------------------------------------------------------------
    print("\n[4/6] Loading model + LoRA (independent, rank=32)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ------------------------------------------------------------------
    # 5. 1-batch forward + backward
    # ------------------------------------------------------------------
    print("\n[5/6] 1-batch forward + backward...")
    batch = collator([dataset[0], dataset[1]])
    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"  batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  batch labels shape: {batch['labels'].shape}")

    # Verify padding labels are -100
    amask = batch["attention_mask"]
    labels = batch["labels"]
    pad_mask = amask == 0
    pad_labels = labels[pad_mask]
    if pad_labels.numel() > 0:
        all_pad_neg = bool((pad_labels == -100).all().item())
        print(f"  padding labels all -100: {all_pad_neg}")
        assert all_pad_neg, "padding labels must be -100"
    else:
        print(f"  no padding in this batch (sequences same length)")

    outputs = model(**batch)
    loss = outputs.loss
    print(f"  loss = {loss.item():.6f}")
    print(f"  loss is finite: {torch.isfinite(loss).item()}")

    assert torch.isfinite(loss).item(), "loss must be finite"
    assert not torch.isnan(loss).item(), "loss must not be NaN"

    # Backward
    loss.backward()
    print(f"  backward OK")

    # Check gradients exist on LoRA params
    has_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                print(f"  grad norm ({name[:40]}...): {grad_norm:.6f}")
                break
    assert has_grad, "no LoRA parameter has gradients after backward"

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n[6/6] Summary")
    print(f"  Pipeline:         OK")
    print(f"  Assistant-only:   OK (system/user/pad = -100, assistant supervised)")
    print(f"  Forward:          OK (loss={loss.item():.6f})")
    print(f"  Backward:         OK (gradients on LoRA params)")
    print(f"  Device:           {device}")
    if device == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(0) / (1024**2)
        print(f"  Peak GPU memory:  {peak_mib:.0f} MiB")
    else:
        print(f"  Peak GPU memory:  N/A (CPU mode)")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
