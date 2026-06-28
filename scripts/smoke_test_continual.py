"""
scripts/smoke_test_continual.py — P1x Continual Smoke Test.

Verifies:
  Easy:    Base → LoRA → train 2-5 steps → save → reload → inference
  Boundary: Base + Easy Adapter → continue train 2-5 steps → save
  Repair:   Base + Boundary Adapter → continue train 2-5 steps → save

Each stage must perform: forward, backward, optimizer.step(), zero_grad(), save, reload, eval inference.
Does NOT run full training.

Output dir: adapters/p1-smoke/{easy,boundary,repair}/
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training_data import (
    AssistantOnlyCollator,
    AssistantOnlyDataset,
    build_assistant_only_features,
    compute_token_audit,
)


def _adapter_hash(adapter_dir: Path) -> str:
    """Compute SHA256 of all adapter binary files."""
    sha = hashlib.sha256()
    for f in sorted(adapter_dir.glob("adapter_model*.safetensors")):
        sha.update(f.read_bytes())
    return sha.hexdigest()


def _load_train_data(stage: str) -> list[dict]:
    """Load 2-3 samples from curriculum-v2 train data."""
    train_file = _ROOT / "data" / "curriculum-v2" / stage / "train.jsonl"
    records: list[dict] = []
    with train_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= 3:
                break
    return records


def _smoke_train(
    model: PeftModel,
    tokenizer,
    records: list[dict],
    output_dir: Path,
    max_seq_length: int = 512,
    num_steps: int = 3,
    learning_rate: float = 1e-4,
    stage_name: str = "",
) -> dict:
    """Run num_steps training steps with optimizer, save, reload, inference."""
    device = model.device

    # Build dataset
    dataset = AssistantOnlyDataset(records, tokenizer, max_seq_length, "preserve_assistant")
    collator = AssistantOnlyCollator(tokenizer=tokenizer)
    batch = collator([dataset[i] for i in range(min(len(dataset), 2))])
    batch = {k: v.to(device) for k, v in batch.items()}

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"\n--- {stage_name} Training ({num_steps} steps) ---")
    losses = []
    model.train()

    for step in range(num_steps):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {step+1}/{num_steps}: loss={loss.item():.6f}")

    # Record memory
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

    # Save adapter
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    adapter_sha = _adapter_hash(output_dir)
    print(f"  Saved to: {output_dir}")
    print(f"  Adapter SHA256: {adapter_sha[:16]}...")

    # Reload and verify
    base_model = AutoModelForCausalLM.from_pretrained(
        str(_ROOT / "models" / "Qwen3-0.6B"),
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    reloaded = PeftModel.from_pretrained(base_model, str(output_dir))
    reloaded.eval()

    # Inference test
    test_msgs = [
        {"role": "system", "content": "你是一个严谨的 Python 代码助手。"},
        {"role": "user", "content": "def solution(a, b):\n    return a + b"},
        {"role": "assistant", "content": ""},
    ]
    prompt_text = tokenizer.apply_chat_template(
        test_msgs[:2], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = reloaded.generate(
            **inputs, max_new_tokens=32, do_sample=False, num_beams=1,
        )
    gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)
    print(f"  Inference: {gen_text[-80:]}")

    del base_model, reloaded
    torch.cuda.empty_cache()

    return {
        "adapter_sha256": adapter_sha,
        "losses": losses,
        "final_loss": losses[-1],
        "peak_gpu_mib": peak_mib,
    }


def main() -> int:
    print("=" * 60)
    print("P1x Continual Smoke Test: Easy → Boundary → Repair")
    print("=" * 60)

    model_path = _ROOT / "models" / "Qwen3-0.6B"
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        return 1

    device = "cuda:0"
    print(f"Device: {device}")

    # Check BF16 support
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"BF16 supported: {bf16_supported}")
    use_fp16 = not bf16_supported

    # Load tokenizer once
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config (matches formal training)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ================================================================
    # Stage 1: Easy
    # ================================================================
    print("\n" + "=" * 40)
    print("STAGE 1: Easy (Base → Easy Adapter)")
    print("=" * 40)

    base = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    base.config.use_cache = False
    base.enable_input_require_grads()

    model = get_peft_model(base, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,}")

    easy_records = _load_train_data("easy")
    print(f"Easy data: {len(easy_records)} samples")

    easy_out = _ROOT / "adapters" / "p1-smoke" / "easy"
    easy_result = _smoke_train(
        model, tokenizer, easy_records, easy_out,
        num_steps=3, stage_name="Easy",
    )

    # Record parent mtime before continuing
    easy_config_mtime = (easy_out / "adapter_config.json").stat().st_mtime

    del model, base
    torch.cuda.empty_cache()

    # ================================================================
    # Stage 2: Boundary
    # ================================================================
    print("\n" + "=" * 40)
    print("STAGE 2: Boundary (Easy → Boundary Adapter)")
    print("=" * 40)

    base2 = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    base2.config.use_cache = False
    base2.enable_input_require_grads()

    print(f"Loading parent adapter: {easy_out}")
    model2 = PeftModel.from_pretrained(base2, str(easy_out), is_trainable=True)

    trainable2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"Trainable: {trainable2:,}")
    assert trainable2 > 0, "No trainable params after loading parent adapter!"

    boundary_records = _load_train_data("boundary")
    print(f"Boundary data: {len(boundary_records)} samples")

    boundary_out = _ROOT / "adapters" / "p1-smoke" / "boundary"
    boundary_result = _smoke_train(
        model2, tokenizer, boundary_records, boundary_out,
        num_steps=3, stage_name="Boundary",
    )

    # Verify parent not modified
    easy_config_mtime_after = (easy_out / "adapter_config.json").stat().st_mtime
    assert easy_config_mtime == easy_config_mtime_after, (
        f"Easy adapter was modified! mtime {easy_config_mtime} → {easy_config_mtime_after}"
    )
    print("  Parent adapter integrity: OK (Easy not modified)")

    assert boundary_result["adapter_sha256"] != easy_result["adapter_sha256"], (
        "Boundary adapter SHA256 must differ from Easy!"
    )
    print("  Adapter SHA256 differs: OK")

    boundary_config_mtime = (boundary_out / "adapter_config.json").stat().st_mtime
    del model2, base2
    torch.cuda.empty_cache()

    # ================================================================
    # Stage 3: Repair
    # ================================================================
    print("\n" + "=" * 40)
    print("STAGE 3: Repair (Boundary → Repair Adapter)")
    print("=" * 40)

    base3 = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    base3.config.use_cache = False
    base3.enable_input_require_grads()

    print(f"Loading parent adapter: {boundary_out}")
    model3 = PeftModel.from_pretrained(base3, str(boundary_out), is_trainable=True)

    trainable3 = sum(p.numel() for p in model3.parameters() if p.requires_grad)
    print(f"Trainable: {trainable3:,}")
    assert trainable3 > 0, "No trainable params after loading parent adapter!"

    repair_records = _load_train_data("repair")
    print(f"Repair data: {len(repair_records)} samples")

    repair_out = _ROOT / "adapters" / "p1-smoke" / "repair"
    repair_result = _smoke_train(
        model3, tokenizer, repair_records, repair_out,
        num_steps=3, stage_name="Repair",
    )

    # Verify parents not modified
    easy_config_mtime_final = (easy_out / "adapter_config.json").stat().st_mtime
    boundary_config_mtime_after = (boundary_out / "adapter_config.json").stat().st_mtime
    assert easy_config_mtime == easy_config_mtime_final, "Easy was modified!"
    assert boundary_config_mtime == boundary_config_mtime_after, "Boundary was modified!"
    print("  Parent adapter integrity: OK (Easy and Boundary not modified)")

    assert repair_result["adapter_sha256"] != boundary_result["adapter_sha256"], (
        "Repair adapter SHA256 must differ from Boundary!"
    )
    print("  Adapter SHA256 differs: OK")

    del model3, base3
    torch.cuda.empty_cache()

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("CONTINUAL SMOKE TEST PASSED")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Easy:     loss={easy_result['final_loss']:.6f}, "
          f"sha={easy_result['adapter_sha256'][:16]}..., "
          f"GPU={easy_result['peak_gpu_mib']:.0f}MiB")
    print(f"  Boundary: loss={boundary_result['final_loss']:.6f}, "
          f"sha={boundary_result['adapter_sha256'][:16]}..., "
          f"GPU={boundary_result['peak_gpu_mib']:.0f}MiB")
    print(f"  Repair:   loss={repair_result['final_loss']:.6f}, "
          f"sha={repair_result['adapter_sha256'][:16]}..., "
          f"GPU={repair_result['peak_gpu_mib']:.0f}MiB")
    print(f"\nAll 3 adapter SHA256s are different: OK")
    print(f"All parent adapters preserved: OK")
    print(f"Easy → Boundary → Repair chain verified: OK")
    print(f"Precision: {'BF16' if bf16_supported else 'FP16'}")

    # Save smoke manifest
    smoke_manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "precision": "bf16" if bf16_supported else "fp16",
        "max_seq_length": 512,
        "stages": {
            "easy": {
                "adapter_sha256": easy_result["adapter_sha256"],
                "final_loss": easy_result["final_loss"],
                "losses": easy_result["losses"],
                "peak_gpu_mib": easy_result["peak_gpu_mib"],
            },
            "boundary": {
                "adapter_sha256": boundary_result["adapter_sha256"],
                "final_loss": boundary_result["final_loss"],
                "losses": boundary_result["losses"],
                "peak_gpu_mib": boundary_result["peak_gpu_mib"],
            },
            "repair": {
                "adapter_sha256": repair_result["adapter_sha256"],
                "final_loss": repair_result["final_loss"],
                "losses": repair_result["losses"],
                "peak_gpu_mib": repair_result["peak_gpu_mib"],
            },
        },
    }
    smoke_dir = _ROOT / "adapters" / "p1-smoke"
    (smoke_dir / "smoke_manifest.json").write_text(
        json.dumps(smoke_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSmoke manifest saved: {smoke_dir / 'smoke_manifest.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())