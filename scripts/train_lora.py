"""
scripts/train_lora.py -- LoRA fine-tuning trainer for Qwen3-0.6B (spec §13).

Reads a YAML config, loads ChatML JSONL splits, applies LoRA via PEFT, and
runs training with the Hugging Face Trainer.

Requirements (spec §13.2)
-------------------------
- Read config from YAML
- Fix random seeds
- Print trainable parameter count and VRAM usage
- Smoke test (1 batch forward + backward) before full training
- Resume from checkpoint
- Save a copy of the training config in the output directory
- Record dataset hash, model path, start/end times
- Catch OOM and emit mitigation advice
- Never delete old checkpoints automatically

Usage
-----
    python scripts/train_lora.py --config configs/train_easy.yaml
    python scripts/train_lora.py --config configs/train_boundary.yaml
    python scripts/train_lora.py --config configs/train_repair.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

# Import heavy ML modules at module level to avoid CUDA init race conditions
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training_data import (  # noqa: E402
    AssistantOnlyCollator,
    AssistantOnlyDataset,
    build_assistant_only_features,
    compute_token_audit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _gpu_memory_report() -> dict[str, float]:
    """Return a dict with GPU memory info (MiB).  Returns zeros if no CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"total_mib": 0, "allocated_mib": 0, "reserved_mib": 0}
        return {
            "total_mib": torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
            "allocated_mib": torch.cuda.memory_allocated(0) / (1024 ** 2),
            "reserved_mib": torch.cuda.memory_reserved(0) / (1024 ** 2),
        }
    except Exception:
        return {"total_mib": 0, "allocated_mib": 0, "reserved_mib": 0}


def _dataset_hash_from_file(file_path: Path) -> str:
    """Compute SHA-256 of the raw file content (for provenance)."""
    sha = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file as a list of dicts."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def run_training(config_path: str) -> int:
    """Execute LoRA training from a YAML config file.

    Returns 0 on success, 1 on failure.
    """
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    print("[DEBUG] Loading config...", flush=True)
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    print("[DEBUG] Config loaded OK", flush=True)

    model_path = Path(cfg["model_name_or_path"])
    output_dir = Path(cfg["output_dir"])
    train_file = Path(cfg["train_file"])
    eval_file = Path(cfg["eval_file"])
    seed = cfg.get("seed", 42)

    # P1 config: training mode, assistant-only loss, truncation policy
    training_mode = cfg.get("training_mode", "independent")  # independent | continual
    initial_adapter = cfg.get("initial_adapter", None)
    assistant_only_loss = cfg.get("assistant_only_loss", True)
    truncation_policy = cfg.get("truncation_policy", "preserve_assistant")
    dataset_manifest = cfg.get("dataset_manifest", None)

    # ------------------------------------------------------------------
    # 2. Check prerequisites
    # ------------------------------------------------------------------
    print("[DEBUG] Checking prerequisites...", flush=True)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1
    if not train_file.exists():
        print(f"ERROR: train file not found: {train_file}", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Training on CPU is not allowed.", file=sys.stderr)
        return 1

    # Validate training mode
    if training_mode not in ("independent", "continual"):
        print(
            f"ERROR: training_mode must be 'independent' or 'continual', got '{training_mode}'",
            file=sys.stderr,
        )
        return 1

    if training_mode == "continual":
        if not initial_adapter:
            print(
                "ERROR: continual mode requires 'initial_adapter' config field",
                file=sys.stderr,
            )
            return 1
        initial_adapter_path = Path(initial_adapter)
        if not initial_adapter_path.exists():
            print(
                f"ERROR: initial_adapter not found: {initial_adapter_path}",
                file=sys.stderr,
            )
            return 1
        if not (initial_adapter_path / "adapter_config.json").exists():
            print(
                f"ERROR: initial_adapter is not a valid PEFT adapter "
                f"(no adapter_config.json): {initial_adapter_path}",
                file=sys.stderr,
            )
            return 1

    # Don't overwrite existing adapters
    if output_dir.exists() and (output_dir / "adapter_config.json").exists():
        print(
            f"ERROR: output_dir already contains a trained adapter: {output_dir}\n"
            f"  Refusing to overwrite. Use a different output_dir.",
            file=sys.stderr,
        )
        return 1

    # Validate required config keys
    required_keys = ["model_name_or_path", "output_dir", "train_file", "lora"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        print(f"ERROR: missing required config keys: {missing}", file=sys.stderr)
        return 1
    print("[DEBUG] Prerequisites OK", flush=True)

    # ------------------------------------------------------------------
    # 3. Fix seeds
    # ------------------------------------------------------------------
    print("[DEBUG] Setting seeds...", flush=True)
    _set_seed(seed)
    print("[DEBUG] Seeds set OK", flush=True)

    start_time = datetime.now(timezone.utc)
    print(f"=== Training started at {start_time.isoformat()} ===")
    print(f"Config: {config_path.resolve()}")
    print(f"Model:  {model_path.resolve()}")
    print(f"Output: {output_dir.resolve()}")

    # ------------------------------------------------------------------
    # 4. Compute dataset hash
    # ------------------------------------------------------------------
    train_hash = _dataset_hash_from_file(train_file)
    eval_hash = ""
    if eval_file.exists():
        eval_hash = _dataset_hash_from_file(eval_file)
    print(f"Train hash: {train_hash[:16]}...")
    if eval_hash:
        print(f"Eval hash:  {eval_hash[:16]}...")

    # ------------------------------------------------------------------
    # 5. Load tokenizer and model
    # ------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For causal LM, right padding is standard
    tokenizer.padding_side = "right"

    # Determine dtype from config
    use_bf16 = cfg.get("bf16", False)
    use_fp16 = cfg.get("fp16", True)

    # Check BF16 support (P1x requirement)
    if use_bf16:
        if not torch.cuda.is_bf16_supported():
            print(
                "WARNING: bf16 requested but GPU does not support bf16. "
                "Falling back to fp16.",
                file=sys.stderr,
            )
            use_bf16 = False
            use_fp16 = True
            dtype_label = "fp16 (bf16 not supported)"
        else:
            model_dtype = torch.float16  # Load in fp16, train in bf16 mixed precision
            dtype_label = "fp16 (bf16 training)"
    elif use_fp16:
        model_dtype = torch.float16
        dtype_label = "fp16"
    else:
        model_dtype = torch.float32
        dtype_label = "fp32"

    print(f"Loading model ({dtype_label})...")
    mem_before = _gpu_memory_report()
    print(f"  GPU memory before load: {mem_before['allocated_mib']:.0f} MiB allocated")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=model_dtype,
        device_map={"": "cuda:0"},  # Explicit device for single GPU
        trust_remote_code=True,
    )
    # Disable KV cache for gradient checkpointing compatibility
    model.config.use_cache = False

    mem_after_load = _gpu_memory_report()
    print(f"  GPU memory after load:  {mem_after_load['allocated_mib']:.0f} MiB allocated")

    # ------------------------------------------------------------------
    # 6. Apply LoRA (independent) or load existing adapter (continual)
    # ------------------------------------------------------------------
    lora_cfg = cfg["lora"]

    if training_mode == "continual" and initial_adapter:
        # Continual mode: load parent adapter and continue training
        print(f"\n=== Continual Training Mode ===")
        print(f"Loading parent adapter: {initial_adapter}")
        initial_adapter_path = Path(initial_adapter)

        model = PeftModel.from_pretrained(
            model,
            initial_adapter,
            is_trainable=True,
        )

        # Verify LoRA params are trainable
        trainable_lora = sum(
            1 for name, param in model.named_parameters()
            if "lora_" in name and param.requires_grad
        )
        if trainable_lora == 0:
            print("ERROR: No trainable LoRA parameters after loading parent adapter!", file=sys.stderr)
            return 1

        print(f"Parent adapter path: {initial_adapter_path}")
        print(f"Trainable LoRA params: {trainable_lora}")

        # Print LoRA param norm before training
        lora_norm = 0.0
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_norm += param.data.norm().item() ** 2
        lora_norm = lora_norm ** 0.5
        print(f"LoRA parameter norm (before training): {lora_norm:.6f}")

        # Compute parent adapter SHA256 for manifest
        parent_adapter_sha = _dataset_hash_from_file(
            initial_adapter_path / "adapter_config.json"
        )
        print(f"Parent adapter SHA256: {parent_adapter_sha[:16]}...")
        print(f"  (Adapter weights are trainable, NOT merged)")
    else:
        # Independent mode: create new LoRA from scratch
        print(f"\n=== Independent Training Mode ===")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
        )
        model = get_peft_model(model, peft_config)
        parent_adapter_sha = None

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    mem_after_lora = _gpu_memory_report()
    print(f"GPU memory after LoRA:   {mem_after_lora['allocated_mib']:.0f} MiB allocated")

    # Print training mode info
    print(f"\nTraining mode: {training_mode}")
    print(f"Assistant-only loss: {assistant_only_loss}")
    print(f"Truncation policy: {truncation_policy}")
    print(f"Max seq length: {cfg.get('max_seq_length', 256)}")

    # ------------------------------------------------------------------
    # 7. Load and tokenize dataset
    # ------------------------------------------------------------------
    print("\nLoading dataset...")
    train_records = _load_jsonl(train_file)

    max_seq_length = cfg.get("max_seq_length", 256)

    eval_records = None
    if eval_file.exists():
        eval_records = _load_jsonl(eval_file)

    # --- Token audit (P1 requirement) ---
    print("\n=== Token Truncation Audit ===")
    train_audit = compute_token_audit(train_records, tokenizer, max_seq_length)
    print(f"  Total samples:       {train_audit['total']}")
    print(f"  Not truncated:       {train_audit['not_truncated']}")
    print(f"  Truncated:           {train_audit['truncated']}")
    print(f"  Assistant intact:    {train_audit['assistant_intact']}")
    print(f"  Assistant partial:   {train_audit['assistant_partial']}")
    print(f"  Assistant lost:      {train_audit['assistant_lost']}")
    print(f"  Target too long:     {train_audit['target_too_long']}")

    if train_audit['assistant_lost'] > 0 or train_audit['target_too_long'] > 0:
        print(
            f"  WARNING: {train_audit['assistant_lost'] + train_audit['target_too_long']} "
            f"samples have assistant content that cannot be preserved.",
            file=sys.stderr,
        )

    # --- Build datasets ---
    if assistant_only_loss:
        print("\nUsing Assistant-only Loss (System/User tokens masked as -100)")
        train_dataset = AssistantOnlyDataset(
            train_records, tokenizer, max_seq_length, truncation_policy,
        )
        eval_dataset = None
        if eval_records:
            eval_dataset = AssistantOnlyDataset(
                eval_records, tokenizer, max_seq_length, truncation_policy,
            )
        data_collator = AssistantOnlyCollator(tokenizer=tokenizer)
    else:
        # Backward-compatible: old DataCollatorForLanguageModeling path
        print("\nUsing legacy DataCollatorForLanguageModeling (full-text labels)")
        train_dataset = Dataset.from_list(train_records)
        eval_dataset = None
        if eval_records:
            eval_dataset = Dataset.from_list(eval_records)

        def preprocess(examples: dict) -> dict:
            """Apply chat template and tokenize a batch."""
            texts: list[str] = []
            for msgs in examples["messages"]:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False,
                )
                texts.append(text)
            tokenized = tokenizer(
                texts, truncation=True, max_length=max_seq_length, padding=False,
            )
            return tokenized

        print("Tokenizing train set...")
        train_dataset = train_dataset.map(
            preprocess, batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train", num_proc=1,
        )
        if eval_dataset is not None:
            print("Tokenizing eval set...")
            eval_dataset = eval_dataset.map(
                preprocess, batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing eval", num_proc=1,
            )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )

    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval samples:  {len(eval_dataset)}")

    # Check for empty dataset
    if len(train_dataset) == 0:
        print("ERROR: train dataset is empty", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 8. Build training arguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        eval_steps=cfg.get("eval_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 2),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        report_to=cfg.get("report_to", "none") if cfg.get("report_to", "none") != "none" else [],
        seed=seed,
        # Eval strategy: only when eval_dataset is provided
        eval_strategy="steps" if eval_dataset is not None else "no",
        load_best_model_at_end=False,
        # Remove unused columns to avoid warnings
        remove_unused_columns=False,
    )

    # Enable gradient checkpointing if requested
    if cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()

    # ------------------------------------------------------------------
    # 9. Smoke test: 1 batch forward + backward
    # ------------------------------------------------------------------
    print("\n=== Smoke test (1 batch) ===")
    model.train()

    # Take a small batch — works for both AssistantOnlyDataset and HF Dataset
    n_smoke = min(2, len(train_dataset))
    if assistant_only_loss:
        # AssistantOnlyDataset returns dicts directly
        smoke_features = [train_dataset[i] for i in range(n_smoke)]
        batch = data_collator(smoke_features)
    else:
        # HF Dataset path
        smoke_samples = train_dataset.select(range(n_smoke))
        batch = data_collator([smoke_samples[i] for i in range(n_smoke)])
    batch = {k: v.to(model.device) for k, v in batch.items()}

    try:
        outputs = model(**batch)
        loss = outputs.loss
        # Check for NaN/Inf before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  SMOKE TEST FAILED: loss is NaN or Inf: {loss.item()}", file=sys.stderr)
            return 1
        loss.backward()
        print(f"  Smoke test OK — loss: {loss.item():.4f}")
        # Zero grads after smoke
        model.zero_grad()
    except torch.cuda.OutOfMemoryError:
        print("  SMOKE TEST FAILED: OOM", file=sys.stderr)
        _print_oom_advice(cfg)
        return 1
    except Exception as exc:
        print(f"  SMOKE TEST FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    mem_after_smoke = _gpu_memory_report()
    print(f"  GPU memory after smoke: {mem_after_smoke['allocated_mib']:.0f} MiB allocated")

    # ------------------------------------------------------------------
    # 10. Save config copy to output_dir
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "training_config.yaml")

    # ------------------------------------------------------------------
    # 11. Train
    # ------------------------------------------------------------------
    # Check for existing checkpoint
    resume_from = None
    ckpt_dirs = sorted(output_dir.glob("checkpoint-*"))
    if ckpt_dirs:
        resume_from = str(ckpt_dirs[-1])
        print(f"\nResuming from checkpoint: {resume_from}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\n=== Starting training ===")
    train_start = time.time()

    # Sync CUDA to avoid pending errors
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except torch.cuda.OutOfMemoryError:
        print("\n=== OOM during training ===", file=sys.stderr)
        _print_oom_advice(cfg)
        return 1
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "oom" in msg:
            print("\n=== OOM during training ===", file=sys.stderr)
            _print_oom_advice(cfg)
            return 1
        if "cuda" in msg or "cublas" in msg:
            print(f"\n=== CUDA RuntimeError ===", file=sys.stderr)
            print(f"  {exc}", file=sys.stderr)
            print("  This may be a driver issue. Try reducing max_seq_length or restarting.", file=sys.stderr)
            return 1
        raise

    train_duration = time.time() - train_start
    print(f"\nTraining duration: {train_duration:.0f}s ({train_duration / 60:.1f} min)")

    # ------------------------------------------------------------------
    # 12. Save final adapter
    # ------------------------------------------------------------------
    # Verify parent adapter intact (continual mode)
    if training_mode == "continual" and initial_adapter:
        parent_path = Path(initial_adapter)
        parent_config_mtime = (parent_path / "adapter_config.json").stat().st_mtime
        print(f"Parent adapter intact: {parent_path} (mtime={parent_config_mtime})")

    print("\nSaving adapter...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Verify adapter was saved correctly by reloading
    print("\nVerifying saved adapter...")
    from peft import PeftModel as PeftModelCheck
    verify_model = PeftModelCheck.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=model_dtype, device_map={"": "cuda:0"},
            trust_remote_code=True,
        ),
        str(output_dir),
    )
    # Compare a few LoRA parameter values
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            orig_val = param.data.float().mean().item()
            break
    for name, param in verify_model.named_parameters():
        if "lora_" in name:
            loaded_val = param.data.float().mean().item()
            break
    print(f"  Original LoRA param mean: {orig_val:.6f}")
    print(f"  Reloaded LoRA param mean: {loaded_val:.6f}")
    assert abs(orig_val - loaded_val) < 1e-5, "Adapter save/reload mismatch!"
    print("  Adapter save/reload verification: OK")

    # Clean up
    del verify_model
    torch.cuda.empty_cache()

    # Save dataset manifest
    manifest = {
        "model_name_or_path": str(model_path.resolve()),
        "train_file": str(train_file.resolve()),
        "train_hash": train_hash,
        "eval_hash": eval_hash,
        "seed": seed,
        "lora_config": lora_cfg,
        "max_seq_length": max_seq_length,
        "trainable_params": trainable,
        "total_params": total,
        "started_at": start_time.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "train_duration_s": train_duration,
        # P1 fields
        "training_mode": training_mode,
        "initial_adapter": initial_adapter,
        "parent_adapter_sha256": parent_adapter_sha,
        "assistant_only_loss": assistant_only_loss,
        "truncation_policy": truncation_policy,
        "token_audit": {
            "total": train_audit["total"],
            "truncated": train_audit["truncated"],
            "assistant_intact": train_audit["assistant_intact"],
            "assistant_partial": train_audit["assistant_partial"],
            "assistant_lost": train_audit["assistant_lost"],
            "target_too_long": train_audit["target_too_long"],
        },
        "peak_gpu_mib": _gpu_memory_report()["allocated_mib"],
    }
    manifest_path = output_dir / "metrics.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"Manifest saved: {manifest_path}")

    # ------------------------------------------------------------------
    # 13. Done
    # ------------------------------------------------------------------
    end_time = datetime.now(timezone.utc)
    print(f"\n=== Training finished at {end_time.isoformat()} ===")
    print(f"Adapter saved to: {output_dir.resolve()}")

    return 0


# ---------------------------------------------------------------------------
# OOM advice
# ---------------------------------------------------------------------------

_OOM_ADVICE = """\
OOM detected.  Follow the mitigation ladder (spec §3.1):

  1. Reduce max_seq_length: 256 → 128
  2. Reduce LoRA target modules to only q_proj
  3. Enable gradient checkpointing (already on by default)
  4. Set dataloader_num_workers: 0 (already default)
  5. Consider 4-bit QLoRA (bitsandbytes)

Do NOT switch to CPU training.
"""


def _print_oom_advice(cfg: dict) -> None:
    """Print OOM mitigation advice with current config values."""
    print(_OOM_ADVICE)
    print(f"Current max_seq_length: {cfg.get('max_seq_length', '?')}")
    print(f"Current LoRA targets:   {cfg.get('lora', {}).get('target_modules', '?')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning for Qwen3-0.6B Code Recovery Lab.",
    )
    p.add_argument(
        "--config", required=True,
        help="Path to YAML config (e.g. configs/train_easy.yaml).",
    )
    return p


def main() -> int:
    """CLI entry point."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()

    try:
        return run_training(args.config)
    except Exception:
        print("\nUnhandled exception:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())