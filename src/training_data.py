"""
src/training_data.py -- Assistant-only Loss data processing for P1.

Provides:
  - build_assistant_only_features(): tokenize ChatML with assistant-only labels
  - AssistantOnlyCollator: dynamic padding collator that preserves assistant masks
  - compute_token_audit(): per-sample token length audit

Label mask policy:
  System Token     -> -100
  User Token       -> -100
  Assistant Token  -> input_ids
  Padding Token    -> -100
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Tokenization: build input_ids + labels with assistant-only mask
# ---------------------------------------------------------------------------

def build_assistant_only_features(
    messages: list[dict],
    tokenizer: Any,
    max_seq_length: int = 512,
    *,
    truncation_policy: str = "preserve_assistant",
) -> dict:
    """Tokenize a ChatML conversation and build assistant-only labels.

    Parameters
    ----------
    messages : list[dict]
        ChatML messages: [{"role": "system", ...}, {"role": "user", ...},
        {"role": "assistant", ...}]
    tokenizer : Any
        HuggingFace tokenizer with apply_chat_template support.
    max_seq_length : int
        Maximum sequence length.
    truncation_policy : str
        "preserve_assistant" — never truncate assistant tokens; if the full
        sequence exceeds max_seq_length, truncate from the prompt side.

    Returns
    -------
    dict with keys:
        input_ids : list[int]
        attention_mask : list[int]
        labels : list[int]  (-100 for system/user/pad, input_ids for assistant)
        prompt_len : int
        assistant_len : int
        full_len : int
        truncated : bool
        assistant_status : str  ("intact" | "partial" | "lost" | "target_too_long")
    """
    # Split messages into prompt (system + user) and full (system + user + assistant)
    if len(messages) < 3:
        raise ValueError(
            f"Need at least 3 messages (system, user, assistant), got {len(messages)}"
        )

    prompt_messages = messages[:-1]  # system + user
    # full_messages = messages  # system + user + assistant

    # Tokenize prompt (with generation prompt to get the assistant header).
    # Use tokenize=False + encode() for version-robust behavior: transformers 5.x
    # apply_chat_template(tokenize=True) returns a BatchEncoding, not a list[int].
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Tokenize full conversation
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    prompt_len = len(prompt_ids)
    full_len = len(full_ids)
    assistant_len = max(0, full_len - prompt_len)

    # Verify prefix alignment: prompt_ids must be a prefix of full_ids
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Prompt/full token prefix misalignment detected. "
            f"prompt_len={len(prompt_ids)}, full_len={len(full_ids)}. "
            f"Chat template may not be consistent between add_generation_prompt=True/False."
        )

    # Build labels: -100 for prompt, input_ids for assistant
    labels = list(full_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    truncated = False
    assistant_status = "intact"

    if full_len > max_seq_length:
        truncated = True

        if truncation_policy == "preserve_assistant":
            # If assistant alone exceeds max_seq_length, it's target_too_long
            if assistant_len >= max_seq_length:
                # Can't preserve assistant — mark as target_too_long
                # Still truncate from left (prompt side) as much as possible
                # Keep last max_seq_length tokens (assistant is at the end)
                input_ids = full_ids[-max_seq_length:]
                labels_trunc = labels[-max_seq_length:]
                attention_mask = [1] * max_seq_length
                assistant_status = "target_too_long"
                # assistant_len >= max_seq_length: prompt is completely lost.
                # This is an error state — the sample requires more tokens
                # than the model can handle even for the assistant alone.
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels_trunc,
                    "prompt_len": 0,
                    "assistant_len": assistant_len,
                    "full_len": full_len,
                    "truncated": truncated,
                    "assistant_status": assistant_status,
                }

            # Normal case: truncate prompt from left to fit assistant
            # Keep all assistant tokens, trim prompt tokens from the left
            keep_prompt = max_seq_length - assistant_len
            # Take last keep_prompt tokens of prompt + all assistant tokens
            input_ids = full_ids[prompt_len - keep_prompt:]
            labels_trunc = labels[prompt_len - keep_prompt:]
            attention_mask = [1] * len(input_ids)
            assistant_status = "intact"  # assistant fully preserved

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels_trunc,
                "prompt_len": keep_prompt,
                "assistant_len": assistant_len,
                "full_len": full_len,
                "truncated": truncated,
                "assistant_status": assistant_status,
            }
        else:
            # Default right truncation (not recommended)
            input_ids = full_ids[:max_seq_length]
            labels_trunc = labels[:max_seq_length]
            attention_mask = [1] * max_seq_length

            # Check assistant status after truncation
            if prompt_len >= max_seq_length:
                assistant_status = "lost"
            elif prompt_len + assistant_len > max_seq_length:
                assistant_status = "partial"
            else:
                assistant_status = "intact"

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels_trunc,
                "prompt_len": min(prompt_len, max_seq_length),
                "assistant_len": assistant_len,
                "full_len": full_len,
                "truncated": truncated,
                "assistant_status": assistant_status,
            }

    # No truncation needed
    input_ids = list(full_ids)
    attention_mask = [1] * full_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_len": prompt_len,
        "assistant_len": assistant_len,
        "full_len": full_len,
        "truncated": False,
        "assistant_status": "intact",
    }


# ---------------------------------------------------------------------------
# Custom Data Collator
# ---------------------------------------------------------------------------

@dataclass
class AssistantOnlyCollator:
    """Dynamic padding collator that preserves assistant-only label masks.

    - Pads input_ids with tokenizer.pad_token_id
    - Pads attention_mask with 0
    - Pads labels with -100
    - Does NOT recompute labels — preserves the assistant-only mask
      built by build_assistant_only_features()
    """

    tokenizer: Any
    padding: str = "longest"  # "longest" or "max_length"
    max_length: int | None = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict]) -> dict:
        # Extract sequences
        input_ids_list = [f["input_ids"] for f in features]
        attention_mask_list = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        labels_list = [f["labels"] for f in features]

        # Determine padding length
        if self.padding == "max_length" and self.max_length:
            max_len = self.max_length
        else:
            max_len = max(len(ids) for ids in input_ids_list)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        # Pad sequences
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, amask, labels in zip(input_ids_list, attention_mask_list, labels_list):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_mask.append(amask + [0] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Dataset adapter for Assistant-only Loss
# ---------------------------------------------------------------------------

class AssistantOnlyDataset(Dataset):
    """Wraps a list of ChatML message lists into an assistant-only dataset.

    Each item returns:
        input_ids, attention_mask, labels, prompt_len, assistant_len,
        full_len, truncated, assistant_status
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer: Any,
        max_seq_length: int = 512,
        truncation_policy: str = "preserve_assistant",
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.truncation_policy = truncation_policy
        self._cache: list[dict | None] = [None] * len(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        if self._cache[idx] is not None:
            return self._cache[idx]

        record = self.records[idx]
        messages = record.get("messages")
        if messages is None:
            # Raw Sample format — convert to ChatML
            from src.schemas import Sample, to_chatml
            sample = Sample.model_validate(record)
            messages = to_chatml(sample)["messages"]

        features = build_assistant_only_features(
            messages,
            self.tokenizer,
            max_seq_length=self.max_seq_length,
            truncation_policy=self.truncation_policy,
        )
        self._cache[idx] = features
        return features


# ---------------------------------------------------------------------------
# Token audit helper (used by audit_token_lengths.py and smoke tests)
# ---------------------------------------------------------------------------

def compute_token_audit(
    records: list[dict],
    tokenizer: Any,
    max_seq_length: int = 512,
) -> dict:
    """Compute token length audit for a list of records.

    Returns a dict with:
        total, not_truncated, truncated,
        assistant_intact, assistant_partial, assistant_lost, target_too_long,
        prompt_lens, assistant_lens, full_lens
    """
    prompt_lens: list[int] = []
    assistant_lens: list[int] = []
    full_lens: list[int] = []
    truncated_count = 0
    intact_count = 0
    partial_count = 0
    lost_count = 0
    target_too_long_count = 0

    for record in records:
        messages = record.get("messages")
        if messages is None:
            from src.schemas import Sample, to_chatml
            sample = Sample.model_validate(record)
            messages = to_chatml(sample)["messages"]

        features = build_assistant_only_features(
            messages, tokenizer, max_seq_length=max_seq_length,
        )
        prompt_lens.append(features["prompt_len"])
        assistant_lens.append(features["assistant_len"])
        full_lens.append(features["full_len"])

        if features["truncated"]:
            truncated_count += 1
        else:
            pass

        status = features["assistant_status"]
        if status == "intact":
            intact_count += 1
        elif status == "partial":
            partial_count += 1
        elif status == "lost":
            lost_count += 1
        elif status == "target_too_long":
            target_too_long_count += 1

    return {
        "total": len(records),
        "not_truncated": len(records) - truncated_count,
        "truncated": truncated_count,
        "assistant_intact": intact_count,
        "assistant_partial": partial_count,
        "assistant_lost": lost_count,
        "target_too_long": target_too_long_count,
        "prompt_lens": prompt_lens,
        "assistant_lens": assistant_lens,
        "full_lens": full_lens,
    }
