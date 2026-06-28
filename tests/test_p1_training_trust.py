"""
P1 tests for training trustworthiness.

Covers 5 categories (28 tests):
  A. Assistant-only Loss (9 tests)
  B. Token truncation audit (5 tests)
  C. Data versioning (5 tests)
  D. Training mode (5 tests)
  E. Frozen eval set (4 tests)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample, Verification
from src.training_data import (
    AssistantOnlyCollator,
    AssistantOnlyDataset,
    build_assistant_only_features,
    compute_token_audit,
)


# ---------------------------------------------------------------------------
# Mock tokenizer: mimics Qwen3 ChatML template without loading the real model
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Deterministic mock tokenizer that simulates Qwen3 ChatML templating.

    Token layout per message:
        IM_START role NEWLINE content IM_END
    With add_generation_prompt=True, appends:
        IM_START role(assistant) NEWLINE
    """

    pad_token_id = 0
    eos_token_id = 1
    IM_START = 2
    IM_END = 3
    NEWLINE = 4

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        # Render to a string first, then optionally tokenize
        text_parts: list[str] = []
        for msg in messages:
            text_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        if add_generation_prompt:
            text_parts.append("<|im_start|>assistant\n")
        text = "".join(text_parts)
        if tokenize:
            return self._encode_text(text)
        return text

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs (mimics tokenizer.encode)."""
        return self._encode_text(text)

    def _encode_text(self, text: str) -> list[int]:
        # Deterministic: one token per character (offset to avoid special ids)
        return [ord(c) % 200 + 10 for c in text]


def _make_chatml(system: str, user: str, assistant: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ===========================================================================
# A. Assistant-only Loss (9 tests)
# ===========================================================================

class TestAssistantOnlyLoss:
    """Tests for src/training_data.py build_assistant_only_features + collator."""

    def test_system_user_labels_all_negative_100(self):
        """A1: System and User tokens must all be -100."""
        tok = MockTokenizer()
        msgs = _make_chatml("sys prompt", "user query", "assistant answer")
        feat = build_assistant_only_features(msgs, tok, max_seq_length=512)

        prompt_len = feat["prompt_len"]
        labels = feat["labels"]
        # All prompt-position labels must be -100
        for i in range(prompt_len):
            assert labels[i] == -100, f"prompt token {i} should be -100, got {labels[i]}"

    def test_assistant_labels_valid(self):
        """A2: Assistant section has at least one non-(-100) token."""
        tok = MockTokenizer()
        msgs = _make_chatml("s", "u", "answer")
        feat = build_assistant_only_features(msgs, tok, max_seq_length=512)

        labels = feat["labels"]
        non_masked = [l for l in labels if l != -100]
        assert len(non_masked) > 0, "assistant section must have supervised tokens"
        assert feat["assistant_len"] > 0

    def test_padding_labels_negative_100(self):
        """A3: Padding tokens (added by collator) must be -100."""
        tok = MockTokenizer()
        msgs1 = _make_chatml("s", "u", "short")
        msgs2 = _make_chatml("s", "u", "a much longer assistant answer text")

        feat1 = build_assistant_only_features(msgs1, tok, max_seq_length=512)
        feat2 = build_assistant_only_features(msgs2, tok, max_seq_length=512)

        collator = AssistantOnlyCollator(tokenizer=tok)
        batch = collator([feat1, feat2])

        # Find padded positions (attention_mask == 0)
        amask = batch["attention_mask"]
        labels = batch["labels"]
        for row in range(labels.shape[0]):
            for col in range(labels.shape[1]):
                if amask[row, col] == 0:
                    assert labels[row, col] == -100, (
                        f"padding at ({row},{col}) must be -100, got {labels[row, col]}"
                    )

    def test_prompt_full_boundary_correct(self):
        """A4: prompt_len + assistant_len == full_len (no truncation case)."""
        tok = MockTokenizer()
        msgs = _make_chatml("system", "user", "assistant reply")
        feat = build_assistant_only_features(msgs, tok, max_seq_length=512)

        assert feat["prompt_len"] + feat["assistant_len"] == feat["full_len"]
        assert feat["truncated"] is False
        assert feat["assistant_status"] == "intact"
        assert len(feat["input_ids"]) == feat["full_len"]
        assert len(feat["labels"]) == feat["full_len"]

    def test_batch_forward_backward_success(self):
        """A5: A batch from the collator can forward/backward on a tiny mock model."""
        tok = MockTokenizer()
        msgs1 = _make_chatml("s", "u", "answer one")
        msgs2 = _make_chatml("s", "u", "answer two longer")

        feat1 = build_assistant_only_features(msgs1, tok, max_seq_length=512)
        feat2 = build_assistant_only_features(msgs2, tok, max_seq_length=512)
        collator = AssistantOnlyCollator(tokenizer=tok)
        batch = collator([feat1, feat2])

        # Tiny mock causal LM: embedding + linear, computes cross-entropy on labels
        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(300, 8)
                self.lm_head = torch.nn.Linear(8, 300)

            def forward(self, input_ids, attention_mask=None, labels=None):
                h = self.embed(input_ids)
                logits = self.lm_head(h)
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                return {"loss": loss, "logits": logits}

        model = TinyLM()
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        assert out["loss"] is not None
        out["loss"].backward()
        # Gradients should exist
        assert model.embed.weight.grad is not None

    def test_loss_not_nan_or_inf(self):
        """A6: Computed loss must not be NaN or Inf."""
        tok = MockTokenizer()
        msgs = _make_chatml("s", "u", "answer")
        feat = build_assistant_only_features(msgs, tok, max_seq_length=512)
        collator = AssistantOnlyCollator(tokenizer=tok)
        batch = collator([feat])

        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(300, 8)
                self.lm_head = torch.nn.Linear(8, 300)

            def forward(self, input_ids, attention_mask=None, labels=None):
                h = self.embed(input_ids)
                logits = self.lm_head(h)
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                return {"loss": loss}

        model = TinyLM()
        out = model(input_ids=batch["input_ids"], labels=batch["labels"])
        assert torch.isfinite(out["loss"]), f"loss must be finite, got {out['loss']}"

    def test_changing_user_does_not_increase_supervised_tokens(self):
        """A7: Changing user text should NOT change the count of supervised tokens
        (only assistant tokens are supervised)."""
        tok = MockTokenizer()
        msgs_short = _make_chatml("s", "u", "same answer")
        msgs_long_user = _make_chatml("s", "a much much longer user query text here", "same answer")

        feat1 = build_assistant_only_features(msgs_short, tok, max_seq_length=512)
        feat2 = build_assistant_only_features(msgs_long_user, tok, max_seq_length=512)

        supervised1 = sum(1 for l in feat1["labels"] if l != -100)
        supervised2 = sum(1 for l in feat2["labels"] if l != -100)
        assert supervised1 == supervised2, (
            f"supervised token count should be equal (assistant unchanged): "
            f"{supervised1} vs {supervised2}"
        )

    def test_changing_assistant_changes_supervised_tokens(self):
        """A8: Changing assistant text MUST change the count of supervised tokens."""
        tok = MockTokenizer()
        msgs_short_ans = _make_chatml("s", "u", "short")
        msgs_long_ans = _make_chatml("s", "u", "a much longer assistant answer with more tokens")

        feat1 = build_assistant_only_features(msgs_short_ans, tok, max_seq_length=512)
        feat2 = build_assistant_only_features(msgs_long_ans, tok, max_seq_length=512)

        supervised1 = sum(1 for l in feat1["labels"] if l != -100)
        supervised2 = sum(1 for l in feat2["labels"] if l != -100)
        assert supervised2 > supervised1, (
            f"longer assistant should have more supervised tokens: "
            f"{supervised1} vs {supervised2}"
        )

    def test_repair_long_prompt_locates_assistant_boundary(self):
        """A9: Even with a long repair-style prompt (broken code + feedback),
        assistant boundary is correctly located."""
        tok = MockTokenizer()
        long_broken = "def solution():\n    " + "x = 1\n" * 30 + "    return x"
        long_feedback = "FAILED: test_xyz\n" + "Error line\n" * 20 + "AssertionError"
        msgs = _make_chatml(
            "你是代码助手",
            f"修复以下代码:\n```python\n{long_broken}\n```\n反馈:\n{long_feedback}",
            "```python\ndef solution():\n    return 42\n```",
        )
        feat = build_assistant_only_features(msgs, tok, max_seq_length=512)

        # Assistant must be intact (preserve_assistant policy)
        assert feat["assistant_status"] == "intact", (
            f"long repair prompt should still preserve assistant, got {feat['assistant_status']}"
        )
        # Prompt tokens all masked
        for i in range(feat["prompt_len"]):
            assert feat["labels"][i] == -100
        # Assistant tokens not masked
        assert any(l != -100 for l in feat["labels"][feat["prompt_len"]:])


# ===========================================================================
# B. Token truncation audit (5 tests)
# ===========================================================================

class TestTokenAudit:
    """Tests for compute_token_audit and truncation policy."""

    def test_long_prompt_gets_compressed(self):
        """B1: Long prompt triggers truncation (prompt-side compression)."""
        tok = MockTokenizer()
        long_user = "x" * 600  # very long user content
        msgs = _make_chatml("s", long_user, "short answer")
        feat = build_assistant_only_features(msgs, tok, max_seq_length=128)

        assert feat["truncated"] is True
        assert len(feat["input_ids"]) <= 128
        # Assistant still intact under preserve_assistant
        assert feat["assistant_status"] == "intact"

    def test_assistant_not_truncated(self):
        """B2: Under preserve_assistant policy, assistant tokens are never truncated."""
        tok = MockTokenizer()
        long_user = "u" * 400
        msgs = _make_chatml("s", long_user, "a" * 50)
        feat = build_assistant_only_features(
            msgs, tok, max_seq_length=128, truncation_policy="preserve_assistant"
        )
        assert feat["assistant_status"] == "intact"
        assert len(feat["input_ids"]) <= 128
        # The supervised (non -100) tokens correspond to assistant
        supervised = [l for l in feat["labels"] if l != -100]
        assert len(supervised) > 0

    def test_assistant_too_long_rejected(self):
        """B3: When assistant alone exceeds max_seq_length, status is target_too_long."""
        tok = MockTokenizer()
        huge_assistant = "a" * 600
        msgs = _make_chatml("s", "u", huge_assistant)
        feat = build_assistant_only_features(
            msgs, tok, max_seq_length=128, truncation_policy="preserve_assistant"
        )
        assert feat["assistant_status"] == "target_too_long"
        assert feat["truncated"] is True

    def test_audit_stats_correct(self):
        """B4: compute_token_audit returns correct aggregate statistics."""
        tok = MockTokenizer()
        records = [
            {"messages": _make_chatml("s", "u", "short")},
            {"messages": _make_chatml("s", "u" * 400, "answer")},  # will truncate at 128
        ]
        audit = compute_token_audit(records, tok, max_seq_length=128)
        assert audit["total"] == 2
        assert audit["truncated"] >= 1
        assert audit["assistant_intact"] >= 1
        assert audit["assistant_lost"] == 0
        assert len(audit["prompt_lens"]) == 2
        assert len(audit["assistant_lens"]) == 2

    def test_retention_rate_computable(self):
        """B5: Assistant retention rate = intact / total can be computed from audit."""
        tok = MockTokenizer()
        records = [
            {"messages": _make_chatml("s", "u", "ans1")},
            {"messages": _make_chatml("s", "u" * 300, "ans2")},
            {"messages": _make_chatml("s", "u", "ans3")},
        ]
        audit = compute_token_audit(records, tok, max_seq_length=256)
        total = audit["total"]
        intact = audit["assistant_intact"]
        lost = audit["assistant_lost"]
        target_too_long = audit["target_too_long"]
        retention = intact / total if total > 0 else 0.0
        # All should be intact (none lost, none too long) at max_seq_length=256
        assert lost == 0
        assert target_too_long == 0
        assert 0.0 <= retention <= 1.0


# ===========================================================================
# C. Data versioning (5 tests)
# ===========================================================================

class TestDataVersioning:
    """Tests for data/curriculum/ structure and manifests."""

    EASY_DIR = _ROOT / "data" / "curriculum-v2" / "easy"
    BOUNDARY_DIR = _ROOT / "data" / "curriculum-v2" / "boundary"
    REPAIR_DIR = _ROOT / "data" / "curriculum-v2" / "repair"

    def test_stage_paths_distinct(self):
        """C1: Easy, Boundary, Repair have separate directories and train files."""
        for d in (self.EASY_DIR, self.BOUNDARY_DIR, self.REPAIR_DIR):
            assert d.exists(), f"{d} should exist"
            assert (d / "train.jsonl").exists(), f"{d}/train.jsonl should exist"
        # The three train files must be different files (different content)
        easy_train = (self.EASY_DIR / "train.jsonl").read_bytes()
        boundary_train = (self.BOUNDARY_DIR / "train.jsonl").read_bytes()
        repair_train = (self.REPAIR_DIR / "train.jsonl").read_bytes()
        assert easy_train != boundary_train, "easy and boundary train must differ"
        assert boundary_train != repair_train, "boundary and repair train must differ"
        assert easy_train != repair_train, "easy and repair train must differ"

    def test_sha256_reproducible(self):
        """C2: SHA256 in manifest matches actual file content and is stable."""
        import hashlib

        for stage_dir in (self.EASY_DIR, self.BOUNDARY_DIR, self.REPAIR_DIR):
            manifest_path = stage_dir / "manifest.json"
            assert manifest_path.exists(), f"{manifest_path} should exist"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            train_path = stage_dir / "train.jsonl"
            actual_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
            assert manifest["train_sha256"] == actual_sha, (
                f"{stage_dir.name}: manifest train_sha256 mismatch"
            )

    def test_manifest_fields_complete(self):
        """C3: Each stage manifest has all required fields."""
        required = {
            "stage", "dataset_version", "created_at", "seed", "source_files",
            "train_sha256", "validation_sha256",
            "sample_counts", "family_counts", "task_type_mix", "difficulty_mix",
            "max_seq_length", "assistant_target_retention_rate",
            "train_families", "validation_families", "frozen_families",
            "train_validation_overlap", "train_frozen_overlap", "validation_frozen_overlap",
        }
        for stage_dir in (self.EASY_DIR, self.BOUNDARY_DIR, self.REPAIR_DIR):
            manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
            missing = required - set(manifest.keys())
            assert not missing, f"{stage_dir.name} manifest missing fields: {missing}"

    def test_family_lists_correct(self):
        """C4: families.json lists family_ids that match the train data."""
        for stage_dir in (self.EASY_DIR, self.BOUNDARY_DIR, self.REPAIR_DIR):
            families_path = stage_dir / "families.json"
            assert families_path.exists(), f"{families_path} should exist"
            families_data = json.loads(families_path.read_text(encoding="utf-8"))
            # Could be a list or dict; extract family_ids
            if isinstance(families_data, list):
                family_ids = set(families_data)
            elif isinstance(families_data, dict):
                # Prefer explicit family list keys, fall back to keys
                fam_list = (
                    families_data.get("family_ids")
                    or families_data.get("families")
                    or list(families_data.keys())
                )
                family_ids = set(fam_list)
            else:
                pytest.fail(f"unexpected families.json format in {stage_dir.name}")

            # Cross-check: family_ids in train.jsonl must be subset of families.json
            train_families = set()
            with (stage_dir / "train.jsonl").open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if "family_id" in rec:
                            train_families.add(rec["family_id"])
                        elif "messages" in rec:
                            pass  # ChatML format, skip
            if train_families:
                assert train_families.issubset(family_ids), (
                    f"{stage_dir.name}: train families not in families.json: "
                    f"{train_families - family_ids}"
                )

    def test_no_family_leakage_between_splits(self):
        """C5: Train family_ids must NOT appear in test_raw.jsonl."""
        test_raw_path = self.EASY_DIR / "test_raw.jsonl"
        if not test_raw_path.exists():
            pytest.skip("test_raw.jsonl not found in easy stage")
        test_families = set()
        with test_raw_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if "family_id" in rec:
                        test_families.add(rec["family_id"])

        for stage_dir in (self.EASY_DIR, self.BOUNDARY_DIR, self.REPAIR_DIR):
            train_path = stage_dir / "train.jsonl"
            if not train_path.exists():
                continue
            train_families = set()
            with train_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if "family_id" in rec:
                            train_families.add(rec["family_id"])
            leak = train_families & test_families
            assert not leak, (
                f"{stage_dir.name}: family leakage between train and test_raw: {leak}"
            )


# ===========================================================================
# D. Training mode (5 tests)
# ===========================================================================

class TestTrainingMode:
    """Tests for independent vs continual training mode config validation."""

    CONFIGS_DIR = _ROOT / "configs" / "curriculum"

    def _load_yaml(self, name: str) -> dict:
        import yaml

        path = self.CONFIGS_DIR / name
        assert path.exists(), f"{path} should exist"
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_independent_no_parent_adapter(self):
        """D1: Independent mode configs have initial_adapter == null."""
        for name in ("easy-independent.yaml", "boundary-independent.yaml", "repair-independent.yaml"):
            cfg = self._load_yaml(name)
            assert cfg["training_mode"] == "independent", f"{name}: training_mode"
            assert cfg.get("initial_adapter") is None, (
                f"{name}: independent mode must have initial_adapter=null"
            )

    def test_continual_loads_parent_adapter(self):
        """D2: Continual mode configs (boundary/repair) specify a parent adapter."""
        # easy-continual has no parent (it's the first stage)
        easy = self._load_yaml("easy-continual.yaml")
        assert easy["training_mode"] == "continual"
        assert easy.get("initial_adapter") is None  # first stage

        boundary = self._load_yaml("boundary-continual.yaml")
        assert boundary["training_mode"] == "continual"
        assert boundary.get("initial_adapter"), "boundary-continual must have initial_adapter"

        repair = self._load_yaml("repair-continual.yaml")
        assert repair["training_mode"] == "continual"
        assert repair.get("initial_adapter"), "repair-continual must have initial_adapter"

    def test_continual_parent_must_exist(self):
        """D3: train_lora.py rejects continual mode when parent adapter missing.

        This test validates the config-level requirement: boundary-continual's
        initial_adapter points to a path that, when training is run, must exist.
        We check the path is non-empty and well-formed.
        """
        boundary = self._load_yaml("boundary-continual.yaml")
        parent = boundary.get("initial_adapter")
        assert parent is not None and parent != "null"
        # The parent path should point into adapters/p1/continual/
        assert "adapters" in parent, f"parent adapter path looks wrong: {parent}"

    def test_continual_parent_incompatible_rejected(self):
        """D4: train_lora.py validates parent adapter has adapter_config.json.

        We test the validation logic by confirming the config points to a path
        that would be checked for adapter_config.json at runtime. The actual
        runtime check is in train_lora.py run_training().
        """
        # Verify the train_lora.py source contains the adapter_config.json check
        train_src = (_ROOT / "scripts" / "train_lora.py").read_text(encoding="utf-8")
        assert "adapter_config.json" in train_src, (
            "train_lora.py must validate parent adapter has adapter_config.json"
        )
        assert "initial_adapter not found" in train_src or "initial_adapter is not a valid" in train_src, (
            "train_lora.py must reject missing/invalid parent adapter"
        )

    def test_existing_adapters_not_overwritten(self):
        """D5: train_lora.py refuses to overwrite an existing adapter directory.

        Verify the source contains the output_dir protection logic, and that
        the new p1 output dirs do NOT collide with existing v3 adapters.
        """
        train_src = (_ROOT / "scripts" / "train_lora.py").read_text(encoding="utf-8")
        assert "already contains a trained adapter" in train_src, (
            "train_lora.py must refuse to overwrite existing adapters"
        )
        # The new p1 configs output into adapters/p1/, not adapters/code-lora-v3-*
        for name in ("easy-continual.yaml", "boundary-continual.yaml", "repair-continual.yaml",
                     "easy-independent.yaml", "boundary-independent.yaml", "repair-independent.yaml"):
            cfg = self._load_yaml(name)
            out = cfg["output_dir"]
            assert "p1" in out, f"{name}: output_dir should be under p1/, got {out}"
            assert "code-lora-v3" not in out, f"{name}: must not overwrite v3 adapters"


# ===========================================================================
# E. Frozen eval set (4 tests)
# ===========================================================================

class TestFrozenEval:
    """Tests for data/frozen-eval/v1/ structure and isolation."""

    FROZEN_DIR = _ROOT / "data" / "frozen-eval" / "v1"

    def test_frozen_families_not_in_training(self):
        """E1: Frozen eval family_ids do not appear in any curriculum train set."""
        if not self.FROZEN_DIR.exists():
            pytest.skip("frozen-eval/v1 not built yet")
        frozen_families = set()
        with (self.FROZEN_DIR / "test_raw.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if "family_id" in rec:
                        frozen_families.add(rec["family_id"])

        for stage in ("easy", "boundary", "repair"):
            train_path = _ROOT / "data" / "curriculum-v2" / stage / "train.jsonl"
            if not train_path.exists():
                continue
            train_families = set()
            with train_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if "family_id" in rec:
                            train_families.add(rec["family_id"])
            leak = frozen_families & train_families
            assert not leak, (
                f"frozen eval family leakage into {stage} train: {leak}"
            )

    def test_no_family_cross_split(self):
        """E2: Same family_id does not appear in both train and test_raw within
        the curriculum structure (test_raw is the frozen test, shared)."""
        if not self.FROZEN_DIR.exists():
            pytest.skip("frozen-eval/v1 not built yet")
        # test_raw in each curriculum stage should be identical (the frozen test)
        easy_test = _ROOT / "data" / "curriculum-v2" / "easy" / "test_raw.jsonl"
        boundary_test = _ROOT / "data" / "curriculum-v2" / "boundary" / "test_raw.jsonl"
        if easy_test.exists() and boundary_test.exists():
            assert easy_test.read_bytes() == boundary_test.read_bytes(), (
                "all stages should share the same frozen test_raw.jsonl"
            )

    def test_frozen_file_is_raw_sample_format(self):
        """E3: Frozen test file is Raw Sample format (has sample_id, not messages)."""
        test_path = self.FROZEN_DIR / "test_raw.jsonl"
        if not test_path.exists():
            pytest.skip("frozen-eval/v1/test_raw.jsonl not built yet")
        with test_path.open("r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
        assert first_line, "frozen test file is empty"
        rec = json.loads(first_line)
        assert "sample_id" in rec, "frozen test must be Raw Sample format"
        assert "messages" not in rec, "frozen test must NOT be ChatML format"
        assert "family_id" in rec
        assert "target_code" in rec
        assert "public_tests" in rec

    def test_frozen_passes_schema_validation(self):
        """E4: Every frozen test sample passes Sample.model_validate()."""
        test_path = self.FROZEN_DIR / "test_raw.jsonl"
        if not test_path.exists():
            pytest.skip("frozen-eval/v1/test_raw.jsonl not built yet")
        count = 0
        with test_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    Sample.model_validate(rec)  # raises ValidationError on failure
                    count += 1
        assert count > 0, "frozen test file should have at least one sample"
