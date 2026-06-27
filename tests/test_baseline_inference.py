"""Tests for scripts/baseline_inference.py

Covers:
- bytes_to_gb: unit conversion helper
- check_model_exists: guard logic (no model needed)
- build_prompt: chat-template wrapper (mock tokenizer)
- main() missing-model guard: returns non-zero without attempting model load

All tests are GPU-free and fast.  No real model or CUDA device is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.baseline_inference import (
    build_prompt,
    bytes_to_gb,
    check_model_exists,
)


# ---------------------------------------------------------------------------
# bytes_to_gb
# ---------------------------------------------------------------------------

class TestBytesToGb:
    def test_zero_bytes(self) -> None:
        assert bytes_to_gb(0) == 0.0

    def test_one_gb(self) -> None:
        assert bytes_to_gb(1024 ** 3) == pytest.approx(1.0)

    def test_two_gb(self) -> None:
        assert bytes_to_gb(2 * 1024 ** 3) == pytest.approx(2.0)

    def test_half_gb(self) -> None:
        assert bytes_to_gb(512 * 1024 * 1024) == pytest.approx(0.5, rel=1e-3)

    def test_1_5_gb(self) -> None:
        assert bytes_to_gb(int(1.5 * 1024 ** 3)) == pytest.approx(1.5, rel=1e-3)

    def test_result_is_float(self) -> None:
        assert isinstance(bytes_to_gb(1024 ** 3), float)

    def test_rounding_to_3_places(self) -> None:
        # 1 byte: 1 / (1024^3) ~ 9.313e-10; rounded to 3dp -> 0.0
        result = bytes_to_gb(1)
        assert result == 0.0  # rounds to 0.000


# ---------------------------------------------------------------------------
# check_model_exists
# ---------------------------------------------------------------------------

class TestCheckModelExists:
    def test_empty_dir_returns_false(self, tmp_path: Path) -> None:
        assert check_model_exists(tmp_path) is False

    def test_dir_with_config_returns_true(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert check_model_exists(tmp_path) is True

    def test_nonexistent_dir_returns_false(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        assert check_model_exists(missing) is False

    def test_dir_without_config_but_with_other_files(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        assert check_model_exists(tmp_path) is False


# ---------------------------------------------------------------------------
# build_prompt (mock tokenizer)
# ---------------------------------------------------------------------------

def _make_tokenizer(return_value: str) -> MagicMock:
    """Create a mock tokenizer whose apply_chat_template returns *return_value*."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = return_value
    return tok


class TestBuildPrompt:
    def test_returns_non_empty_string(self) -> None:
        instruction = "Write a Python max() function."
        fake_prompt = "<|im_start|>user\nWrite a Python max() function.<|im_end|>\n<|im_start|>assistant\n"
        tok = _make_tokenizer(fake_prompt)
        result = build_prompt(tok, instruction)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_contains_instruction(self) -> None:
        instruction = "Please return the largest element."
        fake_prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        tok = _make_tokenizer(fake_prompt)
        result = build_prompt(tok, instruction)
        assert instruction in result

    def test_apply_chat_template_called_once(self) -> None:
        tok = _make_tokenizer("dummy")
        build_prompt(tok, "test instruction")
        tok.apply_chat_template.assert_called_once()

    def test_messages_contain_instruction(self) -> None:
        instruction = "Test instruction for messages check."
        tok = _make_tokenizer("dummy prompt")
        build_prompt(tok, instruction)
        call_args = tok.apply_chat_template.call_args
        # messages is the first positional argument
        messages = call_args[0][0]
        assert any(msg.get("content") == instruction for msg in messages)

    def test_user_role_in_messages(self) -> None:
        tok = _make_tokenizer("dummy prompt")
        build_prompt(tok, "some instruction")
        call_args = tok.apply_chat_template.call_args
        messages = call_args[0][0]
        roles = [msg.get("role") for msg in messages]
        assert "user" in roles

    def test_add_generation_prompt_is_true(self) -> None:
        tok = _make_tokenizer("dummy prompt")
        build_prompt(tok, "test")
        _, kwargs = tok.apply_chat_template.call_args
        assert kwargs.get("add_generation_prompt") is True

    def test_tokenize_is_false(self) -> None:
        tok = _make_tokenizer("dummy prompt")
        build_prompt(tok, "test")
        _, kwargs = tok.apply_chat_template.call_args
        assert kwargs.get("tokenize") is False


# ---------------------------------------------------------------------------
# main() missing-model guard
# ---------------------------------------------------------------------------

class TestMainMissingModelGuard:
    """Verify that main() fails fast (without loading anything) when the model
    directory does not contain config.json."""

    def test_returns_nonzero_for_missing_model(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            sys, "argv",
            ["baseline_inference.py", "--model", str(tmp_path)],
        )
        import scripts.baseline_inference as bi
        result = bi.main()
        assert result != 0, "main() must return non-zero when model is missing"

    def test_does_not_call_load_when_model_missing(self, tmp_path: Path, monkeypatch) -> None:
        """main() must short-circuit before load_model_and_tokenizer when model absent."""
        monkeypatch.setattr(
            sys, "argv",
            ["baseline_inference.py", "--model", str(tmp_path)],
        )
        import scripts.baseline_inference as bi
        with patch.object(bi, "load_model_and_tokenizer") as mock_load:
            bi.main()
            mock_load.assert_not_called()

    def test_nonzero_even_with_other_files(self, tmp_path: Path, monkeypatch) -> None:
        """Model dir with non-config files (but no config.json) still fails."""
        (tmp_path / "some_random_file.txt").write_text("not a model", encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv",
            ["baseline_inference.py", "--model", str(tmp_path)],
        )
        import scripts.baseline_inference as bi
        result = bi.main()
        assert result != 0
