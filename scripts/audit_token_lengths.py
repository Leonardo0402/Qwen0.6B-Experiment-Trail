"""
scripts/audit_token_lengths.py -- Token truncation auditor for Qwen3-0.6B.

For a given curriculum stage (easy / boundary / repair), loads the training
JSONL, applies the Qwen3 chat template to each sample, and reports how many
samples / assistant spans survive truncation at ``--max-seq-length``.

Output
------
- audits/token-length/{stage}.json   per-stage detailed stats
- audits/token-length/summary.md     human-readable combined report

Supported input formats
-----------------------
The script accepts EITHER:
  - ChatML JSONL: each line is {"messages": [system, user, assistant]}
  - Raw Sample JSONL: each line is a Sample dict (see src/schemas.py);
    it is validated via ``Sample.from_json_line`` and converted to ChatML
    with ``to_chatml`` so task_type / difficulty can be reported.

Usage
-----
    python scripts/audit_token_lengths.py --stage easy --max-seq-length 512
    python scripts/audit_token_lengths.py --stage boundary --max-seq-length 512
    python scripts/audit_token_lengths.py --stage repair   --max-seq-length 512

When ``data/curriculum/{stage}.jsonl`` does not exist, the script falls back
to ``data/splits/train.jsonl`` and emits a notice.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path setup so this script can be run directly (python scripts/audit_*.py)
# and still import the project's ``src`` package.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample, to_chatml  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = str(_ROOT / "models" / "Qwen3-0.6B")
DEFAULT_MAX_SEQ_LENGTH = 512
VALID_STAGES = ("easy", "boundary", "repair")


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    """Yield parsed JSON objects from a JSONL file (blank lines skipped)."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_stage_file(stage: str) -> tuple[Path, bool]:
    """Return (path, used_fallback) for the given stage.

    Tries ``data/curriculum/{stage}.jsonl`` first; if it does not exist,
    falls back to ``data/splits/train.jsonl``.
    """
    primary = _ROOT / "data" / "curriculum" / f"{stage}.jsonl"
    if primary.exists():
        return primary, False
    fallback = _ROOT / "data" / "splits" / "train.jsonl"
    return fallback, True


# ---------------------------------------------------------------------------
# Sample normalisation
# ---------------------------------------------------------------------------

def normalise_record(raw: dict) -> tuple[list[dict], str, Any]:
    """Return (messages, task_type, difficulty) for a raw JSONL record.

    Detects the record format:
      - ChatML: has a top-level ``messages`` field.
      - Raw Sample: validates via ``Sample.from_json_line`` then converts.

    For ChatML records, ``task_type`` / ``difficulty`` are read from the
    top-level fields if present (build_dataset.py emits plain ChatML without
    them, so they may be ``"unknown"`` / ``None``).
    """
    if "messages" in raw:
        messages = raw["messages"]
        task_type = raw.get("task_type", "unknown")
        difficulty = raw.get("difficulty")
        return messages, task_type, difficulty

    # Try raw Sample format.
    try:
        sample = Sample(**raw)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"record is neither ChatML nor a valid Sample: {exc}") from exc
    chatml = to_chatml(sample)
    return chatml["messages"], sample.task_type, sample.difficulty


def split_messages(messages: list[dict]) -> tuple[list[dict], list[dict], Optional[dict]]:
    """Split a ChatML messages list into (prompt_messages, full_messages, assistant_msg).

    prompt_messages = every message up to and including the LAST user turn
    full_messages   = prompt_messages + assistant message (if present)
    assistant_msg   = the trailing assistant message, or None
    """
    # Find the last user message index.
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx < 0:
        # No user message — treat the whole list as the prompt.
        prompt_messages = list(messages)
        assistant_msg = None
    else:
        prompt_messages = list(messages[: last_user_idx + 1])
        # The message immediately after the last user message, if it is an
        # assistant message, is the response.
        if last_user_idx + 1 < len(messages) and messages[last_user_idx + 1].get("role") == "assistant":
            assistant_msg = messages[last_user_idx + 1]
        else:
            assistant_msg = None

    full_messages = list(prompt_messages)
    if assistant_msg is not None:
        full_messages.append(assistant_msg)
    return prompt_messages, full_messages, assistant_msg


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenise_sample(tokenizer: Any, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, full_ids) for a ChatML messages list.

    prompt_ids = apply_chat_template(system+user, tokenize=True,
                                     add_generation_prompt=True)
    full_ids   = apply_chat_template(system+user+assistant, tokenize=True,
                                     add_generation_prompt=False)
    """
    prompt_messages, full_messages, assistant_msg = split_messages(messages)

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if assistant_msg is None:
        # No assistant turn: full_ids == prompt_ids (assistant_len = 0).
        return list(prompt_ids), list(prompt_ids)

    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return list(prompt_ids), list(full_ids)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[int], pct: float) -> int:
    """Return the pct-th percentile (0..100) of a SORTED list of ints.

    Uses linear interpolation between closest ranks (same method as
    numpy.percentile default).  Returns 0 for an empty list.
    """
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return int(round(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac))


def _percentile_summary(values: list[int]) -> dict[str, int]:
    """Return {p50,p90,p95,p99,max} for an arbitrary list of ints."""
    s = sorted(values)
    return {
        "p50": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
        "max": s[-1] if s else 0,
    }


def _mean(values: list[int]) -> float:
    return round(statistics.fmean(values), 2) if values else 0.0


def _length_distribution(samples: list[dict], key: str) -> dict[str, dict[str, Any]]:
    """Group samples by ``key`` and return per-group length stats.

    Each value is a dict with: count, prompt_mean, assistant_mean,
    full_mean, truncated_count, and p50/p90/p95/p99/max for each span.
    """
    groups: dict[str, list[dict]] = {}
    for s in samples:
        g = str(s[key])
        groups.setdefault(g, []).append(s)

    out: dict[str, dict[str, Any]] = {}
    for g, items in sorted(groups.items()):
        prompts = [s["prompt_len"] for s in items]
        assistants = [s["assistant_len"] for s in items]
        fulls = [s["full_len"] for s in items]
        out[g] = {
            "count": len(items),
            "prompt_mean": _mean(prompts),
            "assistant_mean": _mean(assistants),
            "full_mean": _mean(fulls),
            "truncated_count": sum(1 for s in items if s["truncated"]),
            "prompt": _percentile_summary(prompts),
            "assistant": _percentile_summary(assistants),
            "full": _percentile_summary(fulls),
        }
    return out


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def audit_stage(
    tokenizer: Any,
    stage: str,
    data_path: Path,
    max_seq_length: int,
    used_fallback: bool,
) -> dict:
    """Run the token-length audit for one stage and return the stats dict."""
    per_sample: list[dict] = []
    parse_errors: list[str] = []
    sample_index = 0

    for raw in _iter_jsonl(data_path):
        try:
            messages, task_type, difficulty = normalise_record(raw)
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(f"#{sample_index}: {exc}")
            sample_index += 1
            continue

        try:
            prompt_ids, full_ids = tokenise_sample(tokenizer, messages)
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(f"#{sample_index} tokenise: {exc}")
            sample_index += 1
            continue

        prompt_len = len(prompt_ids)
        full_len = len(full_ids)
        # assistant span = the suffix of full_ids past the prompt prefix.
        # Under the chat template this is well-defined as long as full_ids
        # starts with prompt_ids (which it does for the standard Qwen3
        # template).  If for any reason full_len < prompt_len we clamp to 0.
        assistant_len = max(0, full_len - prompt_len)

        truncated = full_len > max_seq_length
        # Tokens of the assistant that survive truncation.
        # If max_seq_length <= prompt_len, no assistant tokens survive.
        assistant_kept = max(0, max_seq_length - prompt_len) if truncated else assistant_len
        if not truncated:
            assistant_state = "intact"
        elif assistant_kept <= 0:
            assistant_state = "lost"
        elif assistant_kept < assistant_len:
            assistant_state = "partial"
        else:
            # Shouldn't normally happen (would imply full_len <= max_seq_length),
            # but kept as a safe fallback.
            assistant_state = "intact"

        per_sample.append({
            "index": sample_index,
            "task_type": task_type,
            "difficulty": difficulty,
            "prompt_len": prompt_len,
            "assistant_len": assistant_len,
            "full_len": full_len,
            "truncated": truncated,
            "assistant_kept": assistant_kept,
            "assistant_state": assistant_state,
        })
        sample_index += 1

    n_total = len(per_sample)
    n_truncated = sum(1 for s in per_sample if s["truncated"])
    n_intact_full = n_total - n_truncated
    n_asst_intact = sum(1 for s in per_sample if s["assistant_state"] == "intact")
    n_asst_partial = sum(1 for s in per_sample if s["assistant_state"] == "partial")
    n_asst_lost = sum(1 for s in per_sample if s["assistant_state"] == "lost")

    prompts = [s["prompt_len"] for s in per_sample]
    assistants = [s["assistant_len"] for s in per_sample]
    fulls = [s["full_len"] for s in per_sample]

    stats = {
        "stage": stage,
        "data_path": str(data_path.relative_to(_ROOT)) if data_path.is_relative_to(_ROOT) else str(data_path),
        "used_fallback": used_fallback,
        "max_seq_length": max_seq_length,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_total": n_total,
        "untruncated_count": n_intact_full,
        "truncated_count": n_truncated,
        "assistant_intact_count": n_asst_intact,
        "assistant_partial_count": n_asst_partial,
        "assistant_lost_count": n_asst_lost,
        "prompt_mean": _mean(prompts),
        "assistant_mean": _mean(assistants),
        "full_mean": _mean(fulls),
        "prompt": _percentile_summary(prompts),
        "assistant": _percentile_summary(assistants),
        "full": _percentile_summary(fulls),
        "by_task_type": _length_distribution(per_sample, "task_type"),
        "by_difficulty": _length_distribution(per_sample, "difficulty"),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:20],
    }
    return stats


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_stage_json(stats: dict, out_dir: Path) -> Path:
    """Write a single stage's stats dict to ``{out_dir}/{stage}.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stats['stage']}.json"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    return path


def write_summary_md(all_stats: list[dict], out_dir: Path) -> Path:
    """Write the combined Markdown summary across all stages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.md"

    lines: list[str] = []
    lines.append("# Token Length Audit Summary")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).isoformat()}"
    )
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    header = (
        "| Stage | MaxLen | Samples | Untrunc | Trunc | "
        "Asst Intact | Asst Partial | Asst Lost | "
        "Prompt P50/P90/P95/P99/Max | Asst P50/P90/P95/P99/Max | "
        "Full P50/P90/P95/P99/Max | Source |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 12)

    for s in all_stats:
        p = s["prompt"]
        a = s["assistant"]
        f = s["full"]
        src = "fallback" if s["used_fallback"] else "curriculum"
        lines.append(
            f"| {s['stage']} | {s['max_seq_length']} | {s['sample_total']} | "
            f"{s['untruncated_count']} | {s['truncated_count']} | "
            f"{s['assistant_intact_count']} | {s['assistant_partial_count']} | "
            f"{s['assistant_lost_count']} | "
            f"{p['p50']}/{p['p90']}/{p['p95']}/{p['p99']}/{p['max']} | "
            f"{a['p50']}/{a['p90']}/{a['p95']}/{a['p99']}/{a['max']} | "
            f"{f['p50']}/{f['p90']}/{f['p95']}/{f['p99']}/{f['max']} | "
            f"{src} |"
        )

    lines.append("")
    lines.append("## Per-stage detail")
    lines.append("")

    for s in all_stats:
        lines.append(f"### Stage: {s['stage']}")
        lines.append("")
        lines.append(f"- Data source: `{s['data_path']}`"
                     f"{' (fallback to data/splits/train.jsonl)' if s['used_fallback'] else ''}")
        lines.append(f"- max_seq_length: **{s['max_seq_length']}**")
        lines.append(f"- Total samples: **{s['sample_total']}**")
        lines.append(f"- Untruncated: {s['untruncated_count']}  "
                     f"Truncated: {s['truncated_count']}")
        lines.append(f"- Assistant intact: {s['assistant_intact_count']}  "
                     f"partial: {s['assistant_partial_count']}  "
                     f"lost: {s['assistant_lost_count']}")
        lines.append("")
        lines.append("#### Length percentiles (tokens)")
        lines.append("")
        lines.append("| Span | P50 | P90 | P95 | P99 | Max | Mean |")
        lines.append("|---|---|---|---|---|---|---|")
        for name in ("prompt", "assistant", "full"):
            d = s[name]
            lines.append(
                f"| {name} | {d['p50']} | {d['p90']} | {d['p95']} | "
                f"{d['p99']} | {d['max']} | {s[name + '_mean']} |"
            )
        lines.append("")

        # By task_type
        if s["by_task_type"]:
            lines.append("#### By TaskType")
            lines.append("")
            lines.append("| TaskType | Count | Trunc | Prompt P50/Max | Asst P50/Max | Full P50/Max |")
            lines.append("|---|---|---|---|---|---|")
            for g, d in s["by_task_type"].items():
                lines.append(
                    f"| {g} | {d['count']} | {d['truncated_count']} | "
                    f"{d['prompt']['p50']}/{d['prompt']['max']} | "
                    f"{d['assistant']['p50']}/{d['assistant']['max']} | "
                    f"{d['full']['p50']}/{d['full']['max']} |"
                )
            lines.append("")

        # By difficulty
        if s["by_difficulty"]:
            lines.append("#### By Difficulty")
            lines.append("")
            lines.append("| Difficulty | Count | Trunc | Prompt P50/Max | Asst P50/Max | Full P50/Max |")
            lines.append("|---|---|---|---|---|---|")
            for g, d in s["by_difficulty"].items():
                lines.append(
                    f"| {g} | {d['count']} | {d['truncated_count']} | "
                    f"{d['prompt']['p50']}/{d['prompt']['max']} | "
                    f"{d['assistant']['p50']}/{d['assistant']['max']} | "
                    f"{d['full']['p50']}/{d['full']['max']} |"
                )
            lines.append("")

        if s["parse_error_count"]:
            lines.append(f"#### Parse errors: {s['parse_error_count']}")
            for err in s["parse_errors"]:
                lines.append(f"- {err}")
            lines.append("")

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit token lengths and assistant-truncation for a curriculum stage.",
    )
    p.add_argument(
        "--stage", choices=VALID_STAGES, required=True,
        help="Curriculum stage to audit (easy / boundary / repair).",
    )
    p.add_argument(
        "--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Max sequence length for truncation check (default: {DEFAULT_MAX_SEQ_LENGTH}).",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Path to the Qwen3-0.6B tokenizer directory (default: models/Qwen3-0.6B).",
    )
    p.add_argument(
        "--data-file", default=None,
        help="Override input JSONL path. Defaults to data/curriculum/{stage}.jsonl "
             "with fallback to data/splits/train.jsonl.",
    )
    p.add_argument(
        "--out-dir", default=str(_ROOT / "audits" / "token-length"),
        help="Output directory for audit JSON and summary.md.",
    )
    p.add_argument(
        "--all-stages", action="store_true",
        help="Run for all three stages (easy/boundary/repair) and produce a combined summary.",
    )
    return p


def _load_tokenizer(model_path: str) -> Any:
    """Load the Qwen3 tokenizer (deferred import for pytest-safety on Windows)."""
    try:
        from transformers import AutoTokenizer
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"transformers not available: {exc}") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> int:
    """CLI entry point."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args()

    model_path = Path(args.model).resolve()
    if not (model_path / "config.json").exists() and not (model_path / "tokenizer_config.json").exists():
        print(f"ERROR: tokenizer not found at {model_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    stages = list(VALID_STAGES) if args.all_stages else [args.stage]

    print(f"Loading tokenizer from {model_path} ...")
    try:
        tokenizer = _load_tokenizer(str(model_path))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to load tokenizer: {exc}", file=sys.stderr)
        return 1

    all_stats: list[dict] = []
    for stage in stages:
        if args.data_file:
            data_path = Path(args.data_file)
            used_fallback = False
            if not data_path.exists():
                print(f"ERROR: --data-file not found: {data_path}", file=sys.stderr)
                return 1
        else:
            data_path, used_fallback = resolve_stage_file(stage)
            if not data_path.exists():
                print(
                    f"ERROR: no data found for stage '{stage}': neither "
                    f"{_ROOT / 'data' / 'curriculum' / f'{stage}.jsonl'} nor "
                    f"fallback {data_path} exist.",
                    file=sys.stderr,
                )
                return 1

        if used_fallback:
            print(
                f"[{stage}] data/curriculum/{stage}.jsonl not found; "
                f"falling back to {data_path}"
            )
        print(f"[{stage}] auditing {data_path} (max_seq_length={args.max_seq_length}) ...")

        stats = audit_stage(
            tokenizer=tokenizer,
            stage=stage,
            data_path=data_path,
            max_seq_length=args.max_seq_length,
            used_fallback=used_fallback,
        )

        json_path = write_stage_json(stats, out_dir)
        print(
            f"[{stage}] samples={stats['sample_total']}  "
            f"truncated={stats['truncated_count']}  "
            f"asst_intact={stats['assistant_intact_count']}  "
            f"asst_partial={stats['assistant_partial_count']}  "
            f"asst_lost={stats['assistant_lost_count']}"
        )
        print(f"[{stage}] wrote {json_path}")
        all_stats.append(stats)

    summary_path = write_summary_md(all_stats, out_dir)
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
