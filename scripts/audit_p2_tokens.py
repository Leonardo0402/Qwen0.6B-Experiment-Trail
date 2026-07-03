"""P2 Token audit for max_seq_length=384."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer
from src.training_data import compute_token_audit


def main() -> int:
    tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = {}
    for stage in ["stage1-code", "stage2-boundary", "stage3-repair"]:
        p = _ROOT / "data" / "p2-curriculum" / stage / "train.jsonl"
        if not p.exists():
            continue
        records = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        audit = compute_token_audit(records, tok, max_seq_length=384)
        retention = audit["assistant_intact"] / max(1, audit["total"])
        results[stage] = {
            "total": audit["total"],
            "truncated": audit["truncated"],
            "assistant_intact": audit["assistant_intact"],
            "assistant_partial": audit["assistant_partial"],
            "assistant_lost": audit["assistant_lost"],
            "target_too_long": audit["target_too_long"],
            "retention_rate": retention,
        }
        print(
            f"{stage}: total={audit['total']}, intact={audit['assistant_intact']}, "
            f"lost={audit['assistant_lost']}, too_long={audit['target_too_long']}, "
            f"retention={retention:.4f}"
        )

    out = _ROOT / "reports" / "p2" / "token-audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Token audit saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
