"""P3: Paired statistics for P2 model comparison.

Issue #1 P3 requirements:
- Use identical sample_id across Base, Stage2, Continual Stage3-v2,
  Independent Stage3, Anti-forgetting Stage3-v3 (and Stage1 if available).
- Output per-sample win/loss/unchanged.
- Output per-family net gain/loss.
- McNemar test or paired bootstrap CI.
- Per bug_type repair success rate.

Inputs (all must use the FIXED stratified-120 subset):
  evaluations/p2/base.json
  evaluations/p2/stage1-code.json  (optional, not in Issue comparison)
  evaluations/p2/stage2-boundary.json
  evaluations/p2/stage3-repair.json  (Continual Stage3-v2)
  evaluations/p2/independent-stage3.json  (P1)
  evaluations/p2/stage3-v3-antiforget.json  (P2)

Outputs:
  reports/p2/paired-stats.json
  reports/p2/paired-stats.md (summary)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = _ROOT / "evaluations" / "p2"
_OUT_DIR = _ROOT / "reports" / "p2"

# Canonical comparison order per Issue #1 P3
MODELS: list[tuple[str, str]] = [
    ("base", "Base"),
    ("stage2-boundary", "Stage2-v2"),
    ("stage3-repair", "Stage3-v2-Continual"),
    ("independent-stage3", "Stage3-Independent"),  # P1
    ("stage3-v3-antiforget", "Stage3-v3-Antiforget"),  # P2
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> dict[str, Any] | None:
    f = _EVAL_DIR / f"{name}.json"
    if not f.exists():
        return None
    with f.open(encoding="utf-8") as fh:
        return json.load(fh)


def _passed(o: dict) -> bool:
    """A sample passes iff both public and hidden tests pass."""
    return bool(o.get("public_passed") and o.get("hidden_passed"))


def _outcomes_by_id(doc: dict) -> dict[str, dict]:
    return {o["sample_id"]: o for o in doc.get("outcomes", [])}


# ---------------------------------------------------------------------------
# McNemar test (no scipy dependency)
# ---------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via binomial.

    b = #samples where model_A passed, model_B failed
    c = #samples where model_A failed, model_B passed
    Under H0 (no difference), b+c ~ Binomial(b+c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    # Two-sided exact: 2 * min(P(X<=min(b,c)), P(X>=max(b,c)))
    # Equivalently: 2 * sum_{k=0}^{min(b,c)} C(n,k) * 0.5^n  (capped at 1.0)
    from math import comb

    k_min = min(b, c)
    p_one_tail = sum(comb(n, k) for k in range(k_min + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * p_one_tail)


def paired_bootstrap_ci(
    pass_a: list[bool], pass_b: list[bool], n_boot: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """Paired bootstrap 95% CI for the difference in pass rate (b - a)."""
    import random

    rng = random.Random(seed)
    n = len(pass_a)
    if n == 0:
        return (0.0, 0.0)
    diffs = [(1 if b else 0) - (1 if a else 0) for a, b in zip(pass_a, pass_b)]
    observed = sum(diffs) / n
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_diff = sum(diffs[i] for i in idx) / n
        boots.append(boot_diff)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Per-sample, per-family, per-bug_type
# ---------------------------------------------------------------------------

def per_sample_compare(doc_a: dict, doc_b: dict) -> dict:
    """Compare two models sample-by-sample."""
    a_by = _outcomes_by_id(doc_a)
    b_by = _outcomes_by_id(doc_b)
    common = sorted(set(a_by) & set(b_by))
    win, loss, unchanged = 0, 0, 0
    flips_to_pass = []
    flips_to_fail = []
    for sid in common:
        pa = _passed(a_by[sid])
        pb = _passed(b_by[sid])
        if pa == pb:
            unchanged += 1
        elif (not pa) and pb:
            win += 1
            flips_to_pass.append(sid)
        else:
            loss += 1
            flips_to_fail.append(sid)
    n = len(common)
    base_rate = sum(_passed(a_by[s]) for s in common) / n if n else 0
    other_rate = sum(_passed(b_by[s]) for s in common) / n if n else 0
    # McNemar counts
    b_count = sum(1 for s in common if _passed(a_by[s]) and not _passed(b_by[s]))
    c_count = sum(1 for s in common if not _passed(a_by[s]) and _passed(b_by[s]))
    p_value = mcnemar_exact(b_count, c_count)
    # Bootstrap CI
    pass_a = [_passed(a_by[s]) for s in common]
    pass_b = [_passed(b_by[s]) for s in common]
    ci_lo, ci_hi = paired_bootstrap_ci(pass_a, pass_b)
    return {
        "n_compared": n,
        "win": win,
        "loss": loss,
        "unchanged": unchanged,
        "rate_a": base_rate,
        "rate_b": other_rate,
        "delta": other_rate - base_rate,
        "mcnemar_b": b_count,
        "mcnemar_c": c_count,
        "mcnemar_p_two_sided": p_value,
        "bootstrap_95ci": [ci_lo, ci_hi],
        "flips_to_pass": flips_to_pass,
        "flips_to_fail": flips_to_fail,
    }


def per_family_compare(doc_a: dict, doc_b: dict) -> dict:
    """Family passes iff all samples in that family pass. Compare net gain/loss."""
    a_by = _outcomes_by_id(doc_a)
    b_by = _outcomes_by_id(doc_b)
    common = sorted(set(a_by) & set(b_by))
    fam_a: dict[str, list[bool]] = defaultdict(list)
    fam_b: dict[str, list[bool]] = defaultdict(list)
    for sid in common:
        f = a_by[sid].get("family_id", "?")
        fam_a[f].append(_passed(a_by[sid]))
        fam_b[f].append(_passed(b_by[sid]))
    fam_pass_a = {f: (len(v) > 0 and all(v)) for f, v in fam_a.items()}
    fam_pass_b = {f: (len(v) > 0 and all(v)) for f, v in fam_b.items()}
    gained, lost = [], []
    for f in fam_pass_a:
        if fam_pass_a[f] and not fam_pass_b[f]:
            lost.append(f)
        elif (not fam_pass_a[f]) and fam_pass_b[f]:
            gained.append(f)
    return {
        "n_families": len(fam_pass_a),
        "families_gained": sorted(gained),
        "families_lost": sorted(lost),
        "net_gain": len(gained) - len(lost),
        "pass_a": sum(fam_pass_a.values()),
        "pass_b": sum(fam_pass_b.values()),
    }


def per_bug_type_repair(doc: dict) -> dict:
    """For repair task_types (static_repair, execution_repair),
    compute success rate per bug_type extracted from sample_id."""
    by_bug: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0})
    for o in doc.get("outcomes", []):
        tt = o.get("task_type", "")
        if tt not in ("static_repair", "execution_repair"):
            continue
        sid = o.get("sample_id", "")
        # sample_id format: mbpp_<N>_sr_<bug_type> or mbpp_<N>_er_<bug_type>
        parts = sid.split("_")
        bug_type = "?"
        if len(parts) >= 4 and parts[-2] in ("sr", "er"):
            bug_type = parts[-1]
        elif "branch_deletion" in sid:
            bug_type = "branch_deletion"
        elif "off_by_one" in sid:
            bug_type = "off_by_one"
        elif "return_value_error" in sid:
            bug_type = "return_value_error"
        elif "initialization_error" in sid:
            bug_type = "initialization_error"
        elif "type_error" in sid:
            bug_type = "type_error"
        elif "condition_error" in sid:
            bug_type = "condition_error"
        elif "aggregation_error" in sid:
            bug_type = "aggregation_error"
        elif "index_error" in sid:
            bug_type = "index_error"
        by_bug[bug_type]["total"] += 1
        if _passed(o):
            by_bug[bug_type]["passed"] += 1
    out = {}
    for bug, s in by_bug.items():
        out[bug] = {
            "total": s["total"],
            "passed": s["passed"],
            "pass_rate": s["passed"] / s["total"] if s["total"] else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = {}
    for key, _label in MODELS:
        d = _load(key)
        if d is None:
            print(f"WARN: evaluations/p2/{key}.json not found - skipping")
            continue
        docs[key] = d
    if len(docs) < 2:
        print("ERROR: need at least 2 model evals for paired stats")
        return 1

    # Verify same sample_ids across all docs (sanity)
    sid_sets = {k: set(_outcomes_by_id(d)) for k, d in docs.items()}
    common = set.intersection(*sid_sets.values()) if len(sid_sets) > 1 else next(iter(sid_sets.values()))
    print(f"Models loaded: {list(docs)}")
    print(f"Common sample IDs: {len(common)}")

    # Per-pair comparisons (consecutive in MODELS order)
    pairs = []
    keys = list(docs)
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        pairs.append({
            "pair": [a, b],
            "sample_compare": per_sample_compare(docs[a], docs[b]),
            "family_compare": per_family_compare(docs[a], docs[b]),
        })
    # Also include Base vs each later model for direct comparison
    for k in keys[2:]:
        pairs.append({
            "pair": ["base", k],
            "sample_compare": per_sample_compare(docs["base"], docs[k]),
            "family_compare": per_family_compare(docs["base"], docs[k]),
        })

    # Per-model bug_type repair stats
    bug_stats = {k: per_bug_type_repair(d) for k, d in docs.items()}

    result = {
        "models": [k for k in docs],
        "common_sample_count": len(common),
        "pair_comparisons": pairs,
        "per_model_bug_type_repair": bug_stats,
    }

    out_json = _OUT_DIR / "paired-stats.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved JSON: {out_json}")

    # Markdown summary
    md_lines = ["# P3: Paired Statistics Summary", ""]
    md_lines.append(f"- Models compared: {', '.join(docs)}")
    md_lines.append(f"- Common sample IDs: {len(common)}")
    md_lines.append("")
    md_lines.append("## Per-pair sample-level comparison")
    md_lines.append("")
    md_lines.append("| Pair | N | Win | Loss | Unchanged | Δ rate | McNemar b/c | p (2-sided) | 95% CI |")
    md_lines.append("|------|---|-----|------|-----------|--------|-------------|-------------|--------|")
    for p in pairs:
        a, b = p["pair"]
        s = p["sample_compare"]
        ci = s["bootstrap_95ci"]
        md_lines.append(
            f"| {a} → {b} | {s['n_compared']} | {s['win']} | {s['loss']} | {s['unchanged']} | "
            f"{s['delta']:+.4f} | {s['mcnemar_b']}/{s['mcnemar_c']} | {s['mcnemar_p_two_sided']:.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )
    md_lines.append("")
    md_lines.append("## Per-pair family-level comparison")
    md_lines.append("")
    md_lines.append("| Pair | Families | Gained | Lost | Net | A pass | B pass |")
    md_lines.append("|------|----------|--------|------|-----|--------|--------|")
    for p in pairs:
        a, b = p["pair"]
        f = p["family_compare"]
        md_lines.append(
            f"| {a} → {b} | {f['n_families']} | {len(f['families_gained'])} | "
            f"{len(f['families_lost'])} | {f['net_gain']:+d} | {f['pass_a']} | {f['pass_b']} |"
        )
    md_lines.append("")
    md_lines.append("## Per bug_type repair success rate")
    md_lines.append("")
    bug_types = sorted({bt for d in bug_stats.values() for bt in d})
    header = "| Bug type | " + " | ".join(docs) + " |"
    sep = "|----------|" + "|".join(["---"] * len(docs)) + "|"
    md_lines.append(header)
    md_lines.append(sep)
    for bt in bug_types:
        row = f"| {bt} |"
        for k in docs:
            s = bug_stats[k].get(bt, {"pass_rate": None, "total": 0, "passed": 0})
            if s["total"] == 0:
                row += " - |"
            else:
                row += f" {s['passed']}/{s['total']} ({s['pass_rate']*100:.1f}%) |"
        md_lines.append(row)
    md_lines.append("")
    out_md = _OUT_DIR / "paired-stats.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
