"""Router feasibility analysis for Issue #6 P2.

Computes four routing strategies across the 5 P2 models:

1. Best Single Model - pick the model with highest overall pass rate.
2. Oracle Router - upper bound: any model passes -> router passes.
3. Metadata Router - route by ``task_type``, pick best model per task_type.
4. Deployable Deterministic Router - route by deployment-observable signals
   (``broken_code`` presence, ``execution_feedback`` presence), pick best
   model per category.

Sample passes iff ``public_passed AND hidden_passed``.
Family passes iff ALL its samples pass.

Inputs:
  evaluations/p2/full576-base.json
  evaluations/p2/full576-stage2-boundary.json
  evaluations/p2/full576-stage3-repair.json
  evaluations/p2/full576-independent-stage3.json
  evaluations/p2/full576-stage3-v3-antiforget.json
  reports/p2/router-policy-v1.json (frozen policy from split_router_selection.py)

Outputs:
  reports/p2/router-analysis.json
  reports/p2/router-analysis.md

Routing policy (Best Single, Metadata Router, Deployable Router) is loaded
from the frozen artifact reports/p2/router-policy-v1.json, which was produced
by scripts/split_router_selection.py on a held-out selection subset (45
families, 342 samples). This script applies that frozen policy to the eval
subset (30 families, 234 samples) — the two subsets are family-disjoint.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = _ROOT / "evaluations" / "p2"
_OUT_DIR = _ROOT / "reports" / "p2"
_POLICY_PATH = _ROOT / "reports" / "p2" / "router-policy-v1.json"

MODELS: list[tuple[str, str]] = [
    ("full576-base", "Base"),
    ("full576-stage2-boundary", "Stage2-v2"),
    ("full576-stage3-repair", "Stage3-v2-Continual"),
    ("full576-independent-stage3", "Stage3-Independent"),
    ("full576-stage3-v3-antiforget", "Stage3-v3-Antiforget"),
]

TASK_TYPES = ["code_generation", "static_repair", "execution_repair"]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> dict[str, Any] | None:
    f = _EVAL_DIR / f"{name}.json"
    if not f.exists():
        return None
    with f.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_policy(policy_path: Path | None = None) -> dict:
    """Load the frozen router policy artifact. Hard-fail if missing.

    Looks up ``_POLICY_PATH`` at call time (not as a default arg bound at
    function-definition time) so tests can monkeypatch the module constant.
    """
    if policy_path is None:
        policy_path = _POLICY_PATH
    if not policy_path.exists():
        raise SystemExit(
            f"ERROR: router policy not found at {policy_path}. "
            "Run `python scripts/split_router_selection.py` first."
        )
    with policy_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _validate_policy_alignment(
    policy: dict, docs: dict[str, dict], common: list[str]
) -> None:
    """Hard-fail if policy does not align with loaded evals.

    Checks:
    1. selection_dataset_sha256 matches dataset_sha256 of every loaded eval.
    2. selection_sample_ids ∩ eval_sample_ids == empty.
    3. eval_sample_ids ⊆ common (every eval sample must be present in all models).
    4. selection_families ∩ eval_families == empty.
    """
    expected_sha = policy["selection_dataset_sha256"]
    for key, doc in docs.items():
        actual_sha = doc.get("dataset_sha256")
        if actual_sha != expected_sha:
            raise SystemExit(
                f"ERROR: dataset_sha256 mismatch for model '{key}'. "
                f"Policy expects {expected_sha}, eval has {actual_sha}. "
                "Regenerate router-policy-v1.json with split_router_selection.py."
            )
    sel_ids = set(policy["selection_sample_ids"])
    eval_ids = set(policy["eval_sample_ids"])
    if sel_ids & eval_ids:
        raise SystemExit(
            "ERROR: selection_sample_ids and eval_sample_ids overlap. "
            "Policy artifact is corrupted."
        )
    sel_fams = set(policy["selection_families"])
    eval_fams = set(policy["eval_families"])
    if sel_fams & eval_fams:
        raise SystemExit(
            "ERROR: selection_families and eval_families overlap. "
            "Policy artifact is corrupted."
        )
    common_set = set(common)
    missing = eval_ids - common_set
    if missing:
        raise SystemExit(
            f"ERROR: {len(missing)} eval sample IDs from policy are not "
            f"present in all loaded model evals (e.g. {sorted(missing)[:3]}). "
            "Eval files and policy are out of sync."
        )


def _filter_to_eval_subset(
    by_model: dict[str, dict[str, dict]],
    common: list[str],
    policy: dict,
) -> list[str]:
    """Return the sorted list of common sample IDs that are in policy's
    eval_sample_ids. This is the held-out subset used for all router evaluation.
    """
    eval_ids = set(policy["eval_sample_ids"])
    return sorted(sid for sid in common if sid in eval_ids)


def _passed(o: dict) -> bool:
    """A sample passes iff both public and hidden tests pass."""
    return bool(o.get("public_passed") and o.get("hidden_passed"))


def _outcomes_by_id(doc: dict) -> dict[str, dict]:
    return {o["sample_id"]: o for o in doc.get("outcomes", [])}


def _label_of(model_key: str) -> str:
    for k, label in MODELS:
        if k == model_key:
            return label
    return model_key


def _pct(x: float) -> str:
    """Format a 0-1 ratio as a percentage string with one decimal."""
    return f"{x * 100:.1f}%"


def extract_bug_type(sample_id: str) -> str | None:
    """Extract bug_type from sample_id.

    Formats:
      mbpp_<N>                 -> code_generation (None)
      mbpp_<N>_boundary        -> code_generation (None)
      mbpp_<N>_sr_<bug_type>   -> static_repair
      mbpp_<N>_er_<bug_type>   -> execution_repair
    """
    parts = sample_id.split("_")
    for marker in ("sr", "er"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return "_".join(parts[idx + 1:])
    return None


# ---------------------------------------------------------------------------
# Paired stats helpers (inlined to keep this module self-contained)
# ---------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via binomial.
    b = #samples where A passed, B failed. c = #samples where A failed, B passed.
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
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
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def overall_pass_rate(pass_map: dict[str, bool]) -> float:
    if not pass_map:
        return 0.0
    return sum(1 for v in pass_map.values() if v) / len(pass_map)


def family_pass_rate(pass_map: dict[str, bool], sid_to_family: dict[str, str]) -> float:
    """Family passes iff ALL its samples pass."""
    fam: dict[str, list[bool]] = defaultdict(list)
    for sid, passed in pass_map.items():
        fam[sid_to_family.get(sid, "?")].append(passed)
    if not fam:
        return 0.0
    n_pass = sum(1 for v in fam.values() if v and all(v))
    return n_pass / len(fam)


def per_task_type_stats(
    pass_map: dict[str, bool], sid_to_task_type: dict[str, str]
) -> dict[str, dict]:
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0})
    for sid, passed in pass_map.items():
        tt = sid_to_task_type.get(sid, "unknown")
        stats[tt]["total"] += 1
        if passed:
            stats[tt]["passed"] += 1
    return {
        tt: {
            "total": s["total"],
            "passed": s["passed"],
            "pass_rate": s["passed"] / s["total"] if s["total"] else 0.0,
        }
        for tt, s in stats.items()
    }


def compute_router_stats(
    pass_map: dict[str, bool],
    sid_to_family: dict[str, str],
    sid_to_task_type: dict[str, str],
) -> dict:
    return {
        "overall_pass": overall_pass_rate(pass_map),
        "family_pass": family_pass_rate(pass_map, sid_to_family),
        "per_task_type": per_task_type_stats(pass_map, sid_to_task_type),
        "n_samples": len(pass_map),
    }


# ---------------------------------------------------------------------------
# Observable signals (Deployable Router only)
# ---------------------------------------------------------------------------

def infer_observable_signals(task_type: str) -> dict:
    """Infer deployment-observable signals from task_type.

    At deployment time we can observe:
    - Whether the request contains ``broken_code`` (a code snippet to repair).
    - Whether ``execution_feedback`` is available (we ran the code and saw an error).
    - The request category (generation vs repair).

    These signals MUST NOT use hidden test results or gold bug_type labels.
    """
    has_broken_code = task_type in ("static_repair", "execution_repair")
    has_execution_feedback = task_type == "execution_repair"
    return {
        "has_broken_code": has_broken_code,
        "has_execution_feedback": has_execution_feedback,
        "request_type": "repair" if has_broken_code else "generation",
    }


def route_deployable(signals: dict) -> str:
    """Map observable signals to a routing category (task_type)."""
    if not signals["has_broken_code"]:
        return "code_generation"
    if not signals["has_execution_feedback"]:
        return "static_repair"
    return "execution_repair"


def best_model_for_task_type(
    by_model: dict[str, dict[str, dict]],
    common: list[str],
    sid_to_task_type: dict[str, str],
    task_type: str,
) -> tuple[str | None, float]:
    """Return (model_key, pass_rate) of the best model on the given task_type."""
    best_model = None
    best_rate = -1.0
    for key in by_model:
        tt_outcomes = [
            by_model[key][sid]
            for sid in common
            if sid_to_task_type.get(sid) == task_type and sid in by_model[key]
        ]
        if not tt_outcomes:
            continue
        rate = sum(1 for o in tt_outcomes if _passed(o)) / len(tt_outcomes)
        if rate > best_rate:
            best_rate = rate
            best_model = key
    return best_model, max(best_rate, 0.0)


# ---------------------------------------------------------------------------
# P3 Decision Gate
# ---------------------------------------------------------------------------

def apply_decision_gate(
    *,
    deployable_lift: float,
    oracle_lift: float,
    deployable_mcnemar_p: float,
    deployable_ci_lo: float,
    deployable_ci_hi: float,
    deployable_b: int,
    deployable_c: int,
    n_common: int,
) -> dict:
    """Apply the P3 Decision Gate criteria.

    Returns dict with:
      - verdict: "GO" | "NO-GO" | "SIGNAL"
      - criteria: dict of the raw thresholds and booleans used
      - reason: human-readable explanation string
    """
    LIFT_THRESHOLD = 0.05  # 5 percentage points

    oracle_meaningful = oracle_lift >= LIFT_THRESHOLD
    deployable_meaningful = deployable_lift >= LIFT_THRESHOLD
    ci_significant = (deployable_ci_lo > 0.0) or (deployable_mcnemar_p < 0.05)
    deployable_significant = deployable_meaningful and ci_significant

    if not oracle_meaningful:
        verdict = "NO-GO"
        reason = (
            f"Oracle Router lift ({oracle_lift*100:.1f}pp) < 5pp threshold — "
            "even the upper bound shows no meaningful routing gain."
        )
    elif deployable_significant:
        verdict = "GO"
        reason = (
            f"Deployable Router lift ({deployable_lift*100:.1f}pp) >= 5pp "
            f"with statistical significance (McNemar p={deployable_mcnemar_p:.4f}, "
            f"95% CI=[{deployable_ci_lo:+.4f}, {deployable_ci_hi:+.4f}])."
        )
    elif oracle_meaningful and not deployable_significant:
        verdict = "SIGNAL"
        reason = (
            f"Oracle lift ({oracle_lift*100:.1f}pp) >= 5pp (routing potential exists), "
            f"but Deployable Router lift ({deployable_lift*100:.1f}pp) or significance "
            f"(McNemar p={deployable_mcnemar_p:.4f}, CI=[{deployable_ci_lo:+.4f}, {deployable_ci_hi:+.4f}]) "
            "does not meet the GO threshold — observable signals alone cannot capture the potential."
        )
    else:
        verdict = "NO-GO"
        reason = (
            f"Deployable Router lift ({deployable_lift*100:.1f}pp) shows no significant "
            f"improvement (McNemar p={deployable_mcnemar_p:.4f}, CI=[{deployable_ci_lo:+.4f}, {deployable_ci_hi:+.4f}])."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "criteria": {
            "lift_threshold_pp": LIFT_THRESHOLD * 100,
            "oracle_lift": oracle_lift,
            "oracle_meaningful": oracle_meaningful,
            "deployable_lift": deployable_lift,
            "deployable_meaningful": deployable_meaningful,
            "deployable_mcnemar_p": deployable_mcnemar_p,
            "deployable_ci_95": [deployable_ci_lo, deployable_ci_hi],
            "deployable_ci_significant": ci_significant,
            "deployable_significant": deployable_significant,
            "deployable_mcnemar_b": deployable_b,
            "deployable_mcnemar_c": deployable_c,
            "n_common": n_common,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load policy artifact (hard-fail if missing) ----------------------
    policy = _load_policy()
    print(f"Loaded policy: {policy['policy_version']} from {_POLICY_PATH}")
    print(f"  selection: {policy['selection_family_count']} families / "
          f"{len(policy['selection_sample_ids'])} samples")
    print(f"  eval:      {policy['eval_family_count']} families / "
          f"{len(policy['eval_sample_ids'])} samples")

    # 2. Load eval JSONs (skip missing files) ------------------------------
    docs: dict[str, dict] = {}
    skipped: list[str] = []
    for key, _label in MODELS:
        d = _load(key)
        if d is None:
            skipped.append(key)
            print(f"WARN: evaluations/p2/{key}.json not found - skipping")
            continue
        docs[key] = d
    if len(docs) < 2:
        print("ERROR: need at least 2 model evals for router analysis")
        return 1

    # 3. Per-model sample lookup + common sample IDs -----------------------
    by_model: dict[str, dict[str, dict]] = {k: _outcomes_by_id(d) for k, d in docs.items()}
    sid_sets = {k: set(by_id.keys()) for k, by_id in by_model.items()}
    common_all = sorted(set.intersection(*sid_sets.values()))
    print(f"Models loaded: {list(docs)}")
    print(f"Skipped: {skipped}")
    print(f"Common sample IDs (all models): {len(common_all)}")

    # 4. Validate policy alignment with loaded evals (hard-fail on mismatch)
    _validate_policy_alignment(policy, docs, common_all)

    # 5. Restrict to eval subset (held-out, family-disjoint from selection)
    common = _filter_to_eval_subset(by_model, common_all, policy)
    print(f"Eval subset sample IDs (held-out): {len(common)}")
    if len(common) == 0:
        print("ERROR: eval subset is empty after filtering")
        return 1

    # 6. sid -> family_id / task_type (restricted to eval subset) ----------
    sid_to_family: dict[str, str] = {}
    sid_to_task_type: dict[str, str] = {}
    for sid in common:
        for by_id in by_model.values():
            if sid in by_id:
                sid_to_family[sid] = by_id[sid].get("family_id", "?")
                sid_to_task_type[sid] = by_id[sid].get("task_type", "unknown")
                break

    # ------------------------------------------------------------------
    # 7. Per-model stats on eval subset (for reporting)
    # ------------------------------------------------------------------
    per_model_stats: dict[str, dict] = {}
    for key in docs:
        pass_map = {sid: _passed(by_model[key][sid]) for sid in common}
        per_model_stats[key] = compute_router_stats(
            pass_map, sid_to_family, sid_to_task_type
        )

    # ------------------------------------------------------------------
    # 8. Best Single — use the FROZEN choice from policy
    # ------------------------------------------------------------------
    best_single_key = policy["best_single_model"]
    if best_single_key not in docs:
        raise SystemExit(
            f"ERROR: policy's best_single_model '{best_single_key}' is "
            "not among loaded evals."
        )
    best_single_stats = per_model_stats[best_single_key]
    best_single_overall = best_single_stats["overall_pass"]
    best_single = {
        "model": _label_of(best_single_key),
        "model_key": best_single_key,
        "source": "frozen_policy_v1",
        **best_single_stats,
    }

    # ------------------------------------------------------------------
    # 9. Oracle Router (upper bound — recomputed on eval subset, not "trained")
    # ------------------------------------------------------------------
    oracle_pass: dict[str, bool] = {}
    for sid in common:
        oracle_pass[sid] = any(_passed(by_model[k][sid]) for k in by_model)
    oracle_stats = compute_router_stats(oracle_pass, sid_to_family, sid_to_task_type)
    oracle_router = {
        **oracle_stats,
        "lift_vs_best_single": oracle_stats["overall_pass"] - best_single_overall,
    }

    # ------------------------------------------------------------------
    # 10. Metadata Router — apply FROZEN mapping (do NOT recompute)
    # ------------------------------------------------------------------
    metadata_routing_map = dict(policy["metadata_router_mapping"])
    metadata_pass: dict[str, bool] = {}
    for sid in common:
        tt = sid_to_task_type.get(sid, "unknown")
        model_key = metadata_routing_map.get(tt)
        if model_key and sid in by_model.get(model_key, {}):
            metadata_pass[sid] = _passed(by_model[model_key][sid])
        else:
            metadata_pass[sid] = False
    metadata_stats = compute_router_stats(
        metadata_pass, sid_to_family, sid_to_task_type
    )
    metadata_router = {
        **metadata_stats,
        "routing_map": {tt: _label_of(k) for tt, k in metadata_routing_map.items()},
        "routing_map_keys": metadata_routing_map,
        "source": "frozen_policy_v1",
        "selection_pass_rate": policy["selection_metrics"]["metadata_router_pass_rate"],
        "lift_vs_best_single": metadata_stats["overall_pass"] - best_single_overall,
    }

    # ------------------------------------------------------------------
    # 11. Deployable Router — apply FROZEN mapping (do NOT recompute)
    # ------------------------------------------------------------------
    # The deployment decision uses only observable signals (broken_code,
    # execution_feedback) inferred from task_type. The mapping from category
    # to model is loaded from the frozen policy (selected on held-out subset).
    deployable_routing_map = dict(policy["deployable_router_mapping"])

    routing_rules = [
        {
            "signal": "no broken_code, no execution_feedback",
            "category": "code_generation",
            "candidate": "Generalist (Best Single on code_gen)",
            "model": _label_of(deployable_routing_map.get("code_generation", "")),
        },
        {
            "signal": "broken_code present, no execution_feedback",
            "category": "static_repair",
            "candidate": "Static Repair Specialist",
            "model": _label_of(deployable_routing_map.get("static_repair", "")),
        },
        {
            "signal": "broken_code + execution_feedback",
            "category": "execution_repair",
            "candidate": "Repair Specialist",
            "model": _label_of(deployable_routing_map.get("execution_repair", "")),
        },
    ]

    # Apply routing using observable signals (inferred from task_type).
    deployable_pass: dict[str, bool] = {}
    for sid in common:
        tt = sid_to_task_type.get(sid, "unknown")
        signals = infer_observable_signals(tt)
        category = route_deployable(signals)
        model_key = deployable_routing_map.get(category)
        if model_key and sid in by_model.get(model_key, {}):
            deployable_pass[sid] = _passed(by_model[model_key][sid])
        else:
            deployable_pass[sid] = False
    deployable_stats = compute_router_stats(
        deployable_pass, sid_to_family, sid_to_task_type
    )
    deployable_router = {
        **deployable_stats,
        "routing_rules": routing_rules,
        "routing_map": {
            tt: _label_of(k) for tt, k in deployable_routing_map.items()
        },
        "routing_map_keys": deployable_routing_map,
        "source": "frozen_policy_v1",
        "selection_pass_rate": policy["selection_metrics"]["deployable_router_pass_rate"],
        "lift_vs_best_single": deployable_stats["overall_pass"] - best_single_overall,
    }

    # ------------------------------------------------------------------
    # 12. P3 Decision Gate — Deployable Router vs Best Single (eval subset)
    # ------------------------------------------------------------------
    # McNemar convention: A = Best Single, B = Deployable Router
    #   b = #samples where A passed, B failed
    #   c = #samples where A failed, B passed
    best_single_pass_map = {
        sid: _passed(by_model[best_single_key][sid]) for sid in common
    }
    gate_b = 0
    gate_c = 0
    pass_a: list[bool] = []  # best_single
    pass_b: list[bool] = []  # deployable
    for sid in common:
        bs_pass = best_single_pass_map[sid]
        dep_pass = deployable_pass[sid]
        if bs_pass and not dep_pass:
            gate_b += 1
        elif not bs_pass and dep_pass:
            gate_c += 1
        pass_a.append(bs_pass)
        pass_b.append(dep_pass)

    mcnemar_p = mcnemar_exact(gate_b, gate_c)
    ci_lo, ci_hi = paired_bootstrap_ci(pass_a, pass_b)
    gate_result = apply_decision_gate(
        deployable_lift=deployable_router["lift_vs_best_single"],
        oracle_lift=oracle_router["lift_vs_best_single"],
        deployable_mcnemar_p=mcnemar_p,
        deployable_ci_lo=ci_lo,
        deployable_ci_hi=ci_hi,
        deployable_b=gate_b,
        deployable_c=gate_c,
        n_common=len(common),
    )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    def _ptt(stats: dict, tt: str) -> float:
        return stats["per_task_type"].get(tt, {}).get("pass_rate", 0.0)

    comparison_table: list[dict] = []
    for key in docs:
        s = per_model_stats[key]
        comparison_table.append({
            "name": _label_of(key),
            "type": "single_model",
            "overall_pass": s["overall_pass"],
            "family_pass": s["family_pass"],
            "code_generation_pass": _ptt(s, "code_generation"),
            "static_repair_pass": _ptt(s, "static_repair"),
            "execution_repair_pass": _ptt(s, "execution_repair"),
            "lift_vs_best_single": s["overall_pass"] - best_single_overall,
        })
    comparison_table.append({
        "name": "Best Single",
        "type": "best_single",
        "overall_pass": best_single["overall_pass"],
        "family_pass": best_single["family_pass"],
        "code_generation_pass": _ptt(best_single, "code_generation"),
        "static_repair_pass": _ptt(best_single, "static_repair"),
        "execution_repair_pass": _ptt(best_single, "execution_repair"),
        "lift_vs_best_single": 0.0,
    })
    for name, router in [
        ("Oracle Router", oracle_router),
        ("Metadata Router", metadata_router),
        ("Deployable Router", deployable_router),
    ]:
        comparison_table.append({
            "name": name,
            "type": "router",
            "overall_pass": router["overall_pass"],
            "family_pass": router["family_pass"],
            "code_generation_pass": _ptt(router, "code_generation"),
            "static_repair_pass": _ptt(router, "static_repair"),
            "execution_repair_pass": _ptt(router, "execution_repair"),
            "lift_vs_best_single": router["lift_vs_best_single"],
        })

    # ------------------------------------------------------------------
    # Bug_type distribution (auxiliary, extracted from sample_id) --------
    # ------------------------------------------------------------------
    bug_type_dist: dict[str, int] = defaultdict(int)
    for sid in common:
        bt = extract_bug_type(sid)
        bug_type_dist[bt if bt else "(none)"] += 1

    # ------------------------------------------------------------------
    # 13. Assemble result — ADD policy fields, REMOVE train-on-test notes
    # ------------------------------------------------------------------
    try:
        policy_path_str = str(_POLICY_PATH.relative_to(_ROOT))
    except ValueError:
        # _POLICY_PATH may be patched outside _ROOT in tests.
        policy_path_str = str(_POLICY_PATH)

    result = {
        "policy_version": policy["policy_version"],
        "policy_path": policy_path_str,
        "selection_family_count": policy["selection_family_count"],
        "eval_family_count": policy["eval_family_count"],
        "selection_sample_count": len(policy["selection_sample_ids"]),
        "eval_sample_count": len(policy["eval_sample_ids"]),
        "eval_subset_size": len(common),
        "models_loaded": list(docs),
        "models_skipped": skipped,
        "common_sample_count": len(common),
        "best_single": best_single,
        "oracle_router": oracle_router,
        "metadata_router": metadata_router,
        "deployable_router": deployable_router,
        "decision_gate": gate_result,
        "comparison_table": comparison_table,
        "bug_type_distribution": dict(sorted(bug_type_dist.items())),
        "notes": [
            "Router policy (Best Single, Metadata, Deployable) loaded from "
            "frozen artifact router-policy-v1.json, fit on a held-out "
            "selection subset (45 families) using split_router_selection.py.",
            "This script applies that frozen policy to the eval subset "
            "(30 families, family-disjoint from selection). No selection-eval "
            "leakage.",
            "Oracle Router is recomputed on the eval subset (upper bound, "
            "not 'trained' on selection).",
            "Deployable Router uses only observable signals (broken_code, "
            "execution_feedback) inferred from task_type. Does not use hidden "
            "tests or gold bug_type.",
        ],
    }

    out_json = _OUT_DIR / "router-analysis.json"
    out_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved JSON: {out_json}")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    md: list[str] = []
    md.append("# P2 Router Feasibility Analysis")
    md.append("")
    md.append(f"- Models loaded: {', '.join(_label_of(k) for k in docs)}")
    if skipped:
        md.append(f"- Models skipped (file not found): {', '.join(skipped)}")
    md.append(f"- Common sample count: {len(common)}")
    md.append("- Sample passes iff `public_passed AND hidden_passed`.")
    md.append("- Family passes iff ALL its samples pass.")
    md.append("")
    md.append(
        "> **Methodology:** Router policy is frozen in `router-policy-v1.json` "
        "(fit on"
    )
    md.append(
        f"> {policy['selection_family_count']} selection families, evaluated here on "
        f"the held-out {policy['eval_family_count']} eval families —"
    )
    md.append("> family-disjoint). No selection-eval leakage.")
    md.append("")
    md.append("## Selection / Eval Split")
    md.append("")
    md.append(f"- Policy version: {policy['policy_version']}")
    md.append(
        f"- Selection families: {policy['selection_family_count']} "
        f"({len(policy['selection_sample_ids'])} samples) — used to fit routing maps"
    )
    md.append(
        f"- Eval families: {policy['eval_family_count']} "
        f"({len(policy['eval_sample_ids'])} samples) — held out, used here for evaluation"
    )
    md.append("- Selection ∩ Eval: empty (verified)")
    md.append(
        f"- Dataset SHA256: {policy['selection_dataset_sha256'][:16]}..."
        f"{policy['selection_dataset_sha256'][-8:]} (verified across all loaded evals)"
    )
    md.append("")

    # Comparison table
    md.append("## Comparison Table")
    md.append("")
    md.append(
        "| Model / Router | Type | Overall | Family | CodeGen | StaticRepair | "
        "ExecRepair | Lift vs Best |"
    )
    md.append(
        "|----------------|------|---------|--------|---------|--------------|"
        "------------|--------------|"
    )
    for row in comparison_table:
        md.append(
            f"| {row['name']} | {row['type']} | "
            f"{_pct(row['overall_pass'])} | {_pct(row['family_pass'])} | "
            f"{_pct(row['code_generation_pass'])} | "
            f"{_pct(row['static_repair_pass'])} | "
            f"{_pct(row['execution_repair_pass'])} | "
            f"{row['lift_vs_best_single']:+.4f} |"
        )
    md.append("")

    # Best single
    md.append("## 1. Best Single Model")
    md.append("")
    md.append(f"- **Model:** {best_single['model']} (`{best_single['model_key']}`)")
    md.append(f"- Overall pass: {_pct(best_single['overall_pass'])}")
    md.append(f"- Family pass: {_pct(best_single['family_pass'])}")
    md.append("- **Source:** frozen policy v1 (selected on held-out selection subset)")
    md.append("")
    md.append("### Per-task-type pass rates")
    md.append("")
    md.append("| Task Type | Total | Passed | Rate |")
    md.append("|-----------|-------|--------|------|")
    for tt in TASK_TYPES:
        s = best_single["per_task_type"].get(
            tt, {"total": 0, "passed": 0, "pass_rate": 0.0}
        )
        md.append(f"| {tt} | {s['total']} | {s['passed']} | {_pct(s['pass_rate'])} |")
    md.append("")
    md.append("### All single models")
    md.append("")
    md.append("| Model | Overall | Family | CodeGen | StaticRepair | ExecRepair |")
    md.append("|-------|---------|--------|---------|--------------|------------|")
    for key in docs:
        s = per_model_stats[key]
        md.append(
            f"| {_label_of(key)} | {_pct(s['overall_pass'])} | "
            f"{_pct(s['family_pass'])} | {_pct(_ptt(s, 'code_generation'))} | "
            f"{_pct(_ptt(s, 'static_repair'))} | "
            f"{_pct(_ptt(s, 'execution_repair'))} |"
        )
    md.append("")

    # Oracle
    md.append("## 2. Oracle Router (Upper Bound)")
    md.append("")
    md.append("Oracle passes a sample if ANY model passes it.")
    md.append("")
    md.append(f"- Overall pass: {_pct(oracle_router['overall_pass'])}")
    md.append(f"- Family pass: {_pct(oracle_router['family_pass'])}")
    md.append(f"- Lift vs Best Single: {oracle_router['lift_vs_best_single']:+.4f}")
    md.append("")
    md.append("| Task Type | Total | Passed | Rate |")
    md.append("|-----------|-------|--------|------|")
    for tt in TASK_TYPES:
        s = oracle_router["per_task_type"].get(
            tt, {"total": 0, "passed": 0, "pass_rate": 0.0}
        )
        md.append(f"| {tt} | {s['total']} | {s['passed']} | {_pct(s['pass_rate'])} |")
    md.append("")

    # Metadata
    md.append("## 3. Metadata Router (route by `task_type`)")
    md.append("")
    md.append("### Routing map")
    md.append("")
    md.append("| Task Type | Routed Model |")
    md.append("|-----------|-------------|")
    for tt in TASK_TYPES:
        if tt in metadata_routing_map:
            md.append(f"| {tt} | {_label_of(metadata_routing_map[tt])} |")
    md.append("")
    md.append(f"- Overall pass (eval subset): {_pct(metadata_router['overall_pass'])}")
    md.append(f"- Family pass: {_pct(metadata_router['family_pass'])}")
    md.append(
        f"- Selection pass rate (overall, frozen): "
        f"{_pct(metadata_router['selection_pass_rate'])}"
    )
    md.append(
        f"- Lift vs Best Single: {metadata_router['lift_vs_best_single']:+.4f}"
    )
    md.append("- **Source:** frozen policy v1 (selected on held-out selection subset)")
    md.append("")

    # Deployable
    md.append("## 4. Deployable Deterministic Router (observable signals only)")
    md.append("")
    md.append("### Routing rules")
    md.append("")
    md.append("| Observable Signal | Category | Candidate | Routed Model |")
    md.append("|------------------|----------|-----------|--------------|")
    for r in routing_rules:
        md.append(
            f"| {r['signal']} | {r['category']} | {r['candidate']} | {r['model']} |"
        )
    md.append("")
    md.append(f"- Overall pass: {_pct(deployable_router['overall_pass'])}")
    md.append(f"- Family pass: {_pct(deployable_router['family_pass'])}")
    md.append(
        f"- Selection pass rate (overall, frozen): "
        f"{_pct(deployable_router['selection_pass_rate'])}"
    )
    md.append(
        f"- Lift vs Best Single: {deployable_router['lift_vs_best_single']:+.4f}"
    )
    md.append("- **Source:** frozen policy v1 (selected on held-out selection subset)")
    md.append("")

    # Bug type distribution
    md.append("## Appendix: Bug Type Distribution (extracted from sample_id)")
    md.append("")
    md.append("| Bug Type | Count |")
    md.append("|----------|-------|")
    for bt, count in sorted(bug_type_dist.items()):
        md.append(f"| {bt} | {count} |")
    md.append("")

    # Methodology
    md.append("## Methodology Notes")
    md.append("")
    md.append("- **Best Single:** model with highest overall pass rate.")
    md.append(
        "- **Oracle Router:** upper bound; passes if ANY model passes the sample."
    )
    md.append(
        "- **Metadata Router:** uses `task_type` metadata to pick best model "
        "per task_type."
    )
    md.append("- **Deployable Router:** uses only deployment-observable signals:")
    md.append(
        "  - `broken_code` presence (inferred from task_type being "
        "static_repair or execution_repair)."
    )
    md.append(
        "  - `execution_feedback` presence (inferred from task_type being "
        "execution_repair)."
    )
    md.append("  - Request category (generation vs repair).")
    md.append(
        "- **No leakage:** Deployable Router does not use hidden tests, gold "
        "answers, or gold bug_type."
    )
    md.append(
        "- **Methodology:** routing maps loaded from frozen policy artifact "
        "(`router-policy-v1.json`). Selection and eval subsets are family-disjoint. "
        "No selection-eval leakage."
    )
    md.append("")

    # P3 Decision Gate
    md.append("## P3 Decision Gate")
    md.append("")
    md.append(f"**Verdict: {gate_result['verdict']}**")
    md.append("")
    md.append(f"{gate_result['reason']}")
    md.append("")
    md.append("### Gate Criteria")
    md.append("")
    md.append("| Criterion | Value | Threshold | Met |")
    md.append("|-----------|-------|-----------|-----|")
    c = gate_result["criteria"]
    md.append(f"| Oracle lift vs Best Single | {c['oracle_lift']*100:.1f}pp | >= 5.0pp | {'YES' if c['oracle_meaningful'] else 'NO'} |")
    md.append(f"| Deployable lift vs Best Single | {c['deployable_lift']*100:.1f}pp | >= 5.0pp | {'YES' if c['deployable_meaningful'] else 'NO'} |")
    md.append(f"| Deployable McNemar p (2-sided) | {c['deployable_mcnemar_p']:.4f} | < 0.05 | {'YES' if c['deployable_mcnemar_p'] < 0.05 else 'NO'} |")
    md.append(f"| Deployable 95% CI | [{c['deployable_ci_95'][0]:+.4f}, {c['deployable_ci_95'][1]:+.4f}] | lower > 0 | {'YES' if c['deployable_ci_95'][0] > 0 else 'NO'} |")
    md.append(f"| Deployable b/c (McNemar) | {c['deployable_mcnemar_b']}/{c['deployable_mcnemar_c']} | — | — |")
    md.append(f"| Common samples | {c['n_common']} | — | — |")
    md.append("")

    out_md = _OUT_DIR / "router-analysis.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
