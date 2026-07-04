"""Holdout router policy generation for P2 router analysis.

Splits the 75 Full-576 eval families into disjoint selection (45, 60%) and
eval (30, 40%) subsets using a deterministic seed (42). Router rules
(Best Single, Metadata Router, Deployable Router) are determined on the
**selection subset only**, then frozen into a versioned artifact
``reports/p2/router-policy-v1.json``. The eval subset is held out so that
lift/CI computed downstream (Task 2) is free of train-on-test bias.

Reuses routing primitives from ``scripts.compute_router_analysis``:
  - ``_passed`` — sample passes iff public_passed AND hidden_passed
  - ``_outcomes_by_id`` — sample_id -> outcome dict
  - ``infer_observable_signals`` / ``route_deployable`` — observable-signal routing
  - ``MODELS`` / ``TASK_TYPES`` — canonical model and task-type lists

Inputs:
  evaluations/p2/full576-{base,stage2-boundary,stage3-repair,independent-stage3,stage3-v3-antiforget}.json
  data/p2-curriculum/frozen-eval-v2/test_raw.jsonl (for dataset SHA256)

Output:
  reports/p2/router-policy-v1.json
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.compute_router_analysis import (
    MODELS,
    TASK_TYPES,
    _outcomes_by_id,
    _passed,
    infer_observable_signals,
    route_deployable,
)

_EVAL_DIR = _ROOT / "evaluations" / "p2"
_OUT_DIR = _ROOT / "reports" / "p2"
_FROZEN_EVAL = _ROOT / "data" / "p2-curriculum" / "frozen-eval-v2" / "test_raw.jsonl"

SEED = 42
SELECTION_RATIO = 0.60  # 60% selection, 40% eval
POLICY_VERSION = "v1"


# ---------------------------------------------------------------------------
# Family split
# ---------------------------------------------------------------------------

def compute_split(
    families: list[str], seed: int = SEED
) -> tuple[list[str], list[str]]:
    """Deterministically split families into (selection, eval) subsets.

    Sorts families alphabetically, then uses ``random.Random(seed).sample``
    to draw the selection subset. If the 60% split is uneven, the larger
    portion goes to selection (via ``math.ceil``).

    Hard guarantee: the two subsets are disjoint and partition the input.
    """
    sorted_families = sorted(families)
    n_total = len(sorted_families)
    n_selection = math.ceil(n_total * SELECTION_RATIO)
    n_eval = n_total - n_selection

    rng = random.Random(seed)
    selection = sorted(rng.sample(sorted_families, n_selection))
    selection_set = set(selection)
    eval_ = sorted([f for f in sorted_families if f not in selection_set])

    # Hard assertion: family isolation.
    assert set(selection).isdisjoint(set(eval_)), "selection and eval families overlap"
    assert len(selection) + len(eval_) == n_total, "split does not cover all families"
    assert len(selection) == n_selection, (
        f"expected {n_selection} selection families, got {len(selection)}"
    )
    assert len(eval_) == n_eval, f"expected {n_eval} eval families, got {len(eval_)}"
    return selection, eval_


# ---------------------------------------------------------------------------
# Best model selection (with MODELS-list tie-breaking)
# ---------------------------------------------------------------------------

def _best_model_for_subset(
    by_model: dict[str, dict[str, dict]],
    candidate_sids: list[str],
) -> tuple[str | None, float]:
    """Return (model_key, pass_rate) of the best model on candidate_sids.

    Iterates models in ``MODELS`` order so ties are broken by earlier position.
    """
    best_key: str | None = None
    best_rate = -1.0
    for key, _label in MODELS:
        if key not in by_model or not candidate_sids:
            continue
        passes = sum(1 for sid in candidate_sids if _passed(by_model[key][sid]))
        rate = passes / len(candidate_sids)
        if rate > best_rate:
            best_rate = rate
            best_key = key
    return best_key, max(best_rate, 0.0)


def _pass_rate(pass_map: dict[str, bool]) -> float:
    if not pass_map:
        return 0.0
    return sum(1 for v in pass_map.values() if v) / len(pass_map)


# ---------------------------------------------------------------------------
# Policy computation (pure function — testable without disk IO)
# ---------------------------------------------------------------------------

def compute_policy(
    docs: dict[str, dict], dataset_sha256: str
) -> dict[str, Any]:
    """Compute the router policy artifact from a dict of {model_key: eval_doc}.

    Uses sample IDs common to ALL provided model docs (intersection), matching
    the fairness convention in ``compute_router_analysis.main``.

    Parameters
    ----------
    docs : dict[str, dict]
        Mapping of model_key -> eval document (each has an ``outcomes`` array).
    dataset_sha256 : str
        SHA256 of the frozen eval dataset (computed by the caller from disk).

    Returns
    -------
    dict
        The router policy artifact conforming to the schema in the task brief.
    """
    by_model: dict[str, dict[str, dict]] = {
        k: _outcomes_by_id(d) for k, d in docs.items()
    }

    # Common sample IDs = intersection across all loaded models (fair comparison).
    sid_sets = [set(by_id.keys()) for by_id in by_model.values()]
    common = sorted(set.intersection(*sid_sets)) if sid_sets else []

    # sid -> family_id / task_type (from any model that has the sample).
    sid_to_family: dict[str, str] = {}
    sid_to_task_type: dict[str, str] = {}
    for sid in common:
        for by_id in by_model.values():
            if sid in by_id:
                sid_to_family[sid] = by_id[sid].get("family_id", "?")
                sid_to_task_type[sid] = by_id[sid].get("task_type", "unknown")
                break

    all_families = sorted(set(sid_to_family.values()))
    selection_families, eval_families = compute_split(all_families)
    selection_set = set(selection_families)

    # Selection subset of samples; eval subset is the remainder.
    selection_sids = [sid for sid in common if sid_to_family[sid] in selection_set]
    eval_sids = [sid for sid in common if sid_to_family[sid] not in selection_set]

    # ------------------------------------------------------------------
    # Best Single (highest overall pass rate on selection subset).
    # ------------------------------------------------------------------
    best_single_key, best_single_rate = _best_model_for_subset(
        by_model, selection_sids
    )

    # ------------------------------------------------------------------
    # Metadata Router mapping: best model per task_type on selection subset.
    # ------------------------------------------------------------------
    metadata_router_mapping: dict[str, str] = {}
    for tt in TASK_TYPES:
        tt_sids = [
            sid for sid in selection_sids if sid_to_task_type.get(sid) == tt
        ]
        best_key, _rate = _best_model_for_subset(by_model, tt_sids)
        if best_key is not None:
            metadata_router_mapping[tt] = best_key

    # ------------------------------------------------------------------
    # Deployable Router mapping: best model per route_deployable category.
    # Category is derived from observable signals (no gold task_type label
    # used at deployment time). Since route_deployable is a deterministic
    # function of task_type, the categories are code_generation /
    # static_repair / execution_repair — but we group by the *derived*
    # category, not by task_type directly.
    # ------------------------------------------------------------------
    sids_by_category: dict[str, list[str]] = defaultdict(list)
    for sid in selection_sids:
        tt = sid_to_task_type.get(sid, "unknown")
        signals = infer_observable_signals(tt)
        category = route_deployable(signals)
        sids_by_category[category].append(sid)

    deployable_router_mapping: dict[str, str] = {}
    for category, cat_sids in sids_by_category.items():
        best_key, _rate = _best_model_for_subset(by_model, cat_sids)
        if best_key is not None:
            deployable_router_mapping[category] = best_key

    # ------------------------------------------------------------------
    # Apply each strategy on the selection subset and record pass rates.
    # ------------------------------------------------------------------
    best_single_pass = {
        sid: _passed(by_model[best_single_key][sid]) for sid in selection_sids
    }
    best_single_rate = _pass_rate(best_single_pass)

    metadata_pass: dict[str, bool] = {}
    for sid in selection_sids:
        tt = sid_to_task_type.get(sid, "unknown")
        model_key = metadata_router_mapping.get(tt)
        if model_key and sid in by_model.get(model_key, {}):
            metadata_pass[sid] = _passed(by_model[model_key][sid])
        else:
            metadata_pass[sid] = False
    metadata_rate = _pass_rate(metadata_pass)

    deployable_pass: dict[str, bool] = {}
    for sid in selection_sids:
        tt = sid_to_task_type.get(sid, "unknown")
        signals = infer_observable_signals(tt)
        category = route_deployable(signals)
        model_key = deployable_router_mapping.get(category)
        if model_key and sid in by_model.get(model_key, {}):
            deployable_pass[sid] = _passed(by_model[model_key][sid])
        else:
            deployable_pass[sid] = False
    deployable_rate = _pass_rate(deployable_pass)

    oracle_pass = {
        sid: any(_passed(by_model[k][sid]) for k in by_model)
        for sid in selection_sids
    }
    oracle_rate = _pass_rate(oracle_pass)

    # ------------------------------------------------------------------
    # Assemble policy artifact.
    # ------------------------------------------------------------------
    policy: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_dataset_sha256": dataset_sha256,
        "selection_family_count": len(selection_families),
        "eval_family_count": len(eval_families),
        "selection_families": selection_families,
        "eval_families": eval_families,
        "selection_sample_ids": selection_sids,
        "eval_sample_ids": eval_sids,
        "models": [k for k, _ in MODELS],
        "best_single_model": best_single_key,
        "metadata_router_mapping": metadata_router_mapping,
        "deployable_router_mapping": deployable_router_mapping,
        "selection_metrics": {
            "best_single_pass_rate": best_single_rate,
            "metadata_router_pass_rate": metadata_rate,
            "deployable_router_pass_rate": deployable_rate,
            "oracle_router_pass_rate": oracle_rate,
        },
    }
    return policy


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def compute_sha256(path: Path) -> str:
    """Compute the SHA256 of a file, reading in binary mode."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all 5 model evals.
    docs: dict[str, dict] = {}
    for key, _label in MODELS:
        f = _EVAL_DIR / f"{key}.json"
        if not f.exists():
            print(f"ERROR: {f} not found")
            return 1
        with f.open(encoding="utf-8") as fh:
            docs[key] = json.load(fh)

    if not _FROZEN_EVAL.exists():
        print(f"ERROR: frozen eval dataset not found at {_FROZEN_EVAL}")
        return 1

    dataset_sha = compute_sha256(_FROZEN_EVAL)
    policy = compute_policy(docs, dataset_sha256=dataset_sha)

    out = _OUT_DIR / "router-policy-v1.json"
    out.write_text(
        json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {out}")
    print(f"Best Single: {policy['best_single_model']}")
    print(
        f"Families: {policy['selection_family_count']} selection / "
        f"{policy['eval_family_count']} eval"
    )
    print(f"Dataset SHA256: {dataset_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
