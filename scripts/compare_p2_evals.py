"""Compare P2 evaluation results across models, with per-task-type breakdown."""
import json
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    ("base", "Base"),
    ("stage1-code", "Stage1-Code"),
    ("stage2-boundary", "Stage2-Boundary"),
    ("stage3-repair", "Stage3-Repair"),
]


def load_eval(name: str) -> dict:
    f = _ROOT / "evaluations" / "p2" / f"{name}.json"
    if not f.exists():
        return None
    return json.load(open(f))


def per_task_type(outcomes: list) -> dict:
    """Break down pass rate by task_type."""
    stats = defaultdict(lambda: {"total": 0, "passed": 0, "syntax_ok": 0, "format_ok": 0})
    for o in outcomes:
        t = o.get("task_type", "unknown")
        stats[t]["total"] += 1
        # For generation: public_passed. For repair: public_passed too.
        if o.get("public_passed") and o.get("hidden_passed"):
            stats[t]["passed"] += 1
        if o.get("syntax_ok"):
            stats[t]["syntax_ok"] += 1
        if o.get("format_ok"):
            stats[t]["format_ok"] += 1
    return dict(stats)


def per_family(outcomes: list) -> dict:
    """Aggregate per family: a family passes if ALL its samples pass."""
    fam_outcomes = defaultdict(lambda: {"total": 0, "passed": 0})
    for o in outcomes:
        f = o.get("family_id", "unknown")
        fam_outcomes[f]["total"] += 1
        if o.get("public_passed") and o.get("hidden_passed"):
            fam_outcomes[f]["passed"] += 1
    # Family passes only if all samples pass
    fam_pass = {}
    for f, s in fam_outcomes.items():
        fam_pass[f] = (s["passed"] == s["total"] and s["total"] > 0)
    return fam_pass


def main() -> None:
    results = {}
    for name, label in MODELS:
        d = load_eval(name)
        if d is None:
            continue
        outcomes = d.get("outcomes", [])
        results[name] = {
            "label": label,
            "metrics": d["metrics"],
            "per_task_type": per_task_type(outcomes),
            "family_pass": per_family(outcomes),
            "outcomes": outcomes,
        }

    # Print per-task-type comparison
    print("=" * 90)
    print("Per-Task-Type Comparison")
    print("=" * 90)
    task_types = ["code_generation", "static_repair", "execution_repair"]
    for t in task_types:
        print(f"\n--- {t} ---")
        print(f"{'Model':<20} {'Total':>6} {'Passed':>8} {'Rate':>8} {'Syntax':>8} {'Format':>8}")
        for name, _ in MODELS:
            if name not in results:
                continue
            tt = results[name]["per_task_type"].get(t, {})
            total = tt.get("total", 0)
            passed = tt.get("passed", 0)
            syn = tt.get("syntax_ok", 0)
            fmt = tt.get("format_ok", 0)
            rate = passed / total if total > 0 else 0
            print(f"{results[name]['label']:<20} {total:>6} {passed:>8} {rate:>8.3f} {syn/total if total else 0:>8.3f} {fmt/total if total else 0:>8.3f}")

    # Family-level pass comparison
    print("\n" + "=" * 90)
    print("Family-Level Pass Comparison")
    print("=" * 90)
    for name, _ in MODELS:
        if name not in results:
            continue
        fp = results[name]["family_pass"]
        passed = sum(1 for v in fp.values() if v)
        total = len(fp)
        print(f"{results[name]['label']:<20} {passed}/{total} families passed ({passed/total*100 if total else 0:.1f}%)")

    # Per-family delta (Stage3 vs Base)
    if "base" in results and "stage3-repair" in results:
        base_fp = results["base"]["family_pass"]
        s3_fp = results["stage3-repair"]["family_pass"]
        improved = []
        regressed = []
        for f in base_fp:
            if f in s3_fp:
                if s3_fp[f] and not base_fp[f]:
                    improved.append(f)
                elif not s3_fp[f] and base_fp[f]:
                    regressed.append(f)
        print(f"\nStage3 vs Base:")
        print(f"  New passing families: {len(improved)}")
        print(f"  Regressed families:   {len(regressed)}")
        print(f"  Net improvement:      {len(improved) - len(regressed)}")

    # Error category analysis for repair samples
    print("\n" + "=" * 90)
    print("Error Category Analysis (Repair samples)")
    print("=" * 90)
    for name, _ in MODELS:
        if name not in results:
            continue
        repair_outcomes = [o for o in results[name]["outcomes"] if "repair" in o.get("task_type", "")]
        if not repair_outcomes:
            continue
        # Count by bug type (extracted from sample_id)
        bug_types = defaultdict(lambda: {"total": 0, "passed": 0})
        for o in repair_outcomes:
            sid = o.get("sample_id", "")
            # Extract bug type from sample_id like mbpp_611_sr_return_value_error
            parts = sid.split("_")
            if "sr" in parts:
                idx = parts.index("sr")
                bug = "_".join(parts[idx+1:]) if idx+1 < len(parts) else "unknown"
            elif "er" in parts:
                idx = parts.index("er")
                bug = "_".join(parts[idx+1:]) if idx+1 < len(parts) else "unknown"
            else:
                bug = "unknown"
            bug_types[bug]["total"] += 1
            if o.get("public_passed") and o.get("hidden_passed"):
                bug_types[bug]["passed"] += 1
        print(f"\n{results[name]['label']}:")
        for bug, s in sorted(bug_types.items()):
            rate = s["passed"] / s["total"] if s["total"] else 0
            print(f"  {bug:<30} {s['passed']}/{s['total']} ({rate:.3f})")

    # Save comparison JSON
    comparison = {}
    for name, _ in MODELS:
        if name not in results:
            continue
        comparison[name] = {
            "label": results[name]["label"],
            "metrics": results[name]["metrics"],
            "per_task_type": results[name]["per_task_type"],
            "family_pass_count": sum(1 for v in results[name]["family_pass"].values() if v),
            "family_total": len(results[name]["family_pass"]),
        }
    out = _ROOT / "evaluations" / "p2" / "comparison.json"
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved comparison to {out}")


if __name__ == "__main__":
    main()
