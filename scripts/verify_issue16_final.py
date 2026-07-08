"""Verify all Issue #16 acceptance criteria after frozen v4 evaluation.

Acceptance criteria:
1. train ∩ frozen_v4 sample_ids = 0
2. train ∩ frozen_v4 family_ids = 0
3. scored_frozen_v4_n_total = 365
4. canary_count = 100, canary_passed = true, canary excluded from scored metrics
5. adapter metrics eval_hash matches committed validation-v2 file
6. frozen eval input hash matches committed frozen-v4 test_raw.jsonl
"""
import json, hashlib, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_jsonl(path):
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]

def main():
    results = []

    # --- Criteria 1 & 2: Frozen exclusion in manifests ---
    for candidate in ["balanced-limited", "repair-limited"]:
        manifest = json.loads((_ROOT / f"data/p3-limited/{candidate}/manifest.json").read_text())
        fe = manifest["frozen_exclusion"]
        ok1 = fe["frozen_sample_ids_in_train"] == 0
        ok2 = fe["frozen_family_ids_in_train"] == 0
        results.append((f"[{candidate}] train ∩ frozen_v4 sample_ids = 0", ok1, fe["frozen_sample_ids_in_train"]))
        results.append((f"[{candidate}] train ∩ frozen_v4 family_ids = 0", ok2, fe["frozen_family_ids_in_train"]))

    # --- Criteria 5: adapter metrics eval_hash matches validation-v2 file ---
    val_v2_hash = sha256_file(_ROOT / "data/p3-curriculum/validation-v2/validation.jsonl")
    for candidate in ["balanced-limited", "repair-limited"]:
        metrics = json.loads((_ROOT / f"adapters/p3/{candidate}/metrics.json").read_text())
        ok = metrics["eval_hash"] == val_v2_hash
        results.append((f"[{candidate}] eval_hash matches validation-v2", ok,
                        f"{metrics['eval_hash'][:16]}... vs {val_v2_hash[:16]}..."))

    # --- Criteria 3, 4, 6: Frozen v4 eval results ---
    frozen_v4_hash = sha256_file(_ROOT / "data/frozen-eval/v4/test_raw.jsonl")
    frozen_manifest = json.loads((_ROOT / "data/frozen-eval/v4/manifest.json").read_text())

    for candidate in ["balanced-limited", "repair-limited"]:
        eval_path = _ROOT / f"evaluations/p3-limited/{candidate}-frozen-v4.json"
        if not eval_path.exists():
            results.append((f"[{candidate}] frozen-v4 eval file exists", False, "file not found"))
            continue

        eval_result = json.loads(eval_path.read_text())

        # Criterion 3: scored_frozen_v4_n_total = 365
        scored_n = eval_result.get("scored_sample_count", eval_result.get("metrics", {}).get("n_total", -1))
        ok3 = scored_n == 365
        results.append((f"[{candidate}] scored_frozen_v4_n_total = 365", ok3, f"actual: {scored_n}"))

        # Criterion 4: canary_count = 100, canary_passed = true, canary excluded
        canary = eval_result.get("canary", {})
        canary_passed = canary.get("passed", False)
        canary_excluded = eval_result.get("canary_excluded_count", 0)
        ok4a = canary_passed is True
        ok4b = canary_excluded == 100
        results.append((f"[{candidate}] canary_passed = true", ok4a, f"actual: {canary_passed}"))
        results.append((f"[{candidate}] canary_excluded_count = 100", ok4b, f"actual: {canary_excluded}"))

        # Criterion 6: frozen eval input hash matches committed frozen-v4 test_raw.jsonl
        eval_dataset_sha = eval_result.get("dataset_sha256", "")
        ok6 = eval_dataset_sha == frozen_v4_hash
        results.append((f"[{candidate}] frozen eval input hash matches test_raw.jsonl", ok6,
                        f"{eval_dataset_sha[:16]}... vs {frozen_v4_hash[:16]}..."))

    # --- Summary ---
    print("\n" + "=" * 80)
    print("Issue #16 Acceptance Criteria Verification")
    print("=" * 80)
    all_pass = True
    for desc, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {desc}" + (f" ({detail})" if detail else ""))

    print("=" * 80)
    if all_pass:
        print("ALL CRITERIA PASSED — Issue #16 is resolved.")
    else:
        print("SOME CRITERIA FAILED — Issue #16 is NOT resolved.")
    print("=" * 80)

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
