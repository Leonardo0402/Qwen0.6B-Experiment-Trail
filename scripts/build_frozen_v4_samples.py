"""scripts/build_frozen_v4_samples.py -- Build Frozen v4 = v3 + repair buckets."""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample, Verification
from src.validators import verify_sample, verify_broken_is_broken
from scripts.build_execution_repair import build_repair_samples

SCHEMA_VERSION = 1
GENERATOR_NAME = "build_frozen_v4_samples.py"
SEED = 42
CANARY_CODE = 'def canary_always_fails():\n    raise AssertionError("canary")'
_PLACEHOLDER_VER = Verification(syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False)

V3_DIR = _ROOT / "data" / "frozen-eval" / "v3"
V4_DIR = _ROOT / "data" / "frozen-eval" / "v4"

def _load_jsonl(path):
    samples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(Sample.from_json_line(line))
    return samples

def _write_jsonl(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(s.to_json_line() + "\n")

def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    # Step 1: Load v3 content
    v3_samples = _load_jsonl(V3_DIR / "test_raw.jsonl")
    v3_families = json.loads((V3_DIR / "families.json").read_text(encoding="utf-8"))
    family_ids = v3_families["families"]
    print(f"Loaded v3: {len(v3_samples)} samples, {len(family_ids)} families")
    
    # Extract code samples for repair generation
    code_samples = [s for s in v3_samples if s.variant_type == "code"]
    print(f"Code samples for repair generation: {len(code_samples)}")
    
    # Step 2: Generate repair samples
    new_repair_samples = []
    rejected = []
    sr_count = 0
    er_count = 0
    
    for sample in code_samples:
        try:
            pairs = build_repair_samples(sample, timeout_s=10.0, seed=SEED)
        except Exception as e:
            rejected.append({
                "sample_id": sample.sample_id,
                "family_id": sample.family_id,
                "rejection_reason": f"build_repair_samples_error: {e}",
            })
            continue
        
        if not pairs:
            rejected.append({
                "sample_id": sample.sample_id,
                "family_id": sample.family_id,
                "rejection_reason": "no_failing_bug_variants",
            })
            continue
        
        for sr, er in pairs:
            # Verify static_repair
            if sr is not None:
                sr_verified = verify_sample(sr, pytest_timeout_s=10.0)
                sr_broken = False
                try:
                    sr_broken = verify_broken_is_broken(sr, pytest_timeout_s=10.0)
                except ValueError:
                    sr_broken = False
                
                sr_updated = sr.model_copy(update={
                    "verified": sr_verified.is_accepted,
                    "verification": sr_verified.verification,
                    "variant_type": "static_repair",
                })
                
                if sr_verified.is_accepted and sr_broken:
                    new_repair_samples.append(sr_updated)
                    sr_count += 1
                else:
                    rejected.append({
                        "sample_id": sr.sample_id,
                        "family_id": sr.family_id,
                        "variant_type": "static_repair",
                        "rejection_reason": f"sr_verify={sr_verified.is_accepted}, sr_broken={sr_broken}",
                    })
            
            # Verify execution_repair
            if er is not None:
                er_verified = verify_sample(er, pytest_timeout_s=10.0)
                er_broken = False
                try:
                    er_broken = verify_broken_is_broken(er, pytest_timeout_s=10.0)
                except ValueError:
                    er_broken = False
                
                er_updated = er.model_copy(update={
                    "verified": er_verified.is_accepted,
                    "verification": er_verified.verification,
                    "variant_type": "execution_repair",
                })
                
                if er_verified.is_accepted and er_broken:
                    new_repair_samples.append(er_updated)
                    er_count += 1
                else:
                    rejected.append({
                        "sample_id": er.sample_id,
                        "family_id": er.family_id,
                        "variant_type": "execution_repair",
                        "rejection_reason": f"er_verify={er_verified.is_accepted}, er_broken={er_broken}",
                    })
    
    print(f"Repair samples generated: static_repair={sr_count}, execution_repair={er_count}")
    print(f"Rejected: {len(rejected)}")
    
    # Step 3: Generate canary_repair samples (top 30 families by family_id)
    canary_repair_samples = []
    sorted_code = sorted(code_samples, key=lambda s: s.family_id)[:30]
    for sample in sorted_code:
        canary_repair = Sample(
            sample_id=f"{sample.sample_id}_canary_repair",
            family_id=sample.family_id,
            difficulty=sample.difficulty,
            task_type="static_repair",
            language="python",
            skill_tags=list(sample.skill_tags),
            instruction=sample.instruction + "\n\n以下代码存在错误，请找出并修复，使其能通过所有测试用例。",
            broken_code=CANARY_CODE,
            execution_feedback=None,
            target_code=sample.target_code,
            public_tests=sample.public_tests,
            hidden_tests=sample.hidden_tests,
            verified=False,
            verification=_PLACEHOLDER_VER,
            generator=GENERATOR_NAME,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_version="v4",
            variant_type="canary_repair",
        )
        canary_repair_samples.append(canary_repair)
    
    print(f"Canary repair samples: {len(canary_repair_samples)}")
    
    # Step 4: Combine and write v4
    V4_DIR.mkdir(parents=True, exist_ok=True)
    
    all_v4_samples = v3_samples + new_repair_samples + canary_repair_samples
    _write_jsonl(V4_DIR / "test_raw.jsonl", all_v4_samples)
    
    # Write families.json (same as v3)
    v4_families = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "frozen_version": "v4",
        "base_version": "v3",
        "frozen_family_count": len(family_ids),
        "families": family_ids,
    }
    (V4_DIR / "families.json").write_text(
        json.dumps(v4_families, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    
    # Write rejected.jsonl
    with (V4_DIR / "rejected.jsonl").open("w", encoding="utf-8") as fh:
        for r in rejected:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Count variants
    variant_breakdown = {}
    for s in all_v4_samples:
        vt = s.variant_type or "unknown"
        variant_breakdown[vt] = variant_breakdown.get(vt, 0) + 1
    
    # Write manifest
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "frozen_version": "v4",
        "base_version": "v3",
        "generator": GENERATOR_NAME,
        "frozen_family_count": len(family_ids),
        "total_sample_count": len(all_v4_samples),
        "variant_breakdown": variant_breakdown,
        "v3_content_preserved": True,
        "v3_sample_count": len(v3_samples),
        "repair_buckets_added": True,
        "repair_sample_counts": {
            "static_repair": sr_count,
            "execution_repair": er_count,
            "canary_repair": len(canary_repair_samples),
        },
        "generation_environment": f"Python {sys.version_info.major}.{sys.version_info.minor} (ast.unparse required)",
        "test_raw_sha256": _sha256(V4_DIR / "test_raw.jsonl"),
        "families_sha256": _sha256(V4_DIR / "families.json"),
        "rejected_count": len(rejected),
        "rejected_sha256": _sha256(V4_DIR / "rejected.jsonl") if rejected else "",
        "immutability": {
            "write_once": True,
            "any_change_requires": "v5",
        },
    }
    (V4_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    
    print(f"\nv4 built successfully:")
    print(f"  Total samples: {len(all_v4_samples)}")
    print(f"  Variant breakdown: {variant_breakdown}")
    print(f"  Rejected: {len(rejected)}")
    print(f"  Output: {V4_DIR}")

if __name__ == "__main__":
    sys.exit(main())
