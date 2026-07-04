# P3 Capability Expansion v2 — Implementation Plan

Branch: `feat/p3-capability-expansion-v2`
Issue: #9
Started: 2026-07-04
Scope: P3.0–P3.4 only (data + tests + Readiness Gate). NO full training this session.

## Global Constraints (binding)

1. MBPP only — no HumanEval/LeetCode/MBPP-sanitized (AGENTS.md L30)
2. RTX 3050 4GB VRAM
3. Assistant-only loss; System/User/Pad tokens excluded
4. Assistant target answers 100% preserved (no silent truncation)
5. Continual chain integrity (parent adapter DAG)
6. Frozen eval immutability (v3 write-once)
7. Adapters append-only (no overwrite)
8. Train/Val/Frozen families pairwise disjoint (not just 3-way empty)
9. P2 Train replay is the ONLY allowed historical intersection (explicit whitelist)
10. Canonical verified sample pool (no direct stage manifest concat, per-family cap)
11. 2300–3100 = final effective train.jsonl samples per candidate (no duplication to pad)
12. Frozen v3: public≥2, hidden≥3; verify_broken_is_broken; real execution_feedback
13. Checkpoint 3-tier: 25-50 steps loss / 0.25 epoch probe (60-90 family-stratified) / 1 epoch full val
14. Best checkpoint by full Validation Composite only (never frozen v3)
15. Cross-split dedup: unresolved=0 before partition; unconfirmed high-similarity → quarantine
16. HF pin source revision + fingerprint + normalized SHA + verified SHA
17. Benchmark contamination: test split reused → manifest + reports mark `benchmark_contaminated=true`, `standard_mbpp_test_claims_disallowed=true`
18. Two candidates share identical Train/Validation/Frozen family partition (only ratios differ)
19. Sample schema: add optional `variant_type`/`bug_type`/`source_split` (backward compatible); no new boundary task_type
20. import + verify split: importer only normalizes (verified=false); verify_imported_mbpp.py runs real validation

## Strict Execution Order (不可乱序)

```
1. baseline lock
2. schema extend
3. import_mbpp.py fix + verify_imported_mbpp.py new
4. run import + verify + source audit
5. cross-split dedup audit (unresolved=0 required)
6. family registry
7. frozen v3 candidate reservation (120 candidates)
8. frozen v3 samples + verify + freeze (80-100 final)
9. P3 validation + train family partition (pairwise disjoint + P2 replay whitelist)
10. canonical verified sample pool (sample_id dedup + per-family cap)
11. balanced generalist train data (30/20/20/30)
12. repair specialist train data (15/15/30/40)
13. training config + 3-tier checkpoint evaluator
14. readiness gate report (GO/FIX FIRST) — SESSION END
```

## Task Details

### Task 1: Lock Historical Baseline
- File: `reports/p3/p3-baseline-lock.json`
- Record for Base / Stage3-Independent / Stage3-v3-Antiforget:
  - adapter_path, weight_sha256, config_sha256
  - historical eval set sha256 (frozen-eval-v2)
  - historical held-out metrics (from reports/p2/)
  - training config sha256
  - created_at
- Tests: field completeness, SHA format, 3 models present

### Task 2: Extend Sample Schema
- File: `src/schemas.py`
- Add optional fields: `variant_type: str | None`, `bug_type: str | None`, `source_split: str | None`
- Backward compatible (existing JSON without these fields loads fine)
- Update serialization (to_dict / from_dict / to_chatml unaffected)
- No new task_type (boundary is variant_type, not task_type)
- Tests: schema load without new fields (backward compat), with new fields, serialization round-trip, metrics unaffected

### Task 3: Fix import_mbpp.py + New verify_imported_mbpp.py
- import_mbpp.py changes:
  - Default verified=false, verification all false (no preset)
  - per-split manifest: manifest.{split}.json + manifest.index.json (SHA256/count/task_id_range per split)
  - Pin HF source revision (datasets.load_dataset revision param)
  - Record dataset_fingerprint, source_revision, normalized_sha256
  - benchmark_contaminated field (true for test split only)
  - standard_mbpp_test_claims_disallowed field
  - normalized/verified/rejected subdirectories (or normalized/{split}.jsonl + verified/{split}.jsonl + rejected/{split}.jsonl)
  - No pytest execution in importer
- New scripts/verify_imported_mbpp.py:
  - Loads normalized/{split}.jsonl
  - Calls src.validators.verify_sample per sample
  - Hard checks: public_assertions >= 2, hidden_assertions >= 3
  - Repair samples: verify_broken_is_broken + execution_feedback from real failure
  - Writes verified/{split}.jsonl (verified=true) + rejected/{split}.jsonl (with reason)
  - Updates manifest.{split}.json with verified_sha256, verified_count, rejected_count
- Tests: importer default verified=false, per-split manifest no overwrite, contamination flag, verifier hard checks, rejected recording

### Task 4: Run Import + Verify + Source Audit
- Run: import_mbpp.py --split test, --split validation, --split train (re-run for new manifest format)
- Run: verify_imported_mbpp.py for each split
- Generate reports/p3/mbpp-source-audit.json:
  - actual split names, sample counts, task_id ranges
  - missing/duplicate task IDs
  - dataset fingerprint, source revision
  - normalized SHA256, verified SHA256
  - verified/rejected counts per split
  - conclusion: LIKELY_FEASIBLE or INFEASIBLE
- If actual new families < 240 (180 train + 60 val minimum), stop and escalate

### Task 5: Cross-Split Semantic Dedup Audit
- New scripts/audit_cross_split_dedup.py
- Checks: normalized instruction hash, target code hash, test-suite hash, function signature, AST structural hash, token n-gram similarity
- Output: reports/p3/cross-split-dedup-audit.json (stats) + reports/p3/cross-split-dedup-review-queue.jsonl (high-sim pairs)
- Hard requirement: unresolved=0 before partition
- Unconfirmed high-similarity families → quarantine list (excluded from Frozen/Val/Train)
- Tests: dedup detection, quarantine recording, unresolved counter

### Task 6: Build Family Registry
- data/family-registry.json: family_id, source_task_id, source_split, usage (list), first_commit, dataset_version, sample_ids
- src/family_registry.py: is_used(), families_with_usage(), claim(), assert_pairwise_disjoint(usages, whitelist=[])
- scripts/build_family_registry.py: backfill P2 374 families + new imported families (usage=[] pending claim)
- Tests: backfill correct, claim idempotent, pairwise disjoint with whitelist

### Task 7: Frozen v3 Candidate Reservation
- From test source pool (minus dedup quarantine), seed=42, stratified sample 120 candidate families
- File: data/frozen-eval/v3/candidates.json (candidate list only, no samples yet)
- Hard assert: candidates disjoint with P2 used families + quarantine
- Registry claim usage=frozen_v3_candidate
- Tests: disjoint assertions, candidate count=120, registry consistency

### Task 8: Build Frozen v3 Samples + Verify + Freeze
- Generate on 120 candidates: 1 code_gen + 1 boundary + 1-2 static_repair + 1-2 exec_repair ≈ 5/family → ~600 samples
- Hidden test hard gate: public>=2, hidden>=3; generate boundary tests if insufficient, verify with Reference Code
- Boundary as variant_type="boundary"
- Verify: Reference passes public+hidden / Bugged fails >=1 / Repair passes all / Canary fails / schema OK
- Families failing verification → rejected.jsonl, removed from frozen v3
- Count qualified families:
  - >=100: freeze 100 (first 100 by seed)
  - 80-99: freeze actual count, document
  - <80: FIX FIRST (stop)
- Write data/frozen-eval/v3/{families.json, test_raw.jsonl, manifest.json, rejected.jsonl}
- Compute test_sha256, write-once (any future change → v4)
- Training code hard-blocks reading this directory (path check in collator/trainer)
- Tests: schema, verification fields, hidden gate, canary fail, SHA256 consistency, immutability assertion, path block

### Task 9: P3 Validation + Train Family Partition
- P3 Validation: prefer official validation split (~90), exclude quarantine; if insufficient, supplement from test pool minus frozen_v3
- P3 Train (new): test pool minus frozen_v3 minus validation, >=180 required
- P3 Train (replay): P2 224 train families (whitelist allowed intersection with P2)
- Total Train families >= 404
- File: data/p3-curriculum/family-partition.json
- Pairwise disjoint assertions (not just 3-way):
  - train ∩ validation = ∅
  - train ∩ frozen_v3 = ∅
  - validation ∩ frozen_v3 = ∅
  - All three ∩ P2 frozen_v2 = ∅
  - P2 replay whitelist: train_p3_replay ∩ train_p2 = 224 (allowed, documented)
- Registry claim
- Tests: pairwise disjoint, P2 frozen disjoint, count thresholds, whitelist documentation

### Task 10: Canonical Verified Sample Pool
- New src/sample_pool.py + scripts/build_sample_pool.py
- Pool = P3 new train samples (from Task 8/9 families) + P2 replay samples (from P2 stage manifests)
- Deduplicate by sample_id
- Index by family/task_type/variant_type/bug_type
- Per-family contribution cap (prevent over-representation)
- Tests: sample_id uniqueness, per-family cap, indexing correct

### Task 11: Balanced Generalist Train Data
- Sample from canonical pool at 30/20/20/30 (Code/Boundary/Static/Exec), ±3pp
- 2300-3100 final effective samples (404×7 is capacity estimate only, actual yield from verified+deduped pool)
- Shared partition (identical to Repair Specialist)
- Report family count vs sample count separately
- Output: data/p3-curriculum/balanced-generalist/{train,validation,manifest,families,token_audit,rejected}.jsonl/json
- Tests: ratio ±3pp, partition consistency, schema, verification fields

### Task 12: Repair Specialist Train Data
- Same partition, ratio 15/15/30/40 (Code/Boundary/Static/Exec), ±3pp
- Output: data/p3-curriculum/repair-specialist/...
- Tests: same as Task 11

### Task 13: Training Config + 3-Tier Checkpoint Evaluator
- configs/p3/{balanced-generalist,repair-specialist}.yaml per Issue #9 §6.1
- LoRA rank=16, alpha=32, dropout=0.05, 7 modules (q/k/v/o/gate/up/down)
- BF16 runtime check torch.cuda.is_bf16_supported(), explicit FP16 fallback only if unsupported (no silent)
- 3-tier monitoring:
  - Tier 1 (every 25-50 steps): train_loss, eval_loss, lr, gpu_mem, nan/inf check
  - Tier 2 (every 0.25 epoch): 60-90 family-stratified probe (fixed seed) + Composite Score
  - Tier 3 (every 1 epoch): full validation + Composite Score
- Early stop: probe signal triggers, full validation confirms
- Best checkpoint: by full Validation Composite only (never frozen v3, never probe)
- Composite Score formulas per candidate type (Issue #9 §6.5)
- Early stopping triggers (Issue #9 §6.6)
- Tests: config schema, BF16 check, Composite formula, early stop triggers, tier scheduling

### Task 14: Readiness Gate Report
- File: reports/p3/p3-training-readiness-report.md
- 9 PASS checks:
  1. Frozen v3 frozen (SHA locked)
  2. Train/Val/Frozen pairwise disjoint (zero leakage)
  3. Assistant retention = 100%
  4. Silent target truncation = 0
  5. Canary all fail
  6. GPU smoke: forward+backward+optimizer.step+eval+save+reload+inference pass
  7. Output dirs don't exist (no overwrite)
  8. CPU CI green
  9. P3 baseline lock present
- Conclusion: GO_FOR_P3_TRAINING or FIX_FIRST
- SESSION END (no actual training)
- Tests: all 9 checks executed, conclusion logic

---

## Binding Amendments (v2.1) — 2026-07-04

Appended after user review of v2 plan body. These 12 amendments are BINDING
and supersede any conflicting language in the original plan body above.

### A1. Import / Verify Pipeline Split (Item 1)
- `import_mbpp.py` ONLY downloads + normalizes. Outputs `verified=false`
  with all-false `Verification`. NO pytest in the importer.
- `verify_imported_mbpp.py` is the separate verifier script.
- Directory layout: `normalized/{split}.jsonl`, `verified/{split}.jsonl`,
  `rejected/{split}.jsonl` — each with its own SHA-256 in the manifest.
- Verifier internally reuses `src.validators.verify_sample`.

### A2. P3 Verifier Hard Checks (Item 2) — PAD-THEN-VERIFY
- `verify_sample` treats empty `hidden_tests` as vacuous true. Therefore the
  P3 verifier must additionally HARD-check:
  - `public_tests.count("assert ") >= 2` (HARD reject)
  - `hidden_tests.count("assert ") >= 3` (HARD reject, AFTER padding)
- **Pad-then-verify flow** (user-approved 2026-07-04):
  1. `import_mbpp.py`: normalize (verified=false) — unchanged
  2. `verify_imported_mbpp.py`:
     a. Load normalized samples
     b. **Pad hidden tests** via shared `src/hidden_test_padding.pad_hidden_tests`
        (extracted from `build_frozen_v3_samples.py`)
     c. HARD check: public>=2, hidden>=3 (on padded sample)
     d. Repair samples: `verify_broken_is_broken` + `execution_feedback`
        must contain a failure marker (real failure output)
     e. Real `verify_sample` (syntax + pytest + ruff + timeout)
     f. Write verified/rejected + update manifest
- Supersedes Task 4 fix `b46fcb7` (which used SOFT warning for hidden>=3).
  Raw MBPP samples (1 hidden assert) now get padded BEFORE the hard check.
- Frozen v3 / training-data verifiers also enforce hidden>=3 HARD.

### A3. Frozen v3 Pipeline Consolidation (Item 3)
- Full frozen v3 pipeline (reservation → generation → enhancement → verify →
  dedup → freeze) is one logical unit.
- Freeze decision: >=100 freeze 100 / 80-99 freeze actual / <80 FIX FIRST.
- `families.json` / `test_raw.jsonl` / SHA write-once immutable.

### A4. Pairwise Disjoint Enforcement (Item 4)
- `assert_pairwise_disjoint(usages, whitelist=[])` checks EVERY PAIR C(n,2),
  not just 3-way intersection.
- P2 Train replay is the ONLY allowed historical intersection — explicitly
  whitelisted.

### A5. Sample Schema Extension (Item 5)
- Optional backward-compatible: `variant_type`, `bug_type`, `source_split`.
- No new `task_type` (boundary is `variant_type`).

### A6. Canonical Verified Sample Pool (Item 6)
- NO direct Stage manifest concatenation for P2 replay.
- Pool: sample_id dedup → index by family/task/variant/bug → per-family
  cap → sample by two candidate ratios.

### A7. Effective Sample Count (Item 7)
- 2300-3100 = final effective `train.jsonl` per candidate.
- 404x7 is capacity estimate ONLY. No duplication to pad.

### A8. Three-Tier Checkpoint (Item 8)
- Tier 1 (25-50 steps): loss/LR/GPU/NaN
- Tier 2 (0.25 epoch): 60-90 family-stratified probe + Composite
- Tier 3 (1 epoch): full validation + Composite
- Early stop: probe triggers, full validation confirms.
- Best checkpoint: full Validation Composite ONLY.

### A9. Cross-Split Dedup Gate (Item 9)
- `unresolved=0` HARD prerequisite before partition.
- Unconfirmed high-similarity → quarantine.

### A10. HuggingFace Source Pinning (Item 10)
- Pin source revision. Manifest records: `source_revision`,
  `dataset_fingerprint`, `normalized_sha256`, `verified_sha256`.

### A11. Baseline Lock First (Item 11)
- `reports/p3/p3-baseline-lock.json` is the FIRST implementation task.
- Before any data construction or training config change.

### A12. Readiness Gate Only (Item 12)
- Session executes ONLY to Readiness Gate (Task 14).
- NO full training.

### Rework Note (2026-07-04)
- A2 pad-then-verify requires rework of Tasks 3-8:
  - Task 3-redo: extract `pad_hidden_tests` to `src/hidden_test_padding.py`,
    modify `verify_imported_mbpp.py` to pad-then-verify
  - Task 4-redo: re-run import + verify (with pad)
  - Task 5-redo: re-run dedup on new verified set
  - Task 6-redo: rebuild family registry
  - Task 7-redo: re-run candidate reservation
  - Task 8-redo: re-run frozen v3 samples build
- Tasks 1-2 unaffected (baseline lock + schema already compliant).
