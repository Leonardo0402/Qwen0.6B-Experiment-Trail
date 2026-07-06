# MBPP Family Audit for P3 Planning

Audit performed: 2026-07-04
Scope: `e:\agent\Qwen\qwen3-code-lab` — read-only audit for Issue #9 P3 Capability Expansion planning.

## MBPP Source Dataset

- Source: `google-research-datasets/mbpp` (HuggingFace, Apache-2.0)
- Imported splits: **`train` only**
- Total imported samples: **374**
- Task ID range: **`mbpp_601` → `mbpp_974`** (contiguous, no gaps — every integer 601..974 is present)
- Imported files:
  - `data/external/mbpp/manifest.json` (split=train, sample_count=374, sha256=fb45cb32…)
  - `data/external/mbpp/normalized/train.jsonl` (374 lines, one Sample per line)
- Other available splits (NOT yet imported):
  - `test` — typical size on HuggingFace is ~500 samples (task_ids ~1–500)
  - `validation` — typical size ~90 samples (task_ids ~501–590)
  - `prompt` — prompt-only / auxiliary, not directly usable as Samples
  - `audit` — auxiliary audit metadata
  - Note: exact sizes were not fetched (HuggingFace dataset page timed out); sizes are the canonical MBPP values from the dataset card / `mbpp` README. Verify via `datasets.load_dataset("google-research-datasets/mbpp", split="test")` before import.
- The import script `scripts/import_mbpp.py` already supports `--split {train,test,validation,prompt}`, so no new infrastructure is required to import additional splits.

## P2 Family Usage

All P2 family lists come from the canonical P2 partition file
`data/p2-curriculum/family-partition.json` (seed=42, three-way disjoint).

| Bucket | Family count | Source |
|---|---|---|
| P2 train families | **224** | `family-partition.json::train_families` |
| P2 validation families | **75** | `family-partition.json::validation_families` |
| P2 frozen-v1 families (a.k.a. partition frozen) | **75** | `family-partition.json::frozen_families` |
| P2 frozen-v2 families | **75** | `data/p2-curriculum/frozen-eval-v2/manifest.json::test_families` (identical to partition frozen) |
| Router selection families | **45** | `reports/p2/router-policy-v1.json::selection_families` (subset of frozen-v2) |
| Router eval families | **30** | `reports/p2/router-policy-v1.json::eval_families` (subset of frozen-v2; 45+30=75 covers all frozen-v2) |

Stage manifests (`stage1-code`, `stage2-boundary`, `stage3-repair`, `stage3-repair-v3`) all reuse the SAME partition lists — they only differ in which subset of `train_families` is materialized into each stage's `train.jsonl`. They introduce NO new family IDs.

The P1-era `data/frozen-eval/v1/manifest.json` uses 9 NON-MBPP families (`fam_l0_*`, `fam_l1_*`, `fam_l2_*`) from the project's hand-authored data pool, not from MBPP. These are orthogonal to the MBPP family namespace and irrelevant to P3 MBPP capacity.

- **Total unique MBPP families used by P2: 374** (= 224 + 75 + 75; overlaps all empty by construction)
- **List of all P2-used family IDs:** the union is `mbpp_fam_601` … `mbpp_fam_974` (every MBPP train family). The three disjoint subsets are listed in `data/p2-curriculum/family-partition.json`.

## Available (Unused) Families

- Total MBPP train families available: **374**
- Total MBPP train families used by P2: **374**
- **Total unused families in the currently-imported MBPP data: 0**
- Unused family ID list: `[]` (empty)

Every MBPP train family is already assigned to exactly one of train / validation / frozen. There is no slack.

## Capacity Assessment

- Can we meet Issue #9's requirement of 60–100 new frozen-v3 families? **NO**
- Can we meet Issue #9's requirement of ≥180 new train families? **NO**

If NO to either, what are the options?

1. **Import MBPP `test` split** via `python scripts/import_mbpp.py --split test`.
   - Expected yield: ~500 new families (task_ids ~1–500), disjoint from the imported `train` split.
   - Recommended primary option — single command, no new code needed.
2. **Import MBPP `validation` split** via `python scripts/import_mbpp.py --split validation`.
   - Expected yield: ~90 new families (task_ids ~501–590).
   - Useful as an additional held-out bucket; recommend as secondary.
3. **Generate variants via existing pipeline** (`scripts/inject_bugs.py`, `scripts/generate_boundary_variants.py`, `scripts/build_execution_repair.py`).
   - ⚠️ Caveat: these scripts create new Samples inside EXISTING family IDs (e.g. `mbpp_611_er_*` belongs to family `mbpp_fam_611`). They DO NOT create new family IDs, so they cannot by themselves satisfy the "new families" requirement. They are only useful for enriching samples inside new families once those families are imported.
4. **Hand-author new families** (the P1-era `fam_l0_*` / `fam_l1_*` / `fam_l2_*` approach used by `data/frozen-eval/v1`).
   - Slow, manual, doesn't scale to ≥180 families. Not recommended as primary.

## Existing Data Generation Pipeline

### Relevant scripts

| Script | One-line description |
|---|---|
| `scripts/import_mbpp.py` | Downloads MBPP from HuggingFace (`datasets.load_dataset`), converts each task to canonical `Sample` (task_type=code_generation), writes `normalized/<split>.jsonl` + `manifest.json`. Supports `--split`. |
| `scripts/build_p2_curriculum.py` | P2.2 curriculum integration builder. Calls `inject_bugs`, `build_execution_repair`, `generate_boundary_variants` to produce stage1-code / stage2-boundary / stage3-repair / frozen-eval-v2 / family-partition.json. |
| `scripts/build_curriculum_data.py` | P1 spec §7 curriculum replay data builder for `data/curriculum/{easy,boundary,repair}/`. |
| `scripts/build_dataset.py` | General dataset builder (split_by_family wrapper). |
| `scripts/build_execution_repair.py` | Execution-driven repair sample builder: inject bugs → run pytest → keep failing variants → compress feedback → emit static_repair + execution_repair Samples. |
| `scripts/build_frozen_eval.py` | Builds the expanded P1 frozen eval set from `data/splits/test_raw.jsonl` + untrained families. |
| `scripts/build_stage3_v3_antiforget.py` | Stage3-v3 anti-forgetting mix per Issue #1 P2. |
| `scripts/inject_bugs.py` | 8 deterministic AST bug-injection operators (condition_error, off_by_one, return_value_error, index_error, initialization_error, aggregation_error, branch_deletion, type_error). |
| `scripts/generate_boundary_variants.py` | Boundary-condition variant generator: extends test suite with empty / single / boundary-value cases (difficulty+1). |
| `scripts/generate_tasks.py` | Generic task generator. |
| `scripts/generate_reference.py` | Reference solution generator. |
| `scripts/mutate_code.py` | Code mutation utilities. |
| `scripts/augment_instructions.py` | Instruction paraphrasing/augmentation. |
| `scripts/rebuild_curriculum_v2.py` | Rebuilds `data/curriculum-v2/` from the verified pool. |
| `scripts/split_router_selection.py` | Splits frozen-v2 into router selection (45) / eval (30) subsets. |
| `scripts/update_p2_manifests.py` | Refreshes P2 manifest hashes/counts. |
| `scripts/audit_p2_dataset.py` / `audit_p2_tokens.py` / `audit_token_lengths.py` | Audits of dataset integrity, token counts, length distributions. |
| `scripts/verify_samples.py` | Sample schema + execution verification. |

### Relevant `src/` modules

- `src/schemas.py` — `Sample`, `Verification` dataclasses; `to_chatml`.
- `src/dataset_builder.py` — `split_by_family` (leak-proof family partitioner, fixed seed).
- `src/curriculum.py` — curriculum ratios / stage assembly helpers.
- `src/training_data.py` — ChatML training-data preparation.
- `src/sandbox.py` — `run_python_code` / `run_pytest` sandboxed execution.
- `src/validators.py` — `verify_sample` (syntax, pytest, ruff, timeout).
- `src/metrics.py` — evaluation metrics.

### Sample schema (first entry of `data/p2-curriculum/stage1-code/train.jsonl`)

Key fields (a `Sample` record):

```
{
  "sample_id":      "mbpp_917",
  "family_id":      "mbpp_fam_917",
  "difficulty":     1,                       // 0..4
  "task_type":      "code_generation",        // code_generation | static_repair | execution_repair
  "language":       "python",
  "skill_tags":     ["function"],             // derived from prompt keywords
  "instruction":    "Write a function to find the sequences of one upper case letter followed by lower case letters.\n\nFunction signature: def text_uppercase_lowercase(text):",
  "broken_code":    null,                     // present for *_repair samples
  "execution_feedback": null,                 // present for execution_repair samples
  "target_code":    "import re\r\ndef text_uppercase_lowercase(text):\r\n        ...",
  "public_tests":   "assert text_uppercase_lowercase(\"AaBbGg\")==('Found a match!')\n\n...",
  "hidden_tests":   "assert text_uppercase_lowercase(\"PYTHON\")==('Not matched!')",
  "verified":       true,
  "verification":   {"syntax_ok": true, "pytest_ok": true, "ruff_ok": false, "timeout": false},
  "generator":      "mbpp-importer",          // or "inject_bugs", "generate_boundary_variants", "build_execution_repair", ...
  "created_at":     "2026-07-02T14:05:40.715694+00:00",
  "dataset_version":"mbpp-v1"                 // or "p2.2", "p2.3-antiforget", ...
}
```

The schema is unchanged across `data/external/mbpp/normalized/train.jsonl`, all `data/curriculum*/`, and all `data/p2-curriculum/` files. The same `Sample` dataclass is used end-to-end.

## Recommendations

**Primary recommendation:** Import MBPP `test` and `validation` splits before starting P3 curriculum construction.

Concrete steps (no new code required — `scripts/import_mbpp.py` already supports `--split`):

1. `python scripts/import_mbpp.py --split test --output-dir data/external/mbpp`
   → yields `data/external/mbpp/normalized/test.jsonl` (~500 new families, task_ids ~1–500).
   ⚠️ Caveat: the current `--output-dir` writes a single `manifest.json` overwriting the train manifest. Either (a) run with separate output dirs (`data/external/mbpp-test`, `data/external/mbpp-validation`) and union the families at the partition step, or (b) extend `import_mbpp.py` to merge manifests. Verify before running.
2. `python scripts/import_mbpp.py --split validation`
   → yields ~90 more new families (task_ids ~501–590).
3. Combine the three splits (train + test + validation ≈ 964 families total) into a new P3 family-partition with three-way disjoint buckets:
   - reserve 60–100 new families for **frozen-v3** (Issue #9 lower bound is 60, target 100),
   - allocate ≥180 new families to **P3 train** (Issue #9 lower bound),
   - allocate a P3 validation bucket disjoint from both.
4. After new families are imported, reuse the existing P2 data-generation pipeline (`build_p2_curriculum.py` → `inject_bugs` + `generate_boundary_variants` + `build_execution_repair`) to materialize per-stage curricula. The pipeline is family-ID-agnostic — it just needs a `--input` JSONL of new MBPP Samples.

**Yield check:**
- MBPP `test` (~500) + `validation` (~90) ≈ 590 new families.
- Issue #9 asks for 60–100 frozen-v3 + ≥180 new train = at least 240 new families.
- 590 ≫ 240, so importing just `test` (or `test` + `validation` together) gives ample headroom (≈2.5× margin over the minimum).

**Why not variant generation?** The existing variant generators (`inject_bugs`, `generate_boundary_variants`, `build_execution_repair`) emit Samples with the SAME `family_id` as their source (e.g. `mbpp_611_er_*` belongs to `mbpp_fam_611`). They cannot manufacture new family IDs. They are essential for STAGE assembly inside a family but useless for ADDING families to the partition. New families must come from new source tasks — i.e. the MBPP test/validation splits.

**Risk to verify before import:**
- Confirm MBPP `test`/`validation` task_ids really are disjoint from `train` (601–974). Canonical MBPP assigns `test` = 1–500, `validation` = 501–590, `train` = 601–974, so they should be disjoint by construction — but a one-line sanity check (`set(test_ids) & set(train_ids) == set()`) after import is cheap insurance.
- Confirm `import_mbpp.py` does not overwrite `data/external/mbpp/manifest.json` for the existing `train` split when re-run with `--split test`. Either use separate output dirs or refactor to write per-split manifests (`manifest.train.json`, `manifest.test.json`, …).
