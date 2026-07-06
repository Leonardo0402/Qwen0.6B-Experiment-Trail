# Task 5 Brief: Cross-Split Semantic Dedup Audit

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Tasks 1-4 complete, 955 verified samples across 3 splits)
- Verified data: `data/external/mbpp/verified/{train,test,validation}.jsonl`
- Plan file: .superpowers/sdd/p3-plan.md (Global Constraint #15: unresolved=0 before partition)

## Goal
Write `scripts/audit_cross_split_dedup.py` that checks for semantic duplicates across MBPP splits, then run it to produce:
- `reports/p3/cross-split-dedup-audit.json` (statistics)
- `reports/p3/cross-split-dedup-review-queue.jsonl` (high-similarity pairs needing review)
- `reports/p3/cross-split-dedup-quarantine.json` (final quarantine list — families excluded from P3)

## Dedup Checks (6 methods)

### 1. Normalized instruction hash (exact match)
- Normalize: `instruction.strip().lower()` + collapse whitespace (`re.sub(r'\s+', ' ', text)`)
- Compute SHA256 of normalized instruction
- Cross-split pairs with same hash → EXACT_DUPLICATE

### 2. Target code hash (exact match)
- Normalize: strip comments (`# ...` lines), strip whitespace, collapse blank lines
- Compute SHA256 of normalized target_code
- Cross-split pairs with same hash → EXACT_DUPLICATE

### 3. Test-suite hash (exact match)
- Concatenate `public_tests + "\n" + hidden_tests`, normalize whitespace
- Compute SHA256
- Cross-split pairs with same hash → EXACT_DUPLICATE

### 4. Function signature (exact match)
- Parse `target_code` with `ast.parse()`
- Extract all `ast.FunctionDef` nodes: name + arg names (ignore annotations/defaults)
- Format as `"func_name(arg1,arg2,...)"`
- If multiple functions, join with `;`
- Cross-split pairs with same signature → HIGH_SIMILARITY (review queue, not auto-duplicate — different functions can share signatures)

### 5. AST structural hash (structural similarity)
- Parse `target_code` with `ast.parse()`
- Walk tree, replace all `ast.Name` nodes' `id` with `"VAR"`, all `ast.Constant` nodes' `value` with `"CONST"`
- `ast.dump(tree, annotate_fields=False)` → SHA256
- Cross-split pairs with same structural hash → HIGH_SIMILARITY (review queue)

### 6. Token n-gram similarity (near-duplicate detection)
- Tokenize `instruction`: split on non-alphanumeric (`re.findall(r'[a-z0-9]+', text.lower())`)
- Compute 3-gram set (sliding window of 3 tokens)
- Jaccard similarity = |intersection| / |union|
- For each cross-split pair, compute Jaccard
- Pairs with Jaccard > 0.7 → HIGH_SIMILARITY (review queue)
- Note: O(n^2) comparison — for 955 samples that's ~450k pairs. Use blocking/bucketing: first group by first 2 chars of instruction hash, only compare within same bucket. This reduces to manageable size.

## Output Files

### reports/p3/cross-split-dedup-audit.json
```json
{
  "generated_at": "<iso8601>",
  "total_samples_checked": 955,
  "splits_checked": ["train", "test", "validation"],
  "checks_performed": ["instruction_hash", "code_hash", "test_hash", "func_signature", "ast_structural", "ngram_3"],
  "exact_duplicates": {
    "count": <int>,
    "pairs": [{"sample_a": "mbpp_XXX", "split_a": "train", "sample_b": "mbpp_YYY", "split_b": "test", "method": "instruction_hash"}]
  },
  "high_similarity": {
    "count": <int>,
    "pairs": [{"sample_a": "...", "sample_b": "...", "method": "ngram_3", "score": 0.85}]
  },
  "quarantined_families": {
    "count": <int>,
    "families": ["mbpp_fam_XXX", ...]
  },
  "unresolved": {
    "count": 0,
    "note": "All high-similarity pairs auto-quarantined per P3 plan (unconfirmed → quarantine)"
  },
  "conclusion": "PASS" | "FAIL"
}
```
- `conclusion`: PASS if unresolved=0 (all high-sim auto-quarantined), FAIL otherwise

### reports/p3/cross-split-dedup-review-queue.jsonl
One line per high-similarity pair:
```json
{"sample_a": "mbpp_XXX", "split_a": "train", "sample_b": "mbpp_YYY", "split_b": "test", "method": "ngram_3", "score": 0.85, "status": "auto_quarantined"}
```

### reports/p3/cross-split-dedup-quarantine.json
```json
{
  "generated_at": "<iso8601>",
  "quarantine_reason": "Cross-split semantic duplicates or high-similarity pairs (unconfirmed). Excluded from P3 Frozen/Val/Train partition.",
  "quarantined_families": ["mbpp_fam_XXX", ...],
  "count": <int>,
  "auto_quarantine_rules": ["exact_duplicate", "high_similarity_unconfirmed"]
}
```

## Quarantine Logic (binding)
1. **Exact duplicate** (instruction/code/test hash match across splits) → both families auto-quarantined
2. **High similarity** (func_signature match, AST structural match, or n-gram > 0.7) → both families auto-quarantined (per plan: "未人工确认的高相似 family 一律 quarantine")
3. **unresolved=0**: all high-sim pairs are auto-quarantined, so unresolved count is always 0 after this script runs
4. Quarantine list is the union of all families involved in any duplicate or high-sim pair

## CLI
```
python scripts/audit_cross_split_dedup.py --input-dir data/external/mbpp --report-dir reports/p3
```

## Tests (tests/test_cross_split_dedup.py)
Use synthetic samples (no real MBPP data needed):

1. `test_exact_instruction_duplicate_detected`: two samples with same normalized instruction → exact duplicate
2. `test_exact_code_duplicate_detected`: two samples with same normalized target_code → exact duplicate
3. `test_exact_test_duplicate_detected`: two samples with same test-suite → exact duplicate
4. `test_func_signature_match_detected`: two samples with same function signature → high similarity
5. `test_ast_structural_match_detected`: two samples with same AST structure (different var names) → high similarity
6. `test_ngram_high_similarity_detected`: two samples with Jaccard > 0.7 → high similarity
7. `test_ngram_low_similarity_not_flagged`: two samples with Jaccard < 0.7 → not flagged
8. `test_quarantine_list_built`: families in any duplicate/high-sim pair → quarantined
9. `test_unresolved_always_zero`: after script runs, unresolved=0 (all auto-quarantined)
10. `test_disjoint_splits_no_duplicates`: three splits with no overlap → 0 duplicates, 0 quarantine, conclusion PASS

## Constraints
- Read from `data/external/mbpp/verified/{split}.jsonl` (verified samples only)
- Do NOT modify any existing files outside `scripts/`, `tests/`, `reports/p3/`
- Do NOT modify src/ files
- n-gram comparison must be efficient (use bucketing/blocking to avoid O(n^2) on 955 samples)
- AST parsing must handle syntax errors gracefully (skip sample with warning, don't crash)

## Report File
Write to: `.superpowers/sdd/task-5-report.md`
Include: total samples checked, exact duplicate count, high similarity count, quarantined family count, conclusion.

Return: status, commit hash, test summary, concerns.

## Commit
- Stage: `scripts/audit_cross_split_dedup.py`, `tests/test_cross_split_dedup.py`, `reports/p3/cross-split-dedup-audit.json`, `reports/p3/cross-split-dedup-review-queue.jsonl`, `reports/p3/cross-split-dedup-quarantine.json`
- Commit message: `feat(p3): cross-split semantic dedup audit with quarantine`
- Single commit.

## Working Directory
e:\agent\Qwen\qwen3-code-lab

## Test Verification
`cd e:\agent\Qwen\qwen3-code-lab ; python -m pytest tests/test_cross_split_dedup.py -v`

All 10 tests must pass.
