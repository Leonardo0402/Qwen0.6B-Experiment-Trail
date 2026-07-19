# P4.1b Protocol Ablation — v4 vs v3 Analysis

## 1. Experiment Context

- **v3** (pre-fix): `max_new_tokens=128`, greedy regex `extract_json`, no default injection for `action_id`/`safety_flags`
- **v4** (post-fix): `max_new_tokens=256`, `raw_decode`-based `extract_json`, structural default injection for `action_id`/`reason_short`/`expected_observation`/`safety_flags`
- **Common**: 3 protocols × 2 configs × 40 tasks = 240 trajectories, 480 steps per combination, 2880 steps total
- **Verdict (both)**: `FIX_PROMPT_FIRST`

## 2. Metrics Comparison

### 2.1 schema_valid_rate (primary verdict metric)

| Combination | v3 | v4 | Delta |
|-------------|-----|-----|-------|
| dsl-base | 0.00% | 0.00% | 0 |
| dsl-repair-lora | 6.04% | 0.00% | **-6.04pp** |
| json-base | 0.00% | 0.00% | 0 |
| json-repair-lora | 0.00% | 0.00% | 0 |
| tag-base | 2.29% | 10.42% | **+8.13pp** |
| tag-repair-lora | 1.67% | 1.04% | -0.63pp |

**Protocol averages**:
- dsl: v3=3.02% → v4=0.00% (regression)
- json: v3=0.00% → v4=0.00% (unchanged)
- tag: v3=1.98% → v4=5.73% (improvement)

### 2.2 format_parse (step-level)

| Combination | v3 | v4 | Delta |
|-------------|-----|-----|-------|
| dsl-base | 0.00% | 80.00% | **+80.00pp** |
| dsl-repair-lora | 47.08% | 100.00% | **+52.92pp** |
| json-base | 2.50% | 45.00% | **+42.50pp** |
| json-repair-lora | 100.00% | 100.00% | 0 |
| tag-base | 2.29% | 20.00% | **+17.71pp** |
| tag-repair-lora | 6.25% | 100.00% | **+93.75pp** |

**Format parse failure count**: v3=2193 → v4=744 (**-66% reduction**)

### 2.3 Failure Taxonomy

| Failure Class | v3 | v4 | Delta |
|---------------|-----|-----|-------|
| FORMAT_PARSE_FAIL | 2193 | 744 | **-1449 (-66%)** |
| SCHEMA_VALIDATION_FAIL | 639 | 2081 | **+1442** |
| REPEATED_ACTION_LOOP | 240 | 240 | 0 |

## 3. Key Findings

### 3.1 Format Fix Succeeded (root cause addressed)

The v3 `FIX_PROMPT_FIRST` verdict was triggered by 2193 FORMAT_PARSE_FAIL out of 2880 steps (76%). The root cause — greedy regex `_BARE_JSON_RE = r"(\{.*\})"` capturing multiple concatenated JSON objects as one string — was fixed in v4 via `json.JSONDecoder().raw_decode()`.

**Evidence**: Format parse failures dropped 66% (2193 → 744). Four combinations now achieve 80-100% format_parse (vs v3's max 47%).

### 3.2 Schema Validation Became New Bottleneck

The 1449 format-fixed steps converted directly to 1442 new SCHEMA_VALIDATION_FAIL. The model now produces parseable output, but the output does not match the Action schema.

**Root cause analysis** (from trajectory inspection):
- Model emits finish/propose_patch actions with wrong argument shape (e.g., `{"action_type":"finish","reason":"..."}` instead of required `success_criterion`, `tests_passed`, `identification_verified`, `summary`)
- Model omits required parameters for parameterized actions (read_file needs `path`, search_text needs `query`+`path`)
- Prompt examples only show read_file/propose_patch; Qwen3-0.6B cannot infer parameter schemas for the other 9 actions

### 3.3 Action-Type Distribution Skewed

- **json-repair-lora**: 468/480 steps emit `finish` (all fail schema) — model collapses to finish-only
- **tag-base**: 47 successful `inspect_task` (no-arg action) — model succeeds only when no parameters needed
- **dsl-repair-lora**: 100% format_parse but 0% schema_valid — format perfect, args always wrong

### 3.4 Verdict Still FIX_PROMPT_FIRST (Correct)

v4 verdict rule 2 triggers: all protocols' avg schema_valid_rate < 30% (dsl=0%, json=0%, tag=5.73%). The fix moved the bottleneck from format to schema, but did not cross the 30% threshold.

## 4. Recommendations for v5

### 4.1 Prompt Enhancement (Primary)

1. **Per-action examples**: Extend prompt to show all 11 action types with their required arguments, especially `finish` (4 required fields), `propose_patch`/`apply_patch` (file_path, old_text, new_text), `search_text` (query, path)
2. **Argument schema hints**: Include a compact parameter list per action (e.g., `finish(success_criterion, tests_passed, identification_verified, summary)`)
3. **Negative examples**: Show common mistakes (e.g., "Do NOT use `reason` for finish; use `success_criterion`")

### 4.2 Schema Relaxation (Secondary, if prompt fix insufficient)

The `finish` action requires 4 complex fields (`success_criterion: TaskSuccessCriterion`, `tests_passed: bool`, `identification_verified: bool`, `summary: str`). For a 0.6B model this is demanding. Options:
- Simplify `TaskSuccessCriterion` enum
- Make `identification_verified` default to `false`
- Split `finish` into `finish_simple` (no args) and `finish_verified` (full args)

### 4.3 Metric Interpretation

- **format_parse**: v4 demonstrates the JSON extraction fix is effective — keep this signal
- **schema_valid**: Now the binding constraint; v5 must target >30% to escape FIX_PROMPT_FIRST
- **arguments_valid = schema_valid**: In all v4 combinations these are identical, confirming schema failures are purely argument-shape mismatches (not action_type or safety_flags issues)

## 5. Conclusion

The v4 fix successfully addressed the v3 root cause (FORMAT_PARSE_FAIL dropped 66%), but exposed a deeper issue: Qwen3-0.6B cannot infer Action argument schemas from minimal prompts. The verdict remains `FIX_PROMPT_FIRST`, now pointing at argument-shape education rather than format extraction. v5 should focus on per-action prompt examples and possibly schema simplification for the finish action.
