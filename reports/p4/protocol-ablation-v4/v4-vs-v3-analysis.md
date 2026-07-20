# P4.1b Protocol Ablation — v4 vs v3 Analysis

> **Status**: Post-hoc analysis document. NOT a core experiment artifact.
> Listed in `reports/p4/protocol-ablation-v4/` directory (13 files total) but
> deliberately excluded from `artifact-manifest.json` (which lists the 11 core
> experiment artifacts only).
>
> **v5 status**: PROPOSED — NOT AUTHORIZED — OUTSIDE PR #30. No v5 work is
> implemented or approved in this PR.

## 1. Experiment Context

- **v3** (pre-fix state): `max_new_tokens=128`, greedy regex `extract_json`,
  no default injection for `action_id`/`safety_flags`.
- **v4** (post-fix state): `max_new_tokens=256`, `raw_decode`-based
  `extract_json`, structural default injection for `action_id` /
  `reason_short` / `expected_observation` / `safety_flags`, tightened
  protocol formatting behavior.
- **Common**: 3 protocols × 2 configs × 40 tasks = 240 trajectories,
  480 steps per combination, 2880 steps total.
- **Verdict (both)**: `FIX_PROMPT_FIRST`.

### 1.1 v4 is a state comparison, not a single-variable causal ablation

v3 → v4 changes multiple variables simultaneously:

1. `max_new_tokens`: 128 → 256 (token budget).
2. JSON extraction: greedy regex `r"(\{.*\})"` → `json.JSONDecoder().raw_decode()`.
3. Structural metadata defaults: `action_id`, `reason_short`,
   `expected_observation`, `safety_flags` are now injected by the protocol
   layer when the model omits them.
4. Protocol prompt formatting behavior was tightened (see commits
   `475b162`, `dbe3445`, `7d5dbc2`).

`raw_decode()` only affects the JSON protocol. Tag and DSL formatters are
unaffected by that specific change, yet their `format_parse` rates also
shifted substantially (see §2.2). Therefore the v3 → v4 delta cannot be
attributed to `raw_decode` alone.

The existing v3/v4 comparison is a **state comparison** between two recorded
experiment snapshots, not a controlled single-variable causal ablation.
The numerical deltas below describe what changed between the two states;
they do not isolate the contribution of any single fix.

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

**Format parse failure count**: v3=2193 → v4=744 (-1449, **-66% relative**).

### 2.3 Failure Taxonomy

| Failure Class | v3 | v4 | Delta |
|---------------|-----|-----|-------|
| FORMAT_PARSE_FAIL | 2193 | 744 | -1449 (-66%) |
| SCHEMA_VALIDATION_FAIL | 639 | 2081 | **+1442** |
| REPEATED_ACTION_LOOP | 240 | 240 | 0 |

## 3. Key Findings

### 3.1 Format failures decreased under the joint v4 changes

The v3 state had 2193 FORMAT_PARSE_FAIL out of 2880 steps (76%). Under the
v4 state — which simultaneously changed token budget, JSON extraction,
default injection, and protocol prompt formatting — format parse failures
dropped by 66% to 744. Four combinations now achieve 80-100% format_parse
(vs v3's max 47%).

Because v4 changed multiple variables at once (see §1.1), the 66% reduction
cannot be attributed to `raw_decode()` alone. The JSON-specific
`raw_decode` fix is one contributing factor; the larger token budget and
the structural default injection for `action_id` / `safety_flags` are
independent contributing factors, and Tag/DSL format improvements must
come from the broader prompt-formatting changes rather than from
`raw_decode` (which only affects JSON).

### 3.2 Schema Validation Became the New Bottleneck

The 1449 format-fixed steps converted almost 1:1 into 1442 new
SCHEMA_VALIDATION_FAIL. The model now produces parseable output, but the
output does not match the Action schema.

**Observations from trajectory inspection**:
- Model emits `finish` / `propose_patch` actions with wrong argument shape
  (e.g., `{"action_type":"finish","reason":"..."}` instead of the required
  `success_criterion`, `tests_passed`, `identification_verified`,
  `summary`).
- Model omits required parameters for parameterized actions:
  - `read_file` requires `path`
  - `search_text` requires `query` (its actual schema is
    `search_text(query: str, file_glob: str | None = None, max_results: int = 20)`;
    the earlier v4 draft of this document incorrectly listed `path` as a
    required field — `path` is not part of `SearchTextArgs`)
- Prompt examples only show `read_file` / `propose_patch`; Qwen3-0.6B
  cannot infer parameter schemas for the other 9 actions.

### 3.3 Action-Type Distribution Skewed

- **json-repair-lora**: 468/480 steps emit `finish` (all fail schema) —
  model collapses to finish-only.
- **tag-base**: 47 successful `inspect_task` (no-arg action) — model
  succeeds only when no parameters are needed.
- **dsl-repair-lora**: 100% format_parse but 0% schema_valid — format
  perfect, args always wrong.

### 3.4 Verdict Still FIX_PROMPT_FIRST (Correct)

v4 verdict rule 2 triggers: all protocols' avg `schema_valid_rate` < 30%
(dsl=0%, json=0%, tag=5.73%). The state change moved the bottleneck from
format to schema, but did not cross the 30% threshold.

## 4. v5 Recommendations (PROPOSED — NOT AUTHORIZED — OUTSIDE PR #30)

> **Important**: Everything in this section is a proposal for a future
> experiment. PR #30 does NOT implement any of it. No v5 code, no v5
> training, no v5 prompt change, no Action Schema change, no `finish`
> default injection is included in this PR. Starting v5 requires explicit
> user authorization in a separate issue.

The v4 evidence suggests two complementary directions for a future v5
experiment. They are listed here for planning purposes only.

### 4.1 Prompt Enhancement (primary candidate)

1. **Compact action signatures** — list every action with its argument
   shape inline in the system prompt so the model does not have to guess
   (e.g., `finish(success_criterion, tests_passed, identification_verified,
   summary)`).
2. **State-gated actions** — show the current-state allowed actions
   explicitly (e.g., "before finish, run_tests is required").
3. **Argument positive examples** — one positive example per action.
4. **Argument negative examples** — show common mistakes (e.g., "Do NOT
   use `reason` for `finish`; use `success_criterion`").
5. **Premature finish prevention** — explicit rule that `finish` without
   prior `run_tests` is rejected.

### 4.2 Schema / Contract Direction (future candidate — NOT RECOMMENDING SILENT DEFAULTS)

The `finish` action currently requires 4 fields (`success_criterion:
TaskSuccessCriterion`, `tests_passed: bool`, `identification_verified:
bool`, `summary: str`). For a 0.6B model this is demanding.

This document does NOT recommend injecting silent business defaults such as
`missing identification_verified → default false` or
`missing tests_passed → default false`. Silent defaults would contradict
the Issue #32 Trust Repair principle that unknown or missing fields must
hard-fail, not be silently normalized into a different valid action.

A more principled future direction is:

> **Runtime-authoritative finish**: the model requests
> `finish(summary)`; the harness independently decides whether tests and
> identification evidence permit termination, and records the
> authoritative `tests_passed` / `identification_verified` values itself.

This direction would move the `tests_passed` and `identification_verified`
fields from model-authored input to harness-authored observation, which
preserves the strict-validation principle while reducing the model's
argument burden.

**This direction belongs to a future P4.1c Action Contract experiment.**
PR #30 does not implement it. PR #30 does not modify the `finish` schema.
PR #30 does not inject business defaults.

### 4.3 Metric Interpretation

- **format_parse**: v4 demonstrates the joint format changes are
  effective — keep this signal.
- **schema_valid**: Now the binding constraint; v5 must target >30% to
  escape `FIX_PROMPT_FIRST`.
- **arguments_valid = schema_valid**: In all v4 combinations these are
  identical, confirming schema failures are purely argument-shape
  mismatches (not `action_type` or `safety_flags` issues).

## 5. v4 Artifact Inventory

The `reports/p4/protocol-ablation-v4/` directory contains **13 files**:

- **11 core experiment artifacts** (listed in `artifact-manifest.json`):
  `baseline-lock.json`, `comparison-matrix.json`, `comparison-report.md`,
  `failure-taxonomy.json`, `verdict.json`, and the 6 trajectory JSONL
  files under `trajectories/`.
- **1 manifest file**: `artifact-manifest.json` itself (self-excluded
  from its own listing).
- **1 post-hoc analysis document**: `v4-vs-v3-analysis.md` (this file;
  not a core experiment artifact, not listed in the manifest).

The manifest's `artifact_count` field is 11, not 13. The directory file
count is 13. Both numbers are correct and refer to different concepts.

## 6. Conclusion

Under the v4 state, the format bottleneck observed in v3 was substantially
reduced (FORMAT_PARSE_FAIL -66%) by the joint changes to token budget,
JSON extraction, default injection, and protocol prompt formatting. The
bottleneck shifted to SCHEMA_VALIDATION_FAIL (+1442), indicating that
Qwen3-0.6B cannot infer Action argument schemas from minimal prompts.

The verdict remains `FIX_PROMPT_FIRST`, now pointing at argument-shape
education rather than format extraction. v5 directions are proposed
(§4) but not authorized or implemented in this PR.
