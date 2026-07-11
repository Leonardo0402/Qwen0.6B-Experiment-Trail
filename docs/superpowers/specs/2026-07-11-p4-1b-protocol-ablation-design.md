# P4.1b Protocol & Harness Ablation — Design Spec

> Issue: #29
> Phase: P4.1b (after P4.1 VERIFIED, before P4.2 Agent SFT)
> Date: 2026-07-11
> Base branch: main (commit 8ac2c5f, PR #28 merged)

## 1. Problem Statement

P4.1 smoke results revealed a critical finding: Qwen3-0.6B achieves 100% JSON parse rate but **0% schema validity**. The model can produce JSON-formatted text, but never produces a schema-valid Action. Every trajectory hit max_steps without a single valid action. This suggests the full Action JSON schema (discriminated union + nested arguments + safety_flags + 5 required base fields) is too high-entropy for a 0.6B model.

**Hypothesis:** A lower-entropy action protocol (XML Tag or One-line DSL) may improve parse validity, schema validity, and loop stability compared with full Action JSON.

**Non-goal:** This phase does not train, fine-tune, or modify the model. It only compares protocol formats via smoke runs on the untrained/adapted model.

## 2. Verified Starting State

- P4.1 verdict: `VERIFIED_GO_FOR_P4_AGENT_SFT` (PR #28 merged, 10/10 gates PASS)
- Model: `models/Qwen3-0.6B/` (1.4GB safetensors, available locally)
- Adapter: `adapters/p3/repair-limited/` (available locally)
- Micro-tasks: 40 tasks in `data/p4-agent/micro-tasks-v0/manifest.json`
- P4.1 smoke: 100% json_parse_rate, 0% schema_valid_rate (both base and repair-lora)

## 3. Scope

### In Scope
- Define 3 action protocols (JSON baseline, XML Tag, One-line DSL)
- Implement deterministic parsers for each protocol
- Implement protocol-aware prompt templates
- Add protocol abstraction to ModelActionProvider
- Run smoke on 40 micro-tasks × 3 protocols × 2 configs = 240 runs
- Produce protocol comparison report with metrics and failure taxonomy
- Output final verdict for P4.2 training target

### Out of Scope
- SFT / DPO / PPO / GRPO / RL training
- Model replacement or external datasets
- Unrestricted shell / network / Git push tools
- P4.2 training readiness (separate issue)
- Issue #27 training protocol decision (separate issue, after #29)

## 4. Architecture

### 4.1 Module Structure

```
src/protocols/
├── __init__.py          # Exports ProtocolBase, JsonProtocol, TagProtocol, DslProtocol
├── base.py              # ProtocolBase ABC + ProtocolDiagnostics dataclass
├── json_protocol.py     # Extracted from ModelActionProvider (baseline)
├── tag_protocol.py      # XML Tag protocol
└── dsl_protocol.py      # One-line DSL protocol

scripts/
└── run_protocol_ablation.py  # T6-T8: smoke run + report + verdict

tests/
├── test_protocol_json.py     # T3: JSON parser tests
├── test_protocol_tag.py      # T3: Tag parser tests
├── test_protocol_dsl.py      # T3: DSL parser tests
└── test_protocol_ablation.py # T6-T8 integration tests

reports/p4/protocol-ablation/
├── comparison-report.md
├── comparison-matrix.json
├── failure-taxonomy.json
├── baseline-lock.json
└── trajectories/
    ├── json-base.jsonl
    ├── json-repair-lora.jsonl
    ├── tag-base.jsonl
    ├── tag-repair-lora.jsonl
    ├── dsl-base.jsonl
    └── dsl-repair-lora.jsonl
```

### 4.2 ProtocolBase Interface

```python
class ProtocolBase(ABC):
    """Abstract base for action protocols."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol identifier (e.g. 'json', 'tag', 'dsl')."""

    @abstractmethod
    def build_system_prompt(self, task_context: str) -> str:
        """Build system prompt with protocol format instructions and tool semantics."""

    @abstractmethod
    def parse_output(self, raw: str) -> tuple[Action | SentinelAction, ProtocolDiagnostics]:
        """Parse model output into Action or SentinelAction with diagnostics."""
```

### 4.3 ModelActionProvider Changes

```python
class ModelActionProvider(ActionProvider):
    def __init__(self, model_path=..., adapter_path=...,
                 protocol: ProtocolBase | None = None,  # NEW
                 max_new_tokens=128):
        self._protocol = protocol or JsonProtocol()  # backward-compatible default
```

The provider delegates `build_system_prompt` and `parse_output` to the protocol object. All existing ModelActionProvider tests pass unchanged when `protocol=None`.

### 4.4 Data Flow

```
Task → ModelActionProvider.next_action(state)
         ↓
    protocol.build_system_prompt(task_context)  →  model.generate()
         ↓
    protocol.parse_output(raw_output)  →  (Action | SentinelAction, ProtocolDiagnostics)
         ↓
    AgentEvaluator dispatches action  →  tool result  →  next step
```

## 5. Protocol Specifications

### 5.1 Protocol A — Action JSON (Baseline)

**Format:**
```json
{"action_type": "read_file", "path": "solution.py", "reason_short": "inspect failing file"}
```

**Parser logic (extracted from existing ModelActionProvider):**
1. `extract_json(raw)` — regex extracts fenced `json` code block or bare `{...}` object
2. `json.loads()` → Pydantic `Action` discriminated-union validation
3. On failure: `repair_json(raw)` — format-only repair (strip markdown fences, remove trailing commas, append missing closing braces). Never alters semantics.
4. All failures → `SentinelAction(is_invalid=True)`

**Prompt:** Instructs model to output strict Action JSON with all required fields.

### 5.2 Protocol B — XML Tag

**Format:**
```
<action>
tool: read_file
path: solution.py
</action>
```

**Parser logic:**
1. `extract_tag(raw, "action")` — regex extracts `<action>...</action>` block
2. Parse key-value pairs (`key: value` per line)
3. Map `tool` → `action_type`, remaining keys map to action arguments
4. For multiline content (`old_text`, `new_text`): use `<content>...</content>` subtag
5. Pydantic `Action` validation
6. Failures → `SentinelAction(is_invalid=True)`

**Special handling:**
- `<think>...</think>` blocks are ignored (not passed to parser)
- Unknown keys hard-fail (not silently ignored)
- Unclosed tags hard-fail

### 5.3 Protocol C — One-line DSL

**Format:**
```
ACTION read_file path=solution.py
```

**Parser logic:**
1. `extract_dsl(raw)` — find line starting with `ACTION `
2. Parse `ACTION <action_type> <key>=<value> <key>=<value>...`
3. For multiline content: heredoc syntax (`<<EOF...EOF`)
4. Map to Action constructor arguments
5. Pydantic `Action` validation
6. Failures → `SentinelAction(is_invalid=True)`

**Heredoc example for propose_patch:**
```
ACTION propose_patch file_path=solution.py old_text<<EOF
def old():
    pass
EOF new_text<<EOF
def new():
    return 42
EOF
```

**Special handling:**
- Unknown action_type hard-fails
- Missing parameter values hard-fail
- Malformed heredoc hard-fails

### 5.4 Common Constraints

- All 3 protocols map to the **same Pydantic Action schema** (Action definition unchanged)
- All 3 protocols retain `safety_flags` checks (network/sensitive-path hard-reject)
- All 3 parsers return `(Action | SentinelAction, ProtocolDiagnostics)`
- No new dependencies (pydantic + regex only)

## 6. Diagnostics

### 6.1 ProtocolDiagnostics

```python
@dataclass
class ProtocolDiagnostics:
    raw_output: str
    format_parse_ok: bool       # Protocol format syntax correct (JSON/Tag/DSL)
    schema_valid: bool          # Pydantic Action validation passed
    safety_valid: bool          # safety_flags check passed (no network/sensitive path)
    action_type_valid: bool     # action_type in allowed 11 types
    arguments_valid: bool       # Argument types and values valid
    repair_attempted: bool      # Format repair was attempted
    repair_success: bool        # Format repair succeeded
    latency_ms: int             # Parse latency
    failure_class: str | None   # Failure classification (see §6.3)
```

This fixes the P4.1 issue where `schema_valid`, `safety_valid`, `action_type_valid`, and `arguments_valid` were all set to `True` together. Each dimension is now computed independently.

### 6.2 Aggregated Metrics (T7 Report)

Per protocol × per config, aggregated from trajectory `step_diagnostics`:

| Metric | Calculation |
|--------|-------------|
| `format_parse_rate` | `sum(format_parse_ok) / total_steps` |
| `schema_valid_rate` | `sum(schema_valid) / total_steps` |
| `safety_valid_rate` | `sum(safety_valid) / total_steps` |
| `action_type_valid_rate` | `sum(action_type_valid) / total_steps` |
| `arguments_valid_rate` | `sum(arguments_valid) / total_steps` |
| `forbidden_action_count` | `sum(not safety_valid)` |
| `unknown_action_count` | `sum(not action_type_valid)` |
| `task_success_rate` | `sum(success) / total_tasks` |
| `finish_without_tests_count` | From evaluator metrics |
| `finish_claim_mismatch_count` | From evaluator metrics |
| `max_steps_hit_rate` | `sum(steps_executed >= max_steps) / total_tasks` |
| `runtime_crash_count` | Count of crashed trajectories |

### 6.3 Failure Taxonomy

Each failed step is classified into one of:

```
FORMAT_PARSE_FAIL        — Protocol format syntax error
SCHEMA_VALIDATION_FAIL   — Format correct but Pydantic validation failed
UNKNOWN_ACTION_TYPE      — action_type not in 11 allowed types
FORBIDDEN_ACTION         — safety_flags triggered network/sensitive path
INVALID_PATH             — Path validation failed (absolute, .., URL)
EMPTY_OR_USELESS_ACTION  — Empty output or meaningless content
MODEL_REFUSAL_OR_CHATTER — Model refused or produced irrelevant content
REPEATED_ACTION_LOOP     — 3+ consecutive identical actions
```

## 7. T6 Smoke Run Flow

### 7.1 Script: `scripts/run_protocol_ablation.py`

```
Step 1: Baseline Lock (T1)
  - Record commit SHA, micro-task manifest SHA, evaluator SHA
  - Record model path, adapter path, generation config
  - Write to reports/p4/protocol-ablation/baseline-lock.json

Step 2: Run all 6 combinations (T6)
  for protocol in [JsonProtocol, TagProtocol, DslProtocol]:
    for config in [base, repair-lora]:
      a. Create ModelActionProvider(protocol=protocol, model_path=..., adapter_path=...)
      b. Load model (_load_model())
      c. For each of 40 micro-tasks:
         - MicroTaskWorkspace.from_task(task_dir)
         - AgentEvaluator(ws, provider, task_id, max_steps=12)
         - Record trajectory + step_diagnostics
         - ws.cleanup()
      d. Write trajectories/{protocol}-{config}.jsonl
      e. Aggregate metrics into summary

Step 3: Generate Comparison Report (T7)
  - Assemble 6 results into comparison-matrix.json
  - Group failure statistics by protocol
  - Generate comparison-report.md

Step 4: Output Verdict (T8)
  - Apply decision rules (see §7.2)
  - Write verdict to report
```

### 7.2 T8 Verdict Decision Rules

| Condition | Verdict |
|-----------|---------|
| Any alternative protocol's `schema_valid_rate` significantly better than JSON (>30%) AND `safety_valid_rate` not degraded | `TRY_TAG_PROTOCOL_FOR_P4_2` or `TRY_DSL_FOR_P4_2` |
| All protocols have `schema_valid_rate` < 30% | `FIX_PROMPT_FIRST` |
| JSON baseline has highest `schema_valid_rate` | `KEEP_ACTION_JSON` |
| Evaluator issues make metrics unreliable | `FIX_EVALUATOR_FIRST` |
| Protocol change risk too high | `STOP_PROTOCOL_CHANGE` |

### 7.3 Constraints

- Deterministic generation: `temperature=0.0, do_sample=False` for all runs
- Model reloaded between config combinations (no adapter residue)
- Independent temp workspace per task (`MicroTaskWorkspace.from_task`)
- Total: 240 agent loops (6 groups × 40 tasks), estimated ~2-3 hours

## 8. Testing Strategy

### 8.1 T3 Parser Unit Tests (TDD)

**test_protocol_json.py (10 tests):**
- `valid_action_parses` — complete JSON action parses successfully
- `fenced_json_parses` — `json` code block format parses
- `malformed_json_fails` — JSON syntax error returns SentinelAction
- `unknown_action_type_fails` — unknown action_type returns SentinelAction
- `forbidden_path_fails` — network path / `..` rejected
- `missing_required_field_fails` — missing action_type returns SentinelAction
- `repair_fixes_trailing_comma` — format repair succeeds
- `repair_does_not_change_semantics` — repair only fixes format
- `empty_output_fails` — empty output returns SentinelAction
- `multiline_content_handled` — propose_patch multiline content parses

**test_protocol_tag.py (9 tests):**
- `valid_tag_action_parses` — `<action>tool: read_file\npath: x.py</action>`
- `think_block_ignored` — `<think>...</think>` does not affect parsing
- `content_subtag_handled` — `<content>` multiline content
- `unclosed_tag_fails` — unclosed `<action>` returns SentinelAction
- `unknown_key_fails` — unknown key hard-fails
- Plus action_type/path/forbidden tests (same as JSON)

**test_protocol_dsl.py (9 tests):**
- `valid_dsl_action_parses` — `ACTION read_file path=solution.py`
- `heredoc_multiline_handled` — `old_text<<EOF...EOF`
- `malformed_dsl_fails` — not starting with `ACTION` returns SentinelAction
- `missing_value_fails` — `path=` with no value returns SentinelAction
- Plus action_type/path/forbidden tests (same as JSON)

### 8.2 T6 Integration Tests

**test_protocol_ablation.py (7 tests):**
- `baseline_lock_records_all_fields` — T1 baseline lock complete
- `all_6_combinations_executed` — 3 protocols × 2 configs all executed
- `same_task_set_for_all` — all combinations use same 40 tasks
- `deterministic_generation_config` — temperature=0.0, do_sample=False
- `comparison_matrix_has_all_metrics` — T7 report has all metrics
- `failure_taxonomy_classified` — failed steps classified
- `verdict_is_valid_enum` — T8 verdict in allowed values

### 8.3 Regression Tests

- `test_agent_model_provider.py` — all existing tests pass (protocol=None defaults to JSON)
- `test_agent_evaluator.py` — all existing tests pass (evaluator unchanged)
- P4.1 readiness 10 gates — all pass

## 9. Safety and Resource Constraints

- Same safe tool layer (no shell/network/credential/Git push)
- Path-restricted, patch-oriented, test-driven, rollback-capable
- All fits in RTX 3050 4GB VRAM (model loaded with float16)
- No training, no external data, no model weights committed
- Deterministic generation (greedy decoding, no sampling)

## 10. Acceptance Criteria

- [ ] Existing Action JSON remains the baseline (Protocol A)
- [ ] At least 2 alternative protocols specified and tested (Protocol B, C)
- [ ] Each protocol has a deterministic parser
- [ ] Unknown or malformed actions hard-fail
- [ ] No silent no-op exists
- [ ] All protocol runs use the same task set (40 micro-tasks)
- [ ] All protocol runs record reproducible generation settings
- [ ] All tool calls are replayable
- [ ] Safety violations are counted
- [ ] Protocol comparison report produced with 12+ metrics
- [ ] Failure taxonomy classifies all failed steps
- [ ] Final verdict is one of: KEEP_ACTION_JSON / TRY_TAG_PROTOCOL_FOR_P4_2 / TRY_DSL_FOR_P4_2 / FIX_PROMPT_FIRST / FIX_EVALUATOR_FIRST / STOP_PROTOCOL_CHANGE
- [ ] All parser unit tests pass (T3)
- [ ] All integration tests pass (T6-T8)
- [ ] All regression tests pass (existing test suite)
- [ ] No new dependencies introduced

## 11. PR Relationship

- `Refs #29` (protocol ablation)
- `Refs #28` (P4.1 roadmap blockers, merged)
- Does NOT close #29 until verdict is produced and accepted
- Does NOT close #27 (separate training protocol decision)
