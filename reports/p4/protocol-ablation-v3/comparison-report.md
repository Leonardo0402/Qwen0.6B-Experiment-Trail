# P4.1b Protocol Ablation — Comparison Report (protocol-ablation-v3)

## Overview

- Protocols: 3
- Configs: 2
- Total combinations: 6
- Report dir: protocol-ablation-v3

## Metrics by Protocol x Config

Rates shown as `numerator / denominator = rate`.

| Protocol | Config | schema_valid | arguments_valid | task_success | max_steps_hit | unknown_actions | finish_no_tests | crashes |
|----------|--------|--------------|-----------------|--------------|---------------|-----------------|-----------------|---------|
| dsl | base | 0/480 = 0.00% | 0/480 = 0.00% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| dsl | repair-lora | 29/480 = 6.04% | 29/480 = 6.04% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| json | base | 0/480 = 0.00% | 0/480 = 0.00% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| json | repair-lora | 0/480 = 0.00% | 0/480 = 0.00% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| tag | base | 11/480 = 2.29% | 11/480 = 2.29% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| tag | repair-lora | 8/480 = 1.67% | 8/480 = 1.67% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |

## Detailed Step-Level Metrics

| Protocol | Config | total_steps | format_parse | safety_valid | action_type_valid |
|----------|--------|-------------|--------------|--------------|-------------------|
| dsl | base | 480 | 0/480 = 0.00% | 0/480 = 0.00% | 0/480 = 0.00% |
| dsl | repair-lora | 480 | 226/480 = 47.08% | 154/480 = 32.08% | 226/480 = 47.08% |
| json | base | 480 | 12/480 = 2.50% | 12/480 = 2.50% | 12/480 = 2.50% |
| json | repair-lora | 480 | 480/480 = 100.00% | 480/480 = 100.00% | 480/480 = 100.00% |
| tag | base | 480 | 11/480 = 2.29% | 11/480 = 2.29% | 11/480 = 2.29% |
| tag | repair-lora | 480 | 30/480 = 6.25% | 30/480 = 6.25% | 30/480 = 6.25% |

## Failure Taxonomy

| Failure Class | Count |
|---------------|-------|
| EMPTY_OR_USELESS_ACTION | 0 |
| FORBIDDEN_ACTION | 0 |
| FORMAT_PARSE_FAIL | 2193 |
| INVALID_PATH | 0 |
| MODEL_REFUSAL_OR_CHATTER | 0 |
| REPEATED_ACTION_LOOP | 240 |
| SCHEMA_VALIDATION_FAIL | 639 |
| UNKNOWN_ACTION_TYPE | 0 |

## Protocol Comparison Summary

- **dsl**: avg schema_valid_rate = 3.02%
- **json**: avg schema_valid_rate = 0.00%
- **tag**: avg schema_valid_rate = 1.98%

## Verdict

**FIX_PROMPT_FIRST**
