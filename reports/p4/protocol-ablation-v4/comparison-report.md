# P4.1b Protocol Ablation — Comparison Report (protocol-ablation-v4)

## Overview

- Protocols: 3
- Configs: 2
- Total combinations: 6
- Report dir: protocol-ablation-v4

## Metrics by Protocol x Config

Rates shown as `numerator / denominator = rate`.

| Protocol | Config | schema_valid | arguments_valid | task_success | max_steps_hit | unknown_actions | finish_no_tests | crashes |
|----------|--------|--------------|-----------------|--------------|---------------|-----------------|-----------------|---------|
| dsl | base | 0/480 = 0.00% | 0/480 = 0.00% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| dsl | repair-lora | 0/480 = 0.00% | 0/480 = 0.00% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| json | base | 0/480 = 0.00% | 0/480 = 0.00% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| json | repair-lora | 0/480 = 0.00% | 0/480 = 0.00% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| tag | base | 50/480 = 10.42% | 50/480 = 10.42% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |
| tag | repair-lora | 5/480 = 1.04% | 5/480 = 1.04% | 0/40 = 0.00% | 40/40 = 100.00% | 0 | 0 | 0 |

## Detailed Step-Level Metrics

| Protocol | Config | total_steps | format_parse | safety_valid | action_type_valid |
|----------|--------|-------------|--------------|--------------|-------------------|
| dsl | base | 480 | 384/480 = 80.00% | 384/480 = 80.00% | 384/480 = 80.00% |
| dsl | repair-lora | 480 | 480/480 = 100.00% | 480/480 = 100.00% | 480/480 = 100.00% |
| json | base | 480 | 216/480 = 45.00% | 216/480 = 45.00% | 216/480 = 45.00% |
| json | repair-lora | 480 | 480/480 = 100.00% | 480/480 = 100.00% | 480/480 = 100.00% |
| tag | base | 480 | 96/480 = 20.00% | 96/480 = 20.00% | 96/480 = 20.00% |
| tag | repair-lora | 480 | 480/480 = 100.00% | 480/480 = 100.00% | 480/480 = 100.00% |

## Failure Taxonomy

| Failure Class | Count |
|---------------|-------|
| EMPTY_OR_USELESS_ACTION | 0 |
| FORBIDDEN_ACTION | 0 |
| FORMAT_PARSE_FAIL | 744 |
| INVALID_PATH | 0 |
| MODEL_REFUSAL_OR_CHATTER | 0 |
| REPEATED_ACTION_LOOP | 240 |
| SCHEMA_VALIDATION_FAIL | 2081 |
| UNKNOWN_ACTION_TYPE | 0 |

## Protocol Comparison Summary

- **dsl**: avg schema_valid_rate = 0.00%
- **json**: avg schema_valid_rate = 0.00%
- **tag**: avg schema_valid_rate = 5.73%

## Verdict

**FIX_PROMPT_FIRST**
