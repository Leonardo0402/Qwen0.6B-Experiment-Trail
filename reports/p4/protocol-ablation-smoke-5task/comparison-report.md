# P4.1b Protocol Ablation — Comparison Report

## Overview

- Protocols: 3
- Configs: 2
- Total combinations: 6

## Metrics by Protocol x Config

| Protocol | Config | format_parse_rate | schema_valid_rate | safety_valid_rate | action_type_valid_rate | arguments_valid_rate | forbidden_count | unknown_count | task_success_rate | finish_no_tests | finish_mismatch | max_steps_hit_rate | crashes |
|----------|--------|-------------------|-------------------|-------------------|------------------------|----------------------|-----------------|----------------|-------------------|------------------|-----------------|---------------------|---------|
| dsl | base | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 60 | 0.00% | 0 | 0 | 100.00% | 0 |
| dsl | repair-lora | 21.67% | 3.33% | 3.33% | 21.67% | 3.33% | 0 | 47 | 0.00% | 0 | 0 | 100.00% | 0 |
| json | base | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 60 | 0.00% | 0 | 0 | 100.00% | 0 |
| json | repair-lora | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0 | 0 | 0.00% | 0 | 0 | 100.00% | 0 |
| tag | base | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 60 | 0.00% | 0 | 0 | 100.00% | 0 |
| tag | repair-lora | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 60 | 0.00% | 0 | 0 | 100.00% | 0 |

## Failure Taxonomy

| Failure Class | Count |
|---------------|-------|
| EMPTY_OR_USELESS_ACTION | 0 |
| FORBIDDEN_ACTION | 0 |
| FORMAT_PARSE_FAIL | 298 |
| INVALID_PATH | 0 |
| MODEL_REFUSAL_OR_CHATTER | 0 |
| REPEATED_ACTION_LOOP | 30 |
| SCHEMA_VALIDATION_FAIL | 0 |
| UNKNOWN_ACTION_TYPE | 0 |

## Protocol Comparison Summary

- **dsl**: avg schema_valid_rate = 1.67%
- **json**: avg schema_valid_rate = 50.00%
- **tag**: avg schema_valid_rate = 0.00%

## Verdict

**KEEP_ACTION_JSON**
