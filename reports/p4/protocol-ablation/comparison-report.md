# P4.1b Protocol Ablation — Comparison Report

## Overview

- Protocols: 3
- Configs: 2
- Total combinations: 6

## Metrics by Protocol x Config

| Protocol | Config | format_parse_rate | schema_valid_rate | safety_valid_rate | action_type_valid_rate | arguments_valid_rate | forbidden_count | unknown_count | task_success_rate | finish_no_tests | finish_mismatch | max_steps_hit_rate | crashes |
|----------|--------|-------------------|-------------------|-------------------|------------------------|----------------------|-----------------|----------------|-------------------|------------------|-----------------|---------------------|---------|
| dsl | base | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 480 | 0.00% | 0 | 0 | 100.00% | 0 |
| dsl | repair-lora | 32.29% | 8.12% | 8.12% | 32.29% | 8.12% | 0 | 325 | 0.00% | 0 | 0 | 100.00% | 0 |
| json | base | 2.50% | 0.00% | 0.00% | 2.50% | 0.00% | 0 | 468 | 0.00% | 0 | 0 | 100.00% | 0 |
| json | repair-lora | 96.25% | 96.25% | 96.25% | 96.25% | 96.25% | 0 | 18 | 0.00% | 0 | 0 | 100.00% | 0 |
| tag | base | 2.29% | 2.29% | 2.29% | 2.29% | 2.29% | 0 | 469 | 0.00% | 0 | 0 | 100.00% | 0 |
| tag | repair-lora | 6.25% | 1.67% | 1.67% | 6.25% | 1.67% | 0 | 450 | 0.00% | 0 | 0 | 100.00% | 0 |

## Failure Taxonomy

| Failure Class | Count |
|---------------|-------|
| EMPTY_OR_USELESS_ACTION | 0 |
| FORBIDDEN_ACTION | 0 |
| FORMAT_PARSE_FAIL | 2282 |
| INVALID_PATH | 0 |
| MODEL_REFUSAL_OR_CHATTER | 0 |
| REPEATED_ACTION_LOOP | 240 |
| SCHEMA_VALIDATION_FAIL | 78 |
| UNKNOWN_ACTION_TYPE | 0 |

## Protocol Comparison Summary

- **dsl**: avg schema_valid_rate = 4.06%
- **json**: avg schema_valid_rate = 48.12%
- **tag**: avg schema_valid_rate = 1.98%

## Verdict

**KEEP_ACTION_JSON**
