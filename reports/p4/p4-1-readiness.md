# P4.1 Readiness Report

**Verdict:** GO_FOR_P4_AGENT_SFT

**Date:** 2026-07-11T01:48:07.559591
**Commit:** 071ab2a

## Gates

| Gate | Status | Evidence |
|---|---|---|
| 01_p4_0_baseline_lock | PASS | locked at 7ccd06c |
| 02_test_pass_replay_authoritative | PASS | 3 passed in 28.34s |
| 03_unknown_action_hard_fails | PASS | 2 passed in 1.03s |
| 04_all_11_actions_dispatched | PASS | 2 passed in 1.46s |
| 05_inspect_error_surfaces_stdout | PASS | 2 passed in 0.57s |
| 06_all_5_corruption_types_tested | PASS | 5 passed in 42.77s |
| 07_model_smoke_base | PASS | loaded (full), 40 trajectories, json_parse=1.00, schema_valid=0.00, forbidden=0, tool_dispatch_ok=1.00, max_step_stop=40 |
| 08_model_smoke_repair_lora | PASS | loaded (full), 40 trajectories, json_parse=1.00, schema_valid=0.00, forbidden=0, tool_dispatch_ok=1.00, max_step_stop=40 |
| 09_sft_dataset | PASS | 1350 trajectories, train=920 val=130 heldout=220 |
| 10_no_training_no_external_data | PASS | no training, no external data, no weights committed |

## Test Evidence (§2.8)

### Local Tests

- **Exact command:** `C:\Users\20385\AppData\Local\Programs\Python\Python311\python.exe -m pytest tests/ -p no:warnings --tb=no -q -m "not gpu" --timeout=120 --ignore=tests/test_data_pipeline.py --ignore=tests/test_p3_readiness_gate.py --junit-xml=C:\Users\20385\AppData\Local\Temp\pytest_junit_du04vmi5.xml`
- **Test count:** 1285
- **Pass count:** 1284
- **Fail count:** 0
- **Skip count:** 1
- **Warning count:** 0
- **Error count:** 0
- **Runtime:** 1074.17s
- **Python:** 3.11.7
- **Platform:** Windows-10-10.0.26200-SP0
- **Machine:** AMD64
- **Commit SHA:** 071ab2a

### GitHub CI

- **Status:** not run (local execution)

**Endpoint:** GO_FOR_P4_AGENT_SFT

P4.1 is complete. `GO_FOR_P4_AGENT_SFT` authorizes considering P4.2 (Agent SFT training). It does NOT authorize training. Training requires a separate P4.2 issue + user approval.