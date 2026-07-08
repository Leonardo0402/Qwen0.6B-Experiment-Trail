# P4.0 Phase F — Scripted Teacher Trajectories Report

## Summary
- Total trajectories: 40
- Total steps: 400
- Mean steps per trajectory: 10.00
- Success (types 2-8): 35
- Identify-only (type 1): 5

## Tool distribution (aggregate)
| Action type | Count |
|---|---|
| apply_patch | 40 |
| finish | 40 |
| inspect_error | 45 |
| inspect_task | 40 |
| list_files | 40 |
| propose_patch | 40 |
| read_file | 45 |
| run_tests | 75 |
| write_memory | 35 |

## Per-task-type breakdown
| Type | Tasks | Steps | Pattern |
|---|---|---|---|
| locate_failing_function | 5 | 7 each | identify only |
| one_line_fix | 5 | 10 each | standard |
| add_boundary_check | 5 | 10 each | standard |
| update_helper | 5 | 10 each | standard |
| repair_after_pytest | 5 | 10 each | standard |
| avoid_editing_tests | 5 | 11 each | standard + extra read |
| recover_from_failed_patch | 5 | 12 each | failed patch + recover |
| finish_after_tests_pass | 5 | 10 each | standard |

## Generation
- Script: scripts/generate_scripted_agent_trajectories.py
- Output: data/p4-agent/trajectories-v0/scripted.jsonl
- Format: one Trajectory JSON object per line (40 lines)
- All steps verified=True (real tool execution)
