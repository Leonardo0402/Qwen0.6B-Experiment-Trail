# Task: task_032

## Goal
Fix the bug in saturate(). The obvious fix may not work — inspect errors and retry.

## Constraints
- Do not modify test_*.py files

## Hints
- The function should return lo when x < lo, hi when x > hi, else x. The current code returns the wrong bound for each case.
