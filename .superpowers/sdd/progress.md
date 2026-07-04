# Subagent-Driven Development Progress Ledger

Branch: feat/p2.2-ci-router-validation
Started: 2026-07-03

## Tasks

- Task 1: Fix compare_p2_evals.py silent-skip bug — complete (commits 80c03de..deb8db9, review clean, Approved)
- Task 2: Extend compute_router_analysis.py with P3 Decision Gate — complete (commits deb8db9..5dbaf8b, review clean, Approved)
- Prep: Commit compute_paired_stats.py full576 rename — complete (commit 48efe4e)
- Task 3: Write generate_full576_report.py for Full-576 comparison report — complete (commits 48efe4e..6670523, review APPROVED_WITH_MINOR)
- Chore: Commit eval scripts + SDD process files — complete (commit d6f553d)
- Final whole-branch code review — complete (APPROVED_WITH_MINOR, no Critical/Important)
- Fix: UTF-8 encoding in generate_full576_report load_json — complete (commit d90ec9b)

## Final Review Minor Findings (triaged)

### Fixed:
- **#4 load_json encoding** (FIXED in d90ec9b): `generate_full576_report.py` load_json now uses `p.open(encoding="utf-8")` with `with` statement. Prevents mojibake when reading em-dashes from router-analysis.json on Windows.

### Deferred (acceptable for merge):
- **#1 §6 pair labels**: Raw model keys in Paired Statistics table (consistent with upstream compute_paired_stats.py)
- **#2 Test style**: Bare functions vs class grouping (valid pytest, style preference)
- **#3 Weak assertion**: `label in md` not section-scoped (still verifies model presence)
- **#5 Dead else branch**: `compute_router_analysis.py:299-304` unreachable else in apply_decision_gate (cosmetic)
- **#6 Stale comment**: `compute_paired_stats.py` "Issue #1" → "Issue #6" (cosmetic)

## Remaining Work (data-dependent, automated)

### Automation pipeline (running, verified correct):
- **Job 1** (`job-0670342adf0543e8bdc5dd371b4d2ca9`): Base Full-576 eval — in progress (~48%, 278/576)
- **Job 2** (`job-47f22e3d087442a6a608bb407ad7b0b7`): waits for Base → runs 4 remaining models sequentially
- **Job 3** (`job-cd2801f011ba4b6a9799b40ed1767fec`): waits for all 5 eval files → runs 4 analysis scripts → emits `POST_EVAL_PIPELINE_DONE_MARKER`

### Steps 2-4 are automated by Job 3:
2. compare_p2_evals.py → full576-comparison.json
3. compute_paired_stats.py → full576-paired-stats.json/md
4. compute_router_analysis.py → router-analysis.json/md (P3 Decision Gate)
5. generate_full576_report.py → p2-full576-comparison-report.md

### Pipeline COMPLETED (2026-07-04 ~11:12):
- All 5 eval files generated ✅
- Post-eval pipeline ran successfully ✅ (compare + paired_stats + router_analysis + report)
- P3 Decision Gate verdict: **SIGNAL** ✅
  - Oracle lift: 13.2pp (routing potential exists)
  - Deployable lift: 0.3pp (observable signals insufficient)
  - McNemar p: 0.7905 | 95% CI: [-0.0087, +0.0156]

### Full-576 Pass@1 Results:
1. Base: 16.4%
2. Stage3-v3-Antiforget: 15.7%
3. Stage3-Independent: 15.0%
4. Stage2-v2: 14.3%
5. Stage3-v2-Continual: 11.4%

### Git delivery:
- Local commit: f81a571 (12 files, 94827 insertions) ✅
- Push: BLOCKED — proxy at 127.0.0.1:7897 not running ❌
- PR + merge: pending push

### To complete:
1. Start proxy (Clash/V2Ray at port 7897)
2. `git push`
3. Create PR: `gh pr create --title "P2 Full-576 Evaluation + Router Analysis (P3: SIGNAL)" --body "..."`
4. Merge to main
