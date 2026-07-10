# Task 8 Report — Phase E: GPU smoke tests (base + repair-lora)

**Status:** DONE_WITH_CONCERNS

**Commit SHA:** `7069daf`

**BASE:** `2f244cd` (T7 commit) → HEAD: `7069daf`

---

## 1. What was implemented

- **Created** `tests/test_agent_model_provider_gpu.py` (78 lines) — verbatim from the task brief. Two GPU smoke tests:
  - `test_model_provider_smoke_base` — loads base Qwen3-0.6B (`adapter_path=None`), runs task_001 with `max_steps=12`, asserts `forbidden_action_count == 0` and `len(provider.diagnostics) > 0`.
  - `test_model_provider_smoke_repair_lora` — loads Qwen3-0.6B + `adapters/p3/repair-limited`, same assertions.
  - Module-level `pytestmark = pytest.mark.gpu` ensures both tests are deselected by CI's `-m "not gpu"`.
- **Modified** `pyproject.toml` — registered the `gpu` marker under `[tool.pytest.ini_options]` (Pre-flight Correction 1). Prevents `PytestUnknownMarkWarning` and documents the marker's purpose:
  ```toml
  markers = [
      "gpu: tests requiring a physical GPU (skipped in CI, run manually on RTX 3050 before PR merge)",
  ]
  ```

## 2. Collection + skip evidence

**Command 1 (collect-only):**
```
py -3.11 -m pytest tests/test_agent_model_provider_gpu.py --collect-only -p no:warnings
```
Output:
```
tests/test_agent_model_provider_gpu.py::test_model_provider_smoke_base
tests/test_agent_model_provider_gpu.py::test_model_provider_smoke_repair_lora

2 tests collected in 0.33s
```
Result: **2 tests collected** ✓

**Command 2 (skip via marker exclusion):**
```
py -3.11 -m pytest tests/test_agent_model_provider_gpu.py -v -p no:warnings -m "not gpu"
```
Output:
```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\agent\Qwen\qwen3-code-lab
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.155.7, timeout-2.4.0, xdist-3.8.0
collected 2 items / 2 deselected / 0 selected

============================ 2 deselected in 0.29s ============================
```
Exit code: 5 (pytest's standard "no tests collected" code).

Result: **2 deselected** — see concern below.

## 3. Broader regression check

```
py -3.11 -m pytest tests/test_agent_model_provider.py tests/test_agent_evaluator.py -p no:warnings -q --timeout=120
```
Output:
```
...............................                                          [100%]

============================== 31 passed in 6.93s ==============================
```
Result: **31 passed, 0 failed** — no regressions from the `pyproject.toml` markers addition. ✓

## 4. Diffstat

```
git diff --stat 2f244cd HEAD
```
Output:
```
 pyproject.toml                         |  3 ++
 tests/test_agent_model_provider_gpu.py | 78 ++++++++++++++++++++++++++++++++++
 2 files changed, 81 insertions(+)
```

## 5. Corrections applied

- **Correction 1 (gpu marker registration):** Applied. The `markers` list was added to `[tool.pytest.ini_options]` in `pyproject.toml` with the exact entry specified in the brief. Without this, pytest would emit `PytestUnknownMarkWarning` for the `pytestmark = pytest.mark.gpu` declaration in the new test file. ✓

## 6. Self-review findings

- **Surgical changes:** Only `tests/test_agent_model_provider_gpu.py` (new) and `pyproject.toml` (3-line addition) were modified. No source files touched. `src/agent_trajectory.py` (frozen) untouched. ✓
- **Test code verbatim:** The test file matches the brief byte-for-byte (no creative additions, no extra error handling, no refactors). ✓
- **Staging discipline:** Only the two intended files were staged and committed. The `.superpowers/sdd/` doc modifications (progress.md, task briefs, review packages) were left unstaged. ✓
- **No emojis, no incidental docstrings, no refactors.** ✓
- **Python version:** Used `py -3.11` as specified. ✓
- **Commit message:** Matches the brief verbatim. ✓

## 7. Concerns

### Concern A: "2 SKIPPED" vs "2 deselected" wording mismatch

The brief's Step 2 expected output was "2 SKIPPED (no GPU marker active)" for the command `py -3.11 -m pytest tests/test_agent_model_provider_gpu.py -v -p no:warnings -m "not gpu"`. The actual pytest behavior with `-m "not gpu"` is **deselection**, not **skipping**:

- **Deselect** (what `-m` does): tests matching the exclusion marker are removed from the collection entirely → exit code 5 when all tests are deselected, output "2 deselected / 0 selected".
- **Skip** (what `pytest.mark.skip` / `skipif` does): tests are collected and reported as SKIPPED with an "s" in the progress bar, exit code 0.

Both mechanisms achieve the functional goal — GPU tests do not execute in CI — but the brief's wording was imprecise. The marker-based deselection pattern (`-m "not gpu"`) is the correct and conventional way to exclude GPU tests in CI, so the implementation is functionally correct. The exit code 5 is expected when an all-GPU file is run with `-m "not gpu"` and is not an error.

If "2 SKIPPED" output is strictly required (e.g., to satisfy a downstream check that greps for "skipped"), an alternative would be to use `@pytest.mark.skipif` with an environment check instead of `-m "not gpu"`. However, that would deviate from the brief's `pytestmark = pytest.mark.gpu` pattern, so I kept the brief's verbatim code. Flagging for parent agent awareness.

### Concern B: GPU execution not verified

As expected in this environment (no GPU), the two tests were not actually executed — only collected and deselected. The actual GPU smoke run on the RTX 3050 must happen manually before PR merge, as the brief specifies. This is by design, not a defect.

## 8. One-line test summary

2 GPU smoke tests created and registered with `gpu` marker; collected successfully (2/2) and excluded via `-m "not gpu"` (2 deselected, exit 5); 31 existing unit tests pass with no regressions; actual GPU execution deferred to manual RTX 3050 run before PR merge.
