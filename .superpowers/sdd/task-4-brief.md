## Task 4: Phase C — inspect_error returns stdout+stderr capped 8KB

**Files:**
- Modify: `src/agent_tools.py` (`tool_inspect_error` last_test branch)
- Modify: `tests/test_agent_tools.py` (+2 tests)

**Interfaces:**
- Consumes: `TestObservation.stdout`, `TestObservation.stderr`
- Produces: `ErrorObservation.content` = stdout+stderr[:8192] for `last_test` source

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_tools.py`:

```python
# --- Task 4: inspect_error returns stdout+stderr capped 8KB ---

def test_inspect_error_returns_stdout_on_test_failure():
    """Failed run_tests writes traceback to stdout; inspect_error must
    surface it, not return empty content."""
    test_obs = TestObservation(
        passed=False, returncode=1,
        stdout="AssertionError: expected 5 but got 4",
        stderr="",
        summary="1 failed",
    )
    result = tool_inspect_error(
        error_source="last_test",
        last_test_observation=test_obs,
        last_patch_observation=None,
    )
    assert result.content != "", "inspect_error returned empty content for stdout-only failure"
    assert "AssertionError" in result.content


def test_inspect_error_caps_at_8kb():
    """stdout+stderr > 8KB → content truncated to exactly 8192 chars."""
    big_stdout = "x" * 10000
    test_obs = TestObservation(
        passed=False, returncode=1,
        stdout=big_stdout, stderr="",
        summary="1 failed",
    )
    result = tool_inspect_error(
        error_source="last_test",
        last_test_observation=test_obs,
        last_patch_observation=None,
    )
    assert len(result.content) == 8192, \
        f"expected 8192 chars (8KB cap), got {len(result.content)}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.11 -m pytest tests/test_agent_tools.py::test_inspect_error_returns_stdout_on_test_failure tests/test_agent_tools.py::test_inspect_error_caps_at_8kb -v -p no:warnings`
Expected: FAIL — first test fails because `content=stderr` (empty); second fails because no cap applied.

- [ ] **Step 3: Implement the fix**

In `src/agent_tools.py`, in `tool_inspect_error`, replace the `last_test` branch:

```python
if error_source == "last_test":
    if last_test_observation is None:
        raise ValueError("no prior run_tests observation")
    raw = last_test_observation.stdout + "\n" + last_test_observation.stderr
    capped = raw[:8192]
    return ErrorObservation(source="last_test", content=capped)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_agent_tools.py -v -p no:warnings`
Expected: All tests PASS. No regressions.

- [ ] **Step 5: Commit**

```bash
git add src/agent_tools.py tests/test_agent_tools.py
git commit -m "feat(p4-1): Phase C — inspect_error returns stdout+stderr capped 8KB"
```

---

