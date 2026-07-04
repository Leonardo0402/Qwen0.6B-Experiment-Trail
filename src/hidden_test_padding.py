"""Shared module for padding hidden tests with boundary-condition asserts.

Extracted from scripts/build_frozen_v3_samples.py per P3 plan v2.1 Amendment A2.
Used by both build_frozen_v3_samples.py (Frozen v3 build) and
verify_imported_mbpp.py (import verifier pad-then-verify flow).

The padding logic:
1. Parse target_code to find the first top-level function.
2. Extract template call args from existing public tests.
3. Generate boundary calls by varying each argument.
4. Execute all boundary calls in one harness run.
5. Build bare-assert tests from non-exception results.
6. Append the new tests to hidden_tests until target_count is reached.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.generate_boundary_variants import (  # noqa: E402
    _ast_to_value,
    _boundary_values_for_type,
    _classify_arg,
    _execute_boundary_calls,
    _extract_test_calls,
    _parse_function_info,
)
from src.schemas import Sample, Verification  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_HIDDEN_COUNT: int = 3


# ---------------------------------------------------------------------------
# pad_hidden_tests
# ---------------------------------------------------------------------------

def pad_hidden_tests(
    sample: Sample,
    *,
    target_count: int = TARGET_HIDDEN_COUNT,
) -> tuple[Sample, Optional[str]]:
    """Pad *sample.hidden_tests* with boundary-condition asserts until the
    ``assert `` substring count reaches *target_count*.

    Returns ``(padded_sample, rejection_reason)``. *rejection_reason* is
    ``None`` on success, or one of:
      - ``hidden_padding_failed_syntax_error``
      - ``hidden_padding_failed_no_functions``
      - ``hidden_padding_insufficient``

    The padding helper must NOT modify ``target_code``, ``public_tests``,
    or ``instruction``. Only ``hidden_tests`` is extended.

    Performance note: this implementation calls ``_execute_boundary_calls``
    (one ``run_python_code`` subprocess) to capture expected outputs for all
    boundary candidates in a single harness run. The generated asserts are
    guaranteed to pass against *target_code* (because the expected output
    was captured by calling the function), so no per-test ``run_pytest``
    verification is needed. This is a deliberate optimisation over the
    brief's suggested per-test ``run_pytest`` loop.
    """
    current_count = sample.hidden_tests.count("assert ")
    if current_count >= target_count:
        return sample, None

    # Parse target_code to find the first top-level function.
    info = _parse_function_info(sample.target_code)
    if info is None:
        # Could be a SyntaxError or no function def. Distinguish by re-parsing.
        try:
            ast.parse(sample.target_code)
        except SyntaxError:
            return sample, "hidden_padding_failed_syntax_error"
        return sample, "hidden_padding_failed_no_functions"

    func_name, param_names = info

    # Extract template call args from existing public tests.
    test_calls = _extract_test_calls(sample.public_tests, func_name)
    if not test_calls:
        return sample, "hidden_padding_insufficient"

    template_args = test_calls[0]
    try:
        base_values = [_ast_to_value(a) for a in template_args]
    except ValueError:
        return sample, "hidden_padding_insufficient"

    # Generate boundary calls by varying each argument.
    boundary_calls: list[tuple[str, list[Any]]] = []
    for i, arg in enumerate(template_args):
        if i >= len(param_names):
            break
        type_name = _classify_arg(arg)
        boundaries = _boundary_values_for_type(type_name, arg)
        for label, value in boundaries:
            new_args = list(base_values)
            new_args[i] = value
            boundary_calls.append((f"{param_names[i]}_{label}", new_args))

    if not boundary_calls:
        return sample, "hidden_padding_insufficient"

    # Execute all boundary calls in one harness run.
    exec_results = _execute_boundary_calls(
        sample.target_code, func_name, param_names, boundary_calls
    )
    if not exec_results:
        return sample, "hidden_padding_insufficient"

    # Build bare-assert tests from non-exception results.
    added_tests: list[str] = []
    for res in exec_results:
        if current_count + len(added_tests) >= target_count:
            break
        if res.get("exception_type") is not None:
            continue  # skip exception-raising cases (no bare assert form)
        args = res["args"]
        output = res["output"]
        args_repr = ", ".join(repr(a) for a in args)
        test_line = f"assert {func_name}({args_repr}) == {output}"
        added_tests.append(test_line)

    if current_count + len(added_tests) < target_count:
        return sample, "hidden_padding_insufficient"

    new_hidden = sample.hidden_tests.rstrip() + "\n\n" + "\n\n".join(added_tests)
    return sample.model_copy(update={"hidden_tests": new_hidden}), None


# ---------------------------------------------------------------------------
# normalize_public_tests_for_pytest
# ---------------------------------------------------------------------------

def normalize_public_tests_for_pytest(public_tests: str) -> str:
    """Ensure *public_tests* can be collected by pytest when mixed formats exist.

    ``generate_boundary_variant`` appends pytest test functions (with
    ``from solution import`` and ``def test_*``) to the original bare-assert
    public tests. This creates a mixed file where bare asserts run before
    the import line, causing ``NameError`` during pytest collection (because
    ``_normalize_test_code`` in ``src/sandbox.py`` returns the code as-is when
    it detects ``from solution`` or ``def test``).

    Fix: if there are bare asserts before any ``from solution`` line, prepend
    ``from solution import *`` at the top so the bare asserts resolve the
    function under test. The import is redundant for the pytest test
    functions (which have their own import) but harmless.
    """
    if not public_tests or not public_tests.strip():
        return public_tests
    if "from solution" not in public_tests:
        return public_tests  # pure bare asserts; _normalize_test_code handles it
    if "assert " not in public_tests:
        return public_tests
    lines = public_tests.split("\n")
    import_line_idx: int | None = None
    for i, line in enumerate(lines):
        if line.strip().startswith("from solution"):
            import_line_idx = i
            break
    if import_line_idx is None:
        return public_tests
    before_import = "\n".join(lines[:import_line_idx])
    if "assert " in before_import:
        return "from solution import *\n\n" + public_tests
    return public_tests
