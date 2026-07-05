"""
scripts/generate_boundary_variants.py -- Boundary-condition variant generator (P2.2).

Takes correct code_generation samples and produces new code_generation samples
with enhanced boundary-condition test cases.  The target_code is unchanged
(verified correct), but the test suite is extended with boundary tests that
exercise edge cases:

  - Empty list / empty string
  - Single element
  - Boundary values (0, -1, large)
  - Duplicate elements
  - None input (if the function handles it or raises a documented exception)

The new sample keeps task_type=code_generation but has difficulty+1 (capped at 4),
because the enhanced test suite demands more robust handling.

Usage
-----
    python scripts/generate_boundary_variants.py --input <sample.jsonl> --output <output.jsonl> --seed 42
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sandbox import run_python_code  # noqa: E402
from src.schemas import Sample, Verification  # noqa: E402


_PLACEHOLDER_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)
_DATASET_VERSION = "p2.2"
_GENERATOR = "boundary_variant_v1"


# ---------------------------------------------------------------------------
# Function signature parsing
# ---------------------------------------------------------------------------

def _parse_function_info(code: str) -> Optional[tuple[str, list[str]]]:
    """Extract (function_name, param_names) from the first top-level def.

    Returns None if no function definition is found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            params = [arg.arg for arg in node.args.args]
            return node.name, params
    return None


# ---------------------------------------------------------------------------
# Test-call argument extraction
# ---------------------------------------------------------------------------

def _extract_test_calls(test_code: str, func_name: str) -> list[list[ast.expr]]:
    """Extract argument lists from `assert func_name(...)` calls in tests.

    Returns a list of arg-lists, one per call site found.
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []
    calls: list[list[ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == func_name:
                calls.append(list(node.args))
    return calls


def _classify_arg(arg: ast.expr) -> str:
    """Classify an AST argument expression as 'list', 'str', 'int', 'float', or 'unknown'."""
    if isinstance(arg, ast.List):
        return "list"
    if isinstance(arg, ast.Constant):
        if isinstance(arg.value, str):
            return "str"
        if isinstance(arg.value, bool):
            return "bool"
        if isinstance(arg.value, int):
            return "int"
        if isinstance(arg.value, float):
            return "float"
        if arg.value is None:
            return "none"
    return "unknown"


# ---------------------------------------------------------------------------
# Boundary value generation
# ---------------------------------------------------------------------------

def _boundary_values_for_type(type_name: str, example: ast.expr) -> list[tuple[str, Any]]:
    """Generate (label, value) boundary pairs for the given type.

    The 'example' AST node is used to infer element types for lists.
    """
    if type_name == "list":
        # Infer element type from the example list
        elem_type = "int"
        if isinstance(example, ast.List) and example.elts:
            elem_type = _classify_arg(example.elts[0])
        boundaries: list[tuple[str, Any]] = [
            ("empty", []),
            ("single", [_default_for_type(elem_type)]),
            ("duplicates", [_default_for_type(elem_type)] * 3),
        ]
        if elem_type == "int":
            boundaries.append(("with_zero", [0]))
            boundaries.append(("with_negative", [-1]))
        return boundaries
    if type_name == "str":
        return [
            ("empty_string", ""),
            ("single_char", "a"),
            ("repeated", "aaa"),
            ("with_spaces", "a b c"),
        ]
    if type_name == "int":
        return [
            ("zero", 0),
            ("negative_one", -1),
            ("one", 1),
            ("large", 1000000),
        ]
    if type_name == "float":
        return [
            ("zero_float", 0.0),
            ("small_negative", -0.001),
            ("pi", 3.14),
        ]
    return []


def _default_for_type(type_name: str) -> Any:
    if type_name == "int":
        return 1
    if type_name == "str":
        return "x"
    if type_name == "float":
        return 1.0
    return 1


# ---------------------------------------------------------------------------
# Boundary test execution
# ---------------------------------------------------------------------------

def _build_boundary_harness(
    func_name: str,
    param_names: list[str],
    boundary_calls: list[tuple[str, list[Any]]],
) -> str:
    """Build a Python harness that executes the function with boundary inputs.

    Outputs a JSON array of results:
      [{"label": ..., "args": [...], "output": repr, "exception_type": None/str}
    """
    harness_lines = [
        "import json, sys, traceback",
        f"from solution import {func_name}",
        "",
        "results = []",
    ]
    for label, args in boundary_calls:
        # Use repr to serialize args safely
        args_repr = ", ".join(repr(a) for a in args)
        harness_lines.append("try:")
        harness_lines.append(f"    _result = {func_name}({args_repr})")
        harness_lines.append(f'    results.append({{"label": {label!r}, "args": {args!r}, "output": repr(_result), "exception_type": None, "exception_msg": None}})')
        harness_lines.append("except Exception as _e:")
        harness_lines.append(f'    results.append({{"label": {label!r}, "args": {args!r}, "output": None, "exception_type": type(_e).__name__, "exception_msg": str(_e)}})')
    harness_lines.append("print(json.dumps(results, ensure_ascii=False))")
    return "\n".join(harness_lines)


def _execute_boundary_calls(
    target_code: str,
    func_name: str,
    param_names: list[str],
    boundary_calls: list[tuple[str, list[Any]]],
) -> list[dict]:
    """Execute the correct code with boundary inputs and return parsed results."""
    harness = _build_boundary_harness(func_name, param_names, boundary_calls)
    result = run_python_code(
        harness,
        timeout_s=10.0,
        extra_files={"solution.py": target_code},
    )
    if result.timed_out or result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Test generation from execution results
# ---------------------------------------------------------------------------

def _generate_boundary_tests(
    func_name: str,
    param_names: list[str],
    exec_results: list[dict],
) -> str:
    """Generate pytest test functions from boundary execution results."""
    test_lines: list[str] = [
        "import pytest",
        f"from solution import {func_name}",
        "",
    ]
    for res in exec_results:
        label = res["label"]
        args = res["args"]
        test_name = f"test_boundary_{label}"
        args_repr = ", ".join(repr(a) for a in args)
        if res["exception_type"] is None:
            # Normal result — assert equality
            output = res["output"]
            test_lines.append(f"def {test_name}():")
            test_lines.append(f"    assert {func_name}({args_repr}) == {output}")
            test_lines.append("")
        else:
            # Exception — assert it's raised
            exc_type = res["exception_type"]
            test_lines.append(f"def {test_name}():")
            test_lines.append(f"    with pytest.raises({exc_type}):")
            test_lines.append(f"        {func_name}({args_repr})")
            test_lines.append("")
    return "\n".join(test_lines)


# ---------------------------------------------------------------------------
# Boundary variant sample construction
# ---------------------------------------------------------------------------

def generate_boundary_variant(
    sample: Sample,
    *,
    max_boundary_tests: int = 8,
) -> Optional[Sample]:
    """Generate a boundary-variant sample with enhanced test cases.

    Returns None if no boundary tests could be generated.
    """
    info = _parse_function_info(sample.target_code)
    if info is None:
        return None
    func_name, param_names = info

    # Extract example calls from existing tests
    test_calls = _extract_test_calls(sample.public_tests, func_name)
    if not test_calls:
        return None

    # Use the first test call as a template for argument types
    template_args = test_calls[0]

    # Convert template args to Python values once (so we can swap in boundary
    # values without re-running _ast_to_value on already-converted values).
    try:
        base_values = [_ast_to_value(a) for a in template_args]
    except ValueError:
        return None

    # Generate boundary calls by varying each argument
    boundary_calls: list[tuple[str, list[Any]]] = []
    for i, arg in enumerate(template_args):
        if i >= len(param_names):
            break
        type_name = _classify_arg(arg)
        boundaries = _boundary_values_for_type(type_name, arg)
        for label, value in boundaries:
            # Build a call with the boundary value at position i,
            # other args from the template.
            new_args = list(base_values)
            new_args[i] = value
            boundary_calls.append((f"{param_names[i]}_{label}", new_args))
            if len(boundary_calls) >= max_boundary_tests:
                break
        if len(boundary_calls) >= max_boundary_tests:
            break

    if not boundary_calls:
        return None

    # Execute boundary calls against the correct code
    exec_results = _execute_boundary_calls(
        sample.target_code, func_name, param_names, boundary_calls
    )
    if not exec_results:
        return None

    # Generate boundary test code
    boundary_tests = _generate_boundary_tests(func_name, param_names, exec_results)

    # Combine original tests + boundary tests
    enhanced_tests = sample.public_tests.rstrip() + "\n\n" + boundary_tests

    # Create the new sample with difficulty+1 (capped at 4)
    new_difficulty = min(sample.difficulty + 1, 4)
    new_sample_id = f"{sample.sample_id}_boundary"

    created_at = datetime.now(timezone.utc).isoformat()

    return Sample(
        sample_id=new_sample_id,
        family_id=sample.family_id,
        difficulty=new_difficulty,
        task_type="code_generation",
        language="python",
        skill_tags=list(sample.skill_tags) + ["boundary"],
        instruction=sample.instruction,
        broken_code=None,
        execution_feedback=None,
        target_code=sample.target_code,
        public_tests=enhanced_tests,
        hidden_tests=sample.hidden_tests,
        verified=False,
        verification=_PLACEHOLDER_VER,
        generator=_GENERATOR,
        created_at=created_at,
        dataset_version=_DATASET_VERSION,
    )


def _ast_to_value(node: ast.expr) -> Any:
    """Convert an AST literal node to its Python value."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_ast_to_value(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_ast_to_value(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ast_to_value(node.operand)
    raise ValueError(f"Cannot convert AST node to value: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Multi-variant boundary generation (Issue #14 P6.2 / Wave 4-G1)
# ---------------------------------------------------------------------------

def generate_boundary_variants_multi(
    sample: Sample,
    *,
    max_variants: int = 2,
    max_boundary_tests_per_variant: int = 4,
) -> list[Sample]:
    """Generate up to max_variants boundary variants with disjoint edge sets.

    Each variant uses a non-overlapping subset of boundary values so the
    edge-set combinations are unique across variants (Issue #14 P6.2).

    Strategy:
      1. Collect all boundary values per argument position (existing logic).
      2. Round-robin partition them into N disjoint groups (one per variant).
      3. For each non-empty group, execute boundary calls against target_code
         and build a pytest test suite from the results.
      4. Return one Sample per non-empty group; sample_id suffix encodes the
         variant index (``_boundary_v0``, ``_boundary_v1``, ...).

    Parameters
    ----------
    sample:
        Source code_generation sample (target_code must be correct).
    max_variants:
        Maximum number of boundary variants to produce (cap at 3).
    max_boundary_tests_per_variant:
        Maximum boundary test cases per variant sample.

    Returns
    -------
    list[Sample]
        Possibly empty; one entry per non-empty disjoint edge-set.
    """
    info = _parse_function_info(sample.target_code)
    if info is None:
        return []
    func_name, param_names = info

    test_calls = _extract_test_calls(sample.public_tests, func_name)
    if not test_calls:
        return []
    template_args = test_calls[0]

    try:
        base_values = [_ast_to_value(a) for a in template_args]
    except ValueError:
        return []

    # Build all boundary calls per argument position
    all_calls_by_arg: list[list[tuple[str, list[Any]]]] = []
    for i, arg in enumerate(template_args):
        if i >= len(param_names):
            break
        type_name = _classify_arg(arg)
        boundaries = _boundary_values_for_type(type_name, arg)
        arg_calls: list[tuple[str, list[Any]]] = []
        for label, value in boundaries:
            new_args = list(base_values)
            new_args[i] = value
            arg_calls.append((f"{param_names[i]}_{label}", new_args))
        if arg_calls:
            all_calls_by_arg.append(arg_calls)

    if not all_calls_by_arg:
        return []

    # Round-robin partition into N disjoint groups so each variant gets a
    # non-overlapping subset of boundary labels per argument position.
    n_variants = max(1, min(max_variants, 3))
    variant_calls: list[list[tuple[str, list[Any]]]] = [
        [] for _ in range(n_variants)
    ]
    for arg_calls in all_calls_by_arg:
        for i, call in enumerate(arg_calls):
            target_v = i % n_variants
            variant_calls[target_v].append(call)

    # Build one Sample per non-empty variant
    variants: list[Sample] = []
    new_difficulty = min(sample.difficulty + 1, 4)
    created_at = datetime.now(timezone.utc).isoformat()

    for v_idx, calls in enumerate(variant_calls):
        if not calls:
            continue
        # Cap per variant to keep test suites tractable
        calls = calls[:max_boundary_tests_per_variant]

        exec_results = _execute_boundary_calls(
            sample.target_code, func_name, param_names, calls
        )
        if not exec_results:
            continue

        boundary_tests = _generate_boundary_tests(
            func_name, param_names, exec_results
        )
        enhanced_tests = sample.public_tests.rstrip() + "\n\n" + boundary_tests

        new_sample_id = f"{sample.sample_id}_boundary_v{v_idx}"

        variants.append(Sample(
            sample_id=new_sample_id,
            family_id=sample.family_id,
            difficulty=new_difficulty,
            task_type="code_generation",
            language="python",
            skill_tags=list(sample.skill_tags) + ["boundary"],
            instruction=sample.instruction,
            broken_code=None,
            execution_feedback=None,
            target_code=sample.target_code,
            public_tests=enhanced_tests,
            hidden_tests=sample.hidden_tests,
            verified=False,
            verification=_PLACEHOLDER_VER,
            generator=_GENERATOR,
            created_at=created_at,
            dataset_version=_DATASET_VERSION,
        ))

    return variants


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate boundary-condition variant samples."
    )
    p.add_argument("--input", required=True, help="Input JSONL of code_generation samples.")
    p.add_argument("--output", required=True, help="Output JSONL of boundary variants.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (kept for API stability).")
    return p


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_variants = 0
    n_samples = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out_fh:
        with in_path.open(encoding="utf-8") as in_fh:
            for line in in_fh:
                line = line.strip()
                if not line:
                    continue
                n_samples += 1
                data = json.loads(line)
                try:
                    sample = Sample(**data)
                except Exception as exc:
                    print(f"  SKIP invalid sample: {exc}", file=sys.stderr)
                    n_skipped += 1
                    continue

                variant = generate_boundary_variant(sample)
                if variant is None:
                    n_skipped += 1
                    continue

                out_fh.write(variant.to_json_line() + "\n")
                n_variants += 1

    print(
        f"generate_boundary_variants: {n_samples} samples -> {n_variants} variants "
        f"({n_skipped} skipped) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
