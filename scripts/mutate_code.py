"""
scripts/mutate_code.py -- AST-based bug mutator (spec §11.2).

Reads verified code_generation samples, applies mutation operators to their
reference implementations, and produces ``static_repair`` and
``execution_repair`` samples for every variant that genuinely fails >=1 test.

Mutation operators (11 total) -- covering all spec §11.2 categories
-------------------------------------------------------------------
  typo_identifier    -- rename first Load-context identifier to a misspelling
                        (compiles, then NameError at runtime -> test failure)
  flip_comparison    -- < ↔ >, > ↔ <, == ↔ !=, in ↔ not in  (first occurrence)
  wrong_arithmetic   -- first + → -, - → +, * → + in BinOp / AugAssign
  off_by_one_minus1  -- first range(...) stop arg -1
  off_by_one_plus1   -- first range(...) stop arg +1
  wrong_sort_dir     -- add/flip reverse=True in first sorted()/list.sort()
  wrong_index_plus1  -- first integer literal subscript index +1
  flip_bool_return   -- first True/False literal return → opposite
  drop_guard         -- delete first leading guard `if ...: raise/return` in a
                        function body (empty-input / boundary guard removal)
  remove_first_extend -- delete first .extend() call in any function
  flip_slice_step    -- change first [::-1] step to [::1]

Spec §11.2 category coverage: typo (typo_identifier), off-by-one
(off_by_one_*), wrong comparison (flip_comparison), drop empty-input/boundary
guard (drop_guard), wrong sort direction (wrong_sort_dir), wrong return
structure (flip_bool_return / remove_first_extend / flip_slice_step), wrong
index (wrong_index_plus1).

Each operator uses ast.NodeTransformer + ast.unparse (Python >= 3.9).
Only mutations that (a) compile and (b) fail >= 1 test are kept.

Usage
-----
    python scripts/mutate_code.py [--in PATH] [--out PATH]
                                  [--seed S] [--max-per-sample N]

Exit codes
----------
    0   success
    1   error
"""

from __future__ import annotations

import argparse
import ast
import copy
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._io import load_samples_file  # noqa: E402
from src.sandbox import run_pytest  # noqa: E402
from src.schemas import Sample, Verification  # noqa: E402
from src.validators import compile_check  # noqa: E402


def per_sample_seed(base_seed: int, sample_id: str) -> int:
    """Derive a stable, per-sample RNG seed from a base seed and sample_id.

    Uses zlib.crc32 (a STABLE hash) rather than builtin hash() -- which is
    salted per-process via PYTHONHASHSEED and would make runs non-reproducible.
    Mixing the id into the seed means each source sample gets a different
    operator shuffle, so the same first-N operators do not dominate the dataset.
    """
    return base_seed ^ (zlib.crc32(sample_id.encode("utf-8")) & 0xFFFF)

# ---------------------------------------------------------------------------
# Placeholder verification (for newly created repair samples)
# ---------------------------------------------------------------------------

_PLACEHOLDER_VER = Verification(
    syntax_ok=False, pytest_ok=False, ruff_ok=False, timeout=False
)
_DATASET_VERSION = "v1.0"
_GENERATOR = "mutator_v1"


# ---------------------------------------------------------------------------
# AST mutation operators
# ---------------------------------------------------------------------------

def _try_transform(code: str, transformer: ast.NodeTransformer) -> Optional[str]:
    """Parse *code*, apply *transformer*, unparse.  Returns None on failure or no change."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    new_tree = transformer.visit(copy.deepcopy(tree))
    if not getattr(transformer, "changed", False):
        return None
    ast.fix_missing_locations(new_tree)
    try:
        result = ast.unparse(new_tree)
    except Exception:
        return None
    return result if result.strip() != code.strip() else None


# --- Operator: typo_identifier (spec §11.2 "typo / rename a var") -------------


class _TypoRenameIdentifier(ast.NodeTransformer):
    """Rename the FIRST Load-context identifier to a misspelling.

    The renamed name gets a ``_typo`` suffix that is never defined anywhere, so
    the code still COMPILES but raises ``NameError`` the moment that expression
    executes.  pytest reports the NameError as a collection/runtime failure
    (num_failed >= 1), which the keep-only-failing filter retains.

    Only Load contexts are renamed (uses, not bindings) and function parameters
    in the signature are ``ast.arg`` nodes -- not ``ast.Name`` -- so they are
    left untouched, guaranteeing the def itself still imports cleanly.
    """

    def __init__(self) -> None:
        self.changed = False

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.changed:
            return node
        if isinstance(node.ctx, ast.Load):
            self.changed = True
            return ast.Name(id=node.id + "_typo", ctx=ast.Load())
        return node


# --- Operator: flip_comparison -----------------------------------------------

# Aggressive flip: wrong direction / wrong equality — more likely to fail tests.
_FLIP_MAP: dict[type, type] = {
    ast.Lt: ast.Gt,
    ast.Gt: ast.Lt,
    ast.LtE: ast.Gt,
    ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


class _FlipFirstComparison(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if self.changed:
            return node
        new_ops = []
        for op in node.ops:
            flip = _FLIP_MAP.get(type(op))
            if flip and not self.changed:
                new_ops.append(flip())
                self.changed = True
            else:
                new_ops.append(op)
        if not self.changed:
            # No flippable op found in this Compare; keep traversing
            self.generic_visit(node)
            return node
        return ast.Compare(
            left=node.left, ops=new_ops, comparators=node.comparators
        )


# --- Operator: wrong_arithmetic ----------------------------------------------


class _WrongArithmeticOp(ast.NodeTransformer):
    """Flip first + ↔ - or * → + in BinOp or AugAssign."""

    def __init__(self) -> None:
        self.changed = False

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        if isinstance(node.op, ast.Add):
            self.changed = True
            return ast.BinOp(left=node.left, op=ast.Sub(), right=node.right)
        if isinstance(node.op, ast.Sub):
            self.changed = True
            return ast.BinOp(left=node.left, op=ast.Add(), right=node.right)
        if isinstance(node.op, ast.Mult):
            self.changed = True
            return ast.BinOp(left=node.left, op=ast.Add(), right=node.right)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        if isinstance(node.op, ast.Add):
            self.changed = True
            return ast.AugAssign(target=node.target, op=ast.Sub(), value=node.value)
        if isinstance(node.op, ast.Sub):
            self.changed = True
            return ast.AugAssign(target=node.target, op=ast.Add(), value=node.value)
        if isinstance(node.op, ast.Mult):
            self.changed = True
            return ast.AugAssign(target=node.target, op=ast.Add(), value=node.value)
        return node


# --- Operator: off_by_one_minus1 / off_by_one_plus1 -------------------------


def _make_range_transformer(delta: int) -> type:
    """Return a NodeTransformer class that adjusts range() last arg by *delta*."""

    class _OffByOneRange(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if self.changed:
                return node
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "range"
                and node.args
            ):
                self.changed = True
                args = list(node.args)
                last = args[-1]
                op = ast.Add() if delta > 0 else ast.Sub()
                args[-1] = ast.BinOp(
                    left=last, op=op, right=ast.Constant(value=abs(delta))
                )
                return ast.Call(func=node.func, args=args, keywords=node.keywords)
            return node

    return _OffByOneRange


_OffByOneMinus1 = _make_range_transformer(-1)
_OffByOnePlus1 = _make_range_transformer(+1)


# --- Operator: wrong_sort_direction ------------------------------------------


class _WrongSortDirection(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        is_sort_call = (isinstance(node.func, ast.Name) and node.func.id == "sorted") or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("sort", "sorted")
        )
        if not is_sort_call:
            return node
        self.changed = True
        new_kw: list[ast.keyword] = []
        has_reverse = False
        for kw in node.keywords:
            if kw.arg == "reverse":
                has_reverse = True
                if isinstance(kw.value, ast.Constant):
                    new_kw.append(
                        ast.keyword(
                            arg="reverse",
                            value=ast.Constant(value=not kw.value.value),
                        )
                    )
                else:
                    new_kw.append(kw)
            else:
                new_kw.append(kw)
        if not has_reverse:
            new_kw.append(
                ast.keyword(arg="reverse", value=ast.Constant(value=True))
            )
        return ast.Call(func=node.func, args=node.args, keywords=new_kw)


# --- Operator: wrong_index_plus1 ---------------------------------------------


class _WrongIndexPlus1(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            self.changed = True
            return ast.Subscript(
                value=node.value,
                slice=ast.Constant(value=node.slice.value + 1),
                ctx=node.ctx,
            )
        return node


# --- Operator: flip_bool_return ----------------------------------------------


class _FlipBoolReturn(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if self.changed:
            return node
        if node.value is not None and isinstance(node.value, ast.Constant):
            if node.value.value is True:
                self.changed = True
                return ast.Return(value=ast.Constant(value=False))
            if node.value.value is False:
                self.changed = True
                return ast.Return(value=ast.Constant(value=True))
        return node


# --- Operator: remove_first_extend -------------------------------------------


def _remove_first_extend(code: str) -> Optional[str]:
    """Delete the first result.extend() (or any .extend()) statement."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    removed: list[bool] = [False]

    class _Remover(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            if removed[0]:
                return node
            new_body = []
            for stmt in node.body:
                if (
                    not removed[0]
                    and isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr == "extend"
                ):
                    removed[0] = True
                    continue  # drop this statement
                new_body.append(stmt)
            node.body = new_body if removed[0] else node.body
            return node

    new_tree = _Remover().visit(copy.deepcopy(tree))
    if not removed[0]:
        return None
    ast.fix_missing_locations(new_tree)
    try:
        result = ast.unparse(new_tree)
    except Exception:
        return None
    return result if result.strip() != code.strip() else None


# --- Operator: drop_guard (spec §11.2 "drop empty-input guard") --------------


def _drop_first_guard(code: str) -> Optional[str]:
    """Delete the first guard statement found in a function body.

    A guard is a top-level ``if`` statement (no ``else``) whose body is a single
    ``raise`` or ``return`` -- e.g. ``if n < 0: raise ValueError(...)``,
    ``if len(unique) < 2: raise ...``, ``if x < lo: return lo``.  The guard need
    not be the very first statement (e.g. second_largest's guard is the 2nd
    statement); the first one encountered while scanning the body in order is
    removed.  Removing it means empty / boundary / negative inputs are no longer
    rejected or handled, so the boundary test cases fail.  Only the first such
    guard in the first function is removed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    removed: list[bool] = [False]

    class _Dropper(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            if removed[0]:
                return node
            new_body: list[ast.stmt] = []
            for stmt in node.body:
                if (
                    not removed[0]
                    and isinstance(stmt, ast.If)
                    and not stmt.orelse
                    and len(stmt.body) == 1
                    and isinstance(stmt.body[0], (ast.Raise, ast.Return))
                ):
                    removed[0] = True
                    continue  # drop the guard
                new_body.append(stmt)
            if removed[0]:
                # A function body must not be empty; pad with `pass` if needed.
                node.body = new_body or [ast.Pass()]
            return node

    new_tree = _Dropper().visit(copy.deepcopy(tree))
    if not removed[0]:
        return None
    ast.fix_missing_locations(new_tree)
    try:
        result = ast.unparse(new_tree)
    except Exception:
        return None
    return result if result.strip() != code.strip() else None


# --- Operator: flip_slice_step -----------------------------------------------


class _FlipSliceStep(ast.NodeTransformer):
    """Change first [::-1] to [::1] (no-op step removes reversal)."""

    def __init__(self) -> None:
        self.changed = False

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        if isinstance(node.slice, ast.Slice) and node.slice.step is not None:
            step = node.slice.step
            is_neg_one = (
                isinstance(step, ast.UnaryOp)
                and isinstance(step.op, ast.USub)
                and isinstance(step.operand, ast.Constant)
                and step.operand.value == 1
            ) or (isinstance(step, ast.Constant) and step.value == -1)
            if is_neg_one:
                self.changed = True
                new_slice = ast.Slice(
                    lower=node.slice.lower,
                    upper=node.slice.upper,
                    step=ast.Constant(value=1),
                )
                return ast.Subscript(
                    value=node.value, slice=new_slice, ctx=node.ctx
                )
        return node


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

def _op_typo_identifier(code: str) -> Optional[str]:
    return _try_transform(code, _TypoRenameIdentifier())


def _op_drop_guard(code: str) -> Optional[str]:
    return _drop_first_guard(code)


def _op_flip_comparison(code: str) -> Optional[str]:
    return _try_transform(code, _FlipFirstComparison())


def _op_wrong_arithmetic(code: str) -> Optional[str]:
    return _try_transform(code, _WrongArithmeticOp())


def _op_off_by_one_minus1(code: str) -> Optional[str]:
    return _try_transform(code, _OffByOneMinus1())


def _op_off_by_one_plus1(code: str) -> Optional[str]:
    return _try_transform(code, _OffByOnePlus1())


def _op_wrong_sort_dir(code: str) -> Optional[str]:
    return _try_transform(code, _WrongSortDirection())


def _op_wrong_index_plus1(code: str) -> Optional[str]:
    return _try_transform(code, _WrongIndexPlus1())


def _op_flip_bool_return(code: str) -> Optional[str]:
    return _try_transform(code, _FlipBoolReturn())


def _op_remove_first_extend(code: str) -> Optional[str]:
    return _remove_first_extend(code)


def _op_flip_slice_step(code: str) -> Optional[str]:
    return _try_transform(code, _FlipSliceStep())


MUTATORS: dict[str, Callable[[str], Optional[str]]] = {
    "typo_identifier": _op_typo_identifier,
    "flip_comparison": _op_flip_comparison,
    "wrong_arithmetic": _op_wrong_arithmetic,
    "off_by_one_minus1": _op_off_by_one_minus1,
    "off_by_one_plus1": _op_off_by_one_plus1,
    "wrong_sort_dir": _op_wrong_sort_dir,
    "wrong_index_plus1": _op_wrong_index_plus1,
    "flip_bool_return": _op_flip_bool_return,
    "drop_guard": _op_drop_guard,
    "remove_first_extend": _op_remove_first_extend,
    "flip_slice_step": _op_flip_slice_step,
}


def apply_all_mutators(code: str) -> dict[str, str]:
    """Apply all operators; return {op_name: mutated_code} for applicable ones.

    Operators that produce no change or invalid syntax are excluded.
    The returned dict may be empty if no operator is applicable.
    """
    result: dict[str, str] = {}
    for name, op in MUTATORS.items():
        mutated = op(code)
        if mutated is not None and mutated.strip() != code.strip():
            result[name] = mutated
    return result


# ---------------------------------------------------------------------------
# Verify a mutation and capture execution feedback
# ---------------------------------------------------------------------------


# Upper bound on the assembled execution_feedback string.  This is enforced
# here at the assembly site (rather than relying on the sandbox's per-stream
# cap) so the bound on what lands in a sample is self-documenting and stable
# even if stdout + stderr are concatenated.
MAX_FEEDBACK_CHARS = 4000
_FEEDBACK_TRUNCATION_MARKER = "...[truncated]"


def _format_feedback(stdout: str, stderr: str, returncode: Optional[int]) -> str:
    """Format pytest output as execution_feedback text (capped at MAX_FEEDBACK_CHARS)."""
    parts: list[str] = []
    if stdout.strip():
        parts.append(stdout.strip())
    if stderr.strip():
        parts.append(stderr.strip())
    if not parts:
        parts.append(f"[pytest exited with code {returncode}]")
    feedback = "\n".join(parts)
    if len(feedback) > MAX_FEEDBACK_CHARS:
        feedback = feedback[:MAX_FEEDBACK_CHARS] + _FEEDBACK_TRUNCATION_MARKER
    return feedback


def mutate_and_get_feedback(
    broken_code: str,
    sample: Sample,
    *,
    pytest_timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """Run broken_code against sample's tests.

    Returns
    -------
    (is_broken, execution_feedback)
        is_broken is True when >= 1 test fails.
        execution_feedback is the captured pytest output (empty string if not broken).
    """
    # We call run_pytest directly rather than validators.verify_broken_is_broken
    # because we need the captured pytest stdout/stderr to populate
    # execution_feedback for the execution_repair sample; the bool-returning
    # helper only reports pass/fail and does not expose that output.
    # Public tests first
    pub = run_pytest(broken_code, sample.public_tests, timeout_s=pytest_timeout_s)
    if pub.timed_out or pub.num_failed >= 1:
        return True, _format_feedback(pub.stdout, pub.stderr, pub.returncode)

    # Hidden tests
    hidden = (sample.hidden_tests or "").strip()
    if hidden:
        hid = run_pytest(broken_code, hidden, timeout_s=pytest_timeout_s)
        if hid.timed_out or hid.num_failed >= 1:
            return True, _format_feedback(hid.stdout, hid.stderr, hid.returncode)

    return False, ""


# ---------------------------------------------------------------------------
# Core: generate repair samples from one reference sample
# ---------------------------------------------------------------------------


def generate_repair_samples(
    sample: Sample,
    *,
    max_per_sample: int = 3,
    pytest_timeout_s: float = 10.0,
    seed: int = 42,
) -> list[tuple[Optional[Sample], Optional[Sample]]]:
    """Produce (static_repair, execution_repair) pairs from *sample*.

    For each mutation operator that produces a genuinely broken variant:
    - A ``static_repair`` Sample is created (broken_code, no feedback).
    - An ``execution_repair`` Sample is created (broken_code + feedback).

    Mutations that (a) fail to compile, or (b) pass all tests, are silently
    discarded.  At most ``max_per_sample`` pairs are returned.

    Parameters
    ----------
    sample:
        A verified code_generation Sample.  Its target_code is mutated.
    max_per_sample:
        Maximum number of (static, exec) pairs to return.
    pytest_timeout_s:
        Per-pytest timeout.
    seed:
        RNG seed for deterministic operator ordering.

    Returns
    -------
    list of (Optional[Sample], Optional[Sample])
        Each tuple is (static_repair, execution_repair) or (None, None) when
        a slot cannot be created (e.g., empty execution_feedback).
        The outer list has at most max_per_sample entries.
    """
    import random

    rng = random.Random(seed)
    candidates = apply_all_mutators(sample.target_code)
    op_names = list(candidates.keys())
    rng.shuffle(op_names)

    created_at = datetime.now(timezone.utc).isoformat()
    results: list[tuple[Optional[Sample], Optional[Sample]]] = []

    for op_name in op_names:
        if len(results) >= max_per_sample:
            break

        broken_code = candidates[op_name]

        # Compile check
        ok, _ = compile_check(broken_code)
        if not ok:
            continue

        # Run tests and check
        is_broken, feedback = mutate_and_get_feedback(
            broken_code, sample, pytest_timeout_s=pytest_timeout_s
        )
        if not is_broken:
            continue  # mutation passes all tests — discard

        sr_id = f"{sample.sample_id}_sr_{op_name}"
        er_id = f"{sample.sample_id}_er_{op_name}"

        repair_instruction = (
            f"{sample.instruction}\n\n"
            "以下代码存在错误，请找出并修复，使其能通过所有测试用例。"
        )
        exec_instruction = (
            f"{sample.instruction}\n\n"
            "以下代码存在错误，执行后出现以下问题，请修复代码。"
        )

        # static_repair: same difficulty as original
        static_repair = Sample(
            sample_id=sr_id,
            family_id=sample.family_id,
            difficulty=sample.difficulty,
            task_type="static_repair",
            language="python",
            skill_tags=list(sample.skill_tags),
            instruction=repair_instruction,
            broken_code=broken_code,
            execution_feedback=None,
            target_code=sample.target_code,
            public_tests=sample.public_tests,
            hidden_tests=sample.hidden_tests,
            verified=False,
            verification=_PLACEHOLDER_VER,
            generator=_GENERATOR,
            created_at=created_at,
            dataset_version=_DATASET_VERSION,
        )

        # execution_repair: difficulty 3 (L3 = Execution-feedback repair)
        exec_repair: Optional[Sample] = None
        if feedback.strip():
            exec_repair = Sample(
                sample_id=er_id,
                family_id=sample.family_id,
                difficulty=3,
                task_type="execution_repair",
                language="python",
                skill_tags=list(sample.skill_tags),
                instruction=exec_instruction,
                broken_code=broken_code,
                execution_feedback=feedback,
                target_code=sample.target_code,
                public_tests=sample.public_tests,
                hidden_tests=sample.hidden_tests,
                verified=False,
                verification=_PLACEHOLDER_VER,
                generator=_GENERATOR,
                created_at=created_at,
                dataset_version=_DATASET_VERSION,
            )

        results.append((static_repair, exec_repair))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Produce static_repair and execution_repair samples via AST mutation.",
    )
    p.add_argument("--in", dest="in_path",
                   default="data/generated/reference.jsonl",
                   help="Input JSONL of verified code_generation samples.")
    p.add_argument("--out",
                   default="data/generated/mutations.jsonl",
                   help="Output JSONL of repair samples.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed.")
    p.add_argument("--max-per-sample", type=int, default=3,
                   help="Max (static, exec) pairs per source sample.")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="Per-pytest timeout in seconds.")
    return p


def main() -> int:
    """CLI entry point."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    samples = load_samples_file(in_path)

    print(f"mutate_code: {len(samples)} input samples from {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_static = 0
    n_exec = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for sample in samples:
            # Per-sample seed so each source's operator shuffle differs and the
            # same first-N operators do not dominate the output dataset.
            pairs = generate_repair_samples(
                sample,
                max_per_sample=args.max_per_sample,
                pytest_timeout_s=args.timeout,
                seed=per_sample_seed(args.seed, sample.sample_id),
            )
            if not pairs:
                n_skipped += 1
                print(f"  SKIP {sample.sample_id}: no valid mutations")
                continue
            for sr, er in pairs:
                if sr is not None:
                    fh.write(sr.to_json_line() + "\n")
                    n_static += 1
                if er is not None:
                    fh.write(er.to_json_line() + "\n")
                    n_exec += 1

    total = n_static + n_exec
    print(
        f"mutate_code: wrote {total} repair samples "
        f"({n_static} static_repair, {n_exec} execution_repair) -> {out_path}"
    )
    if n_skipped:
        print(f"  {n_skipped} source samples had no valid mutations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
