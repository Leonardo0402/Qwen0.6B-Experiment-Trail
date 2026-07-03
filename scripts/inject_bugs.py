"""
scripts/inject_bugs.py -- Deterministic bug injection (P2.2 §7.1–7.8).

Implements 8 distinct bug-injection operators using AST transforms.
Each operator changes exactly one site (first applicable occurrence) so
the resulting bugged code is deterministic for a given input.

Bug types
---------
  7.1 condition_error      : < -> <=, > -> >=, == -> !=
  7.2 off_by_one           : range(n) -> range(n-1) or range(n+1)
  7.3 return_value_error   : return X -> return None / wrong variable
  7.4 index_error          : items[0] -> items[1], items[-1] -> items[0]
  7.5 initialization_error : total = 0 -> total = 1, result = [] -> result = None
  7.6 aggregation_error    : min -> max, sum -> len, any -> all
  7.7 branch_deletion      : drop or invert first if-branch
  7.8 type_error           : [] -> None, 0 -> "", literal type mismatch

Functions
---------
  inject_bug_xxx(code, seed=42) -> (bugged_code, description) | None
  inject_all_bugs(code, seed=42) -> [(bug_type, bugged_code, description), ...]

Usage
-----
    python scripts/inject_bugs.py --input <sample.jsonl> --output <output.jsonl> --seed 42
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Core AST transform helper
# ---------------------------------------------------------------------------

def _try_transform(code: str, transformer: ast.NodeTransformer) -> Optional[str]:
    """Parse *code*, apply *transformer*, unparse. None on failure or no change."""
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


# ---------------------------------------------------------------------------
# 7.1 condition_error : < -> <=, > -> >=, == -> !=
# ---------------------------------------------------------------------------

_COND_MAP_FIRST: dict[type, type] = {
    ast.Lt: ast.LtE,
    ast.Gt: ast.GtE,
    ast.Eq: ast.NotEq,
}


class _ConditionErrorTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if self.changed:
            return node
        new_ops = []
        for op in node.ops:
            flip = _COND_MAP_FIRST.get(type(op))
            if flip and not self.changed:
                new_ops.append(flip())
                self.changed = True
            else:
                new_ops.append(op)
        if not self.changed:
            self.generic_visit(node)
            return node
        return ast.Compare(left=node.left, ops=new_ops, comparators=node.comparators)


def inject_bug_condition_error(code: str, seed: int = 42) -> tuple[str, str] | None:
    """7.1: Flip first comparison operator (< -> <=, > -> >=, == -> !=)."""
    result = _try_transform(code, _ConditionErrorTransformer())
    if result is None:
        return None
    return result, "条件错误: 比较运算符被修改 (< -> <=, > -> >=, == -> !=)"


# ---------------------------------------------------------------------------
# 7.2 off_by_one : range(n) -> range(n-1) or range(n+1)
# ---------------------------------------------------------------------------

def _make_off_by_one_transformer(delta: int) -> ast.NodeTransformer:
    class _T(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if self.changed:
                return node
            if (isinstance(node.func, ast.Name) and node.func.id == "range" and node.args):
                self.changed = True
                args = list(node.args)
                last = args[-1]
                op = ast.Add() if delta > 0 else ast.Sub()
                args[-1] = ast.BinOp(left=last, op=op, right=ast.Constant(value=abs(delta)))
                return ast.Call(func=node.func, args=args, keywords=node.keywords)
            return node
    return _T()


def inject_bug_off_by_one(code: str, seed: int = 42) -> tuple[str, str] | None:
    """7.2: Adjust first range() stop argument by -1 (fallback +1 if -1 yields identical)."""
    t = _make_off_by_one_transformer(-1)
    result = _try_transform(code, t)
    if result is not None:
        return result, "Off-by-one: range() 上界减 1 (range(n) -> range(n-1))"
    t2 = _make_off_by_one_transformer(1)
    result2 = _try_transform(code, t2)
    if result2 is not None:
        return result2, "Off-by-one: range() 上界加 1 (range(n) -> range(n+1))"
    return None


# ---------------------------------------------------------------------------
# 7.3 return_value_error : return X -> return None / wrong variable
# ---------------------------------------------------------------------------

class _ReturnNoneTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if self.changed:
            return node
        if node.value is not None and not isinstance(node.value, ast.Constant):
            self.changed = True
            return ast.Return(value=ast.Constant(value=None))
        return node


class _ReturnWrongVarTransformer(ast.NodeTransformer):
    """Replace return X with return Y where Y is a different in-scope name."""
    def __init__(self) -> None:
        self.changed = False
        self._scope_names: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Collect candidate names from args and body assignments
        candidates: list[str] = []
        for arg in node.args.args:
            candidates.append(arg.arg)
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        candidates.append(tgt.id)
        self._scope_names = candidates
        self.generic_visit(node)
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if self.changed:
            return node
        if node.value is not None and isinstance(node.value, ast.Name):
            current = node.value.id
            for alt in self._scope_names:
                if alt != current:
                    self.changed = True
                    return ast.Return(value=ast.Name(id=alt, ctx=ast.Load()))
        return node


def inject_bug_return_value_error(code: str, seed: int = 42) -> tuple[str, str] | None:
    """7.3: Change first return to return None (fallback: wrong variable)."""
    result = _try_transform(code, _ReturnNoneTransformer())
    if result is not None:
        return result, "返回值错误: return 语句改为 return None"
    result2 = _try_transform(code, _ReturnWrongVarTransformer())
    if result2 is not None:
        return result2, "返回值错误: return 语句返回了错误的变量"
    return None


# ---------------------------------------------------------------------------
# 7.4 index_error : items[0] -> items[1], items[-1] -> items[0]
# ---------------------------------------------------------------------------

class _IndexErrorTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
            self.changed = True
            new_val = sl.value + 1 if sl.value >= 0 else 0
            return ast.Subscript(
                value=node.value,
                slice=ast.Constant(value=new_val),
                ctx=node.ctx,
            )
        return node


def inject_bug_index_error(code: str, seed: int = 42) -> tuple[str, str] | None:
    """7.4: Shift first integer subscript index (items[0] -> items[1], items[-1] -> items[0])."""
    result = _try_transform(code, _IndexErrorTransformer())
    if result is None:
        return None
    return result, "索引错误: 列表下标被修改 (items[0] -> items[1] 或 items[-1] -> items[0])"


# ---------------------------------------------------------------------------
# 7.5 initialization_error : total = 0 -> total = 1, result = [] -> result = None
# ---------------------------------------------------------------------------

class _InitializationErrorTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if self.changed:
            return node
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.value is not None:
            val = node.value
            if isinstance(val, ast.Constant):
                if val.value == 0:
                    self.changed = True
                    return ast.Assign(
                        targets=node.targets, value=ast.Constant(value=1)
                    )
                if isinstance(val.value, str):
                    self.changed = True
                    return ast.Assign(
                        targets=node.targets, value=ast.Constant(value=None)
                    )
            if isinstance(val, ast.List) and len(val.elts) == 0:
                self.changed = True
                return ast.Assign(
                    targets=node.targets, value=ast.Constant(value=None)
                )
        return node


def inject_bug_initialization_error(code: str, seed: int = 42) -> tuple[str, str] | None:
    """7.5: Corrupt first initialization (total=0 -> total=1, result=[] -> result=None)."""
    result = _try_transform(code, _InitializationErrorTransformer())
    if result is None:
        return None
    return result, "初始化错误: 变量初始值被破坏 (total=0 -> total=1, result=[] -> result=None)"


# ---------------------------------------------------------------------------
# 7.6 aggregation_error : min -> max, sum -> len, any -> all
# ---------------------------------------------------------------------------

_AGG_MAP: dict[str, str] = {
    "min": "max",
    "max": "min",
    "sum": "len",
    "any": "all",
    "all": "any",
}


class _AggregationErrorTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        if isinstance(node.func, ast.Name) and node.func.id in _AGG_MAP:
            old = node.func.id
            new = _AGG_MAP[old]
            self.changed = True
            return ast.Call(
                func=ast.Name(id=new, ctx=ast.Load()),
                args=node.args,
                keywords=node.keywords,
            )
        return node


def inject_bug_aggregation_error(code: str, seed: int = 42) -> tuple[str, str] | None:
    """7.6: Replace first aggregation builtin (min -> max, sum -> len, any -> all)."""
    result = _try_transform(code, _AggregationErrorTransformer())
    if result is None:
        return None
    return result, "聚合函数错误: 内置聚合函数被替换 (min -> max, sum -> len, any -> all)"


# ---------------------------------------------------------------------------
# 7.7 branch_deletion : delete or invert first if-branch
# ---------------------------------------------------------------------------

class _BranchDeletionTransformer(ast.NodeTransformer):
    """Remove the first if-statement body (replace with pass)."""
    def __init__(self) -> None:
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if self.changed:
            return node
        new_body: list[ast.stmt] = []
        for stmt in node.body:
            if not self.changed and isinstance(stmt, ast.If):
                self.changed = True
                # Replace the if-body with pass (effectively deleting the branch)
                new_body.append(ast.Pass())
            else:
                new_body.append(stmt)
        if self.changed:
            node.body = new_body or [ast.Pass()]
        return node


class _BranchInvertTransformer(ast.NodeTransformer):
    """Invert the condition of the first if-statement."""
    def __init__(self) -> None:
        self.changed = False

    def visit_If(self, node: ast.If) -> ast.AST:
        if self.changed:
            return node
        self.changed = True
        return ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=node.test),
            body=node.body,
            orelse=node.orelse,
        )


def inject_bug_branch_deletion(code: str, seed: int = 42) -> tuple[str, str] | None:
    """7.7: Delete first if-branch (fallback: invert condition)."""
    result = _try_transform(code, _BranchDeletionTransformer())
    if result is not None:
        return result, "分支删除: 关键 if 分支被删除"
    result2 = _try_transform(code, _BranchInvertTransformer())
    if result2 is not None:
        return result2, "分支反转: 关键 if 条件被取反"
    return None


# ---------------------------------------------------------------------------
# 7.8 type_error : [] -> None, 0 -> "", literal type mismatch
# ---------------------------------------------------------------------------

class _TypeErrorTransformer(ast.NodeTransformer):
    """Replace first list literal [] with None or first numeric 0 with empty string."""
    def __init__(self) -> None:
        self.changed = False

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if self.changed:
            return node
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            val = node.value
            if isinstance(val, ast.List) and len(val.elts) == 0:
                self.changed = True
                return ast.Assign(
                    targets=node.targets, value=ast.Constant(value=None)
                )
            if isinstance(val, ast.Constant) and val.value == 0:
                self.changed = True
                return ast.Assign(
                    targets=node.targets, value=ast.Constant(value="")
                )
        return node


def inject_bug_type_error(code: str, seed: int = 42) -> tuple[str, str] | None:
    """7.8: Introduce type mismatch ([] -> None, 0 -> '')."""
    result = _try_transform(code, _TypeErrorTransformer())
    if result is None:
        return None
    return result, "类型错误: 初始值类型不匹配 (result=[] -> result=None, total=0 -> total=空字符串)"


# ---------------------------------------------------------------------------
# Registry + inject_all_bugs
# ---------------------------------------------------------------------------

BUG_TYPES: list[str] = [
    "condition_error",
    "off_by_one",
    "return_value_error",
    "index_error",
    "initialization_error",
    "aggregation_error",
    "branch_deletion",
    "type_error",
]

_INJECTORS: dict[str, callable] = {
    "condition_error": inject_bug_condition_error,
    "off_by_one": inject_bug_off_by_one,
    "return_value_error": inject_bug_return_value_error,
    "index_error": inject_bug_index_error,
    "initialization_error": inject_bug_initialization_error,
    "aggregation_error": inject_bug_aggregation_error,
    "branch_deletion": inject_bug_branch_deletion,
    "type_error": inject_bug_type_error,
}


def inject_all_bugs(code: str, seed: int = 42) -> list[tuple[str, str, str]]:
    """Return [(bug_type, bugged_code, description), ...] for all applicable injectors.

    Operators that produce no change or invalid syntax are excluded.
    The returned list may be empty if no operator is applicable.
    """
    results: list[tuple[str, str, str]] = []
    for bug_type in BUG_TYPES:
        injector = _INJECTORS[bug_type]
        outcome = injector(code, seed=seed)
        if outcome is not None:
            bugged_code, description = outcome
            if bugged_code.strip() != code.strip():
                results.append((bug_type, bugged_code, description))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inject deterministic bugs into code_generation samples."
    )
    p.add_argument("--input", required=True, help="Input JSONL of code_generation samples.")
    p.add_argument("--output", required=True, help="Output JSONL of bug variants.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (unused but kept for API stability).")
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
                sample_id = data.get("sample_id", "unknown")
                family_id = data.get("family_id", "unknown")
                target_code = data.get("target_code", "")

                variants = inject_all_bugs(target_code, seed=args.seed)
                if not variants:
                    n_skipped += 1
                    continue

                for bug_type, bugged_code, description in variants:
                    record = {
                        "sample_id": f"{sample_id}_bug_{bug_type}",
                        "original_sample_id": sample_id,
                        "family_id": family_id,
                        "bug_type": bug_type,
                        "bugged_code": bugged_code,
                        "description": description,
                        "target_code": target_code,
                        "public_tests": data.get("public_tests", ""),
                        "hidden_tests": data.get("hidden_tests", ""),
                        "instruction": data.get("instruction", ""),
                        "difficulty": data.get("difficulty", 1),
                        "skill_tags": data.get("skill_tags", []),
                    }
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_variants += 1

    print(
        f"inject_bugs: {n_samples} samples -> {n_variants} variants "
        f"({n_skipped} samples had no applicable bugs) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
