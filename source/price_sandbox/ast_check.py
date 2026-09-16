"""AST allowlist for LLM-written site price functions.

The generated module may import ``math`` and define one ``quote`` function.
Anything else (other imports, dunders, ``eval``, ``open``, nested defs) is
rejected before ``compile``.
"""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Callable
from typing import Any

ALLOWED_CALL_NAMES = frozenset(
    {
        "abs",
        "bool",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "round",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "next",
        "ValueError",
    }
)

SAFE_METHODS = frozenset(
    {
        "append",
        "count",
        "endswith",
        "get",
        "index",
        "items",
        "join",
        "keys",
        "replace",
        "split",
        "startswith",
        "strip",
        "values",
    }
)

SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "next": next,
    "ValueError": ValueError,
    "True": True,
    "False": False,
    "None": None,
}

_ALLOWED_NODE_TYPES = frozenset(
    {
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.If,
        ast.For,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.Expr,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.IfExp,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Subscript,
        ast.Slice,
        ast.Starred,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.Call,
        ast.keyword,
        ast.Attribute,
        ast.JoinedStr,
        ast.FormattedValue,
        ast.Raise,
        ast.Import,
        ast.ImportFrom,
        ast.alias,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.BitOr,
        ast.BitAnd,
        ast.BitXor,
    }
)


class PriceFunctionError(ValueError):
    """Generated source failed the allowlist or did not define quote()."""


class _AllowlistVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_count = 0

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in _ALLOWED_NODE_TYPES:
            raise PriceFunctionError(f"disallowed syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_count += 1
        if node.name != "quote":
            raise PriceFunctionError("only a function named quote() is allowed")
        if node.decorator_list:
            raise PriceFunctionError("decorators are not allowed")
        if node.args.kwarg is not None or node.args.vararg is not None:
            raise PriceFunctionError("quote() may not take *args or **kwargs")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        raise PriceFunctionError("async functions are not allowed")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raise PriceFunctionError("classes are not allowed")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name != "math" or alias.asname is not None:
                raise PriceFunctionError("only 'import math' is allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module != "math" or node.level:
            raise PriceFunctionError("only 'from math import …' is allowed")
        for alias in node.names:
            if alias.name == "*" or alias.name.startswith("_"):
                raise PriceFunctionError("math import is not a public name")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise PriceFunctionError("dunder attributes are not allowed")
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            self.generic_visit(node)
            return
        if node.attr in SAFE_METHODS:
            self.generic_visit(node)
            return
        raise PriceFunctionError(f"attribute {node.attr!r} is not allowed")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in ALLOWED_CALL_NAMES:
                raise PriceFunctionError(f"call to {func.id!r} is not allowed")
        elif isinstance(func, ast.Attribute):
            if func.attr.startswith("_"):
                raise PriceFunctionError("dunder attributes are not allowed")
            math_call = (
                isinstance(func.value, ast.Name) and func.value.id == "math"
            )
            if not math_call and func.attr not in SAFE_METHODS:
                raise PriceFunctionError(
                    f"call to {func.attr!r} is not allowed"
                )
        else:
            raise PriceFunctionError("call target is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise PriceFunctionError("dunder names are not allowed")
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = node.exc
        if exc is None:
            raise PriceFunctionError("bare raise is not allowed")
        if isinstance(exc, ast.Name):
            if exc.id != "ValueError":
                raise PriceFunctionError("only ValueError may be raised")
        elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            if exc.func.id != "ValueError":
                raise PriceFunctionError("only ValueError may be raised")
        else:
            raise PriceFunctionError("only ValueError may be raised")
        if node.cause is not None:
            raise PriceFunctionError("raise … from is not allowed")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        raise PriceFunctionError("global is not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise PriceFunctionError("nonlocal is not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise PriceFunctionError("lambda is not allowed")

    def visit_While(self, node: ast.While) -> None:
        raise PriceFunctionError("while loops are not allowed")

    def visit_Try(self, node: ast.Try) -> None:
        raise PriceFunctionError("try/except is not allowed")

    def visit_With(self, node: ast.With) -> None:
        raise PriceFunctionError("with is not allowed")

    def visit_Delete(self, node: ast.Delete) -> None:
        raise PriceFunctionError("del is not allowed")

    def visit_Yield(self, node: ast.Yield) -> None:
        raise PriceFunctionError("yield is not allowed")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        raise PriceFunctionError("yield from is not allowed")

    def visit_Await(self, node: ast.Await) -> None:
        raise PriceFunctionError("await is not allowed")


def validate_price_ast(tree: ast.AST) -> None:
    """Raise PriceFunctionError if *tree* is not an allowed price module."""
    visitor = _AllowlistVisitor()
    visitor.visit(tree)
    if visitor.function_count != 1:
        raise PriceFunctionError("source must define exactly one function, quote()")


def source_sha256(source: str) -> str:
    normalized = source.strip().replace("\r\n", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compile_quote(source: str) -> Callable[..., Any]:
    """Parse, allowlist, compile, and return the ``quote`` callable."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PriceFunctionError(f"invalid Python: {exc}") from exc
    validate_price_ast(tree)
    compiled = compile(tree, "<price_function>", "exec")
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "math": math}
    exec(compiled, namespace, namespace)
    fn = namespace.get("quote")
    if not callable(fn):
        raise PriceFunctionError("source must define quote()")
    return fn
