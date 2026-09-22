"""Render a parameterized statement for logs and LangSmith spans."""

from __future__ import annotations

from typing import Any


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "ARRAY[" + ", ".join(_sql_literal(v) for v in value) + "]"
    if hasattr(value, "isoformat"):
        value = value.isoformat()
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _render_sql(sql: str, params: list[Any]) -> str:
    parts = sql.split("%s")
    if len(parts) != len(params) + 1:
        return sql
    out = [parts[0]]
    for part, param in zip(parts[1:], params):
        out.append(_sql_literal(param))
        out.append(part)
    return "".join(out)


def _trace_sql_param(value: Any) -> Any:
    """Keep LangSmith SQL readable; pgvector literals are thousands of floats."""
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        return "<vector>"
    return value


def _drop_embedding_input(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inputs.items() if key != "embedding"}


def _attach_run_sql(sql: str, params: list[Any]) -> None:
    """Put interpolated SQL on the current LangSmith span, if any."""
    try:
        from langsmith import get_current_run_tree
    except ImportError:
        return
    run = get_current_run_tree()
    if run is None:
        return
    rendered = _render_sql(sql, [_trace_sql_param(p) for p in params])
    try:
        inputs = dict(run.inputs or {})
        inputs["sql"] = rendered
        run.inputs = inputs
    except Exception:
        return
