"""Fail-closed SQL policy for a narrow single-SELECT workflow."""
from __future__ import annotations
import re
try:
    import sqlglot
    from sqlglot import exp
except ImportError as exc:
    sqlglot, exp, _IMPORT_ERROR = None, None, exc
else:
    _IMPORT_ERROR = None

class SQLPolicyError(ValueError):
    pass

def validate_read_only_sql(sql: str, dialect: str = "sqlite", max_length: int = 10000,
                           allowed_tables: set[str] | None = None) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise SQLPolicyError("Empty SQL is not allowed.")
    candidate = sql.strip()
    if len(candidate) > max_length:
        raise SQLPolicyError("SQL exceeds configured length limit.")
    if "\x00" in candidate:
        raise SQLPolicyError("NUL characters are not allowed.")
    if sqlglot is None:
        raise SQLPolicyError("SQL parser unavailable; query rejected.") from _IMPORT_ERROR
    try:
        statements = [x for x in sqlglot.parse(candidate, read=dialect) if x is not None]
    except Exception as exc:
        raise SQLPolicyError("SQL parsing failed; query rejected.") from exc
    if len(statements) != 1:
        raise SQLPolicyError("Exactly one SQL statement is allowed.")
    tree = statements[0]
    if not isinstance(tree, exp.Select):
        raise SQLPolicyError("Only a SELECT statement is allowed.")
    forbidden = ("Insert", "Update", "Delete", "Create", "Drop", "Alter", "Command",
                 "Copy", "Transaction", "Grant", "Revoke", "Merge", "Into", "Lock",
                 "Attach", "Detach", "Pragma", "Execute", "Call")
    types = tuple(getattr(exp, n) for n in forbidden if hasattr(exp, n))
    if types and any(tree.find_all(*types)):
        raise SQLPolicyError("Query contains a prohibited operation.")
    if re.search(r"\bINTO\b", candidate, flags=re.I):
        raise SQLPolicyError("SELECT INTO is not allowed.")
    bad_functions = {"LOAD_FILE", "PG_READ_FILE", "PG_READ_BINARY_FILE", "HTTP_GET"}
    for fn in tree.find_all(exp.Func):
        fname = (fn.sql_name() or getattr(fn, "name", "") or "").upper()
        if fname in bad_functions:
            raise SQLPolicyError(f"Function {fname} is not allowed.")
    if allowed_tables is not None:
        names = {t.name.lower() for t in tree.find_all(exp.Table)}
        allowed = {x.lower() for x in allowed_tables}
        if not names or not names.issubset(allowed):
            raise SQLPolicyError("Query references a table outside the allowed dataset.")
    normalized = tree.sql(dialect=dialect)
    if not tree.args.get("limit"):
        tree = tree.limit(100)
        normalized = tree.sql(dialect=dialect)
    return normalized
