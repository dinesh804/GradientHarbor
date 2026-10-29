"""Read-only SQL/MongoDB inspection helpers for GradientHarbor.

Connection secrets are accepted as call arguments only and must never be logged.
Use dedicated database accounts with SELECT/find-only privileges as the final boundary.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from gradientharbor.investigation.scanner import scan_dataframe, _finding
from gradientharbor.security.prompt_guard import scan_untrusted_text

MAX_SAMPLE_ROWS = 1000
MAX_SQL_BYTES = 1_000_000


def scan_sql_script(script: str, dialect: str = "mysql") -> list[dict[str, Any]]:
    """Statically review a SQL script; never execute its contents."""
    if not isinstance(script, str) or not script.strip():
        raise ValueError("SQL script is empty.")
    if len(script.encode("utf-8")) > MAX_SQL_BYTES:
        raise ValueError("SQL script exceeds the 1 MiB review limit.")
    try:
        import sqlglot
        from sqlglot import exp
        statements = [s for s in sqlglot.parse(script, read=dialect) if s is not None]
    except Exception as exc:
        return [_finding("Database/data exposure", "SQL script could not be parsed", "Medium", "High",
                          f"Parser rejected or could not understand the script ({type(exc).__name__}). The script was not executed.",
                          "Review the script manually with a parser for the correct SQL dialect; do not execute untrusted SQL.")]
    findings: list[dict[str, Any]] = []
    mutating_types = tuple(getattr(exp, n) for n in (
        "Insert", "Update", "Delete", "Create", "Drop", "Alter", "Command", "Copy", "Grant", "Revoke",
        "Merge", "Into", "Lock", "Attach", "Detach", "Pragma", "Execute", "Call", "Truncate",
    ) if hasattr(exp, n))
    risky_count = sum(1 for tree in statements if not isinstance(tree, exp.Query) or (mutating_types and any(tree.find_all(*mutating_types))) or re.search(r"\bINTO\b", tree.sql(), re.I))
    if risky_count:
        findings.append(_finding("Database/data exposure", "Non-read-only SQL operation present", "High", "High",
                                 f"{risky_count} of {len(statements)} statement(s) are not clearly read-only SELECT/query statements. The script was inspected but never executed.",
                                 "Review every flagged statement. Run only approved SELECT queries using a dedicated read-only database account.", rows=risky_count))
    if not statements:
        findings.append(_finding("Database/data exposure", "No SQL statements detected", "Info", "High",
                                 "No parseable SQL statements were found; no code was executed.", "Confirm that the uploaded file contains SQL text."))
    secret_hits = len(re.findall(r"(?i)(?:password|passwd|api[_ -]?key|access[_ -]?token|secret)\s*(?:=|:)\s*['\"]?[^\s,'\"]{6,}", script))
    if secret_hits:
        findings.append(_finding("Database/data exposure", "Possible hard-coded credential in SQL script", "High", "Medium",
                                 f"Found {secret_hits} credential-shaped assignment(s). Values are intentionally omitted from the report.",
                                 "Rotate any real exposed credential and move secrets into a secret manager or environment configuration.", rows=secret_hits))
    guard = scan_untrusted_text(script[:20000])
    if guard.get("suspicious"):
        findings.append(_finding("AI-agent & prompt injection", "Prompt-injection-like text in SQL comments or literals", "Medium", "Medium",
                                 f"The script contains suspicious instruction-like text ({', '.join(guard.get('signals', [])[:6])}). Raw SQL text is not included.",
                                 "Treat comments and string literals as untrusted data; never use them as system instructions for an AI agent."))
    return findings


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SAMPLE_ROWS:
        raise ValueError(f"Sample size must be between 1 and {MAX_SAMPLE_ROWS} rows/documents.")
    return limit


def list_relational_tables(dialect: str, host: str, port: int, database: str,
                           username: str, password: str) -> list[str]:
    """List table names using SQLAlchemy's schema inspector; no data is modified."""
    try:
        from sqlalchemy import URL, create_engine, inspect
    except ImportError as exc:
        raise RuntimeError("Install SQLAlchemy and the selected database driver from requirements.txt.") from exc
    if dialect not in {"mysql", "postgresql"}:
        raise ValueError("Supported network SQL databases are MySQL and PostgreSQL.")
    driver = "mysql+pymysql" if dialect == "mysql" else "postgresql+psycopg"
    url = URL.create(driver, username=username, password=password, host=host, port=int(port), database=database)
    engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5} if dialect == "mysql" else {"connect_timeout": 5})
    try:
        with engine.connect() as conn:
            if dialect == "mysql":
                conn.exec_driver_sql("SET SESSION TRANSACTION READ ONLY")
                conn.commit()
            names = inspect(conn).get_table_names()
            return sorted(names)
    finally:
        engine.dispose()


def scan_relational_table(dialect: str, host: str, port: int, database: str,
                          username: str, password: str, table_name: str,
                          sample_rows: int = 500) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch a bounded sample with a reflected identifier, never raw SQL from the user."""
    sample_rows = _validate_limit(sample_rows)
    try:
        from sqlalchemy import URL, MetaData, Table, create_engine, select
        from sqlalchemy.exc import SQLAlchemyError
    except ImportError as exc:
        raise RuntimeError("Install SQLAlchemy and the selected database driver from requirements.txt.") from exc
    if dialect not in {"mysql", "postgresql"}:
        raise ValueError("Supported network SQL databases are MySQL and PostgreSQL.")
    driver = "mysql+pymysql" if dialect == "mysql" else "postgresql+psycopg"
    url = URL.create(driver, username=username, password=password, host=host, port=int(port), database=database)
    engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
    try:
        with engine.connect() as conn:
            if dialect == "mysql":
                conn.exec_driver_sql("SET SESSION TRANSACTION READ ONLY")
                conn.commit()
            trans = conn.begin()
            try:
                if dialect == "postgresql":
                    conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                metadata = MetaData()
                table = Table(table_name, metadata, autoload_with=conn)
                query = select(table).limit(sample_rows)
                frame = pd.read_sql_query(query, conn)
                trans.rollback()
            except Exception:
                if trans.is_active:
                    trans.rollback()
                raise
        return frame, {"format": dialect, "table": table_name, "sampled_rows": len(frame), "sample_limit": sample_rows, "read_only": True}
    except Exception as exc:
        # Avoid returning the exception text: some drivers embed usernames/hosts/SQL in it.
        raise RuntimeError(f"Could not inspect the selected {dialect} table ({type(exc).__name__}). Check connection settings and read-only grants.") from None
    finally:
        engine.dispose()


def list_mongodb_collections(uri: str, database: str) -> list[str]:
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("Install pymongo from requirements.txt to enable MongoDB scanning.") from exc
    if not uri.strip() or not database.strip():
        raise ValueError("MongoDB URI and database name are required.")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, socketTimeoutMS=10000, appname="GradientHarborReadOnlyScanner")
    try:
        client.admin.command("ping")
        return sorted(client[database].list_collection_names())
    except Exception as exc:
        raise RuntimeError(f"Could not list MongoDB collections ({type(exc).__name__}). Check URI, database, network access and read-only grants.") from None
    finally:
        client.close()


def scan_mongodb_collection(uri: str, database: str, collection: str,
                            sample_rows: int = 500) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_rows = _validate_limit(sample_rows)
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("Install pymongo from requirements.txt to enable MongoDB scanning.") from exc
    if not uri.strip() or not database.strip() or not collection.strip():
        raise ValueError("MongoDB URI, database and collection are required.")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, socketTimeoutMS=10000, appname="GradientHarborReadOnlyScanner")
    try:
        cursor = client[database][collection].find({}, limit=sample_rows, max_time_ms=5000)
        docs = list(cursor)
        # Exclude internal MongoDB IDs from analysis/reporting.
        for doc in docs:
            doc.pop("_id", None)
        frame = pd.json_normalize(docs, sep=".") if docs else pd.DataFrame()
        return frame, {"format": "mongodb", "database": database, "collection": collection, "sampled_rows": len(frame), "sample_limit": sample_rows, "read_only": True}
    except Exception as exc:
        raise RuntimeError(f"Could not inspect the selected MongoDB collection ({type(exc).__name__}). Check connection and read-only grants.") from None
    finally:
        client.close()


def findings_for_frame(frame: pd.DataFrame, source_label: str) -> list[dict[str, Any]]:
    """Run the existing tabular scanner on a database sample and label its bounded scope."""
    findings = scan_dataframe(frame, sheet_name=source_label)
    if not findings:
        findings.append(_finding("Database/data exposure", "No candidate findings in sampled data", "Info", "Medium",
                                 f"No configured heuristic matched the sampled records from '{source_label}'. Only a bounded sample was checked.",
                                 "Review scan coverage and ensure database permissions, indexes, and server configuration are assessed separately."))
    return findings


def list_sqlite_tables(database_bytes: bytes) -> list[str]:
    """List tables from an uploaded SQLite file using a temporary read-only copy."""
    import os
    import sqlite3
    import tempfile
    if not database_bytes or len(database_bytes) > 50 * 1024 * 1024:
        raise ValueError("SQLite database must be non-empty and no larger than 50 MiB.")
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(database_bytes)
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            conn.execute("PRAGMA query_only=ON")
            return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        raise RuntimeError(f"Could not inspect SQLite file ({type(exc).__name__}).") from None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def scan_sqlite_table(database_bytes: bytes, table_name: str, sample_rows: int = 500) -> tuple[pd.DataFrame, dict[str, Any]]:
    import os
    import sqlite3
    import tempfile
    sample_rows = _validate_limit(sample_rows)
    if not database_bytes or len(database_bytes) > 50 * 1024 * 1024:
        raise ValueError("SQLite database must be non-empty and no larger than 50 MiB.")
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(database_bytes)
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            conn.execute("PRAGMA query_only=ON")
            available = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if table_name not in available:
                raise ValueError("Selected table was not found in the uploaded SQLite database.")
            escaped = table_name.replace('"', '""')
            frame = pd.read_sql_query(f'SELECT * FROM "{escaped}" LIMIT {sample_rows}', conn)
            return frame, {"format": "sqlite", "table": table_name, "sampled_rows": len(frame), "sample_limit": sample_rows, "read_only": True}
        finally:
            conn.close()
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not inspect SQLite table ({type(exc).__name__}).") from None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
