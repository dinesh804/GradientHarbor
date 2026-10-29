import sqlite3
from io import BytesIO

import pytest

from gradientharbor.investigation.database_sources import (
    scan_sql_script, list_sqlite_tables, scan_sqlite_table, _validate_limit,
)


def test_sql_script_flags_mutating_statements_without_execution():
    pytest.importorskip("sqlglot")
    findings = scan_sql_script("SELECT id FROM users; DROP TABLE users;", "mysql")
    assert any(item["title"] == "Non-read-only SQL operation present" for item in findings)


def test_sql_script_does_not_echo_credential_value():
    pytest.importorskip("sqlglot")
    secret = "SuperSecretValue123456"
    findings = scan_sql_script(f"-- password = '{secret}'\nSELECT 1", "mysql")
    assert any("credential" in item["title"].lower() for item in findings)
    assert secret not in str(findings)


def test_sqlite_scan_is_bounded_and_read_only():
    out = BytesIO()
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE customers (id INTEGER, email TEXT)")
        conn.executemany("INSERT INTO customers VALUES (?, ?)", [(1, "a@example.com"), (2, "b@example.com")])
        conn.commit()
        conn.close()
        with open(path, "rb") as f:
            data = f.read()
    finally:
        os.unlink(path)
    assert list_sqlite_tables(data) == ["customers"]
    frame, meta = scan_sqlite_table(data, "customers", sample_rows=1)
    assert len(frame) == 1
    assert meta["read_only"] is True
    assert meta["sample_limit"] == 1


def test_sqlite_rejects_unknown_table_and_invalid_sample_size():
    with pytest.raises(ValueError, match="Sample size"):
        _validate_limit(1001)
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE existing (id INTEGER)")
        conn.commit()
        conn.close()
        with open(path, "rb") as f:
            valid_db = f.read()
    finally:
        os.unlink(path)
    with pytest.raises(ValueError, match="not found"):
        scan_sqlite_table(valid_db, "missing")
