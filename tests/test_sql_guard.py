import unittest
from gradientharbor.security.sql_guard import validate_read_only_sql, SQLPolicyError

class SQLGuardTests(unittest.TestCase):
    def test_select_allowed_and_limited(self):
        sql=validate_read_only_sql("SELECT city FROM transactions", "sqlite", allowed_tables={"transactions"})
        self.assertIn("LIMIT", sql.upper())
    def test_delete_blocked(self):
        with self.assertRaises(SQLPolicyError):
            validate_read_only_sql("DELETE FROM transactions", "sqlite", allowed_tables={"transactions"})
    def test_multiple_statements_blocked(self):
        with self.assertRaises(SQLPolicyError):
            validate_read_only_sql("SELECT 1; DROP TABLE transactions", "sqlite", allowed_tables={"transactions"})
    def test_other_table_blocked(self):
        with self.assertRaises(SQLPolicyError):
            validate_read_only_sql("SELECT * FROM sqlite_master", "sqlite", allowed_tables={"transactions"})

if __name__ == "__main__": unittest.main()
