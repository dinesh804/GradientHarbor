import unittest

try:
    import yara  # noqa: F401
    YARA_AVAILABLE = True
except ImportError:
    YARA_AVAILABLE = False

from gradientharbor.security.yara_scanner import scan_bytes, MAX_FILE_BYTES, compile_rules


@unittest.skipUnless(YARA_AVAILABLE, "optional yara-python dependency is not installed")
class YaraScannerTests(unittest.TestCase):
    def test_suspicious_powershell_pattern_matches(self):
        data = b"DownloadString( Invoke-Expression"
        result = scan_bytes(data, "sample.ps1")
        self.assertGreaterEqual(result["match_count"], 1)
        self.assertEqual(result["filename"], "sample.ps1")
        self.assertEqual(len(result["sha256"]), 64)

    def test_benign_file_returns_no_matches(self):
        result = scan_bytes(b"Quarterly revenue report: 5000 transactions", "report.txt")
        self.assertEqual(result["match_count"], 0)
        self.assertEqual(result["status"], "no_matches")

    def test_custom_rule_supported(self):
        rule = b'rule Test_Custom { strings: $a = "CUSTOM_MARKER" condition: $a }'
        result = scan_bytes(b"contains CUSTOM_MARKER", "custom.bin", rule)
        self.assertTrue(any(row["rule"] == "Test_Custom" for row in result["findings"]))

    def test_invalid_custom_rule_rejected(self):
        with self.assertRaises(ValueError):
            compile_rules("this is not a yara rule")


class YaraInputSafetyTests(unittest.TestCase):
    def test_file_size_limit_applies_before_dependency_check(self):
        with self.assertRaises(ValueError):
            scan_bytes(b"x" * (MAX_FILE_BYTES + 1), "large.bin")


if __name__ == "__main__":
    unittest.main()
