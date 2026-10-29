import unittest
from gradientharbor.security.integrity import sha256_bytes, verify_bytes
class IntegrityTests(unittest.TestCase):
    def test_integrity(self):
        b=b"trusted text"; h=sha256_bytes(b)
        self.assertTrue(verify_bytes(b,h)); self.assertFalse(verify_bytes(b"modified",h))
if __name__ == "__main__": unittest.main()
