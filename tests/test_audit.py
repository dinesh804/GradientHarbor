import tempfile, unittest, json
from pathlib import Path
from gradientharbor.audit.hash_chain import append_event, verify_chain
from gradientharbor.audit.hash_chain import read_events
from gradientharbor.security.crypto import ensure_keys
from concurrent.futures import ThreadPoolExecutor

class AuditTests(unittest.TestCase):
    def test_valid_chain(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"events.jsonl"; append_event(p,{"action":"allow"}); append_event(p,{"action":"block"})
            self.assertTrue(verify_chain(p)[0])
    def test_tamper_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"events.jsonl"; append_event(p,{"action":"allow"})
            rec=json.loads(p.read_text()); rec["event"]["action"]="tampered"; p.write_text(json.dumps(rec)+"\n")
            self.assertFalse(verify_chain(p)[0])

    def test_concurrent_appends_preserve_a_valid_chain(self):
        ensure_keys()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.jsonl"
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda n: append_event(p, {"action": "concurrent", "number": n}), range(40)))
            valid, detail = verify_chain(p)
            self.assertTrue(valid, detail)
            self.assertEqual(len(read_events(p)), 40)

    def test_append_refuses_to_extend_a_tampered_chain(self):
        ensure_keys()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.jsonl"
            append_event(p, {"action": "original"})
            record = json.loads(p.read_text(encoding="utf-8"))
            record["event"]["action"] = "tampered"
            p.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid audit chain"):
                append_event(p, {"action": "new"})

if __name__ == "__main__": unittest.main()
