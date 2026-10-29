import tempfile, unittest, json
from pathlib import Path
from gradientharbor.audit.hash_chain import append_event, verify_chain

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

if __name__ == "__main__": unittest.main()
