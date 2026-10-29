import tempfile
import unittest
from pathlib import Path
from gradientharbor.observability import record_metric, load_metrics, summarize_metrics

class ObservabilityTests(unittest.TestCase):
    def test_metrics_append_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.jsonl"
            record_metric({"event_type":"workflow", "status":"success", "latency_ms":120, "verifier_status":"pass"}, path)
            record_metric({"event_type":"workflow", "status":"error", "latency_ms":240, "verifier_status":"not_run"}, path)
            record_metric({"event_type":"guardrail", "decision":"blocked"}, path)
            rows = load_metrics(path)
            summary = summarize_metrics(rows)
            self.assertEqual(summary["workflow_runs"], 2)
            self.assertEqual(summary["workflow_successes"], 1)
            self.assertEqual(summary["guardrail_blocks"], 1)
            self.assertEqual(summary["verifier_passes"], 1)

if __name__ == "__main__":
    unittest.main()
