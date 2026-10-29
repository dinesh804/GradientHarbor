"""Privacy-conscious JSONL metrics for agent safety, reliability, and latency."""
from __future__ import annotations
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
METRICS_PATH = BASE / "audit_logs" / "agent_metrics.jsonl"
_LOCK = threading.Lock()

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def record_metric(event: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    """Persist metrics without raw prompts, uploaded content, API keys, or answers."""
    target = Path(path) if path else METRICS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": utc_now(), "event_id": uuid.uuid4().hex, **event}
    with _LOCK:
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    return record

def load_metrics(path: str | Path | None = None, limit: int = 5000) -> list[dict[str, Any]]:
    target = Path(path) if path else METRICS_PATH
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-max(1, min(limit, 20000)):]

def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [r for r in rows if r.get("event_type") == "workflow"]
    stages = [r for r in rows if r.get("event_type") == "agent_stage"]
    blocked = sum(1 for r in rows if r.get("event_type") == "guardrail" and r.get("decision") == "blocked")
    reviewed = sum(1 for r in rows if r.get("event_type") == "guardrail" and r.get("decision") == "review")
    latencies = [float(r["latency_ms"]) for r in runs if isinstance(r.get("latency_ms"), (int, float))]
    success = sum(1 for r in runs if r.get("status") == "success")
    verifier_pass = sum(1 for r in runs if r.get("verifier_status") == "pass")
    verifier_review = sum(1 for r in runs if r.get("verifier_status") == "needs_review")
    return {
        "workflow_runs": len(runs),
        "workflow_successes": success,
        "success_rate_pct": round(100 * success / len(runs), 1) if runs else 0.0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": round(sorted(latencies)[min(len(latencies)-1, int(0.95 * len(latencies)))] , 1) if latencies else 0.0,
        "agent_stages": len(stages),
        "guardrail_blocks": blocked,
        "guardrail_reviews": reviewed,
        "verifier_passes": verifier_pass,
        "verifier_needs_review": verifier_review,
    }
