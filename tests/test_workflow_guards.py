from unittest.mock import patch

import pandas as pd

from gradientharbor.agents import workflow


def test_blocked_answer_skips_verifier_model_call():
    frame = pd.DataFrame({"city": ["Trichy", "Chennai"], "revenue": [100.0, 200.0]})
    leaked_secret = "Customer access token: nvapi-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    with patch.object(workflow, "record_metric"), patch.object(
        workflow,
        "chat",
        side_effect=[
            '{"intent":"analytics","needs_sql":true,"needs_evidence":false,"risk_notes":[]}',
            "SELECT city FROM transactions LIMIT 5",
            leaked_secret,
        ],
    ) as chat_mock:
        result = workflow.run_nemotron_workflow(
            "Which city has the highest revenue?", frame, [], "test-api-key", "test-model"
        )

    # Planner, SQL analyst, and answer synthesizer run; verifier must not run.
    assert chat_mock.call_count == 3
    assert result["verifier_status"] == "not_run"
    assert result["output_screen"]["blocked"] is True
    assert "nvapi-" not in result["answer"]
    assert result["trace"][-1]["agent"] == "Output Safety Gate"
    assert result["trace"][-1]["status"] == "blocked"
