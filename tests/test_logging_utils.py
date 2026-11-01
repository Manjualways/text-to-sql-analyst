import csv

from app import logging_utils


def test_query_logging_records_provider_failure_and_available_usage(
    monkeypatch,
    tmp_path,
):
    log_path = tmp_path / "query_logs.csv"
    monkeypatch.setattr(logging_utils, "LOG_FILE", log_path)

    logging_utils.log_query(
        question="Count tracks",
        status="provider_failure",
        latency_ms=12.5,
        attempts=1,
        prompt_tokens=10,
        completion_tokens=0,
        total_tokens=10,
        error_type="LLMQuotaExceededError",
    )

    with log_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["status"] == "provider_failure"
    assert rows[0]["attempts"] == "1"
    assert rows[0]["prompt_tokens"] == "10"
    assert rows[0]["error_type"] == "LLMQuotaExceededError"
