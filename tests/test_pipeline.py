"""Tests for the text-to-SQL pipeline."""

from itertools import count

import pytest

from app import llm, pipeline


def test_run_pipeline_executes_validated_sql(monkeypatch):
    monkeypatch.setattr(
        llm,
        "last_usage",
        {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "estimated_cost_usd": None,
            "provider_attempts": 0,
            "provider_retries": 0,
        },
    )
    monkeypatch.setattr(pipeline, "retrieve_schema", lambda _question: "Artist(Name)")
    monkeypatch.setattr(
        pipeline,
        "format_results",
        lambda **_kwargs: {"summary": "Three artists found.", "chart_type": "bar"},
    )
    timestamps = count()
    monkeypatch.setattr(
        pipeline.time,
        "perf_counter",
        lambda: next(timestamps) / 10,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda _question, _schema, _error_context: (
            "SELECT Name FROM Artist ORDER BY Name LIMIT 3"
        ),
    )

    result = pipeline.run_pipeline("List three artists")

    assert result["columns"] == ["Name"]
    assert result["row_count"] == 3
    assert result["attempts"] == 1
    assert result["summary"] == "Three artists found."
    assert result["chart_type"] == "bar"
    assert result["latency_ms"] > 0
    assert result["retry_count"] == 0
    assert result["provider_attempts"] == 0


def test_run_pipeline_retries_invalid_sql_at_most_three_times(monkeypatch):
    attempts = []
    monkeypatch.setattr(pipeline, "retrieve_schema", lambda _question: "Artist(Name)")
    monkeypatch.setattr(
        pipeline,
        "format_results",
        lambda **_kwargs: {"summary": "", "chart_type": "table"},
    )

    def generate_sql(_question, _schema, error_context):
        attempts.append(error_context)
        return "DELETE FROM Artist"

    monkeypatch.setattr(pipeline, "generate_sql", generate_sql)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        pipeline.run_pipeline("List artists")

    assert len(attempts) == 3
    assert "DELETE FROM Artist" in attempts[-1]


def test_provider_quota_failure_does_not_retry_sql(monkeypatch):
    generated = []
    logs = []
    monkeypatch.setattr(
        pipeline,
        "retrieve_schema",
        lambda _question: "Artist(Name)",
    )
    monkeypatch.setattr(
        pipeline,
        "log_query",
        lambda **values: logs.append(values),
    )
    monkeypatch.setattr(
        llm,
        "last_usage",
        {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "estimated_cost_usd": None,
            "provider_attempts": 1,
            "provider_retries": 0,
        },
    )

    def fail_generation(*_args):
        generated.append(True)
        raise llm.LLMQuotaExceededError("provider quota exhausted")

    monkeypatch.setattr(pipeline, "generate_sql", fail_generation)

    with pytest.raises(llm.LLMQuotaExceededError):
        pipeline.run_pipeline("List artists")

    assert len(generated) == 1
    assert logs[0]["status"] == "provider_failure"
    assert logs[0]["error_type"] == "LLMQuotaExceededError"


def test_generate_sql_preserves_model_sql(monkeypatch):
    expected_sql = (
        "SELECT AVG(UnitPrice) AS AveragePrice FROM Track"
    )
    prompts = []

    monkeypatch.setattr(
        pipeline,
        "ask_llm",
        lambda prompt: prompts.append(prompt) or expected_sql,
    )

    sql = pipeline.generate_sql(
        "What is the average track price?",
        "Track(UnitPrice)",
    )

    assert sql == expected_sql
    assert "read-only SQLite SELECT" in prompts[0]
    assert "foreign-key relationships" in prompts[0]

    monkeypatch.setattr(pipeline, "ask_llm", lambda _prompt: "  ")
    with pytest.raises(pipeline.SQLGenerationError, match="empty query"):
        pipeline.generate_sql("List artists", "Artist(Name)")
