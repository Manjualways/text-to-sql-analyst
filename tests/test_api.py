from fastapi.testclient import TestClient

from app import llm, pipeline
import app.main as main_module
from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Text-to-SQL Analyst Agent is running!",
        "status": "success",
        "docs": "/docs",
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_endpoint():
    response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()


def test_query_endpoint_returns_structured_result_with_mocked_llm(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "format_results",
        lambda **_kwargs: {"summary": "Ten artists ranked by album count.", "chart_type": "bar"},
    )
    monkeypatch.setattr(
        pipeline,
        "ask_llm",
        lambda _prompt: (
            "SELECT Artist.Name, COUNT(Album.AlbumId) AS album_count "
            "FROM Artist JOIN Album ON Artist.ArtistId = Album.ArtistId "
            "GROUP BY Artist.ArtistId, Artist.Name "
            "ORDER BY album_count DESC LIMIT 10"
        ),
    )

    response = client.post(
        "/query",
        json={"question": "Which 10 artists have the most albums?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["row_count"] == 10
    assert payload["columns"] == ["Name", "album_count"]
    assert payload["attempts"] == 1
    assert payload["summary"] == "Ten artists ranked by album count."
    assert payload["chart_type"] == "bar"
    assert isinstance(payload["latency_ms"], float)


def test_query_endpoint_reports_missing_llm_key(monkeypatch):
    monkeypatch.setattr(llm, "API_KEY", "")
    monkeypatch.setattr(llm, "client", None)

    response = client.post("/query", json={"question": "List some artists"})

    assert response.status_code == 503
    assert "LLM_API_KEY is not configured" in response.json()["detail"]


def test_query_endpoint_reports_provider_quota_as_unavailable(monkeypatch):
    def fail_with_quota(_question):
        raise llm.LLMQuotaExceededError(
            "The LLM provider's quota is exhausted."
        )

    monkeypatch.setattr(main_module, "run_pipeline", fail_with_quota)

    response = client.post(
        "/query",
        json={"question": "List some artists"},
    )

    assert response.status_code == 503
    assert "quota is exhausted" in response.json()["detail"]
