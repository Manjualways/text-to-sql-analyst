from types import SimpleNamespace

import pytest

from app import llm


def test_ask_llm_returns_content_and_records_usage(monkeypatch):
    request_kwargs = {}
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=4,
            total_tokens=16,
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="SELECT 1")
            )
        ],
    )

    def create_completion(**kwargs):
        request_kwargs.update(kwargs)
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=create_completion,
            )
        )
    )
    monkeypatch.setattr(llm, "client", client)
    monkeypatch.setattr(llm, "API_KEY", "test-key")

    assert llm.ask_llm("prompt") == "SELECT 1"
    assert "extra_body" not in request_kwargs
    assert llm.last_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
        "estimated_cost_usd": None,
        "provider_attempts": 1,
        "provider_retries": 0,
    }


def test_ask_llm_wraps_provider_errors(monkeypatch):
    def fail_request(**_kwargs):
        raise ValueError("provider response failed test-key")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fail_request)
        )
    )
    monkeypatch.setattr(llm, "client", client)
    monkeypatch.setattr(llm, "API_KEY", "test-key")

    with pytest.raises(llm.LLMAPIError, match="provider is unavailable") as exc:
        llm.ask_llm("prompt")

    assert exc.value.provider_error == "provider response failed [REDACTED]"
    assert exc.value.__cause__ is None
    assert "test-key" not in str(exc.value)
    assert llm.last_usage == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "estimated_cost_usd": None,
        "provider_attempts": 1,
        "provider_retries": 0,
    }


def test_provider_response_body_is_preserved_and_redacted(monkeypatch):
    class ProviderError(Exception):
        status_code = 400
        response = SimpleNamespace(
            text='{"error":"invalid request","credential":"test-key"}'
        )

    def fail_request(**_kwargs):
        raise ProviderError("provider rejected request")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fail_request)
        )
    )
    monkeypatch.setattr(llm, "client", client)
    monkeypatch.setattr(llm, "API_KEY", "test-key")

    with pytest.raises(llm.LLMAPIError) as exc:
        llm.ask_llm("prompt")

    assert exc.value.status_code == 400
    assert "invalid request" in exc.value.provider_error
    assert "test-key" not in exc.value.provider_error
    assert "[REDACTED]" in exc.value.provider_error
    assert exc.value.__cause__ is None


def test_client_initialization_error_preserves_safe_http_diagnostics(
    monkeypatch,
):
    class InitializationError(Exception):
        status_code = 401
        response = SimpleNamespace(text='{"message":"bad test-key"}')

    def fail_client(**_kwargs):
        raise InitializationError("client initialization rejected")

    monkeypatch.setattr(llm, "API_KEY", "test-key")
    monkeypatch.setattr(llm, "client", None)
    monkeypatch.setattr(llm, "OpenAI", fail_client)

    with pytest.raises(
        llm.LLMAPIError,
        match="client could not be initialized",
    ) as exc:
        llm.ask_llm("prompt")

    assert exc.value.status_code == 401
    assert "bad [REDACTED]" in exc.value.provider_error
    assert "test-key" not in exc.value.provider_error


def test_quota_exhaustion_is_not_retried(monkeypatch):
    calls = []

    class QuotaError(Exception):
        status_code = 429

    def fail_request(**_kwargs):
        calls.append(True)
        raise QuotaError("tokens per day quota exhausted test-key")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fail_request)
        )
    )
    monkeypatch.setattr(llm, "client", client)
    monkeypatch.setattr(llm, "API_KEY", "test-key")

    with pytest.raises(
        llm.LLMQuotaExceededError,
        match="quota is exhausted",
    ) as exc:
        llm.ask_llm("prompt")

    assert len(calls) == 1
    assert llm.last_usage["provider_attempts"] == 1
    assert llm.last_usage["provider_retries"] == 0
    assert "test-key" not in str(exc.value.provider_error)


def test_client_disables_sdk_retries(monkeypatch):
    kwargs = {}

    def create_client(**values):
        kwargs.update(values)
        return object()

    monkeypatch.setattr(llm, "API_KEY", "test-key")
    monkeypatch.setattr(llm, "client", None)
    monkeypatch.setattr(llm, "OpenAI", create_client)

    llm._get_client()

    assert kwargs["max_retries"] == 0


def test_transient_provider_error_uses_bounded_retry(monkeypatch):
    calls = []
    sleeps = []
    response = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="SELECT 1"))
        ],
    )

    class TemporaryError(Exception):
        status_code = 503

    def request(**_kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise TemporaryError("temporarily unavailable")
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=request)
        )
    )
    monkeypatch.setattr(llm, "client", client)
    monkeypatch.setattr(llm, "API_KEY", "test-key")
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    assert llm.ask_llm("prompt") == "SELECT 1"
    assert len(calls) == 2
    assert sleeps == [llm.RETRY_BACKOFF_SECONDS[0]]
    assert llm.last_usage["provider_attempts"] == 2
    assert llm.last_usage["provider_retries"] == 1


def test_empty_provider_response_is_rejected(monkeypatch):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[],
                    usage=None,
                )
            )
        )
    )
    monkeypatch.setattr(llm, "client", client)
    monkeypatch.setattr(llm, "API_KEY", "test-key")

    with pytest.raises(llm.LLMEmptyResponseError, match="no response choices"):
        llm.ask_llm("prompt")


def test_missing_api_key_does_not_create_client(monkeypatch):
    monkeypatch.setattr(llm, "API_KEY", "")
    monkeypatch.setattr(llm, "client", None)

    with pytest.raises(llm.LLMConfigurationError, match="LLM_API_KEY"):
        llm._get_client()

    assert llm.client is None
