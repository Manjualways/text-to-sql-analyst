import re
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.config import API_KEY, BASE_URL, MODEL
from app.logging_utils import get_logger

logger = get_logger(__name__)

MAX_PROVIDER_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (0.5, 1.0)


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM API configuration is invalid."""


class LLMAPIError(RuntimeError):
    """Raised when the configured LLM provider fails a request."""

    def __init__(
        self,
        message: str,
        provider_error: str = "",
        status_code: int | None = None,
        provider_error_type: str = "",
    ):
        super().__init__(message)
        self.provider_error = provider_error
        self.status_code = status_code
        self.provider_error_type = provider_error_type


class LLMRateLimitError(LLMAPIError):
    """Raised when the provider rate-limits a request."""


class LLMQuotaExceededError(LLMRateLimitError):
    """Raised when the provider explicitly reports an exhausted quota."""


class LLMEmptyResponseError(LLMAPIError):
    """Raised when the provider returns no usable completion content."""


client: OpenAI | None = None

last_usage = {
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
    "estimated_cost_usd": None,
    "provider_attempts": 0,
    "provider_retries": 0,
}


def _get_client() -> OpenAI:
    global client

    if not API_KEY or not API_KEY.strip():
        raise LLMConfigurationError(
            "LLM_API_KEY is not configured. Add your API key "
            "to the project .env file."
        )

    if client is None:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            timeout=45.0,
            max_retries=0,
        )

    return client


def _redact_sensitive_text(message: str) -> str:
    if API_KEY:
        message = message.replace(API_KEY, "[REDACTED]")

    message = re.sub(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*",
        "Bearer [REDACTED]",
        message,
    )
    return re.sub(
        r"(?i)\b(?:sk|gsk)[-_][A-Za-z0-9_-]{8,}",
        "[REDACTED]",
        message,
    )


def _safe_error_text(error: Exception) -> str:
    """Keep useful provider diagnostics while redacting credential-like text."""
    message = str(error)
    response = getattr(error, "response", None)
    if response is not None:
        try:
            response_text = response.text
        except Exception as response_error:
            response_text = (
                f"<provider response body unavailable: "
                f"{type(response_error).__name__}>"
            )

        if response_text and response_text not in message:
            message = f"{message}; provider response: {response_text}"

    return _redact_sensitive_text(message)


def _is_quota_exhausted(error: Exception) -> bool:
    detail = _safe_error_text(error).casefold()

    return any(
        marker in detail
        for marker in (
            "tokens per day",
            "token per day",
            "daily token",
            "daily quota",
            "quota exceeded",
            "insufficient_quota",
            "billing hard limit",
            "requires more credits",
            "openrouter_credits",
        )
    )


def _is_transient(error: Exception) -> bool:
    return isinstance(
        error,
        (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError),
    ) or getattr(error, "status_code", None) == 429 or (
        isinstance(getattr(error, "status_code", None), int)
        and getattr(error, "status_code") >= 500
    )


def ask_llm(prompt: str) -> str:
    global last_usage

    last_usage = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "estimated_cost_usd": None,
        "provider_attempts": 0,
        "provider_retries": 0,
    }

    try:
        provider_client = _get_client()
    except LLMConfigurationError:
        raise
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        safe_detail = _safe_error_text(exc)
        logger.error(
            "LLM provider client initialization failed. "
            "Model=%s, Status=%s, ErrorType=%s, Detail=%s",
            MODEL,
            status_code,
            type(exc).__name__,
            safe_detail,
        )
        raise LLMAPIError(
            "The LLM provider client could not be initialized.",
            provider_error=safe_detail,
            status_code=status_code,
            provider_error_type=type(exc).__name__,
        ) from None

    for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
        last_usage["provider_attempts"] = attempt

        try:
            response = provider_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful Text-to-SQL assistant. "
                            "Generate only the SQL requested. "
                            "Follow the user's instructions exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=1024,
                temperature=0,
                extra_body={
                    "include_reasoning": False,
                    "reasoning_effort": "low",
                },
            )

            if not response.choices:
                raise LLMEmptyResponseError(
                    "The LLM provider returned no response choices."
                )

            usage = response.usage
            if usage is not None:
                last_usage.update({
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "estimated_cost_usd": getattr(usage, "cost", None),
                })

            content = response.choices[0].message.content or ""
            if not content.strip():
                raise LLMEmptyResponseError(
                    "The LLM provider returned an empty response."
                )

            logger.info(
                "LLM request completed. Model=%s, usage=%s",
                MODEL,
                last_usage,
            )
            return content

        except LLMEmptyResponseError:
            raise

        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            safe_detail = _safe_error_text(exc)
            logger.warning(
                "LLM provider request failed. Model=%s, status=%s, "
                "error_type=%s, detail=%s",
                MODEL,
                status_code,
                type(exc).__name__,
                safe_detail,
            )

            if status_code == 429 or isinstance(exc, RateLimitError):
                if _is_quota_exhausted(exc):
                    raise LLMQuotaExceededError(
                        "The LLM provider's quota is exhausted. "
                        "No automatic retry was made; check provider quota "
                        "availability before submitting another request.",
                        provider_error=safe_detail,
                        status_code=status_code,
                        provider_error_type=type(exc).__name__,
                    ) from None

                if attempt < MAX_PROVIDER_ATTEMPTS:
                    last_usage["provider_retries"] = attempt
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt - 1])
                    continue
                raise LLMRateLimitError(
                    "The LLM provider is rate-limiting requests. "
                    "Please try again later.",
                    provider_error=safe_detail,
                    status_code=status_code,
                    provider_error_type=type(exc).__name__,
                ) from None

            if _is_transient(exc) and attempt < MAX_PROVIDER_ATTEMPTS:
                last_usage["provider_retries"] = attempt
                time.sleep(RETRY_BACKOFF_SECONDS[attempt - 1])
                continue

            raise LLMAPIError(
                "The LLM provider is unavailable. "
                f"Provider error type: {type(exc).__name__}.",
                provider_error=safe_detail,
                status_code=status_code,
                provider_error_type=type(exc).__name__,
            ) from None

    raise LLMAPIError("The LLM provider is unavailable after bounded retries.")
