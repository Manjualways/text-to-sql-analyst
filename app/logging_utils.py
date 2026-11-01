import csv
import logging
import threading
from datetime import datetime, timezone

from app.config import ROOT_DIR

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "query_logs.csv"

_LOG_LOCK = threading.Lock()

FIELDS = [
    "timestamp",
    "question",
    "status",
    "latency_ms",
    "attempts",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "error_type",
]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_query(
    question: str,
    status: str,
    latency_ms: float = 0,
    attempts: int = 0,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
    error_type: str = "",
) -> None:
    """Append one query's metrics to the CSV log."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "status": status,
        "latency_ms": latency_ms,
        "attempts": attempts,
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else "",
        "completion_tokens": (
            completion_tokens if completion_tokens is not None else ""
        ),
        "total_tokens": total_tokens if total_tokens is not None else "",
        "estimated_cost_usd": (
            estimated_cost_usd if estimated_cost_usd is not None else ""
        ),
        "error_type": error_type,
    }

    with _LOG_LOCK:
        needs_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0

        with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)

            if needs_header:
                writer.writeheader()

            writer.writerow(record)
