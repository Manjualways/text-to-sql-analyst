import csv
import json
import math
import sqlite3
import time
from pathlib import Path

import sqlglot

from app.config import DB_PATH
from app.llm import (
    LLMAPIError,
    LLMConfigurationError,
    LLMQuotaExceededError,
)
from app.pipeline import QueryFailureError, SQLGenerationError, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = PROJECT_ROOT / "eval" / "eval_set.json"
REPORT_PATH = PROJECT_ROOT / "eval" / "eval_results.csv"

# Limit provider usage during development. Set to None for a full benchmark.
MAX_QUESTIONS = 30


def normalize_value(value):
    """Keep numeric result values exact while normalizing non-finite floats."""
    if isinstance(value, bool):
        return value

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    return value


def normalize_rows(columns, rows, ordered=False):
    """Normalize SQLite tuples and pipeline dictionaries consistently."""
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            values = [row.get(column) for column in columns]
        else:
            values = list(row)

        normalized.append(
            tuple(normalize_value(value) for value in values)
        )

    if ordered:
        return normalized

    return sorted(
        normalized,
        key=lambda row: json.dumps(row, sort_keys=True, default=str),
    )


def reference_order_is_significant(sql):
    """Return whether the reference query explicitly specifies output order."""
    expression = sqlglot.parse_one(sql, read="sqlite")
    return expression.args.get("order") is not None


def compare_results(
    expected_columns,
    expected_rows,
    actual_columns,
    actual_rows,
    ordered=False,
):
    """Require matching output aliases, column order, and all result values."""
    columns_match = expected_columns == actual_columns
    if not columns_match:
        return False, False

    expected_data = normalize_rows(expected_columns, expected_rows, ordered)
    actual_data = normalize_rows(actual_columns, actual_rows, ordered)
    return True, expected_data == actual_data


def run_reference_sql(sql):
    """Execute trusted reference SQL against the configured database read-only."""
    uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
    return columns, rows


def _failure_chain(error):
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def classify_failure(error):
    """Classify a failed run by the specific stage that caused the failure."""
    for current in _failure_chain(error):
        if isinstance(current, LLMConfigurationError):
            return "provider_configuration"
        if isinstance(current, LLMAPIError):
            return "provider"
        if isinstance(current, QueryFailureError):
            return current.stage
        if isinstance(current, SQLGenerationError):
            return "sql_generation"
        if isinstance(current, sqlite3.Error):
            return "database_execution"
    return "infrastructure"


def _json_value(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def calculate_metrics(records):
    """Compute accuracy only over provider-independent SQL evaluations."""
    eligible = [row for row in records if row.get("accuracy_eligible")]
    matched = sum(row.get("result_match") is True for row in eligible)
    return {
        "total": len(records),
        "evaluated": len(eligible),
        "matched": matched,
        "accuracy": matched / len(eligible) * 100 if eligible else 0.0,
        "execution_successes": sum(
            row.get("execution_success") is True for row in records
        ),
        "provider_failures": sum(
            row.get("failure_type") in {"provider", "provider_configuration"}
            for row in records
        ),
        "not_attempted_quota_exhausted": sum(
            row.get("failure_type") == "not_attempted_quota_exhausted"
            for row in records
        ),
    }


def _get_attempt_metadata(error):
    for current in _failure_chain(error):
        if isinstance(current, QueryFailureError):
            return {
                "attempts": current.attempts,
                "retry_count": (
                    max(0, current.attempts - 1) + current.provider_retries
                ),
                "provider_attempts": current.provider_attempts,
                "provider_retry_count": current.provider_retries,
                "generated_sql": current.generated_sql,
            }
    return {
        "attempts": "",
        "retry_count": "",
        "provider_attempts": "",
        "provider_retry_count": "",
        "generated_sql": "",
    }


def main():
    with EVAL_SET_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if MAX_QUESTIONS is not None:
        questions = questions[:MAX_QUESTIONS]

    report = []
    quota_exhausted = False
    print(f"Running {len(questions)} evaluation questions...\n")

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        reference_sql = item["expected_sql"]
        started = time.perf_counter()
        expected_columns = []
        expected_rows = []

        if quota_exhausted:
            expected_columns, expected_rows = run_reference_sql(reference_sql)
            report.append({
                "id": question_id,
                "question": question,
                "expected_sql": reference_sql,
                "generated_sql": "",
                "expected_results": _json_value(expected_rows),
                "actual_results": "",
                "ordered_comparison": reference_order_is_significant(
                    reference_sql
                ),
                "execution_success": False,
                "accuracy_eligible": False,
                "result_match": "",
                "expected_rows": len(expected_rows),
                "actual_rows": "",
                "columns_match": "",
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "pipeline_latency_ms": "",
                "sql_generation_latency_ms": "",
                "database_execution_latency_ms": "",
                "result_formatting_latency_ms": "",
                "attempts": "",
                "retry_count": "",
                "provider_attempts": 0,
                "provider_retry_count": 0,
                "prompt_tokens": "",
                "completion_tokens": "",
                "total_tokens": "",
                "estimated_cost_usd": "",
                "failure_type": "not_attempted_quota_exhausted",
                "error": (
                    "Not attempted because the provider previously reported "
                    "an exhausted quota."
                ),
            })
            print(
                f"[SKIPPED] {question_id:02d}. {question} "
                "| provider quota already exhausted"
            )
            continue

        try:
            expected_columns, expected_rows = run_reference_sql(reference_sql)
            ordered = reference_order_is_significant(reference_sql)
            result = run_pipeline(question)

            actual_columns = result["columns"]
            actual_rows = result["rows"]
            columns_match, result_match = compare_results(
                expected_columns,
                expected_rows,
                actual_columns,
                actual_rows,
                ordered=ordered,
            )
            attempts = result.get("attempts", 0)
            retry_count = result.get("retry_count", max(0, attempts - 1))

            report.append({
                "id": question_id,
                "question": question,
                "expected_sql": reference_sql,
                "generated_sql": result.get("sql", ""),
                "expected_results": _json_value(expected_rows),
                "actual_results": _json_value(actual_rows),
                "ordered_comparison": ordered,
                "execution_success": True,
                "accuracy_eligible": True,
                "result_match": result_match,
                "expected_rows": len(expected_rows),
                "actual_rows": len(actual_rows),
                "columns_match": columns_match,
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "pipeline_latency_ms": result.get("latency_ms", ""),
                "sql_generation_latency_ms": result.get(
                    "sql_generation_latency_ms", ""
                ),
                "database_execution_latency_ms": result.get(
                    "database_execution_latency_ms", ""
                ),
                "result_formatting_latency_ms": result.get(
                    "result_formatting_latency_ms", ""
                ),
                "attempts": attempts,
                "retry_count": retry_count,
                "provider_attempts": result.get("provider_attempts", ""),
                "provider_retry_count": result.get(
                    "provider_retry_count", ""
                ),
                "prompt_tokens": result.get("prompt_tokens", ""),
                "completion_tokens": result.get("completion_tokens", ""),
                "total_tokens": result.get("total_tokens", ""),
                "estimated_cost_usd": result.get("estimated_cost_usd", ""),
                "failure_type": "",
                "error": "",
            })

            status = "PASS" if result_match else "MISMATCH"
            print(
                f"[{status}] {question_id:02d}. {question} "
                f"| rows={len(actual_rows)} "
                f"| columns_match={columns_match} "
                f"| attempts={attempts} "
                f"| latency={report[-1]['latency_ms']:.2f} ms"
            )

            if not columns_match:
                print(f"    Expected columns: {expected_columns}")
                print(f"    Actual columns:   {actual_columns}")
            elif not result_match:
                print(f"    Expected rows: {expected_rows[:3]}")
                print(f"    Actual rows:   {actual_rows[:3]}")

        except Exception as exc:
            failure_type = classify_failure(exc)
            quota_exhausted = isinstance(exc, LLMQuotaExceededError)
            provider_failure = failure_type in {
                "provider",
                "provider_configuration",
            }
            eligible = failure_type in {
                "sql_generation",
                "sql_validation",
                "database_execution",
            }
            metadata = _get_attempt_metadata(exc)
            if provider_failure:
                from app import llm

                metadata.update({
                    "provider_attempts": llm.last_usage.get(
                        "provider_attempts", ""
                    ),
                    "provider_retry_count": llm.last_usage.get(
                        "provider_retries", ""
                    ),
                    "retry_count": llm.last_usage.get(
                        "provider_retries", ""
                    ),
                })

            latency_ms = round(
                (time.perf_counter() - started) * 1000,
                2,
            )
            report.append({
                "id": question_id,
                "question": question,
                "expected_sql": reference_sql,
                "generated_sql": metadata["generated_sql"],
                "expected_results": _json_value(expected_rows),
                "actual_results": "",
                "ordered_comparison": (
                    reference_order_is_significant(reference_sql)
                    if expected_columns
                    else ""
                ),
                "execution_success": False,
                "accuracy_eligible": eligible,
                "result_match": False if eligible else "",
                "expected_rows": len(expected_rows) if expected_columns else "",
                "actual_rows": "",
                "columns_match": False if eligible else "",
                "latency_ms": latency_ms,
                "pipeline_latency_ms": getattr(
                    exc,
                    "pipeline_latency_ms",
                    "",
                ),
                "sql_generation_latency_ms": getattr(
                    exc,
                    "sql_generation_latency_ms",
                    "",
                ),
                "database_execution_latency_ms": getattr(
                    exc,
                    "database_execution_latency_ms",
                    "",
                ),
                "result_formatting_latency_ms": getattr(
                    exc,
                    "result_formatting_latency_ms",
                    "",
                ),
                "attempts": metadata["attempts"],
                "retry_count": metadata["retry_count"],
                "provider_attempts": metadata["provider_attempts"],
                "provider_retry_count": metadata["provider_retry_count"],
                "prompt_tokens": "",
                "completion_tokens": "",
                "total_tokens": "",
                "estimated_cost_usd": "",
                "failure_type": failure_type,
                "error": f"{type(exc).__name__}: {exc}",
            })
            status = "PROVIDER FAILURE" if provider_failure else "ERROR"
            print(
                f"[{status}] {question_id:02d}. {question} "
                f"| stage={failure_type} "
                f"| {type(exc).__name__}: {exc}"
            )
            if quota_exhausted:
                print(
                    "Provider reported an exhausted quota; remaining "
                    "questions will be skipped without API calls."
                )

    fieldnames = [
        "id",
        "question",
        "expected_sql",
        "generated_sql",
        "expected_results",
        "actual_results",
        "ordered_comparison",
        "execution_success",
        "accuracy_eligible",
        "result_match",
        "expected_rows",
        "actual_rows",
        "columns_match",
        "latency_ms",
        "pipeline_latency_ms",
        "sql_generation_latency_ms",
        "database_execution_latency_ms",
        "result_formatting_latency_ms",
        "attempts",
        "retry_count",
        "provider_attempts",
        "provider_retry_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "failure_type",
        "error",
    ]

    with REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report)

    metrics = calculate_metrics(report)
    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Questions in benchmark: {metrics['total']}")
    print(f"Validly evaluated:     {metrics['evaluated']}")
    print(f"Execution successes:   {metrics['execution_successes']}")
    print(f"Result matches:        {metrics['matched']}")
    print(f"Accuracy:              {metrics['accuracy']:.2f}%")
    print(f"Excluded provider failures: {metrics['provider_failures']}")
    print(
        "Not attempted after quota exhaustion: "
        f"{metrics['not_attempted_quota_exhausted']}"
    )
    print(f"Average latency:       {sum(float(row['latency_ms']) for row in report) / len(report) if report else 0:.2f} ms")
    print(f"CSV report:            {REPORT_PATH}")


if __name__ == "__main__":
    main()
