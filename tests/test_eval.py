import csv
import json
from pathlib import Path

from app.llm import LLMQuotaExceededError
from app.pipeline import QueryFailureError
from eval import eval as evaluator


def test_result_comparison_preserves_order_when_reference_requires_it():
    expected_columns = ["Name"]
    expected_rows = [("AC/DC",), ("Aaron Copland",)]
    actual_rows = [{"Name": "Aaron Copland"}, {"Name": "AC/DC"}]

    ordered_columns_match, ordered_match = evaluator.compare_results(
        expected_columns,
        expected_rows,
        expected_columns,
        actual_rows,
        ordered=True,
    )
    unordered_columns_match, unordered_match = evaluator.compare_results(
        expected_columns,
        expected_rows,
        expected_columns,
        actual_rows,
        ordered=False,
    )

    assert ordered_columns_match and not ordered_match
    assert unordered_columns_match and unordered_match


def test_result_comparison_requires_exact_aliases_and_values():
    columns_match, result_match = evaluator.compare_results(
        ["AverageInvoiceTotal"],
        [(5.65,)],
        ["AVG(Total)"],
        [(5.65,)],
    )
    assert not columns_match
    assert not result_match

    columns_match, result_match = evaluator.compare_results(
        ["AverageInvoiceTotal"],
        [(5.65,)],
        ["AverageInvoiceTotal"],
        [(5.64,)],
    )
    assert columns_match
    assert not result_match

    columns_match, result_match = evaluator.compare_results(
        ["Value"],
        [(1.000000001,)],
        ["Value"],
        [(1.000000002,)],
    )
    assert columns_match
    assert not result_match


def test_benchmark_reference_order_is_detected():
    assert evaluator.reference_order_is_significant(
        "SELECT Name FROM Artist ORDER BY Name"
    )
    assert not evaluator.reference_order_is_significant(
        "SELECT COUNT(*) AS ArtistCount FROM Artist"
    )


def test_known_benchmark_references_cover_aliases_aggregation_and_ties():
    questions = json.loads(
        (Path(__file__).resolve().parents[1] / "eval" / "eval_set.json")
        .read_text(encoding="utf-8")
    )
    references = {
        question["id"]: question["expected_sql"]
        for question in questions
    }

    assert "AS TrackName" in references[10]
    assert "AS GenreName" in references[10]
    assert "ORDER BY Track.Name" in references[10]
    assert "GROUP BY Artist.ArtistId, Artist.Name" in references[11]
    assert "ORDER BY Artist.Name" in references[11]
    assert "AS TrackCount" in references[14]
    assert "ORDER BY TrackCount DESC, Genre.Name" in references[14]
    assert "ROUND(AVG(Total), 2) AS AverageInvoiceTotal" in references[19]
    assert "GROUP BY BillingCountry" in references[30]
    assert "ORDER BY Revenue DESC, BillingCountry" in references[30]


def test_provider_failures_are_excluded_from_accuracy():
    metrics = evaluator.calculate_metrics([
        {
            "accuracy_eligible": True,
            "execution_success": True,
            "result_match": True,
            "failure_type": "",
        },
        {
            "accuracy_eligible": False,
            "execution_success": False,
            "result_match": "",
            "failure_type": "provider",
        },
        {
            "accuracy_eligible": True,
            "execution_success": False,
            "result_match": False,
            "failure_type": "sql_validation",
        },
    ])

    assert metrics == {
        "total": 3,
        "evaluated": 2,
        "matched": 1,
        "accuracy": 50.0,
        "execution_successes": 1,
        "provider_failures": 1,
        "not_attempted_quota_exhausted": 0,
    }


def test_evaluation_stops_provider_calls_after_quota_exhaustion(
    monkeypatch,
    tmp_path,
):
    questions = [
        {
            "id": index,
            "question": f"Question {index}",
            "expected_sql": "SELECT 1 AS Value",
        }
        for index in range(1, 4)
    ]
    questions_path = tmp_path / "questions.json"
    report_path = tmp_path / "report.csv"
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    provider_calls = []

    def fail_with_quota(_question):
        provider_calls.append(True)
        raise LLMQuotaExceededError("provider quota exhausted")

    monkeypatch.setattr(evaluator, "EVAL_SET_PATH", questions_path)
    monkeypatch.setattr(evaluator, "REPORT_PATH", report_path)
    monkeypatch.setattr(evaluator, "MAX_QUESTIONS", None)
    monkeypatch.setattr(
        evaluator,
        "run_reference_sql",
        lambda _sql: (["Value"], [(1,)]),
    )
    monkeypatch.setattr(evaluator, "run_pipeline", fail_with_quota)

    evaluator.main()

    with report_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(provider_calls) == 1
    assert [row["failure_type"] for row in rows] == [
        "provider",
        "not_attempted_quota_exhausted",
        "not_attempted_quota_exhausted",
    ]
    assert all(row["accuracy_eligible"] == "False" for row in rows)


def test_failure_classification_keeps_provider_and_sql_errors_distinct():
    assert evaluator.classify_failure(
        LLMQuotaExceededError("provider quota exhausted")
    ) == "provider"
    assert evaluator.classify_failure(
        QueryFailureError("invalid SQL", "sql_validation", 3)
    ) == "sql_validation"
