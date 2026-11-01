from app.response_formatter import format_results


def test_format_results_uses_deterministic_summary_and_bar_chart():
    result = format_results(
        question="Revenue by country",
        columns=["BillingCountry", "Revenue"],
        rows=[
            {"BillingCountry": "USA", "Revenue": 523.06},
            {"BillingCountry": "Canada", "Revenue": 303.96},
        ],
    )

    assert result == {
        "summary": (
            "Returned 2 rows. First result: BillingCountry=USA, Revenue=523.06."
        ),
        "chart_type": "bar",
    }


def test_format_results_handles_empty_and_scalar_results():
    assert format_results("count", ["ArtistCount"], []) == {
        "summary": "The query completed successfully but returned no rows.",
        "chart_type": "table",
    }
    assert format_results(
        "count",
        ["ArtistCount"],
        [{"ArtistCount": 275}],
    ) == {
        "summary": "Result: ArtistCount = 275.",
        "chart_type": "table",
    }
