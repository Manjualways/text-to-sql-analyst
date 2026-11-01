from datetime import date, datetime
from numbers import Number


def _row_values(columns: list, row) -> list:
    if isinstance(row, dict):
        return [row.get(column) for column in columns]
    return list(row)


def format_results(question: str, columns: list, rows: list) -> dict:
    """Create a factual summary and chart suggestion without another LLM call."""
    if not rows:
        return {
            "summary": "The query completed successfully but returned no rows.",
            "chart_type": "table",
        }

    sample = _row_values(columns, rows[0])
    numeric_indices = [
        index
        for index, value in enumerate(sample)
        if isinstance(value, Number) and not isinstance(value, bool)
    ]
    category_indices = [
        index
        for index, value in enumerate(sample)
        if value is not None
        and not isinstance(value, Number)
        and not isinstance(value, (date, datetime))
    ]
    date_indices = [
        index
        for index, column in enumerate(columns)
        if "date" in str(column).casefold()
        or "time" in str(column).casefold()
        or (
            index < len(sample)
            and isinstance(sample[index], (date, datetime))
        )
    ]

    chart_type = "table"
    if len(rows) > 1 and numeric_indices and category_indices:
        chart_type = (
            "line"
            if any(index in date_indices for index in category_indices)
            else "bar"
        )

    if len(rows) == 1 and len(columns) == 1:
        summary = f"Result: {columns[0]} = {sample[0]}."
    else:
        first_result = ", ".join(
            f"{column}={value if value is not None else 'NULL'}"
            for column, value in zip(columns[:3], sample[:3])
        )
        summary = (
            f"Returned {len(rows)} rows. First result: {first_result}."
        )

    if len(summary) > 240:
        summary = summary[:237].rstrip() + "..."

    return {
        "summary": summary,
        "chart_type": chart_type,
    }
