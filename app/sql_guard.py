import sqlglot
from sqlglot import exp

def validate_sql(sql: str) -> str:
    sql = sql.strip()

    # Accept an optional trailing semicolon, but reject
    # multiple SQL statements.
    statements = sqlglot.parse(sql, read="sqlite")

    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Exactly one SQL statement is required")

    query = statements[0]

    # Only a SELECT query is allowed, including SELECT-based CTEs.
    if not isinstance(query, exp.Select):
        raise ValueError("Only SELECT queries are allowed")

    # Reject write operations even if they appear inside a query.
    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.Command,
    )

    for node in query.walk():
        if isinstance(node, forbidden):
            raise ValueError("Unsafe SQL operation detected")

    # Add a limit if missing. Never allow more than 100 rows.
    limit = query.args.get("limit")

    if limit is None:
        query = query.limit(100)
    else:
        limit_value = limit.expression

        if isinstance(limit_value, exp.Literal) and limit_value.is_int:
            if not 0 <= int(limit_value.this) <= 100:
                query.set(
                    "limit",
                    exp.Limit(expression=exp.Literal.number(100)),
                )
        elif isinstance(limit_value, exp.Neg) and isinstance(
            limit_value.this, exp.Literal
        ) and limit_value.this.is_int:
            query.set(
                "limit",
                exp.Limit(expression=exp.Literal.number(100)),
            )
        else:
            raise ValueError("LIMIT must be a non-negative integer literal")

    return query.sql(dialect="sqlite")