import sqlite3

from app.config import DB_PATH


class DatabaseNotFoundError(FileNotFoundError):
    """Raised when the configured SQLite database file is unavailable."""


def get_connection():
    if not DB_PATH.is_file():
        raise DatabaseNotFoundError(
            f"Database not found at {DB_PATH}. Check DATABASE_PATH in .env."
        )

    uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection

def get_schema():
    schema = []

    with get_connection() as conn:
        tables = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()

        for item in tables:
            table = item["name"]
            columns = conn.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()
            foreign_keys = conn.execute(
                f'PRAGMA foreign_key_list("{table}")'
            ).fetchall()

            column_info = [
                f"{col['name']} ({col['type'] or 'UNKNOWN'})"
                for col in columns
            ]
            relationships = [
                f"{table}.{foreign_key['from']} -> "
                f"{foreign_key['table']}.{foreign_key['to']}"
                for foreign_key in foreign_keys
            ]

            schema.append({
                "table": table,
                "columns": column_info,
                "relationships": relationships,
                "description": (
                    f"Table {table} contains: "
                    + ", ".join(column_info)
                    + (
                        ". Foreign-key relationships: "
                        + "; ".join(relationships)
                        if relationships
                        else ""
                    )
                ),
            })

    return schema

def execute_sql(sql: str):
    with get_connection() as conn:
        cursor = conn.execute(sql)
        columns = [item[0] for item in cursor.description or []]
        rows = [dict(row) for row in cursor.fetchall()]

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
    }