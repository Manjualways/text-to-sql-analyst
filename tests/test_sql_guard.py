import pytest

from app.sql_guard import validate_sql

def test_select_is_allowed():
    sql = validate_sql("SELECT Name FROM Artist")
    assert sql.upper().startswith("SELECT")
    assert "LIMIT 100" in sql.upper()

def test_existing_limit_is_preserved():
    sql = validate_sql("SELECT Name FROM Artist LIMIT 10")
    assert "LIMIT 10" in sql.upper()

def test_large_limit_is_capped():
    sql = validate_sql("SELECT Name FROM Artist LIMIT 500")
    assert "LIMIT 100" in sql.upper()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT Name FROM Artist LIMIT -1",
        "SELECT Name FROM Artist LIMIT 101",
    ],
)
def test_unbounded_or_excessive_limit_is_capped(sql):
    assert "LIMIT 100" in validate_sql(sql).upper()


def test_non_literal_limit_is_rejected():
    with pytest.raises(ValueError, match="LIMIT"):
        validate_sql("SELECT Name FROM Artist LIMIT 10 + 1")

@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM Artist",
        "UPDATE Artist SET Name = 'X'",
        "INSERT INTO Artist (Name) VALUES ('X')",
        "DROP TABLE Artist",
        "ALTER TABLE Artist ADD COLUMN test TEXT",
    ],
)
def test_write_operations_are_rejected(sql):
    with pytest.raises(ValueError):
        validate_sql(sql)

def test_multiple_statements_are_rejected():
    with pytest.raises(ValueError):
        validate_sql("SELECT Name FROM Artist; DROP TABLE Artist")

def test_cte_select_is_allowed():
    sql = validate_sql(
        "WITH artists AS (SELECT Name FROM Artist) "
        "SELECT Name FROM artists"
    )
    assert "LIMIT 100" in sql.upper()