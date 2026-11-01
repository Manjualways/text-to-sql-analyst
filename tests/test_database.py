"""Tests for database access."""

from app import database


def test_chinook_database_connection_is_read_only():
    with database.get_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM Artist").fetchone()[0] > 0
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1


def test_schema_includes_foreign_key_relationships():
    schema = {item["table"]: item for item in database.get_schema()}

    assert any(
        "Artist.ArtistId -> Album.ArtistId" in relationship
        or "Album.ArtistId -> Artist.ArtistId" in relationship
        for item in schema.values()
        for relationship in item["relationships"]
    )


def test_missing_database_has_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "missing.sqlite")

    try:
        database.get_connection()
    except database.DatabaseNotFoundError as exc:
        assert "Database not found" in str(exc)
    else:
        raise AssertionError("Expected a missing database error")
