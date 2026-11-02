import psycopg
import pytest

from app.config import settings


@pytest.fixture
def conn():
    """A connection whose changes are rolled back after each test."""
    try:
        connection = psycopg.connect(settings.database_url, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("Postgres not available (run `docker compose up -d db`)")
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()
