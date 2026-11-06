import os

import psycopg
import pytest

from app.config import settings


@pytest.fixture
def conn():
    """A connection whose changes are rolled back after each test.

    Locally, database tests are skipped when Postgres isn't running. In CI (GitHub Actions
    sets CI=true) they fail instead: a broken database service must not turn into a green
    run with the database tests silently skipped.
    """
    try:
        connection = psycopg.connect(settings.database_url, connect_timeout=3)
    except psycopg.OperationalError as e:
        if os.environ.get("CI"):
            pytest.fail(f"Postgres not available in CI: {e}")
        pytest.skip("Postgres not available (run `docker compose up -d db`)")
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()
