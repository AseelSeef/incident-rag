from datetime import date

import httpx
import pytest

from app.ingestion import run
from app.ingestion.run import Report, ingest, print_report
from app.ingestion.sources import Source

PARAGRAPHS = [
    (
        "At 14:02 UTC a configuration change was rolled out to the primary database cluster. "
        "Within minutes, connection pools were exhausted and API error rates climbed sharply."
    ),
    (
        "The on-call engineer paged the database team, who identified the faulty change and "
        "began a rollback while traffic was shifted to a secondary region."
    ),
    (
        "Error rates returned to normal at 14:47 UTC. Some customers saw delayed webhooks for "
        "a further twenty minutes while queued events were replayed."
    ),
    (
        "The root cause was a missing validation step in the deploy pipeline, which allowed a "
        "pool size of zero to reach production without review."
    ),
    (
        "Action items: add schema validation for database config, require a canary stage for "
        "all cluster changes, and alert on connection pool saturation."
    ),
]
GOOD_HTML = (
    "<html><body><article><h1>Outage postmortem</h1>"
    + "".join(f"<p>{p}</p>" for p in PARAGRAPHS)
    + "</article></body></html>"
)
SHORT_HTML = "<html><body><p>Brief note.</p></body></html>"

GOOD = "https://ingest.test/good"
GOOD_2 = "https://ingest.test/good-2"
SHORT = "https://ingest.test/short"
MISSING = "https://ingest.test/missing"


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path in ("/good", "/good-2"):
        return httpx.Response(200, text=GOOD_HTML)
    if path == "/short":
        return httpx.Response(200, text=SHORT_HTML)
    return httpx.Response(404)


@pytest.fixture
def client():
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        yield c


@pytest.fixture
def db(conn, monkeypatch):
    """The rollback connection, with commit disabled so ingest's commits are undone too."""
    commits = []
    monkeypatch.setattr(conn, "commit", lambda: commits.append(1))
    conn.commits = commits
    return conn


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(run.time, "sleep", sleeps.append)
    return sleeps


def src(url: str) -> Source:
    return Source(title="Outage", company="Acme", url=url)


def stored_content(conn, url: str):
    row = conn.execute("SELECT content FROM documents WHERE source_url = %s", (url,)).fetchone()
    return row[0] if row else None


def test_stores_good_page(db, client):
    report = ingest(db, client, [src(GOOD)])

    assert report.ok == [GOOD]
    assert report.failed == []
    assert "connection pools were exhausted" in stored_content(db, GOOD)


def test_commits_after_each_document(db, client):
    ingest(db, client, [src(GOOD), src(GOOD_2)])

    assert len(db.commits) == 2


def test_fetch_error_is_recorded_and_not_stored(db, client):
    report = ingest(db, client, [src(MISSING)])

    assert report.ok == []
    assert [url for url, _ in report.failed] == [MISSING]
    assert "404" in report.failed[0][1]
    assert stored_content(db, MISSING) is None


def test_short_page_is_recorded_and_not_stored(db, client):
    report = ingest(db, client, [src(SHORT)])

    assert report.failed == [(SHORT, "no usable text extracted")]
    assert stored_content(db, SHORT) is None


def test_failure_does_not_stop_later_sources(db, client):
    report = ingest(db, client, [src(MISSING), src(SHORT), src(GOOD)])

    assert report.ok == [GOOD]
    assert [url for url, _ in report.failed] == [MISSING, SHORT]


def test_existing_url_is_skipped(db, client):
    ingest(db, client, [src(GOOD)])

    report = ingest(db, client, [src(GOOD)])

    assert report.ok == []
    assert report.skipped == [GOOD]


def test_skipped_url_is_not_fetched(db):
    requested = []

    def recording_handler(request):
        requested.append(str(request.url))
        return handler(request)

    with httpx.Client(transport=httpx.MockTransport(recording_handler)) as c:
        ingest(db, c, [src(GOOD)])
        requested.clear()
        ingest(db, c, [src(GOOD)])

    assert requested == []


def test_changed_date_of_existing_url_is_updated_without_fetching(db):
    requested = []

    def recording_handler(request):
        requested.append(str(request.url))
        return handler(request)

    with httpx.Client(transport=httpx.MockTransport(recording_handler)) as c:
        ingest(db, c, [src(GOOD)])
        requested.clear()
        dated = Source(title="Outage", company="Acme", url=GOOD, date=date(2021, 10, 4))
        report = ingest(db, c, [dated])

    assert requested == []  # not downloaded again
    assert report.skipped == [GOOD]
    assert report.updated == [GOOD]
    (published,) = db.execute(
        "SELECT published FROM documents WHERE source_url = %s", (GOOD,)
    ).fetchone()
    assert published == date(2021, 10, 4)


def test_unchanged_existing_url_is_not_reported_as_updated(db, client):
    ingest(db, client, [src(GOOD)])
    commits_before = len(db.commits)

    report = ingest(db, client, [src(GOOD)])

    assert report.updated == []
    assert len(db.commits) == commits_before  # nothing to save


def test_force_refetches_existing_url(db, client):
    ingest(db, client, [src(GOOD)])

    report = ingest(db, client, [src(GOOD)], force=True)

    assert report.ok == [GOOD]
    assert report.skipped == []


def test_failed_url_is_retried_next_run(db, client):
    ingest(db, client, [src(MISSING)])

    report = ingest(db, client, [src(MISSING)])

    assert report.skipped == []
    assert [url for url, _ in report.failed] == [MISSING]


def test_sleeps_between_downloads_only(db, client, no_sleep):
    ingest(db, client, [src(GOOD), src(SHORT), src(GOOD_2)], delay=2.0)

    assert no_sleep == [2.0, 2.0]


def test_no_sleep_for_skipped_sources(db, client, no_sleep):
    ingest(db, client, [src(GOOD)])
    no_sleep.clear()

    ingest(db, client, [src(GOOD), src(GOOD_2)], delay=2.0)

    assert no_sleep == []


def test_zero_delay_never_sleeps(db, client, no_sleep):
    ingest(db, client, [src(GOOD), src(GOOD_2)], delay=0)

    assert no_sleep == []


def test_print_report(capsys):
    report = Report(ok=["a"], skipped=["b", "c"], updated=["c"], failed=[("d", "boom")])

    print_report(report)

    out = capsys.readouterr().out
    assert "Stored:  1" in out
    assert "Skipped: 2" in out
    assert "Updated: 1" in out
    assert "Failed:  1" in out
    assert "d" in out and "boom" in out


def test_print_report_hides_updated_when_none(capsys):
    print_report(Report(ok=["a"]))

    assert "Updated" not in capsys.readouterr().out
