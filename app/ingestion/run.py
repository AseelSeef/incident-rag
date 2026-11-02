import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psycopg

from app.config import settings
from app.ingestion.db import existing_urls, upsert_document
from app.ingestion.extract import extract_text
from app.ingestion.fetch import FetchError, fetch_html, make_client
from app.ingestion.sources import Source, load_sources

log = logging.getLogger(__name__)

DEFAULT_SOURCES = Path(__file__).resolve().parents[2] / "data" / "sources.csv"


@dataclass
class Report:
    ok: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def ingest(
    conn: psycopg.Connection,
    client: httpx.Client,
    sources: list[Source],
    force: bool = False,
    delay: float = 1.0,
) -> Report:
    """Fetch, extract and store each source. Commits after every document.

    URLs already in the database are skipped unless `force` is set.
    `delay` is the pause in seconds between downloads, to be polite to the sites.
    """
    report = Report()
    done = set() if force else existing_urls(conn)
    fetched_any = False

    for source in sources:
        if source.url in done:
            report.skipped.append(source.url)
            continue

        if fetched_any and delay > 0:
            time.sleep(delay)
        fetched_any = True

        try:
            html = fetch_html(client, source.url)
        except FetchError as e:
            log.warning("%s", e)
            report.failed.append((source.url, str(e)))
            continue

        text = extract_text(html)
        if text is None:
            log.warning("no usable text extracted from %s", source.url)
            report.failed.append((source.url, "no usable text extracted"))
            continue

        doc_id = upsert_document(conn, source, text)
        conn.commit()

        log.info("stored %s as document %s (%s chars)", source.url, doc_id, len(text))
        report.ok.append(source.url)

    return report


def print_report(report: Report) -> None:
    print(f"\nStored:  {len(report.ok)}")
    print(f"Skipped: {len(report.skipped)} (already in database)")
    print(f"Failed:  {len(report.failed)}")
    for url, reason in report.failed:
        print(f"  - {url}\n      {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and store incident postmortems.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="CSV of sources")
    parser.add_argument("--force", action="store_true", help="re-fetch URLs already stored")
    parser.add_argument("--limit", type=int, help="only process the first N sources")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between downloads")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    sources = load_sources(args.sources)
    if args.limit is not None:
        sources = sources[: args.limit]

    with psycopg.connect(settings.database_url) as conn, make_client() as client:
        report = ingest(conn, client, sources, force=args.force, delay=args.delay)

    print_report(report)
    if report.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
