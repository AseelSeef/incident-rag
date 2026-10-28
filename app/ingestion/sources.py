import csv
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path


@dataclass
class Source:
    title: str
    company: str
    url: str
    date: Date | None = None


def load_sources(path: str | Path) -> list[Source]:
    sources: list[Source] = []
    seen_urls: set[str] = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            raw_date = (row.get("date") or "").strip()
            try:
                parsed_date = Date.fromisoformat(raw_date) if raw_date else None
            except ValueError as e:
                raise ValueError(
                    f"Invalid date {raw_date!r} on line {reader.line_num}: {row}"
                ) from e

            sources.append(
                Source(
                    title=(row.get("title") or "").strip(),
                    company=(row.get("company") or "").strip(),
                    url=url,
                    date=parsed_date,
                )
            )
    return sources