from datetime import date

import pytest

from app.ingestion.sources import Source, load_sources

HEADER = "title,company,date,url\n"


def write_csv(tmp_path, content: str):
    path = tmp_path / "sources.csv"
    path.write_text(HEADER + content, encoding="utf-8")
    return path


def test_parses_all_fields(tmp_path):
    path = write_csv(tmp_path, "ChatGPT Outage,OpenAI,2023-03-16,https://example.com/a\n")

    sources = load_sources(path)

    assert sources == [
        Source(
            title="ChatGPT Outage",
            company="OpenAI",
            url="https://example.com/a",
            date=date(2023, 3, 16),
        )
    ]


def test_header_row_is_not_returned_as_source(tmp_path):
    path = write_csv(tmp_path, "Outage,Acme,2024-01-01,https://example.com/a\n")

    sources = load_sources(path)

    assert len(sources) == 1
    assert sources[0].title == "Outage"


def test_quoted_field_with_comma_is_kept_whole(tmp_path):
    path = write_csv(
        tmp_path,
        '"High error rates for ChatGPT, APIs, and Sora",OpenAI,2024-12-26,https://example.com/a\n',
    )

    sources = load_sources(path)

    assert sources[0].title == "High error rates for ChatGPT, APIs, and Sora"


def test_empty_date_becomes_none(tmp_path):
    path = write_csv(tmp_path, "AWS Seoul outage,Amazon,,https://example.com/a\n")

    sources = load_sources(path)

    assert sources[0].date is None


def test_rows_with_empty_url_are_skipped(tmp_path):
    path = write_csv(
        tmp_path,
        "No url,Acme,2024-01-01,\n"
        "Has url,Acme,2024-01-02,https://example.com/a\n",
    )

    sources = load_sources(path)

    assert [s.title for s in sources] == ["Has url"]

def test_trailing_blank_lines_are_ignored(tmp_path):
    path = write_csv(
        tmp_path,
        "Outage,Acme,2024-01-01,https://example.com/a\n"
        "\n"
        ",,,\n"
        "   \n",
    )

    sources = load_sources(path)

    assert [s.title for s in sources] == ["Outage"]


def test_duplicate_urls_are_skipped_and_first_one_wins(tmp_path):
    path = write_csv(
        tmp_path,
        "First,Acme,2024-01-01,https://example.com/a\n"
        "Second,Acme,2024-02-01,https://example.com/a\n"
        "Other,Acme,2024-03-01,https://example.com/b\n",
    )

    sources = load_sources(path)

    assert [s.title for s in sources] == ["First", "Other"]
    assert sources[0].date == date(2024, 1, 1)


def test_whitespace_is_stripped(tmp_path):
    path = write_csv(tmp_path, "  Outage  , Acme ,2024-01-01, https://example.com/a \n")

    sources = load_sources(path)

    assert sources[0].title == "Outage"
    assert sources[0].company == "Acme"
    assert sources[0].url == "https://example.com/a"


def test_duplicate_detection_ignores_whitespace(tmp_path):
    path = write_csv(
        tmp_path,
        "First,Acme,2024-01-01,https://example.com/a\n"
        "Second,Acme,2024-02-01, https://example.com/a \n",
    )

    sources = load_sources(path)

    assert len(sources) == 1


def test_invalid_date_raises_value_error_with_line_number(tmp_path):
    path = write_csv(
        tmp_path,
        "Good,Acme,2024-01-01,https://example.com/a\n"
        "Bad,Acme,not-a-date,https://example.com/b\n",
    )

    with pytest.raises(ValueError, match="'not-a-date' on line 3"):
        load_sources(path)


def test_row_with_invalid_date_but_duplicate_url_is_not_checked(tmp_path):
    path = write_csv(
        tmp_path,
        "First,Acme,2024-01-01,https://example.com/a\n"
        "Second,Acme,not-a-date,https://example.com/a\n",
    )

    sources = load_sources(path)

    assert len(sources) == 1


def test_empty_file_with_only_header_returns_empty_list(tmp_path):
    path = write_csv(tmp_path, "")

    assert load_sources(path) == []


def test_accepts_string_path(tmp_path):
    path = write_csv(tmp_path, "Outage,Acme,2024-01-01,https://example.com/a\n")

    sources = load_sources(str(path))

    assert len(sources) == 1