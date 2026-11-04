import trafilatura

MIN_CHARS = 500

def extract_text(html: str) -> str | None:
    text = trafilatura.extract(html)
    if not text:
        return None
    stripped = text.strip()
    if len(stripped) < MIN_CHARS:
        return None
    return stripped






