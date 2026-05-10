import csv
from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader


class Record:
    def __init__(self, text: str, source: str, locator: str):
        self.text = text
        self.source = source
        self.locator = locator


def load_pdf(path: Path) -> Iterator[Record]:
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield Record(text, str(path), f"page={i}")


def load_txt(path: Path) -> Iterator[Record]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if text:
        yield Record(text, str(path), "whole")


def load_csv(path: Path) -> Iterator[Record]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            if text:
                yield Record(text, str(path), f"row={i}")


_DISPATCH = {".pdf": load_pdf, ".txt": load_txt, ".csv": load_csv}


def load_path(path: Path) -> Iterator[Record]:
    if path.is_file():
        fn = _DISPATCH.get(path.suffix.lower())
        if fn is None:
            return
        yield from fn(path)
        return
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.suffix.lower() in _DISPATCH:
            yield from _DISPATCH[child.suffix.lower()](child)