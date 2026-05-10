from collections.abc import Iterable, Iterator

from .loaders import Record


def _split(text: str, size: int, overlap: int) -> Iterator[tuple[int, str]]:
    assert size > 0
    assert overlap < size
    step = size - overlap
    n = len(text)
    i = 0
    while i < n:
        yield i, text[i : i + size]
        i += step


def chunk(records: Iterable[Record], size: int, overlap: int) -> Iterator[Record]:
    for r in records:
        for offset, piece in _split(r.text, size, overlap):
            piece = piece.strip()
            if piece:
                yield Record(piece, r.source, f"{r.locator};offset={offset}")
