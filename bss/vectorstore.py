import json
from pathlib import Path

import faiss
import numpy as np


class Hit:
    def __init__(self, text: str, source: str, locator: str, score: float):
        self.text = text
        self.source = source
        self.locator = locator
        self.score = score


class FaissStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta: list[dict] = []

    def add(self, vectors: np.ndarray, metas: list[dict]) -> None:
        assert vectors.shape[0] == len(metas)
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)
        self.meta.extend(metas)

    def search(self, query: np.ndarray, k: int) -> list[Hit]:
        if query.ndim == 1:
            query = query[None, :]
        scores, idx = self.index.search(query, k)
        out: list[Hit] = []
        for s, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            m = self.meta[i]
            out.append(Hit(m["text"], m["source"], m["locator"], float(s)))
        return out

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir / "index.faiss"))
        (dir / "meta.json").write_text(json.dumps(self.meta), encoding="utf-8")
        (dir / "dim.txt").write_text(str(self.dim), encoding="utf-8")

    @classmethod
    def load(cls, dir: Path) -> FaissStore:
        dim = int((dir / "dim.txt").read_text().strip())
        store = cls(dim)
        store.index = faiss.read_index(str(dir / "index.faiss"))
        store.meta = json.loads((dir / "meta.json").read_text(encoding="utf-8"))
        return store

    @property
    def size(self) -> int:
        return self.index.ntotal