from .embedder import Embedder
from .vectorstore import FaissStore, Hit


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: FaissStore,
        top_k: int,
        score_threshold: float,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.threshold = score_threshold

    def retrieve(self, query: str) -> list[Hit]:
        vec = self.embedder.embed([query])
        hits = self.store.search(vec, self.top_k)
        return [h for h in hits if h.score >= self.threshold]