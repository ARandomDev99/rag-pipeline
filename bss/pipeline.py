from pathlib import Path

from .chunker import chunk
from .config import CFG
from .embedder import Embedder
from .llm import LLM
from .loaders import load_path
from .prompt import build_messages
from .retriever import Retriever
from .vectorstore import FaissStore

REFUSAL = "I don't know based on the provided documents."


def ingest(paths: list[Path], index_dir: Path = CFG.index_dir) -> int:
    records = []
    for p in paths:
        records.extend(list(load_path(p)))
    chunks = list(chunk(records, CFG.chunk_size, CFG.chunk_overlap))
    if not chunks:
        raise RuntimeError("no chunks produced — check input paths/types")

    embedder = Embedder(CFG.embed_model)
    vectors = embedder.embed([c.text for c in chunks])
    store = FaissStore(embedder.dim)
    store.add(vectors, [vars(c) for c in chunks])
    store.save(index_dir)
    return len(chunks)


def answer(
    question: str,
    index_dir: Path = CFG.index_dir,
) -> dict:
    store = FaissStore.load(index_dir)
    embedder = Embedder(CFG.embed_model)
    if embedder.dim != store.dim:
        raise RuntimeError(
            f"embedder dim {embedder.dim} != index dim {store.dim}; re-ingest"
        )
    retriever = Retriever(embedder, store, CFG.top_k, CFG.score_threshold)

    retrieval_query = question

    hits = retriever.retrieve(retrieval_query)

    if not hits:
        out = REFUSAL
    else:
        msgs = build_messages(question, hits)
        out = LLM().generate(msgs)

    return {"answer": out, "hits": hits}