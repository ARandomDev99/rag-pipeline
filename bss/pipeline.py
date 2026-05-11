from pathlib import Path

from .chunker import chunk
from .config import CFG
from .embedder import Embedder
from .llm import LLM
from .loaders import load_path
from .memory import Memory
from .prompt import build_agent_messages, build_messages
from .retriever import Retriever
from .tools import TOOL_SPECS, ToolContext, dispatch
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
    memory: Memory | None = None,
) -> dict:
    store = FaissStore.load(index_dir)
    embedder = Embedder(CFG.embed_model)
    if embedder.dim != store.dim:
        raise RuntimeError(
            f"embedder dim {embedder.dim} != index dim {store.dim}; re-ingest"
        )
    retriever = Retriever(embedder, store, CFG.top_k, CFG.score_threshold)

    retrieval_query = question
    history_msgs: list[dict] = []
    if memory is not None:
        hint = memory.retrieval_hint(CFG.history_query_chars)
        if hint:
            retrieval_query = f"{question}\n{hint}"
        history_msgs = memory.history_messages()

    hits = retriever.retrieve(retrieval_query)

    if not hits:
        out = REFUSAL
    else:
        msgs = build_messages(question, hits, history=history_msgs)
        out = LLM().generate(msgs)

    if memory is not None:
        memory.add_user(question)
        memory.add_assistant(out)

    return {"answer": out, "hits": hits}


def _assistant_msg_to_dict(msg) -> dict:
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    return dict(msg)


def agent_answer(
    question: str,
    index_dir: Path = CFG.index_dir,
    memory: Memory | None = None,
) -> dict:
    store = FaissStore.load(index_dir)
    embedder = Embedder(CFG.embed_model)
    if embedder.dim != store.dim:
        raise RuntimeError(
            f"embedder dim {embedder.dim} != index dim {store.dim}; re-ingest"
        )
    retriever = Retriever(embedder, store, CFG.top_k, CFG.score_threshold)

    retrieval_query = question
    history_msgs: list[dict] = []
    if memory is not None:
        hint = memory.retrieval_hint(CFG.history_query_chars)
        if hint:
            retrieval_query = f"{question}\n{hint}"
        history_msgs = memory.history_messages()

    initial_hits = retriever.retrieve(retrieval_query)
    msgs = build_agent_messages(question, initial_hits, history=history_msgs)

    ctx = ToolContext(retriever=retriever, store=store, embedder=embedder)
    all_hits = list(initial_hits)
    seen = {(h.source, h.locator) for h in all_hits}

    llm = LLM()
    final_text = ""
    for _ in range(CFG.agent_max_steps):
        msg = llm.generate_tool_step(msgs, tools=TOOL_SPECS)
        msgs.append(_assistant_msg_to_dict(msg))

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            final_text = (msg.content or "").strip()
            break

        for tc in tool_calls:
            result_json, hits = dispatch(
                tc.function.name, tc.function.arguments, ctx
            )
            for h in hits:
                key = (h.source, h.locator)
                if key not in seen:
                    seen.add(key)
                    all_hits.append(h)
            msgs.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result_json}
            )

    if not final_text and all_hits:
        final_msgs = build_messages(question, all_hits, history=history_msgs)
        final_text = llm.generate(final_msgs)

    if not final_text:
        final_text = REFUSAL

    if memory is not None:
        memory.add_user(question)
        memory.add_assistant(final_text)

    return {"answer": final_text, "hits": all_hits}