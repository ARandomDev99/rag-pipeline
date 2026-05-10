import json
from dataclasses import dataclass

from .config import CFG
from .embedder import Embedder
from .retriever import Retriever
from .vectorstore import FaissStore, Hit

LIST_SOURCES_SPEC = {
    "type": "function",
    "function": {
        "name": "list_sources",
        "description": (
            "Return distinct source identifiers (filenames) currently in the index. "
            "Call when the user asks what documents exist, references a document by "
            "name, or before calling retrieve with a source filter."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

RETRIEVE_SPEC = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Search the index for spans relevant to a query. Optionally restrict to "
            "a single source. Use when the existing CONTEXT does not address the "
            "question, a follow-up needs a different angle, or a source-scoped "
            "lookup is required. Do not call speculatively."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "source": {
                    "type": ["string", "null"],
                    "description": (
                        "Exact source identifier from list_sources. Omit or set to "
                        "null for a global search."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_SPECS = [LIST_SOURCES_SPEC, RETRIEVE_SPEC]


@dataclass
class ToolContext:
    retriever: Retriever
    store: FaissStore
    embedder: Embedder


def unique_sources(store: FaissStore) -> list[str]:
    return sorted({m["source"] for m in store.meta})


def retrieve_with_source_filter(
    ctx: ToolContext, query: str, source: str | None
) -> list[Hit]:
    if source:
        k = CFG.top_k * CFG.agent_retrieve_overshoot
    else:
        k = CFG.top_k
    vec = ctx.embedder.embed([query])
    raw = ctx.store.search(vec, k)
    hits = [h for h in raw if h.score >= ctx.retriever.threshold]
    if source:
        hits = [h for h in hits if h.source == source]
    return hits[: CFG.top_k]


def _hits_payload(hits: list[Hit]) -> dict:
    return {
        "hits": [
            {
                "source": h.source,
                "locator": h.locator,
                "score": round(h.score, 3),
                "text": h.text,
            }
            for h in hits
        ]
    }


def dispatch(name: str, args_json: str, ctx: ToolContext) -> tuple[str, list[Hit]]:
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError:
        return json.dumps({"error": "could not parse arguments"}), []

    print(f"Called {name}")

    if name == "list_sources":
        return json.dumps({"sources": unique_sources(ctx.store)}), []

    if name == "retrieve":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return json.dumps({"error": "query is required"}), []
        source = args.get("source")
        if source is not None and not isinstance(source, str):
            return json.dumps({"error": "source must be a string or null"}), []
        hits = retrieve_with_source_filter(ctx, query, source)
        return json.dumps(_hits_payload(hits)), hits

    return json.dumps({"error": f"unknown tool: {name}"}), []
