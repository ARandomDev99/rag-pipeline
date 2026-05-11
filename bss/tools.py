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

KEYWORD_SEARCH_SPEC = {
    "type": "function",
    "function": {
        "name": "keyword_search",
        "description": (
            "Case-insensitive substring scan over chunk text. Use when the "
            "user or question references an exact token that semantic "
            "retrieval would under-rank: identifiers (e.g. 'PMC7501458'), "
            "numbers, dosages, table/section labels, named entities. "
            "Optionally scope to one source. Returns up to top_k matches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "source": {
                    "type": ["string", "null"],
                    "description": (
                        "Exact source identifier from list_sources, or "
                        "null/omitted for a global scan."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
}

GET_NEIGHBORS_SPEC = {
    "type": "function",
    "function": {
        "name": "get_neighbors",
        "description": (
            "Return chunks adjacent to a given (source, locator) within the "
            "same base locator (same page for PDFs, same row for CSVs, same "
            "file for TXT). Use when a retrieved chunk appears truncated "
            "mid-paragraph and you need surrounding context to disambiguate "
            "a claim. Adjacency does NOT cross page/record boundaries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "locator": {
                    "type": "string",
                    "description": "Full locator e.g. 'page=4;offset=800'.",
                },
                "before": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                    "default": 1,
                },
                "after": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                    "default": 1,
                },
            },
            "required": ["source", "locator"],
        },
    },
}

TOOL_SPECS = [
    LIST_SOURCES_SPEC,
    RETRIEVE_SPEC,
    KEYWORD_SEARCH_SPEC,
    GET_NEIGHBORS_SPEC,
]


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


def keyword_search(
    ctx: ToolContext, pattern: str, source: str | None = None
) -> list[Hit]:
    needle = pattern.lower()
    out: list[Hit] = []
    for m in ctx.store.meta:
        if source and m["source"] != source:
            continue
        text = m["text"]
        if needle in text.lower():
            out.append(Hit(text, m["source"], m["locator"], 1.0))
            if len(out) >= CFG.top_k:
                break
    return out


def _split_locator(locator: str) -> tuple[str, int] | None:
    base, _, off = locator.rpartition(";offset=")
    if not base or not off.isdigit():
        return None
    return base, int(off)


def get_neighbors(
    ctx: ToolContext,
    source: str,
    locator: str,
    before: int = 1,
    after: int = 1,
) -> list[Hit]:
    before = max(0, min(before, 3))
    after = max(0, min(after, 3))
    parsed = _split_locator(locator)
    if parsed is None:
        return []
    base, target_off = parsed

    siblings: list[tuple[int, dict]] = []
    for m in ctx.store.meta:
        if m["source"] != source:
            continue
        p = _split_locator(m["locator"])
        if p is None or p[0] != base:
            continue
        siblings.append((p[1], m))

    siblings.sort(key=lambda x: x[0])
    offsets = [s[0] for s in siblings]
    if target_off not in offsets:
        return []
    i = offsets.index(target_off)
    lo = max(0, i - before)
    hi = min(len(siblings), i + after + 1)
    window = [s[1] for s in siblings[lo:hi] if s[0] != target_off]

    return [Hit(m["text"], m["source"], m["locator"], 1.0) for m in window]


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

    if name == "keyword_search":
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return json.dumps({"error": "pattern is required"}), []
        source = args.get("source")
        if source is not None and not isinstance(source, str):
            return json.dumps({"error": "source must be a string or null"}), []
        hits = keyword_search(ctx, pattern, source)
        return json.dumps(_hits_payload(hits)), hits

    if name == "get_neighbors":
        source = args.get("source")
        locator = args.get("locator")
        if not isinstance(source, str) or not source:
            return json.dumps({"error": "source is required"}), []
        if not isinstance(locator, str) or not locator:
            return json.dumps({"error": "locator is required"}), []
        before = args.get("before", 1)
        after = args.get("after", 1)
        if not isinstance(before, int) or not isinstance(after, int):
            return json.dumps({"error": "before/after must be integers"}), []
        hits = get_neighbors(ctx, source, locator, before, after)
        return json.dumps(_hits_payload(hits)), hits

    return json.dumps({"error": f"unknown tool: {name}"}), []
