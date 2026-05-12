# Agentic RAG Pipeline

---

## 1. Setup

Requires Python ≥ 3.14. Dependencies are managed with [`uv`](https://docs.astral.sh/uv/)
(locked in `uv.lock`).

```bash
uv sync
```

### API key

This project can work with any OpenAI-compatible API.

1. Modify `bss/config.py` to adjust relevant fields (such as `openai_base_url` and `llm_model`) as per your requirements.
2. Write your API key into a `.env` file in the project root:
```bash
echo 'OPENAI_API_KEY=sk-or-...' > .env
```

### Embedding model

The embedder uses `BAAI/bge-small-en-v1.5`. Before first use, pull it into your local HuggingFace cache:

```bash
uv run python -m bss download
```

---

## 2. Usage

```bash
# One-time: download the embedding model into cache
uv run python -m bss download

# Index a file or a directory (recursively).
# Supports pdf / txt / csv
uv run python -m bss ingest docs

# One-shot question (no memory)
uv run python -m bss query "What was the sample size in the second study?"

# One-shot, with the agentic tool-calling loop
uv run python -m bss query --agent "List the documents you have indexed."

# Interactive REPL with conversation memory
uv run python -m bss chat

# REPL + agent loop
uv run python -m bss chat --agent
```

`--index-dir <path>` overrides the default `storage/` for any subcommand.

All answers are followed by a list of every
`source (locator) score=...` the model was shown. If retrieval finds nothing above the
similarity threshold (and recovery also fails in agent mode), the system will refuse to answer.

---

## 3. How hallucinations are minimized

1. **During Retrieval:** Only the top-k entries with a cosine score of at least 0.30 are given to the model as context.
   If nothing clears the threshold, the non-agent path immediately refuses to answer without ever calling the LLM.

2. **Strict Prompt:** The system prompt forbids outside knowledge, forbids extrapolating or combining facts into new claims,
   requires an inline citation on every claim,
   and mandates the **exact** refusal string `I don't know based on the provided
   documents.` when the question is unsupported. Openers such as "Based on the documents…",
   "According to…" are explicitly forbidden so that attribution is done by citations.

3. **Deterministic Decoding:** `temperature=0.0` on every LLM call.

---

## 4. Architecture & workflow

| Module           | Responsibility                                                                                                   |
|------------------|------------------------------------------------------------------------------------------------------------------|
| `config.py`      | Contains all customizable parameters (chunk size, top-k, threshold, models, paths)                               |
| `loaders.py`     | Converts files to `Record`s<br/>PDF => 1 record/page<br/>CSV => 1 record/row<br/>Text => 1 record/file           |
| `chunker.py`     | Fixed-size sliding window over each record's text with attached data about the source                            |
| `embedder.py`    | wrapper for `SentenceTransformer`                                                                                |
| `vectorstore.py` | `FaissStore` around `IndexFlatIP`                                                                                |
| `retriever.py`   | Embeds the query, retrieves the k nearest chunks from the FAISS index, then drop hits below specific threshold.  |
| `prompt.py`      | Contains the system prompt and a function that builds the complete prompt by prepending retrieval results        |
| `llm.py`         | OpenAI-compatible client                                                                                         |
| `memory.py`      | Maintains a rolling window of recent turns and provides hints for retrieval                                      |
| `tools.py`       | Contains the four agent tools and handles their dispatch                                                         |
| `pipeline.py`    | Orchestration: `ingest` (load → chunk → embed → index), `answer` (non-agent), and `agent_answer` (tool loop)     |
| `__main__.py`    | Parses CLI options and handles the user interface                                                                |

### Conversation memory (`chat`)

`Memory` keeps a fixed number of recent turns. The number of turns can be adjusted using `CFG.memory_window_turns`.
`history_messages()` is injected between the system and user messages so the LLM can resolve references to previous messages.
`retrieval_hint()` appends the last question and a snippet of the last LLM response to the embedding query, so that
follow-up questions can be answered by retrieval even if they lack standalone context.

### Agentic mode (`--agent`)

`agent_answer` runs the same initial steps as `answer` (load, embed, initial retrieve,
build messages with memory hint), then enters a tool-calling loop
(with maximum number of steps determined by `CFG.agent_max_steps`).

| Function         | Parameters                                 | Purpose                                                                                                                                                                                                                                                     |
|------------------|--------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `list_sources`   | -                                          | Returns a sorted list of distinct sources in index.                                                                                                                                                                                                         |
| `retrieve`       | `query`, `source=None`                     | Embed-and-search. If `source` is `None`, returns results from all sources.                                                                                                                                                                                  |
| `keyword_search` | `pattern`, `source=None`                   | Case-insensitive substring scan over chunk text.                                                                                                                                                                                                            |
| `get_neighbors`  | `source`, `locator`, `before=1`, `after=1` | Retrieves chunks adjacent to a `(source, locator)` within the same base locator (same page for PDFs, row for CSVs, file for TXT) - adjacency never crosses page/record boundaries. `before` and `after` are capped at `3` to avoid context from blowing up. |

The tool-calling loop ends when the LLM responds with no tool calls (that response is the final answer) or when it has run `CFG.agent_max_steps` iterations. After the loop:
- If a final answer was produced, it is returned.
- Otherwise, if any context was gathered (initial retrieval or tool calls), the LLM is called once more without tools to force a final answer over everything collected.
- If no context was gathered at all, the system returns the exact refusal string.
---
AI was used to assist with debugging and generating system prompts.