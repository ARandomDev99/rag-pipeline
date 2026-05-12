import argparse
import sys
from pathlib import Path

from .config import CFG
from .embedder import download_model
from .memory import Memory
from .pipeline import agent_answer, answer, ingest


def _cmd_download(args: argparse.Namespace) -> int:
    print(f"downloading {CFG.embed_model} into local HuggingFace cache ...")
    download_model(CFG.embed_model)
    print("done")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.paths]
    for p in paths:
        if not p.exists():
            print(f"path not found: {p}", file=sys.stderr)
            return 2
    n = ingest(paths, Path(args.index_dir))
    print(f"indexed {n} chunks -> {args.index_dir}")
    return 0


def _print_result(res: dict) -> None:
    print(res["answer"])
    if res["hits"]:
        print("\n--- sources ---")
        for h in res["hits"]:
            print(f"  {h.source} ({h.locator}) score={h.score:.3f}")


def _cmd_query(args: argparse.Namespace) -> int:
    fn = agent_answer if args.agent else answer
    res = fn(args.question, Path(args.index_dir))
    _print_result(res)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    fn = agent_answer if args.agent else answer
    memory = Memory()
    print("type your question, or /exit, /reset")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        if q == "/reset":
            memory = Memory()
            print("memory cleared")
            continue
        res = fn(q, Path(args.index_dir), memory=memory)
        print()
        _print_result(res)
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bss")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("download", help="fetch the embedding model into the local cache (needs network)")
    pd.set_defaults(func=_cmd_download)

    pi = sub.add_parser("ingest", help="index files/dirs")
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--index-dir", default="storage")
    pi.set_defaults(func=_cmd_ingest)

    pq = sub.add_parser("query", help="ask a question (one-shot, no memory)")
    pq.add_argument("question")
    pq.add_argument("--index-dir", default="storage")
    pq.add_argument(
        "--agent",
        action="store_true",
        help="enable tool-calling agent loop (list_sources, retrieve, keyword_search, get_neighbors)",
    )
    pq.set_defaults(func=_cmd_query)

    pc = sub.add_parser("chat", help="interactive REPL with conversation memory")
    pc.add_argument("--index-dir", default="storage")
    pc.add_argument(
        "--agent",
        action="store_true",
        help="enable tool-calling agent loop (list_sources, retrieve, keyword_search, get_neighbors)",
    )
    pc.set_defaults(func=_cmd_chat)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())