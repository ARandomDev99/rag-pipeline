import argparse
import sys
from pathlib import Path

from .pipeline import answer, ingest


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
    res = answer(args.question, Path(args.index_dir))
    _print_result(res)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bss")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="index files/dirs")
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--index-dir", default="storage")
    pi.set_defaults(func=_cmd_ingest)

    pq = sub.add_parser("query", help="ask a question")
    pq.add_argument("question")
    pq.add_argument("--index-dir", default="storage")
    pq.set_defaults(func=_cmd_query)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())