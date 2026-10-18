#!/usr/bin/env python3
"""Independent local code search. No local model or GPU dependency.

python scripts/run_code_search.py --root ~/AiCode --sync
python scripts/run_code_search.py --root ~/AiCode --scan-only
Restart with --sync to resume missing vectors; completed chunks are reused.
"""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home()/"AiCode")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/code-search")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--endpoint-file", type=Path, default=ROOT/"data/autodl/endpoint.json")
    parser.add_argument("--key-file", type=Path, default=ROOT/"data/autodl/inference.key")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--sync", action="store_true", help="启动后扫描并续跑向量化")
    actions.add_argument("--scan-only", action="store_true", help="启动后仅扫描，BM25无需GPU")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("源目录不存在")
    if not 1024 <= args.port <= 65535:
        parser.error("端口需为1024—65535")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.data_dir/".service.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("该数据目录的服务已启动，请使用现有网页。")
    sys.path.insert(0, str(ROOT))
    from omniembed.code_search.app import create_app
    import uvicorn
    app = create_app(root=args.root, data_dir=args.data_dir, endpoint_file=args.endpoint_file, key_file=args.key_file)
    if args.sync or args.scan_only:
        app.state.engine.start_job("sync" if args.sync else "scan")
    print(f"OmniIndex: http://127.0.0.1:{args.port} · {args.root.resolve()}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
