"""命令行：不开网页也能批量入库、检索、评估。

  python -m omniembed serve                       启动服务和 Web 界面
  python -m omniembed watchdog                    启动服务并守护（卡死 / 退出时自动重启）
  python -m omniembed ingest <目录|文件...>         本机文件批量入库（同步处理完再退出）
  python -m omniembed ingest-urls urls.csv         CSV / JSONL / 每行一个 URL
  python -m omniembed search "做饭" --mode visual
  python -m omniembed eval evalset.jsonl --modes hybrid,visual
  python -m omniembed stats
"""
import argparse
import json
import sys
import time
from pathlib import Path


def _ctx():
    from .config import CONFIG
    from .pipeline import Pipeline
    from .retrieval import Index
    from .store import Store
    CONFIG.ensure_dirs()
    st = Store(CONFIG.db_path)
    return CONFIG, st, Pipeline(st), Index(st)


def _drain(st, pipe, ids):
    """同步跑完队列里这些任务（不起后台线程），逐条打印结果。"""
    t0 = time.time()
    done = 0
    while True:
        vid = st.next_queued()
        if not vid:
            break
        pipe.process(vid)
        v = st.get_video(vid)
        done += 1
        msg = "ok" if v["status"] == "ready" else f"FAILED: {v['error']}"
        print(f"  [{done}/{len(ids)}] {v['source_ref']} → {msg}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; status: {st.counts()}")


def cmd_ingest(a):
    from .media import VIDEO_EXT
    cfg, st, pipe, _ = _ctx()
    files = []
    for p in map(Path, a.paths):
        if p.is_dir():
            files += sorted(x for x in p.rglob("*") if x.suffix.lower() in VIDEO_EXT and x.is_file())
        elif p.is_file():
            files.append(p)
    if a.limit:
        files = files[:a.limit]
    meta = {}
    if a.meta:  # 可选：JSONL，按文件名对应 title / description / tags
        for line in Path(a.meta).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                meta[Path(r.get("file", "")).name] = r
    print(f"found {len(files)} videos")
    ids = []
    for f in files:
        m = meta.get(f.name, {})
        ids.append(pipe.submit_path(f, m.get("title", ""), m.get("description", ""), m.get("tags"), m.get("extra")))
    _drain(st, pipe, ids)


def cmd_ingest_urls(a):
    from .server import _rows_from_manifest
    _, st, pipe, _ = _ctx()
    p = Path(a.manifest)
    rows = _rows_from_manifest(p.read_text(encoding="utf-8"), p.name)
    ids = [pipe.submit_url(r["url"], r.get("title", ""), r.get("description", ""), r.get("tags")) for r in rows if r.get("url")]
    print(f"queued {len(ids)} urls")
    _drain(st, pipe, ids)


def cmd_search(a):
    _, st, _, idx = _ctx()
    res, t = idx.search(a.query, a.mode, a.dim, a.k)
    print(f"{len(res)} results · {t}")
    for r in res:
        v = st.get_video(r["video_id"])
        print(f"{r['rank']:>3}  {r['video_id']}  {v['title'][:60]}  {r['scores']}")


def cmd_eval(a):
    from . import evaluation
    _, st, _, idx = _ctx()
    if a.evalset == "auto-title":
        items = evaluation.from_titles(st, "title")
    else:
        text = Path(a.evalset).read_text(encoding="utf-8").strip()
        rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
        items, dropped = evaluation.resolve_gold(st, rows)
        if dropped:
            print(f"dropped {dropped} items (empty query or gold not in library)")
    print(f"{len(items)} queries")
    print(f"{'mode':8s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR@10':>7s} {'nDCG@10':>8s}")
    out = []
    for m in a.modes.split(","):
        r = evaluation.evaluate(idx, items, m, a.dim)
        out.append(r)
        print(f"{m:8s} {r['R@1']:6.1f} {r['R@5']:6.1f} {r['R@10']:6.1f} {r['MRR@10']:7.1f} {r['nDCG@10']:8.1f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("saved", a.out)


def cmd_stats(_):
    cfg, st, _, idx = _ctx()
    print(json.dumps({"data_dir": str(cfg.data_dir), "status": st.counts(), "indexed": idx.size}, ensure_ascii=False, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omniembed", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="启动服务")
    p = sub.add_parser("watchdog", help="启动服务并守护，卡死或退出时自动重启")
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--fails", type=int, default=3)
    p = sub.add_parser("ingest", help="本机文件 / 目录批量入库")
    p.add_argument("paths", nargs="+")
    p.add_argument("--meta", help="JSONL：{file, title, description, tags}")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("ingest-urls", help="URL 清单批量入库")
    p.add_argument("manifest")
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--mode", default="hybrid")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("-k", type=int, default=10)
    p = sub.add_parser("eval")
    p.add_argument("evalset", help="JSON / JSONL 文件，或 auto-title")
    p.add_argument("--modes", default="hybrid,dense,visual,text,bm25")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--out")
    sub.add_parser("stats")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        from .server import main as serve
        return serve()
    if a.cmd == "watchdog":
        from .watchdog import main as wd
        return wd(["--interval", str(a.interval), "--fails", str(a.fails)])
    {"ingest": cmd_ingest, "ingest-urls": cmd_ingest_urls, "search": cmd_search, "eval": cmd_eval, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
