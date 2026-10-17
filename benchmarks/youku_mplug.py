"""用 Youku-mPLUG 检索验证集做端到端基准：走完整入库流水线，再在系统里评估。

输入是一个 manifest（JSONL，每行 {"path": 本地视频, "caption": 中文画面描述}），可由 Youku-mPLUG 的
retrieval_val.csv 下载后生成。视频入库时 *不* 带标题，评测时用描述做查询、视频本身为答案，
所以测的是“文字 → 画面”的能力，不会因为字面匹配而虚高。

  python benchmarks/youku_mplug.py --manifest manifest_val.jsonl --n 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omniembed import evaluation  # noqa: E402
from omniembed.config import CONFIG  # noqa: E402
from omniembed.pipeline import Pipeline  # noqa: E402
from omniembed.retrieval import Index  # noqa: E402
from omniembed.store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.manifest).open(encoding="utf-8")][:a.n]
    CONFIG.ensure_dirs()
    st = Store(CONFIG.db_path)
    pipe, idx = Pipeline(st), Index(st)
    t0 = time.time()
    gold = {}
    for r in rows:
        vid = pipe.submit_path(Path(r["path"]), extra={"benchmark": "youku_mplug_val"})
        gold[vid] = r["caption"]
    while (vid := st.next_queued()):
        pipe.process(vid)
    ingest_s = time.time() - t0
    vids = [st.get_video(v) for v in gold]
    ok = [v for v in vids if v["status"] == "ready"]
    stage = {k: round(sum((v["timings"] or {}).get(k, 0) for v in ok) / max(len(ok), 1), 3)
             for k in ("download_s", "probe_s", "extract_s", "embed_s")}
    print(f"ingested {len(ok)}/{len(rows)} in {ingest_s:.0f}s ({len(ok) / ingest_s:.2f} videos/s); avg stage seconds {stage}")
    items = [{"query": gold[v["id"]], "gold": [v["id"]]} for v in ok]
    st.put_evalset("youku_mplug_val", items, "Youku-mPLUG 验证集：中文画面描述 → 视频（入库时不带标题）")
    res = []
    for mode in ("visual", "dense", "hybrid"):
        for dim in (768, 256):
            r = evaluation.evaluate(idx, items, mode, dim, keep_cases=0)
            r.pop("failures")
            res.append(r)
            print(f"{mode:7s} dim={dim:<4d} R@1 {r['R@1']:5.1f}  R@5 {r['R@5']:5.1f}  R@10 {r['R@10']:5.1f}  nDCG@10 {r['nDCG@10']:5.1f}")
    out = {"n": len(ok), "failed": len(rows) - len(ok), "ingest_s": round(ingest_s, 1),
           "videos_per_s": round(len(ok) / ingest_s, 2), "avg_stage_s": stage, "results": res}
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
