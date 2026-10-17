"""评估：在自己入库的视频上建评测集，计算 R@1/5/10、MRR@10、nDCG@10，并列出错例。

评测集格式（JSON 列表）：[{"query": "红烧肉做法", "gold": ["<video_id>", ...]}, ...]
gold 也可以写 source_ref（上传的文件名或 URL），导入时会换成 video_id，方便直接用业务日志构造。
"""
import math
import time


def resolve_gold(store, items):
    """把 gold 里的文件名 / URL 映射成 video_id；找不到的条目丢弃并计数。"""
    videos, _ = store.list_videos(limit=10**9)
    by_ref = {}
    for v in videos:
        by_ref.setdefault(v["source_ref"], v["id"])
        by_ref.setdefault(v["source_ref"].replace("\\", "/").rsplit("/", 1)[-1], v["id"])
    ids = {v["id"] for v in videos}
    out, dropped = [], 0
    for it in items:
        q = str(it.get("query", "")).strip()
        gold = [g if g in ids else by_ref.get(g) for g in it.get("gold", [])]
        gold = [g for g in gold if g]
        if q and gold:
            out.append({"query": q, "gold": gold})
        else:
            dropped += 1
    return out, dropped


def from_titles(store, field="title", limit=500):
    """自动评测集：以每条视频的标题或描述做查询、该视频本身为标准答案。
    用于 visual 模式时衡量“画面能否对上文字”；用于 hybrid/text 模式会因为字面相同而虚高。"""
    videos, _ = store.list_videos(status="ready", limit=limit)
    return [{"query": v[field], "gold": [v["id"]]} for v in videos if (v.get(field) or "").strip()]


def evaluate(index, items, mode="hybrid", dim=768, ks=(1, 5, 10), keep_cases=20):
    t0 = time.perf_counter()
    hits = {k: 0 for k in ks}
    mrr = ndcg = 0.0
    cases = []
    for it in items:
        res, _ = index.search(it["query"], mode=mode, dim=dim, k=max(max(ks), 10))
        ranked = [r["video_id"] for r in res]
        gold = set(it["gold"])
        first = next((i + 1 for i, v in enumerate(ranked) if v in gold), None)
        for k in ks:
            hits[k] += first is not None and first <= k
        if first and first <= 10:
            mrr += 1 / first
        dcg = sum(1 / math.log2(i + 2) for i, v in enumerate(ranked[:10]) if v in gold)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
        ndcg += dcg / idcg if idcg else 0
        if first != 1 and len(cases) < keep_cases:
            cases.append({"query": it["query"], "gold": it["gold"], "gold_rank": first, "top": ranked[:5]})
    n = max(len(items), 1)
    return {"mode": mode, "dim": dim, "n": len(items),
            **{f"R@{k}": round(100 * hits[k] / n, 2) for k in ks},
            "MRR@10": round(100 * mrr / n, 2), "nDCG@10": round(100 * ndcg / n, 2),
            "elapsed_s": round(time.perf_counter() - t0, 2), "failures": cases}
