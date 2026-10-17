"""召回：BM25 + 向量（标题 / 画面 / 融合）多路召回，RRF 融合。

模式
  hybrid  BM25(标题+描述) 与 融合向量 两路 RRF。默认，评测中比纯向量高约 4 分 nDCG@10
  dense   融合向量（画面与标题向量平均；无标题的视频只用画面）
  visual  只用画面向量：冷启动、标题不可信时用
  text    只用标题向量
  bm25    只用关键词
索引在数据变化（store.version 改变）后的下一次查询时自动重建。
"""
import threading
import time

import numpy as np

from .bm25 import BM25
from .encoder import ENCODER, renorm

MODES = ("hybrid", "dense", "visual", "text", "bm25")
DIMS = (768, 512, 256, 128)
DEPTH = 100  # 每路召回深度


def rrf(lists, k=60):
    sc = {}
    for ranks in lists:
        for r, i in enumerate(ranks):
            sc[i] = sc.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(sc, key=sc.get, reverse=True)


def top(scores, n):
    n = min(n, len(scores))
    if n <= 0:
        return np.zeros(0, dtype=int)
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx])]


class Index:
    def __init__(self, store, encoder=ENCODER):
        self.store, self.enc = store, encoder
        self.version = -1
        self._lock = threading.Lock()
        self.ids, self.pos, self.mats, self.bm25, self._dim_cache = [], {}, {}, None, {}
        self.built_s = None

    def _ensure(self):
        if self.version == self.store.version:
            return
        with self._lock:
            if self.version == self.store.version:
                return
            v = self.store.version
            t = time.time()
            ids, vis = self.store.all_vectors("visual")
            pos = {x: i for i, x in enumerate(ids)}
            mats = {"visual": vis}
            for kind in ("fused", "title"):
                kid, M = self.store.all_vectors(kind)
                full = np.zeros_like(vis)
                has = np.zeros(len(ids), dtype=bool)
                for x, row in zip(kid, M):
                    if x in pos:
                        full[pos[x]] = row
                        has[pos[x]] = True
                if kind == "fused":
                    full[~has] = vis[~has]  # 兜底：没有融合向量时用画面向量
                mats[kind] = full
                mats[kind + "_has"] = has
            texts = []
            for x in ids:
                r = self.store.get_video(x)
                texts.append(f"{r['title']} {r['description']} {' '.join(r['tags'] or [])}")
            self.ids, self.pos, self.mats, self._dim_cache = ids, pos, mats, {}
            self.bm25 = BM25(texts)
            self.version, self.built_s = v, round(time.time() - t, 3)

    def _mat(self, kind, dim):
        key = (kind, dim)
        if key not in self._dim_cache:
            M = self.mats[kind]
            self._dim_cache[key] = renorm(M, dim) if len(M) else M
        return self._dim_cache[key]

    @property
    def size(self):
        self._ensure()
        return len(self.ids)

    def search(self, q: str, mode="hybrid", dim=768, k=20):
        if mode not in MODES or dim not in DIMS:
            raise ValueError("bad mode or dim")
        self._ensure()
        t0 = time.perf_counter()
        timing = {}
        if not self.ids:
            return [], {"total_ms": 0.0}
        qv = None
        if mode != "bm25":
            qv = renorm(self.enc.query([q]), dim)[0]
        timing["encode_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        t1 = time.perf_counter()
        chans = {}
        if mode in ("hybrid", "dense"):
            chans["fused"] = self._mat("fused", dim) @ qv
        if mode == "visual":
            chans["visual"] = self._mat("visual", dim) @ qv
        if mode == "text":
            s = self._mat("title", dim) @ qv
            s[~self.mats["title_has"]] = -1.0  # 没有标题的视频不参与
            chans["title"] = s
        if mode in ("hybrid", "bm25"):
            chans["bm25"] = self.bm25.scores(q)
        timing["score_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        lists = []
        for name, s in chans.items():
            r = top(s, DEPTH)
            if name == "bm25":
                r = [i for i in r if s[i] > 0]
            lists.append(list(r))
        order = rrf(lists) if len(lists) > 1 else lists[0]
        timing["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        out = []
        for rank, i in enumerate(order[:k]):
            out.append({"video_id": self.ids[i], "rank": rank + 1,
                        "scores": {n: round(float(s[i]), 4) for n, s in chans.items()}})
        return out, timing

    def search_vector(self, v: np.ndarray, kind="visual", dim=768, k=20, exclude=None):
        self._ensure()
        if not self.ids:
            return []
        s = self._mat(kind, dim) @ renorm(v, dim)[0]
        if exclude in self.pos:
            s[self.pos[exclude]] = -9
        return [{"video_id": self.ids[i], "rank": r + 1, "scores": {kind: round(float(s[i]), 4)}}
                for r, i in enumerate(top(s, k)) if s[i] > -9]

    def similar(self, vid, k=12):
        self._ensure()
        if vid not in self.pos:
            return []
        return self.search_vector(self.mats["visual"][self.pos[vid]], "visual", 768, k, exclude=vid)

    def search_image(self, img, k=20):
        return self.search_vector(self.enc.images([img]), "visual", 768, k)
