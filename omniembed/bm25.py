"""稀疏矩阵 BM25：10 万文档单次查询约 0.5 ms。中文用 jieba 搜索模式分词。"""
import re

import jieba
import numpy as np
from scipy import sparse

jieba.setLogLevel(60)
_ws = re.compile(r"\s+")
_punct = set("，。！？、#|:：()（）[]【】-_/.,!?;；\"'“”‘’")


def tokenize(s: str):
    return [w for w in jieba.lcut_for_search((s or "").lower()) if not _ws.fullmatch(w) and w not in _punct]


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.n = len(docs)
        self.vocab = {}
        if not docs:
            self.W = sparse.csc_matrix((0, 0), dtype=np.float32)
            return
        rows, cols, vals = [], [], []
        for i, d in enumerate(docs):
            tf = {}
            for w in tokenize(d):
                tf[w] = tf.get(w, 0) + 1
            for w, c in tf.items():
                rows.append(i)
                cols.append(self.vocab.setdefault(w, len(self.vocab)))
                vals.append(c)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(self.n, max(len(self.vocab), 1)), dtype=np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        avg = max(dl.mean(), 1e-9)
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (self.n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / avg)) * idf[tf.col]
        self.W = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, q):
        ids = [self.vocab[w] for w in tokenize(q) if w in self.vocab]
        if not ids or self.n == 0:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(self.W[:, ids].sum(1)).ravel()
