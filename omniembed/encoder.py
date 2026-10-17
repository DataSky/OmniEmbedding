"""EmbeddingGemma 2 封装：懒加载、任务前缀、MRL 截断。

要点（均来自本项目的实测，见 docs/EVALUATION.md）：
- 只能 bf16 / fp32，官方说明 fp16 会出 NaN。
- 文本检索必须带前缀：查询 "task: search result | query: "，文档 "title: … | text: …"；不带前缀 nDCG@10 掉约 19 分。
- 图片 / 视频不加前缀；视频以帧数组输入。
"""
import threading
import time

import numpy as np

from .config import CONFIG

QUERY_PREFIX = "task: search result | query: "


def doc_text(title: str, text: str) -> str:
    return f"title: {title.strip() or 'none'} | text: {text.strip() or title.strip()}"


def renorm(x: np.ndarray, dim: int | None) -> np.ndarray:
    x = np.atleast_2d(x)
    if dim and dim < x.shape[1]:
        x = x[:, :dim]
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class Encoder:
    def __init__(self, model_id=None, device=None):
        self.model_id = model_id or CONFIG.model_id
        self.device = device or CONFIG.device
        self._m = None
        self._lock = threading.Lock()
        self.load_s = None

    @property
    def loaded(self):
        return self._m is not None

    def _model(self):
        if self._m is None:
            with self._lock:
                if self._m is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    dev = self.device
                    if dev == "auto":
                        dev = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
                    t = time.time()
                    m = SentenceTransformer(self.model_id, device=dev, model_kwargs={"torch_dtype": dtype},
                                            config_kwargs={"audio_config": None})  # 视频只用画面，不加载音频编码器
                    m.eval()
                    m.max_seq_length = 8192
                    self.device, self.load_s, self._m = dev, round(time.time() - t, 1), m
        return self._m

    def _enc(self, items, bs):
        import torch
        m = self._model()
        with torch.inference_mode():
            e = m.encode(items, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(e, dtype=np.float32)

    def query(self, texts, bs=64):
        return self._enc([QUERY_PREFIX + t for t in texts], bs)

    def texts(self, texts, prefix="", bs=64, token_budget=32768):
        """任意前缀的文本编码（/v1/embeddings 用）。
        按长度排序后分批，每批 token 总数不超过 token_budget：短文本照常大批处理，
        长文本自动降到小批，避免几条 8k token 的输入把显存占满（实测不限长度时单批可申请 30+ GiB）。"""
        items = [prefix + t for t in texts]
        tok = self._model().tokenizer
        lens = [min(len(tok(t)["input_ids"]), 8192) for t in items]
        order = sorted(range(len(items)), key=lambda i: lens[i])
        out = np.zeros((len(items), 768), dtype=np.float32)
        i = 0
        while i < len(order):
            j = i
            # 同一批按最长的那条补齐，所以用 (条数 × 最长长度) 估算显存
            while j < len(order) and j - i < bs and (j - i + 1) * lens[order[j]] <= max(token_budget, lens[order[j]]):
                j += 1
            idx = order[i:j]
            out[idx] = self._enc([items[k] for k in idx], len(idx))
            i = j
        return out

    def count_tokens(self, texts):
        tok = self._model().tokenizer
        return sum(len(tok(t)["input_ids"]) for t in texts)

    def docs(self, titles, texts, bs=64):
        return self._enc([doc_text(a, b) for a, b in zip(titles, texts)], bs)

    def videos(self, frame_arrays, bs=4):
        """frame_arrays: list of [T,H,W,3] RGB uint8。同一批内帧尺寸可以不同。"""
        return self._enc([{"video": f} for f in frame_arrays], bs)

    def images(self, imgs, bs=16):
        return self._enc(list(imgs), bs)


ENCODER = Encoder()
