"""测试不需要 GPU，也不下载模型：用一个确定性的假编码器替代 EmbeddingGemma 2。
假编码器按文字里的关键词 / 画面主色给出向量，足以检验流水线、存储、召回、评估的逻辑是否正确。
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# 必须在导入 omniembed 之前强制设置：项目根目录的 .env（真实数据目录、访问密钥、局域网地址）不能影响测试
os.environ["OMNI_DATA_DIR"] = tempfile.mkdtemp(prefix="omni_test_")
os.environ["OMNI_API_KEY"] = ""
os.environ["OMNI_HOST"] = "127.0.0.1"
os.environ.pop("OMNI_EMBED_DEFAULT_TASK", None)
os.environ["OMNI_SKIP_WARMUP"] = "1"  # 不在测试里加载真实模型；健康测试手动预热假编码器

COLORS = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}  # OpenCV 用 BGR
WORDS = {"红": "red", "red": "red", "绿": "green", "green": "green", "蓝": "blue", "blue": "blue"}


def _axis(name):
    v = np.zeros(768, dtype=np.float32)
    v[["red", "green", "blue"].index(name)] = 1.0
    return v


class FakeEncoder:
    """texts() 与真实编码器一样经过 HEALTH.inference()，便于测健康状态；
    可通过 hang_s / fail_with / bad_vector 模拟卡死、推理异常、向量异常。"""
    model_id = "fake-encoder"
    device = "cpu"
    dtype = "float32"
    revision = resolved_revision = "test-rev"
    load_s = 0.0
    loaded = True
    hang_s = 0.0
    fail_with = None
    bad_vector = None

    def load(self):
        return self

    def _text(self, s):
        v = np.zeros(768, dtype=np.float32)
        for w, c in WORDS.items():
            if w in s.lower():
                v += _axis(c)
        h = int(hashlib.md5(s.encode()).hexdigest(), 16)
        v[10 + h % 700] += 0.05  # 让不同文本略有区别
        return v / max(np.linalg.norm(v), 1e-9)

    def query(self, texts, bs=64):
        return np.stack([self._text(t) for t in texts])

    def docs(self, titles, texts, bs=64):
        return np.stack([self._text(f"{a} {b}") for a, b in zip(titles, texts)])

    def texts(self, texts, prefix="", bs=64, token_budget=32768, lock_timeout=None):
        import time
        from omniembed.health import HEALTH
        self.last_prefix = prefix
        with HEALTH.inference(lock_timeout):
            if self.hang_s:
                time.sleep(self.hang_s)
            if self.fail_with:
                raise self.fail_with
            if self.bad_vector is not None:
                return np.asarray(self.bad_vector, dtype=np.float32)
            return np.stack([self._text(t) for t in texts])

    def count_tokens(self, texts):
        return sum(len(t) for t in texts)

    def _img(self, rgb):
        m = rgb.reshape(-1, 3).mean(0)  # RGB
        v = np.zeros(768, dtype=np.float32)
        v[:3] = [m[0], m[1], m[2]]
        return v / max(np.linalg.norm(v), 1e-9)

    def videos(self, frame_arrays, bs=4):
        return np.stack([self._img(f) for f in frame_arrays])

    def images(self, imgs, bs=16):
        return np.stack([self._img(np.asarray(i.convert("RGB"))) for i in imgs])


def make_video(path: Path, color="red", seconds=3, fps=10, size=(160, 96)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame[:] = COLORS[color]
    for _ in range(seconds * fps):
        w.write(frame)
    w.release()
    return path


@pytest.fixture()
def tmpdir_path(tmp_path):
    return tmp_path


@pytest.fixture()
def fake_encoder():
    return FakeEncoder()
