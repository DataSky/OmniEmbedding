"""入库流水线：领取任务 → 下载/落盘 → 解析 → 抽帧 → 向量化 → 存储。

每个阶段把状态写回数据库，前端据此显示进度；失败会记录原因，可以单条重试。
同一文件（sha256 相同）已经入库时直接复用向量，不重复编码。
"""
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from . import media
from .config import CONFIG
from .encoder import ENCODER
from .store import Store


class Pipeline:
    def __init__(self, store: Store, cfg=CONFIG, encoder=ENCODER):
        self.store, self.cfg, self.enc = store, cfg, encoder
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads = []
        self.processed = 0
        self.last_error = None

    # ---------- 入队 ----------
    def submit_file(self, tmp_path: Path, filename: str, title="", description="", tags=None, extra=None) -> str:
        vid = self.store.add_video("upload", filename, title, description, tags, extra)
        dst = self.cfg.video_dir / f"{vid}{media.guess_ext(filename)}"
        Path(tmp_path).replace(dst)
        self.store.update_video(vid, path=str(dst))
        self._wake.set()
        return vid

    def submit_url(self, url: str, title="", description="", tags=None, extra=None) -> str:
        vid = self.store.add_video("url", url, title, description, tags, extra)
        self._wake.set()
        return vid

    def submit_path(self, path: Path, title="", description="", tags=None, extra=None) -> str:
        """本机已有的文件：复制到数据目录，原文件不动。"""
        import shutil
        path = Path(path)
        # 不用文件名兜底当标题：IMG_1234 这类名字会污染 BM25 和标题向量；需要时由调用方显式传入
        vid = self.store.add_video("folder", str(path), title, description, tags, extra)
        dst = self.cfg.video_dir / f"{vid}{media.guess_ext(path.name)}"
        shutil.copy2(path, dst)
        self.store.update_video(vid, path=str(dst))
        self._wake.set()
        return vid

    def retry(self, vid):
        self.store.update_video(vid, status="queued", error=None)
        self._wake.set()

    # ---------- 后台 worker ----------
    def start(self):
        n = self.store.requeue_stale()
        if n:
            print(f"[pipeline] requeued {n} unfinished jobs")
        for i in range(max(1, self.cfg.workers)):
            t = threading.Thread(target=self._loop, name=f"pipeline-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        self._wake.set()

    def _loop(self):
        while not self._stop.is_set():
            vid = self.store.next_queued()
            if not vid:
                self._wake.wait(2)
                self._wake.clear()
                continue
            self.process(vid)

    def process(self, vid):
        st, timings = self.store, {}
        v = st.get_video(vid)
        try:
            # 1) 下载 / 落盘
            t = time.time()
            if v["source"] == "url" and not (v["path"] and Path(v["path"]).exists()):
                st.update_video(vid, status="downloading")
                dst = self.cfg.video_dir / f"{vid}{media.guess_ext(v['source_ref'])}"
                media.download(v["source_ref"], dst, self.cfg.max_upload_mb * 2**20, self.cfg.url_timeout_s,
                               self.cfg.allow_private_urls)
                st.update_video(vid, path=str(dst))
                v = st.get_video(vid)
            path = Path(v["path"])
            if not path.exists():
                raise media.MediaError("视频文件不存在")
            sha = media.sha256_file(path)
            timings["download_s"] = round(time.time() - t, 2)
            st.update_video(vid, sha256=sha, bytes=path.stat().st_size)

            # 2) 解析
            st.update_video(vid, status="probing")
            t = time.time()
            info = media.probe(path)
            timings["probe_s"] = round(time.time() - t, 2)
            st.update_video(vid, **{k: info.get(k) for k in ("duration_s", "width", "height", "fps", "codec", "has_audio")})

            # 3) 抽帧 + 封面 + 关键帧缩略图
            st.update_video(vid, status="extracting")
            t = time.time()
            frames, ts = media.extract_frames(path, info["duration_s"], self.cfg.fps, self.cfg.max_frames)
            thumbs = []
            for i, (f, tt) in enumerate(zip(frames, ts)):
                name = f"{vid}_{i}.jpg"
                media.save_thumb(f, self.cfg.cover_dir / name)
                thumbs.append((tt, name))
            st.put_frames(vid, thumbs)
            cover = thumbs[len(thumbs) // 2][1]
            timings["extract_s"] = round(time.time() - t, 2)
            st.update_video(vid, n_frames=len(frames), cover=cover)

            # 4) 向量化（同一文件已入库则复用）
            st.update_video(vid, status="embedding")
            t = time.time()
            dup = st.find_by_sha(sha)
            if dup and dup != vid:
                for kind, rec in st.get_vectors(dup).items():
                    if kind != "title":
                        st.put_vector(vid, kind, rec["vec"], rec["model"])
                timings["reused_from"] = dup
            else:
                vis = self.enc.videos([frames])[0]
                st.put_vector(vid, "visual", vis, self.enc.model_id)
            self._embed_text(vid)
            timings["embed_s"] = round(time.time() - t, 2)
            st.update_video(vid, status="ready", error=None, timings=timings)
            self.processed += 1
        except Exception as ex:
            msg = str(ex) if isinstance(ex, media.MediaError) else f"{type(ex).__name__}: {ex}"
            self.last_error = msg
            if not isinstance(ex, media.MediaError):
                traceback.print_exc()
            st.update_video(vid, status="failed", error=msg[:500], timings=timings)

    def _embed_text(self, vid):
        """标题 + 描述 → 文本向量；再与画面向量平均得到融合向量。元数据修改后单独调用即可，不必重抽帧。"""
        st = self.store
        v = st.get_video(vid)
        vecs = st.get_vectors(vid)
        title, desc = (v["title"] or "").strip(), (v["description"] or "").strip()
        if title or desc:
            tv = self.enc.docs([title], [desc])[0]
            st.put_vector(vid, "title", tv, self.enc.model_id)
        else:
            tv = None
            st.delete_vector(vid, "title")  # 标题被清空时不能留下旧向量
        if "visual" in vecs:
            vis = vecs["visual"]["vec"]
            fused = vis if tv is None else (vis + tv) / 2
            st.put_vector(vid, "fused", fused / max(np.linalg.norm(fused), 1e-12), self.enc.model_id)

    def reembed_text(self, vid):
        self._embed_text(vid)
