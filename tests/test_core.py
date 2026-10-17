"""存储 / 媒体 / 流水线 / 召回 / 评估的单元测试（假编码器，无需 GPU）。"""
import numpy as np
import pytest

from conftest import make_video
from omniembed import evaluation, media
from omniembed.bm25 import BM25
from omniembed.config import Config
from omniembed.pipeline import Pipeline
from omniembed.retrieval import Index, rrf
from omniembed.store import Store


@pytest.fixture()
def env(tmp_path, fake_encoder):
    cfg = Config()
    cfg.data_dir = tmp_path / "data"
    cfg.ensure_dirs()
    st = Store(cfg.db_path)
    return cfg, st, Pipeline(st, cfg, fake_encoder), Index(st, fake_encoder)


def test_store_roundtrip(tmp_path):
    st = Store(tmp_path / "a.db")
    vid = st.add_video("upload", "a.mp4", "标题", "描述", ["x", "y"], {"k": 1})
    st.update_video(vid, status="ready", duration_s=3.0, timings={"probe_s": 0.1})
    v = st.get_video(vid)
    assert v["tags"] == ["x", "y"] and v["extra"] == {"k": 1} and v["timings"]["probe_s"] == 0.1
    st.put_vector(vid, "visual", np.ones(768), "m")
    assert st.get_vectors(vid)["visual"]["vec"].shape == (768,)
    ids, M = st.all_vectors("visual")
    assert ids == [vid] and M.shape == (1, 768)
    with pytest.raises(ValueError):
        st.update_video(vid, bogus=1)
    st.delete_video(vid)
    assert st.get_video(vid) is None and st.get_vectors(vid) == {}  # 向量随视频级联删除


def test_next_queued_claims_once(tmp_path):
    st = Store(tmp_path / "a.db")
    vid = st.add_video("url", "http://x/a.mp4")
    assert st.next_queued() == vid
    assert st.next_queued() is None  # 已被领取，不会重复处理
    assert st.requeue_stale() == 1 and st.next_queued() == vid


def test_probe_and_extract(tmp_path):
    p = make_video(tmp_path / "r.mp4", "red", seconds=4)
    info = media.probe(p)
    assert 3.5 <= info["duration_s"] <= 4.5 and info["width"] == 160
    frames, ts = media.extract_frames(p, info["duration_s"], fps=1, max_frames=8)
    assert 3 <= len(frames) <= 4 and ts == sorted(ts)
    frames, _ = media.extract_frames(p, info["duration_s"], fps=10, max_frames=5)
    assert len(frames) == 5  # 超过上限时均匀取样


def test_probe_rejects_non_video(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"not a video")
    with pytest.raises(media.MediaError):
        media.probe(p)


@pytest.mark.parametrize("url", ["http://127.0.0.1/a.mp4", "http://localhost/a.mp4", "http://10.0.0.5/a.mp4",
                                 "http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "ftp://x/a.mp4"])
def test_download_blocks_private_and_bad_schemes(tmp_path, url):
    with pytest.raises(media.MediaError):
        media.download(url, tmp_path / "a.mp4", 10 * 2**20, 5)


def test_bm25_and_rrf():
    bm = BM25(["红烧肉 做法", "清蒸鱼", "红烧 排骨"])
    s = bm.scores("红烧肉")
    assert s.argmax() == 0 and s[1] == 0
    assert rrf([[1, 2, 3], [3, 1]])[:2] == [1, 3]
    assert BM25([]).scores("x").size == 0


def test_pipeline_end_to_end(env, tmp_path):
    cfg, st, pipe, idx = env
    ids = {}
    for c, title in [("red", "红色视频"), ("green", "绿色视频"), ("blue", "")]:
        ids[c] = pipe.submit_path(make_video(tmp_path / f"{c}.mp4", c), title=title)
    while (vid := st.next_queued()):
        pipe.process(vid)
    assert st.counts() == {"ready": 3}
    v = st.get_video(ids["red"])
    assert v["n_frames"] >= 1 and v["cover"] and v["sha256"] and v["timings"]["embed_s"] >= 0
    assert set(st.get_vectors(ids["red"])) == {"visual", "title", "fused"}
    assert set(st.get_vectors(ids["blue"])) == {"visual", "fused"}  # 没有标题就没有标题向量
    assert len(st.get_frames(ids["red"])) == v["n_frames"]
    assert idx.size == 3
    # 画面召回：查询 "blue" 只靠画面也能找到无标题的蓝色视频
    res, _ = idx.search("blue", mode="visual")
    assert res[0]["video_id"] == ids["blue"]
    # 文本召回只在有标题的视频里找
    res, _ = idx.search("绿", mode="text")
    assert res[0]["video_id"] == ids["green"] and ids["blue"] not in [r["video_id"] for r in res if r["scores"]["title"] > -1]
    res, _ = idx.search("红色", mode="hybrid")
    assert res[0]["video_id"] == ids["red"]
    # 维度截断后仍可检索
    res, _ = idx.search("red", mode="visual", dim=128)
    assert res[0]["video_id"] == ids["red"]
    # 相似视频不包含自己
    assert ids["red"] not in [r["video_id"] for r in idx.similar(ids["red"])]


def test_duplicate_reuses_vectors_and_edit_reembeds(env, tmp_path):
    cfg, st, pipe, idx = env
    p = make_video(tmp_path / "r.mp4", "red")
    a = pipe.submit_path(p, title="第一条")
    b = pipe.submit_path(p, title="第二条")
    while (vid := st.next_queued()):
        pipe.process(vid)
    assert st.get_video(b)["timings"].get("reused_from") == a
    before = idx.size
    st.update_video(b, title="蓝色")
    pipe.reembed_text(b)
    assert idx.size == before  # 索引按 version 重载
    res, _ = idx.search("蓝色", mode="text")
    assert res[0]["video_id"] == b
    st.update_video(b, title="", description="")
    pipe.reembed_text(b)
    assert "title" not in st.get_vectors(b)  # 清空标题后旧标题向量被删除


def test_failed_job_records_error(env, tmp_path):
    cfg, st, pipe, _ = env
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage")
    vid = pipe.submit_path(bad)
    pipe.process(st.next_queued())
    v = st.get_video(vid)
    assert v["status"] == "failed" and v["error"]


def test_evaluation(env, tmp_path):
    cfg, st, pipe, idx = env
    for c, title in [("red", "红"), ("green", "绿"), ("blue", "蓝")]:
        pipe.submit_path(make_video(tmp_path / f"{c}.mp4", c), title=title)
    while (vid := st.next_queued()):
        pipe.process(vid)
    items = evaluation.from_titles(st)
    assert len(items) == 3
    r = evaluation.evaluate(idx, items, "visual")
    assert r["R@1"] == 100.0 and r["nDCG@10"] == 100.0
    rows, dropped = evaluation.resolve_gold(st, [{"query": "红", "gold": ["red.mp4"]}, {"query": "x", "gold": ["nope.mp4"]}])
    assert len(rows) == 1 and dropped == 1
