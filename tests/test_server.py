"""HTTP 接口测试：用 FastAPI TestClient + 假编码器，覆盖上传、URL 校验、元数据 / 向量查看、召回、评估、鉴权、路径安全。"""
import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import FakeEncoder, make_video


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from omniembed import server
    fake = FakeEncoder()
    server.PIPE.enc = fake
    server.INDEX.enc = fake
    with TestClient(server.app) as c:  # 进入时触发 startup，后台 worker 开始处理
        c.server = server
        yield c


def wait_ready(c, ids, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        st = [c.get(f"/api/videos/{i}").json()["status"] for i in ids]
        if all(s in ("ready", "failed") for s in st):
            return st
        time.sleep(0.2)
    raise AssertionError(f"timeout: {st}")


def test_upload_batch_and_metadata(client, tmp_path):
    files = [("files", (f"{c}.mp4", make_video(tmp_path / f"{c}.mp4", c).read_bytes(), "video/mp4")) for c in ("red", "green", "blue")]
    files.append(("files", ("notes.txt", b"hello", "text/plain")))
    r = client.post("/api/videos/upload", files=files, data={"titles": json.dumps(["红色片段", "绿色片段", ""]), "tags": "测试, 颜色"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["queued"]) == 3 and body["errors"][0]["file"] == "notes.txt"
    assert wait_ready(client, body["queued"]) == ["ready"] * 3
    red, green, blue = body["queued"]
    v = client.get(f"/api/videos/{red}").json()
    assert v["title"] == "红色片段" and v["tags"] == ["测试", "颜色"] and "path" not in v
    assert v["frames"] and v["frames"][0]["thumb_url"].startswith("/covers/")
    assert set(v["vectors"]) == {"visual", "title", "fused"}
    assert client.get(f"/api/videos/{blue}").json()["title"] == ""  # 未勾选时不拿文件名当标题
    assert client.get(v["cover_url"]).status_code == 200
    assert client.get(v["video_url"]).status_code == 200
    vec = client.get(f"/api/videos/{red}/vector", params={"kind": "visual", "full": True}).json()
    assert vec["dim"] == 768 and len(vec["values"]) == 768 and abs(vec["stats"]["norm"] - 1) < 0.01
    assert "fused" in vec["cosine_with"]
    assert len(client.get(f"/api/videos/{red}/vector").json()["values"]) == 32
    client.red, client.green, client.blue = red, green, blue


def test_search_modes_and_similar(client):
    r = client.get("/api/search", params={"q": "blue", "mode": "visual"}).json()
    assert r["results"][0]["video_id"] == client.blue and r["results"][0]["cover_url"]
    r = client.get("/api/search", params={"q": "红色", "mode": "hybrid"}).json()
    assert r["results"][0]["video_id"] == client.red and "bm25" in r["results"][0]["scores"]
    for mode in ("dense", "text", "bm25"):
        assert client.get("/api/search", params={"q": "绿", "mode": mode, "dim": 256}).status_code == 200
    assert client.get("/api/search", params={"q": "x", "mode": "nope"}).status_code == 400
    assert client.get("/api/search", params={"q": "x", "dim": 100}).status_code == 400
    sim = client.get(f"/api/videos/{client.red}/similar").json()["results"]
    assert client.red not in [s["video_id"] for s in sim]


def test_search_by_image(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 255, 0)).save(buf, "PNG")
    r = client.post("/api/search/image", files={"file": ("g.png", buf.getvalue(), "image/png")}).json()
    assert r["results"][0]["video_id"] == client.green
    assert client.post("/api/search/image", files={"file": ("x.png", b"nope", "image/png")}).status_code == 400


def test_edit_reembeds(client):
    r = client.patch(f"/api/videos/{client.blue}", json={"title": "蓝色海洋", "tags": "海, 蓝"})
    assert r.status_code == 200 and r.json()["tags"] == ["海", "蓝"]
    assert "title" in client.get(f"/api/videos/{client.blue}").json()["vectors"]
    r = client.get("/api/search", params={"q": "蓝色海洋", "mode": "bm25"}).json()
    assert r["results"][0]["video_id"] == client.blue


def test_urls_validation(client):
    r = client.post("/api/videos/urls", data={"urls": "ftp://x/a.mp4\nnot a url\nhttp://127.0.0.1:9/a.mp4"}).json()
    assert len(r["errors"]) == 2 and len(r["queued"]) == 1
    st = wait_ready(client, r["queued"])
    v = client.get(f"/api/videos/{r['queued'][0]}").json()
    assert st == ["failed"] and "内网" in v["error"]  # 回环地址被 SSRF 防护拒绝
    csv = "url,title,category\nhttp://10.1.2.3/a.mp4,内网,food\n"
    r = client.post("/api/videos/urls", files={"manifest": ("m.csv", csv.encode(), "text/csv")}).json()
    assert len(r["queued"]) == 1
    v = client.get(f"/api/videos/{r['queued'][0]}").json()
    assert v["title"] == "内网" and v["extra"] == {"category": "food"}
    r = client.post("/api/videos/urls", json={"items": [{"url": "https://example.invalid/x.mp4", "title": "t"}]}).json()
    assert len(r["queued"]) == 1


def test_list_retry_and_counts(client):
    r = client.get("/api/videos", params={"status": "failed"}).json()
    assert r["total"] >= 1
    assert client.post(f"/api/videos/{r['items'][0]['id']}/retry").status_code == 200
    s = client.get("/api/stats").json()
    assert s["indexed"] >= 3 and "ready" in s["status"]


def test_evalsets(client):
    r = client.post("/api/evalsets", data={"name": "auto", "auto": "title"}).json()
    assert r["n"] >= 2
    items = [{"query": "green", "gold": ["green.mp4"]}, {"query": "red", "gold": [client.red]}, {"query": "?", "gold": ["missing.mp4"]}]
    r = client.post("/api/evalsets", data={"name": "manual"}, files={"file": ("e.jsonl", "\n".join(json.dumps(x) for x in items).encode(), "application/json")}).json()
    assert r["n"] == 2 and r["dropped"] == 1  # 文件名 green.mp4 解析成了 video_id
    res = client.post("/api/evalsets/manual/run", params={"modes": "visual,hybrid"}).json()
    vis = next(x for x in res["results"] if x["mode"] == "visual")
    assert vis["R@1"] == 100.0 and vis["n"] == 2
    assert {x["name"] for x in client.get("/api/evalsets").json()["items"]} >= {"auto", "manual"}
    assert client.post("/api/evalsets/nope/run").status_code == 404


def test_delete_removes_files(client, tmp_path):
    r = client.post("/api/videos/upload", files=[("files", ("d.mp4", make_video(tmp_path / "d.mp4", "red").read_bytes(), "video/mp4"))]).json()
    vid = r["queued"][0]
    wait_ready(client, [vid])
    v = client.get(f"/api/videos/{vid}").json()
    assert client.delete(f"/api/videos/{vid}").status_code == 200
    assert client.get(f"/api/videos/{vid}").status_code == 404
    assert client.get(v["cover_url"]).status_code == 404


def test_api_key_required_when_set(client):
    cfg = client.server.CONFIG
    cfg.api_key = "secret-test-key"
    try:
        assert client.post("/api/videos/urls", data={"urls": "https://example.invalid/a.mp4"}).status_code == 401
        assert client.post("/api/videos/urls", data={"urls": "https://example.invalid/a.mp4"},
                           headers={"X-API-Key": "secret-test-key"}).status_code == 200
        assert client.get("/api/stats").status_code == 200  # 只读接口不需要
    finally:
        cfg.api_key = ""


def test_web_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "OmniEmbedding" in r.text
    for f in ("app.js", "library.js", "search_eval.js", "app.css"):
        assert client.get(f"/static/{f}").status_code == 200
