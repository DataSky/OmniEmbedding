"""/v1/embeddings 与 /v1/models：OpenAI 格式、任务前缀选择、维度截断、base64、图片输入、错误格式、鉴权。"""
import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import FakeEncoder
from omniembed.openai_api import TASK_PREFIX


@pytest.fixture(scope="module")
def c():
    from omniembed import server
    fake = FakeEncoder()
    fake.model_id = "google/embeddinggemma-2"
    server.PIPE.enc = server.INDEX.enc = fake
    with TestClient(server.app) as client:
        client.fake, client.server = fake, server
        yield client


def png_b64(color):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_models(c):
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert "embeddinggemma-2" in ids and "embeddinggemma-2:query" in ids


def test_openai_shape_single_and_batch(c):
    r = c.post("/v1/embeddings", json={"input": "红烧肉做法", "model": "embeddinggemma-2"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["object"] == "list" and d["model"] == "embeddinggemma-2"
    assert d["data"][0]["object"] == "embedding" and d["data"][0]["index"] == 0 and len(d["data"][0]["embedding"]) == 768
    assert d["usage"]["prompt_tokens"] > 0 and d["usage"]["total_tokens"] == d["usage"]["prompt_tokens"]
    d = c.post("/v1/embeddings", json={"input": ["a", "b", "c"]}).json()
    assert [x["index"] for x in d["data"]] == [0, 1, 2]
    v = np.array(d["data"][0]["embedding"])
    assert abs(np.linalg.norm(v) - 1) < 1e-4  # 已归一化


def test_alias_and_full_model_id(c):
    assert c.post("/v1/embedding", json={"input": "x"}).status_code == 200
    assert c.post("/v1/embeddings", json={"input": "x", "model": "google/embeddinggemma-2"}).status_code == 200


@pytest.mark.parametrize("body,expect", [
    ({"input": "x"}, "query"),                                        # 默认
    ({"input": "x", "task": "query"}, "query"),                        # 扩展字段
    ({"input": "x", "model": "embeddinggemma-2:document"}, "document"),  # 模型名后缀
    ({"input": "x", "model": "embeddinggemma-2:query", "task": "none"}, "none"),  # task 优先于后缀
])
def test_task_selection(c, body, expect):
    d = c.post("/v1/embeddings", json=body).json()
    assert d["task"] == expect and c.fake.last_prefix == TASK_PREFIX[expect]


def test_dimensions_and_base64(c):
    d = c.post("/v1/embeddings", json={"input": "x", "dimensions": 256}).json()
    v = np.array(d["data"][0]["embedding"])
    assert v.size == 256 and abs(np.linalg.norm(v) - 1) < 1e-4  # 截断后重新归一化
    d = c.post("/v1/embeddings", json={"input": "x", "encoding_format": "base64", "dimensions": 128}).json()
    v = np.frombuffer(base64.b64decode(d["data"][0]["embedding"]), dtype="<f4")
    assert v.size == 128


def test_image_input_shares_space(c):
    d = c.post("/v1/embeddings", json={"input": [{"image": "data:image/png;base64," + png_b64((0, 0, 255))}, "blue", "red"]}).json()
    img, blue, red = (np.array(x["embedding"]) for x in d["data"])
    assert img @ blue > img @ red  # 图片与匹配的文字更近


@pytest.mark.parametrize("body,code", [
    ({"input": ""}, 400), ({"input": []}, 400), ({"input": [[1, 2, 3]]}, 400),
    ({"input": "x", "dimensions": 300}, 400), ({"input": "x", "task": "nope"}, 400),
    ({"input": "x", "model": "text-embedding-3-small"}, 404),
    ({"input": [{"image": "https://example.com/a.png"}]}, 400), ({"input": [{"image": "@@@"}]}, 400),
    ({"input": ["x"] * 513}, 400),
])
def test_errors_are_openai_shaped(c, body, code):
    r = c.post("/v1/embeddings", json=body)
    assert r.status_code == code
    assert "message" in r.json()["detail"]["error"]


def test_auth_bearer_and_x_api_key(c, monkeypatch):
    monkeypatch.setattr(c.server.CONFIG, "api_key", "k123")
    assert c.post("/v1/embeddings", json={"input": "x"}).status_code == 401
    assert c.get("/v1/models").status_code == 401
    assert c.post("/v1/embeddings", json={"input": "x"}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.post("/v1/embeddings", json={"input": "x"}, headers={"Authorization": "Bearer k123"}).status_code == 200
    assert c.post("/v1/embeddings", json={"input": "x"}, headers={"X-API-Key": "k123"}).status_code == 200
