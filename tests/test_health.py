"""健康接口契约：/health/live、/health/ready、/health/inference（字段、类型、状态码、no-store、鉴权、状态机）。"""
import threading
import time
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEncoder
from omniembed.health import HEALTH, Health, InferenceError
from omniembed.health_api import check_vector, warmup


@pytest.fixture()
def c():
    from omniembed import server
    fake = FakeEncoder()
    server.PIPE.enc = server.INDEX.enc = fake
    HEALTH.reset()
    HEALTH.timeout_s = 120.0
    with TestClient(server.app) as client:
        client.fake, client.server = fake, server
        yield client
    HEALTH.reset()


def warm(c):
    assert warmup(lambda: c.fake, retry_s=0.01)


ISO = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$"


def test_live_no_auth_and_shape(c, monkeypatch):
    monkeypatch.setattr(c.server.CONFIG, "api_key", "k")
    r = c.get("/health/live")
    d = r.json()
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert d["status"] == "ok" and isinstance(d["uptime_s"], float)
    uuid.UUID(d["boot_id"])
    assert c.get("/health/live").json()["boot_id"] == d["boot_id"]  # 同一进程不变


def test_ready_loading_then_ok(c):
    r = c.get("/health/ready")
    d = r.json()
    assert r.status_code == 503 and d["status"] == "loading" and d["ready"] is False
    assert r.headers["cache-control"] == "no-store"
    assert d["boot_id"] == c.get("/health/live").json()["boot_id"]  # 503 也带同一 boot_id
    warm(c)
    r = c.get("/health/ready")
    d = r.json()
    assert r.status_code == 200 and d["status"] == "ok" and d["ready"] is True  # 布尔，不是字符串
    for k, t in [("model", str), ("model_revision", str), ("device", str), ("dtype", str), ("uptime_s", float),
                 ("busy_for_s", float), ("inference_timeout_s", float)]:
        assert isinstance(d[k], t), k
    assert d["last_error"] is None
    import re
    assert re.match(ISO, d["last_success"])


def test_ready_degraded_after_failure_recovers_after_success(c):
    warm(c)
    c.fake.fail_with = InferenceError("InferenceOOM", "CUDA out of memory at C:\\secret\\path")
    r = c.post("/v1/embeddings", json={"input": "x"})
    assert r.status_code == 503 and r.json()["detail"]["error"]["message"] == "InferenceOOM"
    d = c.get("/health/ready").json()
    assert d["status"] == "degraded" and d["ready"] is False and d["last_error"] == "InferenceOOM"
    assert "secret" not in str(d)  # 只暴露类别
    c.fake.fail_with = None
    assert c.post("/v1/embeddings", json={"input": "x"}).status_code == 200
    d = c.get("/health/ready").json()
    assert d["status"] == "ok" and d["ready"] is True


def test_ready_stalled_and_answers_while_busy(c):
    warm(c)
    HEALTH.timeout_s = 0.3
    c.fake.hang_s = 1.0
    t = threading.Thread(target=lambda: c.post("/v1/embeddings", json={"input": "x"}))
    t.start()
    time.sleep(0.1)
    t0 = time.time()
    d = c.get("/health/ready").json()
    assert time.time() - t0 < 0.5  # 推理在跑也不阻塞 ready
    assert d["status"] == "ok" and d["busy_for_s"] > 0  # 忙碌但未超时
    time.sleep(0.4)
    r = c.get("/health/ready")
    d = r.json()
    assert r.status_code == 503 and d["status"] == "stalled" and d["last_error"] == "InferenceStalled"
    t.join()
    c.fake.hang_s = 0
    assert c.get("/health/ready").json()["status"] == "ok"  # 推理结束后恢复


def test_inference_probe_ok(c):
    warm(c)
    r = c.post("/health/inference")  # 空请求体
    d = r.json()
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert d["ok"] is True and d["status"] == "ok" and d["error"] is None
    assert d["checks"] == {"shape": True, "finite": True, "nonzero": True, "unit_norm": True}
    assert abs(d["norm"] - 1) < 0.05 and d["latency_ms"] >= 0 and d["dimensions"] == 768 and d["task"] == "query"
    assert c.fake.last_prefix == "task: search result | query: "  # 与 /v1/embeddings 默认相同的 query 路径
    assert d["boot_id"] == c.get("/health/live").json()["boot_id"]


@pytest.mark.parametrize("vec,failed", [
    (np.full((1, 768), np.nan), "finite"),
    (np.zeros((1, 768)), "nonzero"),
    (np.ones((1, 768)), "unit_norm"),
    (np.ones((1, 512)) / np.sqrt(512), "shape"),
])
def test_inference_probe_checks_are_real(c, vec, failed):
    warm(c)
    c.fake.bad_vector = vec
    r = c.post("/health/inference")
    d = r.json()
    assert r.status_code == 503 and d["ok"] is False and d["checks"][failed] is False and d["error"] == "VectorCheckFailed"
    assert c.get("/health/ready").json()["status"] == "degraded"


def test_inference_probe_errors(c):
    r = c.post("/health/inference")
    assert r.status_code == 503 and r.json()["status"] == "loading"
    warm(c)
    c.fake.fail_with = RuntimeError("boom at /home/x")
    d = c.post("/health/inference").json()
    assert d["status"] == "degraded" and d["error"] == "InferenceError" and "home" not in str(d)


def test_inference_probe_busy_and_timeout(c, monkeypatch):
    warm(c)
    monkeypatch.setattr(c.server.CONFIG, "probe_wait_s", 0.1)
    HEALTH.lock.acquire()  # 模拟一个长时间占用 GPU 的业务推理
    try:
        d = c.post("/health/inference").json()
        assert d["status"] == "busy" and d["error"] == "InferenceBusy"
    finally:
        HEALTH.lock.release()
    assert c.get("/health/ready").json()["status"] == "ok"  # 等锁超时不算推理失败
    monkeypatch.setattr(c.server.CONFIG, "probe_timeout_s", 0.2)
    c.fake.hang_s = 0.6
    d = c.post("/health/inference").json()
    assert d["status"] == "stalled" and d["error"] == "InferenceTimeout"
    time.sleep(0.6)
    c.fake.hang_s = 0


def test_health_auth(c, monkeypatch):
    monkeypatch.setattr(c.server.CONFIG, "api_key", "k123")
    assert c.get("/health/ready").status_code == 401
    assert c.post("/health/inference").status_code == 401
    assert c.get("/health/ready", headers={"Authorization": "Bearer k123"}).status_code in (200, 503)
    assert c.get("/health/ready", headers={"X-API-Key": "k123"}).status_code in (200, 503)
    r = c.get("/health/ready", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401 and "html" not in r.headers["content-type"]  # 不返回登录页


def test_stats_redacted(c):
    d = c.get("/api/stats").json()
    assert "data_dir" not in d and d["health"] in ("loading", "ok", "degraded", "stalled")


def test_check_vector_unit():
    v = np.zeros((1, 768)); v[0, 0] = 1
    assert check_vector(v)[0]
    assert not check_vector(np.zeros(768))[1]["shape"]  # 一维也不算 [1, 768]


def test_health_state_unit():
    h = Health(timeout_s=10)
    assert h.snapshot()["status"] == "loading"
    h.mark_load_error("ModelLoadError")
    assert h.snapshot()["status"] == "degraded"
    h.mark_warm()
    with h.inference():
        pass
    assert h.snapshot()["status"] == "ok" and h.snapshot()["last_success"]
