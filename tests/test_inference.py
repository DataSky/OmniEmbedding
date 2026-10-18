"""Inference deployment tests: no torch import, model download or GPU required."""
import io
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from omniembed.health import InferenceInputError

from omniembed.inference import (
    MAX_BATCH_SIZE, MAX_IMAGE_BYTES, MAX_JSON_BYTES, MAX_TEXT_CHARS,
    MAX_TITLE_CHARS, MAX_TOTAL_TEXT_CHARS, MAX_VIDEO_BYTES,
    MAX_VIDEO_FRAMES, MAX_VIDEO_MULTIPART_BYTES, create_app,
)

AUTH = {"Authorization": "Bearer test-secret"}


class StubEncoder:
    model_id = "fake-embeddinggemma-2"
    revision = "fixed-model-sha"
    device = "cpu"
    dtype = "float32"

    def __init__(self):
        self.mode = "good"
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False

    def query(self, texts, bs=64):
        self.calls.append(("query", texts))
        if self.block:
            self.entered.set()
            assert self.release.wait(10), "test did not release fake inference"
        if self.mode == "error":
            raise RuntimeError("secret prompt and Hugging Face token must never be exposed")
        if self.mode == "shape":
            return np.ones((len(texts), 512), dtype=np.float32)
        values = np.tile(np.arange(1, 769, dtype=np.float32), (len(texts), 1))
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        if self.mode == "nan":
            values[0, 0] = np.nan
        elif self.mode == "zero":
            values[:] = 0
        elif self.mode == "norm":
            values *= 10
        return values

    def docs(self, titles, texts, bs=64):
        self.calls.append(("document", texts))
        self.calls.append(("document_titles", titles))
        return self.query(texts, bs=bs)

    def images(self, imgs, bs=16):
        self.calls.append(("images", [img.size for img in imgs]))
        return self.query(["image"] * len(imgs), bs=bs)

    def videos(self, frame_arrays, bs=4):
        self.calls.append(("videos", [frames.copy() for frames in frame_arrays], bs))
        return self.query(["video"] * len(frame_arrays), bs=bs)


def wait_ready(client):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get("/health/ready", headers=AUTH)
        if response.status_code == 200:
            return response.json()
        time.sleep(0.01)
    raise AssertionError(response.text)


@pytest.fixture()
def serving():
    encoder = StubEncoder()
    app = create_app(encoder, api_key="test-secret")
    with TestClient(app) as client:
        wait_ready(client)
        yield client, encoder, app


def test_api_key_required(monkeypatch):
    monkeypatch.delenv("OMNI_INFERENCE_API_KEY", raising=False)
    for secret in (None, "", "   "):
        with pytest.raises(ValueError, match="OMNI_INFERENCE_API_KEY"):
            create_app(StubEncoder(), api_key=secret)


def test_authentication_and_minimal_liveness(serving):
    client, encoder, app = serving
    live = client.get("/health/live")
    assert live.status_code == 200
    assert set(live.json()) == {"status", "boot_id", "uptime_s"}
    for path, method in (("/health/ready", client.get), ("/health/inference", client.post),
                         ("/v1/embeddings", client.post), ("/v1/embeddings/image", client.post),
                         ("/v1/embeddings/video", client.post)):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic test-secret"}):
            response = method(path, headers=headers)
            assert response.status_code == 401, response.text
            assert response.headers["www-authenticate"] == "Bearer"
    assert client.get("/docs").status_code == 404


def test_background_warmup_and_unready_status():
    encoder = StubEncoder()
    encoder.block = True
    app = create_app(encoder, api_key="test-secret")
    try:
        with TestClient(app) as client:
            assert encoder.entered.wait(2)
            assert client.get("/health/live").status_code == 200
            response = client.get("/health/ready", headers=AUTH)
            assert response.status_code == 503
            assert response.json()["status"] == "loading"
            assert not response.json()["ready"]
            encoder.release.set()
            assert wait_ready(client)["last_success"]
    finally:
        encoder.release.set()


def test_startup_failure_can_recover_with_deep_probe():
    encoder = StubEncoder()
    encoder.mode = "error"
    app = create_app(encoder, api_key="test-secret")
    with TestClient(app) as client:
        deadline = time.monotonic() + 2
        while app.state.inference.warming and time.monotonic() < deadline:
            time.sleep(0.01)
        ready = client.get("/health/ready", headers=AUTH)
        assert ready.status_code == 503
        assert ready.json()["last_error"] == "RuntimeError"
        assert "secret prompt" not in ready.text
        assert client.post("/v1/embeddings", json={"input": "hello"}, headers=AUTH).status_code == 503
        encoder.mode = "good"
        assert client.post("/health/inference", headers=AUTH).status_code == 200
        assert wait_ready(client)["last_error"] is None


def test_health_self_test_contract(serving):
    client, encoder, app = serving
    response = client.post("/health/inference", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["ready"]
    assert body["checks"] == {"shape": [1, 768], "finite": True, "nonzero": True, "unit_norm": True}
    assert body["boot_id"] == client.get("/health/live").json()["boot_id"]
    assert body["model_revision"] == "fixed-model-sha"
    assert body["device"] == "cpu" and body["dtype"] == "float32"
    assert body["last_success"] and body["last_error"] is None


@pytest.mark.parametrize("dimensions", [128, 256, 512, 768])
def test_text_dimensions_and_query_document_prefix_route(serving, dimensions):
    client, encoder, app = serving
    response = client.post("/v1/embeddings", headers=AUTH,
                           json={"input": ["hello", "中文查询"], "dimensions": dimensions})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "list" and body["dimensions"] == dimensions
    assert body["model"] == encoder.model_id and body["boot_id"]
    assert [row["index"] for row in body["data"]] == [0, 1]
    for row in body["data"]:
        assert row["object"] == "embedding" and len(row["embedding"]) == dimensions
        assert np.linalg.norm(row["embedding"]) == pytest.approx(1.0)
    response = client.post("/v1/embeddings", headers=AUTH,
                           json={"input": "document content", "input_type": "document"})
    assert response.status_code == 200
    assert ("document", ["document content"]) in encoder.calls
    assert ("document_titles", [""]) in encoder.calls


def test_document_titles_and_title_only_input_are_preserved(serving):
    client, encoder, app = serving
    response = client.post("/v1/embeddings", headers=AUTH, json={
        "input_type": "document", "titles": ["标题", "第二个标题"], "input": ["正文", ""],
    })
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 2
    assert ("document", ["正文", ""]) in encoder.calls
    assert ("document_titles", ["标题", "第二个标题"]) in encoder.calls


@pytest.mark.parametrize("payload", [
    {"input": "hello", "titles": ["title"]},
    {"input": "hello", "titles": None},
    {"input_type": "document", "input": ["a", "b"], "titles": ["one"]},
    {"input_type": "document", "input": ["a"], "titles": ["one", "two"]},
    {"input_type": "document", "input": "a", "titles": "title"},
    {"input_type": "document", "input": "a", "titles": [1]},
    {"input_type": "document", "input": "", "titles": ["  "]},
    {"input_type": "document", "input": "", "titles": None},
    {"input_type": "document", "input": "a", "titles": ["x" * (MAX_TITLE_CHARS + 1)]},
    {"input_type": "document", "input": ["x" * MAX_TEXT_CHARS] * 4,
     "titles": ["title"] * 4},
])
def test_document_title_validation_does_not_invoke_model(serving, payload):
    client, encoder, app = serving
    before = len(encoder.calls)
    response = client.post("/v1/embeddings", json=payload, headers=AUTH)
    assert response.status_code == 422, response.text
    assert len(encoder.calls) == before
    assert client.get("/health/ready", headers=AUTH).status_code == 200


def test_document_title_and_total_character_limits_are_inclusive(serving):
    client, encoder, app = serving
    title = "t" * MAX_TITLE_CHARS
    texts = ["x" * (MAX_TEXT_CHARS - MAX_TITLE_CHARS)] * 4
    assert sum(map(len, texts)) + len(title) * 4 == MAX_TOTAL_TEXT_CHARS
    response = client.post("/v1/embeddings", headers=AUTH, json={
        "input_type": "document", "input": texts, "titles": [title] * 4,
    })
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("payload", [
    {"input": ""}, {"input": "   "}, {"input": []}, {"input": 5},
    {"input": ["hello", 3]}, {"input": "hello", "input_type": "image"},
    {"input": "hello", "dimensions": 129}, {"input": "hello", "url": "https://example.com"},
    {"input": ["hello"] * (MAX_BATCH_SIZE + 1)},
    {"input": "x" * (MAX_TEXT_CHARS + 1)},
    {"input": ["x" * MAX_TEXT_CHARS] * (MAX_TOTAL_TEXT_CHARS // MAX_TEXT_CHARS + 1)},
])
def test_bounded_and_strict_text_input(serving, payload):
    client, encoder, app = serving
    before = len(encoder.calls)
    response = client.post("/v1/embeddings", json=payload, headers=AUTH)
    assert response.status_code == 422, response.text
    assert len(encoder.calls) == before
    assert client.get("/health/ready", headers=AUTH).status_code == 200


def test_json_streaming_body_limit_and_content_type(serving):
    client, encoder, app = serving
    assert client.post("/v1/embeddings", content="{}", headers=AUTH).status_code == 415
    response = client.post("/v1/embeddings", content=b"x" * (MAX_JSON_BYTES + 1),
                           headers={**AUTH, "Content-Type": "application/json"})
    assert response.status_code == 413
    # Iterators omit Content-Length: the actual stream must still be bounded.
    response = client.post("/v1/embeddings", content=iter([b"x" * MAX_JSON_BYTES, b"x"]),
                           headers={**AUTH, "Content-Type": "application/json"})
    assert response.status_code == 413
    assert client.post("/v1/embeddings", content=b"{", headers={**AUTH, "Content-Type": "application/json"}).status_code == 422


@pytest.mark.parametrize("mode", ["nan", "zero", "shape", "norm", "error"])
def test_invalid_model_output_degrades_readiness_and_recovers(serving, mode):
    client, encoder, app = serving
    encoder.mode = mode
    response = client.post("/v1/embeddings", json={"input": "sensitive content"}, headers=AUTH)
    assert response.status_code == 503
    ready = client.get("/health/ready", headers=AUTH)
    assert ready.status_code == 503 and ready.json()["last_error"]
    assert "sensitive content" not in ready.text
    assert "secret prompt" not in ready.text
    encoder.mode = "good"
    assert client.post("/health/inference", headers=AUTH).status_code == 200
    assert client.get("/health/ready", headers=AUTH).status_code == 200


def test_busy_does_not_queue_or_block_liveness(serving):
    client, encoder, app = serving
    encoder.block = True
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(client.post, "/v1/embeddings", headers=AUTH, json={"input": "blocked"})
        try:
            assert encoder.entered.wait(2)
            assert client.get("/health/live").status_code == 200
            for path in ("/health/inference", "/v1/embeddings"):
                response = client.post(path, headers=AUTH, json={"input": "second request"})
                assert response.status_code == 429
                assert response.headers["retry-after"] == "2"
            assert client.get("/health/ready", headers=AUTH).status_code == 200
            with app.state.inference.state_lock:
                app.state.inference.busy_since = time.monotonic() - 121
            ready = client.get("/health/ready", headers=AUTH)
            assert ready.status_code == 503 and ready.json()["status"] == "stalled"
            assert ready.json()["busy_for_s"] >= 120
            assert client.post("/health/inference", headers=AUTH).status_code == 503
            assert client.get("/health/live").status_code == 200
        finally:
            encoder.release.set()
        assert result.result(timeout=2).status_code == 200
    assert client.get("/health/ready", headers=AUTH).status_code == 200


def image_bytes(size=(12, 8), color=(0, 255, 0), format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format=format)
    return output.getvalue()


def video_files(contents):
    return [("files", (f"frame-{index}.png", content, "image/png"))
            for index, content in enumerate(contents)]


@pytest.mark.parametrize("dimensions", [128, 256, 512, 768])
def test_video_uses_one_ordered_native_frame_tensor(serving, dimensions):
    client, encoder, app = serving
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    response = client.post(f"/v1/embeddings/video?dimensions={dimensions}", headers=AUTH,
                           files=video_files([image_bytes(color=color) for color in colors]))
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 1
    vector = response.json()["data"][0]["embedding"]
    assert len(vector) == dimensions
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    call = next(call for call in encoder.calls if call[0] == "videos")
    assert call[2] == 1 and len(call[1]) == 1
    frames = call[1][0]
    assert frames.shape == (3, 8, 12, 3) and frames.dtype == np.uint8
    assert frames[:, 0, 0, :].tolist() == [list(color) for color in colors]
    assert not any(call[0] == "images" for call in encoder.calls)


@pytest.mark.parametrize("frame_count", [1, MAX_VIDEO_FRAMES])
def test_video_accepts_frame_count_boundaries(serving, frame_count):
    client, encoder, app = serving
    response = client.post("/v1/embeddings/video", headers=AUTH,
                           files=video_files([image_bytes()] * frame_count))
    assert response.status_code == 200, response.text
    call = next(call for call in encoder.calls if call[0] == "videos")
    assert call[1][0].shape[0] == frame_count


def test_video_upload_and_stream_limits(serving):
    client, encoder, app = serving
    before = len(encoder.calls)
    assert client.post("/v1/embeddings/video", headers=AUTH,
                       files=video_files([image_bytes()] * (MAX_VIDEO_FRAMES + 1))).status_code == 400
    assert client.post("/v1/embeddings/video", headers=AUTH,
                       files=video_files([b"x" * (MAX_VIDEO_BYTES // 2),
                                          b"x" * (MAX_VIDEO_BYTES // 2 + 1)])).status_code == 413
    headers = {**AUTH, "Content-Type": "multipart/form-data; boundary=test"}
    assert client.post("/v1/embeddings/video", headers=headers,
                       content=b"x" * (MAX_VIDEO_MULTIPART_BYTES + 1)).status_code == 413
    assert client.post("/v1/embeddings/video", headers=headers,
                       content=iter([b"x" * MAX_VIDEO_MULTIPART_BYTES, b"x"])).status_code == 413
    assert len(encoder.calls) == before
    assert client.get("/health/ready", headers=AUTH).status_code == 200


def test_video_pixel_budget_counts_all_frames(serving, monkeypatch):
    client, encoder, app = serving
    monkeypatch.setattr("omniembed.inference.MAX_VIDEO_PIXELS", 192)
    response = client.post("/v1/embeddings/video", headers=AUTH,
                           files=video_files([image_bytes(), image_bytes()]))
    assert response.status_code == 200, response.text
    before = len(encoder.calls)
    response = client.post("/v1/embeddings/video", headers=AUTH,
                           files=video_files([image_bytes()] * 3))
    assert response.status_code == 413
    assert len(encoder.calls) == before
    assert client.get("/health/ready", headers=AUTH).status_code == 200


@pytest.mark.parametrize("contents,status", [
    ([image_bytes(), b"broken frame"], 400),
    ([image_bytes(), image_bytes(size=(13, 8))], 400),
    ([image_bytes(format="GIF")], 415),
])
def test_invalid_video_frames_do_not_degrade_readiness(serving, contents, status):
    client, encoder, app = serving
    before = len(encoder.calls)
    response = client.post("/v1/embeddings/video", headers=AUTH, files=video_files(contents))
    assert response.status_code == status, response.text
    assert len(encoder.calls) == before
    assert client.get("/health/ready", headers=AUTH).status_code == 200


def test_video_rejects_animated_frames_and_non_file_protocols(serving):
    client, encoder, app = serving
    animated = io.BytesIO()
    Image.new("RGB", (12, 8), "red").save(
        animated, format="PNG", save_all=True,
        append_images=[Image.new("RGB", (12, 8), "blue")], duration=100, loop=0)
    before = len(encoder.calls)
    assert client.post("/v1/embeddings/video", headers=AUTH,
                       files=video_files([animated.getvalue()])).status_code == 415
    assert client.post("/v1/embeddings/video", headers=AUTH,
                       files={"file": ("frame.png", image_bytes(), "image/png")}).status_code == 422
    assert client.post("/v1/embeddings/video?dimensions=129", headers=AUTH,
                       files=video_files([image_bytes()])).status_code == 422
    assert client.post("/v1/embeddings/video", headers=AUTH,
                       json={"url": "https://example.com/video"}).status_code == 415
    assert client.post("/v1/embeddings/video", headers=AUTH,
                       files=video_files([image_bytes()]), data={"url": "https://example.com"}).status_code == 400
    assert len(encoder.calls) == before
    assert client.get("/health/ready", headers=AUTH).status_code == 200


@pytest.mark.parametrize("mode", ["nan", "zero", "shape", "norm", "error"])
def test_video_model_output_is_validated_and_errors_are_redacted(serving, mode):
    client, encoder, app = serving
    encoder.mode = mode
    response = client.post("/v1/embeddings/video", headers=AUTH,
                           files=video_files([image_bytes()]))
    assert response.status_code == 503
    assert "secret prompt" not in response.text
    assert client.get("/health/ready", headers=AUTH).status_code == 503
    encoder.mode = "good"
    assert client.post("/health/inference", headers=AUTH).status_code == 200


def test_video_does_not_queue_behind_inference(serving):
    client, encoder, app = serving
    encoder.block = True
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(client.post, "/v1/embeddings", headers=AUTH,
                                 json={"input": "blocked"})
        try:
            assert encoder.entered.wait(2)
            response = client.post("/v1/embeddings/video", headers=AUTH,
                                   files=video_files([image_bytes()]))
            assert response.status_code == 429
            assert response.headers["retry-after"] == "2"
            assert not any(call[0] == "videos" for call in encoder.calls)
        finally:
            encoder.release.set()
        assert result.result(timeout=2).status_code == 200


def test_single_image_upload(serving):
    client, encoder, app = serving
    response = client.post("/v1/embeddings/image?dimensions=128", headers=AUTH,
                           files={"file": ("green.png", image_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    assert len(response.json()["data"][0]["embedding"]) == 128
    assert ("images", [(12, 8)]) in encoder.calls
    assert client.post("/v1/embeddings/image", headers=AUTH,
                       files={"file": ("bad.png", b"not an image", "image/png")}).status_code == 400
    assert client.get("/health/ready", headers=AUTH).status_code == 200


def test_image_size_and_multipart_constraints(serving):
    client, encoder, app = serving
    assert client.post("/v1/embeddings/image", headers=AUTH,
                       files={"file": ("large.png", b"x" * (MAX_IMAGE_BYTES + 1), "image/png")}).status_code == 413
    assert client.post("/v1/embeddings/image", headers=AUTH,
                       json={"url": "https://example.com/image.png"}).status_code == 415
    response = client.post("/v1/embeddings/image", headers=AUTH,
                           files=[("file", ("a.png", image_bytes(), "image/png")),
                                  ("file", ("b.png", image_bytes(), "image/png"))])
    assert response.status_code == 400
    assert client.post("/v1/embeddings/image", headers=AUTH,
                       files={"wrong": ("a.png", image_bytes(), "image/png")}).status_code == 422


@pytest.mark.parametrize("device,capability,bf16_supported,expected", [
    ("cuda", 7, True, "float32"),  # T4: reject emulated BF16 support.
    ("cuda:0", 8, True, "bfloat16"),
    ("cuda", 8, False, "float32"),
    ("cpu", 0, False, "float32"),
])
def test_encoder_native_bf16_selection_and_revision(monkeypatch, device, capability, bf16_supported, expected):
    from omniembed.encoder import Encoder

    captured = {}

    class Model:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def eval(self):
            return self

    fake_torch = SimpleNamespace(
        bfloat16="bfloat16", float32="float32", device=lambda value: value,
        cuda=SimpleNamespace(is_available=lambda: device.startswith("cuda"),
                             is_bf16_supported=lambda: bf16_supported,
                             get_device_capability=lambda value: (capability, 0)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    encoder = Encoder(model_id="fake", device=device, revision="immutable-sha", dtype="auto")
    encoder._model()
    assert encoder.dtype == expected
    assert captured["model_kwargs"]["torch_dtype"] == expected
    assert captured["revision"] == "immutable-sha"


def test_encoder_refuses_unsafe_float16():
    from omniembed.encoder import Encoder

    with pytest.raises(ValueError, match="float16 is unsafe"):
        Encoder(model_id="fake", dtype="float16")


@pytest.mark.parametrize("reason", sorted(InferenceInputError.REASONS))
def test_confirmed_input_errors_preserve_readiness_and_safe_diagnostics(serving, monkeypatch, caplog, reason):
    from omniembed.health import InferenceInputError
    client, encoder, app = serving
    before = app.state.inference.snapshot()
    original = encoder.query

    def reject(texts, bs=64):
        raise InferenceInputError(reason)

    monkeypatch.setattr(encoder, "query", reject)
    with caplog.at_level("ERROR", logger="omniembed.inference"):
        response = client.post("/v1/embeddings", headers=AUTH,
                               json={"input": "private source body must not appear in logs"})
    assert response.status_code == 422
    assert response.json()["detail"] == {"code": "InferenceInputError", "reason": reason}
    after = app.state.inference.snapshot()
    assert after["ready"] and after["last_error"] == before["last_error"]
    assert after["last_success"] == before["last_success"]
    assert after["busy_for_s"] == 0
    records = [r for r in caplog.records if r.name == "omniembed.inference"]
    import json
    diagnostic = json.loads(records[-1].message)
    assert diagnostic["category"] == "InferenceInputError"
    assert diagnostic["failure_stage"] == "processor_validation"
    assert diagnostic["operation"] == "text_query"
    assert len(diagnostic["input_sha256"]) == 64
    assert "private source body" not in caplog.text + response.text
    monkeypatch.setattr(encoder, "query", original)
    assert client.post("/v1/embeddings", headers=AUTH, json={"input": "ordinary"}).status_code == 200


def test_input_error_self_test_still_degrades_model(serving, monkeypatch):
    from omniembed.health import InferenceInputError
    client, encoder, app = serving

    def reject(texts, bs=64):
        raise InferenceInputError("processor_image_without_media")

    monkeypatch.setattr(encoder, "query", reject)
    assert client.post("/health/inference", headers=AUTH).status_code == 503
    assert not app.state.inference.snapshot()["ready"]


@pytest.mark.parametrize("category", ["InferenceOOM", "InferenceError", "private-secret-category"])
def test_model_failures_keep_protection_and_redact_diagnostics(serving, monkeypatch, caplog, category):
    from omniembed.health import InferenceError
    client, encoder, app = serving

    def reject(texts, bs=64):
        try:
            raise ValueError("private source content")
        except ValueError as exc:
            raise InferenceError(category, "secret endpoint and credential") from exc

    monkeypatch.setattr(encoder, "query", reject)
    with caplog.at_level("ERROR", logger="omniembed.inference"):
        response = client.post("/v1/embeddings", headers=AUTH, json={"input": "private query"})
    assert response.status_code == 503
    snapshot = app.state.inference.snapshot()
    assert not snapshot["ready"]
    assert snapshot["last_error"] == (category if category != "private-secret-category" else "InferenceError")
    assert '"cause_type": "ValueError"' in caplog.text
    for private in ("private source content", "secret endpoint", "private query", "private-secret-category"):
        assert private not in caplog.text + response.text


def test_confirmed_input_rejection_does_not_poison_shared_health():
    from omniembed.health import Health, InferenceInputError
    health = Health(120)
    health.mark_warm()
    with health.inference():
        pass
    before = health.snapshot()
    with pytest.raises(InferenceInputError):
        with health.inference():
            raise InferenceInputError("processor_video_without_media")
    after = health.snapshot()
    assert after == before
    with pytest.raises(RuntimeError):
        with health.inference():
            raise RuntimeError("real failure")
    assert health.snapshot()["status"] == "degraded"
