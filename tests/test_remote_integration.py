"""Real HTTP protocol integration, using only local fake inference and test keys.

These tests exercise the application's native video ingestion and retrieval
through the same authenticated remote adapter used for Colab. No GPU, model
download, real deployment metadata, or live inference secret is accessed.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
import urllib.request

import numpy as np
import pytest
from PIL import Image
import uvicorn

from conftest import FakeEncoder, make_video
from omniembed.config import Config
from omniembed.encoder import Encoder, create_encoder
from omniembed.inference import create_app
from omniembed.pipeline import Pipeline
from omniembed.remote_encoder import RemoteEncoder, RemoteEncoderError
from omniembed.retrieval import Index
from omniembed.store import Store


TEST_TOKEN = "synthetic-integration-token-not-a-real-secret"
MODEL_REVISION = "synthetic-fixed-model-revision"


class RecordingEncoder(FakeEncoder):
    revision = MODEL_REVISION
    dtype = "float32"

    def __init__(self):
        self.calls = []
        self.fail_video = False
        self.fail_query = False

    def query(self, texts, bs=64):
        self.calls.append(("query", list(texts)))
        if self.fail_query:
            raise RuntimeError("private-query-content-must-not-escape")
        return super().query(texts, bs)

    def docs(self, titles, texts, bs=64):
        self.calls.append(("docs", list(titles), list(texts)))
        return super().docs(titles, texts, bs)

    def videos(self, frame_arrays, bs=4):
        self.calls.append(("videos", [frames.copy() for frames in frame_arrays]))
        if self.fail_video:
            raise RuntimeError("private-video-content-must-not-escape")
        return super().videos(frame_arrays, bs)

    def images(self, imgs, bs=16):
        self.calls.append(("images", [image.copy() for image in imgs]))
        return super().images(imgs, bs)


@pytest.fixture()
def remote_system(tmp_path, monkeypatch):
    # Explicit test paths and loopback URL ensure the repository's running
    # Colab deployment and private key can never be selected by this fixture.
    monkeypatch.delenv("OMNI_INFERENCE_URL", raising=False)
    monkeypatch.delenv("OMNI_MODEL_REVISION", raising=False)
    cfg = Config()
    cfg.data_dir = tmp_path / "data"
    cfg.ensure_dirs()
    colab = cfg.data_dir / "colab"
    colab.mkdir()
    key_file = colab / "synthetic-token.key"
    key_file.write_text(TEST_TOKEN, encoding="ascii")
    key_file.chmod(0o600)
    endpoint_file = colab / "synthetic-endpoint.json"

    encoder = RecordingEncoder()
    app = create_app(encoder=encoder, api_key=TEST_TOKEN)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    host, port = listener.getsockname()
    url = f"http://{host}:{port}"
    endpoint_file.write_text(json.dumps({
        "url": url, "status": "ready",
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    }), encoding="utf-8")
    server = uvicorn.Server(uvicorn.Config(
        app, log_config=None, access_log=False, log_level="error", lifespan="on",
    ))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not (server.started and app.state.inference.snapshot()["ready"]):
            if not thread.is_alive() or time.monotonic() > deadline:
                raise AssertionError("local fake inference server did not become ready")
            time.sleep(0.01)
        encoder.calls.clear()  # Separate background startup warmup from user work.
        remote = RemoteEncoder(model_id=encoder.model_id, revision=MODEL_REVISION,
                               endpoint_file=endpoint_file, key_file=key_file, timeout=5)
        store = Store(cfg.db_path)
        yield SimpleNamespace(cfg=cfg, store=store, encoder=encoder, remote=remote,
                              pipe=Pipeline(store, cfg, remote), index=Index(store, remote),
                              url=url, key_file=key_file, endpoint_file=endpoint_file)
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive(), "local fake inference server did not stop"


def _ingest(system, tmp_path, color, title="", description=""):
    path = make_video(tmp_path / f"{color}.mp4", color, seconds=2)
    video_id = system.pipe.submit_path(path, title=title, description=description)
    assert system.store.next_queued() == video_id
    system.pipe.process(video_id)
    return video_id


def test_remote_native_video_ingestion_text_and_image_retrieval(remote_system, tmp_path):
    system = remote_system
    red = _ingest(system, tmp_path, "red", title="红色画面", description="红色测试片段")
    green = _ingest(system, tmp_path, "green", title="绿色画面")
    blue = _ingest(system, tmp_path, "blue")
    assert system.store.counts() == {"ready": 3}
    assert system.remote.loaded and system.remote.device == "remote"
    assert system.remote.revision == MODEL_REVISION
    assert system.remote.last_success_at and system.remote.last_error is None

    video_calls = [call for call in system.encoder.calls if call[0] == "videos"]
    assert len(video_calls) == 3
    for call in video_calls:
        assert len(call[1]) == 1
        frames = call[1][0]
        assert frames.dtype == np.uint8 and frames.shape == (2, 96, 160, 3)
    # The video path must use the native sequence encoder, rather than using
    # image embeddings or a client-side average in its place.
    assert not any(call[0] == "images" for call in system.encoder.calls)
    assert ("docs", ["红色画面"], ["红色测试片段"]) in system.encoder.calls
    assert ("docs", ["绿色画面"], [""]) in system.encoder.calls
    assert set(system.store.get_vectors(red)) == {"visual", "title", "fused"}
    assert set(system.store.get_vectors(blue)) == {"visual", "fused"}
    for video_id in (red, green, blue):
        for record in system.store.get_vectors(video_id).values():
            assert record["model"] == system.encoder.model_id and record["dim"] == 768
            assert np.isfinite(record["vec"]).all()
            assert np.linalg.norm(record["vec"]) == pytest.approx(1, abs=0.002)

    found, _ = system.index.search("蓝色", mode="visual", dim=768)
    assert found[0]["video_id"] == blue
    found, _ = system.index.search("红色", mode="text", dim=128)
    assert found[0]["video_id"] == red
    found, _ = system.index.search("绿色", mode="hybrid", dim=256)
    assert found[0]["video_id"] == green
    assert ("query", ["蓝色"]) in system.encoder.calls
    with Image.new("RGB", (64, 48), (0, 255, 0)) as image:
        found = system.index.search_image(image)
    assert found[0]["video_id"] == green
    assert any(call[0] == "images" for call in system.encoder.calls)

    # Editing metadata must preserve the remote document title/text contract.
    system.store.update_video(blue, title="蓝色海洋", description="蓝色风景")
    system.pipe.reembed_text(blue)
    assert ("docs", ["蓝色海洋"], ["蓝色风景"]) in system.encoder.calls
    assert "title" in system.store.get_vectors(blue)


def test_remote_failure_marks_ingestion_failed_without_vectors(remote_system, tmp_path):
    system = remote_system
    system.encoder.fail_video = True
    video_id = _ingest(system, tmp_path, "red", title="红色")
    row = system.store.get_video(video_id)
    assert row["status"] == "failed"
    # Upstream now persists only error categories. The remote adapter retains
    # its sanitized diagnostic without leaking the original model exception.
    assert row["error"] == "InferenceError"
    assert "503" in system.remote.last_error
    assert "private-video-content" not in row["error"]
    assert TEST_TOKEN not in row["error"]
    assert system.store.get_vectors(video_id) == {}
    assert system.index.size == 0 and not system.remote.loaded


def test_remote_query_failure_is_explicit_and_does_not_fall_back(remote_system, tmp_path):
    system = remote_system
    _ingest(system, tmp_path, "red", title="红色")
    system.encoder.fail_query = True
    with pytest.raises(RemoteEncoderError) as failure:
        system.index.search("private-query-content", mode="visual")
    message = str(failure.value)
    assert "503" in message and "private-query-content" not in message
    assert TEST_TOKEN not in message
    assert system.remote.last_error == message
    assert system.remote.last_success_at  # Historical success does not imply present health.
    # Keyword retrieval remains available while the remote model is degraded.
    result, _ = system.index.search("红色", mode="bm25")
    assert result


def test_wrong_synthetic_token_cannot_enqueue_a_success(remote_system, tmp_path):
    system = remote_system
    system.key_file.write_text("wrong-synthetic-token", encoding="ascii")
    video_id = _ingest(system, tmp_path, "red")
    row = system.store.get_video(video_id)
    assert row["status"] == "failed" and row["error"] == "InferenceError"
    assert "鉴权失败" in system.remote.last_error
    assert not any(call[0] == "videos" for call in system.encoder.calls)
    assert system.store.get_vectors(video_id) == {}


def test_encoder_factory_selects_remote_without_file_or_network_io(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNI_INFERENCE_URL", raising=False)
    cfg = Config()
    cfg.data_dir = tmp_path / "never-created"
    cfg.encoder_backend = "remote"

    def forbidden(*args, **kwargs):
        raise AssertionError("encoder construction must not read files, load a model, or send HTTP")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(Encoder, "_model", forbidden)
    encoder = create_encoder(cfg)
    assert isinstance(encoder, RemoteEncoder)
    assert encoder.endpoint_file == cfg.data_dir / "colab" / "endpoint.json"
    assert encoder.key_file == cfg.data_dir / "colab" / "inference.key"
    assert not encoder.loaded
    cfg.encoder_backend = "local"
    assert isinstance(create_encoder(cfg), Encoder)
    cfg.encoder_backend = "unrecognized-backend"
    with pytest.raises(ValueError, match="OMNI_ENCODER_BACKEND"):
        create_encoder(cfg)
