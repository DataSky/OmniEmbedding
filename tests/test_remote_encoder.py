"""Real loopback HTTP tests; no Colab, model imports or production secrets."""
from contextlib import ExitStack
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading

import numpy as np
from PIL import Image
import pytest

from omniembed.remote_encoder import (
    DIMENSIONS, MAX_VIDEO_PIXELS, RemoteEncoder, RemoteEncoderError,
)

MODEL, REVISION = "test/model", "test-fixed-revision"


def response(rows=1):
    return {"model": MODEL, "model_revision": REVISION, "dimensions": DIMENSIONS,
            "boot_id": "test-boot", "data": [
                {"index": index, "embedding": [float(axis == index % DIMENSIONS) for axis in range(DIMENSIONS)]}
                for index in range(rows)
            ]}


@pytest.fixture
def servers():
    with ExitStack() as stack:
        def start(handler=None):
            state = {"requests": [], "handler": handler}

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    request = {"path": self.path, "headers": dict(self.headers), "body": body}
                    state["requests"].append(request)
                    if state["handler"] is None:
                        rows = len(json.loads(body)["input"]) if self.path == "/v1/embeddings" else 1
                        code, payload, headers = 200, response(rows), {}
                    else:
                        code, payload, headers = state["handler"](request)
                    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(raw)

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(server.server_close)
            stack.callback(thread.join, 3)
            stack.callback(server.shutdown)
            state["url"] = "http://127.0.0.1:" + str(server.server_port)
            return state
        yield start


@pytest.fixture
def configured(tmp_path, servers):
    server = servers()
    key = tmp_path / "synthetic.key"
    key.write_text("synthetic-test-secret\n")
    key.chmod(0o600)
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(json.dumps({"url": server["url"], "status": "ready",
                                    "expires_at": "2999-01-01T00:00:00Z"}))
    client = RemoteEncoder(model_id=MODEL, revision=REVISION, endpoint_file=endpoint, key_file=key, timeout=2)
    return client, server, endpoint, key


def parts(request):
    raw = ("Content-Type: " + request["headers"]["Content-Type"] + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + request["body"]
    return list(BytesParser(policy=default).parsebytes(raw).iter_parts())


def test_lazy_client_batch_limits_and_order(configured):
    client, server, _, _ = configured
    assert not client.loaded and client.backend == client.device == "remote"
    assert server["requests"] == []

    # The adapter must respect both API row and character budgets. This input
    # fits the 16-row limit but needs two requests because of total characters.
    texts = ["中" * 8192] * 5
    vectors = client.query(texts)
    calls = [json.loads(item["body"]) for item in server["requests"]]
    assert [len(call["input"]) for call in calls] == [4, 1]
    assert calls[0]["input"][0] == texts[0]  # no duplicate task prefix
    assert calls[0]["input_type"] == "query" and "titles" not in calls[0]
    assert vectors.shape == (5, DIMENSIONS) and vectors.dtype == np.float32
    assert np.array_equal(vectors[0], vectors[4])
    assert client.loaded and client.last_success_at and client.last_error is None
    assert client.boot_id == "test-boot"


def test_title_preservation_and_empty_document_body(configured):
    client, server, _, _ = configured
    client.docs(["真实标题", ""], ["", "文档内容"], bs=1)
    payloads = [json.loads(item["body"]) for item in server["requests"]]
    assert payloads == [
        {"input": [""], "input_type": "document", "dimensions": 768, "titles": ["真实标题"]},
        {"input": ["文档内容"], "input_type": "document", "dimensions": 768, "titles": [""]},
    ]


def test_tracks_endpoint_and_key_rotation(configured, servers):
    client, first, endpoint, key = configured
    client.query(["one"])
    second = servers()
    endpoint.write_text(json.dumps({"url": second["url"], "status": "ready"}))
    key.write_text("rotated-synthetic-secret")
    client.query(["two"])
    assert len(first["requests"]) == len(second["requests"]) == 1
    assert first["requests"][0]["headers"]["Authorization"] == "Bearer synthetic-test-secret"
    assert second["requests"][0]["headers"]["Authorization"] == "Bearer rotated-synthetic-secret"


def test_no_redirect_or_environment_proxy_leaks(configured, servers, monkeypatch):
    client, source, _, _ = configured
    target = servers()
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    # It still reaches loopback directly despite an invalid configured proxy.
    client.query(["direct"])
    source["handler"] = lambda request: (307, b"do not follow", {"Location": target["url"] + "/steal"})
    with pytest.raises(RemoteEncoderError, match="重定向"):
        client.query(["private prompt"])
    assert not target["requests"]


@pytest.mark.parametrize("status", [401, 403, 422, 500, 503])
def test_http_errors_are_sanitized(configured, status):
    client, server, _, _ = configured
    server["handler"] = lambda request: (status, b"private prompt synthetic-test-secret http://private", {})
    with pytest.raises(RemoteEncoderError) as caught:
        client.query(["private prompt"])
    assert "synthetic-test-secret" not in str(caught.value)
    assert "private" not in str(caught.value)
    assert client.last_error == str(caught.value) and not client.loaded
    assert caught.value.status_code == 503


def test_busy_retry_is_bounded_and_recovers(configured, monkeypatch):
    client, server, _, _ = configured
    waits = []
    monkeypatch.setattr("omniembed.remote_encoder.time.sleep", waits.append)
    server["handler"] = lambda request: (429, {}, {"Retry-After": "0.1"})
    with pytest.raises(RemoteEncoderError) as caught:
        client.query(["query"])
    assert caught.value.status_code == 429
    assert len(server["requests"]) == 3 and waits == [0.1, 0.1]
    server["handler"] = None
    client.query(["query"])
    assert client.loaded and client.last_error is None
    server["handler"] = lambda request: (503, {}, {})
    with pytest.raises(RemoteEncoderError):
        client.query(["query"])
    # A successful historical call is not a live readiness assertion.
    assert client.loaded and client.last_success_at and client.last_error


@pytest.mark.parametrize("damage", ["shape", "nan", "zero", "norm", "model", "revision", "dimensions", "index", "boot"])
def test_rejects_corrupt_or_incompatible_vectors(configured, damage):
    client, server, _, _ = configured
    value = response(2)
    if damage == "shape":
        value["data"][0]["embedding"] = [1.0]
    elif damage == "nan":
        value["data"][0]["embedding"][0] = float("nan")
    elif damage == "zero":
        value["data"][0]["embedding"] = [0.0] * DIMENSIONS
    elif damage == "norm":
        value["data"][0]["embedding"][0] = 10.0
    elif damage == "model":
        value["model"] = "different/model"
    elif damage == "revision":
        value["model_revision"] = "different-revision"
    elif damage == "dimensions":
        value["dimensions"] = 128
    elif damage == "index":
        value["data"][1]["index"] = 0
    else:
        value["boot_id"] = "invalid boot id"
    server["handler"] = lambda request: (200, value, {})
    with pytest.raises(RemoteEncoderError, match="校验"):
        client.query(["one", "two"])
    assert not client.loaded


def test_response_indices_determine_order(configured):
    client, server, _, _ = configured
    value = response(2)
    value["data"].reverse()
    server["handler"] = lambda request: (200, value, {})
    vectors = client.query(["one", "two"])
    assert vectors[0, 0] == vectors[1, 1] == 1


def test_unconfigured_revision_pins_first_response(configured):
    client, server, _, _ = configured
    client.revision = None
    client.query(["first"])
    assert client.revision == REVISION
    value = response()
    value["model_revision"] = "drift"
    server["handler"] = lambda request: (200, value, {})
    with pytest.raises(RemoteEncoderError, match="版本"):
        client.query(["second"])


def test_image_png_and_native_video_frame_order(configured):
    client, server, _, _ = configured
    image = Image.new("RGB", (8, 6), (1, 2, 3))
    client.images([image])
    image_parts = parts(server["requests"][-1])
    assert len(image_parts) == 1 and image_parts[0].get_param("name", header="content-disposition") == "file"
    with Image.open(io.BytesIO(image_parts[0].get_payload(decode=True))) as restored:
        assert restored.size == image.size and restored.getpixel((0, 0)) == (1, 2, 3)

    frames = np.stack([np.full((6, 8, 3), level, np.uint8) for level in (10, 80, 250)])
    vectors = client.videos([frames])
    request = server["requests"][-1]
    assert request["path"] == "/v1/embeddings/video?dimensions=768"
    frame_parts = parts(request)
    assert len(frame_parts) == 3
    assert all(part.get_param("name", header="content-disposition") == "files" for part in frame_parts)
    for part, expected in zip(frame_parts, frames):
        with Image.open(io.BytesIO(part.get_payload(decode=True))) as restored:
            assert np.array_equal(np.asarray(restored), expected)
    assert vectors.shape == (1, 768)


def test_full_hd_video_budget_preserves_resolution(configured):
    client, server, _, _ = configured
    frames = np.zeros((8, 1080, 1920, 3), np.uint8)
    client.videos([frames])
    for part in parts(server["requests"][-1]):
        with Image.open(io.BytesIO(part.get_payload(decode=True))) as restored:
            assert restored.size == (1920, 1080)


def test_input_validation_happens_before_http(configured):
    client, server, _, _ = configured
    invalid = [
        lambda: client.query([""]),
        lambda: client.query(["ok", "x" * 8193]),
        lambda: client.query(["ok"], bs=0),
        lambda: client.docs(["title"], []),
        lambda: client.docs(["x" * 1025], ["body"]),
        lambda: client.videos([np.zeros((9, 2, 2, 3), np.uint8)]),
        lambda: client.videos([np.zeros((1, 2, 2, 3), np.float32)]),
        lambda: client.videos([np.zeros((1, 1, MAX_VIDEO_PIXELS + 1, 3), np.uint8)]),
    ]
    for call in invalid:
        with pytest.raises(ValueError):
            call()
    assert server["requests"] == []
    for call in (lambda: client.query([]), lambda: client.docs([], []), lambda: client.images([]), lambda: client.videos([])):
        assert call().shape == (0, 768)


def test_private_key_and_no_symlinks(configured, tmp_path):
    client, server, _, key = configured
    key.chmod(0o644)
    with pytest.raises(RemoteEncoderError, match="600"):
        client.query(["test"])
    key.chmod(0o600)
    symlink = tmp_path / "link.key"
    symlink.symlink_to(key)
    client.key_file = symlink
    with pytest.raises(RemoteEncoderError, match="密钥"):
        client.query(["test"])
    assert server["requests"] == []


@pytest.mark.parametrize("url", ["http://example.com", "https://user:secret@example.com", "https://example.com/?key=secret", "https://example.com/#secret"])
def test_invalid_endpoint_never_gets_key(configured, url, monkeypatch):
    client, server, _, _ = configured
    client._url = url
    monkeypatch.setattr("omniembed.remote_encoder._key", lambda path: pytest.fail("must reject URL before reading key"))
    with pytest.raises(RemoteEncoderError):
        client.query(["test"])
    assert not server["requests"]


def test_expired_or_stopped_lease_blocks_requests(configured):
    client, server, endpoint, _ = configured
    endpoint.write_text(json.dumps({"url": server["url"], "expires_at": "2020-01-01T00:00:00Z"}))
    with pytest.raises(RemoteEncoderError, match="到期"):
        client.query(["test"])
    endpoint.write_text(json.dumps({"url": server["url"], "status": "stopped"}))
    with pytest.raises(RemoteEncoderError, match="就绪"):
        client.query(["test"])
    assert not server["requests"]


def test_connection_failures_and_oversized_responses_are_sanitized(configured):
    client, server, _, _ = configured
    client._url = "http://127.0.0.1:1"
    with pytest.raises(RemoteEncoderError, match="连接") as caught:
        client.query(["sensitive user query"])
    assert "127.0.0.1" not in str(caught.value) and "sensitive" not in str(caught.value)
    client._url = None
    server["handler"] = lambda request: (200, b"x" * (1024 * 1024 + 1), {})
    with pytest.raises(RemoteEncoderError, match="大小"):
        client.query(["test"])
