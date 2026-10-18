"""No network, model downloads, real credentials, or third-party dependencies."""
from contextlib import redirect_stdout
from email.parser import BytesParser
from email.policy import default
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.parse
import zlib

from scripts import autodl_service_probe as probe


BOOT = "5d2b86da-70c3-4fd8-af66-a3307d7c688b"
SECRET = "synthetic-unit-test-secret"
URL = "https://example.seetacloud.com:8443"


def ready_payload(**changes):
    result = {"model": probe.MODEL, "model_revision": probe.REVISION,
              "boot_id": BOOT, "ready": True, "status": "ok"}
    result.update(changes)
    return result


def vector_payload(rows=1, dimensions=768):
    return {"object": "list", "model": probe.MODEL,
            "model_revision": probe.REVISION, "boot_id": BOOT,
            "dimensions": dimensions, "data": [
                {"object": "embedding", "index": index,
                 "embedding": [float(axis == index) for axis in range(dimensions)]}
                for index in range(rows)]}


class Clock:
    def __init__(self):
        self.value, self.sleeps = 1000.0, []

    def monotonic(self):
        return self.value

    def wall(self):
        return 1700000000.0 + self.value - 1000

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class FakeHTTP:
    def __init__(self, responses=None, clock=None):
        self.responses = list(responses or [])
        self.requests, self.timeouts = [], []
        self.clock = clock

    def __call__(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            if callable(response):
                return response(request, timeout)
            status, headers, value = response
        elif request.get_method() == "GET":
            status, headers, value = 200, {}, ready_payload()
        else:
            rows = (len(json.loads(request.data)["input"])
                    if request.get_header("Content-type") == "application/json" else 1)
            status, headers, value = 200, {}, vector_payload(rows)
        return status, headers, value if isinstance(value, bytes) else json.dumps(value).encode()


def configured(fake=None, *, clock=None, **kwargs):
    fake = fake or FakeHTTP()
    clock = clock or Clock()
    result = probe.Probe(URL, SECRET, transport=fake, monotonic=clock.monotonic,
                         wall=clock.wall, sleep=clock.sleep, jitter=lambda: 0.1, **kwargs)
    return result, fake, clock


def mime_parts(request):
    prefix = "Content-Type: " + request.get_header("Content-type") + "\r\nMIME-Version: 1.0\r\n\r\n"
    return list(BytesParser(policy=default).parsebytes(prefix.encode() + request.data).iter_parts())


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = self.directory / "client.env"
        self.write_env()

    def write_env(self, content=None):
        self.env.write_text(content or (f"# literal test configuration\nexport OMNI_INFERENCE_URL='{URL}'\n"
                                       f'export OMNI_INFERENCE_API_KEY="{SECRET}"\n'))
        self.env.chmod(0o600)

    def test_env_and_endpoint_override(self):
        self.assertEqual(probe.load_config(self.env), (URL, SECRET))
        endpoint = self.directory / "endpoint.json"
        endpoint.write_text(json.dumps({"url": "https://another.seetacloud.com:8443/", "status": "ready"}))
        self.assertEqual(probe.load_config(self.env, endpoint),
                         ("https://another.seetacloud.com:8443", SECRET))

    def test_env_never_executes_or_expands(self):
        sentinel = self.directory / "executed"
        invalid = [f"$(touch {sentinel})", "`true`", "$HOME", "abc;true", "abc|true", "abc\\def"]
        for attack in invalid:
            with self.subTest(attack=attack):
                self.write_env(f"OMNI_INFERENCE_URL={URL}\nOMNI_INFERENCE_API_KEY={attack}\n")
                with self.assertRaises(probe.ProbeError):
                    probe.load_config(self.env)
                self.assertFalse(sentinel.exists())

    def test_env_rejects_unknown_duplicate_and_missing_assignments(self):
        for suffix in ("\nOTHER=abc\n", "\nOMNI_INFERENCE_API_KEY=duplicate\n"):
            self.write_env()
            self.env.write_text(self.env.read_text() + suffix)
            with self.assertRaises(probe.ProbeError):
                probe.load_config(self.env)
        self.write_env(f"OMNI_INFERENCE_URL={URL}\n")
        with self.assertRaises(probe.ProbeError):
            probe.load_config(self.env)

    def test_permission_symlink_and_endpoint_json_validation(self):
        self.env.chmod(0o644)
        with self.assertRaisesRegex(probe.ProbeError, "0600"):
            probe.load_config(self.env)
        self.env.chmod(0o600)
        symlink = self.directory / "link.env"
        symlink.symlink_to(self.env)
        with self.assertRaisesRegex(probe.ProbeError, "symlink"):
            probe.load_config(symlink)
        endpoint = self.directory / "endpoint.json"
        for value in ("[]", "{}", "not json", '{"url":42}'):
            endpoint.write_text(value)
            with self.assertRaises(probe.ProbeError):
                probe.load_config(self.env, endpoint)

    def test_url_safety(self):
        accepted = (URL, "http://127.0.0.1:6006", "http://[::1]:6006", "http://localhost:6006")
        for url in accepted:
            self.assertEqual(probe.validate_url(url), url)
        rejected = ("http://example.com", "https://secret@example.com", "https://example.com/?key=secret",
                    "https://example.com/#secret", "https://10.0.0.1", "https://169.254.169.254",
                    "file:///etc/passwd", "https://example.com/v1", "https://example.com:0",
                    "https://example.com:65536", "https://bad..example.com", "https://exam ple.com",
                    "https://example.com?", "https://example.com#", "https://example.com\\@evil.com")
        for url in rejected:
            with self.subTest(url=url), self.assertRaises(probe.ProbeError):
                probe.validate_url(url)

    def test_cli_report_no_secret_and_safe_output(self):
        output = self.directory / "result.json"
        captured = io.StringIO()
        with mock.patch.object(probe, "HTTPTransport", return_value=FakeHTTP()), redirect_stdout(captured):
            code = probe.main(["ready", "--env", str(self.env), "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.read_text())["status"], "pass")
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        self.assertNotIn(SECRET, output.read_text() + captured.getvalue())
        self.assertNotIn("Authorization", captured.getvalue())
        self.assertNotIn("embedding\": [", captured.getvalue())
        original = self.env.read_bytes()
        with redirect_stdout(io.StringIO()):
            code = probe.main(["ready", "--env", str(self.env), "--output", str(self.env)])
        self.assertEqual(code, 1)
        self.assertEqual(self.env.read_bytes(), original)

    def test_output_symlink_is_rejected(self):
        link = self.directory / "output.json"
        link.symlink_to(self.env)
        with self.assertRaises(probe.ProbeError):
            probe.write_report(link, {"status": "fail"})


class ContractTests(unittest.TestCase):
    def test_sequential_complete_test_contract(self):
        service, fake, _ = configured()
        result = service.run("test")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(len(fake.requests), 5)
        self.assertEqual([step["name"] for step in result["steps"]],
                         ["ready", "query", "document", "image", "video"])
        self.assertEqual(result["boot_id"], BOOT)
        self.assertGreater(result["check_count"], 50)
        self.assertEqual(result["check_count"], result["passed_check_count"])
        query, document = [json.loads(request.data) for request in fake.requests[1:3]]
        self.assertEqual(query["input_type"], "query")
        self.assertNotIn("titles", query)
        self.assertEqual(set(query), {"input", "input_type", "dimensions"})
        self.assertEqual(document["input_type"], "document")
        self.assertEqual(len(document["titles"]), len(document["input"]))
        self.assertEqual(len(document["input"]), 2)
        image, video = [mime_parts(request) for request in fake.requests[3:]]
        self.assertEqual([part.get_param("name", header="content-disposition") for part in image], ["file"])
        self.assertEqual([part.get_param("name", header="content-disposition") for part in video], ["files", "files"])
        self.assertEqual(video[0].get_payload(decode=True), probe.synthetic_png(0))
        self.assertEqual(video[1].get_payload(decode=True), probe.synthetic_png(1))
        for request in fake.requests[3:]:
            self.assertTrue(request.full_url.endswith("?dimensions=768"))
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertNotIn("embedding", result)

    def test_ready_is_light_get_and_missing_ready_is_failure(self):
        service, fake, _ = configured()
        self.assertEqual(service.run("ready")["status"], "pass")
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(fake.requests[0].get_method(), "GET")
        self.assertTrue(fake.requests[0].full_url.endswith("/health/ready"))
        value = ready_payload()
        del value["ready"]
        service, _, _ = configured(FakeHTTP([(200, {}, value)]))
        result = service.run("ready")
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["error"], "validation_ready_true")

    def test_ready_checks_pinned_identity_and_boot_id(self):
        for change in ({"model": "other"}, {"model_revision": "other"}, {"boot_id": SECRET},
                       {"ready": False}, {"status": "degraded"}):
            with self.subTest(change=change):
                service, fake, _ = configured(FakeHTTP([(200, {}, ready_payload(**change))]))
                result = service.run("test")
                self.assertEqual(result["status"], "fail")
                self.assertEqual(len(fake.requests), 1)
                self.assertNotIn(SECRET, json.dumps(result))

    def test_boot_change_between_requests_stops(self):
        changed = vector_payload()
        changed["boot_id"] = "2d2dc124-fcff-4432-913e-0653c4689824"
        service, fake, _ = configured(FakeHTTP([(200, {}, ready_payload()), (200, {}, changed)]))
        result = service.run("test")
        self.assertEqual(result["error"], "validation_boot_id_stable")
        self.assertEqual(len(fake.requests), 2)

    def test_malformed_vectors_never_pass(self):
        mutations = {
            "wrong_model": lambda v: v.update(model="other"),
            "wrong_revision": lambda v: v.update(model_revision="other"),
            "dimensions": lambda v: v.update(dimensions=512),
            "dimension_bool": lambda v: v.update(dimensions=True),
            "row_count": lambda v: v.update(data=[]),
            "row_index": lambda v: v["data"][0].update(index=1),
            "index_bool": lambda v: v["data"][0].update(index=False),
            "row_object": lambda v: v["data"][0].update(object="other"),
            "length": lambda v: v["data"][0]["embedding"].pop(),
            "nan": lambda v: v["data"][0]["embedding"].__setitem__(3, math.nan),
            "inf": lambda v: v["data"][0]["embedding"].__setitem__(3, math.inf),
            "huge_integer": lambda v: v["data"][0]["embedding"].__setitem__(3, 10 ** 1000),
            "bool": lambda v: v["data"][0]["embedding"].__setitem__(3, False),
            "string": lambda v: v["data"][0]["embedding"].__setitem__(3, SECRET),
            "zero_norm": lambda v: v["data"][0]["embedding"].__setitem__(0, 0),
            "wrong_norm": lambda v: v["data"][0]["embedding"].__setitem__(0, 0.5),
        }
        for name, change in mutations.items():
            with self.subTest(damage=name):
                value = vector_payload()
                change(value)
                service, fake, _ = configured(FakeHTTP([(200, {}, ready_payload()), (200, {}, value)]))
                result = service.run("test")
                self.assertEqual(result["status"], "fail")
                self.assertEqual(len(fake.requests), 2)
                self.assertNotIn(SECRET, json.dumps(result))

    def test_invalid_json_and_unknown_shape_do_not_pass(self):
        for body in (b"invalid", b"null", b"[]", b"\xff"):
            with self.subTest(body=body):
                service, _, _ = configured(FakeHTTP([(200, {}, body)]))
                self.assertEqual(service.run("ready")["status"], "fail")


class RetryAndDeadlineTests(unittest.TestCase):
    def test_429_seconds_and_http_date_then_success(self):
        for header, expected in (("2", 2.1), ("Tue, 14 Nov 2023 22:13:23 GMT", 3.1)):
            with self.subTest(header=header):
                service, fake, clock = configured(FakeHTTP([(429, {"Retry-After": header}, {})]), timeout=10)
                result = service.run("ready")
                self.assertEqual(result["status"], "pass")
                self.assertEqual(len(fake.requests), 2)
                self.assertAlmostEqual(clock.sleeps[0], expected)
                self.assertLess(fake.timeouts[1], fake.timeouts[0])

    def test_429_respects_retry_limit_and_deadline(self):
        service, fake, clock = configured(FakeHTTP([(429, {"Retry-After": "30"}, {})]), timeout=3)
        self.assertEqual(service.run("ready")["error"], "retry_after_exceeds_deadline")
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(clock.sleeps, [])
        service, fake, clock = configured(FakeHTTP([(429, {"Retry-After": "9" * 400}, {})]))
        self.assertEqual(service.run("ready")["error"], "retry_delay_invalid")
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(clock.sleeps, [])
        responses = [(429, {"Retry-After": "0"}, {})] * 3
        service, fake, clock = configured(FakeHTTP(responses), retries=2)
        self.assertEqual(service.run("ready")["error"], "http_429_retry_limit")
        self.assertEqual(len(fake.requests), 3)
        self.assertEqual(len(clock.sleeps), 2)

    def test_transport_timeout_and_5xx_are_never_replayed(self):
        for failure in (TimeoutError(SECRET), urllib.error.URLError(SECRET),
                        (500, {}, {"error": SECRET}), (502, {}, {}), (503, {}, {})):
            with self.subTest(failure=type(failure).__name__):
                # Successful ready, then an ambiguous or failed inference request.
                service, fake, clock = configured(FakeHTTP([(200, {}, ready_payload()), failure]))
                result = service.run("test")
                self.assertEqual(result["status"], "fail")
                self.assertEqual(len(fake.requests), 2)
                self.assertEqual(clock.sleeps, [])
                self.assertNotIn(SECRET, json.dumps(result))

    def test_401_302_and_422_are_not_replayed(self):
        for status in (401, 302, 422):
            service, fake, clock = configured(FakeHTTP([(status, {}, {})]))
            self.assertEqual(service.run("ready")["error"], f"http_{status}")
            self.assertEqual(len(fake.requests), 1)
            self.assertEqual(clock.sleeps, [])

    def test_wait_ready_polls_valid_loading_and_stops_on_identity_error(self):
        loading = (503, {}, ready_payload(ready=False, status="loading"))
        service, fake, clock = configured(FakeHTTP([loading]), timeout=10, interval=2)
        self.assertEqual(service.run("wait-ready")["status"], "pass")
        self.assertEqual(len(fake.requests), 2)
        self.assertEqual(clock.sleeps, [2])
        service, fake, clock = configured(FakeHTTP([(503, {}, ready_payload(model="other", ready=False))]))
        self.assertEqual(service.run("wait-ready")["error"], "validation_model_identity")
        self.assertEqual(len(fake.requests), 1)

    def test_wait_ready_is_bounded_and_never_calls_inference(self):
        loading = (503, {}, ready_payload(ready=False, status="loading"))
        service, fake, clock = configured(FakeHTTP([loading] * 10), timeout=5, interval=2)
        self.assertEqual(service.run("wait-ready")["error"], "readiness_deadline_exceeded")
        self.assertEqual(clock.sleeps, [2, 2])
        self.assertEqual(len(fake.requests), 3)
        self.assertTrue(all(request.get_method() == "GET" for request in fake.requests))

    def test_request_timeouts_use_remaining_budget_and_late_results_fail(self):
        clock = Clock()

        def late(request, timeout):
            clock.value += timeout + 1
            return 200, {}, json.dumps(ready_payload()).encode()

        service, fake, _ = configured(FakeHTTP([late]), clock=clock, timeout=3, http_timeout=60)
        self.assertEqual(service.run("ready")["error"], "total_deadline_exceeded")
        self.assertEqual(fake.timeouts, [3])


class MediaAndTransportTests(unittest.TestCase):
    def test_transport_total_wait_timeout_is_terminal_without_replay(self):
        transport = probe.HTTPTransport()
        request = probe.urllib.request.Request(URL + "/health/ready")
        with mock.patch.object(probe.threading, "Thread") as thread:
            with self.assertRaisesRegex(probe.ProbeError, "transport_timeout_unknown_execution"):
                transport(request, 0.001)
        self.assertEqual(thread.call_count, 1)
        self.assertTrue(thread.call_args.kwargs["daemon"])

    def test_transport_worker_never_exposes_exception_text(self):
        transport = probe.HTTPTransport()
        request = probe.urllib.request.Request(URL + "/health/ready")
        with mock.patch.object(transport, "_exchange", side_effect=RuntimeError(SECRET)):
            with self.assertRaises(probe.ProbeError) as caught:
                transport(request, 1)
        self.assertNotIn(SECRET, str(caught.exception))
        self.assertEqual(caught.exception.code, "transport_failure_unknown_execution")

    def test_png_dimensions_structure_crc_and_distinct_frames(self):
        self.assertNotEqual(probe.synthetic_png(0), probe.synthetic_png(1))
        for index in (0, 1):
            image = probe.synthetic_png(index)
            self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
            position, kinds, raw = 8, [], b""
            while position < len(image):
                length = struct.unpack("!I", image[position:position + 4])[0]
                kind = image[position + 4:position + 8]
                payload = image[position + 8:position + 8 + length]
                crc = struct.unpack("!I", image[position + 8 + length:position + 12 + length])[0]
                self.assertEqual(crc, zlib.crc32(kind + payload) & 0xFFFFFFFF)
                if kind == b"IHDR":
                    self.assertEqual(struct.unpack("!IIBBBBB", payload), (64, 64, 8, 2, 0, 0, 0))
                if kind == b"IDAT":
                    raw += payload
                kinds.append(kind)
                position += 12 + length
            self.assertEqual(kinds, [b"IHDR", b"IDAT", b"IEND"])
            decoded = zlib.decompress(raw)
            self.assertEqual(len(decoded), (64 * 3 + 1) * 64)
            self.assertTrue(all(decoded[offset] == 0 for offset in range(0, len(decoded), 193)))

    def test_transport_disables_proxy_and_redirect_and_keeps_tls_verification(self):
        handlers = []

        def build(*values):
            handlers.extend(values)
            return object()

        with mock.patch.object(probe.urllib.request, "build_opener", side_effect=build):
            probe.HTTPTransport()
        proxy = next(handler for handler in handlers if isinstance(handler, probe.urllib.request.ProxyHandler))
        self.assertEqual(proxy.proxies, {})
        redirect = next(handler for handler in handlers if isinstance(handler, probe.NoRedirect))
        self.assertIsNone(redirect.redirect_request(None, None, 302, "", {}, "https://evil.example"))
        tls = next(handler for handler in handlers if isinstance(handler, probe.urllib.request.HTTPSHandler))
        self.assertTrue(tls._context.check_hostname)
        self.assertEqual(tls._context.verify_mode, probe.ssl.CERT_REQUIRED)


if __name__ == "__main__":
    unittest.main()
