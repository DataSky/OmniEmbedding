"""Control-plane safety and lifecycle tests; all tokens/HTTP responses are fake."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error


SPEC = importlib.util.spec_from_file_location(
    "autodl_service_manager", Path(__file__).parents[1] / "scripts/autodl_service_manager.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
TOKEN = "synthetic-developer-token-no-real-account"


def success(data):
    return {"code": "Success", "data": data, "msg": "never echo this body " + TOKEN}


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, path, payload, token, timeout, **kwargs):
        self.calls.append({"method": method, "path": path, "payload": payload, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClock:
    def __init__(self):
        self.value = 0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class FakeResponse:
    def __init__(self, value, code=200):
        self.raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.code = code

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        return self.raw[:size]


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.config_path = self.directory / "RTX5090.control.json"
        self.config_path.write_text(json.dumps({
            "instance_uuid": "pro-synthetic123", "endpoint_file": "endpoint.json"}))
        self.config = m.load_config(self.config_path)

    def tearDown(self):
        self.temporary.cleanup()

    def manager(self, *responses, clock=None):
        transport = FakeTransport(*responses)
        kwargs = {} if clock is None else {"clock": clock.now, "sleep": clock.sleep}
        return m.ServiceManager(self.config, TOKEN, transport=transport, **kwargs), transport

    def test_defaults_and_relative_paths(self):
        self.assertEqual(self.config["endpoint_file"], self.directory / "endpoint.json")
        self.assertEqual(self.config["payg_price_unit"], "milli_yuan_per_hour")
        self.assertEqual(self.config["max_hourly_yuan"], 8)
        self.assertIn("exec bash ./run-autodl.sh", self.config["start_command"])
        self.assertNotIn("bootstrap", self.config["start_command"])

    def test_host_and_budget_cannot_be_overridden_unsafely(self):
        for update in ({"api_base": "https://evil.example"}, {"max_hourly_yuan": 9},
                       {"poll_interval": 31}, {"timeout_seconds": float("nan")}):
            with self.subTest(update=update):
                self.config_path.write_text(json.dumps({"instance_uuid": "pro-synthetic123", **update}))
                with self.assertRaises(m.ControlError):
                    m.load_config(self.config_path)

    def test_secure_token_file_and_environment(self):
        key = self.directory / "account.token"
        key.write_text(TOKEN + "\n")
        key.chmod(0o600)
        self.assertEqual(m.read_token({"token_file": key}), TOKEN)
        self.assertEqual(m.read_token({"token_env": "FAKE_TOKEN"}, {"FAKE_TOKEN": TOKEN}), TOKEN)
        with self.assertRaises(m.ControlError):
            m.read_token({"token_env": "FAKE_TOKEN"}, {})

    def test_token_permissions_symlink_and_owner_are_checked(self):
        key = self.directory / "account.token"
        key.write_text(TOKEN)
        key.chmod(0o644)
        with self.assertRaises(m.ControlError):
            m.read_token({"token_file": key})
        key.chmod(0o600)
        link = self.directory / "link.token"
        link.symlink_to(key)
        with self.assertRaises(m.ControlError):
            m.read_token({"token_file": link})
        with mock.patch.object(m.os, "getuid", return_value=os.getuid() + 100):
            with self.assertRaises(m.ControlError):
                m.read_token({"token_file": key})

    def test_fifo_token_path_is_rejected_without_blocking(self):
        fifo = self.directory / "account.fifo"
        os.mkfifo(fifo, mode=0o600)
        with self.assertRaises(m.ControlError):
            m.read_token({"token_file": fifo})

    def test_status_outputs_allowlist_and_redacts_actual_token(self):
        manager, transport = self.manager(success("running"), success({
            "payg_price": 2880, "root_password": "PASSWORD_MUST_NOT_LEAK",
            "ssh_command": "SSH_MUST_NOT_LEAK", "jupyter_token": "JUPYTER_MUST_NOT_LEAK",
            "snapshot_gpu_alias_name": "RTX 5090 " + TOKEN,
            "service_6006_domain": "u123-ab.bjb2.seetacloud.com:8443",
            "usage_info": {"jupyter_token": "NESTED_SECRET"}}))
        output = json.dumps(manager.status())
        for secret in (TOKEN, "PASSWORD_MUST_NOT_LEAK", "JUPYTER_MUST_NOT_LEAK",
                       "SSH_MUST_NOT_LEAK", "NESTED_SECRET"):
            self.assertNotIn(secret, output)
        self.assertIn("https://u123-ab.bjb2.seetacloud.com:8443", output)
        self.assertEqual(transport.calls[0]["method"], "GET")
        self.assertEqual(transport.calls[0]["payload"], {"instance_uuid": "pro-synthetic123"})

    def test_query_fallback_only_for_parameter_rejection(self):
        manager, transport = self.manager({"code": "RequestParameterIsWrong"}, success("running"))
        self.assertEqual(manager.state(), "running")
        self.assertFalse(transport.calls[0].get("query", False))
        self.assertTrue(transport.calls[1]["query"])
        manager, transport = self.manager({"code": "InvalidToken", "msg": TOKEN})
        with self.assertRaises(m.ControlError) as caught:
            manager.state()
        self.assertNotIn(TOKEN, str(caught.exception))
        self.assertEqual(len(transport.calls), 1)

    def test_running_power_on_is_idempotent(self):
        manager, transport = self.manager(success("running"))
        self.assertEqual(manager.power_on(), {"status": "running", "changed": False})
        self.assertEqual(len(transport.calls), 1)

    def test_stopped_power_on_checks_price_and_sends_start_command(self):
        manager, transport = self.manager(success("stopped"), success({"payg_price": 2880}), success(None))
        output = manager.power_on()
        self.assertEqual(output["hourly_yuan"], 2.88)
        self.assertFalse(output["service_verified"])
        write = transport.calls[2]
        self.assertTrue(write["mutating"])
        self.assertEqual(write["payload"], {"instance_uuid": "pro-synthetic123", "payload": "gpu",
                                           "start_command": self.config["start_command"]})

    def test_budget_price_units_and_boundary(self):
        for raw, unit, expected in ((2880, "milli_yuan_per_hour", 2.88),
                                    (288, "fen_per_hour", 2.88),
                                    ("2.88", "yuan_per_hour", 2.88),
                                    (8000, "milli_yuan_per_hour", 8)):
            with self.subTest(raw=raw, unit=unit):
                self.assertEqual(m.hourly_price({"payg_price": raw},
                                 {"payg_price_unit": unit, "max_hourly_yuan": 8}), expected)

    def test_missing_ambiguous_invalid_and_over_budget_fail_before_write(self):
        for snapshot in ({}, {"payg_price": 8001}, {"payg_price": -2},
                         {"payg_price": "NaN"}, {"payg_price": True},
                         {"payg_price": 2880, "payg_price_unit": "unknown"}):
            with self.subTest(snapshot=snapshot):
                manager, transport = self.manager(success("stopped"), success(snapshot))
                with self.assertRaises(m.ControlError):
                    manager.power_on()
                self.assertEqual(len(transport.calls), 2)
        with self.assertRaises(m.ControlError):
            m.hourly_price({"payg_price": 2880}, {"max_hourly_yuan": 8})

    def test_transitional_state_does_not_start_or_stop(self):
        for action in ("power_on", "power_off"):
            manager, transport = self.manager(success("starting"))
            with self.assertRaises(m.ControlError):
                getattr(manager, action)()
            self.assertEqual(len(transport.calls), 1)

    def test_power_on_wait_is_bounded_and_never_repeats_write(self):
        clock = FakeClock()
        manager, transport = self.manager(success("stopped"), success({"payg_price": 2880}),
            success(None), success("starting"), success("starting"), success("starting"), clock=clock)
        with self.assertRaisesRegex(m.ControlError, "wait timed out"):
            manager.power_on(wait=True, wait_timeout=8)
        self.assertEqual(clock.sleeps, [5, 3])
        self.assertEqual(sum(call["method"] == "POST" for call in transport.calls), 1)

    def test_successful_power_on_wait_and_explicit_power_off(self):
        clock = FakeClock()
        manager, _ = self.manager(success("stopped"), success({"payg_price": 2880}),
                                  success(None), success("running"), clock=clock)
        self.assertEqual(manager.power_on(wait=True)["status"], "running")
        manager, transport = self.manager(success("running"), success(None), success("stopped"))
        self.assertEqual(manager.power_off(wait=True)["status"], "stopped")
        self.assertTrue(transport.calls[1]["path"].endswith("/power_off"))

    def test_image_save_refuses_running_and_never_shuts_down(self):
        manager, transport = self.manager(success("running"))
        with self.assertRaisesRegex(m.ControlError, "already stopped"):
            manager.image_save("test-image")
        self.assertEqual(len(transport.calls), 1)

    def test_image_save_and_wait(self):
        rows = {"list": [{"image_uuid": "image-test123", "name": "sample", "status": "finished"}], "max_page": 1}
        manager, transport = self.manager(success("stopped"), success({"image_uuid": "image-test123"}), success(rows))
        self.assertEqual(manager.image_save("sample", wait=True)["status"], "finished")
        self.assertTrue(transport.calls[1]["mutating"])
        self.assertFalse(transport.calls[2]["mutating"])

    def test_image_save_missing_id_requires_manual_inspection(self):
        manager, transport = self.manager(success("stopped"), success({}))
        with self.assertRaises(m.ControlError) as caught:
            manager.image_save("sample")
        self.assertTrue(caught.exception.outcome_unknown)
        self.assertEqual(len(transport.calls), 2)

    def test_image_save_wait_failure_retains_resume_id(self):
        clock = FakeClock()
        empty = success({"list": [], "max_page": 0})
        manager, _ = self.manager(success("stopped"), success({"image_uuid": "image-test123"}),
                                 empty, empty, clock=clock)
        with self.assertRaises(m.ControlError) as caught:
            manager.image_save("sample", wait=True, wait_timeout=1)
        self.assertEqual(caught.exception.image_uuid, "image-test123")

    def test_images_traverse_all_pages_and_filter_fields(self):
        manager, transport = self.manager(
            success({"list": [{"image_uuid": "image-first", "name": "name " + TOKEN,
                               "status": "finished", "root_password": "NOPE"}], "max_page": 2}),
            success({"list": [{"image_uuid": "image-second", "status": "saving"}], "max_page": 2}))
        rows = manager.images()
        self.assertEqual(len(rows), 2)
        self.assertEqual([call["payload"]["page_index"] for call in transport.calls], [1, 2])
        self.assertNotIn("NOPE", json.dumps(rows))
        self.assertNotIn(TOKEN, json.dumps(rows))

    def test_image_pagination_anomalies_are_not_silently_accepted(self):
        for data in ({"list": [], "max_page": 2}, {"list": [], "max_page": 1001}):
            manager, _ = self.manager(success(data))
            with self.assertRaises(m.ControlError):
                manager.images()
        page = {"list": [{"image_uuid": "image-repeat"}], "max_page": 2}
        manager, _ = self.manager(success(page), success(page))
        with self.assertRaises(m.ControlError):
            manager.images()

    def test_image_wait_failure_and_timeout_are_bounded(self):
        manager, _ = self.manager(success({"list": [{"image_uuid": "image-test123", "status": "failed"}], "max_page": 1}))
        with self.assertRaisesRegex(m.ControlError, "save failed"):
            manager.image_wait("image-test123")
        clock = FakeClock()
        empty = success({"list": [], "max_page": 0})
        manager, _ = self.manager(empty, empty, clock=clock)
        with self.assertRaisesRegex(m.ControlError, "wait timed out"):
            manager.image_wait("image-test123", wait_timeout=2)
        self.assertEqual(clock.sleeps, [2])

    def test_image_pagination_requests_share_wait_deadline(self):
        clock = FakeClock()
        page = success({"list": [{"image_uuid": "image-unrelated"}], "max_page": 2})
        class SlowTransport(FakeTransport):
            def request(inner, *args, **kwargs):
                inner.calls.append({"timeout": args[4]})
                clock.value += 2
                result = dict(page)
                result["data"] = {"list": [{"image_uuid": "image-other" + str(len(inner.calls))}],
                                  "max_page": 2}
                return result
        transport = SlowTransport()
        manager = m.ServiceManager(self.config, TOKEN, transport=transport,
                                   clock=clock.now, sleep=clock.sleep)
        with self.assertRaisesRegex(m.ControlError, "wait timed out"):
            manager.image_wait("image-test123", wait_timeout=3)
        self.assertEqual([call["timeout"] for call in transport.calls], [3, 1])

    def test_endpoint_validation_rejects_exfiltration_and_ambiguous_urls(self):
        for url in ("http://u.autodl.com:8443", "u.autodl.com.evil.example", "evilautodl.com",
                    "https://user:pass@u.autodl.com", "https://u.autodl.com/x",
                    "https://u.autodl.com?token=secret", "https://127.0.0.1", "u.autodl.com:22",
                    "https://u.autodl.com#fragment", "u.autodl.com:invalid"):
            with self.subTest(url=url), self.assertRaises(m.ControlError):
                m.endpoint_url(url)
        self.assertEqual(m.endpoint_url("u-1.bjb2.seetacloud.com:8443"),
                         "https://u-1.bjb2.seetacloud.com:8443")

    def test_atomic_endpoint_preserves_identity_and_invalidates_stale_ready(self):
        path = self.config["endpoint_file"]
        path.write_text(json.dumps({"url": "https://old.autodl.com", "status": "ready",
                                    "model": "google/embeddinggemma-2", "revision": "pinned"}))
        manager, _ = self.manager(success({"service_6006_domain": "u-1.bjb2.seetacloud.com:8443"}))
        self.assertEqual(manager.refresh_endpoint()["endpoint_status"], "discovered")
        value = json.loads(path.read_text())
        self.assertEqual(value["revision"], "pinned")
        self.assertEqual(value["model"], "google/embeddinggemma-2")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(self.directory.glob(".endpoint-*")), [])

    def test_failed_endpoint_replace_leaves_previous_file_unchanged(self):
        path = self.config["endpoint_file"]
        original = '{"url":"https://old.autodl.com","revision":"pinned"}'
        path.write_text(original)
        with mock.patch.object(m.os, "replace", side_effect=OSError("sensitive path " + TOKEN)):
            with self.assertRaises(m.ControlError):
                m.atomic_endpoint(path, {"url": "https://new.autodl.com"})
        self.assertEqual(path.read_text(), original)
        self.assertEqual(list(self.directory.glob(".endpoint-*")), [])

    def test_endpoint_discovery_invalidates_verification_even_for_same_address(self):
        path = self.config["endpoint_file"]
        url = "https://new.autodl.com"
        path.write_text(json.dumps({"url": url, "status": "ready", "verified_at": "stale",
                                    "boot_id": "old-boot", "revision": "pinned"}))
        self.assertEqual(m.atomic_endpoint(path, {"url": url}), "discovered")
        value = json.loads(path.read_text())
        self.assertNotIn("verified_at", value)
        self.assertNotIn("boot_id", value)
        self.assertEqual(value["revision"], "pinned")

    def test_write_unrecognized_code_requires_manual_inspection(self):
        manager, transport = self.manager(success("stopped"), success({"payg_price": 2880}), {})
        with self.assertRaises(m.ControlError) as caught:
            manager.power_on()
        self.assertTrue(caught.exception.outcome_unknown)
        self.assertEqual(len(transport.calls), 3)

    def test_endpoint_symlink_is_refused(self):
        target = self.directory / "target.json"
        target.write_text("{}")
        self.config["endpoint_file"].symlink_to(target)
        with self.assertRaises(m.ControlError):
            m.atomic_endpoint(self.config["endpoint_file"], {"url": "https://new.autodl.com"})
        self.assertEqual(target.read_text(), "{}")

    def test_http_transport_get_body_and_no_environment_proxy(self):
        transport = m.HTTPTransport()
        transport.opener = mock.Mock()
        transport.opener.open.return_value = FakeResponse(success("running"))
        transport.request("GET", m.PREFIX + "status", {"instance_uuid": "pro-fake123"}, TOKEN, 10)
        request = transport.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, m.API_BASE + m.PREFIX + "status")
        self.assertEqual(json.loads(request.data), {"instance_uuid": "pro-fake123"})
        self.assertEqual(request.get_header("Authorization"), TOKEN)
        self.assertIsNone(m.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example"))
        with mock.patch.dict(os.environ, {"https_proxy": "https://evil.example"}):
            real_transport = m.HTTPTransport()
            proxy_handlers = [handler for handler in real_transport.opener.handlers
                              if isinstance(handler, m.urllib.request.ProxyHandler)]
            self.assertFalse(any(handler.proxies for handler in proxy_handlers))

    def test_write_timeout_is_unknown_and_is_never_retried(self):
        transport = m.HTTPTransport()
        transport.opener = mock.Mock()
        transport.opener.open.side_effect = TimeoutError("headers and token " + TOKEN)
        with self.assertRaises(m.ControlError) as caught:
            transport.request("POST", m.PREFIX + "power_on", {}, TOKEN, 10, mutating=True)
        self.assertTrue(caught.exception.outcome_unknown)
        self.assertNotIn(TOKEN, str(caught.exception))
        self.assertEqual(transport.opener.open.call_count, 1)

    def test_write_bad_response_and_server_error_are_unknown(self):
        for response in (FakeResponse(b"not json " + TOKEN.encode()), FakeResponse([], 200),
                         FakeResponse(success(None), 503)):
            transport = m.HTTPTransport()
            transport.opener = mock.Mock()
            transport.opener.open.return_value = response
            with self.assertRaises(m.ControlError) as caught:
                transport.request("POST", m.PREFIX + "image/save", {}, TOKEN, 10, mutating=True)
            self.assertTrue(caught.exception.outcome_unknown)
            self.assertNotIn(TOKEN, str(caught.exception))

    def test_cli_does_not_print_local_exception_details(self):
        with mock.patch.object(m, "load_config", side_effect=OSError("private credential " + TOKEN)):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(m.main(["--config", str(self.config_path), "status"]), 2)
        self.assertNotIn(TOKEN, output.getvalue())
        self.assertFalse(json.loads(output.getvalue())["ok"])


if __name__ == "__main__":
    unittest.main()
