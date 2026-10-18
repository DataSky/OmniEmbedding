#!/usr/bin/env python3
"""Small, standard-library AutoDL Pro control-plane client.

No account credential is accepted on the command line or emitted in output. The
inference key is deliberately outside this tool's responsibility. See the Pro
API: https://www.autodl.com/docs/instance_pro_api/ .
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import re
import ssl
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


API_BASE = "https://api.autodl.com"
PREFIX = "/api/v1/dev/instance/pro/"
START_COMMAND = ("cd /root/omniembedding-service && mkdir -p logs run "
                 "&& exec bash ./run-autodl.sh")
MAX_BYTES = 2 * 1024 * 1024
MAX_PAGES = 1000
# Verified 2026-10-18 in the public AutoDL Pro startup confirmation UI:
# https://www.autodl.com/assets/index.454ad1a2.js : payg_price / 1e3 yuan/hour.
# API prose names the field but does not itself state the currency multiplier.
DEFAULT_PRICE_UNIT = "milli_yuan_per_hour"


class ControlError(Exception):
    def __init__(self, message, *, outcome_unknown=False, code=None, image_uuid=None):
        super().__init__(message)
        self.outcome_unknown = outcome_unknown
        self.code = code
        self.image_uuid = image_uuid


def require(condition, message):
    if not condition:
        raise ControlError(message)


def finite_number(value, field, *, maximum=3600):
    require(not isinstance(value, bool) and isinstance(value, (int, float)),
            field + " must be a positive finite number")
    require(math.isfinite(value) and 0 < value <= maximum,
            field + " is outside its allowed range")
    return float(value)


def relative_path(value, directory, field):
    require(isinstance(value, str) and bool(value) and "\x00" not in value,
            field + " must be a file path")
    # Do not resolve symlinks before the O_NOFOLLOW credential check.
    return Path(os.path.abspath(directory / value))


def load_json_file(path, *, optional=False):
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(65537)
        require(len(raw) <= 65536, "JSON file exceeds size limit")
        value = json.loads(raw)
        require(isinstance(value, dict), "JSON file must contain an object")
        return value
    except FileNotFoundError:
        if optional:
            return {}
        raise ControlError("configuration file is missing") from None
    except (OSError, ValueError, UnicodeError, RecursionError):
        raise ControlError("JSON file cannot be read or is invalid") from None


def load_config(path):
    path = Path(os.path.abspath(path))
    config = load_json_file(path)
    require(config.get("api_base", API_BASE) == API_BASE,
            "api_base must be https://api.autodl.com")
    instance = config.get("instance_uuid")
    require(isinstance(instance, str) and re.fullmatch(r"pro-[A-Za-z0-9-]{5,80}", instance),
            "instance_uuid must be an AutoDL Pro instance ID")
    config["api_base"] = API_BASE
    config["endpoint_file"] = relative_path(config.get("endpoint_file", "endpoint.json"),
                                             path.parent, "endpoint_file")
    if config.get("token_file"):
        config["token_file"] = relative_path(config["token_file"], path.parent, "token_file")
    else:
        env = config.get("token_env", "AUTODL_DEVELOPER_TOKEN")
        require(isinstance(env, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", env),
                "token_env must name an environment variable")
        config["token_env"] = env
    config["timeout_seconds"] = finite_number(config.get("timeout_seconds", 30),
                                                "timeout_seconds", maximum=120)
    config["poll_interval"] = finite_number(config.get("poll_interval", 5),
                                             "poll_interval", maximum=30)
    config["max_hourly_yuan"] = finite_number(config.get("max_hourly_yuan", 8),
                                               "max_hourly_yuan", maximum=8)
    config["payg_price_unit"] = config.get("payg_price_unit", DEFAULT_PRICE_UNIT)
    require(config["payg_price_unit"] in {"milli_yuan_per_hour", "yuan_per_hour", "fen_per_hour"},
            "payg_price_unit is invalid")
    command = config.get("start_command", START_COMMAND)
    require(isinstance(command, str) and command.strip() and len(command) <= 16384
            and "\x00" not in command, "start_command is invalid")
    config["start_command"] = command
    return config


def read_token(config, environ=None):
    if config.get("token_file"):
        descriptor = None
        try:
            descriptor = os.open(config["token_file"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            info = os.fstat(descriptor)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                    and stat.S_IMODE(info.st_mode) == 0o600,
                    "token_file must be a current-user regular file with mode 0600")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = None
                raw = handle.read(16385)
            require(len(raw) <= 16384, "developer token exceeds size limit")
            token = raw.decode("ascii").strip()
        except (OSError, UnicodeError):
            raise ControlError("token_file cannot be securely read; require a non-symlink mode-0600 file") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
    else:
        token = (os.environ if environ is None else environ).get(config["token_env"], "")
    require(isinstance(token, str) and 16 <= len(token) <= 16384
            and token.isascii() and not any(char.isspace() for char in token)
            and all(ord(char) >= 33 and ord(char) <= 126 for char in token),
            "developer token is missing or invalid; configure token_file or token_env")
    return token


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


class HTTPTransport:
    """Fixed-host TLS with no environment proxy and no credential redirects."""
    def __init__(self):
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def request(self, method, path, payload, token, timeout, *, query=False, mutating=False):
        require(path.startswith(PREFIX) and "?" not in path and "#" not in path,
                "invalid API path")
        url = API_BASE + path
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if query:
            require(method == "GET", "query compatibility is read-only")
            url += "?" + urllib.parse.urlencode(payload)
            body = None
        request = urllib.request.Request(
            url, data=body, method=method,
            headers={"Authorization": token, "Content-Type": "application/json",
                     "Accept": "application/json"})
        try:
            try:
                response = self.opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                status_code = response.code
                raw = response.read(MAX_BYTES + 1)
        except (OSError, urllib.error.URLError, TimeoutError):
            raise ControlError(
                "write outcome unknown; inspect instance/image status before any manual retry"
                if mutating else "read transport failed; no platform change was requested",
                outcome_unknown=mutating) from None
        if len(raw) > MAX_BYTES:
            raise ControlError("API response exceeds size limit", outcome_unknown=mutating)
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            raise ControlError("API response is not valid JSON", outcome_unknown=mutating) from None
        if not 200 <= status_code < 300:
            raise ControlError("API HTTP " + str(status_code),
                               outcome_unknown=mutating and status_code >= 500)
        if not isinstance(result, dict):
            raise ControlError("API response must be an object", outcome_unknown=mutating)
        return result


def safe_code(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", value) else "UnrecognizedCode"


def endpoint_url(domain):
    require(isinstance(domain, str) and domain and len(domain) <= 255
            and not any(char.isspace() for char in domain), "6006 mapping is missing or invalid")
    value = domain if "://" in domain else "https://" + domain
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ControlError("6006 mapping has an invalid port") from None
    hostname = parsed.hostname or ""
    require(parsed.scheme == "https" and not parsed.username and not parsed.password
            and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment
            and port in {None, 443, 8443}
            and re.fullmatch(r"[a-zA-Z0-9.-]+", hostname)
            and any(hostname.lower().endswith("." + suffix)
                    for suffix in ("autodl.com", "seetacloud.com")),
            "6006 mapping must be an HTTPS AutoDL/seetacloud service host without credentials or extra paths")
    return urllib.parse.urlunsplit(("https", parsed.netloc.lower(), "", "", ""))


def hourly_price(snapshot, config):
    """Never infer currency units from the numerical size of a price."""
    raw = snapshot.get("payg_price")
    require(not isinstance(raw, bool) and isinstance(raw, (int, float, str)),
            "snapshot has no explicit payg_price; refuse paid startup")
    unit = snapshot.get("payg_price_unit", config.get("payg_price_unit"))
    require(unit in {"milli_yuan_per_hour", "yuan_per_hour", "fen_per_hour"},
            "payg_price unit is not established; configure verified payg_price_unit before paid startup")
    try:
        price = Decimal(str(raw))
        require(price.is_finite() and price > 0, "payg_price must be positive and finite")
        divisor = {"milli_yuan_per_hour": Decimal(1000), "fen_per_hour": Decimal(100),
                   "yuan_per_hour": Decimal(1)}[unit]
        result = price / divisor
    except InvalidOperation:
        raise ControlError("payg_price is invalid") from None
    require(result <= Decimal(str(config["max_hourly_yuan"])),
            "snapshot hourly price exceeds configured budget; no startup requested")
    return float(result)


def atomic_endpoint(path, update):
    path = Path(path)
    require(not path.is_symlink(), "endpoint_file must not be a symlink")
    existing = load_json_file(path, optional=True)
    existing.update(update)
    # Address discovery alone is not a successful inference/identity test.
    existing["status"] = "discovered"
    for field in ("verified_at", "last_verified_at", "boot_id", "verified_boot_id",
                  "ready_at", "health_checked_at"):
        existing.pop(field, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".endpoint-", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        raise ControlError("endpoint_file atomic update failed") from None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return existing["status"]


class ServiceManager:
    def __init__(self, config, token, *, transport=None, clock=time.monotonic, sleep=time.sleep):
        self.config = config
        self._token = token
        self.transport = transport or HTTPTransport()
        self.clock = clock
        self.sleep = sleep

    def _request_timeout(self, deadline):
        if deadline is None:
            return self.config["timeout_seconds"]
        remaining = deadline - self.clock()
        require(remaining > 0, "wait timed out; inspect status/images before any manual write retry")
        return min(self.config["timeout_seconds"], remaining)

    def _request(self, method, endpoint, payload=None, *, mutating=False, deadline=None):
        payload = payload or {"instance_uuid": self.config["instance_uuid"]}
        kwargs = {"mutating": mutating}
        result = self.transport.request(method, PREFIX + endpoint, payload, self._token,
                                        self._request_timeout(deadline), **kwargs)
        # Official examples specify JSON GET bodies. A previously observed Pro
        # deployment only accepted query parameters; this one read-only fallback
        # is limited to the exact parameter rejection, never network errors/writes.
        if method == "GET" and result.get("code") == "RequestParameterIsWrong":
            result = self.transport.request(method, PREFIX + endpoint, payload, self._token,
                                            self._request_timeout(deadline), query=True,
                                            mutating=False)
        code = safe_code(result.get("code"))
        if code != "Success":
            if self._token in code:
                code = "RedactedCode"
            if mutating and code == "UnrecognizedCode":
                raise ControlError("write outcome unknown; inspect instance/images before any manual retry",
                                   outcome_unknown=True)
            raise ControlError("AutoDL API rejected request: " + code, code=code)
        return result.get("data")

    def _safe(self, value):
        if isinstance(value, str):
            return value.replace(self._token, "[redacted]")[:500]
        if isinstance(value, dict):
            return {key: self._safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._safe(item) for item in value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value if math.isfinite(value) else None
        return value if value is None or isinstance(value, bool) else None

    def state(self, *, deadline=None):
        state = self._request("GET", "status", deadline=deadline)
        require(state in {"running", "stopped", "starting", "stopping", "creating", "restarting", "released"},
                "unrecognized instance state; no mutation requested")
        return state

    def snapshot(self):
        value = self._request("GET", "snapshot")
        require(isinstance(value, dict), "snapshot response is invalid")
        return value

    def status(self):
        state, snapshot = self.state(), self.snapshot()
        fields = ("region_sign", "payg_price", "origin_pay_price", "snapshot_gpu_alias_name",
                  "chip_corp", "cpu_arch", "expand_system_disk_size", "system_init_disk_size",
                  "service_6006_port_protocol")
        safe = {key: snapshot[key] for key in fields if key in snapshot
                and isinstance(snapshot[key], (str, int, float, bool, type(None)))}
        if snapshot.get("service_6006_domain"):
            safe["service_url"] = endpoint_url(snapshot["service_6006_domain"])
        if self.config.get("payg_price_unit"):
            safe["configured_payg_price_unit"] = self.config["payg_price_unit"]
        return self._safe({"instance_uuid": self.config["instance_uuid"], "status": state,
                           "snapshot": safe})

    def wait_state(self, target, timeout):
        timeout = finite_number(timeout, "wait_timeout")
        deadline = self.clock() + timeout
        while True:
            state = self.state(deadline=deadline)
            if state == target:
                return state
            require(state != "released", "instance has been released")
            remaining = deadline - self.clock()
            require(remaining > 0, "instance state wait timed out; inspect status, do not blindly repeat writes")
            self.sleep(min(self.config["poll_interval"], remaining))

    def power_on(self, *, wait=False, wait_timeout=900):
        state = self.state()
        if state == "running":
            return {"status": state, "changed": False}
        require(state == "stopped", "only a stopped instance can be powered on; inspect status")
        price = hourly_price(self.snapshot(), self.config)
        self._request("POST", "power_on", {
            "instance_uuid": self.config["instance_uuid"], "payload": "gpu",
            "start_command": self.config["start_command"]}, mutating=True)
        state = self.wait_state("running", wait_timeout) if wait else "startup_requested"
        return {"status": state, "changed": True, "hourly_yuan": price,
                "service_verified": False}

    def power_off(self, *, wait=False, wait_timeout=900):
        state = self.state()
        if state == "stopped":
            return {"status": state, "changed": False}
        require(state == "running", "only a running instance can be powered off")
        self._request("POST", "power_off", mutating=True)
        return {"status": self.wait_state("stopped", wait_timeout) if wait else "shutdown_requested",
                "changed": True}

    def images(self, *, deadline=None):
        images, seen = [], set()
        page = 1
        while page <= MAX_PAGES:
            data = self._request("POST", "image/private/list",
                                 {"page_index": page, "page_size": 100}, deadline=deadline)
            require(isinstance(data, dict) and isinstance(data.get("list"), list),
                    "image list response is invalid")
            rows = data["list"]
            fields = ("image_uuid", "name", "status", "image_size", "create_at")
            for row in rows:
                require(isinstance(row, dict) and isinstance(row.get("image_uuid"), str),
                        "image entry is invalid")
                image_id = row["image_uuid"]
                require(image_id not in seen, "image pagination repeated an ID; refusing incomplete results")
                seen.add(image_id)
                images.append({key: row[key] for key in fields if key in row
                               and isinstance(row[key], (str, int, float, bool, type(None)))})
            maximum = data.get("max_page")
            require(isinstance(maximum, int) and not isinstance(maximum, bool)
                    and 0 <= maximum <= MAX_PAGES, "image pagination metadata is invalid")
            if page >= maximum:
                return self._safe(images)
            require(bool(rows), "image pagination ended before its advertised last page")
            page += 1
        raise ControlError("image pagination exceeded limit")

    def image_wait(self, image_id, *, wait_timeout=900):
        require(isinstance(image_id, str) and re.fullmatch(r"image-[A-Za-z0-9-]{3,80}", image_id),
                "image_id is invalid")
        deadline = self.clock() + finite_number(wait_timeout, "wait_timeout")
        while True:
            found = next((image for image in self.images(deadline=deadline)
                          if image["image_uuid"] == image_id), None)
            if found and found.get("status") == "finished":
                return found
            require(not found or found.get("status") not in {"failed", "error", "deleted"},
                    "image save failed; inspect the private image list")
            remaining = deadline - self.clock()
            require(remaining > 0, "image wait timed out; inspect images before any manual save retry")
            self.sleep(min(self.config["poll_interval"], remaining))

    def image_save(self, name, *, wait=False, wait_timeout=900):
        require(isinstance(name, str) and 1 <= len(name.strip()) <= 100
                and not any(ord(char) < 32 for char in name), "image name is invalid")
        require(self.state() == "stopped",
                "image-save requires an already stopped instance; this tool will not shut it down automatically")
        data = self._request("POST", "image/save", {
            "instance_uuid": self.config["instance_uuid"], "image_name": name}, mutating=True)
        if not isinstance(data, dict) or not isinstance(data.get("image_uuid"), str):
            raise ControlError("image save accepted but image ID missing; inspect images before retrying",
                               outcome_unknown=True)
        image_id = data["image_uuid"]
        if not re.fullmatch(r"image-[A-Za-z0-9-]{3,80}", image_id):
            raise ControlError("image ID in save response is invalid; inspect images before retrying",
                               outcome_unknown=True)
        if wait:
            try:
                return self.image_wait(image_id, wait_timeout=wait_timeout)
            except ControlError as error:
                raise ControlError(str(error), outcome_unknown=error.outcome_unknown,
                                   image_uuid=self._safe(image_id)) from None
        return self._safe({"image_uuid": image_id, "status": "save_requested"})

    def refresh_endpoint(self):
        snapshot = self.snapshot()
        url = endpoint_url(snapshot.get("service_6006_domain"))
        status = atomic_endpoint(self.config["endpoint_file"], {
            "url": url, "instance_uuid": self.config["instance_uuid"],
            "discovered_at": datetime.now(timezone.utc).isoformat()})
        return self._safe({"url": url, "endpoint_status": status,
                           "service_verified": False})


def parser():
    value = argparse.ArgumentParser(description="AutoDL Pro instance/image control; never prints account credentials")
    value.add_argument("--config", type=Path,
                       default=Path(__file__).with_name("RTX5090.control.json"))
    commands = value.add_subparsers(dest="command", required=True)
    for name in ("status", "images", "refresh-endpoint"):
        commands.add_parser(name)
    for name in ("power-on", "power-off", "image-save"):
        command = commands.add_parser(name)
        command.add_argument("--wait", action="store_true")
        command.add_argument("--wait-timeout", type=float, default=900)
        if name == "image-save":
            command.add_argument("--name", required=True)
    command = commands.add_parser("image-wait")
    command.add_argument("--image-id", required=True)
    command.add_argument("--wait-timeout", type=float, default=900)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        manager = ServiceManager(config, read_token(config))
        command = args.command.replace("-", "_")
        if command in {"power_on", "power_off"}:
            finite_number(args.wait_timeout, "wait_timeout")
            result = getattr(manager, command)(wait=args.wait, wait_timeout=args.wait_timeout)
        elif command == "image_save":
            finite_number(args.wait_timeout, "wait_timeout")
            result = manager.image_save(args.name, wait=args.wait, wait_timeout=args.wait_timeout)
        elif command == "image_wait":
            result = manager.image_wait(args.image_id, wait_timeout=args.wait_timeout)
        else:
            result = getattr(manager, command)()
        print(json.dumps({"ok": True, "data": result}, ensure_ascii=False, allow_nan=False))
        return 0
    except ControlError as error:
        result = {"ok": False, "error": str(error), "outcome_unknown": error.outcome_unknown}
        if error.image_uuid:
            result.update({"accepted_image_uuid": error.image_uuid,
                           "resume_command": "image-wait --image-id " + error.image_uuid})
        print(json.dumps(result, ensure_ascii=False))
        return 2
    except (OSError, ValueError, TypeError, RecursionError):
        # Never print arbitrary exception text, response bodies or headers.
        print(json.dumps({"ok": False, "error": "local input/output or API schema failure",
                          "outcome_unknown": False}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
