#!/usr/bin/env python3
"""独立 AutoDL 推理接口示例，Python >=3.11，仅需 httpx。

安装：python -m pip install httpx==0.28.1
配置 OMNI_INFERENCE_URL 以及 OMNI_INFERENCE_KEY_FILE，或通过 --key-file 指定
当前用户拥有的 0600 文件；也支持 OMNI_INFERENCE_API_KEY 环境变量。不要把 Key
直接放进命令行。服务会自动添加文本任务前缀，调用者传原文即可。

用法：
  python docs/examples/embeddinggemma_client.py health
  python docs/examples/embeddinggemma_client.py probe
  python docs/examples/embeddinggemma_client.py text --text '厨房里正在做饭'
  python docs/examples/embeddinggemma_client.py text --input-type document \
    --text '锅中加入食材，慢火炖煮。' --title '家常炖菜'
  python docs/examples/embeddinggemma_client.py --dimensions 512 image photo.jpg
  python docs/examples/embeddinggemma_client.py video frame01.jpg frame02.jpg

默认只打印摘要。--output PATH 以 0600 权限新建包含向量的 JSON 文件，拒绝覆盖。
导入后用 async with EmbeddingGemmaClient(...)，调用 await client.text/image/video。
一个对象在一个事件循环内串行推理、最多排队 8 个请求。超时预算覆盖本地
排队、HTTP 和重试，不包含此前读文件和此后向量校验、输出文件写入。
这不能协调其他进程/客户端；单实例须由所有调用方共用的调度器保证总并发为 1。
生产应用已有 omniembed.remote_encoder.RemoteEncoder 时优先复用该业务适配器；
此示例用于轻量调用、探针以及可选输出维度，不导入模型或项目 .env 配置。
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import ipaddress
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
import time
from urllib.parse import urlsplit

import httpx

MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIMENSIONS = (128, 256, 512, 768)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class ClientError(RuntimeError):
    """只包含本地说明或 HTTP 状态；不包含请求、Key、服务错误正文。"""


def require(condition, message):
    if not condition:
        raise ClientError(message)


def base_url(value):
    require(isinstance(value, str) and value and not any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value),
            "请配置有效的 HTTPS 服务地址")
    try:
        parsed = urlsplit(value)
        require(parsed.scheme in {"https", "http"} and parsed.hostname and parsed.port != 0
                and parsed.username is None and parsed.password is None
                and not parsed.query and not parsed.fragment, "服务地址格式无效")
        try:
            local = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            local = parsed.hostname.lower() == "localhost"
        require(parsed.scheme == "https" or local, "HTTP 仅允许本机回环地址；公网必须使用 HTTPS")
    except ValueError:
        raise ClientError("服务地址格式无效") from None
    return value.rstrip("/")


def validate_key(value):
    require(isinstance(value, str) and 1 <= len(value) <= 8192
            and all(33 <= ord(c) <= 126 for c in value), "推理 Key 无效")
    return value


def load_key(path=None):
    """只读取明确指定的独立推理 Key；不读取 .env 或 AutoDL 账户 Token。"""
    path = path or os.environ.get("OMNI_INFERENCE_KEY_FILE")
    if not path:
        return validate_key(os.environ.get("OMNI_INFERENCE_API_KEY", ""))
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            info = os.fstat(handle.fileno())
            require(stat.S_ISREG(info.st_mode), "Key 必须来自普通文件")
            if os.name == "posix":
                require(stat.S_IMODE(info.st_mode) == 0o600 and info.st_uid == os.getuid(),
                        "Key 文件须属于当前用户且权限为 0600")
            raw = handle.read(8193)
        require(len(raw) <= 8192, "Key 文件过大")
        return validate_key(raw.decode("ascii").strip())
    except (OSError, UnicodeError):
        raise ClientError("无法读取独立推理 Key 文件") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def retry_after(value):
    """支持秒数或 HTTP-date；不缩短服务给出的等待时间。"""
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            delay = (when - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 2.0
    return max(0.0, delay) if math.isfinite(delay) else 2.0


def read_image(path, maximum):
    try:
        with Path(path).open("rb") as handle:
            content = handle.read(maximum + 1)
    except OSError:
        raise ClientError("无法读取图片/帧文件") from None
    require(0 < len(content) <= maximum, "图片/帧文件为空或超出字节限制")
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif content.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    elif content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ClientError("仅支持 JPEG、PNG 或 WebP 静态图片/帧")
    return content, mime


class EmbeddingGemmaClient:
    def __init__(self, url, key=None, *, timeout_s=120, attempts=3, max_pending=8, transport=None):
        require(type(timeout_s) in {int, float} and math.isfinite(timeout_s) and timeout_s > 0,
                "总超时必须是正有限秒数")
        require(type(attempts) is int and 1 <= attempts <= 3, "尝试次数须为 1–3")
        require(type(max_pending) is int and 0 <= max_pending <= 64, "本地排队上限须为 0–64")
        self.key = validate_key(key) if key is not None else None
        self.url = base_url(url)
        self.timeout_s, self.attempts, self.max_pending = timeout_s, attempts, max_pending
        self._lock, self._pending = asyncio.Lock(), 0
        # 明确关闭环境代理、.netrc 和重定向；保留 TLS 证书校验。
        self.http = httpx.AsyncClient(follow_redirects=False, trust_env=False, transport=transport,
                                     timeout=httpx.Timeout(timeout_s, connect=min(10, timeout_s)))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def _request(self, method, path, *, inference=False, authenticated=True, **kwargs):
        require(not authenticated or self.key is not None, "此路由需要独立推理 Key")
        require(not inference or self._pending < self.max_pending + 1, "本地推理队列已满")
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer " + self.key
        if inference:
            self._pending += 1
        try:
            # 包含本地排队、上传、读响应、429 等待和全部尝试；不是每次重试重新计时。
            async with asyncio.timeout(self.timeout_s):
                if inference:
                    async with self._lock:
                        return await self._attempt(method, path, headers, kwargs)
                return await self._attempt(method, path, headers, kwargs)
        except TimeoutError:
            raise ClientError("排队与 HTTP 总时限已到（含重试）") from None
        except httpx.HTTPError:
            # 不自动重放传输超时：服务器可能仍持有推理锁并继续计算。
            raise ClientError("网络连接失败或超时；未自动重放请求") from None
        finally:
            if inference:
                self._pending -= 1

    async def _attempt(self, method, path, headers, kwargs):
        for attempt in range(self.attempts):
            async with self.http.stream(method, self.url + path, headers=headers, **kwargs) as response:
                if response.status_code == 429:
                    delay = retry_after(response.headers.get("Retry-After")) + random.uniform(0, 0.25)
                else:
                    require(not 300 <= response.status_code < 400, "拒绝重定向，未转发 Key")
                    require(response.status_code == 200, f"服务返回 HTTP {response.status_code}；未自动重试")
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        require(len(raw) + len(chunk) <= MAX_RESPONSE_BYTES, "服务响应超出 2 MiB 限制")
                        raw.extend(chunk)
                    try:
                        payload = json.loads(raw)
                    except (ValueError, UnicodeError, RecursionError):
                        raise ClientError("服务返回无效 JSON") from None
                    require(isinstance(payload, dict), "服务未返回 JSON 对象")
                    return payload
            require(attempt + 1 < self.attempts, "服务繁忙 HTTP 429；已达重试次数上限")
            # 总预算不足时不提前重试；外层 timeout 会在等待期间截止。
            await asyncio.sleep(delay)
        raise ClientError("未收到服务结果")

    @staticmethod
    def _identity(payload):
        require(payload.get("model") == MODEL and payload.get("model_revision") == REVISION,
                "模型或固定 revision 不匹配；禁止混入现有索引")
        boot = payload.get("boot_id")
        require(isinstance(boot, str) and 1 <= len(boot) <= 128
                and all(c.isalnum() or c in "_.:-" for c in boot), "缺少有效 boot_id")

    def _embeddings(self, payload, rows, dimensions):
        self._identity(payload)
        data = payload.get("data")
        require(type(payload.get("dimensions")) is int and payload["dimensions"] == dimensions
                and isinstance(data, list) and len(data) == rows, "返回维度或行数不匹配")
        ordered = [None] * rows
        for record in data:
            require(isinstance(record, dict), "向量记录无效")
            index, vector = record.get("index"), record.get("embedding")
            require(type(index) is int and 0 <= index < rows and ordered[index] is None,
                    "返回 index 缺失、重复或越界")
            require(isinstance(vector, list) and len(vector) == dimensions
                    and all(type(x) in {int, float} and math.isfinite(x) for x in vector),
                    "返回向量形状或有限值校验失败")
            require(abs(math.hypot(*vector) - 1.0) <= 0.001, "返回向量不是单位向量")
            ordered[index] = record
        payload["data"] = ordered
        return payload

    async def health(self, *, live=False):
        payload = await self._request("GET", "/health/live" if live else "/health/ready", authenticated=not live)
        if not live:
            self._identity(payload)
            require(payload.get("ready") is True, "模型尚未就绪")
        require(payload.get("status") == "ok", "健康状态异常")
        return payload

    async def probe(self):
        payload = await self._request("POST", "/health/inference", inference=True)
        self._identity(payload)
        require(payload.get("ready") is True and payload.get("status") == "ok"
                and payload.get("checks") == {"shape": [1, 768], "finite": True,
                                               "nonzero": True, "unit_norm": True}, "真实推理探针失败")
        return payload

    async def text(self, texts, *, input_type="query", titles=None, dimensions=768):
        texts = [texts] if isinstance(texts, str) else list(texts)
        require(input_type in {"query", "document"} and type(dimensions) is int and dimensions in DIMENSIONS,
                "角色或维度无效")
        require(1 <= len(texts) <= 16 and all(isinstance(x, str) and len(x) <= 8192 for x in texts),
                "每批须为 1–16 条字符串，每条最多 8192 字符")
        require(input_type == "document" or titles is None, "query 不接受 titles")
        require(not isinstance(titles, str), "titles 须为字符串列表")
        titles = list(titles) if titles is not None else None
        require(titles is None or len(titles) == len(texts)
                and all(isinstance(x, str) and len(x) <= 1024 for x in titles), "titles 须与 input 等长，每项最多 1024 字符")
        effective_titles = titles if titles is not None else [""] * len(texts)
        require(all(text.strip() or title.strip() for text, title in zip(texts, effective_titles)), "文本和标题不能同时为空")
        require(sum(map(len, texts)) + sum(map(len, effective_titles)) <= 32768, "每批正文与标题合计最多 32768 字符")
        payload = {"input": texts, "input_type": input_type, "dimensions": dimensions}
        if titles is not None:
            payload["titles"] = titles
        result = await self._request("POST", "/v1/embeddings", inference=True, json=payload)
        return self._embeddings(result, len(texts), dimensions)

    async def image(self, path, *, dimensions=768):
        require(type(dimensions) is int and dimensions in DIMENSIONS, "维度须为 128、256、512 或 768")
        content, mime = read_image(path, 10 * 1024 * 1024)
        result = await self._request("POST", "/v1/embeddings/image", inference=True,
                                     params={"dimensions": dimensions}, files={"file": ("image", content, mime)})
        return self._embeddings(result, 1, dimensions)

    async def video(self, frame_paths, *, dimensions=768):
        paths = list(frame_paths)
        require(type(dimensions) is int and dimensions in DIMENSIONS and 1 <= len(paths) <= 8,
                "视频须为 1–8 个有序等尺寸画面帧，维度须有效")
        files, total = [], 0
        for index, path in enumerate(paths):
            content, mime = read_image(path, 32 * 1024 * 1024 - total)
            total += len(content)
            files.append(("files", (f"frame-{index:02d}", content, mime)))
        result = await self._request("POST", "/v1/embeddings/video", inference=True,
                                     params={"dimensions": dimensions}, files=files)
        return self._embeddings(result, 1, dimensions)


async def run(args):
    live = args.command == "health" and args.live
    key = None if live else load_key(args.key_file)
    async with EmbeddingGemmaClient(args.base_url, key, timeout_s=args.timeout, attempts=args.attempts) as client:
        if args.command == "health":
            payload = await client.health(live=args.live)
        elif args.command == "probe":
            payload = await client.probe()
        elif args.command == "text":
            texts = list(args.text or [])
            try:
                for path in args.text_file or []:
                    with path.open("r", encoding="utf-8") as handle:
                        texts.append(handle.read(8193))
            except (OSError, UnicodeError):
                raise ClientError("无法读取 UTF-8 正文文件") from None
            payload = await client.text(texts, input_type=args.input_type, titles=args.title, dimensions=args.dimensions)
        elif args.command == "image":
            payload = await client.image(args.path, dimensions=args.dimensions)
        else:
            payload = await client.video(args.frames, dimensions=args.dimensions)
    if args.output:
        try:
            descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
        except OSError:
            raise ClientError("输出文件写入失败；路径须可写且不能已经存在") from None
    summary = {"ok": True, "operation": args.command}
    for field in ("model", "model_revision", "dimensions", "boot_id", "ready", "status"):
        if field in payload:
            summary[field] = payload[field]
    if "data" in payload:
        summary["rows"] = len(payload["data"])
        summary["unit_norm_checked"] = True
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("OMNI_INFERENCE_URL"), help="HTTPS 基址；也可设 OMNI_INFERENCE_URL")
    parser.add_argument("--key-file", type=Path, help="独立推理 Key 文件；勿传账户 Token")
    parser.add_argument("--dimensions", type=int, choices=DIMENSIONS, default=768)
    parser.add_argument("--timeout", type=float, default=120, help="排队+HTTP+重试总时限，默认 120 秒；不含读写本地文件")
    parser.add_argument("--attempts", type=int, choices=(1, 2, 3), default=3, help="仅 HTTP 429 重试，最多尝试 3 次")
    parser.add_argument("--output", type=Path, help="以 0600 权限新建完整响应 JSON；默认只打印摘要")
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("health", help="轻量 readiness；--live 为免鉴权进程存活探针")
    health.add_argument("--live", action="store_true")
    commands.add_parser("probe", help="真实模型探针，占用推理槽，不用于高频轮询")
    text = commands.add_parser("text", help="原文文本，前缀由服务添加")
    text.add_argument("--input-type", choices=("query", "document"), default="query")
    text.add_argument("--text", action="append", help="可重复；需要保护正文时用 --text-file")
    text.add_argument("--text-file", action="append", type=Path, help="每个 UTF-8 文件作为一条文本，可重复")
    text.add_argument("--title", action="append", help="document 的标题，与输入条数和顺序一致")
    commands.add_parser("image", help="一张静态 JPEG/PNG/WebP").add_argument("path", type=Path)
    commands.add_parser("video", help="按时间顺序列出 1–8 个等尺寸静态帧；不是 MP4").add_argument("frames", nargs="+", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        asyncio.run(run(args))
        return 0
    except ClientError as error:
        print(json.dumps({"ok": False, "error": str(error), "elapsed_s": round(time.monotonic() - started, 3)}, ensure_ascii=False))
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "error": "用户中断"}, ensure_ascii=False))
    except Exception as error:
        # 不输出异常消息或 traceback，避免正文或鉴权资料进入日志。
        print(json.dumps({"ok": False, "error": "调用失败: " + type(error).__name__}, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    sys.exit(main())
