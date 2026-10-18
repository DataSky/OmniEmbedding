#!/usr/bin/env python3
"""Build an unexecuted, self-contained Colab tutorial from reviewed source files.

Run this again after changing any bundled source. No environment files, tokens,
user data, model weights, or Git credentials enter the notebook payload.
Requires the small authoring dependency: ``python -m pip install nbformat``.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
from pathlib import Path
import textwrap
import zipfile

import nbformat


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = (
    "LICENSE",
    "omniembed/__init__.py",
    "omniembed/config.py",
    "omniembed/health.py",
    "omniembed/encoder.py",
    "omniembed/inference.py",
    "omniembed/media.py",
    "scripts/colab_serve.py",
    "requirements-colab.txt",
)


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    source = textwrap.dedent(source).strip() + "\n"
    ast.parse(source)
    return nbformat.v4.new_code_cell(source, execution_count=None, outputs=[])


def source_bundle():
    """An explicit file whitelist prevents accidental credential/data inclusion."""
    stream = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in BUNDLE_FILES:
            path = ROOT / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Required regular source file is missing: {relative}")
            data = path.read_bytes()
            if len(data) > 2 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large source file: {relative}")
            if relative.endswith(".py"):
                ast.parse(data, filename=relative)
            hashes[relative] = hashlib.sha256(data).hexdigest()
            # Fixed ZIP metadata makes rebuilding the same source reproducible.
            member = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, data)
    payload = stream.getvalue()
    return base64.b64encode(payload).decode("ascii"), hashes, hashlib.sha256(payload).hexdigest()


def build_notebook():
    payload, hashes, digest = source_bundle()
    setup = f'''\
        # 只解压生成器白名单内的源码；本单元没有任何密钥或模型权重。
        import base64
        import hashlib
        import io
        import os
        from pathlib import Path, PurePosixPath
        import sys
        import zipfile

        PROJECT_ROOT = Path("/content/OmniEmbedding")
        EXPECTED_SOURCE_HASHES = {hashes!r}
        SOURCE_BUNDLE_SHA256 = {digest!r}
        SOURCE_BUNDLE_BASE64 = {payload!r}

        if PROJECT_ROOT.is_symlink():
            raise RuntimeError("项目目录不能是符号链接，请换一个干净的 Colab 运行时。")
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        if PROJECT_ROOT.resolve() != PROJECT_ROOT:
            raise RuntimeError("项目目录发生路径跳转。")
        bundle_bytes = base64.b64decode(SOURCE_BUNDLE_BASE64, validate=True)
        assert hashlib.sha256(bundle_bytes).hexdigest() == SOURCE_BUNDLE_SHA256
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
            members = archive.infolist()
            if len(members) != len(EXPECTED_SOURCE_HASHES):
                raise RuntimeError("源码包包含重复或额外条目。")
            if {{member.filename for member in members}} != set(EXPECTED_SOURCE_HASHES):
                raise RuntimeError("源码包文件清单与白名单不同。")
            for member in members:
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts or member.is_dir():
                    raise RuntimeError("源码包包含不安全路径。")
                if member.file_size > 2 * 1024 * 1024:
                    raise RuntimeError("源码包单个文件超出限制。")
                source = archive.read(member)
                if hashlib.sha256(source).hexdigest() != EXPECTED_SOURCE_HASHES[member.filename]:
                    raise RuntimeError("源码校验失败。")
                target = PROJECT_ROOT / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or not target.resolve().is_relative_to(PROJECT_ROOT):
                    raise RuntimeError("目标文件存在不安全路径跳转。")
                target.write_bytes(source)
        print("已校验并准备", len(EXPECTED_SOURCE_HASHES), "个源码文件。")
        print("源码包 SHA256:", SOURCE_BUNDLE_SHA256)
        del SOURCE_BUNDLE_BASE64, bundle_bytes
        '''

    cells = [
        markdown("""
            # 在 Google Colab 试运行 EmbeddingGemma 2 推理服务

            ## 目标与边界

            本教程加载 `google/embeddinggemma-2`，提供带 Bearer 鉴权的文本 embedding API，
            然后验证存活、就绪和真实推理。默认只监听 Colab 本机，单次运行最多 **2 小时**；
            公开地址是可选的短期实验功能，默认关闭。

            **健康检测只能发现失效，不能保证 Colab 持续分配 GPU。** Colab 可能回收运行时，
            公开临时地址也会变化；不要把请求或脚本当作规避空闲回收的方法。免费托管运行时
            对远程代理等用途有限制；需要对外服务时，先确认所用运行时与计划允许该用途。
            稳定服务应迁移到有固定端点与可重启 GPU 实例的平台。

            本文件保存时未执行，不包含模型权重、访问令牌或伪造的成功输出。
            它内嵌当前实现的最小源码包，因此无需等待分支推送，也不会从远端静默拉取另一份实现。
            每个源码文件的 SHA256 会在解压时验证；生成器是 `scripts/build_colab_notebook.py`。

            官方资料：[Colab 资源与使用限制](https://research.google.com/colaboratory/faq.html)、
            [模型仓库与访问条款](https://huggingface.co/google/embeddinggemma-2)。
            """),
        markdown("""
            ## 准备运行时

            在 Colab 选择 **运行时 → 更改运行时类型 → GPU**，再从上到下执行。
            需要 Hugging Face 模型访问权限时，先在模型页面接受条款，并把只读令牌保存为
            Colab Secrets 的 `HF_TOKEN`。建议另存一个随机长密钥到
            `OMNI_INFERENCE_API_KEY`，以便在外部健康监测器中使用同一密钥。
            不要把令牌写进代码、截图或对话。

            若没有设置 API 密钥，后续单元会随机生成并只写入权限为 `0600` 的临时文件；
            不会显示它。没有 GPU 时本教程会明确停止，避免误以为 CPU 等价于 GPU 部署。
            """),
        code("""
            import sys
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("没有 CUDA GPU。请在 Colab 更改为 GPU 运行时后从头执行。")
            TORCH_VERSION = torch.__version__
            device_properties = torch.cuda.get_device_properties(0)
            native_bf16 = torch.cuda.is_bf16_supported() and device_properties.major >= 8
            INFERENCE_DTYPE = "bfloat16" if native_bf16 else "float32"
            print("Python:", sys.version.split()[0], "| PyTorch:", TORCH_VERSION)
            print("GPU:", device_properties.name, "| 显存 GiB:", round(device_properties.total_memory / 2**30, 1))
            print("推理精度:", INFERENCE_DTYPE, "（T4 等无原生 BF16 的 GPU 使用 FP32；禁止 FP16）")
            """),
        markdown("""
            ### 1. 准备并校验源码

            下方长字符串只是经过压缩的白名单源码，可从生成器复建。若修改了仓库中的 API 或启动器，
            应先重新生成 notebook，再上传到 Colab。
            """),
        code(setup),
        markdown("""
            ### 2. 安装应用依赖，保留 Colab 的 PyTorch / CUDA

            使用运行时自带的 PyTorch，并为依赖解析添加精确约束；本教程不安装 CUDA 或重新安装 PyTorch。
            如果解析器报告冲突，先使用新的 Colab 运行时并检查依赖版本，不要通过删除约束来强行替换 CUDA 栈。
            """),
        code("""
            import subprocess

            constraints_path = Path("/content/omniembed-torch-constraints.txt")
            import importlib.metadata

            torch_constraints = ["torch==" + TORCH_VERSION]
            for package_name in ("torchvision", "torchaudio"):
                try:
                    torch_constraints.append(package_name + "==" + importlib.metadata.version(package_name))
                except importlib.metadata.PackageNotFoundError:
                    pass
            constraints_path.write_text("\\n".join(torch_constraints) + "\\n", encoding="utf-8")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q",
                 "-r", str(PROJECT_ROOT / "requirements-colab.txt"), "-c", str(constraints_path)],
                check=True,
            )
            installed_torch = subprocess.check_output(
                [sys.executable, "-c", "import torch; print(torch.__version__)"], text=True
            ).strip()
            if installed_torch != TORCH_VERSION:
                raise RuntimeError("PyTorch 版本意外变化，请使用干净运行时重新执行。")
            print("应用依赖已安装；保留 PyTorch", installed_torch)
            """),
        markdown("""
            ### 3. 读取 Secrets，固定模型提交

            `HF_TOKEN` 缺失时可在隐藏输入框填入，已有公开访问权限时可留空。
            模型分支只在这里解析一次，随后固定到 Hugging Face 返回的提交 SHA。
            API 密钥只写入本运行时的 `0600` 文件，不进入命令行、notebook 输出或 endpoint JSON。
            使用自动生成密钥时，外部监测器暂不能共享它；在 Secrets 中设置相同密钥后重跑本单元并重启服务。
            """),
        code("""
            import getpass
            import secrets
            from huggingface_hub import HfApi

            def read_colab_secret(name):
                try:
                    from google.colab import userdata
                    value = userdata.get(name)
                    return value.strip() if isinstance(value, str) else None
                except Exception:
                    return None

            MODEL_ID = "google/embeddinggemma-2"
            hf_token = read_colab_secret("HF_TOKEN")
            if not hf_token:
                hf_token = getpass.getpass("Hugging Face 只读令牌（隐藏；可留空）: ").strip() or None
            try:
                MODEL_REVISION = HfApi().model_info(MODEL_ID, token=hf_token).sha
            except Exception:
                raise RuntimeError("无法解析模型。请检查网络、仓库名称与 HF_TOKEN 的模型访问权限。") from None
            if not MODEL_REVISION:
                raise RuntimeError("模型没有可固定的提交 SHA。")

            KEY_FILE = Path("/content/omniembed.key")
            if KEY_FILE.is_symlink():
                raise RuntimeError("密钥文件不能是符号链接。")
            inference_api_key = read_colab_secret("OMNI_INFERENCE_API_KEY")
            if not inference_api_key:
                inference_api_key = KEY_FILE.read_text().strip() if KEY_FILE.exists() else secrets.token_urlsafe(32)
                print("使用本运行时的私有随机密钥；外部监测请先配置 Colab Secret。")
            if len(inference_api_key) < 32 or any(char.isspace() for char in inference_api_key):
                raise ValueError("OMNI_INFERENCE_API_KEY 至少需要 32 个非空白字符。")
            file_descriptor = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as key_stream:
                os.fchmod(key_stream.fileno(), 0o600)
                key_stream.write(inference_api_key)
            del inference_api_key
            SERVICE_ENV = os.environ.copy()
            SERVICE_ENV.update({"OMNI_MODEL": MODEL_ID, "OMNI_MODEL_REVISION": MODEL_REVISION,
                                "OMNI_DEVICE": "cuda", "OMNI_DTYPE": INFERENCE_DTYPE})
            if hf_token:
                SERVICE_ENV["HF_TOKEN"] = hf_token
            del hf_token
            print("模型:", MODEL_ID)
            print("固定 revision:", MODEL_REVISION)
            """),
        markdown("""
            ### 4. 启动受保护的有限时服务

            默认 `PUBLIC_EXPERIMENT = False`，只做本机测试。仅在所用 Colab 计划允许时将其改为
            `True`，使用短期 Cloudflare 隧道对外实验。公开地址不等于免鉴权，客户端仍需 Bearer 密钥。
            启动器限制总运行时长与重启次数；模型下载或 GPU 被回收时，健康探针不会虚报成功。
            服务到期停止后，Colab 虚拟机仍可能占用配额；用完需手动断开并删除运行时。

            启动日志与端点信息只保存到本运行时，不上传。再次执行本单元前先运行末尾的显式停止操作。
            """),
        code("""
            import socket

            PUBLIC_EXPERIMENT = False
            MAX_RUNTIME_SECONDS = 7200
            API_PORT = 8000
            ENDPOINT_FILE = Path("/content/omniembed-endpoint.json")
            SERVICE_LOG = Path("/content/omniembed-service.log")
            if "service_process" in globals() and service_process.poll() is None:
                raise RuntimeError("当前启动器仍在运行，请先使用末尾的停止单元。")
            with socket.socket() as port_check:
                if port_check.connect_ex(("127.0.0.1", API_PORT)) == 0:
                    raise RuntimeError("端口已被使用，请先停止现有服务。")
            launch_command = [sys.executable, str(PROJECT_ROOT / "scripts/colab_serve.py"),
                              "--key-file", str(KEY_FILE), "--port", str(API_PORT),
                              "--max-runtime", str(MAX_RUNTIME_SECONDS),
                              "--endpoint-file", str(ENDPOINT_FILE)]
            if PUBLIC_EXPERIMENT:
                launch_command.append("--public")
            log_descriptor = os.open(SERVICE_LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(log_descriptor, "w", encoding="utf-8") as log_stream:
                os.fchmod(log_stream.fileno(), 0o600)
                service_process = subprocess.Popen(launch_command, cwd=PROJECT_ROOT, env=SERVICE_ENV,
                                                   stdout=log_stream, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
            print("启动器 PID:", service_process.pid, "| 最大运行秒数:", MAX_RUNTIME_SECONDS)
            print("公开实验开关:", PUBLIC_EXPERIMENT, "| 等待下一单元验证就绪。")
            """),
        markdown("""
            ## 验证服务

            ### 5. 等待模型加载并验证就绪

            最多等待 15 分钟，定时检查本机就绪状态；这只是有上限的启动验证。
            首次下载模型较慢。若启动器退出或超时，查看 `/content/omniembed-service.log`
            和 `/content/logs/inference.log`；
            分享日志前先检查其中的访问地址和其他敏感信息。
            """),
        code("""
            import json
            import time
            import requests

            LOCAL_BASE_URL = f"http://127.0.0.1:{API_PORT}"
            local_client = requests.Session()
            local_client.trust_env = False
            AUTH_HEADERS = {"Authorization": "Bearer " + KEY_FILE.read_text().strip()}
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if service_process.poll() is not None:
                    raise RuntimeError("启动器已退出，请检查私有服务日志。")
                try:
                    ready_response = local_client.get(LOCAL_BASE_URL + "/health/ready", headers=AUTH_HEADERS, timeout=5)
                    if ready_response.status_code in (401, 403):
                        raise RuntimeError("API 密钥不匹配，请停止服务后重新设置 Secrets。")
                    if ready_response.status_code == 200:
                        readiness = ready_response.json()
                        assert readiness["ready"] is True
                        assert readiness["model_revision"] == MODEL_REVISION
                        assert readiness["dtype"] == INFERENCE_DTYPE
                        print("模型已就绪:", json.dumps(readiness, ensure_ascii=False))
                        break
                except requests.RequestException:
                    pass
                time.sleep(5)
            else:
                raise TimeoutError("15 分钟内未就绪，请检查模型权限、显存与服务日志。")
            """),
        markdown("""
            ### 6. 真正运行 embedding，并检查向量

            仅检查 HTTP 200 会漏掉模型卡死或返回 NaN。下方发送真实中文查询，验证二维向量、
            有限数值与 L2 归一化；同时验证无密钥请求被拒绝。此单元消耗一次短推理。
            """),
        code("""
            import numpy as np

            request_started = time.monotonic()
            embedding_response = local_client.post(
                LOCAL_BASE_URL + "/v1/embeddings", headers=AUTH_HEADERS,
                json={"input": ["检索海边日落的视频", "搜索机器学习模型部署教程"],
                      "input_type": "query", "dimensions": 768},
                timeout=120,
            )
            embedding_response.raise_for_status()
            embedding_result = embedding_response.json()
            vectors = np.asarray([item["embedding"] for item in embedding_result["data"]], dtype=np.float32)
            assert vectors.shape == (2, 768), vectors.shape
            assert np.isfinite(vectors).all(), "向量包含 NaN/Inf"
            vector_norms = np.linalg.norm(vectors, axis=1)
            assert np.allclose(vector_norms, 1.0, atol=1e-3), vector_norms
            unauthorized_response = local_client.post(
                LOCAL_BASE_URL + "/v1/embeddings", json={"input": "鉴权测试"}, timeout=10
            )
            assert unauthorized_response.status_code in (401, 403), "无密钥请求未被拒绝"
            live_response = local_client.get(LOCAL_BASE_URL + "/health/live", timeout=10)
            live_response.raise_for_status()
            assert live_response.json()["status"] == "ok"
            deep_response = local_client.post(LOCAL_BASE_URL + "/health/inference", headers=AUTH_HEADERS, timeout=120)
            deep_response.raise_for_status()
            deep_health = deep_response.json()
            assert deep_health["checks"]["shape"] == [1, 768]
            assert all(deep_health["checks"][name] for name in ("finite", "nonzero", "unit_norm"))
            print("真实向量 shape:", vectors.shape, "| L2 norms:", vector_norms.round(6).tolist())
            print("finite: True | live / deep: OK | 无密钥状态码:", unauthorized_response.status_code,
                  "| 请求耗时秒:", round(time.monotonic() - request_started, 2))
            """),
        markdown("""
            ### 7. 读取端点信息，交给外部监测

            端点 JSON 只保存地址、模型等非密钥信息。默认输出本机地址；开启公开实验后临时域名
            可能需要额外几秒才能生成。不要把本机 `127.0.0.1` 地址直接交给外部监测器。
            """),
        code("""
            endpoint_deadline = time.monotonic() + (120 if PUBLIC_EXPERIMENT else 10)
            while time.monotonic() < endpoint_deadline:
                endpoint_info = json.loads(ENDPOINT_FILE.read_text(encoding="utf-8")) if ENDPOINT_FILE.exists() else {}
                if endpoint_info.get("status") == "ready":
                    if PUBLIC_EXPERIMENT and not endpoint_info.get("url", "").startswith("https://"):
                        raise RuntimeError("公开实验没有生成 HTTPS 端点。")
                    print(json.dumps(endpoint_info, ensure_ascii=False, indent=2))
                    break
                if endpoint_info.get("status") in ("failed", "stopped"):
                    raise RuntimeError("启动器已停止，请查看服务日志。")
                time.sleep(2)
            else:
                raise TimeoutError("端点信息未就绪，请查看服务日志；公开实验另查 /content/logs/tunnel.log。")
            """),
        markdown("""
            ## 健康检测设计与后续操作

            将外部监测器运行在你自己的电脑或稳定主机上。存活探针判断进程是否响应；
            就绪探针判断模型是否可接收请求；低频深度探针实际计算短文本向量，验证耗时、
            维度和有限数值。连续失败达到阈值才告警，恢复后再报告恢复，避免一次网络抖动造成误报。

            外部监测器使用端点地址与同一个 API 密钥。公开临时域名失效后，需要人工重新获得运行时、
            执行 notebook，并更新监测地址；不要无限自动重连或模拟交互以规避 Colab 的资源策略。
            对稳定性要求高时，把本 API 镜像迁移到可恢复的 GPU 实例；监测器与请求客户端可继续复用。

            **本交付的验证边界：** notebook 格式、每个代码单元语法和内嵌源码哈希可在本机检查；
            真实模型加载、Colab GPU、HF 权限与公网隧道必须在你的 Colab 会话中依次执行前述单元才能验证。
            不应把“已生成 notebook”写成“云端服务已经上线”。

            ### 手动停止并释放 GPU

            用完后把下一单元的 `STOP_SERVICE` 改为 `True` 再执行。默认值为 `False`，
            因此从头运行全部单元不会在检查完成后立刻关闭服务。停止会终止本 notebook 启动的进程组，
            删除本地密钥和端点文件。最后可通过 Colab 的 **运行时 → 断开连接并删除运行时** 归还资源。
            """),
        code("""
            import signal

            STOP_SERVICE = False
            if STOP_SERVICE:
                if "service_process" in globals() and service_process.poll() is None:
                    os.killpg(service_process.pid, signal.SIGTERM)
                    try:
                        service_process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(service_process.pid, signal.SIGKILL)
                        service_process.wait(timeout=5)
                KEY_FILE.unlink(missing_ok=True)
                ENDPOINT_FILE.unlink(missing_ok=True)
                AUTH_HEADERS.clear()
                SERVICE_ENV.pop("HF_TOKEN", None)
                torch.cuda.empty_cache()
                print("本 notebook 启动的服务已停止；密钥和端点文件已删除。")
            else:
                print("保留当前服务。用完后显式停止，或断开并删除 Colab 运行时。")
            """),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": "Colab_EmbeddingGemma2.ipynb", "provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
        "omniembed_source_bundle_sha256": digest,
        "omniembed_source_hashes": hashes,
    }
    # Stable IDs avoid unrelated diffs every time this source snapshot is rebuilt.
    for index, cell in enumerate(notebook.cells):
        cell.id = f"omniembed-{index:02d}-{hashlib.sha256(cell.source.encode()).hexdigest()[:8]}"
    nbformat.validate(notebook)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/Colab_EmbeddingGemma2.ipynb")
    args = parser.parse_args()
    notebook = build_notebook()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"Saved {args.output} ({len(notebook.cells)} unexecuted cells)")


if __name__ == "__main__":
    main()
