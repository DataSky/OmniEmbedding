"""媒体处理：URL 安全下载、ffprobe 解析、OpenCV 抽帧、封面与关键帧缩略图。

抽帧用 OpenCV 而不是 torchcodec：Windows 上常见的静态版 FFmpeg 没有 torchcodec 需要的共享库。
"""
import hashlib
import ipaddress
import json
import shutil
import socket
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".ts"}


class MediaError(Exception):
    pass


# ---------- URL 下载 ----------
def _check_host(host: str, allow_private: bool):
    """拒绝内网、回环、链路本地地址，防止通过 URL 下载功能访问内网服务（SSRF）。"""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as ex:
        raise MediaError(f"无法解析域名 {host}: {ex}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not allow_private and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                                  or ip.is_multicast or ip.is_unspecified):
            raise MediaError(f"拒绝访问内网或保留地址：{host} → {ip}（如确需访问，设置 OMNI_ALLOW_PRIVATE_URLS=1）")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_private):
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlparse(newurl)
        if u.scheme not in ("http", "https"):
            raise MediaError(f"重定向到不支持的协议：{u.scheme}")
        _check_host(u.hostname or "", self.allow_private)  # 重定向后的地址同样要检查
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, dst: Path, max_bytes: int, timeout: int, allow_private=False) -> int:
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise MediaError("只支持 http / https 地址")
    _check_host(u.hostname, allow_private)
    opener = urllib.request.build_opener(_SafeRedirect(allow_private))
    req = urllib.request.Request(url, headers={"User-Agent": "OmniEmbedding/0.1"})
    tmp = dst.with_suffix(dst.suffix + ".part")
    n = 0
    try:
        with opener.open(req, timeout=timeout) as r, open(tmp, "wb") as f:
            cl = r.headers.get("Content-Length")
            if cl and int(cl) > max_bytes:
                raise MediaError(f"文件过大：{int(cl) / 2**20:.0f} MiB，上限 {max_bytes / 2**20:.0f} MiB")
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                n += len(chunk)
                if n > max_bytes:
                    raise MediaError(f"文件过大：超过 {max_bytes / 2**20:.0f} MiB")
                f.write(chunk)
        tmp.replace(dst)
    except MediaError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as ex:
        tmp.unlink(missing_ok=True)
        raise MediaError(f"下载失败：{type(ex).__name__}: {ex}")
    return n


def guess_ext(name: str) -> str:
    ext = Path(urllib.parse.urlparse(name).path).suffix.lower()
    return ext if ext in VIDEO_EXT else ".mp4"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- 解析 ----------
def probe(path: Path) -> dict:
    """优先用 ffprobe（信息全：编码、音轨），没有 ffprobe 时退回 OpenCV。"""
    info = {}
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                                 capture_output=True, timeout=60, check=True).stdout
            d = json.loads(out)
            v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
            if v is None:
                raise MediaError("文件里没有视频流")
            num, _, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").partition("/")
            info = {"codec": v.get("codec_name"), "width": v.get("width"), "height": v.get("height"),
                    "fps": round(float(num) / float(den or 1), 3) if float(den or 1) else None,
                    "duration_s": float(d.get("format", {}).get("duration") or v.get("duration") or 0) or None,
                    "has_audio": int(any(s.get("codec_type") == "audio" for s in d.get("streams", [])))}
        except MediaError:
            raise
        except Exception:
            info = {}
    if not info.get("duration_s"):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise MediaError("无法打开视频（格式不支持或文件损坏）")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        info.update({"width": info.get("width") or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     "height": info.get("height") or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     "fps": info.get("fps") or round(fps, 3), "duration_s": round(n / fps, 3) if fps else None})
        cap.release()
    if not info.get("duration_s"):
        raise MediaError("读不出视频时长（文件可能损坏）")
    return info


# ---------- 抽帧 ----------
def extract_frames(path: Path, duration_s: float, fps: float = 1.0, max_frames: int = 8):
    """按 fps 计算采样点，超过上限时在全片均匀取 max_frames 个点（与模型 video_processor 的 uniform 策略一致）。
    返回 (帧数组 [T,H,W,3] RGB, 时间点列表)。"""
    n = max(1, int(duration_s * fps))
    k = min(n, max_frames)
    times = list(np.linspace(0, max(duration_s - 0.05, 0), k)) if k > 1 else [min(duration_s / 2, 1.0)]
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames, ts = [], []
    for t in times:
        # 按帧号定位比按毫秒稳定；最后一个点夹到最后一帧，避免越界读不出
        fi = int(round(t * src_fps))
        if n_src:
            fi = min(fi, n_src - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok and fi > 0:  # 部分容器的最后几帧不可定位，往前退一帧再试
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi - 1)
            ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            ts.append(round(fi / src_fps, 2))
    cap.release()
    if not frames:
        raise MediaError("一帧都没有解出来（编码不受支持或文件损坏）")
    return np.stack(frames), ts


def save_thumb(frame_rgb: np.ndarray, dst: Path, width=320):
    h, w = frame_rgb.shape[:2]
    img = cv2.resize(frame_rgb, (width, max(1, int(h * width / w)))) if w > width else frame_rgb
    cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
