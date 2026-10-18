#!/usr/bin/env python3
"""本地短视频 → 有序等尺寸 JPEG 帧，Python >=3.11，依赖 NumPy/OpenCV。

python prepare_video_frames.py clip.mp4 --output-dir prepared/clip01
默认均匀取8个真实帧，长边只缩不放至最多512像素，JPEG质量90。
随后将 frame01.jpg ... frame08.jpg 按顺序交给 embeddinggemma_client.py video。
少于8帧的视频只输出实际可取的不同帧；不会复制帧补齐。长视频应先按镜头或
短片段切分，否则8帧无法覆盖全部内容。这里不下载、不处理音频、不调用模型。
仅使用本地文件和 FFmpeg file 协议；不导入项目配置或读取任何密钥。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

import cv2
import numpy as np

MAX_EDGE = 512


def require(test, message):
    if not test:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, content):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(content)


def prepare(source, output, count=8):
    require(type(count) is int and 1 <= count <= 8, "帧数须为1–8")
    source = source.expanduser().resolve(strict=True)
    info = source.stat()
    require(stat.S_ISREG(info.st_mode), "输入必须是本地普通视频文件")
    output = output.expanduser().absolute()
    require(not os.path.lexists(output), "输出目录已存在，拒绝覆盖；请选择新目录")
    # 即使本地文件伪装成带远程URL的播放列表，也不允许FFmpeg打开网络协议。
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file"
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    created = []
    owns_output = False
    try:
        require(cap.isOpened(), "无法用本地FFmpeg解码器打开视频")
        fps, raw_count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        require(math.isfinite(fps) and fps > 0 and math.isfinite(raw_count) and raw_count >= 1,
                "缺少可靠帧数或FPS；请先把视频转为可正常解码的短片段")
        frame_count = int(raw_count)
        actual_count = min(count, frame_count)
        indices = [min(frame_count - 1, int((i + 0.5) * frame_count / actual_count))
                   for i in range(actual_count)]
        require(len(set(indices)) == actual_count, "采样索引重复")
        source_hash = sha256(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(mode=0o700)  # 原子创建；并行创建同名目录也会失败。
        owns_output = True
        records, source_size, target_size = [], None, None
        for order, index in enumerate(indices, 1):
            require(cap.set(cv2.CAP_PROP_POS_FRAMES, index), "解码器不能定位采样帧")
            ok, frame = cap.read()
            require(ok and isinstance(frame, np.ndarray) and frame.dtype == np.uint8
                    and frame.ndim == 3 and frame.shape[2] == 3, "采样帧解码失败；未补帧或替换")
            position = cap.get(cv2.CAP_PROP_POS_FRAMES)
            require(math.isfinite(position) and abs(position - (index + 1)) < 0.5,
                    "解码器报告的帧位置不匹配，拒绝记录不可靠索引")
            height, width = frame.shape[:2]
            if source_size is None:
                source_size = [width, height]
                scale = min(1.0, MAX_EDGE / max(width, height))
                target_size = [max(1, round(width * scale)), max(1, round(height * scale))]
            require([width, height] == source_size, "视频中途改变尺寸；请先分段或规范化后重试")
            if target_size != source_size:
                frame = cv2.resize(frame, tuple(target_size), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            require(ok, "JPEG编码失败")
            content = encoded.tobytes()
            name = f"frame{order:02d}.jpg"
            write_new(output / name, content)
            created.append(output / name)
            reported_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            records.append({"path": name, "frame_index_zero_based": index,
                            "nominal_timestamp_s": index / fps,
                            "decoder_reported_timestamp_ms": reported_ms if math.isfinite(reported_ms) and reported_ms >= 0 else None,
                            "width": target_size[0], "height": target_size[1],
                            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        after = source.stat()
        require((info.st_size, info.st_mtime_ns, info.st_ctime_ns) ==
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "原视频处理期间发生变化，拒绝发布清单")
        warnings = []
        if actual_count < count:
            warnings.append("原视频帧数不足，只输出不同真实帧，没有复制补齐")
        if frame_count / fps > 60:
            warnings.append("超过60秒：请先分成短片段再编码，整段8帧可能遗漏检索内容")
        manifest = {"schema_version": 1, "source_path": str(source), "source_sha256": source_hash,
                    "source_bytes": info.st_size, "source_frame_count_reported": frame_count,
                    "source_fps_reported": fps, "nominal_duration_s": frame_count / fps,
                    "requested_frames": count, "actual_frames": actual_count,
                    "sampling": "midpoint of equal frame-index intervals; chronological; no repeated padding",
                    "timestamp_note": "index/FPS是名义时间；OpenCV/容器报告可能近似，变帧率不保证精确时间戳",
                    "resize": {"source_width_height": source_size, "output_width_height": target_size,
                               "max_edge": MAX_EDGE, "upscale": False, "preserve_aspect_ratio": True,
                               "interpolation": "INTER_AREA", "jpeg_quality": 90},
                    "decoder": {"backend": cap.getBackendName(), "opencv": cv2.__version__, "numpy": np.__version__,
                                "allowed_protocols": ["file"]},
                    "warnings": warnings, "frames": records}
        write_new(output / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
        created.append(output / "manifest.json")
        return {"ok": True, "output_dir": str(output), "manifest": str(output / "manifest.json"),
                "frames": actual_count, "width": target_size[0], "height": target_size[1], "warnings": warnings}
    except Exception:
        # 仅清理由本次独占创建的文件，不递归删除调用者或其他进程新增的内容。
        for path in reversed(created):
            path.unlink(missing_ok=True)
        if owns_output:
            try:
                output.rmdir()
            except OSError:
                pass
        raise
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="本地短视频文件")
    parser.add_argument("--output-dir", required=True, type=Path, help="尚不存在的新目录")
    parser.add_argument("--frames", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.video, args.output_dir, args.frames), ensure_ascii=False))
        return 0
    except (OSError, ValueError, cv2.error) as error:
        message = str(error) if isinstance(error, ValueError) else "本地视频处理失败: " + type(error).__name__
        print(json.dumps({"ok": False, "error": message}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
