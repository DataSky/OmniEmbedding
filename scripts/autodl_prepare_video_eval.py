#!/usr/bin/env python3
"""Prepare a reproducible real-video MSR-VTT subset; never calls inference.

Requires requests, NumPy and OpenCV. Source files are public and downloaded
without Hugging Face credentials. The source does not declare a dataset license.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import threading
import time

import cv2
import numpy as np
import requests

REPO = "VLM2Vec/MSR-VTT"
REVISION = "947b139d0c591235a3f311e45a31a88461bad0d2"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
PAPER = "https://www.microsoft.com/en-us/research/?p=238188"
ANNOTATION_MIRROR = "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/c1af215a96934854f42683c19c51391aaee6f962/raw_data/MSRVTT_data.json"
# This is the LFS object SHA in VLM2Vec's pinned revision, not a second annotation version.
ANNOTATION_SHA256 = "3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def jsonl(path, records):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                    encoding="utf-8")


class Downloader:
    def __init__(self, output, budget):
        self.output, self.budget = output, budget
        self.network_bytes = 0
        self.lock = threading.Lock()
        self.sessions = threading.local()

    def download(self, relative, local, maximum, url=None):
        path = self.output / local
        if path.is_file():
            data = path.read_bytes()
            if len(data) <= maximum:
                return path, data
            raise RuntimeError("existing file exceeds limit: " + local)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(self.sessions, "http"):
            self.sessions.http = requests.Session()
            self.sessions.http.trust_env = False  # No implicit .netrc authentication or proxy credentials.
        for attempt in range(3):
            try:
                with self.sessions.http.get(url or BASE + relative, stream=True,
                                            timeout=(10, 45)) as response:
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise RuntimeError("expected complete HTTP 200 download")
                    data = bytearray()
                    for block in response.iter_content(64 * 1024):
                        with self.lock:
                            self.network_bytes += len(block)
                            if self.network_bytes > self.budget:
                                raise RuntimeError("download byte budget exceeded")
                        data.extend(block)
                        if len(data) > maximum:
                            raise RuntimeError("source file exceeded size limit")
                temporary = path.with_suffix(path.suffix + ".download")
                temporary.write_bytes(data)
                temporary.replace(path)
                return path, bytes(data)
            except (requests.RequestException, OSError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError("unreachable")


def extract_frames(path, directory, output_root):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("MP4 decode failed: " + path.name)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count < 8 or not np.isfinite(fps) or fps <= 0:
            raise RuntimeError("video has insufficient frames or invalid FPS: " + path.name)
        indices = [min(count - 1, int((i + 0.5) * count / 8)) for i in range(8)]
        assert len(set(indices)) == 8
        directory.mkdir(parents=True, exist_ok=True)
        paths, hashes, original_sizes = [], [], []
        for ordinal, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not read real frame {index}: {path.name}")
            height, width = frame.shape[:2]
            original_sizes.append([width, height])
            scale = min(512 / width, 288 / height)
            target_width, target_height = max(1, round(width * scale)), max(1, round(height * scale))
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((288, 512, 3), dtype=np.uint8)
            x, y = (512 - target_width) // 2, (288 - target_height) // 2
            canvas[y:y + target_height, x:x + target_width] = resized
            destination = directory / f"{ordinal:02d}.jpg"
            success, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                raise RuntimeError("JPEG encode failed")
            data = encoded.tobytes()
            destination.write_bytes(data)
            paths.append(destination.relative_to(output_root).as_posix())
            hashes.append(sha256(data))
        return {"frame_paths": paths, "frame_indices": indices, "frame_sha256": hashes,
                "distinct_sampled_frame_hashes": len(set(hashes)), "original_frame_sizes": original_sizes,
                "original_frame_count": count, "fps": fps, "duration_s": count / fps}
    finally:
        capture.release()


def query_files(output, caption_rows, suffix=""):
    by_text = defaultdict(set)
    for video_id, text in caption_rows:
        by_text[text.strip()].add(video_id)
    queries, qrels = [], []
    for text, videos in sorted(by_text.items()):
        assert text
        query_id = "q-" + sha256(text.encode())[:24]
        queries.append({"id": query_id, "text": text})
        qrels.extend({"query_id": query_id, "corpus_id": video_id, "score": 1}
                     for video_id in sorted(videos))
    jsonl(output / f"queries{suffix}.jsonl", queries)
    jsonl(output / f"qrels{suffix}.jsonl", qrels)
    return {"queries": len(queries), "qrels": len(qrels),
            "queries_with_multiple_positive_videos": sum(len(v) > 1 for v in by_text.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("data/autodl-evaluation-20261018/datasets/video"))
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-download-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    assert 50 <= args.count <= 1000 and 1 <= args.workers <= 16
    assert 0 < args.max_download_bytes <= 1_000_000_000
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cv2.setNumThreads(1)
    downloader = Downloader(output, args.max_download_bytes)
    sources = {}
    for relative in ("README.md", "msrvtt_test_1k.json", "raw_data/MSRVTT_data.json",
                     "raw_data/MSRVTT_train.9k.csv"):
        local = "sources/" + relative
        mirror = ANNOTATION_MIRROR if relative == "raw_data/MSRVTT_data.json" else None
        _, data = downloader.download(relative, local, 50 * 1024 * 1024, mirror)
        if mirror:
            assert sha256(data) == ANNOTATION_SHA256, "annotation mirror differs from pinned original LFS object"
        sources[relative] = {"url": BASE + relative, "download_url": mirror or BASE + relative,
                             "bytes": len(data), "sha256": sha256(data)}
    official = json.loads((output / "sources/msrvtt_test_1k.json").read_text())
    raw = json.loads((output / "sources/raw_data/MSRVTT_data.json").read_text())
    official_by_id = {row["video_id"]: row for row in official}
    assert len(official) == len(official_by_id) == 1000, "expected exact 1K-A source split"
    raw_by_id = {row["video_id"]: row for row in raw["videos"]}
    with (output / "sources/raw_data/MSRVTT_train.9k.csv").open(newline="") as source:
        train_ids = {row["video_id"] for row in csv.DictReader(source)}
    assert len(train_ids) == 9000 and train_ids.isdisjoint(official_by_id), "train/test split overlap"
    assert train_ids | set(official_by_id) == set(raw_by_id), "train/test IDs do not cover the original 10k"
    selected = random.Random(args.seed).sample(sorted(official_by_id), args.count)
    captions = defaultdict(list)
    for row in raw["sentences"]:
        if row["video_id"] in selected:
            captions[row["video_id"]].append({"sen_id": row["sen_id"], "text": row["caption"]})
    assert all(len(captions[video_id]) == 20 for video_id in selected), "expected 20 original captions per video"
    normalize = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
    exact_caption_matches = sum(official_by_id[video_id]["caption"] in
        {row["text"] for row in captions[video_id]} for video_id in selected)
    assert all(normalize(official_by_id[video_id]["caption"]) in
               {normalize(row["text"]) for row in captions[video_id]} for video_id in selected), \
        "1K-A caption does not match original annotation after case/punctuation normalization"

    def prepare(video_id):
        relative = f"raw_videos/{video_id}.mp4"
        path, data = downloader.download(relative, f"mp4/{video_id}.mp4", 64 * 1024 * 1024)
        decoded = extract_frames(path, output / "frames" / video_id, output)
        return {"id": video_id, "mp4_path": path.relative_to(output).as_posix(),
                "source_url": BASE + relative, "mp4_bytes": len(data),
                "mp4_sha256": sha256(data), **decoded}

    completed = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare, video_id): video_id for video_id in selected}
        for future in as_completed(futures):
            video_id = futures[future]
            completed[video_id] = future.result()
            if len(completed) % 16 == 0 or len(completed) == len(selected):
                print(json.dumps({"prepared_videos": len(completed), "total": len(selected),
                                  "downloaded_bytes": downloader.network_bytes}), flush=True)
    videos = [completed[video_id] for video_id in selected]
    assert len({row["mp4_sha256"] for row in videos}) == args.count, "duplicate MP4 content"
    jsonl(output / "corpus.jsonl", ({"id": row["id"], "frame_paths": row["frame_paths"]} for row in videos))
    jsonl(output / "original-caption-mapping.jsonl", (
        {"video_id": video_id, "selection_split": "test_1k", "raw_metadata_split": raw_by_id[video_id]["split"],
         "official_1ka": official_by_id[video_id],
         "all_original_captions": captions[video_id]} for video_id in selected))
    expanded = query_files(output, ((video_id, caption["text"]) for video_id in selected
                                   for caption in captions[video_id]))
    standard = query_files(output, ((video_id, official_by_id[video_id]["caption"])
                                   for video_id in selected), "_1ka")
    file_hashes = {name: sha256((output / name).read_bytes()) for name in (
        "corpus.jsonl", "queries.jsonl", "qrels.jsonl", "queries_1ka.jsonl", "qrels_1ka.jsonl",
        "original-caption-mapping.jsonl")}
    manifest = {
        "dataset": "MSR-VTT real-video fixed subset", "source": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION, "paper": PAPER, "gated": False, "split": "test_1k (1K-A)",
        "original_candidate_count": 1000, "candidate_count": args.count, "seed": args.seed,
        "distinct_original_source_urls": len({official_by_id[video_id]["url"] for video_id in selected}),
        "split_validation": {"train_9k_count": len(train_ids), "test_1k_count": len(official_by_id),
            "intersection_count": 0, "union_count": len(raw_by_id),
            "authority": "msrvtt_test_1k.json and raw_data/MSRVTT_train.9k.csv",
            "raw_metadata_split_warning": "All 10,000 videos in raw MSRVTT_data.json are tagged train, including all official test_1k IDs. That inconsistent field is retained for auditing and is not used to define this benchmark split."},
        "license": None,
        "license_note": "Mirror dataset card does not declare a license. Original video rights remain with their owners; this preparation does not grant redistribution rights.",
        "protocol_note": "Fixed smaller candidate pool, not the full standard 1K-A benchmark. Primary queries use all original captions; *_1ka files retain its official single-caption protocol on the same subset.",
        "caption_rule": "Original text retained in mapping. Query text strips surrounding whitespace only; exact duplicate text is one query with all corresponding sampled videos as positives.",
        "sampling_rule": "random.Random(seed).sample(sorted(official_test_1k_video_ids), count), no replacements or train/validation videos",
        "selected_video_ids": selected, "sample_order_sha256": sha256(("\n".join(selected) + "\n").encode()),
        "frame_paths_base": "directory containing corpus.jsonl", "frame_shape_rgb": [8, 288, 512, 3],
        "frame_rule": "Eight distinct midpoint indices floor((i+0.5)*original_frame_count/8); decode actual MP4 frames; preserve aspect ratio and center-letterbox black to 512x288; JPEG quality 90. No frame duplication or synthetic frames.",
        "caption_statistics": expanded, "official_single_caption_statistics": standard,
        "caption_alignment": {"official_caption_exact_matches": exact_caption_matches,
            "official_caption_matches_after_case_punctuation_normalization": len(selected),
            "normalization_used_only_for_cross_source_validation": True},
        "network_download_bytes_this_run": downloader.network_bytes, "maximum_download_bytes": args.max_download_bytes,
        "unique_source_payload_bytes": sum(row["mp4_bytes"] for row in videos) + sum(row["bytes"] for row in sources.values()),
        "mp4_bytes": sum(row["mp4_bytes"] for row in videos), "sources": sources,
        "artifacts_sha256": file_hashes, "videos": videos,
        "software": {"opencv": cv2.__version__, "numpy": np.__version__},
        "preparation_elapsed_s": round(time.monotonic() - started, 3),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"ok": True, "output": str(output), "videos": len(videos),
                      "captions": sum(len(rows) for rows in captions.values()), **expanded,
                      "network_download_bytes": downloader.network_bytes,
                      "elapsed_s": manifest["preparation_elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
