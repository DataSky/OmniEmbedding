#!/usr/bin/env python3
"""Build an offline Chinese technical report from actual AutoDL evaluation files.

Reads only explicitly named evaluation summaries, manifests, sanitized benchmark
JSONL, telemetry JSONL, paired quality deltas, hybrid/stability analyses,
reviewed-analysis.json, and service-before/after.json.
Never reads credentials, endpoint secrets, snapshots, or the old 4090 report.
Requires matplotlib and numpy. No network access is used.
"""
from __future__ import annotations

import argparse
import base64
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


LABELS = {
    "scifact": "SciFact · 英文科学证据 · 全量",
    "belebele-zho_Hans": "Belebele · 简体中文 · 小候选池",
    "belebele-eng_Latn": "Belebele · 英文 · 小候选池",
    "video-text-zh-subset": "中文视频标题文本检索 · 固定 20K 子集",
    "coco-modality-120": "COCO 跨模态诊断 · 120 候选",
    "coco5k": "COCO 图文检索 · 全量 5K",
    "video": "MSR-VTT 真实视频 · 固定 128 子集",
}
SENSITIVE = re.compile(r"authorization|password|secret|access.?token|api.?key|private.?key", re.I)
JWT = re.compile(r"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}")
COLORS = ["#1768ac", "#e07a19", "#2b8c65", "#ab49a5", "#bf4141", "#57737a", "#856b24"]


def clean(value):
    if isinstance(value, dict):
        return {str(k): ("[已隐藏敏感字段]" if SENSITIVE.search(str(k)) else clean(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        return JWT.sub("[已隐藏 Token]", value)
    return value


def esc(value):
    return html.escape(str(clean(value)), quote=True)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=2, percent=False):
    if value is None:
        return "待测"
    if isinstance(value, bool):
        return "是" if value else "否"
    if finite(value):
        if percent:
            return f"{value * 100:.{digits}f}%"
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:,.{digits}f}"
    return str(value)


def table(headers, rows, cls=""):
    if not rows:
        return '<p class="pending">待测：尚无对应的完整结果文件。</p>'
    return (f'<div class="table-scroll {cls}"><table><thead><tr>' +
            "".join(f"<th>{esc(x)}</th>" for x in headers) + "</tr></thead><tbody>" +
            "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows) +
            "</tbody></table></div>")


def render_value(value):
    value = clean(value)
    if isinstance(value, dict):
        rendered = []
        for key, child in value.items():
            shown = render_value(child)
            count = value.get("count", value.get("n"))
            if key == "p99" and finite(child) and (not finite(count) or count < 1000):
                shown += f'<small>探索性 p99 · n={esc(fmt(count))}</small>'
            rendered.append(f"<dt>{esc(key)}</dt><dd>{shown}</dd>")
        return '<dl class="analysis">' + "".join(rendered) + "</dl>"
    if isinstance(value, list):
        return "<ul>" + "".join(f"<li>{render_value(x)}</li>" for x in value) + "</ul>"
    return "<p>" + esc("未提供" if value is None else value).replace("\n", "<br>") + "</p>"


class Report:
    def __init__(self, root):
        self.root = root.resolve()
        self.charts = self.root / "charts"
        self.charts.mkdir(parents=True, exist_ok=True)
        self.inputs, self.warnings, self.csv_rows = set(), [], []
        self.figures = {}
        self.quality = {}
        self.manifests = {}
        self.cases = []
        self.performance = []
        self.telemetry = []
        self.soak = []
        self.reviewed = None
        self.deltas = None
        self.hybrid = None
        self.stability_analysis = None
        self.hardware = None
        self.supporting_files = []
        self.run_settings = {}
        self.service = {}
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.alpha": .19,
                             "figure.dpi": 130, "savefig.dpi": 170,
                             "axes.prop_cycle": matplotlib.cycler(color=COLORS),
                             "svg.fonttype": "none"})

    def within(self, path):
        path = path.resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return None
        # Even malformed input paths cannot cause the reporter to read a secret file.
        if re.search(r"(?:inference\.key|credentials|snapshot|\.env)(?:$|[/\.])", str(path), re.I):
            return None
        return path

    def read_json(self, path):
        path = self.within(path)
        if path is None or not path.is_file():
            return None
        self.inputs.add(path)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            self.warnings.append(f"{path.relative_to(self.root)} 尚不可解析：{type(error).__name__}；视为待完成。")
            return None

    def read_lines(self, path):
        path = self.within(path)
        if path is None or not path.is_file():
            return
        self.inputs.add(path)
        bad = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
                    else:
                        bad += 1
                except ValueError:
                    bad += 1
        if bad:
            self.warnings.append(f"{path.relative_to(self.root)} 有 {bad} 行无法解析，已跳过。")

    def rel(self, path):
        return str(path.relative_to(self.root))

    def link(self, path):
        return f'<a href="{quote(self.rel(path))}">{esc(self.rel(path))}</a>'

    @staticmethod
    def location(path):
        name = str(path).lower()
        if "public" in name:
            return "公网"
        if "local" in name or "loopback" in name:
            return "服务器本机"
        return "路径未标注"

    def discover(self):
        for path in sorted(self.root.glob("datasets/*/manifest.json")):
            value = self.read_json(path)
            if isinstance(value, dict):
                self.manifests[path.parent.name] = value
        for path in sorted(self.root.glob("quality/*/summary.json")):
            value = self.read_json(path)
            if isinstance(value, dict) and isinstance(value.get("dimensions"), dict):
                self.quality[path.parent.name] = {**value, "_path": path}
        unique = {}
        for path in sorted(self.root.glob("performance/**/summary.json"), key=lambda x: x.stat().st_mtime):
            value = self.read_json(path)
            if not isinstance(value, dict):
                continue
            self.performance.append({**value, "_path": path})
            for case in value.get("cases", []):
                if not isinstance(case, dict):
                    continue
                key = tuple(case.get(k) for k in ["run_id", "case_run_id", "phase", "case"])
                if key in unique and {k: v for k, v in unique[key].items() if not k.startswith("_")} != case:
                    self.warnings.append(f"同一性能 case 有多份不同摘要，采用后写入的 {self.rel(path)}。")
                unique[key] = {**case, "_path": path, "_location": self.location(path)}
        self.cases = list(unique.values())
        self.reviewed = self.read_json(self.root / "reviewed-analysis.json")
        if not isinstance(self.reviewed, dict):
            self.reviewed = None
        self.deltas = self.read_json(self.root / "quality-dimension-deltas.json")
        if not isinstance(self.deltas, dict):
            self.deltas = None
        self.hybrid = self.read_json(self.root / "hybrid-results.json")
        if not isinstance(self.hybrid, dict):
            self.hybrid = None
        self.stability_analysis = self.read_json(self.root / "stability-analysis.json")
        if not isinstance(self.stability_analysis, dict):
            self.stability_analysis = None
        hardware = self.read_json(self.root / "hardware.json")
        if isinstance(hardware, dict):
            allowed = ["gpu", "gpu_memory_mib", "driver", "torch", "cuda", "server_kernel",
                       "server_cgroup_memory_bytes", "server_cpu_quota", "model", "revision", "dtype",
                       "gpu_hourly_cny", "api_worker_count", "inference_lock_capacity", "max_text_batch",
                       "encoder_micro_batch", "client_system", "client_arch", "client_python"]
            self.hardware = {key: hardware[key] for key in allowed if key in hardware}
        supporting = [self.root / name for name in ["REPRODUCE.txt", "recommended-client-settings.json",
                      "quality-review-final.json", "performance/performance-review.json", "source-snapshots/cloud-manifest.json"]]
        supporting += sorted(self.root.glob("evaluation-source-manifest*.json"))
        for path in supporting:
            path = self.within(path)
            if path is not None and path.is_file():
                self.inputs.add(path)
                self.supporting_files.append(path)
        for name in ["service-before.json", "service-after.json"]:
            value = self.read_json(self.root / name)
            if isinstance(value, dict):
                self.service[name] = {key: value.get(key) for key in ["model", "model_revision", "device", "dtype", "boot_id", "ready", "status", "uptime_s"]}
        for path in [self.root / "telemetry.jsonl", *sorted(self.root.glob("performance/**/telemetry.jsonl"))]:
            if path.is_file():
                for record in self.read_lines(path):
                    fields = ["time_utc", "elapsed_s", "gpu_util_percent", "gpu_memory_mib", "gpu_total_mib",
                              "gpu_temperature_c", "gpu_power_w", "worker_pid", "worker_rss_kib", "error_class"]
                    self.telemetry.append({**{k: record.get(k) for k in fields}, "_path": path})
        # Sanitized raw benchmark records provide soak chronology; only whitelisted fields are kept.
        for path in sorted(self.root.glob("performance/**/raw-*.jsonl")):
            for record in self.read_lines(path):
                if record.get("record_type") == "run_start":
                    self.run_settings[record.get("run_id")] = {key: record.get(key) for key in ["input_type", "mode"]}
                if record.get("record_type") == "request" and str(record.get("phase", "")).startswith("soak"):
                    fields = ["run_id", "at", "phase", "case", "success", "status", "latency_ms",
                              "started_offset_s", "finished_offset_s", "boot_id", "anchor_cosine"]
                    self.soak.append({**{k: record.get(k) for k in fields}, "_path": path})

    @staticmethod
    def workload(case):
        name = str(case.get("case", ""))
        match = re.search(r"(\d+)x(\d+)_real_jpeg", name)
        if match:
            return f"真实 JPEG · {match[1]}×{match[2]}"
        if "256px" in name and (name.startswith("image_") or name.startswith("video_")):
            return "合成 PNG · 256×256"
        if name.startswith(("text_", "document_")):
            return "文本"
        if name.startswith("health_"):
            return "探活"
        return "校验／其他"

    def text_role(self, case):
        name = str(case.get("case", ""))
        if not name.startswith(("text_", "document_")):
            return None
        return self.run_settings.get(case.get("run_id"), {}).get("input_type") or ("document" if name.startswith("document_") else None)

    @staticmethod
    def tail_value(distribution, percentile="p99"):
        value, count = distribution.get(percentile), distribution.get("count", distribution.get("n"))
        shown = esc(fmt(value))
        if value is not None and percentile == "p99" and (not finite(count) or count < 1000):
            shown += f'<small>探索性 · n={esc(fmt(count))}</small>'
        return shown

    def save_figure(self, key, figure, caption):
        figure.tight_layout()
        for extension in ["png", "svg"]:
            figure.savefig(self.charts / f"{key}.{extension}", bbox_inches="tight", facecolor="white")
        plt.close(figure)
        encoded = base64.b64encode((self.charts / f"{key}.png").read_bytes()).decode()
        self.figures[key] = (f'<figure><img src="data:image/png;base64,{encoded}" alt="{esc(caption)}">'
                             f'<figcaption>{esc(caption)}</figcaption></figure>')

    def figure(self, key, missing):
        return self.figures.get(key, f'<p class="pending">待测：{esc(missing)}</p>')

    @staticmethod
    def metric_ci(value, metric):
        for name in ["positive_source_cluster_ci95", "query_bootstrap_ci95"]:
            ci = value.get(name, {}).get(metric)
            if isinstance(ci, list) and len(ci) == 2 and all(finite(x) for x in ci):
                return ci, name
        return None, None

    def quality_records(self):
        records = []
        for dataset, summary in self.quality.items():
            multimodal = bool(summary.get("reverse_text_candidates_use_document_prefix"))
            for dimension, result in summary["dimensions"].items():
                if not str(dimension).isdigit() or not isinstance(result, dict):
                    continue
                for direction, data in [("文字→媒体" if multimodal else "文字→文字", result),
                                        ("媒体→文字", result.get("media_to_text"))]:
                    if not isinstance(data, dict):
                        continue
                    records.append({"dataset": dataset, "dimension": int(dimension), "direction": direction,
                                    "multimodal": multimodal, "value": data, "summary": summary})
        return records

    def chart_quality(self):
        records = self.quality_records()
        for key, multimodal, metric, title in [("text-quality", False, "nDCG@10", "Text retrieval: nDCG@10"),
                                               ("multimodal-quality", True, "Hit@1", "Cross-modal retrieval: Hit@1")]:
            datasets = sorted({r["dataset"] for r in records if r["multimodal"] == multimodal})
            if not datasets:
                continue
            columns = min(2, len(datasets))
            rows = math.ceil(len(datasets) / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(7.1 * columns, 4.1 * rows), squeeze=False)
            for ax, dataset in zip(axes.flat, datasets):
                relevant = [r for r in records if r["dataset"] == dataset and r["multimodal"] == multimodal]
                directions = sorted({r["direction"] for r in relevant})
                for i, direction in enumerate(directions):
                    points = sorted([r for r in relevant if r["direction"] == direction], key=lambda r: r["dimension"])
                    xs, ys, low, high = [], [], [], []
                    for point in points:
                        y = point["value"].get("metrics", {}).get(metric)
                        if not finite(y):
                            continue
                        ci, _ = self.metric_ci(point["value"], metric)
                        xs.append(point["dimension"]); ys.append(y)
                        low.append(max(0, y - ci[0]) if ci else 0)
                        high.append(max(0, ci[1] - y) if ci else 0)
                    if xs:
                        label = {"文字→文字": "text -> text", "文字→媒体": "text -> media", "媒体→文字": "media -> text"}[direction]
                        ax.errorbar(xs, ys, yerr=[low, high], marker="o", capsize=3, linewidth=1.8, label=label, color=COLORS[i])
                baseline = self.quality[dataset].get("lexical_char_tfidf_baseline", {}).get("metrics", {}).get(metric)
                if not multimodal and finite(baseline):
                    ax.axhline(baseline, linestyle="--", color="#777777", label="char TF-IDF baseline")
                ax.set_title(dataset, loc="left", fontweight="bold")
                ax.set_xlabel("Embedding dimensions"); ax.set_ylabel(metric)
                ax.set_xticks([128, 256, 512, 768]); ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.legend(fontsize=8)
            for ax in list(axes.flat)[len(datasets):]: ax.set_visible(False)
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
            self.save_figure(key, fig, "各数据集独立展示；误差线优先采用正例来源聚类 bootstrap 95% 区间，缺失时采用 query bootstrap。没有计算跨数据集总平均。")

    def chart_latency(self):
        cases = [c for c in self.cases if c.get("phase") == "latency" and c["_location"] == "公网"
                 and finite(c.get("text_chars")) and c.get("batch", 0) > 0 and c.get("success_count", 0) > 0]
        if not cases:
            return
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))
        groups = defaultdict(list)
        for c in cases:
            groups[(c.get("text_chars"), c.get("run_id"))].append(c)
        for (length, run), points in groups.items():
            points.sort(key=lambda c: c["batch"])
            xs = [p["batch"] for p in points]
            axes[0].plot(xs, [p.get("success_http_latency_ms", {}).get("p95", float("nan")) for p in points], marker="o", label=f"{length} chars")
            axes[1].plot(xs, [p.get("success_items_per_second", float("nan")) for p in points], marker="o", label=f"{length} chars")
        for ax in axes:
            ax.set_xlabel("Items per request (batch)"); ax.set_xscale("log", base=2)
            ticks = sorted({p["batch"] for p in cases}); ax.set_xticks(ticks, labels=ticks)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Validated success HTTP p95 (ms)")
        axes[1].set_ylabel("Validated successful items / second")
        fig.suptitle("Public endpoint, C=1: batch size and text length", fontsize=14, fontweight="bold")
        self.save_figure("public-batch", fig, "公网、单个在途请求的真实结果。吞吐只统计向量校验通过的成功条目，并包含完整测量窗口与收尾时间；未将快速拒绝计为成功吞吐。")

    def chart_load(self):
        # Closed-loop concurrency and scheduled arrivals answer different questions.
        # Select representative protocols from metadata, never by which result is best.
        raw = [c for c in self.cases if c.get("phase") == "load" and c.get("target_rps") is None and finite(c.get("concurrency"))]
        candidates = defaultdict(list)
        for case in raw:
            candidates[(case["_location"], case["case"], case.get("run_id"))].append(case)
        chosen = []
        for location in sorted({case["_location"] for case in raw}):
            series = [points for (where, name, run), points in candidates.items() if where == location]
            for role in ["query", "document"]:
                options = [points for points in series if self.text_role(points[0]) == role]
                if options:
                    def rank(points):
                        first = points[0]
                        chars, batch = first.get("text_chars"), first.get("batch")
                        preferred = (chars == 16 and batch == 1) if role == "query" else (chars == 1024 and batch == 16)
                        return (not preferred, abs((chars or 0) - (16 if role == "query" else 1024)),
                                abs((batch or 0) - (1 if role == "query" else 16)), -first["_path"].stat().st_mtime)
                    chosen.extend(min(options, key=rank))
            for prefix in ["image_", "video_2frames_", "video_8frames_"]:
                options = [points for points in series if str(points[0].get("case", "")).startswith(prefix)
                           and "_real_jpeg" in str(points[0].get("case", ""))]
                if options:
                    selected = min(options, key=lambda p: ("512x288_real_jpeg" not in p[0]["case"], -p[0]["_path"].stat().st_mtime))
                    chosen.extend(selected)
        cases = chosen
        locations = [x for x in ["公网", "服务器本机", "路径未标注"] if any(c["_location"] == x for c in cases)]
        if not cases:
            return
        fig, axes = plt.subplots(2, len(locations), figsize=(7.0 * len(locations), 8), squeeze=False)
        labels = {"公网": "Public endpoint", "服务器本机": "Server loopback", "路径未标注": "Unlabelled location"}
        for column, location in enumerate(locations):
            groups = defaultdict(list)
            for c in cases:
                if c["_location"] == location:
                    groups[(c["case"], c.get("run_id"))].append(c)
            for i, ((case, run), points) in enumerate(groups.items()):
                points.sort(key=lambda c: c["concurrency"])
                xs = [p["concurrency"] for p in points]
                label = case
                axes[0, column].plot(xs, [p.get("success_rps", float("nan")) for p in points], marker="o", label=label, color=COLORS[i % len(COLORS)])
                axes[1, column].plot(xs, [p.get("http_429_rate") if finite(p.get("http_429_rate")) else float("nan") for p in points], marker="o", color=COLORS[i % len(COLORS)])
                for p in points:
                    if not p.get("sample_valid") or p.get("interrupted") or p.get("drain_timeout"):
                        axes[0, column].scatter([p["concurrency"]], [p.get("success_rps", 0)], marker="x", color="black", s=65, zorder=5)
            for ax in axes[:, column]:
                ax.set_xlabel("Concurrent in-flight HTTP requests (C)"); ax.set_xscale("log", base=2)
                ticks = sorted({p["concurrency"] for p in cases if p["_location"] == location})
                ax.set_xticks(ticks, labels=ticks)
            axes[0, column].set_title(labels[location], loc="left", fontweight="bold")
            axes[0, column].set_ylabel("Validated successful requests / second")
            axes[1, column].set_ylabel("HTTP 429 / started HTTP attempts")
            axes[1, column].yaxis.set_major_formatter(PercentFormatter(1))
            axes[0, column].legend(fontsize=7, loc="best")
        fig.suptitle("Raw load: goodput and overload rejection", fontsize=14, fontweight="bold")
        self.save_figure("load-concurrency", fig, "仅展示 target_rps 为空的原始闭环并发实验，公网与服务器本机分面。每位置按预先定义的工作负载元数据选择短 query、长 document 批次及真实 JPEG 图片／2帧／8帧视频，每类取最新符合协议的一组，不依据结果优劣筛选；其他用例及合成 PNG 保留在全表。黑叉表示样本不足、中断或收尾超时。429 比率分母为实际发起的 HTTP 尝试，包含传输失败。")

    def chart_arrivals(self):
        cases = [c for c in self.cases if c.get("phase") == "load" and finite(c.get("target_rps"))]
        if not cases:
            return
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
        groups = defaultdict(list)
        for case in cases:
            groups[(case["_location"], case.get("case"), case.get("concurrency"), self.text_role(case))].append(case)
        locations = {"公网": "Public", "服务器本机": "Server loopback", "路径未标注": "Unlabelled"}
        for i, ((location, name, concurrency, role), points) in enumerate(groups.items()):
            points.sort(key=lambda c: c["target_rps"])
            xs = [p["target_rps"] for p in points]
            label = f"{locations[location]} | {name} | C={concurrency}"
            color = COLORS[i % len(COLORS)]
            metrics = [("success_rps", None), ("success_http_latency_ms", "p95"),
                       ("success_client_queue_ms", "p95"), ("success_end_to_end_ms", "p95")]
            for ax, (key, quantile) in zip(axes.flat, metrics):
                ys = [p.get(key) if quantile is None else p.get(key, {}).get(quantile) for p in points]
                ax.plot(xs, [v if finite(v) else float("nan") for v in ys], marker="o", color=color, label=label)
        for ax in axes.flat:
            ax.set_xlabel("Planned request arrival rate (requests / second)")
            ax.set_xticks(sorted({c["target_rps"] for c in cases}))
            ax.legend(fontsize=7)
        axes[0, 0].set_ylabel("Validated successful requests / second")
        axes[0, 1].set_ylabel("Successful HTTP p95 (ms)")
        axes[1, 0].set_ylabel("Successful client-queue p95 (ms)")
        axes[1, 1].set_ylabel("Scheduled arrival to full body p95 (ms)")
        fig.suptitle("Scheduled arrivals: achieved goodput and queue delay", fontsize=14, fontweight="bold")
        self.save_figure("scheduled-arrivals", fig, "固定计划到达率实验的横轴是 target_rps，不是并发 C；仅连接相同位置、工作负载与 C 的实际测点，不跨协议求平均。成功吞吐包含收尾时间；HTTP、客户端排队、计划到达到响应 body 收完分开度量，不含随后的本机 JSON 解析与向量校验。即使没有 429，也可能在客户端积累队列，不能只凭拒绝率判断到达率可持续。")

    def arrival_section(self):
        rows = []
        for case in sorted([c for c in self.cases if c.get("phase") == "load" and finite(c.get("target_rps"))],
                           key=lambda c: (c["_location"], c.get("case", ""), c["target_rps"])):
            rows.append([esc(case["_location"]), esc(case.get("case")), esc(fmt(case.get("concurrency"))),
                         esc(fmt(case.get("target_rps"))), esc(fmt(case.get("arrival_window_s"))), esc(fmt(case.get("elapsed_s"))),
                         esc(fmt(case.get("offered_requests"))), esc(fmt(case.get("http_attempts"))), esc(fmt(case.get("success_count"))),
                         esc(fmt(case.get("success_rps"))), esc(fmt(case.get("success_http_latency_ms", {}).get("p95"))),
                         esc(fmt(case.get("success_client_queue_ms", {}).get("p95"))), esc(fmt(case.get("success_end_to_end_ms", {}).get("p95"))),
                         esc(fmt(case.get("http_429_rate"), percent=True)), esc(json.dumps(case.get("error_counts", {}), ensure_ascii=False))])
        body = '<h3>固定到达率与客户端排队</h3>' + self.figure("scheduled-arrivals", "计划到达率实验摘要尚未完成。")
        body += table(["位置", "用例", "C", "计划 RPS", "到达窗口 s", "含收尾总时长 s", "计划请求", "实际 HTTP 尝试", "成功数", "成功 RPS", "HTTP p95 ms", "客户端排队 p95 ms", "计划到达至 body 收完 p95 ms", "429 比率", "错误／客户端丢弃"], rows)
        return body + '<p class="note">这组实验固定请求的计划到达节奏，由 C 限制实际同时发出的 HTTP 请求数；与闭环并发压测口径不同。客户端尚未发出的请求也可能等待、丢弃或在收尾阶段超时。表中延迟分位数仅针对成功请求，须同时查看计划数、实际 HTTP 尝试数、成功数及错误计数，避免忽略排队和未完成请求。</p>'

    def chart_soak(self):
        records = [r for r in self.soak if r.get("phase") == "soak" and finite(r.get("started_offset_s"))]
        if not records:
            return
        fig, axes = plt.subplots(2, 1, figsize=(13, 6.7), sharex=True)
        groups = defaultdict(list)
        for record in records: groups[(str(record["_path"].parent.name), record.get("run_id"))].append(record)
        for (location, run), rows in groups.items():
            start = min(r["started_offset_s"] for r in rows)
            bins = defaultdict(list)
            for r in rows: bins[int((r["started_offset_s"] - start) // 30)].append(r)
            xs, p50s, p95s, rates = [], [], [], []
            for bucket, values in sorted(bins.items()):
                good = [r["latency_ms"] for r in values if r.get("success") and finite(r.get("latency_ms"))]
                xs.append((bucket * 30 + 15) / 60)
                p50s.append(float(np.quantile(good, .5)) if good else float("nan"))
                p95s.append(float(np.quantile(good, .95)) if good else float("nan"))
                rates.append(sum(not r.get("success") for r in values) / len(values))
            axes[0].plot(xs, p50s, label=f"{location}: p50")
            axes[0].plot(xs, p95s, linestyle="--", label=f"{location}: p95")
            axes[1].plot(xs, rates, marker=".", label=location)
        axes[0].set_ylabel("Successful inference HTTP latency (ms)"); axes[0].legend(fontsize=8)
        axes[1].set_ylabel("Unsuccessful inference fraction"); axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].set_ylim(bottom=0)
        axes[1].set_xlabel("Minutes since first mixed-soak inference request"); axes[1].legend(fontsize=8)
        fig.suptitle("Mixed-modality soak: 30-second bins", fontsize=14, fontweight="bold")
        self.save_figure("soak-timeline", fig, "由原始请求记录按 30 秒窗口计算，只包括混合推理，探活和锚点请求单独记录。末尾窗口可能不满 30 秒；短期曲线不能证明长期无内存泄漏或长期可用性 SLA。")

    def chart_telemetry(self):
        if not self.telemetry:
            return
        fig, axes = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True)
        fields = [("gpu_util_percent", 1, "GPU utilization (%)"), ("gpu_memory_mib", 1024, "GPU memory used (GiB)"),
                  ("worker_rss_kib", 1024 * 1024, "Worker process RSS (GiB)")]
        groups = defaultdict(list)
        for row in self.telemetry: groups[row["_path"]].append(row)
        any_data = False
        for path, rows in groups.items():
            rows = sorted([r for r in rows if finite(r.get("elapsed_s"))], key=lambda r: r["elapsed_s"])
            for ax, (field, scale, label) in zip(axes, fields):
                points = [r for r in rows if finite(r.get(field))]
                if points:
                    any_data = True
                    ax.plot([r["elapsed_s"] / 60 for r in points], [r[field] / scale for r in points], linewidth=1, label=self.rel(path))
                ax.set_ylabel(label)
                if points: ax.legend(fontsize=8)
        if not any_data:
            plt.close(fig); return
        axes[-1].set_xlabel("Minutes since telemetry sampling began")
        fig.suptitle("Read-only GPU and worker memory telemetry", fontsize=14, fontweight="bold")
        times = sorted(str(r["time_utc"]) for r in self.telemetry if r.get("time_utc"))
        time_range = f"实际 UTC 时间范围：{times[0]} 至 {times[-1]}。" if times else "源文件未提供 UTC 起止时间。"
        self.save_figure("resource-timeline", fig, "时间轴从各自遥测采集开始计算，覆盖完整遥测采集区间，未裁剪成 soak 窗口。" + time_range + "期间包含不同工作负载；首次扩大 batch 后的显存缓存台阶增长、或区间首尾差，均不能单独判为泄漏。缺失字段不补零。")

    def dimension_delta_section(self):
        rows = []
        source = "quality-dimension-deltas.json"
        for dataset in (self.deltas or {}).get("datasets", []):
            if dataset.get("status") != "complete": continue
            for direction, result in dataset.get("directions", {}).items():
                for dimension, metrics in result.get("dimensions", {}).items():
                    fields = [esc(dataset.get("dataset")), esc(direction), esc(dimension),
                              esc(fmt(result.get("query_count"))), esc(fmt(result.get("positive_source_clusters")))]
                    for metric in ["nDCG@10", "Hit@1"]:
                        value = metrics.get(metric, {})
                        ci = value.get("paired_positive_source_cluster_ci95_pp")
                        number = value.get("delta_pp")
                        shown = f"{number:+.3f}" if finite(number) else "待测"
                        shown += f'<small>[{fmt(ci[0], 3)}, {fmt(ci[1], 3)}]</small>' if isinstance(ci, list) and len(ci) == 2 else '<small>cluster CI 不可估</small>'
                        fields.append(shown)
                    if str(dimension) != "768": rows.append(fields)
                    for metric, value in metrics.items():
                        if not isinstance(value, dict) or not finite(value.get("delta_pp")): continue
                        ci = value.get("paired_positive_source_cluster_ci95_pp")
                        self.csv_rows.append({"category": "quality_dimension_delta", "dataset_or_case": dataset.get("dataset"),
                                              "direction": direction, "dimension": dimension, "queries": result.get("query_count"),
                                              "metric": metric, "value": value["delta_pp"], "unit": "percentage_points_vs_768",
                                              "ci95_low": ci[0] if ci else "", "ci95_high": ci[1] if ci else "",
                                              "ci_method": "paired_positive_source_cluster_ci95_pp", "source": source})
        body = '<h3>相对 768 维的配对质量变化</h3>'
        body += table(["数据集", "方向", "维度", "查询", "来源簇", "nDCG@10 Δ pp [cluster 95% CI]", "Hit@1 Δ pp [cluster 95% CI]"], rows)
        if (self.deltas or {}).get("errors"):
            body += '<details><summary>尚未完成的配对／严格协议分析</summary>' + render_value(self.deltas["errors"]) + '</details>'
        body += '<p class="note">Δ = 当前维度 − 768 维，单位是绝对百分点 pp。区间来自同一查询/同一正例来源簇的配对重采样，不能用两条独立置信区间相减。区间仅针对固定候选池，属于逐项 95% 区间，未进行多重比较校正；区间跨零不等于证明等价。</p>'
        return body

    def strict_video_section(self):
        supplements = (self.deltas or {}).get("video_1ka_supplements", [])
        body = '<h3>官方单 caption / 1K-A 标注的固定视频子集补充</h3>'
        if not supplements:
            return body + '<p class="pending">待测：quality-dimension-deltas.json 尚无严格官方 caption 协议的补充结果。主视频实验采用扩展 caption 协议，不能替代该项。</p>'
        rows, protocols, delta_rows = [], [], []
        for result in supplements:
            if result.get("status") != "complete":
                body += '<p class="pending">' + esc(result.get("dataset", "video")) + '：未完成，' + esc(result.get("error", "无结果")) + '</p>'
                continue
            protocols.append([esc(result.get("dataset")), esc(fmt(result.get("candidate_count"))),
                              esc(fmt(result.get("source_query_rows"))), esc(fmt(result.get("unique_exact_caption_queries"))),
                              esc(fmt(result.get("original_standard_candidate_count"))),
                              esc(result.get("query_vectors", {}).get("method")),
                              esc(result.get("caption_document_vectors", {}).get("method"))])
            for dimension, directions in result.get("dimensions", {}).items():
                for direction, value in directions.items():
                    fields = [esc(result.get("dataset")), esc(dimension), esc(direction), esc(fmt(value.get("queries"))), esc(fmt(value.get("corpus")))]
                    for metric in ["nDCG@10", "MRR@10", "Hit@1", "Hit@5", "Hit@10", "Recall@10"]:
                        number = value.get("metrics", {}).get(metric)
                        ci, kind = self.metric_ci(value, metric)
                        shown = esc(fmt(number, percent=True))
                        if ci: shown += f'<small>[{fmt(ci[0], percent=True)}, {fmt(ci[1], percent=True)}]</small>'
                        fields.append(shown)
                    rows.append(fields)
                    for metric, number in value.get("metrics", {}).items():
                        if not finite(number): continue
                        ci, kind = self.metric_ci(value, metric)
                        self.csv_rows.append({"category": "strict_video_1ka_subset", "dataset_or_case": result.get("dataset"),
                                              "direction": direction, "dimension": dimension, "queries": value.get("queries"), "corpus": value.get("corpus"),
                                              "metric": metric, "value": number, "unit": "fraction", "ci95_low": ci[0] if ci else "",
                                              "ci95_high": ci[1] if ci else "", "ci_method": kind or "", "source": "quality-dimension-deltas.json"})
            for direction, value in result.get("paired_dimension_deltas", {}).items():
                for dimension, metrics in value.get("dimensions", {}).items():
                    for metric, data in metrics.items():
                        if not finite(data.get("delta_pp")): continue
                        ci = data.get("paired_positive_source_cluster_ci95_pp")
                        self.csv_rows.append({"category": "strict_video_1ka_dimension_delta", "dataset_or_case": result.get("dataset"),
                                              "direction": direction, "dimension": dimension, "metric": metric, "value": data["delta_pp"],
                                              "unit": "percentage_points_vs_768", "ci95_low": ci[0] if ci else "", "ci95_high": ci[1] if ci else "",
                                              "ci_method": "paired_positive_source_cluster_ci95_pp", "source": "quality-dimension-deltas.json"})
                        if str(dimension) != "768" and metric in {"nDCG@10", "Hit@1"}:
                            delta_rows.append([esc(result.get("dataset")), esc(direction), esc(dimension), esc(metric),
                                               esc(f"{data['delta_pp']:+.3f}"), esc(f"[{fmt(ci[0], 3)}, {fmt(ci[1], 3)}]" if ci else "不可估")])
        body += table(["数据集", "视频候选", "官方 caption 行", "去重后查询", "原标准候选数", "query 向量来源", "document 向量来源"], protocols)
        body += table(["数据集", "维度", "方向", "查询", "候选", "nDCG@10", "MRR@10", "Hit@1", "Hit@5", "Hit@10", "严格 Recall@10"], rows)
        body += '<details><summary>官方 caption 子集：相对 768 维的配对变化</summary>' + table(["数据集", "方向", "维度", "指标", "Δ pp", "cluster 95% CI pp"], delta_rows) + '</details>'
        return body + '<p class="note">此项严格保留原官方 caption 文本和标签，在同一固定较小视频候选池上测量；不是完整标准 1K-A 榜单。重复 caption 仅按字节精确匹配合并并取其全部正例，不使用近似替换或对角线标签。反向 caption 候选独立以 document 前缀编码。</p>'

    def quality_section(self):
        rows = []
        for record in self.quality_records():
            value = record["value"]
            fields = [record["dataset"], str(record["dimension"]), record["direction"], fmt(value.get("queries")), fmt(value.get("corpus"))]
            for metric in ["nDCG@10", "MRR@10", "Hit@1", "Hit@10", "Recall@10"]:
                number = value.get("metrics", {}).get(metric)
                ci, kind = self.metric_ci(value, metric)
                shown = fmt(number, percent=True)
                if ci:
                    shown += f'<small>[{fmt(ci[0], percent=True)}, {fmt(ci[1], percent=True)}]</small>'
                fields.append(shown)
            rows.append([esc(x) if i < 5 else x for i, x in enumerate(fields)])
            for metric, number in value.get("metrics", {}).items():
                if not finite(number): continue
                ci, kind = self.metric_ci(value, metric)
                self.csv_rows.append({"category": "quality", "dataset_or_case": record["dataset"], "direction": record["direction"],
                                      "dimension": record["dimension"], "metric": metric, "value": number, "unit": "fraction",
                                      "ci95_low": ci[0] if ci else "", "ci95_high": ci[1] if ci else "", "ci_method": kind or "",
                                      "queries": value.get("queries"), "corpus": value.get("corpus"), "source": self.rel(record["summary"]["_path"])})
        baseline_rows = []
        for name, summary in self.quality.items():
            baseline = summary.get("lexical_char_tfidf_baseline")
            if not isinstance(baseline, dict): continue
            baseline_rows.append([esc(name)] + [esc(fmt(baseline.get("metrics", {}).get(metric), percent=True)) for metric in ["nDCG@10", "MRR@10", "Hit@10", "Recall@10"]])
            for metric, number in baseline.get("metrics", {}).items():
                if finite(number):
                    self.csv_rows.append({"category": "lexical_baseline", "dataset_or_case": name, "metric": metric,
                                          "value": number, "unit": "fraction", "source": self.rel(summary["_path"])})
        section = self.figure("text-quality", "文本质量摘要尚未完成。") + self.figure("multimodal-quality", "图文／真实视频质量摘要尚未完成。")
        section += table(["数据集", "维度", "方向", "查询", "候选", "nDCG@10", "MRR@10", "Hit@1", "Hit@10", "严格 Recall@10"], rows)
        section += '<p class="note">方括号为 95% 区间。Hit@K 表示至少命中一个正例的查询比例；严格 Recall@K 是命中正例数 / 该查询全部正例数后再按查询平均。常见 COCO “R@K”通常指前者，本报告不混用。</p>'
        section += '<details><summary>字符 TF-IDF 词面基线</summary><p>这是字符 2–4 gram TF-IDF 诊断基线，不是 BM25；不与其他数据集分数求总平均。</p>'
        section += table(["数据集", "nDCG@10", "MRR@10", "Hit@10", "严格 Recall@10"], baseline_rows) + '</details>'
        section += self.dimension_delta_section() + self.strict_video_section()
        return section

    def hybrid_section(self):
        intro = '<p class="note">此项是独立列出的探索性检索诊断：字符 TF-IDF 不是 BM25。公开评测集错例可以启发融合方案，但在相同公开集合上的结果不构成独立验证；固定参数的提升仍须在未用于方案设计的业务检索集上复核，也不代表已经修改线上服务。</p>'
        if self.hybrid is None:
            return intro + '<p class="pending">待完成：尚无 hybrid-results.json 的 CPU 融合诊断结果。</p>'
        methods = {"dense768": "768维稠密检索", "char_tfidf": "字符 TF-IDF（非 BM25）", "rrf": "固定参数 RRF 融合"}
        rows, differences, incomplete, harmful = [], [], [], []
        for dataset in self.hybrid.get("datasets", []):
            if not isinstance(dataset, dict): continue
            if dataset.get("status") != "complete":
                incomplete.append({"dataset": dataset.get("dataset"), "status": dataset.get("status"), "error": dataset.get("error")})
                continue
            counts = dataset.get("counts", {})
            for method, value in dataset.get("methods", {}).items():
                if not isinstance(value, dict): continue
                fields = [esc(dataset.get("dataset")), esc(methods.get(method, method)),
                          esc(fmt(value.get("queries", counts.get("queries")))), esc(fmt(value.get("corpus", counts.get("corpus"))))]
                for metric in ["nDCG@10", "MRR@10", "Hit@1", "Hit@10", "strictRecall@10"]:
                    number = value.get("metrics", {}).get(metric)
                    ci, kind = self.metric_ci(value, metric)
                    shown = esc(fmt(number, percent=True))
                    if ci: shown += f'<small>[{fmt(ci[0], percent=True)}, {fmt(ci[1], percent=True)}]</small>'
                    fields.append(shown)
                rows.append(fields)
                for metric, number in value.get("metrics", {}).items():
                    if not finite(number): continue
                    ci, kind = self.metric_ci(value, metric)
                    self.csv_rows.append({"category": "hybrid_exploratory", "dataset_or_case": dataset.get("dataset"),
                                          "direction": method, "dimension": self.hybrid.get("dimension"),
                                          "queries": value.get("queries", counts.get("queries")), "corpus": value.get("corpus", counts.get("corpus")),
                                          "metric": metric, "value": number, "unit": "fraction", "ci95_low": ci[0] if ci else "",
                                          "ci95_high": ci[1] if ci else "", "ci_method": kind or "", "source": "hybrid-results.json"})
            for comparison, metrics in dataset.get("paired_deltas", {}).items():
                for metric, value in metrics.items():
                    if not isinstance(value, dict) or not finite(value.get("delta_pp")): continue
                    ci = value.get("paired_positive_source_cluster_ci95_pp")
                    if (comparison == "rrf_minus_dense768" and metric == "nDCG@10" and value["delta_pp"] < 0
                            and isinstance(ci, list) and len(ci) == 2 and finite(ci[1]) and ci[1] < 0):
                        harmful.append(f"{dataset.get('dataset')}：nDCG@10 变化 {value['delta_pp']:+.3f} pp，来源簇配对 95% 区间 [{ci[0]:.3f}, {ci[1]:.3f}] pp。")
                    self.csv_rows.append({"category": "hybrid_exploratory_paired_delta", "dataset_or_case": dataset.get("dataset"),
                                          "direction": comparison, "dimension": self.hybrid.get("dimension"), "metric": metric,
                                          "value": value["delta_pp"], "unit": "percentage_points",
                                          "ci95_low": ci[0] if ci else "", "ci95_high": ci[1] if ci else "",
                                          "ci_method": "paired_positive_source_cluster_ci95_pp", "source": "hybrid-results.json"})
                    if metric in {"nDCG@10", "Hit@1", "strictRecall@10"}:
                        differences.append([esc(dataset.get("dataset")), esc(comparison), esc(metric),
                                            esc(f"{value['delta_pp']:+.3f}"),
                                            esc(f"[{fmt(ci[0], 3)}, {fmt(ci[1], 3)}]" if ci else "cluster CI 不可估")])
        body = intro + '<h3>固定参数与实验状态</h3>'
        if harmful:
            body += '<aside class="warning"><strong>固定参数 RRF 并非默认改进</strong><p>本次以下数据集相对稠密检索的 nDCG@10 下降，且源文件中的逐项配对区间整体为负：</p>' + render_value(harmful) + '<p>这仅是当前固定候选池的探索性证据，不能直接推广为所有业务场景或独立验证结论。</p></aside>'
        body += render_value({"status": self.hybrid.get("status"), "dimension": self.hybrid.get("dimension"),
                              "parameters": self.hybrid.get("parameters", {})})
        body += table(["数据集", "方法", "查询", "候选", "nDCG@10", "MRR@10", "Hit@1", "Hit@10", "严格 Recall@10"], rows)
        body += '<details><summary>RRF 相对两条基线的配对变化（探索性）</summary>'
        body += table(["数据集", "比较", "指标", "Δ pp", "cluster 95% CI pp"], differences) + '</details>'
        if incomplete or self.hybrid.get("errors"):
            body += '<details><summary>未完成项与源错误记录</summary>' + render_value(incomplete or self.hybrid["errors"]) + '</details>'
        return body + '<p class="note">严格 Recall@10 的分母是查询的全部已标注正例；Hit@10 是至少命中一个正例。各数据集分别报告，不计算跨数据集总平均。百分点评估采用源文件中的配对区间，不能把两条独立区间相减。</p>'

    def performance_section(self):
        rows = []
        for case in self.cases:
            latency = case.get("success_http_latency_ms", {})
            valid = bool(case.get("sample_valid")) and not case.get("interrupted") and not case.get("drain_timeout")
            role = self.text_role(case)
            rows.append([esc(case["_location"]), esc(case.get("phase")), esc(case.get("case")), esc(self.workload(case)), esc(role or "不适用／未标注"), esc(fmt(case.get("concurrency"))),
                         esc(fmt(case.get("batch"))), esc(fmt(case.get("text_chars"))), esc(fmt(case.get("success_count"))),
                         esc(fmt(case.get("success_rps"))), esc(fmt(case.get("success_items_per_second"))),
                         esc(fmt(latency.get("p50"))), esc(fmt(latency.get("p95"))), self.tail_value(latency),
                         esc(fmt(case.get("http_429_rate"), percent=True)), esc("有效" if valid else "不足／负例／中断"),
                         esc(json.dumps(case.get("http_status_counts", {}), ensure_ascii=False))])
            flattened = {"success_rps": (case.get("success_rps"), "requests/s"),
                         "success_items_per_second": (case.get("success_items_per_second"), "items/s"),
                         "http_429_rate": (case.get("http_429_rate"), "fraction"), "success_count": (case.get("success_count"), "requests"),
                         "elapsed_s": (case.get("elapsed_s"), "seconds"),
                         "offered_requests": (case.get("offered_requests"), "requests"),
                         "http_attempts": (case.get("http_attempts"), "requests"),
                         "http_429_count": (case.get("http_429_count"), "requests"),
                         **{f"http_latency_{p}": (latency.get(p), "ms") for p in ["p50", "p95", "p99", "max"]},
                         **{f"end_to_end_{p}": (case.get("success_end_to_end_ms", {}).get(p), "ms") for p in ["p50", "p95", "p99"]},
                         **{f"client_queue_{p}": (case.get("success_client_queue_ms", {}).get(p), "ms") for p in ["p50", "p95", "p99"]}}
            for metric, (number, unit) in flattened.items():
                if finite(number):
                    count = case.get("success_count")
                    if metric.startswith("end_to_end_"): count = case.get("success_end_to_end_ms", {}).get("count", count)
                    elif metric.startswith("client_queue_"): count = case.get("success_client_queue_ms", {}).get("count", count)
                    elif metric.startswith("http_latency_"): count = latency.get("count", count)
                    exploratory = metric.endswith("p99") and (not finite(count) or count < 1000)
                    self.csv_rows.append({"category": "performance", "dataset_or_case": case.get("case"), "phase": case.get("phase"),
                                          "location": case["_location"], "concurrency": case.get("concurrency"), "batch": case.get("batch"),
                                          "target_rps": case.get("target_rps"), "arrival_window_s": case.get("arrival_window_s"),
                                          "workload": self.workload(case), "input_type": role or "", "sample_count": count,
                                          "interpretation": "exploratory_p99" if exploratory else "",
                                          "text_chars": case.get("text_chars"), "run_id": case.get("run_id"), "case_run_id": case.get("case_run_id"),
                                          "sample_valid": valid, "metric": metric, "value": number, "unit": unit, "source": self.rel(case["_path"])})
        body = self.figure("public-batch", "公网 batch / 文本长度测试尚未完成。")
        body += self.figure("load-concurrency", "原始并发负载测试尚未完成。")
        body += '<p class="note">C 是同时在途 HTTP 请求数，batch 是单次请求内条目数，两者必须分别配置。成功 RPS 与成功条目/秒分别报告；快速返回的 429 不构成有效容量。拒绝率分母为实际发起的 HTTP 尝试，包含传输失败，尚未发出的客户端队列丢弃不计入该分母。公网延迟包含网络、代理和服务处理，服务器本机结果不能直接当作公网体验。</p>'
        body += self.arrival_section()
        body += '<details><summary>全部性能用例（含负例、探活及样本有效性）</summary>'
        body += table(["位置", "阶段", "用例", "工作负载", "文字角色", "C", "batch", "字符数", "成功数", "成功 RPS", "条目/s", "p50 ms", "p95 ms", "p99 ms", "429 比率", "样本", "HTTP 状态"], rows) + '</details>'
        body += '<p class="note">真实 JPEG（按用例名记录实际尺寸）与合成 256×256 PNG 分开标注，编码、分辨率、图像内容和视频帧数均会影响延迟。样本数少于 1,000 的所有 p99 都标为探索性；短窗口 p95 / p99 只描述当前有限样本。soak 中按类型累计的较长窗口结果可作为更稳的参考，仍不构成长期尾延迟保证。</p>'
        sanity = []
        seen = set()
        for summary in self.performance:
            for check in summary.get("sanity_checks", []):
                key = check.get("run_id"), check.get("case")
                if key in seen: continue
                seen.add(key)
                sanity.append([esc(self.location(summary["_path"])), esc(check.get("case")),
                               esc(fmt(check.get("expected_status"))), esc(fmt(check.get("actual_status"))),
                               esc("通过" if check.get("passed") else "未通过")])
                self.csv_rows.append({"category": "input_validation", "dataset_or_case": check.get("case"),
                                      "location": self.location(summary["_path"]), "run_id": check.get("run_id"),
                                      "metric": "expected_behavior_passed", "value": int(bool(check.get("passed"))),
                                      "unit": "boolean_0_or_1", "source": self.rel(summary["_path"])})
        body += '<details><summary>鉴权及非法输入的预期响应</summary>' + table(["位置", "用例", "预期 HTTP 状态", "实际 HTTP 状态", "结论"], sanity) + '</details>'
        return body

    def stability_window_section(self):
        body = '<h3>实际 soak 窗口的专项分析</h3>'
        if self.stability_analysis is None:
            return body + '<p class="pending">待完成：尚无 stability-analysis.json 中的实际 soak 窗口资源趋势与稳定性分析。全实验遥测曲线不能代替这一窗口专项判断。</p>'
        analysis = self.stability_analysis
        window = analysis.get("window", {})
        body += '<p class="note">以下仅使用源文件明确标注的实际 soak 窗口；全实验遥测另列。首次扩大 batch 后的 PyTorch 显存缓存台阶不能单独视为泄漏。</p>'
        body += table(["分析状态", "开始 UTC", "结束 UTC", "实际单调时长 s", "墙上时钟时长 s", "成功混合推理 RPS"],
                      [[esc(analysis.get("status")), esc(window.get("start_utc", "待测")), esc(window.get("end_utc", "待测")),
                        esc(fmt(window.get("monotonic_elapsed_seconds"))), esc(fmt(window.get("wall_elapsed_seconds"))),
                        esc(fmt(analysis.get("validated_mixed_request_goodput_per_second"))) ]])
        request_rows = []
        request_groups = [("混合推理合计", analysis.get("mixed_load", {})), *list(analysis.get("by_case", {}).items()),
                          ("健康探活", analysis.get("health_probes", {})), ("固定锚点推理", analysis.get("anchors", {}))]
        for name, stats in request_groups:
            latency = stats.get("success_latency_ms", {})
            attempts, successes = stats.get("attempts"), stats.get("validated_successes")
            failures = attempts - successes if finite(attempts) and finite(successes) else None
            request_rows.append([esc(name), esc(fmt(attempts)), esc(fmt(successes)), esc(fmt(failures)),
                                 esc(fmt(latency.get("p50"))), esc(fmt(latency.get("p95"))), self.tail_value(latency),
                                 esc(json.dumps(stats.get("status_counts", {}), ensure_ascii=False)),
                                 esc(json.dumps(stats.get("error_counts", {}), ensure_ascii=False))])
        body += table(["请求类别", "实际尝试", "校验成功", "未成功", "p50 ms", "p95 ms", "p99 ms", "HTTP 状态", "错误计数"], request_rows)
        body += table(["观察到的 boot_id 数", "观察到 boot 变化事件", "前后记录满足同一 boot 验证", "结束后 ready", "锚点最低余弦"],
                      [[esc(fmt(len(analysis.get("boot_ids", [])))), esc(fmt(analysis.get("observed_boot_change_events"))),
                        esc(fmt(analysis.get("before_after_same_observed_boot"))), esc(fmt(analysis.get("after_ready"))),
                        esc(fmt(analysis.get("anchor_cosine_vs_first_successful_reference", {}).get("min"), 8))]])
        resources = analysis.get("soak_window_resources", {})
        labels = {"gpu_util_percent": ("GPU 利用率", "%", 1), "gpu_memory_mib": ("GPU 显存", "MiB", 1),
                  "gpu_temperature_c": ("GPU 温度", "°C", 1), "gpu_power_w": ("GPU 功耗", "W", 1),
                  "worker_rss_kib": ("worker RSS", "MiB", 1024)}
        resource_rows = []
        for field, metric in resources.get("metrics", {}).items():
            name, unit, divisor = labels.get(field, (field, "源单位", 1))
            normalized = lambda key: metric.get(key) / divisor if finite(metric.get(key)) else None
            resource_rows.append([esc(name), esc(unit), esc(fmt(metric.get("n"))),
                                  esc(fmt(normalized("first_five_minutes_after_60s_median"))),
                                  esc(fmt(normalized("last_five_minutes_median"))),
                                  esc(fmt(normalized("late_minus_early_median"), 4)),
                                  esc(fmt(normalized("ols_slope_per_minute_of_minute_medians"), 4))])
        body += '<h3>soak 窗口内的资源变化</h3>'
        body += table(["资源", "单位", "有效样本", "热身后首五分钟中位数", "末五分钟中位数", "后段 − 前段", "热身后分钟中位数斜率 / 分钟"], resource_rows)
        body += '<p class="note">前段统计跳过最初 60 秒后取五分钟，后段取结束前五分钟。斜率对热身后的整分钟中位数做描述性拟合，不能单独判定泄漏，更不能把整场质量编码和扩 batch 的首尾缓存差纳入这项判断。</p>'
        window_rows = []
        for segment in analysis.get("five_minute_latency_windows", []):
            for case, stats in segment.get("by_case", {}).items():
                latency = stats.get("success_latency_ms", {})
                window_rows.append([esc(f"{fmt(segment.get('start_minute'))}–{fmt(segment.get('end_minute'))}"),
                                    esc(case), esc(fmt(stats.get("validated_successes"))), esc(fmt(latency.get("p50"))),
                                    esc(fmt(latency.get("p95"))), self.tail_value(latency)])
        body += '<details><summary>每五分钟、按工作负载的尾延迟</summary>'
        body += table(["起止分钟", "用例", "成功样本", "p50 ms", "p95 ms", "p99 ms"], window_rows) + '</details>'
        body += '<p class="note">适用边界：单次约 30 分钟、重复固定素材；媒体性能反复使用同一张真实图和同一视频的有序帧。未做独立重复实验、长周期 SLA 验证或崩溃、断网、OOM、重启等故障注入。素材多样性的检索质量由另一组公开评测单独衡量；这里不包含客户端素材准备、MP4 抽帧、对象存储、ANN 索引和应用链路。</p>'
        body += '<details><summary>专项分析全部源字段与边界说明</summary>' + render_value(analysis) + '</details>'
        def scalars(value, path="", sample_count=None):
            if isinstance(value, dict):
                sample_count = value.get("count", value.get("n", sample_count))
                for key, child in value.items():
                    if not SENSITIVE.search(str(key)):
                        yield from scalars(child, f"{path}.{key}" if path else str(key), sample_count)
            elif isinstance(value, list):
                for index, child in enumerate(value): yield from scalars(child, f"{path}[{index}]", sample_count)
            elif finite(value):
                yield path, value, sample_count
        for metric, number, count in scalars(analysis):
            exploratory = metric.endswith(".p99") and (not finite(count) or count < 1000)
            self.csv_rows.append({"category": "soak_window_analysis", "dataset_or_case": "actual_soak_window",
                                  "metric": metric, "value": number, "unit": "as_recorded_in_source",
                                  "sample_count": count if count is not None else "",
                                  "interpretation": "exploratory_p99" if exploratory else "",
                                  "source": "stability-analysis.json"})
        return body

    def stability_section(self):
        rows = []
        for summary in self.performance:
            if not any(str(c.get("phase", "")).startswith("soak") for c in summary.get("cases", [])):
                continue
            cases = [c for c in summary.get("cases", []) if c.get("phase") == "soak"]
            anchors = [c.get("anchor_min_cosine") for c in summary.get("cases", []) if finite(c.get("anchor_min_cosine"))]
            rows.append([self.link(summary["_path"]), esc(fmt(sum(c.get("success_count", 0) for c in cases))),
                         esc(fmt(sum(c.get("offered_requests", 0) for c in cases))), esc(fmt(summary.get("boot_id_changed"))),
                         esc(fmt(min(anchors) if anchors else None, digits=6)), esc(fmt(summary.get("invalid_jsonl_lines")))])
        body = self.stability_window_section()
        body += self.figure("soak-timeline", "混合负载 soak 原始记录尚未完成。")
        body += table(["soak 摘要", "成功推理请求", "发起推理请求", "观察到 boot_id 变化", "锚点最低余弦", "无法解析记录"], rows)
        by_type = []
        for case in self.cases:
            if case.get("phase") != "soak": continue
            latency = case.get("success_http_latency_ms", {})
            by_type.append([esc(case["_location"]), esc(case.get("case")), esc(self.workload(case)),
                            esc(fmt(latency.get("count", case.get("success_count")))), esc(fmt(case.get("elapsed_s"))),
                            esc(fmt(latency.get("p50"))), esc(fmt(latency.get("p95"))), self.tail_value(latency),
                            esc(fmt(case.get("http_429_rate"), percent=True))])
        body += '<h3>soak 按工作负载累计的长窗口延迟</h3>'
        body += table(["位置", "用例", "工作负载", "成功样本", "统计窗口 s", "p50 ms", "p95 ms", "p99 ms", "429 比率"], by_type)
        body += self.figure("resource-timeline", "GPU / worker RSS 遥测尚未提供。")
        if self.telemetry:
            numbers = lambda field: [r[field] for r in self.telemetry if finite(r.get(field))]
            gpu, rss = numbers("gpu_memory_mib"), numbers("worker_rss_kib")
            pids = sorted({r["worker_pid"] for r in self.telemetry if r.get("worker_pid") is not None})
            body += table(["遥测记录", "GPU 显存峰值 MiB", "worker RSS 峰值 MiB", "观察到的 worker PID 数", "采集错误记录"],
                          [[esc(fmt(len(self.telemetry))), esc(fmt(max(gpu) if gpu else None)),
                            esc(fmt(max(rss) / 1024 if rss else None)), esc(fmt(len(pids))),
                            esc(fmt(sum(bool(r.get("error_class")) for r in self.telemetry))) ]])
            for path in sorted({r["_path"] for r in self.telemetry}):
                for field, unit in [("gpu_util_percent", "percent"), ("gpu_memory_mib", "MiB"),
                                    ("gpu_temperature_c", "Celsius"), ("gpu_power_w", "watts"), ("worker_rss_kib", "KiB")]:
                    values = [r[field] for r in self.telemetry if r["_path"] == path and finite(r.get(field))]
                    if not values: continue
                    for stat, value in [("min", min(values)), ("max", max(values)), ("first", values[0]), ("last", values[-1])]:
                        self.csv_rows.append({"category": "telemetry", "dataset_or_case": field, "metric": stat,
                                              "value": value, "unit": unit, "source": self.rel(path)})
        return body + '<p class="note">稳定性结论仅限实际记录的测试时长与负载；探活成功不等于每次模型推理成功，未观察到重启也不构成长期 SLA。</p>'

    def coverage_section(self):
        rows = []
        for name, manifest in self.manifests.items():
            counts = manifest.get("counts", {})
            if not counts and name == "video":
                counts = {"corpus": manifest.get("candidate_count"), "queries": manifest.get("query_count"), "qrels": manifest.get("qrel_count")}
            if name in self.quality:
                first = next(iter(self.quality[name].get("dimensions", {}).values()), {})
                counts = {**counts, "corpus": first.get("corpus", counts.get("corpus")), "queries": first.get("queries", counts.get("queries"))}
            source = manifest.get("sources", [{}])[0] if isinstance(manifest.get("sources"), list) else {}
            revision = manifest.get("revision") or source.get("revision")
            license_note = manifest.get("license") or manifest.get("license_note") or "源文件未声明"
            rows.append([esc(LABELS.get(name, name)), esc(fmt(counts.get("corpus"))), esc(fmt(counts.get("queries"))),
                         esc(manifest.get("split", "源文件未标注")), esc("已评分" if name in self.quality else "待测／可选诊断"),
                         esc(revision or "见 manifest"), esc(license_note)])
        return table(["评测集", "候选", "查询", "split", "进度", "revision", "数据许可"], rows)

    def hardware_section(self):
        body = '<h3>硬件、容器配额与推理配置</h3>'
        if self.hardware is None:
            return body + '<p class="pending">待提供：尚无 hardware.json 中的硬件与运行约束记录。</p>'
        value = self.hardware
        memory_gib = value.get("server_cgroup_memory_bytes")
        memory_gib = memory_gib / (1024 ** 3) if finite(memory_gib) else None
        rows = [
            ["GPU", value.get("gpu"), "所用 GPU 型号"],
            ["GPU 显存", fmt(value.get("gpu_memory_mib")) + " MiB", "GPU 可见显存记录"],
            ["NVIDIA 驱动", value.get("driver"), "服务端驱动版本"],
            ["PyTorch / CUDA", f"{value.get('torch', '待测')} / {value.get('cuda', '待测')}", "实际软件环境"],
            ["容器 CPU 配额", fmt(value.get("server_cpu_quota")) + " vCPU", "采用容器 CPU quota"],
            ["容器内存上限", fmt(memory_gib) + " GiB", "采用 cgroup memory limit"],
            ["计算精度", value.get("dtype"), "BF16 配置以源记录为准"],
            ["HTTP worker 数", fmt(value.get("api_worker_count")), "服务进程数"],
            ["推理锁容量", fmt(value.get("inference_lock_capacity")), "同一时刻可进入模型执行的请求数"],
            ["HTTP 文本 batch 上限", fmt(value.get("max_text_batch")), "单次 API 请求的条目上限"],
            ["编码器内部 micro batch", fmt(value.get("encoder_micro_batch")), "模型内部批处理，与 HTTP 并发和 HTTP batch 分开"],
            ["GPU 小时费用", fmt(value.get("gpu_hourly_cny")) + " 元/小时", "源记录的 GPU 计费价格"],
            ["服务端内核", value.get("server_kernel"), "实际系统版本"],
            ["公网测试客户端", f"{value.get('client_system', '待测')} / {value.get('client_arch', '待测')} / Python {value.get('client_python', '待测')}", "客户端运行环境"],
            ["模型", value.get("model"), value.get("revision")],
        ]
        body += table(["项目", "记录配置", "口径"], [[esc(x) for x in row] for row in rows])
        body += '<p class="note">CPU 与内存采用容器 cgroup 配额。宿主机可见的 affinity 或总内存属于共享宿主机信息，不能当作本实例的独占资源。配置来源：' + self.link(self.root / "hardware.json") + '。</p>'
        units = {"gpu_memory_mib": "MiB", "server_cgroup_memory_bytes": "bytes", "server_cpu_quota": "vCPU",
                 "gpu_hourly_cny": "CNY/hour", "api_worker_count": "workers", "inference_lock_capacity": "requests",
                 "max_text_batch": "items/request", "encoder_micro_batch": "items/microbatch"}
        for key, unit in units.items():
            if finite(value.get(key)):
                self.csv_rows.append({"category": "hardware_environment", "dataset_or_case": "deployment_environment",
                                      "metric": key, "value": value[key], "unit": unit, "source": "hardware.json"})
        return body

    @staticmethod
    def timing_protocol_section():
        return '''<h3>计时边界与热身条件</h3><div class="protocol"><p>HTTP 延迟（latency_ms）从客户端开始尝试请求计时，到响应 body 完整接收并退出 HTTP 读取上下文结束。之后本机 JSON 解析、向量形状／有限值／范数校验和锚点比对的耗时不在该 HTTP 延迟内；“成功”资格仍要求通过这些校验。</p>
<p>固定到达率实验的 scheduled E2E 是“客户端等待计划到达后的排队时间 + 上述 HTTP 时间”，终点同样是响应 body 接收完。它不是完整业务端到端耗时，不包含上游业务、素材准备、MP4 抽帧、对象存储、ANN 检索、RAG 或应用处理。成功吞吐的分母则使用完整测量窗口及收尾时间。</p>
<p>本轮测试针对已加载并热身的服务，未测试实例启动或模型加载的 cold start。soak 是否覆盖媒体以实际请求 case 和分类型结果为准；通用 run_start.include_media CLI 标志不用于推断该模式的媒体覆盖，soak 的固定轮换逻辑会执行其实际列出的图像与视频用例。</p></div>'''

    def sources_section(self):
        rows = []
        # Sort the immutable provenance table last, after discovery and plotting.
        for path in sorted(self.inputs):
            if not path.is_file(): continue
            raw = path.read_bytes()
            rows.append([self.link(path), esc(fmt(len(raw))), f'<code>{hashlib.sha256(raw).hexdigest()}</code>'])
        body = ''
        if self.supporting_files:
            body += '<h3>复现、客户端建议与独立复核材料</h3><ul>'
            body += "".join('<li>' + self.link(path) + '</li>' for path in self.supporting_files) + '</ul>'
        body += '<details><summary>本地原始文件索引与 SHA256</summary><p>链接相对于报告目录；分享单个 HTML 时图表仍可显示，原始数据链接需要一同保留目录。报告无需外部脚本、样式或图片。</p>'
        body += table(["文件", "字节", "SHA256"], rows) + '</details>'
        links = [("模型官方说明", "https://huggingface.co/google/embeddinggemma-2"),
                 ("MTEB 官方任务定义", "https://docs.mteb.org/overview/available_tasks/retrieval/"),
                 ("BEIR SciFact", "https://huggingface.co/datasets/BeIR/scifact"),
                 ("Belebele", "https://huggingface.co/datasets/mteb/belebele"),
                 ("C-MTEB VideoRetrieval", "https://huggingface.co/datasets/C-MTEB/VideoRetrieval"),
                 ("MTEB / M-BEIR COCO5K", "https://huggingface.co/datasets/mteb/mbeir_mscoco_task0"),
                 ("MSR-VTT 镜像", "https://huggingface.co/datasets/VLM2Vec/MSR-VTT"),
                 ("COCO 使用条款", "https://cocodataset.org/#termsofuse")]
        body += '<p class="external">' + " · ".join(f'<a href="{esc(url)}">{esc(label)}</a>' for label, url in links) + '</p>'
        return body

    def build(self):
        self.discover()
        self.chart_quality(); self.chart_latency(); self.chart_load(); self.chart_arrivals(); self.chart_soak(); self.chart_telemetry()
        reviewed = self.reviewed or {}
        status = "评测完成 · 结论已复核" if self.reviewed else "评测进行中 · 结论待核准"
        headline = reviewed.get("headline", "正在形成可核查的质量、性能与并发结论")
        concurrency = reviewed.get("concurrency_recommendation") or reviewed.get("recommended_concurrency")
        recommendation = render_value(concurrency) if concurrency is not None else '<p class="pending">待核准：尚无 reviewed-analysis.json 中的并发建议。请先完成公网 / 本机负载和稳定性数据，再填写复核结论。</p>'
        # No automatic numeric capacity recommendation is synthesized from partial data.
        top = f'<div class="status">{esc(status)}</div><h1>EmbeddingGemma 2<br>推理服务技术评测</h1><p class="lead">{esc(headline)}</p>'
        top += '<div class="cards">' + "".join(f'<div class="card"><strong>{esc(value)}</strong><span>{esc(label)}</span></div>' for label, value in [
            ("已完成质量评分的数据集", len(self.quality)), ("已记录性能用例", len(self.cases)),
            ("只读资源遥测记录", len(self.telemetry))]) + '</div>'
        top += '<div class="decision"><h2>并发与批处理建议</h2>' + recommendation + '</div>'
        if self.reviewed:
            remaining = {k: v for k, v in reviewed.items() if k not in {"headline", "concurrency_recommendation", "recommended_concurrency"}}
            top += render_value(remaining)
        else:
            top += '<p class="note">当前报告只展示已经产生的真实结果。未完成项保持“待测”，不会引用旧硬件报告或推测缺失数字。</p>'
        quality = self.quality_section()
        hybrid = self.hybrid_section()
        performance = self.performance_section()
        stability = self.stability_section()
        protocol = self.hardware_section() + self.timing_protocol_section() + self.coverage_section()
        protocol += '''<div class="protocol"><p>质量评测使用固定模型 revision 与服务实际 query/document 前缀，进行精确余弦排序。768 维向量截取前 512 / 256 / 128 维后重新 L2 归一化；同一维度的查询与候选相匹配。图像与视频不加文字前缀。反向媒体→文字的 caption 候选另走 document 编码，不复用正向 query 向量。</p>
<p>COCO5K 是固定 revision 的全量 5,000 图及全部 caption/qrels，保留多正例。中文 20K 是完整正例保留的固定候选子集；MSR-VTT 是真实视频固定 128 候选子集，按原 test_1k 分割取样，不代表完整 1K-A 榜单；Belebele 只有 488 个候选，属于语言诊断。视频标题文本检索与真实视频帧检索分别报告。</p>
<p>公开测试集可能出现在模型训练材料中，无法据此证明训练污染已排除。公开基准是回归与相对比较证据，仍需补充真实业务检索、长尾失败查询和困难负例。已标注的数据许可不能替代原图片或视频权利；未声明的许可明确保留为未知。</p>
<p>性能以真正完成并通过向量形状、有限值和范数校验的响应为成功；429、传输错误、超时另列。原始负载测试不重试，包含完整到达窗口和收尾时间。负例测试的预期 4xx 是输入校验结果，不能混入正常流量故障率。HTTP p95 是成功响应条件下的分位数，须结合拒绝率和成功 RPS 解读。</p></div>'''
        runtime_rows = []
        for name, values in self.service.items():
            runtime_rows.append([esc(name)] + [esc(fmt(values.get(k))) for k in ["model", "model_revision", "device", "dtype", "boot_id", "ready"]])
        protocol += table(["健康记录", "模型", "revision", "设备", "精度", "boot_id", "ready"], runtime_rows)
        warning_block = '<aside class="warning"><strong>输入检查提示</strong>' + render_value(self.warnings) + '</aside>' if self.warnings else ''
        sources = self.sources_section()
        css = '''*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#182638;font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1420px;margin:36px auto;padding:44px 48px;background:white;border:1px solid #dce3ea;border-radius:12px}h1{font-size:42px;line-height:1.15;letter-spacing:-1px;margin:18px 0}h2{font-size:25px;margin:38px 0 16px;line-height:1.3}h3{font-size:18px}p{margin:10px 0}.lead{font-size:21px;max-width:1100px;color:#3b536d}.status{display:inline-block;background:#e9f3fa;color:#1768ac;padding:5px 12px;border-radius:20px;font-weight:650;font-size:13px}.meta{font-size:12px;color:#64768a}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:25px 0}.card{padding:20px;background:#f6f9fc;border:1px solid #dfe8ef;border-radius:9px}.card strong{font-size:31px;display:block;line-height:1.25}.card span{color:#5a7188;font-size:13px}.decision{padding:18px 24px;background:#eef7f4;border-left:4px solid #2b8c65;border-radius:5px}.decision h2{margin:0 0 10px;font-size:21px}.note,.pending{padding:13px 16px;background:#f5f8fc;border-radius:5px;color:#52677b}.pending{border-left:3px solid #b27b20;background:#fff8ea}.warning{background:#fff4e5;padding:16px}.table-scroll{overflow-x:auto;margin:16px 0;border:1px solid #dfe5ec;border-radius:5px}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.5}th{background:#edf3f8;color:#2d475f;text-align:left;font-weight:650;white-space:nowrap}td,th{padding:10px 11px;border-bottom:1px solid #e4eaf0;vertical-align:top}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafcfe}td small{display:block;font-size:10px;color:#677b90;white-space:nowrap}td code{font-size:10px;word-break:break-all}a{color:#1768ac;text-decoration:none}a:hover{text-decoration:underline}details{margin:18px 0;padding:12px 16px;border:1px solid #dce5ef;border-radius:6px}summary{font-weight:650;cursor:pointer}figure{margin:25px 0}figure img{width:100%;height:auto;display:block}figcaption{color:#64768a;font-size:12px;margin-top:8px}.analysis dt{font-weight:650;color:#314f68;margin-top:12px}.analysis dd{margin-left:0}.analysis ul{padding-left:22px}.protocol{max-width:1160px}.external{font-size:13px}.toc{display:flex;flex-wrap:wrap;gap:17px;padding:14px 0;border-top:1px solid #dfe7ef;border-bottom:1px solid #dfe7ef;margin:28px 0}.footer{padding-top:18px;border-top:1px solid #dde5ed;margin-top:32px;font-size:12px;color:#61768b}@media(max-width:800px){main{margin:0;padding:25px 18px;border:0;border-radius:0}h1{font-size:33px}.lead{font-size:18px}.cards{gap:8px}.card{padding:12px}.card strong{font-size:25px}}@media print{body{background:white}main{padding:0;margin:0;border:0;max-width:none}details{break-inside:avoid}figure{break-inside:avoid}.toc{display:none}a{color:inherit}}'''
        generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EmbeddingGemma 2 推理服务技术评测</title><style>{css}</style></head><body><main>{top}
<nav class="toc"><a href="#quality">检索质量</a><a href="#hybrid">探索性融合诊断</a><a href="#performance">性能与并发</a><a href="#stability">稳定性与资源</a><a href="#protocol">实验协议与边界</a><a href="#sources">来源与原始文件</a></nav>
{warning_block}<section id="quality"><h2>检索质量与维度取舍</h2>{quality}</section>
<section id="hybrid"><h2>探索性：词面与稠密检索融合</h2>{hybrid}</section>
<section id="performance"><h2>公网性能与原始并发负载</h2>{performance}</section>
<section id="stability"><h2>稳定性与资源使用</h2>{stability}</section>
<section id="protocol"><h2>实验协议、数据覆盖与适用边界</h2>{protocol}</section>
<section id="sources"><h2>来源与可复核材料</h2><p><a href="results.csv">下载全部数字 results.csv</a>；charts 目录保留可独立使用的 PNG 与 SVG 科学图表。</p>{sources}</section>
<p class="footer">报告生成时间（UTC）：{esc(generated)}。内容来自本目录的实际评测文件；HTML 内嵌全部图表，离线打开无需联网。</p></main></body></html>'''
        (self.root / "REPORT.html").write_text(document, encoding="utf-8")
        fieldnames = ["category", "dataset_or_case", "direction", "dimension", "location", "phase", "workload", "input_type", "concurrency", "batch", "text_chars",
                      "run_id", "case_run_id", "sample_valid", "queries", "corpus", "metric", "value", "unit", "ci95_low", "ci95_high", "ci_method", "source"]
        fieldnames += ["sample_count", "interpretation", "target_rps", "arrival_window_s"]
        with (self.root / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(self.csv_rows)
        return {"report": str(self.root / "REPORT.html"), "csv_rows": len(self.csv_rows), "quality_datasets": len(self.quality),
                "performance_cases": len(self.cases), "figures": sorted(self.figures), "warnings": self.warnings,
                "reviewed_analysis_present": bool(self.reviewed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("--root must be an existing evaluation directory")
    print(json.dumps(Report(args.root).build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
