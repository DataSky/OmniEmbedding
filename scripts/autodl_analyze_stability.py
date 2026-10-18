#!/usr/bin/env python3
"""CPU-only analysis of completed mixed-soak records and matching telemetry.

No service calls or credentials are used. Whole-experiment telemetry is kept
separate from the completed soak window. Memory trends are descriptive, not a
leak detector or a long-term availability guarantee.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def distribution(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0}
    assert np.isfinite(array).all()
    return {"n": len(array), "min": float(array.min()),
            "p50": float(np.quantile(array, .5)),
            "p95": float(np.quantile(array, .95)),
            "p99": float(np.quantile(array, .99)),
            "max": float(array.max())}


def request_stats(records):
    successful = [row for row in records if row["success"]]
    latencies = [row["latency_ms"] for row in successful]
    return {"attempts": len(records), "validated_successes": len(successful),
            "status_counts": dict(Counter(str(row.get("status")) for row in records)),
            "error_counts": dict(Counter(row["errorclass"] for row in records if row.get("errorclass"))),
            "success_latency_ms": distribution(latencies),
            "p99_note": ("Empirical p99 for this single run; independent repeated runs were not performed."
                         if len(latencies) >= 1000 else "Exploratory p99: fewer than 1000 successful samples.")}


def resource_analysis(rows, start, end):
    output = {"samples": len(rows), "window_seconds": end - start,
              "telemetry_errors": dict(Counter(row["error_class"] for row in rows if row.get("error_class"))),
              "worker_pids": sorted({row["worker_pid"] for row in rows if row.get("worker_pid")}),
              "metrics": {}}
    fields = ["gpu_util_percent", "gpu_memory_mib", "gpu_temperature_c", "gpu_power_w", "worker_rss_kib"]
    for field in fields:
        valid = [(timestamp(row["time_utc"]), row[field]) for row in rows
                 if isinstance(row.get(field), (int, float)) and np.isfinite(row[field])]
        if not valid:
            continue
        times, values = map(np.asarray, zip(*valid))
        metric = distribution(values)
        metric["first_five_minutes_after_60s_median"] = (
            float(np.median(values[(times >= start + 60) & (times < start + 360)]))
            if np.any((times >= start + 60) & (times < start + 360)) else None)
        metric["last_five_minutes_median"] = (
            float(np.median(values[times >= end - 300]))
            if np.any(times >= end - 300) else None)
        first = metric["first_five_minutes_after_60s_median"]
        last = metric["last_five_minutes_median"]
        metric["late_minus_early_median"] = last - first if first is not None and last is not None else None
        buckets = defaultdict(list)
        for point, value in valid:
            if point >= start + 60:
                buckets[int((point - start) // 60)].append(value)
        points = [(minute, float(np.median(items))) for minute, items in sorted(buckets.items())]
        # Last partially observed minute is omitted from the descriptive slope.
        points = [(minute, value) for minute, value in points if (minute + 1) * 60 <= end - start]
        metric["post_warmup_minute_medians"] = [{"minute": minute, "median": value} for minute, value in points]
        metric["ols_slope_per_minute_of_minute_medians"] = (
            float(np.polyfit([x for x, _ in points], [y for _, y in points], 1)[0])
            if len(points) >= 2 else None)
        output["metrics"][field] = metric
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    raw_paths = sorted((root / "performance/public-soak").glob("raw-*.jsonl"))
    assert raw_paths, "No public-soak raw records"
    records = [row for path in raw_paths for row in lines(path)]
    starts = [row for row in records if row.get("record_type") == "case_start" and row.get("phase") == "soak"]
    ends = [row for row in records if row.get("record_type") == "case_end" and row.get("phase") == "soak"]
    assert len(starts) == len(ends) == 1, "Expected exactly one completed dedicated soak"
    first, last = starts[0], ends[0]
    assert first["case_run_id"] == last["case_run_id"] and not last["interrupted"]
    assert last["elapsed_s"] >= 1800, "Dedicated soak shorter than planned 30 minutes"
    start, end = timestamp(first["at"]), timestamp(last["at"])
    assert end > start
    requests = [row for row in records if row.get("record_type") == "request" and
                row.get("case_run_id") == first["case_run_id"]]
    actual = [row for row in requests if row.get("started_offset_s") is not None]
    main_load = [row for row in actual if row["phase"] == "soak"]
    grouped = defaultdict(list)
    for row in main_load:
        grouped[row["case"]].append(row)
    assert len(grouped) == 5, "Unexpected mixed workload"
    assert max(map(len, grouped.values())) - min(map(len, grouped.values())) <= 1
    boot_ids = sorted({row["boot_id"] for row in actual if row.get("boot_id")})
    boot_changes = [row for row in records if row.get("record_type") == "boot_observed" and row.get("changed")]
    anchors = [row for row in actual if row["phase"] == "soak_anchor"]
    probes = [row for row in actual if row["phase"] == "soak_probe"]
    before = json.loads((root / "service-before.json").read_text())
    after_path = root / "service-after.json"
    after = json.loads(after_path.read_text()) if after_path.exists() else {}
    telemetry_path = root / "telemetry.jsonl"
    all_telemetry = lines(telemetry_path)
    telemetry = [row for row in all_telemetry if start <= timestamp(row["time_utc"]) <= end]
    assert len(telemetry) >= 1700, "Missing most of the 1-second soak telemetry"
    windows = []
    for minute in range(0, 30, 5):
        selected = [row for row in main_load if minute * 60 <= timestamp(row["at"]) - start < (minute + 5) * 60]
        windows.append({"start_minute": minute, "end_minute": minute + 5,
                        "by_case": {name: request_stats([row for row in selected if row["case"] == name])
                                    for name in sorted(grouped)}})
    result = {
        "status": "complete", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "One public client; one globally in-flight inference request; sequential five-case round robin; no retry. Health and anchor requests also run sequentially.",
        "window": {"start_utc": first["at"], "end_utc": last["at"],
                   "monotonic_elapsed_seconds": last["elapsed_s"], "wall_elapsed_seconds": end - start},
        "mixed_load": request_stats(main_load),
        "validated_mixed_request_goodput_per_second": sum(row["success"] for row in main_load) / last["elapsed_s"],
        "by_case": {name: request_stats(rows) for name, rows in sorted(grouped.items())},
        "health_probes": request_stats(probes), "anchors": request_stats(anchors),
        "anchor_cosine_vs_first_successful_reference": distribution([row["anchor_cosine"] for row in anchors if "anchor_cosine" in row]),
        "boot_ids": boot_ids, "observed_boot_change_events": len(boot_changes),
        "before_boot_id": before.get("boot_id"), "after_boot_id": after.get("boot_id"),
        "before_after_same_observed_boot": bool(after) and boot_ids == [before.get("boot_id")] == [after.get("boot_id")],
        "after_ready": after.get("ready"),
        "soak_window_resources": resource_analysis(telemetry, start, end),
        "five_minute_latency_windows": windows,
        "limitations": [
            "30 minutes of repeated fixed inputs, one run; no 24-hour SLA or independent repeated-run estimate.",
            "The media workload repeats one real image and ordered frames from one fixed video; quality diversity is evaluated separately.",
            "Client preparation, MP4 extraction, object storage, ANN indexing, RAG and application latency are not included.",
            "Memory summaries and slopes describe this window only. Allocator caches, input size, and process memory may change; this is not a proof of absence of leaks.",
            "Whole-experiment telemetry also includes quality encoding and expanding batches; its first/last memory values must not be interpreted as a soak leak test.",
            "No deliberate crash, network partition, OOM, worker restart, or power-cycle was injected into the live instance."
        ],
        "source_files": [{"path": str(path.relative_to(root)), "sha256": digest(path)}
                         for path in [*raw_paths, telemetry_path, root / "service-before.json", *([after_path] if after else [])]]
    }
    (root / "stability-analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "seconds": last["elapsed_s"],
                      "mixed_requests": len(main_load), "mixed_failures": sum(not row["success"] for row in main_load),
                      "boot_ids": boot_ids, "telemetry_samples": len(telemetry)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
