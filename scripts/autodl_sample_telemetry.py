#!/usr/bin/env python3
"""Bounded, read-only NVIDIA and /proc telemetry; contains no credentials."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--supervisor-pid-file", type=Path,
                        default=Path("/root/omniembedding-service/run/supervisor.pid"))
    args = parser.parse_args()
    assert args.duration > 0 and args.interval >= 0.5
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    fields = ["utilization.gpu", "utilization.memory", "memory.used", "memory.total",
              "temperature.gpu", "power.draw"]
    names = ["gpu_util_percent", "memory_util_percent", "gpu_memory_mib", "gpu_total_mib",
             "gpu_temperature_c", "gpu_power_w"]
    started = time.monotonic()
    with args.output.open("a", buffering=1) as output:
        while not stop and time.monotonic() - started < args.duration:
            beginning = time.monotonic()
            record = {"time_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_s": beginning - started}
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                                         "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=5, check=True)
                values = result.stdout.strip().splitlines()[0].split(",")
                record.update({name: float(value.strip()) for name, value in zip(names, values)})
                memory = {}
                for line in Path("/proc/meminfo").read_text().splitlines():
                    name, value = line.split(":", 1)
                    if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                        memory[name + "_kib"] = int(value.split()[0])
                record.update(memory)
                supervisor = int(args.supervisor_pid_file.read_text())
                children = Path(f"/proc/{supervisor}/task/{supervisor}/children").read_text().split()
                if children:
                    status = Path(f"/proc/{children[0]}/status").read_text()
                    record["worker_pid"] = int(children[0])
                    for line in status.splitlines():
                        if line.startswith("VmRSS:"):
                            record["worker_rss_kib"] = int(line.split()[1])
            except Exception as error:
                record["error_class"] = type(error).__name__
            output.write(json.dumps(record) + "\n")
            time.sleep(max(0, args.interval - (time.monotonic() - beginning)))


if __name__ == "__main__":
    main()
