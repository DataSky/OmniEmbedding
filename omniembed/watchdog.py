"""看门狗：启动服务进程并持续检查健康接口，卡死或进程无响应时自动重启。

Python 无法打断一个卡在 GPU 上的计算，/health/ready 只能如实报告 stalled；恢复只能靠重启进程。
用法：python -m omniembed watchdog（参数取自 .env / 环境变量，与 serve 相同）

重启条件（任一满足）：
  - 服务进程退出
  - /health/live 连续 N 次没有响应（进程卡死，连 HTTP 都不回）
  - /health/ready 连续 N 次返回 stalled
loading / degraded 不重启：加载中会自己好；degraded（如一次显存不足）通常在下一次成功推理后恢复，
重启也解决不了权重或显卡问题，只会反复加载。
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .config import CONFIG


def _get(url, key, timeout):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as ex:
        try:
            return ex.code, json.loads(ex.read() or b"{}")
        except json.JSONDecodeError:
            return ex.code, {}
    except Exception:
        return None, {}


def log(msg):
    print(f"[watchdog {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(interval_s=15.0, fails_to_restart=3, startup_grace_s=600.0, max_restarts_per_hour=6):
    host = "127.0.0.1" if CONFIG.host in ("0.0.0.0", "::") else CONFIG.host
    base = f"http://{host}:{CONFIG.port}"
    cmd = [sys.executable, "-m", "omniembed", "serve"]
    restarts = []
    while True:
        log(f"starting: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=os.environ.copy())
        started, boot_id, live_fail, stalled = time.monotonic(), None, 0, 0
        reason = None
        while reason is None:
            time.sleep(interval_s)
            if proc.poll() is not None:
                reason = f"process exited with code {proc.returncode}"
                break
            code, live = _get(base + "/health/live", None, 5)
            if code != 200:
                # 启动阶段（导入库、加载模型）HTTP 还没起来是正常的
                live_fail = live_fail + 1 if time.monotonic() - started > 60 else 0
                if live_fail >= fails_to_restart:
                    reason = f"/health/live unreachable {live_fail} times"
                continue
            live_fail = 0
            if boot_id and live.get("boot_id") != boot_id:
                log(f"boot_id changed {boot_id} -> {live.get('boot_id')} (another process on the port?)")
            boot_id = live.get("boot_id")
            code, ready = _get(base + "/health/ready", CONFIG.api_key, 5)
            status = ready.get("status")
            if code == 401:
                log("/health/ready returned 401: check OMNI_API_KEY")
            if status == "stalled":
                stalled += 1
                log(f"stalled ({stalled}/{fails_to_restart}), busy_for_s={ready.get('busy_for_s')}")
                if stalled >= fails_to_restart:
                    reason = "inference stalled"
            else:
                stalled = 0
                if status == "loading" and time.monotonic() - started > startup_grace_s:
                    log(f"still loading after {startup_grace_s:.0f}s (last_error={ready.get('last_error')})")
        log(f"restarting: {reason}")
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()  # 卡在 GPU 上的进程通常不响应 terminate
            proc.wait(timeout=20)
        now = time.monotonic()
        restarts = [t for t in restarts if now - t < 3600] + [now]
        if len(restarts) > max_restarts_per_hour:
            log(f"{len(restarts)} restarts within an hour; backing off 10 minutes")
            time.sleep(600)
            restarts = []


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="omniembed watchdog")
    ap.add_argument("--interval", type=float, default=15.0, help="检查间隔（秒）")
    ap.add_argument("--fails", type=int, default=3, help="连续几次异常后重启")
    a = ap.parse_args(argv)
    try:
        run(a.interval, a.fails)
    except KeyboardInterrupt:
        log("stopped")


if __name__ == "__main__":
    main()
