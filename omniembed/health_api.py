"""/health/live、/health/ready、/health/inference，以及启动预热。契约见 docs/HEALTH.md。"""
import concurrent.futures as cf
import threading
import time
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from .config import CONFIG
from .encoder import QUERY_PREFIX
from .health import BOOT_ID, HEALTH, PROBE_TEXT, InferenceBusy, category_of, uptime_s

NORM_TOL = 0.05
NO_STORE = {"Cache-Control": "no-store"}


def check_vector(v):
    """对实际算出的向量逐项检查；返回 (全部通过, 各项结果, 模长)。"""
    v = np.asarray(v)
    shape_ok = v.ndim == 2 and v.shape == (1, 768)
    finite = bool(np.isfinite(v).all())
    norm = float(np.linalg.norm(v)) if finite and v.size else float("nan")
    checks = {"shape": shape_ok, "finite": finite, "nonzero": bool(finite and norm > 1e-12),
              "unit_norm": bool(finite and abs(norm - 1.0) <= NORM_TOL)}
    return all(checks.values()), checks, (round(norm, 6) if finite else None)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def identity(enc):
    return {"boot_id": BOOT_ID, "model": enc.model_id,
            "model_revision": getattr(enc, "resolved_revision", None) or getattr(enc, "revision", None),
            "device": getattr(enc, "device", None), "dtype": getattr(enc, "dtype", None)}


def _probe_once(enc, lock_timeout):
    t = time.perf_counter()
    v = enc.texts([PROBE_TEXT], QUERY_PREFIX, lock_timeout=lock_timeout)
    ok, checks, norm = check_vector(v)
    if not ok:
        HEALTH.record_failure("VectorCheckFailed")
    return ok, checks, norm, round((time.perf_counter() - t) * 1000, 1)


class Prober:
    """真实推理探测。并发的探测请求合并成一次（共享结果），不重复占用 GPU，也不会因“正在探测”而误报失败。"""

    def __init__(self, get_encoder):
        self.get_encoder = get_encoder
        self._pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="health-probe")
        self._lock = threading.Lock()
        self._inflight = None

    def run(self):
        with self._lock:
            if self._inflight is None or self._inflight.done():
                self._inflight = self._pool.submit(_probe_once, self.get_encoder(), CONFIG.probe_wait_s)
            fut = self._inflight
        return fut.result(timeout=CONFIG.probe_timeout_s)


def warmup(get_encoder, retry_s=30.0, stop=None):
    """后台加载模型并做一次真实推理验证；只有验证通过 ready 才会变成 true。失败则隔一段时间重试。"""
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            enc = get_encoder()
            enc.load()
            ok, _, _, _ = _probe_once(enc, None)
            if ok:
                HEALTH.mark_warm()
                print(f"[health] model warm: {enc.model_id}@{getattr(enc, 'resolved_revision', '?')} on {getattr(enc, 'device', '?')}")
                return True
            HEALTH.mark_load_error("VectorCheckFailed")
        except Exception as ex:  # noqa: BLE001  记录类别，日志里留原因
            HEALTH.mark_load_error(category_of(ex) if not isinstance(ex, (OSError, ValueError)) else "ModelLoadError")
            print(f"[health] warmup failed: {type(ex).__name__}: {str(ex)[:200]}")
        stop.wait(retry_s)
    return False


def build_router(auth_dep, get_encoder):
    r = APIRouter()
    prober = Prober(get_encoder)

    @r.get("/health/live")
    def live():
        return JSONResponse({"status": "ok", "boot_id": BOOT_ID, "uptime_s": uptime_s()}, headers=NO_STORE)

    @r.get("/health/ready", dependencies=[Depends(auth_dep)])
    def ready():
        s = HEALTH.snapshot()
        body = {"status": s["status"], "ready": s["ready"], **identity(get_encoder()), "uptime_s": uptime_s(),
                "busy_for_s": s["busy_for_s"], "inference_timeout_s": HEALTH.timeout_s,
                "last_success": s["last_success"], "last_error": s["last_error"]}
        return JSONResponse(body, status_code=200 if s["ready"] else 503, headers=NO_STORE)

    @r.post("/health/inference", dependencies=[Depends(auth_dep)])
    def inference():
        enc = get_encoder()
        base = {**identity(enc), "task": "query", "dimensions": 768, "checked_at": _now()}
        s = HEALTH.snapshot()
        if s["status"] == "loading":
            return JSONResponse({**base, "status": "loading", "ok": False, "checks": None, "error": None},
                                status_code=503, headers=NO_STORE)
        try:
            ok, checks, norm, ms = prober.run()
        except cf.TimeoutError:
            return JSONResponse({**base, "status": "stalled", "ok": False, "checks": None, "error": "InferenceTimeout"},
                                status_code=503, headers=NO_STORE)
        except InferenceBusy:
            return JSONResponse({**base, "status": "busy", "ok": False, "checks": None, "error": "InferenceBusy"},
                                status_code=503, headers=NO_STORE)
        except Exception as ex:  # noqa: BLE001
            return JSONResponse({**base, "status": "degraded", "ok": False, "checks": None, "error": category_of(ex)},
                                status_code=503, headers=NO_STORE)
        body = {**base, "status": "ok" if ok else "degraded", "ok": ok, "latency_ms": ms, "norm": norm,
                "checks": checks, "error": None if ok else "VectorCheckFailed"}
        return JSONResponse(body, status_code=200 if ok else 503, headers=NO_STORE)

    return r
