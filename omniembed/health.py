"""健康状态：进程存活、模型就绪、推理是否卡死。契约见 docs/HEALTH.md。

所有 GPU 推理都经过 HEALTH.inference() 这把锁串行执行，所以：
- 业务、入库、探测走同一条推理路径，显存占用可控；
- busy_for_s 就是“当前这次推理已经跑了多久”，超过 inference_timeout_s 判为 stalled；
- /health/ready 只读这里的内存状态，不碰推理锁，模型再忙也能立即回答。
对外只暴露错误类别（InferenceError / InferenceOOM / …），不暴露异常文本、路径或用户输入。
"""
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import CONFIG

BOOT_ID = str(uuid.uuid4())  # 每次进程启动生成一次
_T0 = time.monotonic()
PROBE_TEXT = "A small embedding health check."


class InferenceError(Exception):
    """带对外可见类别的推理错误；str(ex) 只进服务日志。"""

    def __init__(self, category="InferenceError", detail=""):
        super().__init__(detail or category)
        self.category = category


class InferenceBusy(InferenceError):
    def __init__(self):
        super().__init__("InferenceBusy")


def category_of(ex: BaseException) -> str:
    if isinstance(ex, InferenceError):
        return ex.category
    return "InferenceOOM" if "OutOfMemory" in type(ex).__name__ else "InferenceError"


def utc_iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds") if ts else None


def uptime_s():
    return round(time.monotonic() - _T0, 1)


class Health:
    def __init__(self, timeout_s):
        self.timeout_s = float(timeout_s)
        self.lock = threading.Lock()   # 推理锁：同一时间只有一个推理在 GPU 上
        self._st = threading.Lock()    # 保护下面的字段，持有时间极短
        self.reset()

    def reset(self):
        with self._st:
            self.warmed = False
            self.load_error = None
            self.busy_since = None
            self.last_success = None   # 墙钟时间，只用于展示
            self.last_error = None
            self._seq = 0              # 用序号判断最近一次是成功还是失败，不受校时影响
            self._ok_seq = 0
            self._fail_seq = 0

    # ---------- 推理记录 ----------
    @contextmanager
    def inference(self, lock_timeout=None):
        if not self.lock.acquire(timeout=-1 if lock_timeout is None else max(lock_timeout, 0)):
            raise InferenceBusy()
        with self._st:
            self.busy_since = time.monotonic()
        try:
            yield
        except BaseException as ex:
            if not isinstance(ex, InferenceBusy):
                self.record_failure(category_of(ex))
            raise
        else:
            self.record_success()
        finally:
            with self._st:
                self.busy_since = None
            self.lock.release()

    def record_success(self):
        with self._st:
            self._seq += 1
            self._ok_seq = self._seq
            self.last_success = time.time()

    def record_failure(self, category):
        with self._st:
            self._seq += 1
            self._fail_seq = self._seq
            self.last_error = category

    def mark_warm(self):
        with self._st:
            self.warmed, self.load_error = True, None

    def mark_load_error(self, category):
        with self._st:
            self.load_error = category
            self.last_error = category

    # ---------- 读取 ----------
    def snapshot(self):
        with self._st:
            busy = 0.0 if self.busy_since is None else time.monotonic() - self.busy_since
            if busy > self.timeout_s:
                status = "stalled"
            elif not self.warmed:
                status = "degraded" if self.load_error else "loading"
            elif self._fail_seq > self._ok_seq:
                status = "degraded"
            else:
                status = "ok"
            return {"status": status, "ready": status == "ok", "busy_for_s": round(busy, 1),
                    "last_success": utc_iso(self.last_success),
                    "last_error": "InferenceStalled" if status == "stalled" else self.last_error}


HEALTH = Health(CONFIG.inference_timeout_s)
