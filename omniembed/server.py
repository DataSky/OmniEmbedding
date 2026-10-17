"""HTTP 服务：上传 / 批量入库、任务进度、元数据与向量查看、召回、评估、Web 界面。

默认只监听 127.0.0.1。设置 OMNI_API_KEY 后，所有 /api 写操作需要请求头 X-API-Key；
要监听其他地址必须先设置 OMNI_API_KEY，否则拒绝启动。
"""
import csv
import io
import json
import os
import secrets
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import __version__, evaluation, media
from .config import CONFIG
from .encoder import ENCODER
from .health import HEALTH, InferenceError, category_of
from .health_api import build_router as build_health_router
from .health_api import warmup
from .openai_api import build_router
from .pipeline import Pipeline
from .retrieval import DIMS, MODES, Index
from .store import Store

CONFIG.ensure_dirs()
(CONFIG.data_dir / "tmp").mkdir(exist_ok=True)
STORE = Store(CONFIG.db_path)
PIPE = Pipeline(STORE)
INDEX = Index(STORE)
WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="OmniEmbedding", version=__version__)


def auth(x_api_key: str | None = Header(default=None)):
    if CONFIG.api_key and not secrets.compare_digest(x_api_key or "", CONFIG.api_key):
        raise HTTPException(401, "缺少或错误的 X-API-Key")


def embed_auth(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)):
    """/v1 接口按 OpenAI 习惯认 Authorization: Bearer <key>，也接受 X-API-Key。
    向量计算占 GPU，设置了密钥时这里读接口也要求密钥。"""
    if not CONFIG.api_key:
        return
    tok = x_api_key or ""
    if authorization and authorization.lower().startswith("bearer "):
        tok = authorization[7:].strip()
    if not secrets.compare_digest(tok, CONFIG.api_key):
        raise HTTPException(401, {"error": {"message": "invalid api key", "type": "invalid_request_error"}},
                            headers={"Cache-Control": "no-store", "WWW-Authenticate": "Bearer"})


app.include_router(build_router(embed_auth, lambda: INDEX.enc))
app.include_router(build_health_router(embed_auth, lambda: INDEX.enc))
WARMUP_STOP = threading.Event()


@app.on_event("startup")
def _start():
    PIPE.start()
    if os.environ.get("OMNI_SKIP_WARMUP") != "1":  # 测试里关闭，用假编码器手动预热
        threading.Thread(target=warmup, args=(lambda: INDEX.enc,), kwargs={"stop": WARMUP_STOP},
                         name="warmup", daemon=True).start()


@app.on_event("shutdown")
def _stop():
    WARMUP_STOP.set()
    PIPE.stop()


@app.exception_handler(ValueError)
async def _value_error(_: Request, ex: ValueError):
    return JSONResponse({"detail": str(ex)}, status_code=400)


@app.exception_handler(InferenceError)
async def _inference_error(_: Request, ex: InferenceError):
    """推理失败（显存不足、模型异常、推理锁等待超时）返回 503 和错误类别；原因只写服务日志。"""
    print(f"[inference] {ex.category}: {ex}")
    return JSONResponse({"detail": {"error": {"message": ex.category, "type": "server_error"}}}, status_code=503,
                        headers={"Retry-After": "5"})


# ---------- 入库 ----------
def _parse_tags(s):
    if not s:
        return []
    if isinstance(s, list):
        return [str(x).strip() for x in s if str(x).strip()]
    return [x.strip() for x in str(s).replace("，", ",").split(",") if x.strip()]


async def _save_upload(f: UploadFile) -> Path:
    """流式写入临时文件并限制大小，避免整段读入内存。"""
    limit = CONFIG.max_upload_mb * 2**20
    fd, name = tempfile.mkstemp(dir=CONFIG.data_dir / "tmp", suffix=".part")
    os.close(fd)  # mkstemp 返回已打开的句柄；Windows 上不关闭就无法移动文件
    tmp = Path(name)
    n = 0
    with open(tmp, "wb") as out:
        while chunk := await f.read(1 << 20):
            n += len(chunk)
            if n > limit:
                out.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"{f.filename} 超过 {CONFIG.max_upload_mb} MiB")
            out.write(chunk)
    return tmp


@app.post("/api/videos/upload", dependencies=[Depends(auth)])
async def upload(files: list[UploadFile] = File(...), titles: str = Form(""), description: str = Form(""),
                 tags: str = Form(""), filename_as_title: bool = Form(False)):
    """批量上传视频文件。titles 可选：JSON 数组，与文件一一对应。
    默认不拿文件名当标题（IMG_1234 这类名字会干扰检索），filename_as_title=true 时才用。"""
    if len(files) > CONFIG.max_batch:
        raise HTTPException(400, f"单次最多 {CONFIG.max_batch} 个文件")
    try:
        tl = json.loads(titles) if titles.strip().startswith("[") else []
    except json.JSONDecodeError:
        tl = []
    out, errors = [], []
    for i, f in enumerate(files):
        name = Path(f.filename or "video.mp4").name
        if Path(name).suffix.lower() not in media.VIDEO_EXT:
            errors.append({"file": name, "error": "不支持的文件类型"})
            continue
        tmp = await _save_upload(f)
        title = tl[i] if i < len(tl) and tl[i] else (Path(name).stem if filename_as_title else "")
        out.append(PIPE.submit_file(tmp, name, title, description, _parse_tags(tags)))
    return {"queued": out, "errors": errors}


def _rows_from_manifest(text: str, filename: str):
    """支持 CSV（表头含 url）、JSONL（每行 {"url": ...}）、或每行一个 URL 的纯文本。"""
    text = text.lstrip("﻿")
    if filename.endswith(".jsonl") or text.lstrip().startswith("{"):
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    first = text.splitlines()[0] if text.strip() else ""
    if filename.endswith(".csv") or ("url" in first.lower() and "," in first):
        return list(csv.DictReader(io.StringIO(text)))
    return [{"url": l.strip()} for l in text.splitlines() if l.strip() and not l.startswith("#")]


@app.post("/api/videos/urls", dependencies=[Depends(auth)])
async def add_urls(request: Request, manifest: UploadFile | None = File(None), urls: str = Form("")):
    """批量提交视频地址：表单 urls（每行一个），或上传 CSV / JSONL 清单（字段 url, title, description, tags）。
    也接受 JSON 请求体 {"items": [{"url": ..., "title": ...}]}。"""
    rows = []
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        rows = body.get("items") or [{"url": u} for u in body.get("urls", [])]
    else:
        if manifest is not None:
            rows += _rows_from_manifest((await manifest.read()).decode("utf-8", "replace"), manifest.filename or "")
        rows += _rows_from_manifest(urls, "") if urls.strip() else []
    if len(rows) > CONFIG.max_batch:
        raise HTTPException(400, f"单次最多 {CONFIG.max_batch} 条")
    out, errors = [], []
    for r in rows:
        url = str(r.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            errors.append({"url": url, "error": "只支持 http / https 地址"})
            continue
        out.append(PIPE.submit_url(url, r.get("title") or "", r.get("description") or "", _parse_tags(r.get("tags")),
                                   {k: v for k, v in r.items() if k not in ("url", "title", "description", "tags")}))
    return {"queued": out, "errors": errors}


@app.post("/api/videos/folder", dependencies=[Depends(auth)])
async def add_folder(body: dict):
    """导入本机目录（递归）。只在服务监听本机地址时开放，避免远程读取服务器文件。"""
    if CONFIG.host not in ("127.0.0.1", "localhost", "::1"):
        raise HTTPException(403, "目录导入只在本机模式下可用")
    root = Path(str(body.get("path", ""))).expanduser()
    if not root.is_dir():
        raise HTTPException(400, f"目录不存在：{root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in media.VIDEO_EXT and p.is_file())
    limit = int(body.get("limit") or CONFIG.max_batch)
    out = [PIPE.submit_path(p, tags=_parse_tags(body.get("tags"))) for p in files[:limit]]
    return {"found": len(files), "queued": out}


# ---------- 任务与元数据 ----------
def _public(v):
    v = dict(v)
    v["cover_url"] = f"/covers/{v['cover']}" if v.get("cover") else None
    v["video_url"] = f"/api/videos/{v['id']}/file" if v.get("path") else None
    v.pop("path", None)  # 不向前端暴露服务器文件路径
    return v


@app.get("/api/videos")
def list_videos(status: str | None = None, q: str | None = None, limit: int = 50, offset: int = 0):
    rows, total = STORE.list_videos(status, q, max(1, min(limit, 500)), max(0, offset))
    return {"total": total, "items": [_public(v) for v in rows]}


@app.get("/api/videos/{vid}")
def get_video(vid: str):
    v = STORE.get_video(vid)
    if not v:
        raise HTTPException(404, "not found")
    out = _public(v)
    out["frames"] = [dict(f, thumb_url=f"/covers/{f['thumb']}") for f in STORE.get_frames(vid)]
    out["vectors"] = {k: {"dim": r["dim"], "model": r["model"], "norm": round(float(np.linalg.norm(r["vec"])), 4),
                          "created_at": r["created_at"]} for k, r in STORE.get_vectors(vid).items()}
    return out


@app.get("/api/videos/{vid}/vector")
def get_vector(vid: str, kind: str = "visual", full: bool = False):
    """向量数据：统计信息 + 前 32 维（full=true 返回全部 768 维）+ 与其他向量的余弦。"""
    vecs = STORE.get_vectors(vid)
    if kind not in vecs:
        raise HTTPException(404, f"no {kind} vector")
    v = vecs[kind]["vec"]
    return {"video_id": vid, "kind": kind, "dim": int(v.size), "model": vecs[kind]["model"],
            "stats": {"norm": round(float(np.linalg.norm(v)), 4), "mean": round(float(v.mean()), 5),
                      "std": round(float(v.std()), 5), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)},
            "cosine_with": {k: round(float(r["vec"] @ v), 4) for k, r in vecs.items() if k != kind},
            "values": [round(float(x), 5) for x in (v if full else v[:32])]}


@app.patch("/api/videos/{vid}", dependencies=[Depends(auth)])
def edit_video(vid: str, body: dict):
    """修改标题 / 描述 / 标签；文本向量与融合向量随之重算，不重新抽帧。"""
    v = STORE.get_video(vid)
    if not v:
        raise HTTPException(404, "not found")
    fields = {k: body[k] for k in ("title", "description") if k in body}
    if "tags" in body:
        fields["tags"] = _parse_tags(body["tags"])
    STORE.update_video(vid, **fields)
    if v["status"] == "ready" and ({"title", "description"} & fields.keys()):
        PIPE.reembed_text(vid)
    return _public(STORE.get_video(vid))


@app.post("/api/videos/{vid}/retry", dependencies=[Depends(auth)])
def retry(vid: str):
    if not STORE.get_video(vid):
        raise HTTPException(404, "not found")
    PIPE.retry(vid)
    return {"ok": True}


@app.delete("/api/videos/{vid}", dependencies=[Depends(auth)])
def delete_video(vid: str, delete_file: bool = True):
    v = STORE.get_video(vid)
    if not v:
        raise HTTPException(404, "not found")
    frames = STORE.get_frames(vid)
    STORE.delete_video(vid)
    if delete_file:
        if v.get("path"):
            Path(v["path"]).unlink(missing_ok=True)
        for f in frames:
            (CONFIG.cover_dir / f["thumb"]).unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/videos/{vid}/file")
def video_file(vid: str):
    v = STORE.get_video(vid)
    if not v or not v.get("path") or not Path(v["path"]).exists():
        raise HTTPException(404, "not found")
    p = Path(v["path"]).resolve()
    if CONFIG.video_dir.resolve() not in p.parents:
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.get("/api/stats")
def stats():
    """不需要密钥，所以只返回类别和计数：不含服务器路径、异常文本或用户提交的 URL。"""
    enc, h = INDEX.enc, HEALTH.snapshot()
    return {"version": __version__, "model": enc.model_id, "model_revision": getattr(enc, "resolved_revision", None),
            "model_loaded": enc.loaded, "device": enc.device, "dtype": getattr(enc, "dtype", None),
            "model_load_s": enc.load_s, "health": h["status"], "status": STORE.counts(), "indexed": INDEX.size,
            "index_build_s": INDEX.built_s, "processed_since_start": PIPE.processed, "last_error": PIPE.last_error,
            "frames": {"fps": CONFIG.fps, "max_frames": CONFIG.max_frames}, "auth": bool(CONFIG.api_key)}


# ---------- 召回 ----------
def _hydrate(results):
    out = []
    for r in results:
        v = STORE.get_video(r["video_id"])
        if v:
            out.append({**r, "title": v["title"], "description": v["description"], "tags": v["tags"],
                        "duration_s": v["duration_s"], "cover_url": f"/covers/{v['cover']}" if v.get("cover") else None,
                        "video_url": f"/api/videos/{v['id']}/file"})
    return out


@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=300), mode: str = "hybrid", dim: int = 768, k: int = 20):
    if mode not in MODES or dim not in DIMS:
        raise HTTPException(400, f"mode 取 {MODES}，dim 取 {DIMS}")
    res, timing = INDEX.search(q, mode, dim, max(1, min(k, 100)))
    return {"q": q, "mode": mode, "dim": dim, "timing": timing, "results": _hydrate(res)}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), k: int = 20):
    raw = await file.read()
    if len(raw) > 10 * 2**20:
        raise HTTPException(413, "图片超过 10 MiB")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "无法识别的图片")
    return {"results": _hydrate(INDEX.search_image(img, max(1, min(k, 100))))}


@app.get("/api/videos/{vid}/similar")
def similar(vid: str, k: int = 12):
    return {"results": _hydrate(INDEX.similar(vid, max(1, min(k, 50))))}


# ---------- 评估 ----------
@app.get("/api/evalsets")
def list_evalsets():
    return {"items": STORE.list_evalsets()}


@app.post("/api/evalsets", dependencies=[Depends(auth)])
async def put_evalset(name: str = Form(...), description: str = Form(""), file: UploadFile | None = File(None),
                      items: str = Form(""), auto: str = Form("")):
    """导入评测集：上传 JSON / JSONL（{"query", "gold"}），或 auto=title|description 从已入库视频自动生成。"""
    name = name.strip()[:64]
    if not name:
        raise HTTPException(400, "需要评测集名称")
    if auto in ("title", "description"):
        rows = evaluation.from_titles(STORE, auto)
        description = description or f"自动生成：以视频{'标题' if auto == 'title' else '描述'}为查询、视频本身为答案"
    else:
        text = (await file.read()).decode("utf-8", "replace") if file is not None else items
        text = text.lstrip("﻿").strip()
        try:
            rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
        except json.JSONDecodeError as ex:
            raise HTTPException(400, f"JSON 解析失败：{ex}")
    rows, dropped = evaluation.resolve_gold(STORE, rows)
    if not rows:
        raise HTTPException(400, "没有可用条目（query 为空或 gold 找不到对应视频）")
    STORE.put_evalset(name, rows, description)
    return {"name": name, "n": len(rows), "dropped": dropped}


@app.post("/api/evalsets/{name}/run")
def run_eval(name: str, modes: str = "hybrid,dense,visual,text,bm25", dim: int = 768):
    es = STORE.get_evalset(name)
    if not es:
        raise HTTPException(404, "评测集不存在")
    ms = [m for m in modes.split(",") if m in MODES]
    if not ms or dim not in DIMS:
        raise HTTPException(400, "bad modes / dim")
    return {"name": name, "n": len(es["items"]), "results": [evaluation.evaluate(INDEX, es["items"], m, dim) for m in ms]}


# ---------- 静态资源与入口 ----------
app.mount("/covers", StaticFiles(directory=CONFIG.cover_dir), name="covers")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


def main():
    import uvicorn
    if CONFIG.host not in ("127.0.0.1", "localhost", "::1") and not CONFIG.api_key:
        raise SystemExit("监听非本机地址前必须设置 OMNI_API_KEY（否则任何人都能上传和删除）")
    print(f"[omni] data dir: {CONFIG.data_dir}")
    print(f"[omni] open http://{CONFIG.host}:{CONFIG.port}")
    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port)


if __name__ == "__main__":
    main()
