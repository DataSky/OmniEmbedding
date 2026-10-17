"""OpenAI 兼容的向量接口：POST /v1/embeddings（别名 /v1/embedding）、GET /v1/models。

可以直接被 openai SDK、LangChain、LlamaIndex、Dify 等当作 "OpenAI Embeddings" 使用。
EmbeddingGemma 2 需要按任务加前缀，通用客户端不会区分查询和文档，所以提供三种方式指定：
  1. 请求体 "task"（扩展字段）：query / document / similarity / classification / clustering / code / qa / fact / none
  2. 模型名后缀："embeddinggemma-2:query"、"embeddinggemma-2:document" …（适合只能填模型名的客户端）
  3. 都不指定时用 OMNI_EMBED_DEFAULT_TASK（默认 query）。实测（VideoRetrieval nDCG@10）：查询 :query + 文档 :document 57.9；
     两边都用 query 前缀 53.3、都用 document 50.0、都用 similarity 44.6、都不加 38.9。所以不区分时默认 query。
另外支持 "dimensions"（MRL：768 / 512 / 256 / 128）、"encoding_format": "base64"、
以及图片输入：{"image": "data:image/jpeg;base64,..."}（与文本在同一向量空间，可做以图搜文 / 以文搜图）。
"""
import base64
import binascii
import io
import os
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from PIL import Image

from .encoder import ENCODER, renorm

TASK_PREFIX = {
    "query": "task: search result | query: ",
    "search_query": "task: search result | query: ",
    "document": "title: none | text: ",
    "search_document": "title: none | text: ",
    "passage": "title: none | text: ",
    "similarity": "task: sentence similarity | query: ",
    "sts": "task: sentence similarity | query: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "code": "task: code retrieval | query: ",
    "qa": "task: question answering | query: ",
    "fact": "task: fact checking | query: ",
    "none": "",
}
DIMS = (768, 512, 256, 128)
MAX_INPUTS = 512
MAX_CHARS = 32_000  # 约 8k token 上限；超出由分词器截断，这里只防异常大的请求
MAX_IMAGE_BYTES = 10 * 2**20


class EmbeddingRequest(BaseModel):
    input: str | list | dict
    model: str | None = None
    encoding_format: str | None = "float"
    dimensions: int | None = None
    task: str | None = None
    user: str | None = None


def _model_names(enc):
    base = enc.model_id.rsplit("/", 1)[-1]
    return base, [enc.model_id, base]


def _resolve_task(req: EmbeddingRequest):
    task = (req.task or "").strip().lower()
    if not task and req.model and ":" in req.model:
        task = req.model.rsplit(":", 1)[1].strip().lower()
    if not task:
        task = os.environ.get("OMNI_EMBED_DEFAULT_TASK", "query").strip().lower()
    if task not in TASK_PREFIX:
        raise HTTPException(400, {"error": {"message": f"unknown task '{task}', use one of {sorted(TASK_PREFIX)}",
                                            "type": "invalid_request_error"}})
    return task


def _decode_image(s: str):
    if not isinstance(s, str):
        raise ValueError("image must be a base64 string or data URL")
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    elif s.startswith(("http://", "https://")):
        # 不在这里抓取远程图片：避免把向量接口变成服务器端代理
        raise ValueError("remote image URLs are not supported; send base64 / data URL")
    try:
        raw = base64.b64decode(s, validate=False)
    except binascii.Error:
        raise ValueError("invalid base64 image")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image larger than 10 MiB")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise ValueError("cannot decode image")


def _items(inp):
    """归一化成 [("text", str) | ("image", PIL)]。token 数组输入（OpenAI 允许）不支持。"""
    if isinstance(inp, (str, dict)):
        inp = [inp]
    if not isinstance(inp, list) or not inp:
        raise ValueError("input must be a non-empty string, list of strings, or list of {image: ...} objects")
    if len(inp) > MAX_INPUTS:
        raise ValueError(f"at most {MAX_INPUTS} inputs per request")
    out = []
    for x in inp:
        if isinstance(x, str):
            if not x.strip():
                raise ValueError("input strings must not be empty")
            out.append(("text", x[:MAX_CHARS]))
        elif isinstance(x, dict) and "image" in x:
            out.append(("image", _decode_image(x["image"])))
        elif isinstance(x, dict) and "text" in x:
            out.append(("text", str(x["text"])[:MAX_CHARS]))
        elif isinstance(x, (int, list)):
            raise ValueError("token-id inputs are not supported; send text")
        else:
            raise ValueError("unsupported input item")
    return out


def build_router(auth_dep, get_encoder=lambda: ENCODER):
    """get_encoder 便于测试时换成假编码器。"""
    r = APIRouter()

    @r.get("/v1/models", dependencies=[Depends(auth_dep)])
    def models():
        base, _ = _model_names(get_encoder())
        now = int(time.time())
        ids = [base] + [f"{base}:{t}" for t in ("query", "document", "similarity")]
        return {"object": "list", "data": [{"id": i, "object": "model", "created": now, "owned_by": "google"} for i in ids]}

    def embeddings(req: EmbeddingRequest):
        enc = get_encoder()
        base, accepted = _model_names(enc)
        if req.model and req.model.split(":", 1)[0] not in accepted:
            raise HTTPException(404, {"error": {"message": f"model '{req.model}' not found; available: {base}",
                                                "type": "invalid_request_error"}})
        dim = req.dimensions or 768
        if dim not in DIMS:
            raise HTTPException(400, {"error": {"message": f"dimensions must be one of {DIMS}", "type": "invalid_request_error"}})
        task = _resolve_task(req)
        try:
            items = _items(req.input)
        except ValueError as ex:
            raise HTTPException(400, {"error": {"message": str(ex), "type": "invalid_request_error"}})
        vecs = np.zeros((len(items), 768), dtype=np.float32)
        ti = [i for i, (k, _) in enumerate(items) if k == "text"]
        ii = [i for i, (k, _) in enumerate(items) if k == "image"]
        if ti:
            vecs[ti] = enc.texts([items[i][1] for i in ti], TASK_PREFIX[task])
        if ii:
            vecs[ii] = enc.images([items[i][1] for i in ii])
        vecs = renorm(vecs, dim)
        # 图片按官方 280 token/张计
        n_tok = enc.count_tokens([TASK_PREFIX[task] + items[i][1] for i in ti]) + 280 * len(ii)
        data = []
        for i, v in enumerate(vecs):
            emb = base64.b64encode(v.astype("<f4").tobytes()).decode() if req.encoding_format == "base64" else v.tolist()
            data.append({"object": "embedding", "index": i, "embedding": emb})
        return {"object": "list", "data": data, "model": req.model or base,
                "usage": {"prompt_tokens": n_tok, "total_tokens": n_tok}, "task": task, "dimensions": dim}

    r.add_api_route("/v1/embeddings", embeddings, methods=["POST"], dependencies=[Depends(auth_dep)])
    r.add_api_route("/v1/embedding", embeddings, methods=["POST"], dependencies=[Depends(auth_dep)],
                    include_in_schema=False)
    return r
