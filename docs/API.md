# HTTP 接口

默认地址 `http://127.0.0.1:8770`。设置了 `OMNI_API_KEY` 时，标 ✎ 的写操作需要请求头 `X-API-Key`。
FastAPI 自带的交互式文档在 `/docs`。

## 入库

| 方法 | 路径 | 说明 |
|---|---|---|
| ✎ POST | `/api/videos/upload` | multipart：`files`（可多个）、`titles`（可选，JSON 数组，与文件一一对应）、`description`、`tags`（逗号分隔）、`filename_as_title`（默认 false） |
| ✎ POST | `/api/videos/urls` | 三种写法任选：表单 `urls`（每行一个）；表单 `manifest` 文件（CSV / JSONL，字段 `url, title, description, tags`，其余列存入 `extra`）；JSON 请求体 `{"items": [{"url": ..., "title": ...}]}` |
| ✎ POST | `/api/videos/folder` | JSON `{"path": "D:/videos", "limit": 200, "tags": "..."}`。只在监听本机地址时可用 |

返回 `{"queued": [video_id, ...], "errors": [...]}`。处理在后台进行，用下面的接口查状态。

状态依次为 `queued → downloading → probing → extracting → embedding → ready`，失败为 `failed`（`error` 字段写明原因）。

## 视频与元数据

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/videos?status=&q=&limit=&offset=` | 列表；`q` 按标题 / 描述 / 来源模糊匹配 |
| GET | `/api/videos/{id}` | 完整元数据 + `frames`（时间点、缩略图）+ `vectors`（各类向量的维度、模型、模长） |
| GET | `/api/videos/{id}/vector?kind=visual&full=false` | 向量数据：统计（模长、均值、标准差、最值）、与同一视频其他向量的余弦、取值（默认前 32 维，`full=true` 全部 768 维）。`kind` 取 `visual` / `title` / `fused` |
| GET | `/api/videos/{id}/file` | 视频文件 |
| GET | `/api/videos/{id}/similar?k=12` | 画面相似的视频 |
| ✎ PATCH | `/api/videos/{id}` | JSON `{"title", "description", "tags"}`；会重算标题向量和融合向量，不重新抽帧 |
| ✎ POST | `/api/videos/{id}/retry` | 失败任务重新排队 |
| ✎ DELETE | `/api/videos/{id}?delete_file=true` | 删除记录、向量、缩略图，默认同时删除视频文件 |
| GET | `/api/stats` | 各状态数量、已入库数、编码后端、模型版本、抽帧设置；`remote_last_success_at` 为最近业务调用成功时间，读取此接口不访问远端 |
| GET | `/api/service-health?profile=local&hours=3&limit=180` | 按 `local` / `colab` 返回最近 3 小时日志及双服务开关；省略 profile 恢复选中的页签。只读历史，不触发探测，配置 API Key 后需 `X-API-Key` |
| ✎ PATCH | `/api/service-health/profiles/{profile}` | JSON `{"enabled":false}` 启停该服务的健康监测，或 `{"selected":true}` 保存选中页签；两者独立。设置持久化，停用返回 `DISABLED`，保留历史，不新建或续租实例 |

## 召回

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/search?q=&mode=hybrid&dim=768&k=20` | `mode`：`hybrid` / `dense` / `visual` / `text` / `bm25`；`dim`：768 / 512 / 256 / 128。结果带各路分数和耗时 |
| POST | `/api/search/image` | multipart `file`（图片，≤10 MiB），按画面向量找视频 |

## 评估

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/evalsets` | 评测集列表 |
| ✎ POST | `/api/evalsets` | multipart：`name`，加 `file`（JSON 数组或 JSONL，每条 `{"query", "gold": [...]}`）或 `auto=title`（用视频标题自动生成）。`gold` 可写 video_id、上传时的文件名或 URL |
| POST | `/api/evalsets/{name}/run?modes=hybrid,visual&dim=768` | 返回每种召回方式的 R@1/5/10、MRR@10、nDCG@10、耗时，以及最多 20 条错例 |

## 向量服务（OpenAI 兼容）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/embeddings`（别名 `/v1/embedding`） | 请求 `{"input": str | [str | {"image": base64/data URL}], "model", "dimensions", "encoding_format", "task"}`；返回 OpenAI 格式 `{"object": "list", "data": [{"object": "embedding", "index", "embedding"}], "model", "usage"}`，另附 `task` 与 `dimensions` |
| GET | `/v1/models` | 可用模型名：`embeddinggemma-2` 及 `:query` / `:document` / `:similarity` 后缀 |

鉴权：设置了 `OMNI_API_KEY` 时这两个接口都需要 `Authorization: Bearer <key>` 或 `X-API-Key`（向量计算占用 GPU，所以读操作也要求密钥）。
单次最多 512 条输入；每条文本最多 32,000 字符（超过 8192 token 的部分被截断）；图片 ≤10 MiB，不接受远程图片 URL。
错误按 OpenAI 格式返回 `{"detail": {"error": {"message", "type"}}}`。不支持 token id 数组输入。

```bash
curl http://192.168.31.204:8770/v1/embeddings -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "embeddinggemma-2:query", "input": ["how to cook pork belly"], "dimensions": 256}'
```

## 示例

```bash
# 批量上传
curl -F files=@a.mp4 -F files=@b.mp4 -F 'titles=["红烧肉做法","周末露营"]' http://127.0.0.1:8770/api/videos/upload
# 提交地址清单
curl -F manifest=@urls.csv http://127.0.0.1:8770/api/videos/urls
# 搜索
curl -G http://127.0.0.1:8770/api/search --data-urlencode "q=厨房里做菜" -d mode=visual
```

Windows 的 Git Bash 里 `curl -F` 传中文会按系统代码页编码，标题会乱码；请用网页、Python 或 PowerShell 7 调用。
