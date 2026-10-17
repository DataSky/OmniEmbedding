# 健康检查接口

三个接口供外部监控判断进程存活、模型就绪和真实推理能力。探测不写入视频库或检索索引。
所有响应带 `Cache-Control: no-store`；时间戳为带时区的 UTC ISO 8601（如 `2026-10-17T04:50:00+00:00`）；
`ready`、`ok` 和 `checks` 里的值都是 JSON 布尔。

| 接口 | 鉴权 | 检查内容 | 是否运行模型 |
|---|---|---|---|
| `GET /health/live` | 无 | HTTP 进程能否响应 | 否 |
| `GET /health/ready` | Bearer | 已预热、模型可用、推理未卡死 | 否，只读内存状态 |
| `POST /health/inference` | Bearer | 实际编码一条固定文本并校验向量 | 是 |

鉴权：`Authorization: Bearer <OMNI_API_KEY>`，也接受 `X-API-Key`。缺失或错误返回 401 JSON，不重定向。
未设置 `OMNI_API_KEY` 时不校验（只在监听本机地址时允许）。

## 实现方式

- 所有 GPU 推理（入库、搜索、`/v1/embeddings`、探测）经同一把推理锁串行执行。
  `busy_for_s` 是当前这次推理已运行的时间（单调时钟），超过 `inference_timeout_s` 判为 `stalled`。
- `/health/ready` 只读内存里的状态，不获取推理锁，模型再忙也立即返回。
- 启动后后台线程加载模型，并用一次真实推理做校验（与 `/health/inference` 相同的检查）；通过之前 `ready` 一直为 `loading`。
  加载或校验失败时为 `degraded`，30 秒后自动重试。
- `boot_id` 在进程启动时生成一次；`uptime_s` 用单调时钟。
- 对外只暴露错误类别，不含异常文本、路径、密钥或用户输入；完整原因只写服务日志。

## GET /health/live

不依赖模型、数据库、GPU。只要 HTTP 进程能响应就返回 200。

```json
{"status": "ok", "boot_id": "3e0f7ca1-7dd3-474d-bcce-3416fb48bd30", "uptime_s": 123.4}
```

## GET /health/ready

```json
{
  "status": "ok",
  "ready": true,
  "boot_id": "3e0f7ca1-7dd3-474d-bcce-3416fb48bd30",
  "model": "google/embeddinggemma-2",
  "model_revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
  "device": "cuda:0",
  "dtype": "bfloat16",
  "uptime_s": 123.4,
  "busy_for_s": 0.0,
  "inference_timeout_s": 120.0,
  "last_success": "2026-10-17T04:50:00+00:00",
  "last_error": null
}
```

| 情况 | HTTP | `status` | `ready` |
|---|---|---|---|
| 已预热，最近一次推理成功 | 200 | `ok` | `true` |
| 正在加载或预热 | 503 | `loading` | `false` |
| 加载失败、最近一次推理失败或向量校验失败 | 503 | `degraded` | `false` |
| 当前推理运行超过 `inference_timeout_s` | 503 | `stalled` | `false` |

- 503 也返回全部字段，`boot_id` 与模型身份不变。
- 正常忙碌（推理在跑但未超时）返回 200 / `ok`，`busy_for_s` 大于 0。
- `degraded` 在下一次推理成功后自动恢复为 `ok`。
- `last_success`：最近一次任何推理（业务或探测）成功的时间，没有则为 `null`。
- `last_error` 可能的值：`InferenceError`、`InferenceOOM`（显存不足，已释放缓存，进程继续服务）、
  `VectorCheckFailed`、`ModelLoadError`、`InferenceStalled`；没有错误为 `null`。
- `dtype` 填实际值：GPU 上为 `bfloat16`（官方说明 fp16 会出 NaN），CPU 上为 `float32`。

## POST /health/inference

空请求体。服务内部编码固定文本 `A small embedding health check.`，走与 `/v1/embeddings` 相同的 query 前缀、精度和推理锁。
对实际返回值检查：

| 检查 | 条件 |
|---|---|
| `shape` | 数组形状严格为 `[1, 768]` |
| `finite` | 全部有限，无 NaN / Inf |
| `nonzero` | L2 模长大于 `1e-12` |
| `unit_norm` | \|模长 − 1\| ≤ `0.05` |

成功（200）：

```json
{
  "status": "ok",
  "ok": true,
  "boot_id": "3e0f7ca1-7dd3-474d-bcce-3416fb48bd30",
  "model": "google/embeddinggemma-2",
  "model_revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
  "device": "cuda:0",
  "dtype": "bfloat16",
  "task": "query",
  "dimensions": 768,
  "checked_at": "2026-10-17T05:12:30+00:00",
  "latency_ms": 18.4,
  "norm": 0.998,
  "checks": {"shape": true, "finite": true, "nonzero": true, "unit_norm": true},
  "error": null
}
```

失败都返回 503，字段相同：

| 情况 | `status` | `error` | `checks` |
|---|---|---|---|
| 模型仍在加载 | `loading` | `null` | `null` |
| 向量检查未通过 | `degraded` | `VectorCheckFailed` | 本次实际结果，失败项为 `false` |
| 推理抛错 / 显存不足 | `degraded` | `InferenceError` / `InferenceOOM` | `null` |
| 业务推理占用 GPU，`OMNI_PROBE_WAIT`（默认 10 秒）内没排到 | `busy` | `InferenceBusy` | `null` |
| 排到了但 `OMNI_PROBE_TIMEOUT`（默认 30 秒）内没算完 | `stalled` | `InferenceTimeout` | `null` |

- `busy` 不算推理失败，不会让 `/health/ready` 变成 `degraded`；向量检查失败和推理抛错会。
- 同时到达的多个探测合并为一次推理并共享结果，探测不会在 GPU 上排长队。
- 建议监控间隔不短于 30 秒；探测一次约占 GPU 20 ms。

## 看门狗

`python -m omniembed watchdog` 启动服务并每 15 秒检查一次，以下任一情况连续 3 次即重启服务进程：
进程退出、`/health/live` 无响应、`/health/ready` 为 `stalled`。`loading` 与 `degraded` 不触发重启。
一小时内重启超过 6 次则暂停 10 分钟，避免反复加载。

注册为 Windows 开机自启服务（需要管理员，示例使用 [NSSM](https://nssm.cc/)）：

```powershell
nssm install OmniEmbedding "E:\path\to\.venv\Scripts\python.exe" "-m omniembed watchdog"
nssm set OmniEmbedding AppDirectory "E:\Codes\OmniEmbedding"
nssm set OmniEmbedding AppStdout "E:\Codes\OmniEmbedding\logs\service.log"
nssm set OmniEmbedding AppStderr "E:\Codes\OmniEmbedding\logs\service.log"
nssm start OmniEmbedding
```

## 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `OMNI_INFERENCE_TIMEOUT` | `120` | 单次推理超过多少秒判为 `stalled` |
| `OMNI_PROBE_WAIT` | `10` | 探测等推理锁的最长时间 |
| `OMNI_PROBE_TIMEOUT` | `30` | 探测整体最长时间 |
| `OMNI_MODEL_REVISION` | `914f7f8…` | 固定权重提交；设为 `main` 跟随最新（已入库向量可能与新查询不一致） |

## 命令行检查

```bash
curl -s http://192.168.31.204:8770/health/live
curl -s -H "Authorization: Bearer $KEY" http://192.168.31.204:8770/health/ready
curl -s -X POST -H "Authorization: Bearer $KEY" http://192.168.31.204:8770/health/inference
```
