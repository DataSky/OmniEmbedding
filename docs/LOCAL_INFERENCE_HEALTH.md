# 本地推理服务健康接口实现说明

目标服务为 `192.168.31.204:8770`，模型为 `google/embeddinggemma-2`。
为现有推理服务增加下面三个接口，供 OmniEmbedding 的外部监控判断进程存活、模型就绪和真实推理能力。
接口复用业务已经加载的模型、推理执行器与状态；探测不写入视频库或检索索引。

## 当前检测目标

`127.0.0.1:8770/#health` 提供“本地电脑”和“Google Colab”两个独立页签，
默认选中并监测 `192.168.31.204:8770`，Colab 默认停用。
每个服务有独立开关、日志和密钥，页签切换只改变查看对象，开关控制后台健康请求。
业务编码目标与健康监控目标是独立配置，使用本地已有视频库可点击“打开本地服务”。

内网服务已经补齐三个健康接口；live、带密钥的 ready 和真实 inference 均已实测返回 200。
新版服务的深度检查响应使用 `ok:true`、`task:"query"`、`dimensions:768` 和布尔
`checks.shape:true`；完整契约见 [健康检查接口](HEALTH.md)。
外部监控兼容该新版格式及下文原 Colab 格式，GET ready 仍严格要求 `ready:true`。

## 接口职责

| 接口 | 鉴权 | 检查内容 | 是否运行模型 |
|---|---|---|---|
| `GET /health/live` | 无需密钥 | 当前 HTTP 进程能否及时响应 | 否 |
| `GET /health/ready` | Bearer | 已完成预热、模型可用、推理未卡死 | 否，只读内存状态 |
| `POST /health/inference` | Bearer | 实际执行一条固定文本并验证向量 | 是 |

现有监控按上述路径调用。字段名、大小写、JSON 类型和 HTTP 状态码应按下文实现。
特别是 `ready` 和检查结果必须为布尔值，不能使用字符串 `"true"` 或数字 `1`。
所有时间戳使用带时区的 UTC ISO 8601 格式，网页负责转换为北京时间。
响应设置 `Cache-Control: no-store`，避免代理缓存掩盖故障。

## 进程存活接口

`GET /health/live` 不依赖模型锁、数据库、外部网络或 GPU 操作。
模型加载中、推理繁忙、模型降级时，只要 HTTP 进程可响应，此接口仍返回 200。

```json
{
  "status": "ok",
  "boot_id": "3e0f7ca1-7dd3-474d-bcce-3416fb48bd30",
  "uptime_s": 123.4
}
```

`boot_id` 在模型服务进程每次启动时生成一次 UUID。同一进程的全部健康与推理响应使用同一个值，
每次请求都重新生成会导致监控误报重启。`uptime_s` 使用单调时钟计算，避免系统校时造成跳变。

单 GPU 服务建议单推理 worker。多个独立模型进程需分别监测实例，
避免请求被随机分配到不同 `boot_id` 后产生假重启告警。

## 模型就绪接口

请求头为 `Authorization: Bearer <ACCESS_KEY>`。
可以同时兼容 `X-API-Key`，但必须支持 Bearer，因为现有监控使用 Bearer。
缺失或错误密钥返回 401；不要返回登录页面或重定向。

服务启动时在后台加载并预热模型，只有一次真实预热验证成功后，才允许就绪接口返回 200。
仅检查 `model_loaded=true`、GPU 存在或 CUDA 可用不足以证明模型能正确推理。
后续 GET 请求只读取状态，不触发加载、预热或新的推理。

成功响应为 200，以下版本和设备值仅示意，应填实际加载的值：

```json
{
  "status": "ok",
  "ready": true,
  "boot_id": "3e0f7ca1-7dd3-474d-bcce-3416fb48bd30",
  "model": "google/embeddinggemma-2",
  "model_revision": "实际加载权重的固定提交版本",
  "device": "cuda:0",
  "dtype": "float32",
  "uptime_s": 123.4,
  "busy_for_s": 0.0,
  "inference_timeout_s": 120.0,
  "last_success": "2026-10-17T04:50:00+00:00",
  "last_error": null
}
```

`last_success` 是最近一次业务或探测推理成功的时间，没有成功记录时为 `null`。
监控另行记录最近一次深度探测成功时间；这两个时间不能混淆。

| 情况 | HTTP | `status` | `ready` |
|---|---|---|---|
| 模型完成预热，可以推理 | 200 | `ok` | `true` |
| 正在加载或预热 | 503 | `loading` | `false` |
| 模型推理失败或向量校验失败 | 503 | `degraded` | `false` |
| 真实推理执行超过配置的超时 | 503 | `stalled` | `false` |

503 响应也保留同一 `boot_id`、模型身份和状态字段。
普通业务忙碌期间可以继续返回 `200 / ok / ready:true`，并更新 `busy_for_s`。
超过推理超时后必须转为 `stalled`，不能无限返回“忙碌”隐藏卡死。

`last_error` 最多暴露脱敏的错误类别，例如 `InferenceError`、`InferenceStalled`。
不要返回密钥、用户文本、文件路径、完整异常或堆栈。

## 真实推理检查接口

请求为带 Bearer 的 **空请求体 POST**，无需 JSON 请求体或调用方传入测试文本。
接口内部选一条固定短文本，例如 `A small embedding health check.`，
通过与业务相同的 query 编码路径执行一条 768 维 embedding。
模型版本、任务前缀、精度和执行器均与 `/v1/embeddings` 一致。

对实际返回值计算以下检查，全部通过才返回 200：

- 数组形状严格为 `[1, 768]`。
- 所有数值有限，不含 NaN 或 Inf。
- 向量模长大于 `1e-12`。
- L2 模长接近 1；与仓库实现对齐可使用绝对容差 `0.05`。

原 Colab 协议的成功响应如下，`checks` 必须来自本次实际计算，不能写死为成功：

```json
{
  "status": "ok",
  "ready": true,
  "boot_id": "3e0f7ca1-7dd3-474d-bcce-3416fb48bd30",
  "model": "google/embeddinggemma-2",
  "model_revision": "实际加载权重的固定提交版本",
  "checks": {
    "shape": [1, 768],
    "finite": true,
    "nonzero": true,
    "unit_norm": true
  }
}
```

可以附加实际检查耗时等字段；无需返回完整向量。
健康接口仅验证单次 query，图片、视频能力应各自进行功能验收，不能由文本探测成功代替。

新版内网协议通过 `ok:true` 表示本次探测成功，`checks.shape:true` 表示实际形状检查通过，
并同时返回 `dimensions:768`、`task:"query"`；无需重复提供 `ready`。
两种协议都必须返回布尔 `finite:true`、`nonzero:true`、`unit_norm:true`。
存在明确的否定或矛盾字段时按失败处理，不能仅凭 HTTP 200、`status:"ok"` 或形状检查通过就判健康。
新版业务占用 GPU 时还可返回 `503 / status:busy / error:InferenceBusy`，外部监控将其单列为 BUSY。

| 情况 | HTTP | 处理 |
|---|---|---|
| 缺失或错误密钥 | 401 | 不进入模型执行 |
| GPU 正在执行其他请求 | 429 | 带 `Retry-After: 2`，不排长队、不启动另一份模型 |
| 模型加载中 | 503 | `status:loading`、`ready:false` |
| 推理卡死 | 503 | `status:stalled`、`ready:false`，交给进程守护处理 |
| 推理异常或向量校验失败 | 503 | `status:degraded`、`ready:false`，记录脱敏错误类别 |

模型已加载但因瞬时错误进入 `degraded` 时，允许此接口再次执行真实探测。
探测成功后恢复就绪状态；不要仅因 `ready:false` 永久拒绝探测，否则无法恢复。

## 与业务共享推理状态

将已有业务推理封装为统一执行入口，所有文本、图片、视频和健康推理共享同一个模型、
推理并发限制、忙碌开始时间和错误状态。可以用独立的短锁保护状态读写，
但不能给健康接口另设一把与业务无关的 GPU 锁。

业务锁卡死时，独立健康模型或独立执行路径仍可能返回成功，造成假健康。
HTTP 请求超时或客户端断开，也不能提前释放仍在计算的 GPU 锁。
计算真正结束后再释放，卡死由守护进程终止并重启模型进程。
异步 Web 路由应在线程池或模型工作进程运行阻塞计算，让存活和只读状态接口仍能响应。

仓库已有实现可参考：

- [`omniembed/inference.py`](../omniembed/inference.py)：`InferenceState`、预热、共享执行锁、向量校验和三个路由。
- [`tests/test_inference.py`](../tests/test_inference.py)：加载、繁忙、卡死、无效向量、恢复和鉴权测试。
- [`scripts/monitor_colab.py`](../scripts/monitor_colab.py)：当前监控调用顺序及字段校验。

接入已有服务时复用它的模型实例与业务执行入口，不另起一份 `Encoder`。
仅复制健康路由而不接入业务状态，无法提供真实的就绪和推理健康判断。

## 外部观测与日志汇总

当前观测约定为每 60 秒读取 live 和 ready，首次及之后每 300 秒调用一次真实推理检查。
轻探测默认超时 10 秒，真实推理检查独立超时 45 秒；响应不得超过 64 KiB，不跟随重定向。
监控每轮保存一条结构化日志，即使状态未变化也记录，包含时间、结果、HTTP 状态、延迟与 `boot_id`。

首次存活、就绪与真实推理检查均通过即判 HEALTHY。连续 3 次失败判 DOWN；
实际故障、实例重启或更换端点后，连续 2 次成功恢复 HEALTHY，期间显示确认进度。
监测器重启会在验证当前实例后续接最近 180 秒内的同一实例观测；旧实例或陈旧记录不用于续接。
429 单列 BUSY，
401/403 判 AUTH_ERROR。推理检查失败后，只有再次真实探测成功才能清除该故障。
没有日志或最后记录超过 180 秒显示 UNKNOWN 或 STALE，不能把“没有错误记录”当作健康。

本机网页的 `GET /api/service-health?profile=local&hours=3&limit=180` 只读这些历史记录，
不执行 HTTP 探测或模型推理。Codex 每 3 小时读取该汇总，状态正常且不变时保持安静。
这条汇总接口由外部观测端提供，内网推理服务先完成上述三个 `/health/*` 接口即可。

## 内网监测配置与业务兼容边界

内网监测已经接入；监测页签不改变业务推理地址或业务接口格式。

- 监测端仅显式允许 `192.168.31.204:8770` 使用 HTTP，其他远程地址默认仍要求 HTTPS，
  密钥不经环境代理或重定向转发。业务远程编码客户端的 HTTPS 规则保持独立。
- 内网服务的 `endpoint.json`、`inference.key`、`health.json` 与 `probe-log.jsonl`
  位于 `data/services/local/`，不会被 Colab 租约守护覆盖。
- 常驻内网服务忽略 Colab 到期时间或释放状态，Colab 到期不会停止本地监测。
- `data/services/profiles.json` 保存开关和页签选择。停用服务返回 `DISABLED`，
  取消后续健康请求且不清空历史；重新打开页面或重启 Web 服务后保持该设置。
- 健康接口兼容后，业务接入还需对齐 `task` 与 `input_type/titles`、
  文本任务前缀和多模态格式。当前远程编码器还会验证业务响应中的 `model_revision`、
  `boot_id`、维度及向量值；仅补健康接口不会自动解决这些协议差异。

## 手工验收

在调用端环境变量 `ACCESS_KEY` 中配置密钥；不要把真实密钥放进文档、URL 或截图。
以下示例直连当前局域网地址，客户端未授权的重定向不会被跟随。
最后一条命令会执行一次真实模型推理，前两条不会。

```bash
curl --noproxy '*' --max-time 10 --fail-with-body -i \
  http://192.168.31.204:8770/health/live

curl --noproxy '*' --max-time 10 --fail-with-body -i \
  -H "Authorization: Bearer ${ACCESS_KEY}" \
  http://192.168.31.204:8770/health/ready

curl --noproxy '*' --max-time 10 --fail-with-body -i -X POST \
  -H "Authorization: Bearer ${ACCESS_KEY}" \
  http://192.168.31.204:8770/health/inference
```

验收至少覆盖：无密钥 401、启动中 ready 503、预热后 ready 200、真实探测的数值检查、
繁忙 429、超时 stalled、故障后恢复，以及进程重启后 `boot_id` 改变。
坏输出、繁忙和超时用测试替身模拟，避免为验收故意耗尽真实 GPU 内存或卡死业务进程。
