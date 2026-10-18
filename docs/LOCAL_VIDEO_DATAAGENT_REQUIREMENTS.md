# 本地视频数据准备与推理接入需求

执行方：DataAgent。消费方：OmniEmbedding。版本：1.0。日期：2026 年 10 月 17 日。

目标是先把原始视频及其业务元数据可靠地落盘，交付可校验、可重试、可追溯的数据包，再由 OmniEmbedding 完成抽帧、embedding 和检索。原始数据保留在本地，换用本地推理服务或 Colab 时沿用同一份数据契约。

本次 DataAgent 必须完成阶段 A 的数据准备。阶段 B 定义后续入库与推理验收要求；只有目标应用的导入入口及视频推理能力通过验证后才执行。健康页显示正常、下载任务完成、接口返回已入队，都不能代替 embedding 成功。

## 任务范围

| 阶段 | 工作 | 完成标志 |
|---|---|---|
| A 数据准备 | 读取源记录、下载原视频、校验媒体、整理元数据、生成清单、保留失败账本 | `DATA_READY`，数据包可被另一个进程完整读取和校验 |
| B 小批量入库 | 选择已验证的导入方式，提交阶段 A 的视频和元数据，核对任务及向量，执行检索 | `INDEX_READY`，至少完成本次试验批的结果对账 |
| C 扩量 | 按增量游标、容量及并发预算继续采集和入库 | 仅在 A、B 验收后开展，保留每批报告 |

先固定最多 20 条源记录作为试验批。如果可用记录不足 20 条，记录实际数量。下载失败的记录也必须保留在账本和失败报告中，不能通过悄悄替换样本伪造全成功结果。数据源类型、实际表名、访问方式及全量规模由执行环境提供，不猜测不存在的字段或凭证。

批次有计划记录且全部通过时才标记 `DATA_READY`；部分成功为 `PARTIAL_DATA_READY`，全部失败为 `DATA_FAILED`，没有发现记录为 `NO_DATA`。后面三种情况同样交付实际清单、账本和报告，不能用空清单宣称准备成功。记录级状态仍采用后文的任务状态枚举。

本次不要求生成 ASR、OCR、视频摘要、字幕、镜头切分或秒级片段索引。如果源数据已经包含这些内容，保留原文及来源，暂不自动混入检索文本。默认索引单位是一条完整视频。

## 当前项目可以直接使用的能力

以下现状以当前仓库代码为准；实际部署仍需核对其版本与接口能力。

| 项目 | 当前行为 | 对 DataAgent 的约束 |
|---|---|---|
| 本地文件入库 | `python -m omniembed ingest <目录> --meta <JSONL>` | JSONL 使用 `file,title,description,tags,extra`；不会自动读取逐视频同名 JSON |
| 清单关联 | CLI 按 `Path(file).name` 关联元数据 | 同一导入批内所有视频的 basename 必须唯一，即使位于不同子目录 |
| 文件保存 | 本地导入复制原件到应用 `data/videos` | 原件和导入副本通常各占一份空间 |
| 文本向量 | 标题和描述生成文本向量 | `extra` 不自动参与 embedding；tags 参与 BM25，但不参与文本向量 |
| 视频向量 | 采样有序帧，调用模型原生视频编码，得到一条视觉向量 | 逐帧图片向量的平均值不能冒充原生视频向量 |
| 文件重复 | 相同 SHA256 可复用已就绪记录的视觉向量 | 仍可能新增视频行、复制文件及抽帧，不等于完整幂等入库 |
| HTTP 文件上传 | `/api/videos/upload` 支持视频、标题、描述和 tags | 不支持逐条 `extra`；多文件请求共用 description、tags |
| HTTP 目录入库 | `/api/videos/folder` 只在绑定 loopback 时开放 | 路径属于服务所在电脑；该入口也不读取逐条元数据清单 |
| 健康页 | 本地电脑、Colab 独立监测，读取历史摘要 | 监测开关不切换业务推理目标，不验证全部业务接口兼容性 |

不要把本机的 `C:/...`、`D:/...` 或 `/Users/...` 路径发给另一台机器，并期待对方能读取该文件。跨机器需要传输文件内容，或先同步到服务所在机器，再使用对应机器的路径。

## 输入与配置

### 原始记录最小映射

| 输入 | 必需 | 说明 |
|---|---|---|
| `source_system` | 是 | 来源命名空间，例如 `internal_video`；小写 ASCII 字母开头，以字母或数字结束，分隔符可用单个下划线或短横线 |
| `source_id` | 是 | 来源中的稳定视频主键，按字符串读取，保留前导零，不能使用随机数或临时文件路径替代 |
| 视频定位信息 | 是 | 已授权的文件路径、对象存储定位、下载 API 参数或下载 URL；只供下载器使用 |
| title、description、tags | 否 | 读取真实业务字段；缺失时分别置空字符串、空字符串、空数组 |
| 来源链接和时间 | 否 | 稳定详情页链接、源记录创建和更新时间；无法确定则置 null |
| 其他业务字段 | 否 | 按字段白名单映射到 `extra`，例如类目、已有转写、来源状态 |

执行前输出 `field-mapping.json`，记录真实源字段到标准字段的映射、时间单位、时区、缺失处理和来源筛选条件。不能把假定的字段名当成已经存在的表结构。来源没有稳定 ID 时，应先确定可重放的业务组合键，不能每次重新生成 ID。

长整型业务 ID 应直接按字符串读取；不能先经过浮点数、JavaScript 不安全整数或科学计数法，再转回字符串，否则可能已经丢失精度。

建议运行配置如下，名称为 DataAgent 实现约定，不是 OmniEmbedding 已有命令行参数：

```yaml
source_system: internal_video
source_config_ref: DATA_SOURCE_CONFIG
output_root: /path/to/video-dataset
pilot_limit: 20
download_concurrency: 4
download_timeout_s: 120
max_download_attempts: 3
max_video_bytes: 524288000
embedding_enabled: false
embedding_profile: local
embedding_concurrency: 1
```

`source_config_ref` 引用执行环境已有配置；不在配置样例、交付清单和日志中填写真实凭证。500 MiB 是本轮默认单文件上限，来自当前应用默认上传限制，可经显式配置调整；超限记录报告 `SIZE_LIMIT`，不要静默截断文件。视频下载并发与 GPU 请求并发分别控制，不能因为下载开了 4 路，就向模型同时发送 4 路任务。

## 交付目录

```text
video-dataset/
  README.md
  field-mapping.json
  manifests/
    batch-20261017-001.jsonl       # 本批成功落盘且校验通过的视频
    current.jsonl                  # 当前有效版本视图，每个 record_id 最多一条
  metadata/
    batch-20261017-001.omniembed.jsonl
  videos/
    rec_<记录哈希>__<内容哈希前16位>.mp4
  source-metadata/                  # 可选，已脱敏的原始业务元数据
  state/
    tasks.sqlite                   # DataAgent 自己的任务账本，不是应用数据库
  reports/
    batch-20261017-001.summary.json
    batch-20261017-001.failures.jsonl
    batch-20261017-001.validation.json
  logs/
    events.jsonl
```

所有清单和日志使用 UTF-8，无 BOM。JSONL 每行是一个完整 JSON 对象；不能使用 Python 字典字符串、数组包裹整份 JSONL、NaN 或 Infinity。

批次清单发布后保持不变；`current.jsonl` 可通过临时文件加原子替换更新。路径均相对于 `output_root`，使用 `/` 分隔。禁止绝对路径、`..`、反斜线及指向根目录外的符号链接。视频文件名使用稳定哈希；原始文件名另存字段，不拿文件名生成业务标题。

命名规则为 `videos/<record_id>__<完整内容 SHA256 前16位>.<真实容器扩展名>`。如果短内容哈希发生冲突，扩展到完整内容哈希并更新清单，不能覆盖不同内容。全批 basename 唯一性是强校验项。

## 标准视频记录

机器契约以 [JSON Schema](schemas/local-video-manifest.schema.json) 为准；完整结构示例在 [标准清单示例](examples/local-video-manifest.example.jsonl)。示例使用虚构来源及媒体参数，只说明结构，不包含真实视频，不能当作验收数据。

主清单每行代表一个已验证的视频版本。失败任务不写入该成功清单，单独进入任务账本和失败报告。

| 字段 | 类型 | 规则 |
|---|---|---|
| `schema_version` | string | 固定 `1.0` |
| `record_id` | string | `rec_` 加稳定身份哈希，算法见下文 |
| `source.system` | string | 来源命名空间 |
| `source.id` | string | 来源稳定主键，保留前导零 |
| `source.url` | string 或 null | 已脱敏稳定详情页 URL，不放临时下载签名、账号口令或访问密钥 |
| `source.created_at`、`source.updated_at` | string 或 null | 源系统业务时间，必须知道其时区；未知为 null |
| `source.fetched_at` | string | 本次读取源记录的时间 |
| `asset.path` | string | 根目录下的相对视频路径 |
| `asset.sha256` | string | 原始视频全部字节的 SHA256，64 位小写十六进制 |
| `asset.bytes` | integer | 实际落盘字节数，必须大于 0 |
| `asset.downloaded_at` | string | 完整下载及校验完成时间 |
| `asset.original_filename`、`asset.mime_type` | string 或 null | 原始文件名和识别到的媒体类型；未知为 null |
| `metadata.title` | string | 真实标题，可为空 |
| `metadata.description` | string | 真实说明，可为空；完整保留，不为适配模型偷偷截断 |
| `metadata.tags` | string array | 去空、去重、顺序稳定，不使用逗号拼接字符串 |
| `metadata.language` | string 或 null | 已知语言代码，例如 `zh-CN`；未知为 null |
| `metadata.published_at` | string 或 null | 业务发布时间，未知为 null |
| `metadata.author_id`、`metadata.author_name` | string 或 null | 可用且允许保留的业务作者标识；ID 为字符串 |
| `metadata_sha256` | string | 对规范化后的 metadata 对象计算，算法见下文 |
| `media.duration_s` | number | 实测时长，单位秒，大于 0 |
| `media.width`、`media.height` | integer | 有效像素尺寸，均大于 0 |
| `media.fps` | number 或 null | 可确定时为正数；不能确定时为 null |
| `media.codec` | string 或 null | 实测编码，不能确定时为 null |
| `media.has_audio` | boolean 或 null | 明确有或没有音轨；未检查或无法判断为 null |
| `extra` | object | 白名单业务字段，不参与当前默认文本 embedding |

时间使用 ISO 8601，输出统一为 UTC `Z`，例如 `2026-10-17T06:00:00Z`。数值时间戳先按映射表明确秒或毫秒，再转换；不能根据位数不一致的数据随意猜测。北京展示时间由消费端转换，不把北京时间字符串当 UTC 保存。

如果源数据只有日期而没有时刻，或无法确定其时区，保留原值到 `extra`，对应标准时间字段置 null，不伪造午夜时间。

### 身份和内容哈希

以下算法是跨批次关联约定。`source.system`、`source.id` 禁止包含 NUL；来源主键不做会改变身份的隐式清洗。

```python
import hashlib
import json

record_id = "rec_" + hashlib.sha256(
    (source_system + "\0" + source_id).encode("utf-8")
).hexdigest()

metadata_bytes = json.dumps(
    metadata, ensure_ascii=False, sort_keys=True,
    separators=(",", ":"), allow_nan=False,
).encode("utf-8")
metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
```

metadata 的七个字段必须齐全：`title,description,tags,language,published_at,author_id,author_name`。缺失值按前文补空字符串、空数组或 null 后再计算哈希。字符串先统一 Unicode NFC、换行统一 LF，标题、描述去首尾空白；tags 去首尾空白、去空、去重后按 Unicode 码点排序。保留必要的原始业务内容时，另存已脱敏原始元数据，不往检索字段填充整份源 JSON。

三个标识承担不同职责：`record_id` 标识业务视频，`asset.sha256` 标识某一份原始文件，`metadata_sha256` 标识可索引元信息。它们都不等于 OmniEmbedding 入库后生成的内部 `video_id`。相同文件可能属于不同业务记录，不得只按文件哈希合并掉合法的来源关系。

## 下载与媒体校验

1. 按稳定游标读取源数据，保存本批样本 ID 和采集条件。增量任务优先使用来源提供的游标或 `更新时间加主键` 的稳定顺序，并设置本批时间上界；重试不会跳过未完成记录。
2. 下载到目标目录内的 `.part` 临时文件，流式计算字节数和 SHA256。对响应状态、超时、大小限制和实际内容做校验；HTTP 200 返回 HTML 登录页不算视频下载成功。
3. 断点续传前验证源文件版本或 ETag 等信息，以及服务端是否正确支持 Range。不满足时从头重下，不能把两个版本的字节拼接成一个文件。
4. 解析容器、视频轨道、时长、分辨率等。至少解码出一帧；对可以定位的正常视频，检查开头、中间及靠近结尾的采样点。媒体解析和解码失败独立报告，不能只检查扩展名或文件大小。
5. 全部通过后原子改名到最终路径，再把成功状态与清单记录发布。崩溃残留的 `.part` 文件不能出现在成功清单中。

保留原始字节，不静默转码、降画质、去音轨或替换成抽帧图片。原始容器不能处理时，报告 `UNSUPPORTED_MEDIA`；如以后需要转码，保留原件、派生文件、转换参数和两份哈希，不能覆盖原件冒充原始数据。

每次重跑先检查已完成文件的存在性、字节数和内容哈希；一致时跳过下载。来源更新或 URL 刷新但字节相同，不重复保存一份内容版本。来源同一 ID 对应新视频字节时保留旧版本文件，更新当前视图，同时记录版本变更。

磁盘预算至少包括原始文件、应用导入副本、在途临时文件、缩略图与派生结果。当前 CLI 会复制视频，因此通常需要预留原视频总量两倍以上的空间。容量不足时停止领取新任务并给出所需空间，不能删除原件来强行继续。

## 元数据整理与兼容清单

生成与每批主清单一一对应的 CLI 清单，字段结构见 [兼容清单示例](examples/omniembed-metadata.example.jsonl)。

```json
{
  "file": "rec_<记录哈希>__<内容哈希前16位>.mp4",
  "title": "源数据中的视频标题",
  "description": "源数据中的视频说明",
  "tags": ["源标签"],
  "extra": {
    "data_contract": {
      "record_id": "rec_<记录哈希>",
      "source": {},
      "asset_sha256": "<完整内容哈希>",
      "metadata_sha256": "<元数据哈希>",
      "media": {}
    },
    "business": {}
  }
}
```

以上片段说明字段映射，完整、合法的机器示例以示例文件为准。`file` 必须取 `asset.path` 的 basename；`title,description,tags` 原样映射规范化 metadata。`data_contract.source`、`data_contract.media` 放入主记录对应的完整对象，`business` 放入原 `extra`。这样导入应用后仍能通过 `extra.data_contract.record_id` 追溯源记录。

标题或描述缺失时保留空值，并在报告中统计。不能让大模型编造标题、把文件名当标题，或用标签填充“原始描述”。已有 ASR、OCR、字幕及业务评价可以存入 `extra` 或单独文件，但当前消费端没有自动索引这些字段。

## 任务账本与幂等

下载状态和入库状态分别保存，不用一个 `success` 字段混在一起。

| 账本字段 | 要求 |
|---|---|
| `record_id`、源主键、批次 ID | 稳定关联，可查回来源 |
| `asset_sha256`、`metadata_sha256`、本地相对路径 | 内容版本及元数据版本 |
| `source_record_fingerprint` | 对规范化后的稳定 source 字段、metadata、extra 计算快照哈希，排除采集时间、下载签名及重试状态，识别仅业务扩展信息变化 |
| `data_status` | `DISCOVERED / DOWNLOADING / VALIDATING / DATA_READY / DATA_FAILED` |
| `ingest_status` | `NOT_SUBMITTED / SUBMITTING / QUEUED / INDEX_READY / INDEX_FAILED / SUBMIT_UNKNOWN` |
| `omni_video_id` | 应用返回的内部视频 ID；未提交为 null |
| `attempts`、`next_retry_at`、阶段时间 | 支持限次重试和断点续跑 |
| `error_code`、`error_stage`、脱敏说明 | 明确失败发生在采集、下载、媒体、提交或推理阶段 |
| `embedding_profile`、输入指纹和处理版本 | 仅进入阶段 B 后记录，不用监测 tab 推断业务配置 |

账本可以使用 DataAgent 自己的 SQLite，必须位于独立路径，不得直接修改 OmniEmbedding 的 `omni.db`。每个业务 ID 的每个内容版本及来源快照版本应有唯一任务记录。快照哈希使用与 metadata 相同的 JSON 规范化方法，并在实现中固定参与字段；不要因 `metadata_sha256` 未变就忽略 `extra` 或来源属性的更新。主清单不保存不断变化的重试状态，账本不替代最终清单。

重复执行相同批次不能新增下载文件或再次提交已经确认入库的版本。仅修改标题或描述时，应保留视觉向量并更新文本及融合结果；仅修改其他业务属性时，不应无条件重算视频 embedding。当前应用尚无外部 ID upsert 契约，不能把这些目标误写成已经实现的能力。

提交接口超时不代表服务端没有成功入队。此时进入 `SUBMIT_UNKNOWN`，先按外部 ID 或幂等键查询和对账，不得立即重传制造重复行。现有 CLI、HTTP 上传均不提供完整的外部幂等保证，阶段 B 必须先选定补齐方式。

## 推理与入库接入要求

### 首选的数据流

```text
原始数据源
  → DataAgent 下载原视频并整理元数据
  → 标准数据包及兼容清单
  → OmniEmbedding 的统一导入入口
  → 本地抽帧及元数据处理
  → 已验证的 embedding 服务
  → 应用向量存储、检索和状态查询
```

DataAgent 不需要另外建设向量库，也不需要为了采集任务自己加载一份模型。优先由应用负责生成和保存 `visual / title / fused` 向量，以及查询侧的同模型编码，避免数据导入用一个模型、检索用另一个模型。

### 当前可用路径与需要补齐的能力

| 路径 | 适用条件 | 注意事项 |
|---|---|---|
| 现有 CLI 加 `--meta` | 数据和应用在同一电脑，能安排独占导入窗口，编码后端已验证 | 会复制文件、可能加载本地模型；不要与占用同一 GPU 的 Web 进程重复加载模型，也不要与其并行写同一应用库 |
| 现有 HTTP upload | 临时验证单视频上传及文本、标签，允许来源关联暂存在 DataAgent 账本 | 单条提交可带各自描述，但不能完整传入 `extra`，不作为完整元数据交付验收方式 |
| 在线完整导入入口 | 后续在应用中补齐 | 应接收逐条元数据和文件、稳定外部 ID、幂等键，返回内部 `video_id`；重试可查回已接收任务 |

阶段 A 不依赖新增 API。推荐后续补齐统一在线导入能力，再执行阶段 B；不要猜一个不存在的 `/api/import` 接口并反复调用。如果 DataAgent 仅负责数据采集，交付 A 的完整数据包及 B 的接入缺口报告即可。

现有 CLI 的使用示例：

```bash
python -m omniembed ingest /path/to/video-dataset/videos \
  --meta /path/to/video-dataset/metadata/batch-20261017-001.omniembed.jsonl
```

上述命令是入库及推理操作，阶段 A 不执行。目录必须只包含本次允许提交的文件；CLI `--limit` 不替代批次清单对账。若目录包含多个历史批次，应先构造仅覆盖本批文件的输入路径集合。

CLI 在另一个进程中写入后，正在运行的 Web 不保证立即刷新其进程内索引；使用独占导入方式时，需要安排应用重新载入索引并验收。不能看到 CLI 退出就直接宣布网页已可检索。

### 模型与请求协议

本轮目标模型为 `google/embeddinggemma-2`，权重提交为 `914f7f89142e33e77833254d9c9b90c3cef7303b`，主存储维度为 768。查询与视频、文本索引必须使用一致的权重和已记录的预处理约定。需要 128、256、512 维检索时，再按应用已有约定截断并重新归一化，不把不同维度的原始结果混存进同一索引。

本地电脑的已知服务入口是 `http://192.168.31.204:8770`。`127.0.0.1` 代表调用者自己的电脑，不能用它替代跨机器服务地址。凭证通过环境变量或私有文件注入，文档和日志不包含真实 Access key。

当前仓库包含两套不同的协议，不能只因路径相同就直接替换目标地址：

| 服务实现 | 文本请求 | 视觉能力 |
|---|---|---|
| 主 Web 应用 `server.py` | `/v1/embeddings` 使用 `task: document/query`、`model`、`dimensions` | 图片以 base64 对象进入同一接口；当前主应用没有 multipart 原生视频路由 |
| 推理专用应用 `inference.py` | `/v1/embeddings` 使用 `input_type: document/query`、文档 `titles`、`dimensions` | 提供 `/v1/embeddings/image` 和接收有序帧的 `/v1/embeddings/video` |

主 Web 文本协议的最小检查请求如下。它用于验证接口能力，不是视频入库，也不证明与项目 `docs(title, description)` 的输入模板完全一致。

```json
{
  "model": "embeddinggemma-2",
  "input": ["一个介绍红烧肉做法的视频"],
  "task": "document",
  "dimensions": 768,
  "encoding_format": "float"
}
```

检索文本使用 `task: query`。不能把所有文档按默认 query 方式编码。主 Web 的 document 前缀与项目独立 titles 模板存在差异，正式入库必须明确选择模板和适配实现，不能默认两者等价。

现有 `RemoteEncoder` 面向推理专用应用的协议，要求返回模型提交、启动 ID 和向量校验信息，并拒绝普通远程 HTTP。因此健康监测已放行该内网地址，并不意味着现有业务远程编码也已兼容。实际接入必须显式处理协议、鉴权、内网传输和原生视频能力，不得绕过所有地址与模型校验来强行成功。

### 视频及文本的验收约定

- 采用当前应用抽帧策略：按视频时长和 1 fps 预算，在全片均匀取最多 8 帧，不是只取开头 8 秒。保留真实时间点、顺序、抽帧与预处理版本。
- 原生视频请求返回一个完整视频的视觉向量。仅支持单图接口时报告 `VIDEO_EMBEDDING_UNSUPPORTED`，不得改为帧向量平均并标成原生视频编码成功。
- 若最终使用推理专用应用，其当前视频接口要求 1 至 8 张同尺寸有序帧，编码总大小不超过 32 MiB，总像素不超过 32,000,000。超过时报告并按已约定的派生策略处理，原视频不改动。
- 原始文本不因服务限制丢失。正式提交前校验目标协议的正文、标题、批次及 token 上限；超限时记录 `TEXT_POLICY_REQUIRED`，待确定可追溯的分块或截断策略后处理，不依赖服务静默截断。
- embedding 结果必须校验行数与索引对应、768 维、全部数值有限、非零且接近单位范数。检查原始 float32 输出时范数容差可用 0.05，并记录是否重新归一化；不能接收 NaN、全零或错位批次。
- 每次推理保留模型 ID、确认的权重提交、维度、模态、任务或模板版本、输入指纹、预处理版本、完成时间和服务返回的 boot ID（如有）。boot ID 是运行观测信息，不应单独导致相同权重的所有视频重新编码。
- 当前应用向量表主要存模型名，尚未完整存储上述版本信息；在消费端补齐前，先保留独立处理报告，不能宣称向量已具备完整版本追溯能力。

### 重试与服务不可用

阶段 A 不请求模型健康或深探测接口，不启停 Colab，不申请或续租实例。采集不依赖模型在线；模型暂时不可用时，原始数据准备仍可完成。

阶段 B 默认单路 GPU 请求。对 429 和明确的推理 BUSY，优先遵守 `Retry-After`，否则采用有抖动的退避；可从 2 秒开始，最高 60 秒，每轮最多 5 次，耗尽后持久化为等待或失败，不能无限占用进程。5xx 和连接超时可限次重试；401/403 停止该批推理并报告鉴权错误；400/413/415/422 先修正输入，不原样无限重试。

两阶段分别保存进度。切换 local 与 Colab 必须显式选择业务配置并验证协议，不根据健康页当前 tab 或监测 enabled 状态自动切换，也不因故障自动混用不同模型提交的结果。

## 报告与错误格式

`summary.json` 至少包含批次 ID、起止时间、筛选条件、计划记录数、各终态数量、原始总字节数、可解码数、元数据缺失数量、重复与版本变化数量、实际重试次数、失败原因分布和是否达到 `DATA_READY` / `INDEX_READY`。未执行阶段 B 时，推理与入库统计为 null 或明确 `not_run`，不能写成 0 失败的已验收结果。

失败 JSONL 至少包含：

```json
{
  "batch_id": "batch-20261017-001",
  "record_id": "rec_<记录哈希>",
  "stage": "download",
  "error_code": "DOWNLOAD_TIMEOUT",
  "message": "下载超过配置时限",
  "attempts": 3,
  "retryable": true,
  "occurred_at": "2026-10-17T06:00:00Z"
}
```

建议统一错误码包括 `SOURCE_RECORD_INVALID / SOURCE_ACCESS_DENIED / DOWNLOAD_TIMEOUT / DOWNLOAD_NOT_FOUND / SIZE_LIMIT / DISK_FULL / HASH_MISMATCH / INVALID_MEDIA / UNSUPPORTED_MEDIA / METADATA_INVALID / AUTH_ERROR / INFERENCE_BUSY / VIDEO_EMBEDDING_UNSUPPORTED / TEXT_POLICY_REQUIRED / SUBMIT_UNKNOWN / EMBEDDING_INVALID / INGEST_FAILED`。

日志只输出必要的业务 ID、阶段、耗时、大小和脱敏错误。临时签名 URL、Cookie、Authorization 和真实 Access key 不进入交付清单、失败说明或日志。原始平台响应如需保留，先按业务白名单脱敏。

## 验收标准

### 阶段 A

1. 每条已发现记录均有账本结果，计划记录数可与 `DATA_READY + DATA_FAILED + 明确待处理数` 对平；不存在悄悄丢失的记录。
2. 成功主清单逐行通过 JSON Schema 校验，并使用日期格式检查；还需执行文件存在性、根目录边界、字节数、真实 SHA256、record ID 和元数据哈希的语义校验。Schema 单独通过不代表文件存在或内容一致。
3. 每条成功记录对应有效视频轨道，时长、尺寸正确，能够解码；伪装成视频的 HTML、空文件和损坏文件不能通过。
4. 主清单和 CLI 元数据清单一一对应，basename 全批唯一；空标题、中文、特殊原始文件名、缺失标签都能正确处理。
5. 重跑同一试验批不重复下载；强制中断后可恢复，`.part` 不混入成功结果；文件被替换或损坏时会重新校验并报告。
6. 改变原始视频内容和仅改变元数据两种情况能分开识别；稳定主键、内容身份及内部应用 ID 不混用。
7. 数据包迁移到另一根目录后，不修改清单也能通过路径及哈希验证；成功记录不依赖原电脑的绝对路径。
8. 报告说明实际样本覆盖和所有失败原因，交付中不含真实凭证。负向测试使用专用夹具，不破坏原始业务文件。

### 阶段 B

1. 先证明实际服务协议、模型权重和原生视频能力符合约定；若不符合，交付具体接入缺口，不把阶段 A 失败化，也不伪造推理成功。
2. 入库返回或对账得到每条 `record_id` 对应的 `omni_video_id`；任务最终进入 ready，并核对视频、来源、标题、描述、tags 和 extra 的映射。
3. 应用内部每条视频有合法 `visual` 和 `fused` 向量；有非空标题或描述时还有合法 `title` 向量。空文本视频不要求伪造文本向量。
4. 相同批次重复提交不增加相同业务内容版本的应用记录；模拟提交超时后能对账，不重复上传。
5. 至少准备 5 条人工可解释的查询，覆盖试验批中不同主题，检查预期视频是否进入 top 10，并记录排名和失败案例；此为链路验收，不预设小样本检索质量结论。
6. 模拟繁忙、鉴权错误和模型不可用时，任务能限次退避或保留待处理状态，数据包不丢失，Colab 不会被自动启用。

## DataAgent 最终交付

交付可运行的采集、下载、导出及校验入口，说明真实命令和依赖；交付阶段 A 的目录、两份逐行清单、任务账本、字段映射、校验报告和失败清单。命令参数如由 DataAgent 新增，必须明确属于采集工具，不能冒充 OmniEmbedding 现有参数。

最后给出一份结果说明：原始来源及筛选范围、实际发现及成功数、失败及待处理数、文件总大小、元数据缺失情况、重复和版本变化、数据包根目录，以及阶段 B 是否执行。阶段 B 未通过时指出缺少的导入或推理能力和相关请求结果，不要求用户从大段日志中自行猜测。

## 项目实现参考

- [命令行与元数据映射](../omniembed/cli.py)
- [入库流水线](../omniembed/pipeline.py)
- [应用存储结构](../omniembed/store.py)
- [媒体解析与抽帧](../omniembed/media.py)
- [HTTP 接口说明](API.md)
- [主应用向量协议](../omniembed/openai_api.py)
- [推理专用应用协议](../omniembed/inference.py)
- [现有远程编码适配器](../omniembed/remote_encoder.py)
