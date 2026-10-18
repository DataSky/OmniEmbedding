# OmniEmbedding

基于 Google [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)（740M，多模态，Apache 2.0）的短视频检索工具链：
**上传 → 抽帧 → 解析 → 向量化 → 存储 → 召回 → 评估**，一个进程、一个 SQLite 文件、一个网页界面。

适合用来：在自己的视频上验证"按画面搜视频""以图搜视频""标题 + 画面混合召回"的效果，或作为短视频搜索召回层的原型。

## 最新进展（2026-10-18）

- 新增带鉴权的独立推理服务和远程编码客户端，支持文本、图片和视频帧序列；已有 AutoDL 服务时，可按下方快速开始直接调用，本机无需安装 PyTorch 或下载模型。Colab 保留为短时实验入口。
- 新增本地素材检索 Demo，包含安全导入、封面 / 视频 / 文本三路向量、关键词与融合召回、Galaxy 投影、SQLite 存储与调用追踪。使用说明见 [本地 Demo](docs/LOCAL_DEMO.html)。
- 新增 AutoDL 部署、进程管理、真实推理探针、环境备份及恢复工具；[运维指南](docs/AUTODL_OPERATIONS.html)区分了停止服务、实例关机、保存镜像和数据保留规则。
- 补充 [Colab 部署](docs/COLAB.md)、[本地健康监测](docs/LOCAL_INFERENCE_HEALTH.md)、调用示例、素材清单规范及回归测试。
- 保存两份报告的源码：[系统总结](reports/embedding-system-summary/src/content/report/ReportContent.jsx)、[表示与时间信息](reports/embedding-representation-paper/src/content/report/ReportContent.jsx)。公开版本不附带私有封面和视频帧；实测数值与相关性结论的适用范围见报告正文。

仓库不包含服务密钥、模型权重、业务素材、运行数据库、原始验收记录或报告离线导出；文档中的 `data/` 链接指向使用者本机生成的文件。

## 功能

| 环节 | 做什么 |
|---|---|
| 上传 | 网页拖拽批量上传；提交视频地址（每行一个，或 CSV / JSONL 清单，可带标题、描述、标签和任意业务字段）；导入本机目录；命令行批量入库 |
| 抽帧 | 1 fps，超过上限均匀取样（默认最多 8 帧）；保存封面和每帧缩略图 |
| 解析 | ffprobe 读取时长、分辨率、帧率、编码、是否有音轨；没有 ffprobe 时退回 OpenCV；计算 SHA-256 |
| 向量化 | 每条视频三种向量：画面（帧序列）、标题（标题 + 描述，带文档前缀）、融合（两者平均）；768 维，可截断到 512 / 256 / 128 |
| 存储 | SQLite：元数据、任务状态、各阶段耗时、向量（float16）、抽帧记录、评测集。同一文件重复上传时复用向量 |
| 召回 | 五种方式：混合（BM25 + 融合向量 RRF）、融合向量、仅画面、仅标题向量、仅 BM25；以图搜视频；相似视频 |
| 评估 | 导入评测集（查询 → 标准答案，答案可写 video_id / 文件名 / URL），一次跑全部召回方式，给出 R@1/5/10、MRR@10、nDCG@10 和错例 |
| 查看 | 每条视频的完整元数据、抽帧缩略图、各类向量的统计、768 维取值分布和原始数值、向量间余弦、画面相似视频；可在线修改标题 / 描述 / 标签并自动重算向量 |

## 快速开始

以下命令在仓库根目录执行。已有 AutoDL 推理服务时，建议让本机负责应用、文件与索引，远端 GPU 负责向量化；本机无需显卡。先取得代码：

```bash
git clone https://github.com/DataSky/OmniEmbedding.git
cd OmniEmbedding
```

### 推荐：直接调用已有 AutoDL 推理服务

此路径适合 Python 3.11 / 3.12。示例客户端只安装 `httpx`，模型与 PyTorch 已在远端服务中；首次部署、镜像恢复与开关机操作见 [AutoDL 运维指南](docs/AUTODL_OPERATIONS.html)。下面准备的是**独立推理服务密钥**，无需 AutoDL 账户 Token。

```bash
python3.12 -m venv .venv-client
.venv-client/bin/python -m pip install httpx==0.28.1
mkdir -p data/autodl
install -m 600 /path/to/private/inference.key data/autodl/inference.key
```

将自己的服务基址写入 `data/autodl/endpoint.json`，不带 `/v1` 或健康检查路径。地址仅为占位示例；密钥文件须属于当前用户且权限为 `0600`。

```json
{
  "url": "https://your-inference-host.example",
  "status": "ready"
}
```

轻量示例脚本从环境变量取得地址、从文件读取密钥；本地 Demo 和 OmniIndex 则直接读取上述两个文件。

```bash
export OMNI_INFERENCE_URL="$(.venv-client/bin/python -c 'import json; print(json.load(open("data/autodl/endpoint.json"))["url"])')"
export OMNI_INFERENCE_KEY_FILE="$PWD/data/autodl/inference.key"

# 轻量就绪检查，不运行模型
.venv-client/bin/python docs/examples/embeddinggemma_client.py health

# 确认就绪后再调用；服务自动添加任务前缀，传原文即可
.venv-client/bin/python docs/examples/embeddinggemma_client.py text \
  --text '厨房里正在切菜的画面'
.venv-client/bin/python docs/examples/embeddinggemma_client.py text \
  --input-type document --text '锅中翻炒番茄与鸡蛋。' --title '番茄炒蛋'
.venv-client/bin/python docs/examples/embeddinggemma_client.py image /absolute/path/photo.jpg
.venv-client/bin/python docs/examples/embeddinggemma_client.py video \
  /absolute/path/frame01.jpg /absolute/path/frame02.jpg
```

视频输入是按时间排序、同宽高的 **1–8 张静态帧**，整段返回一条向量，MP4 需先在本机抽帧；[抽帧示例](docs/AUTODL_OPERATIONS.html#quickstart)给出了依赖与命令。文本支持 `--text-file /absolute/path/query.txt`，避免正文进入命令历史。默认输出校验后的摘要；需要完整向量时，将 `--output /absolute/path/new-result.json` 放在 `text` / `image` / `video` 之前，输出文件以 `0600` 创建且拒绝覆盖。

AutoDL 上运行的是独立推理 API：

| 调用 | 协议 |
|---|---|
| 文本 | `POST /v1/embeddings`，JSON：`input`、`input_type=query/document`、`dimensions`；仅 `document` 可带等长 `titles` |
| 图片 | `POST /v1/embeddings/image?dimensions=768`，multipart 字段 `file` |
| 视频帧 | `POST /v1/embeddings/video?dimensions=768`，重复 multipart 字段 `files` |

以上请求使用 `Authorization: Bearer`，示例客户端负责读取并发送密钥。默认 768 维，支持 128 / 256 / 512 / 768；比较的向量须使用相同模型版本和维度。**此独立 API 不接受 `model`、`task`、`encoding_format`，也没有 `/v1/models` 或 `/v1/embedding`；请使用上面的示例客户端，勿照抄下方 8770 业务服务的 OpenAI SDK 调用。** 完整约束见 [准确的调用接口](docs/AUTODL_OPERATIONS.html#api)。

`endpoint.json` 中的 `status: ready` 只是连接配置，不能证明当前在线。持续可用取决于实例保持运行、端口映射正确、模型就绪及网络连通；若健康请求返回 AutoDL nginx 的 HTML 404，先核对实例状态与当前公网映射并更新端点，不能据此认定 GPU 或模型已损坏。启动 / 恢复后可额外运行 `embeddinggemma_client.py probe` 做真实推理验收，它会占用推理槽。当前单 GPU 只有一个推理槽，多个客户端应统一调度；429 表示繁忙，增加并发并不能解决。

### 本地素材检索 Demo

沿用上节的 `endpoint.json` 与 `inference.key`。使用 Python 3.11 或 3.12，安装脚本会创建独立应用环境，无需激活 `.venv-client`：

```bash
OMNI_DEMO_SETUP_PYTHON=python3.12 bash scripts/local_demo.sh install  # 首次安装
bash scripts/local_demo.sh start    # 打开 http://127.0.0.1:8780/demo/
bash scripts/local_demo.sh status
bash scripts/local_demo.sh stop
```

素材导入与检索流程见[本地 Demo 使用说明](docs/LOCAL_DEMO.html)。关闭网页或停止本地应用不会关闭远程 GPU 实例，也不会停止 AutoDL 实例计费；实例关机需另外操作。

### AiCode 代码与文档混合检索

独立的 OmniIndex 服务扫描本机目录，将 UTF-8 源码和文本快照保存到 SQLite，提供 BM25、AutoDL 向量和加权 RRF 混合召回，以及原行号、分块、向量与排名贡献的白盒查看。快照和索引保存在本机，需要编码的代码分块与查询会发送到配置的远端服务。使用 Python 3.11 / 3.12：

```bash
python3.12 -m venv .venv-code-search
source .venv-code-search/bin/activate
python -m pip install -r requirements-code-search.txt
python scripts/run_code_search.py --root ~/AiCode --sync  # http://127.0.0.1:8790/
# 仅建立本地快照与 BM25，不依赖 GPU：改用 --scan-only
```

沿用上节的连接文件；需要显式指定时，为启动命令添加 `--endpoint-file ./data/autodl/endpoint.json --key-file ./data/autodl/inference.key`。任务支持暂停与断点续跑；PDF、Office、图片、依赖、构建产物和敏感文件不在当前正文解析范围内。完整设计、连接配置、API、排除规则与规模边界见 [OmniIndex 使用说明](docs/CODE_SEARCH.md)。

质量评测会固定语料、题集与算法，分别记录来源证据、编码覆盖、分词、候选、排序和Recall/MRR。网页“质量评测”页可查看各轮日志；自动研究只用开发集选参，随后用留出集检查门槛，未通过则保留现有参数。入口与实验范围见 [检索研究说明](docs/CODE_SEARCH_RESEARCH.md)。

最新版完整方案、端到端流程、召回机制、优化实测、运维与审计示例见 [OmniIndex 完整方案与实践指南](docs/OMNIINDEX_SOLUTION.md)。
本次针对 AiCode 目录的详细执行过程、处理记录与当前验收结果见 [AiCode 执行汇总](docs/AICODE_EXECUTION_SUMMARY.md)。

### 本机加载模型（可选）

没有远程服务或需要本机独立运行时，使用 Python 3.12+ 安装完整环境。NVIDIA GPU 可加速，CPU 也能运行但向量化更慢。以下为 CUDA 12.8 示例，PyTorch 安装需匹配自己的平台与驱动：

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
OMNI_ENCODER_BACKEND=local python -m omniembed serve  # http://127.0.0.1:8770
```

此完整环境首次向量化会下载模型（约 1.5 GB）；国内网络可设置 `HF_ENDPOINT=https://hf-mirror.com`。已有业务应用也可通过下方环境变量切换 AutoDL 后重启：

```bash
unset OMNI_INFERENCE_URL  # 由端点文件提供地址，避免旧环境变量覆盖
export OMNI_ENCODER_BACKEND=remote
export OMNI_INFERENCE_ENDPOINT_FILE="$PWD/data/autodl/endpoint.json"
export OMNI_INFERENCE_KEY_FILE="$PWD/data/autodl/inference.key"
python -m omniembed serve
```

这些是本机业务进程的配置；AutoDL 推理进程仍用 `local` 加载 GPU 模型。远程请求失败会明确报错，不会回退到本机下载模型。仅需轻量远程应用时直接使用上面的 Demo 或 OmniIndex。

### Colab 推理服务（短时实验）

Colab 使用同一套独立推理协议；业务应用设置 `OMNI_ENCODER_BACKEND=remote`，并将 `OMNI_INFERENCE_ENDPOINT_FILE` / `OMNI_INFERENCE_KEY_FILE` 指向自己的 Colab 连接文件：
[部署与健康监测说明](docs/COLAB.md)、[自包含 Colab Notebook](notebooks/Colab_EmbeddingGemma2.ipynb)。
现有网页的「服务健康」页提供本地电脑 / Colab 双页签及独立监测开关，默认本地开启、Colab 停用。
汇总最近 3 小时的探测日志；停用后灰显并保留历史，读取页面不触发 GPU 推理。
监控在 Colab 外由网页后端运行，区分存活、就绪和真实推理；默认实验窗口为 2 小时。
健康探针不能延长 Colab 租期，长期服务需要稳定的 GPU 宿主。

### 命令行

```bash
python -m omniembed ingest D:/videos --meta meta.jsonl     # 目录批量入库；meta.jsonl 可选：{"file": "a.mp4", "title": "...", "tags": [...]}
python -m omniembed ingest-urls urls.csv                   # CSV（url,title,description,tags,...）/ JSONL / 每行一个 URL
python -m omniembed search "厨房里做菜" --mode visual
python -m omniembed eval evalset.jsonl --modes hybrid,visual --out result.json
python -m omniembed stats
python -m omniembed watchdog                              # 启动服务并守护，卡死或退出时自动重启
```

评测集格式（JSON 数组或 JSONL）：

```json
{"query": "红烧肉做法", "gold": ["a1b2c3d4e5f60789", "hongshaorou.mp4"]}
```

## 配置

全部用环境变量，见 [`.env.example`](.env.example)。常用的：

| 变量 | 默认 | 说明 |
|---|---|---|
| `OMNI_DATA_DIR` | `./data` | 数据库、视频、缩略图都在这里，建议指到大容量盘 |
| `OMNI_ENCODER_BACKEND` | `local` | `remote` 调用 AutoDL / Colab 独立推理服务，失败不会回退本机 |
| `OMNI_INFERENCE_ENDPOINT_FILE` | `data/colab/endpoint.json` | 业务远程编码器的默认路径；接 AutoDL 时设为 `data/autodl/endpoint.json`；Demo / OmniIndex 默认使用 AutoDL 路径 |
| `OMNI_INFERENCE_KEY_FILE` | `data/colab/inference.key` | 独立推理密钥；接 AutoDL 时设为 `data/autodl/inference.key`，须属于当前用户且权限为 `0600` |
| `OMNI_INFERENCE_TIMEOUT_S` | `120` | 业务远程编码器的请求与重试时限，秒 |
| `OMNI_MAX_FRAMES` | `8` | 每条视频最多抽几帧 |
| `OMNI_HOST` / `OMNI_PORT` | `127.0.0.1` / `8770` | 监听地址 |
| `OMNI_API_KEY` | 空 | 设置后所有写操作要带 `X-API-Key` 请求头；**监听非本机地址时必须设置** |
| `OMNI_MAX_UPLOAD_MB` | `500` | 单个视频上限（上传和 URL 下载都适用） |
| `OMNI_ALLOW_PRIVATE_URLS` | `0` | 默认拒绝下载内网 / 回环地址，防止被用来探测内网 |

## 业务服务作为向量服务使用（8770，OpenAI 兼容）

本节针对 `python -m omniembed serve` 启动的业务服务；它可用本机模型，也可作为调用 AutoDL 的本机后端。**AutoDL 独立推理服务请使用上方协议**，两者的路由、请求字段与密钥不同。此节的 `OMNI_API_KEY` 是业务服务密钥，远端推理密钥由后端从文件读取。

`POST /v1/embeddings`（别名 `/v1/embedding`）与 `GET /v1/models` 和 OpenAI Embeddings 接口格式一致，
可直接在 openai SDK、LangChain、LlamaIndex、Dify 等里当 "OpenAI Embeddings" 填入：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8770/v1", api_key="<OMNI_API_KEY>")
r = client.embeddings.create(model="embeddinggemma-2:query", input=["红烧肉做法", "周末露营"], dimensions=256)
print(len(r.data[0].embedding))   # 256
```

EmbeddingGemma 2 要按任务加前缀，通用客户端不会区分查询和文档，所以用下面任一方式指定（优先级从高到低）：

| 方式 | 写法 |
|---|---|
| 请求体扩展字段 | `"task": "query"`，可选 `query` / `document` / `similarity` / `classification` / `clustering` / `code` / `qa` / `fact` / `none` |
| 模型名后缀 | `"model": "embeddinggemma-2:query"` 或 `:document`（只能填模型名的客户端用这个） |
| 默认 | 环境变量 `OMNI_EMBED_DEFAULT_TASK`，默认 `query` |

`task` 和下面的图片输入是仓库扩展；用 OpenAI SDK 传 `task` 时放入 `extra_body={"task": "document"}`，或直接使用模型名后缀。不要把图片输入当作通用 OpenAI Embeddings 客户端都支持的能力。

做检索时查询用 `:query`、入库文档用 `:document` 效果最好。在 VideoRetrieval（10 万优酷标题）上实测 nDCG@10：
查询 `:query` + 文档 `:document` 57.9；两边都不区分时，都用 `query` 53.3、都用 `document` 50.0、都用 `similarity` 44.6、都不加前缀 38.9。
所以不区分时默认 `query`；能区分就一定要区分。
其他参数：`dimensions`（768 / 512 / 256 / 128，Matryoshka 截断后重新归一化）、`encoding_format: "base64"`（float32 小端）。
图片输入：`{"input": [{"image": "data:image/jpeg;base64,..."}]}`，与文本在同一向量空间。
设置了 `OMNI_API_KEY` 时需要 `Authorization: Bearer <key>`（或 `X-API-Key`）。

## 实测数据

都在一台 RTX 4090（48 GB）+ Windows 11 上测得，bf16。完整评测报告（含文本检索、多模态、量化、短视频专项）见 [docs/EVALUATION.md](docs/EVALUATION.md)。

**端到端（本仓库流水线）**：Youku-mPLUG 检索验证集抽 200 段真实优酷短视频（中位时长约 16 秒），入库时不带标题，
用人工写的中文画面描述做查询、在这 200 段里找对应视频：

| 召回方式 | 维度 | R@1 | R@5 | R@10 | nDCG@10 |
|---|---|---|---|---|---|
| 仅画面 | 768 | 69.5 | 89.5 | 92.5 | 82.2 |
| 仅画面 | 256 | 68.0 | 86.5 | 92.0 | 80.0 |

入库速度 3.11 段/秒（200 段 64 秒，含解析、抽帧、缩略图、向量化、写库），每段平均：解析 0.03 s、抽帧 0.11 s、向量化 0.16 s。
复现：`python benchmarks/youku_mplug.py --manifest manifest_val.jsonl --n 200`。

**几个对使用方式有影响的结论**（来自评测报告）：

- 抽帧 8 / 16 / 32 帧的检索精度相差不到 1 分，速度是 5.6 / 3.1 / 2.1 段/秒，所以默认 8 帧。
- 视频向量至少用 256 维：256 维比 768 维低 3–4 分，128 维低约 16 分。
- 文本检索必须带任务前缀（本项目已内置），不带会让 nDCG@10 掉约 19 分。
- 标题类查询里有大量专名（剧名、人名），BM25 + 向量混合比纯向量高约 4 分 nDCG@10，所以默认用混合。
- 音频检索偏弱（Clotho 文→音频 R@1 7.9），本项目视频向量只用画面，不加载音频编码器。
- 只能用 bf16 / fp32，官方说明 fp16 会出 NaN。

## 结构

```
omniembed/
  config.py      环境变量配置
  store.py       SQLite：videos / vectors / frames / evalsets
  media.py       URL 安全下载、ffprobe 解析、OpenCV 抽帧、缩略图
  encoder.py     EmbeddingGemma 2 封装（懒加载、前缀、MRL 截断）
  pipeline.py    后台入库流水线（状态机 + 重试 + 去重）
  bm25.py        稀疏矩阵 BM25（jieba 分词）
  retrieval.py   多路召回 + RRF、以图搜、相似视频；数据变化后自动重建索引
  evaluation.py  评测集解析与 R@k / MRR / nDCG 计算
  health.py      推理锁与健康状态（就绪 / 降级 / 卡死）
  health_api.py  /health/live、/health/ready、/health/inference 与启动预热
  openai_api.py  OpenAI 兼容 /v1/embeddings
  watchdog.py    看门狗
  server.py      FastAPI 接口
  cli.py         命令行
  web/           网页界面（无构建步骤的原生 JS）
benchmarks/      端到端基准脚本
tests/           pytest，用假编码器，不需要 GPU 和模型
```

接口列表见 [docs/API.md](docs/API.md)，健康检查与看门狗见 [docs/HEALTH.md](docs/HEALTH.md)。运行测试：`pip install -r requirements-dev.txt && pytest`。

## 规模与限制

- 召回是精确内积（numpy），几万条视频以内单次打分在毫秒级。更大规模请把 `retrieval.py` 的打分换成 FAISS / Milvus 等 ANN 索引，存储层不用改。
- 后台流水线默认单线程（`OMNI_WORKERS`），GPU 编码是瓶颈，多开线程收益有限。
- URL 下载只支持直链（http / https 指向视频文件），不解析视频网站的播放页。
- 测试和基准里用到的公开数据集（Youku-mPLUG、MSR-VTT 等）版权归原作者，仅供研究，不随仓库分发。

## 许可证

代码：Apache License 2.0。模型 EmbeddingGemma 2 为 Apache 2.0，使用时还需遵守 [Gemma 禁止用途政策](https://ai.google.dev/gemma/prohibited_use_policy)。
