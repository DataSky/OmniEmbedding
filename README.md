# OmniEmbedding

基于 Google [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)（740M，多模态，Apache 2.0）的短视频检索工具链：
**上传 → 抽帧 → 解析 → 向量化 → 存储 → 召回 → 评估**，一个进程、一个 SQLite 文件、一个网页界面。

适合用来：在自己的视频上验证"按画面搜视频""以图搜视频""标题 + 画面混合召回"的效果，或作为短视频搜索召回层的原型。

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

需要 Python 3.10+，有 NVIDIA 显卡更快（只用 CPU 也能跑，向量化会慢很多）。

```bash
git clone https://github.com/DataSky/OmniEmbedding.git
cd OmniEmbedding
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # 按自己的 CUDA 版本选
pip install -r requirements.txt
python -m omniembed serve        # 打开 http://127.0.0.1:8770
```

首次向量化时会从 Hugging Face 下载模型（约 1.5 GB）。国内网络可设置 `HF_ENDPOINT=https://hf-mirror.com`。

### 命令行

```bash
python -m omniembed ingest D:/videos --meta meta.jsonl     # 目录批量入库；meta.jsonl 可选：{"file": "a.mp4", "title": "...", "tags": [...]}
python -m omniembed ingest-urls urls.csv                   # CSV（url,title,description,tags,...）/ JSONL / 每行一个 URL
python -m omniembed search "厨房里做菜" --mode visual
python -m omniembed eval evalset.jsonl --modes hybrid,visual --out result.json
python -m omniembed stats
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
| `OMNI_MAX_FRAMES` | `8` | 每条视频最多抽几帧 |
| `OMNI_HOST` / `OMNI_PORT` | `127.0.0.1` / `8770` | 监听地址 |
| `OMNI_API_KEY` | 空 | 设置后所有写操作要带 `X-API-Key` 请求头；**监听非本机地址时必须设置** |
| `OMNI_MAX_UPLOAD_MB` | `500` | 单个视频上限（上传和 URL 下载都适用） |
| `OMNI_ALLOW_PRIVATE_URLS` | `0` | 默认拒绝下载内网 / 回环地址，防止被用来探测内网 |

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
  server.py      FastAPI 接口
  cli.py         命令行
  web/           网页界面（无构建步骤的原生 JS）
benchmarks/      端到端基准脚本
tests/           pytest，用假编码器，不需要 GPU 和模型
```

接口列表见 [docs/API.md](docs/API.md)。运行测试：`pip install -r requirements-dev.txt && pytest`。

## 规模与限制

- 召回是精确内积（numpy），几万条视频以内单次打分在毫秒级。更大规模请把 `retrieval.py` 的打分换成 FAISS / Milvus 等 ANN 索引，存储层不用改。
- 后台流水线默认单线程（`OMNI_WORKERS`），GPU 编码是瓶颈，多开线程收益有限。
- URL 下载只支持直链（http / https 指向视频文件），不解析视频网站的播放页。
- 测试和基准里用到的公开数据集（Youku-mPLUG、MSR-VTT 等）版权归原作者，仅供研究，不随仓库分发。

## 许可证

代码：Apache License 2.0。模型 EmbeddingGemma 2 为 Apache 2.0，使用时还需遵守 [Gemma 禁止用途政策](https://ai.google.dev/gemma/prohibited_use_policy)。
