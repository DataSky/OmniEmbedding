"""运行配置：全部可用环境变量覆盖，默认值适合单机 GPU。"""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv():
    """读取项目根目录的 .env（KEY=VALUE，# 注释）。已存在的环境变量优先，不被覆盖。"""
    p = Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


@dataclass
class Config:
    # 数据目录：数据库、视频、封面都在这里。默认在项目目录下的 data/，可指到大容量盘
    data_dir: Path = field(default_factory=lambda: Path(_env("OMNI_DATA_DIR", Path(__file__).resolve().parent.parent / "data")))
    model_id: str = field(default_factory=lambda: _env("OMNI_MODEL", "google/embeddinggemma-2"))
    device: str = field(default_factory=lambda: _env("OMNI_DEVICE", "auto"))
    # 抽帧：评测显示 1fps、最多 8 帧与 16/32 帧精度相差 <1 分，速度快 1.8–2.7 倍
    fps: float = field(default_factory=lambda: _env("OMNI_FPS", 1.0, float))
    max_frames: int = field(default_factory=lambda: _env("OMNI_MAX_FRAMES", 8, int))
    # 限制：单个视频大小、URL 下载大小与超时、单次批量条数
    max_upload_mb: int = field(default_factory=lambda: _env("OMNI_MAX_UPLOAD_MB", 500, int))
    url_timeout_s: int = field(default_factory=lambda: _env("OMNI_URL_TIMEOUT", 60, int))
    max_batch: int = field(default_factory=lambda: _env("OMNI_MAX_BATCH", 200, int))
    allow_private_urls: bool = field(default_factory=lambda: _env("OMNI_ALLOW_PRIVATE_URLS", "0") == "1")
    # 服务
    host: str = field(default_factory=lambda: _env("OMNI_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env("OMNI_PORT", 8770, int))
    api_key: str = field(default_factory=lambda: _env("OMNI_API_KEY", ""))
    workers: int = field(default_factory=lambda: _env("OMNI_WORKERS", 1, int))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "omni.db"

    @property
    def video_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def cover_dir(self) -> Path:
        return self.data_dir / "covers"

    def ensure_dirs(self):
        for d in (self.data_dir, self.video_dir, self.cover_dir):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
