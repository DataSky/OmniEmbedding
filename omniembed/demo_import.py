"""Read untrusted video export data without executing or serving its source files.

Only structured metadata and covers/videos with approved extensions are copied.
Returned media paths always belong to the caller's destination. Metadata remains
untrusted text: consumers must escape it and must never render it as HTML/SQL.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile
import unicodedata

MAX_ENTRIES = 20_000
MAX_RECORDS = 20_000
MAX_TOTAL_BYTES = 16 * 1024**3
MAX_MEMBER_BYTES = 512 * 1024**2
MAX_METADATA_BYTES = 16 * 1024**2
MAX_COVER_BYTES = 32 * 1024**2
CHUNK = 1024**2
COVER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
CLICK_WARNING = "原始 click_cnt / click_dv 等为行为点击信息，不是人工相关性标签，不能直接作为独立检索评测真值。"


class ImportValidationError(ValueError):
    """Unsafe, ambiguous or malformed data; no source instructions are executed."""


def _cancel(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("导入已取消；未发布的临时文件已清理。")


def _progress(callback, phase, completed, total, message):
    if callback is not None:
        callback({"phase": phase, "completed": completed, "total": total, "message": message})


def _name(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ImportValidationError("素材路径为空或过长。")
    value = unicodedata.normalize("NFC", value)
    if "\\" in value or ":" in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ImportValidationError("素材路径包含不安全字符。")
    if value.startswith("/") or ".." in value.split("/"):
        raise ImportValidationError("拒绝绝对路径或越界路径。")
    while value.startswith("./"):
        value = value[2:]
    value = value.rstrip("/")
    path = PurePosixPath(value)
    if not path.parts or len(path.parts) > 32 or path.is_absolute() or any(p in {".", ".."} for p in path.parts):
        raise ImportValidationError("素材路径不合法。")
    return path.as_posix()


def _signature(path):
    value = path.lstat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _json(content):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ImportValidationError("JSON 包含重复字段。")
            result[key] = value
        return result
    try:
        return json.loads(content, object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ImportValidationError("JSON 包含非有限数值。")))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ImportValidationError("metadata.json 无法解析。") from error


def _text(value, limit=300):
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]{0,2000}>", "", text)
    text = " ".join("".join(c for c in text if ord(c) >= 32 and ord(c) != 127).split())
    return text[:limit]


def _keywords(record):
    value = record.get("keywords", [])
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            decoded = None
        value = decoded if isinstance(decoded, list) else re.split(r"[|;,]", value)
    if not isinstance(value, list):
        value = []
    result = []
    top = _text(record.get("keyword_top"))
    for item in ([top] if top else []) + value:
        keyword = _text(item)
        if keyword and keyword not in result:
            result.append(keyword)
    return result[:1000]


class _Source:
    def __init__(self, path, cancelled=None):
        self.original_path = Path(path).expanduser().absolute()
        if self.original_path.is_symlink():
            raise ImportValidationError("输入路径不能是符号链接。")
        self.path = self.original_path.resolve(strict=True)
        self.type = "directory" if self.path.is_dir() else "tar"
        self.cancelled = cancelled
        self.tar = None
        self.files = {}
        self.signatures = {}
        self.names = set()
        self.total_bytes = 0
        self.source_signature = _signature(self.path)

    def __enter__(self):
        try:
            if self.type == "directory":
                self._directory()
            else:
                if not stat.S_ISREG(self.path.stat().st_mode):
                    raise ImportValidationError("输入必须是 tar 文件或已解包目录。")
                try:
                    self.tar = tarfile.open(self.path, "r:*")
                    for member in self.tar:
                        _cancel(self.cancelled)
                        name = _name(member.name)
                        if member.issym() or member.islnk() or member.issparse() or not (member.isdir() or member.isfile()):
                            raise ImportValidationError("包内包含符号链接、硬链接、设备或其他特殊成员。")
                        self._register(name, member.size if member.isfile() else 0)
                        if member.isfile():
                            self.files[name] = member
                except (tarfile.TarError, OSError) as error:
                    raise ImportValidationError("无法安全读取 tar 文件。") from error
            self._select_metadata()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.tar is not None:
            self.tar.close()

    def _register(self, name, size):
        if name in self.names:
            raise ImportValidationError("包内存在重复路径。")
        self.names.add(name)
        if len(self.names) > MAX_ENTRIES or size < 0 or size > MAX_MEMBER_BYTES:
            raise ImportValidationError("包内成员数量或单文件大小超过限制。")
        self.total_bytes += size
        if self.total_bytes > MAX_TOTAL_BYTES:
            raise ImportValidationError("包内总文件大小超过限制。")

    def _directory(self):
        for current, directories, files in os.walk(self.path, followlinks=False):
            for child in sorted(directories + files):
                _cancel(self.cancelled)
                path = Path(current) / child
                value = path.lstat()
                if not (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)) or (stat.S_ISREG(value.st_mode) and value.st_nlink > 1):
                    raise ImportValidationError("目录输入包含链接或特殊文件。")
                name = _name(path.relative_to(self.path).as_posix())
                self._register(name, value.st_size if stat.S_ISREG(value.st_mode) else 0)
                if stat.S_ISREG(value.st_mode):
                    self.files[name] = path
                    self.signatures[name] = _signature(path)

    def _select_metadata(self):
        candidates = [name for name in self.files if PurePosixPath(name).name == "metadata.json" and len(PurePosixPath(name).parts) <= 2]
        if not candidates:
            candidates = [name for name in self.files if PurePosixPath(name).name == "metadata.csv" and len(PurePosixPath(name).parts) <= 2]
        if len(candidates) != 1:
            raise ImportValidationError("需要且只能有一份明确的 metadata.json（或 metadata.csv）。")
        for name in self.names:
            if any(parent.as_posix() in self.files for parent in PurePosixPath(name).parents):
                raise ImportValidationError("成员路径的父路径被其他文件占用。")
        self.metadata_name = candidates[0]
        parent = PurePosixPath(self.metadata_name).parent
        self.prefix = "" if str(parent) == "." else parent.as_posix() + "/"
        selected = {}
        for name, member in self.files.items():
            if not name.startswith(self.prefix):
                continue
            relative = name[len(self.prefix):]
            suffix = PurePosixPath(name).suffix.lower()
            kind = ("metadata" if relative in {"metadata.json", "metadata.csv"}
                    else "cover" if relative.startswith("covers/") and suffix in COVER_EXTENSIONS
                    else "video" if relative.startswith("videos/") and suffix in VIDEO_EXTENSIONS else None)
            if kind:
                size = member.size if self.tar else self.signatures[name][3]
                limit = MAX_METADATA_BYTES if kind == "metadata" else MAX_COVER_BYTES if kind == "cover" else MAX_MEMBER_BYTES
                if size > limit:
                    raise ImportValidationError("结构化元数据或媒体文件超过对应大小限制。")
                selected[name] = kind
        self.selected = selected

    def open(self, name):
        _cancel(self.cancelled)
        if name not in self.selected:
            raise ImportValidationError("尝试读取白名单外文件。")
        if self.tar:
            return self.tar.extractfile(self.files[name])
        path = self.files[name]
        if _signature(path) != self.signatures[name]:
            raise ImportValidationError("导入时源文件发生变化。")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        stream = os.fdopen(descriptor, "rb")
        value = os.fstat(stream.fileno())
        if (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns) != self.signatures[name]:
            stream.close()
            raise ImportValidationError("源文件被替换。")
        return stream

    def unchanged(self):
        if _signature(self.path) != self.source_signature:
            raise ImportValidationError("导入期间源包或源目录发生变化。")
        if self.type == "directory":
            for name in self.selected:
                if _signature(self.files[name]) != self.signatures[name]:
                    raise ImportValidationError("导入期间源文件发生变化。")

    def metadata(self):
        with self.open(self.metadata_name) as stream:
            content = stream.read(MAX_METADATA_BYTES + 1)
        if len(content) > MAX_METADATA_BYTES:
            raise ImportValidationError("元数据过大。")
        if self.metadata_name.endswith(".json"):
            value = _json(content)
        else:
            try:
                reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
                if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ImportValidationError("CSV 字段重复或为空。")
                records = []
                for row in reader:
                    if None in row:
                        raise ImportValidationError("CSV 行的字段数量不匹配。")
                    for field in ["keywords", "kw_detail", "tab_names", "related_tabs"]:
                        try:
                            if row.get(field):
                                row[field] = json.loads(row[field])
                        except (ValueError, RecursionError):
                            pass
                    records.append(row)
                    if len(records) > MAX_RECORDS:
                        raise ImportValidationError("元数据记录数量过多。")
                value = {"records": records}
            except (UnicodeDecodeError, csv.Error) as error:
                raise ImportValidationError("metadata.csv 无法解析。") from error
        if isinstance(value, list):
            records, source_metadata = value, {}
        elif isinstance(value, dict) and isinstance(value.get("records"), list):
            records, source_metadata = value["records"], {k: v for k, v in value.items() if k != "records"}
        else:
            raise ImportValidationError("元数据需要 records 数组。")
        if len(records) > MAX_RECORDS or any(not isinstance(row, dict) for row in records):
            raise ImportValidationError("元数据记录过多或记录不是对象。")
        return records, source_metadata

    def reference(self, value, kind):
        if not value:
            return None, None
        try:
            name = _name(value)
        except ImportValidationError:
            return None, "已拒绝不安全的媒体引用。"
        if name in self.selected and self.selected[name] == kind:
            return name, None
        name = self.prefix + name
        if self.selected.get(name) == kind:
            return name, None
        return None, "元数据引用的媒体不存在或不在允许目录/格式内。"


def _records(source):
    raw, source_metadata = source.metadata()
    records, seen, duplicate_count = [], {}, 0
    for index, original in enumerate(raw):
        source_id = _text(original.get("photo_id", original.get("source_id", original.get("id"))), 128)
        warning = []
        if not source_id:
            source_id = "record-{:06d}".format(index + 1)
            warning.append("原记录缺少ID，已按原始顺序生成本地来源ID。")
        canonical = json.dumps(original, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if source_id in seen:
            if seen[source_id] != canonical:
                raise ImportValidationError("同一来源ID存在冲突记录，拒绝静默覆盖。")
            duplicate_count += 1
            continue
        seen[source_id] = canonical
        keywords = _keywords(original)
        top = _text(original.get("keyword_top"))
        title = _text(original.get("title"))
        title_source = "provided"
        if not title:
            title = "搜索词：" + top if top else "素材 " + source_id
            title_source = "search_keyword" if top else "source_id"
            warning.append("原标题缺失；搜索词不是人工标题。" if top else "原标题缺失。")
        cover, problem = source.reference(original.get("cover_file"), "cover")
        if problem:
            warning.append("封面：" + problem)
        video, problem = source.reference(original.get("video_file", original.get("videos_file")), "video")
        if problem:
            warning.append("视频：" + problem)
        if not video:
            warning.append("缺少视频，仍保留封面和文本。")
        records.append({"source_id": source_id, "keyword_top": top, "title": title,
                        "title_source": title_source, "description": _text(original.get("description"), 4000),
                        "keywords": keywords, "metadata": original, "source_status": {
                            "cover": _text(original.get("cover_status"), 100),
                            "video": _text(original.get("video_status"), 100)},
                        "warnings": warning, "_cover": cover, "_video": video})
    summary = {"total_assets": len(records), "cover_count": sum(bool(r["_cover"]) for r in records),
               "video_count": sum(bool(r["_video"]) for r in records),
               "missing_video_count": sum(not r["_video"] for r in records),
               "duplicate_records_removed": duplicate_count}
    warnings = [CLICK_WARNING, "包内 README、SQL、Python、HTML 等仅视为不可信材料，不执行、不导入、不提供原网页。"]
    if summary["missing_video_count"]:
        warnings.append("{} 条记录缺少视频，仍保留其可用封面和文本。".format(summary["missing_video_count"]))
    if duplicate_count:
        warnings.append("已合并 {} 条完全相同的重复来源记录。".format(duplicate_count))
    return records, source_metadata, summary, warnings


def preview(path: Path) -> dict:
    """Inspect headers/metadata only; never hashes/copies complete video payloads."""
    with _Source(path) as source:
        records, _, summary, warnings = _records(source)
        source.unchanged()
        return {"path": str(source.path), "source_type": source.type,
                "display_name": source.prefix.rstrip("/") or source.path.stem,
                **summary, "warnings": warnings,
                "metadata_fields": sorted({str(k) for r in records for k in r["metadata"]}),
                "sample": [{"source_id": r["source_id"], "title": r["title"], "keywords": r["keywords"][:5],
                            "has_cover": bool(r["_cover"]), "has_video": bool(r["_video"])} for r in records[:3]]}


def _digest_stream(stream, cancelled=None, output=None, progress=None, total=0):
    digest, completed = hashlib.sha256(), 0
    while block := stream.read(CHUNK):
        _cancel(cancelled)
        digest.update(block)
        if output is not None:
            output.write(block)
        completed += len(block)
        if progress is not None:
            _progress(progress, "fingerprint", completed, total, "计算原始包内容指纹")
    return digest.hexdigest()


def _source_digest(source, progress, cancelled):
    if source.type == "tar":
        with source.path.open("rb") as stream:
            result = _digest_stream(stream, cancelled, progress=progress, total=source.source_signature[3])
    else:
        digest = hashlib.sha256()
        for index, name in enumerate(sorted(source.selected)):
            _cancel(cancelled)
            with source.open(name) as stream:
                file_hash = _digest_stream(stream, cancelled)
            digest.update(name.encode() + b"\0" + file_hash.encode() + b"\0")
            _progress(progress, "fingerprint", index + 1, len(source.selected), "计算所选目录内容指纹")
        result = digest.hexdigest()
    source.unchanged()
    return result


def _magic(header, suffix):
    if suffix in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")
    if suffix == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix == ".webp":
        return header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    if suffix in {".mp4", ".mov", ".m4v"}:
        return header[4:8] in {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"}
    if suffix in {".webm", ".mkv"}:
        return header.startswith(b"\x1aE\xdf\xa3")
    return suffix == ".avi" and header.startswith(b"RIFF") and header[8:12] == b"AVI "


def _cache_result(target, source_hash, cancelled):
    if target.is_symlink() or not target.is_dir():
        raise ImportValidationError("内容目录已存在但不安全，拒绝覆盖。")
    manifest = target / "import-manifest.json"
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > MAX_METADATA_BYTES * 4:
        raise ImportValidationError("已存在内容目录没有有效导入清单，拒绝覆盖。")
    result = _json(manifest.read_bytes())
    if (not isinstance(result, dict) or not isinstance(result.get("dataset"), dict)
            or not isinstance(result.get("assets"), list)
            or result["dataset"].get("source_sha256") != source_hash):
        raise ImportValidationError("已有导入指纹不匹配。")
    checked = set()
    for asset in result.get("assets", []):
        for kind in ["cover", "video"]:
            value = asset.get(kind + "_path")
            if not value:
                continue
            path = Path(value)
            if path.is_symlink() or target.resolve() not in path.resolve().parents or not path.is_file():
                raise ImportValidationError("已有媒体路径不安全或缺失。")
            if value not in checked:
                with path.open("rb") as stream:
                    digest = _digest_stream(stream, cancelled)
                if digest != asset.get(kind + "_sha256"):
                    raise ImportValidationError("已有媒体内容校验失败。")
                checked.add(value)
    return result


def materialize(path: Path, dest: Path, progress=None, cancelled=None) -> dict:
    """Copy selected data into dest/<source SHA>; never execute a package member.

    progress receives one dict with phase/completed/total/message. Cancellation
    raises InterruptedError. Copies are private, ordinary files, never hardlinks
    to external originals. Reimports reuse verified content instead of copying it.
    """
    _cancel(cancelled)
    destination = Path(dest).expanduser().resolve()
    with _Source(path, cancelled) as source:
        if source.type == "directory" and (destination == source.path or source.path in destination.parents):
            raise ImportValidationError("导入输出不能放在原始目录内。")
        records, source_metadata, summary, warnings = _records(source)
        _progress(progress, "inspect", len(records), len(records), "安全检查和元数据解析完成")
        source_hash = _source_digest(source, progress, cancelled)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source_hash
        if os.path.lexists(target):
            result = _cache_result(target, source_hash, cancelled)
            _progress(progress, "complete", len(records), len(records), "复用已校验的相同内容")
            return result
        lock = destination / ("." + source_hash + ".lock")
        try:
            lock.mkdir(mode=0o700)
        except FileExistsError as error:
            raise ImportValidationError("同一内容正在导入；请等待当前任务完成。") from error
        staging = None
        try:
            staging = Path(tempfile.mkdtemp(prefix="." + source_hash[:12] + ".partial-", dir=destination))
            media, content_paths = {}, {}
            required = sorted({name for r in records for name in [r["_cover"], r["_video"]] if name})
            for index, name in enumerate(required):
                _cancel(cancelled)
                suffix = PurePosixPath(name).suffix.lower()
                temporary = staging / (".media-" + str(index))
                with source.open(name) as incoming, temporary.open("xb") as output:
                    os.chmod(temporary, 0o600)
                    header = incoming.read(16)
                    if not _magic(header, suffix):
                        raise ImportValidationError("媒体头与允许格式不匹配；拒绝伪装的网页或其他内容。")
                    digest = hashlib.sha256(header)
                    output.write(header)
                    while block := incoming.read(CHUNK):
                        _cancel(cancelled)
                        digest.update(block)
                        output.write(block)
                sha = digest.hexdigest()
                relative = content_paths.setdefault(sha, "media/" + sha + suffix)
                final = staging / relative
                final.parent.mkdir(mode=0o700, exist_ok=True)
                if final.exists():
                    temporary.unlink()
                else:
                    temporary.rename(final)
                media[name] = {"path": str(target / relative), "sha256": sha}
                _progress(progress, "copy", index + 1, len(required), "复制并校验允许的媒体文件")
            for name, kind in source.selected.items():
                if kind == "metadata":
                    folder = staging / "metadata"
                    folder.mkdir(mode=0o700, exist_ok=True)
                    with source.open(name) as incoming, (folder / PurePosixPath(name).name).open("xb") as output:
                        os.chmod(output.name, 0o600)
                        shutil.copyfileobj(incoming, output, CHUNK)
            assets = []
            for record in records:
                asset = {k: v for k, v in record.items() if not k.startswith("_")}
                asset["id"] = "asset_" + hashlib.sha256((source_hash + "\0" + record["source_id"]).encode()).hexdigest()[:24]
                for kind in ["cover", "video"]:
                    item = media.get(record["_" + kind], {})
                    asset[kind + "_path"] = item.get("path")
                    asset[kind + "_sha256"] = item.get("sha256")
                assets.append(asset)
            result = {"dataset": {"id": "ds_" + source_hash[:20], "name": source.prefix.rstrip("/") or source.path.stem,
                                  "source_path": str(source.path), "source_sha256": source_hash,
                                  "source_type": source.type, "summary": {**summary, "warnings": warnings},
                                  "source_metadata": source_metadata}, "assets": assets}
            with (staging / "import-manifest.json").open("x", encoding="utf-8") as output:
                os.chmod(output.name, 0o600)
                json.dump(result, output, ensure_ascii=False, allow_nan=False, indent=2)
            source.unchanged()
            _cancel(cancelled)
            if os.path.lexists(target):
                raise ImportValidationError("导入目录在写入期间出现；拒绝覆盖。")
            staging.rename(target)
            _progress(progress, "complete", len(records), len(records), "安全导入完成；未执行包内代码")
            return result
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)
            lock.rmdir()
