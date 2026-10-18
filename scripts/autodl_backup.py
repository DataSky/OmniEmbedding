#!/usr/bin/env python3
"""Whitelist environment backup; never stops a service or changes AutoDL resources.

Requires Python 3.8+ and only its standard library. CLI create/restore are
Linux-only and use /root/omniembedding-service. This is a
file-level, quiescence-checked backup, NOT an application/database-consistent
snapshot. An installed Conda environment is usable only at the original absolute
prefix on a compatible Linux base/CPU architecture/GPU driver. It is not conda-pack.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone


ROOT = Path("/root/omniembedding-service")
MODEL = "cache/huggingface/hub/models--google--embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MANIFEST = "BACKUP-MANIFEST.json"
SCRIPTS = {"autodl_bootstrap.sh", "autodl_verify.py", "autodl_ops.py", "autodl_backup.py"}
MAX_MEMBERS = 1_000_000
MAX_BYTES = 512 * 1024**3
MAX_MANIFEST = 64 * 1024**2
CHUNK = 1024**2


class BackupError(RuntimeError):
    pass


def _safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise BackupError("Invalid archive path")
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or str(path) != name or any(p in {".", ".."} for p in path.parts):
        raise BackupError("Absolute, noncanonical or escaping archive path")
    return path


def _excluded(name):
    parts = PurePosixPath(name).parts
    for part in parts:
        lower = part.lower()
        if (lower in {"inference.key", "token", "stored_tokens", ".netrc", ".pypirc", ".locks"}
                or lower == ".env" or lower.startswith(".env.")
                or lower.startswith("credentials") or lower.endswith((".incomplete", ".key"))):
            return True
    return name.startswith((".conda/pkgs/", ".conda/.cache/", ".conda/conda-bld/")) or name in {
        ".conda/pkgs", ".conda/.cache", ".conda/conda-bld"}


def _allowed_file(name):
    path = _safe_name(name)
    if _excluded(name):
        return False
    parts = path.parts
    return (name.startswith(".conda/") or name.startswith(MODEL + "/")
            or (parts[0] == "omniembed" and len(parts) > 1 and path.suffix == ".py")
            or (len(parts) == 1 and (name in {"run-autodl.sh", "LICENSE"} or name.startswith("requirements")))
            or (len(parts) == 2 and parts[0] == "scripts" and parts[1] in SCRIPTS))


def _allowed_dir(name):
    _safe_name(name)
    if _excluded(name):
        return False
    return (name == ".conda" or name.startswith(".conda/")
            or name == "omniembed" or name.startswith("omniembed/") or name == "scripts"
            or name == MODEL or name.startswith(MODEL + "/")
            or MODEL.startswith(name + "/"))


def _signature(path):
    value = path.lstat()
    target = os.readlink(path) if stat.S_ISLNK(value.st_mode) else None
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns, target)


def _inventory(source):
    """No traversal through symlinks; inventory is also repeated after compression."""
    if source.is_symlink() or not source.is_dir():
        raise BackupError("Source must be a real directory")
    found = {}

    def add(path):
        name = path.relative_to(source).as_posix()
        if _excluded(name):
            return
        signature = _signature(path)
        mode = signature[2]
        if stat.S_ISDIR(mode):
            if _allowed_dir(name):
                found[name] = signature
        elif stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            if _allowed_file(name):
                found[name] = signature
        elif _allowed_file(name):
            raise BackupError("Special files are forbidden in the backup whitelist")

    for base_name in [".conda", "omniembed", MODEL]:
        base = source / base_name
        if not os.path.lexists(base):
            raise BackupError("Required environment/code/model directory is missing: " + base_name)
        ancestors = list(PurePosixPath(base_name).parents)[:-1]
        for ancestor in reversed(ancestors):
            path = source / str(ancestor)
            if path.is_symlink() or not path.is_dir():
                raise BackupError("A selected directory has a symlink/non-directory ancestor")
            add(path)
        if base.is_symlink() or not base.is_dir():
            raise BackupError("Selected root directories must not be symlinks")
        add(base)
        for current, dirs, files in os.walk(base, followlinks=False):
            current = Path(current)
            for child in sorted(dirs + files):
                add(current / child)
            dirs[:] = [d for d in dirs if not (current / d).is_symlink()
                       and not _excluded((current / d).relative_to(source).as_posix())]
    for path in source.iterdir():
        if path.name in {"run-autodl.sh", "LICENSE"} or path.name.startswith("requirements"):
            add(path)
    scripts = source / "scripts"
    if os.path.lexists(scripts):
        if scripts.is_symlink() or not scripts.is_dir():
            raise BackupError("scripts must be a real directory")
        add(scripts)
        for name in sorted(SCRIPTS):
            path = scripts / name
            if os.path.lexists(path):
                add(path)
    if "run-autodl.sh" not in found:
        raise BackupError("run-autodl.sh is required")
    return dict(sorted(found.items()))


def _check_links(entries):
    """Resolve links including link ancestors; reject dangling links and cycles."""
    def target_parts(target):
        if not target or "\0" in target or "\\" in target:
            raise BackupError("Invalid symlink target")
        if target.startswith("/"):
            try:
                relative = PurePosixPath(target).relative_to(PurePosixPath(str(ROOT)))
            except ValueError as error:
                raise BackupError("Unknown absolute symlink target outside the service prefix") from error
            return [], list(relative.parts)
        return None, list(PurePosixPath(target).parts)

    for name, entry in entries.items():
        for ancestor in PurePosixPath(name).parents:
            if str(ancestor) in entries and entries[str(ancestor)]["kind"] != "directory":
                raise BackupError("An archive member has a non-directory ancestor")
        if entry["kind"] != "symlink":
            continue
        reset, pending = target_parts(entry["target"])
        resolved = list(PurePosixPath(name).parent.parts) if reset is None else reset
        resolved = [p for p in resolved if p != "."]
        hops = 0
        while pending:
            part = pending.pop(0)
            if part == ".":
                continue
            if part == "..":
                if not resolved:
                    raise BackupError("Escaping symlink target")
                resolved.pop()
                continue
            resolved.append(part)
            prefix = "/".join(resolved)
            child = entries.get(prefix)
            if child and child["kind"] == "symlink":
                hops += 1
                if hops > 40:
                    raise BackupError("Symlink cycle or excessive link chain")
                reset, extra = target_parts(child["target"])
                resolved = resolved[:-1] if reset is None else reset
                pending = extra + pending
        target = "/".join(resolved)
        if target not in entries:
            raise BackupError("Symlink target is absent from the selected backup")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _metadata(source):
    python = source / ".conda/bin/python"
    if not python.exists():
        raise BackupError("Installed environment Python is missing")
    result = subprocess.run([str(python), "-m", "pip", "freeze", "--all"],
                            capture_output=True, text=True, timeout=120, check=False,
                            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                                 "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    if result.returncode:
        raise BackupError("pip freeze failed; its output is withheld")
    freeze, omitted = [], 0
    for line in result.stdout.splitlines():
        # Direct URL/editable references can contain credentials; do not record them.
        if re.fullmatch(r"[A-Za-z0-9_.-]+={2,3}[A-Za-z0-9_.+!-]+", line):
            freeze.append(line)
        elif line.strip():
            omitted += 1
    return {"platform": platform.platform(), "system": platform.system(),
            "machine": platform.machine(), "glibc": list(platform.libc_ver()),
            "pip_freeze": freeze, "pip_freeze_direct_or_unrecognized_lines_omitted": omitted}


def _entry(name, signature):
    mode = signature[2]
    kind = "directory" if stat.S_ISDIR(mode) else "symlink" if stat.S_ISLNK(mode) else "file"
    entry = {"path": name, "kind": kind, "mode": stat.S_IMODE(mode) & 0o777,
             "size": signature[3] if kind != "directory" else 0,
             "mtime_ns": signature[4], "mtime_seconds": signature[4] // 10**9}
    if kind == "symlink":
        entry["target"] = signature[6]
    return entry


def _write_exclusive(path, content):
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _create(source, output, metadata=None):
    source, output = source.resolve(), output.absolute()
    output = output.parent.resolve() / output.name
    if output == source or source in output.parents:
        raise BackupError("Backup output must be outside the service root")
    if not output.parent.is_dir():
        raise BackupError("Backup output parent must already exist")
    paths = [output, Path(str(output) + ".manifest.json"), Path(str(output) + ".sha256")]
    partials = [Path(str(path) + ".partial") for path in paths]
    if any(os.path.lexists(path) for path in paths + partials):
        raise BackupError("An output or .partial file already exists; choose a new name")
    before = _inventory(source)
    entries = {name: _entry(name, signature) for name, signature in before.items()}
    if len(entries) > MAX_MEMBERS or sum(e["size"] for e in entries.values()) > MAX_BYTES:
        raise BackupError("Selected source exceeds the archive safety limits")
    _check_links(entries)
    environment = metadata if metadata is not None else _metadata(source)
    model_snapshot = MODEL + "/snapshots/" + REVISION
    if model_snapshot not in entries or entries[model_snapshot]["kind"] != "directory":
        raise BackupError("The pinned model revision is absent from the selected cache")
    manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_prefix": str(ROOT), "model": "google/embeddinggemma-2", "model_revision": REVISION,
                "environment": environment, "entries": [],
                "consistency": "File metadata checked before and after archiving; NOT an application/database-consistent snapshot.",
                "restore_constraints": "Same absolute prefix and a compatible Linux base/CPU/GPU driver are required; not conda-pack.",
                "credential_policy": "Known credential filenames and direct-reference pip metadata are excluded; no service key is supplied. Review source code for hardcoded credentials before sharing."}
    try:
        with partials[0].open("xb") as raw:
            os.chmod(partials[0], 0o600)
            with gzip.GzipFile(filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for name, entry in entries.items():
                        path = source / name
                        if _signature(path) != before[name]:
                            raise BackupError("Selected source metadata changed before reading")
                        info = tarfile.TarInfo(name)
                        info.mode, info.mtime = entry["mode"], entry["mtime_ns"] // 10**9
                        if entry["kind"] == "directory":
                            info.type = tarfile.DIRTYPE
                            archive.addfile(info)
                        elif entry["kind"] == "symlink":
                            info.type, info.linkname = tarfile.SYMTYPE, entry["target"]
                            archive.addfile(info)
                        else:
                            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                            with os.fdopen(descriptor, "rb") as stream:
                                value = os.fstat(stream.fileno())
                                actual = (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                                          value.st_mtime_ns, value.st_ctime_ns, None)
                                if actual != before[name]:
                                    raise BackupError("A selected file changed or was replaced")
                                digest = hashlib.sha256()

                                class Reader:
                                    def read(self, amount=-1):
                                        block = stream.read(amount)
                                        digest.update(block)
                                        return block

                                info.size = entry["size"]
                                archive.addfile(info, Reader())
                                entry["sha256"] = digest.hexdigest()
                    if _inventory(source) != before:
                        raise BackupError("Selected files/directories changed during backup; no backup published")
                    manifest["entries"] = list(entries.values())
                    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
                    info = tarfile.TarInfo(MANIFEST)
                    info.mode, info.size = 0o600, len(manifest_bytes)
                    archive.addfile(info, io.BytesIO(manifest_bytes))
            raw.flush()
            os.fsync(raw.fileno())
        archive_sha = _sha(partials[0])
        if _inventory(source) != before:
            raise BackupError("Selected files/directories changed before publication; no backup published")
        _write_exclusive(partials[1], manifest_bytes)
        _write_exclusive(partials[2], (archive_sha + "  " + output.name + "\n").encode())
        # Publish the archive last so its presence implies both metadata sidecars exist.
        for index in [1, 2, 0]:
            os.replace(partials[index], paths[index])
        return {"archive": str(output), "sha256": archive_sha, "members": len(entries),
                "uncompressed_file_bytes": sum(e["size"] for e in entries.values() if e["kind"] == "file"),
                "credentials_included": False, "service_stopped": False}
    finally:
        for path in partials:
            path.unlink(missing_ok=True)


def _json(content):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BackupError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(content, object_pairs_hook=unique)


def _member(info):
    name = info.name
    _safe_name(name)
    if info.islnk() or info.issparse() or not (info.isdir() or info.issym() or info.isfile()):
        raise BackupError("Unsupported tar member type")
    if info.mode & ~0o777:
        raise BackupError("Special permission bits in archive")
    kind = "directory" if info.isdir() else "symlink" if info.issym() else "file"
    if not (_allowed_dir(name) if kind == "directory" else _allowed_file(name)):
        raise BackupError("Archive member is outside the whitelist")
    if info.size < 0 or info.size > MAX_BYTES or (kind != "file" and info.size):
        raise BackupError("Invalid archive member size")
    if not isinstance(info.mtime, (int, float)) or info.mtime < 0 or int(info.mtime) != info.mtime:
        raise BackupError("Invalid archive modification time")
    return {"path": name, "kind": kind, "mode": info.mode, "mtime_seconds": int(info.mtime),
            "size": len(os.fsencode(info.linkname)) if kind == "symlink" else info.size,
            **({"target": info.linkname} if kind == "symlink" else {})}


def _validate_manifest(manifest, observed):
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("source_prefix") != str(ROOT)
            or manifest.get("model") != "google/embeddinggemma-2"
            or manifest.get("model_revision") != REVISION):
        raise BackupError("Unsupported backup manifest or prefix/model revision")
    allowed_keys = {"schema_version", "created_at_utc", "source_prefix", "model", "model_revision",
                    "environment", "entries", "consistency", "restore_constraints", "credential_policy"}
    if set(manifest) != allowed_keys:
        raise BackupError("Unknown/missing manifest metadata fields")
    environment = manifest.get("environment")
    if not isinstance(environment, dict) or environment.get("system") != "Linux":
        raise BackupError("The recorded environment must be Linux")
    if not isinstance(environment.get("pip_freeze"), list) or any(
            not isinstance(line, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+={2,3}[A-Za-z0-9_.+!-]+", line)
            for line in environment["pip_freeze"]):
        raise BackupError("Unsafe/non-pinned pip metadata in manifest")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(observed):
        raise BackupError("Manifest/member count mismatch")
    expected = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("path") in expected:
            raise BackupError("Invalid or duplicate manifest member")
        name = entry.get("path")
        _safe_name(name)
        expected_keys = {"path", "kind", "mode", "size", "mtime_ns", "mtime_seconds"}
        if entry.get("kind") == "file":
            expected_keys.add("sha256")
        elif entry.get("kind") == "symlink":
            expected_keys.add("target")
        if set(entry) != expected_keys:
            raise BackupError("Unknown/missing manifest member fields")
        expected[name] = entry
        if not isinstance(entry.get("mtime_ns"), int) or entry["mtime_ns"] < 0:
            raise BackupError("Invalid modification time")
        if entry.get("mtime_seconds") != entry["mtime_ns"] // 10**9:
            raise BackupError("Manifest modification times disagree")
        actual = observed.get(name)
        if actual is None or any(actual.get(k) != entry.get(k) for k in actual):
            raise BackupError("Manifest member hash/size/type/mode/target mismatch")
    for required in [".conda", "omniembed", MODEL, MODEL + "/snapshots/" + REVISION]:
        if expected.get(required, {}).get("kind") != "directory":
            raise BackupError("Required environment/model directory absent")
    if expected.get("run-autodl.sh", {}).get("kind") not in {"file", "symlink"}:
        raise BackupError("Startup script absent")
    _check_links(expected)
    return expected


def _verify(archive_path):
    archive_path = archive_path.resolve(strict=True)
    sidecar = Path(str(archive_path) + ".sha256")
    side_manifest = Path(str(archive_path) + ".manifest.json")
    if sidecar.stat().st_size > 4096 or side_manifest.stat().st_size > MAX_MANIFEST:
        raise BackupError("Oversized metadata sidecar")
    line = sidecar.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}  [^\r\n]+", line):
        raise BackupError("Invalid SHA256 sidecar")
    digest = _sha(archive_path)
    if digest != line[:64]:
        raise BackupError("Archive SHA256 mismatch")
    observed, manifest_bytes, total = {}, None, 0
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for info in archive:
            if info.name == MANIFEST:
                if manifest_bytes is not None or not info.isfile() or not 0 < info.size <= MAX_MANIFEST:
                    raise BackupError("Invalid/duplicate embedded manifest")
                manifest_bytes = archive.extractfile(info).read()
                continue
            if manifest_bytes is not None:
                raise BackupError("Archive data after final manifest")
            if info.name in observed or len(observed) >= MAX_MEMBERS:
                raise BackupError("Duplicate/excessive archive members")
            entry = _member(info)
            total += info.size
            if total > MAX_BYTES:
                raise BackupError("Archive exceeds extraction size limit")
            if info.isfile():
                digest_file = hashlib.sha256()
                stream = archive.extractfile(info)
                while block := stream.read(CHUNK):
                    digest_file.update(block)
                entry["sha256"] = digest_file.hexdigest()
            observed[info.name] = entry
    if manifest_bytes is None or side_manifest.read_bytes() != manifest_bytes:
        raise BackupError("Missing embedded manifest or sidecar mismatch")
    manifest = _json(manifest_bytes)
    entries = _validate_manifest(manifest, observed)
    return manifest, entries, digest


def _rename_noreplace(source, target):
    # Linux renameat2(RENAME_NOREPLACE) prevents a racing empty directory from
    # being overwritten. Do not silently degrade to overwrite-capable os.rename.
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renameat2", None)
    if function is None:
        raise BackupError("Atomic no-replace rename is unavailable on this Linux base")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise BackupError("Restore target appeared; refusing to overwrite it")
        raise OSError(error, os.strerror(error))


def _restore(archive_path, target, new_key=False, committer=_rename_noreplace):
    if os.path.lexists(target):
        raise BackupError("Restore target already exists; no overwrite is allowed")
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise BackupError("Restore parent must be a real existing directory")
    manifest, entries, digest = _verify(archive_path)
    staging = Path(tempfile.mkdtemp(prefix=".omniembedding-restore-", dir=target.parent))
    try:
        seen = set()
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for info in archive:
                if info.name == MANIFEST:
                    content = archive.extractfile(info).read(MAX_MANIFEST + 1)
                    if _json(content) != manifest:
                        raise BackupError("Archive manifest changed after verification")
                    continue
                entry = _member(info)
                expected = entries.get(info.name)
                if info.name in seen or expected is None or any(entry.get(k) != expected.get(k) for k in entry):
                    raise BackupError("Archive member changed after verification")
                seen.add(info.name)
                path = staging / info.name
                if info.isdir():
                    path.mkdir(mode=0o700, parents=True, exist_ok=True)
                elif info.isfile():
                    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    hashed = hashlib.sha256()
                    with path.open("xb") as output:
                        stream = archive.extractfile(info)
                        while block := stream.read(CHUNK):
                            output.write(block)
                            hashed.update(block)
                    if hashed.hexdigest() != expected["sha256"]:
                        raise BackupError("Restored file hash mismatch")
                    path.chmod(expected["mode"])
                    os.utime(path, ns=(expected["mtime_ns"], expected["mtime_ns"]))
        if seen != set(entries) or _sha(archive_path) != digest:
            raise BackupError("Archive changed while restoring")
        # Links are made only after all regular writes, never traversed by extraction.
        for name, entry in entries.items():
            if entry["kind"] == "symlink":
                path = staging / name
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                path.symlink_to(entry["target"])
        for name, entry in sorted(entries.items(), key=lambda x: len(PurePosixPath(x[0]).parts), reverse=True):
            if entry["kind"] == "directory":
                path = staging / name
                path.chmod(entry["mode"])
                os.utime(path, ns=(entry["mtime_ns"], entry["mtime_ns"]))
        # Inspect the actual staged filesystem, not only bytes entering write().
        for name, entry in entries.items():
            path = staging / name
            value = path.lstat()
            if entry["kind"] == "file":
                if not stat.S_ISREG(value.st_mode) or value.st_size != entry["size"] or _sha(path) != entry["sha256"]:
                    raise BackupError("Staged regular file failed its final hash/size check")
            elif entry["kind"] == "symlink":
                if not stat.S_ISLNK(value.st_mode) or os.readlink(path) != entry["target"]:
                    raise BackupError("Staged symlink target mismatch")
            elif not stat.S_ISDIR(value.st_mode):
                raise BackupError("Staged directory type mismatch")
            if entry["kind"] != "symlink" and stat.S_IMODE(value.st_mode) != entry["mode"]:
                raise BackupError("Staged file/directory permission mismatch")
        for name in ["logs", "run"]:
            (staging / name).mkdir(mode=0o700)
        if new_key:
            _write_exclusive(staging / "inference.key", (secrets.token_urlsafe(48) + "\n").encode())
        committer(staging, target)
        return {"restored_to": str(target), "archive_sha256": digest, "service_started": False,
                "new_key_generated": bool(new_key),
                "next_step": "Check compatibility, model cache and readiness before starting the existing runner."
                if new_key else "Provide your own inference.key with mode 0600 before starting; no key was restored."}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _linux_root():
    if platform.system() != "Linux":
        raise BackupError("create/restore require Linux and the fixed /root/omniembedding-service prefix")
    if ROOT.is_symlink():
        raise BackupError("The fixed service root must not be a symlink")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Back up the fixed service root without stopping it")
    create.add_argument("--output", type=Path, required=True, help="New tar.gz outside the service root")
    verify = commands.add_parser("verify", help="Verify archive plus .manifest.json and .sha256 sidecars")
    verify.add_argument("archive", type=Path)
    restore = commands.add_parser("restore", help="Restore only when the fixed service root does not exist")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--new-key", action="store_true", help="Generate a new 0600 service key without displaying it")
    args = parser.parse_args()
    try:
        if args.command == "verify":
            manifest, entries, digest = _verify(args.archive)
            result = {"verified": True, "sha256": digest, "members": len(entries),
                      "model_revision": manifest["model_revision"], "credentials_included": False}
        else:
            _linux_root()
            result = _create(ROOT, args.output) if args.command == "create" else _restore(args.archive, ROOT, args.new_key)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (BackupError, OSError, ValueError, tarfile.TarError, subprocess.SubprocessError) as error:
        print("Backup operation failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
