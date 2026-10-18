#!/usr/bin/env python3
"""Build a private fixed retrieval subset; optional explicit remote encoding.

Production SQLite and source files remain read-only. Gold windows deliberately
make this a coverage-controlled benchmark, not an estimate of whole-corpus
quality. Only original stored chunk text is sent to the encoder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omniembed.code_search.ingest import diagnostic_path, excluded_relative_path, safe_error, sensitive_content
from omniembed.code_search.search import Search
from omniembed.code_search.store import DIMENSIONS, MODEL, REVISION, Store
from omniembed.remote_encoder import RemoteEncoder, RemoteEncoderError


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        temporary.chmod(0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


class Source(Store):
    """Pin one existing SQLite view without Store initialization/migrations."""

    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only=ON")
        self.db.execute("BEGIN")

    @contextmanager
    def connect(self):
        yield self.db

    def close(self):
        self.db.rollback()
        self.db.close()


def validated_fixture(value):
    value = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    if not isinstance(value, dict) or not 1 <= len(value.get("queries", [])) <= 100:
        raise ValueError("invalid_benchmark_fixture")
    cases, seen = [], set()
    for case in value["queries"]:
        ident, query, split = case.get("id"), case.get("query"), case.get("split", "development")
        if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", ident)
                or ident in seen or not isinstance(query, str) or not query.strip()
                or sensitive_content(ident) or sensitive_content(query)
                or split not in {"development", "holdout"}):
            raise ValueError("invalid_benchmark_case")
        seen.add(ident)
        golds = []
        for gold in case.get("gold", []):
            path = gold.get("path")
            if (not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts
                    or excluded_relative_path(path) or sensitive_content(path)
                    or type(gold.get("start_line")) is not int or type(gold.get("end_line")) is not int
                    or not 1 <= gold["start_line"] <= gold["end_line"]
                    or any(not isinstance(gold.get(k), str) or not re.fullmatch(r"[a-f0-9]{64}", gold[k])
                           for k in ("source_sha256", "evidence_sha256"))):
                raise ValueError("invalid_benchmark_gold")
            golds.append({k: gold[k] for k in ("path", "start_line", "end_line", "source_sha256", "evidence_sha256")})
        cases.append({"id": ident, "query": query, "split": split, "gold": golds,
                      **{k: case[k] for k in ("project", "extension") if case.get(k)}})
    return {"version": value.get("version", 1), "scope": "Fixed coverage-controlled benchmark subset", "queries": cases}


def build_benchmark(source_db, fixture, output_dir, *, root=None, max_chunks=500, seed=20261018,
                    window_lines=8, hard_negatives=20, encode=False, encoder=None,
                    endpoint_file=None, key_file=None):
    if not 1 <= max_chunks <= 500 or not 0 <= window_lines <= 100 or not 0 <= hard_negatives <= 100:
        raise ValueError("invalid_benchmark_limits")
    fixture = validated_fixture(fixture)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    events = output / "events.jsonl"

    def log(stage, **fields):
        with events.open("a", encoding="utf-8") as handle:
            events.chmod(0o600)
            handle.write(json.dumps({"at": utc(), "stage": stage, **fields}, ensure_ascii=False, allow_nan=False) + "\n")

    source = Source(source_db)
    selected, reasons, file_safety = {}, defaultdict(set), {}
    rejected = Counter()
    try:
        counts = {table: source.db.execute("SELECT count(*) FROM " + table).fetchone()[0]
                  for table in ("files", "chunks", "vectors")}
        log("source_opened", source_counts=counts, fixture_sha256=digest(json.dumps(fixture, sort_keys=True)),
            max_chunks=max_chunks, seed=seed)

        def get_chunk(ident):
            row = source.db.execute("SELECT c.*,f.path,f.project,f.extension,f.sha256 source_sha256 "
                                    "FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.id=?", (ident,)).fetchone()
            return dict(row) if row else None

        def safe(chunk):
            if chunk is None:
                return False
            reason = excluded_relative_path(chunk["path"])
            text = chunk.get("embedding_text")
            if reason is None and (not isinstance(text, str) or not text.strip() or len(text) > 8192):
                reason = "invalid_embedding_text"
            if reason is None and (sensitive_content(chunk["path"]) or sensitive_content(chunk["content"]) or sensitive_content(text)):
                reason = "sensitive_content"
            if reason is None:
                file_id = chunk["file_id"]
                if file_id not in file_safety:
                    record = source.db.execute("SELECT content FROM files WHERE id=?", (file_id,)).fetchone()
                    file_safety[file_id] = bool(record and not sensitive_content(record["content"]))
                if not file_safety[file_id]:
                    reason = "sensitive_source_snapshot"
            if reason:
                rejected[reason] += 1
                return False
            return True

        def add(chunk, reason):
            if not safe(chunk):
                return False
            if chunk["id"] not in selected and len(selected) >= max_chunks:
                return False
            selected[chunk["id"]] = chunk
            reasons[chunk["id"]].add(reason)
            return True

        gold_manifest = []
        for case in fixture["queries"]:
            for gold in case["gold"]:
                rows = source.db.execute("SELECT c.id FROM chunks c JOIN files f ON f.id=c.file_id "
                                         "WHERE f.path=? AND f.sha256=? AND c.start_line<=? AND c.end_line>=? "
                                         "ORDER BY c.start_line,c.id",
                                         (gold["path"], gold["source_sha256"], gold["end_line"] + window_lines,
                                          max(1, gold["start_line"] - window_lines))).fetchall()
                matching = []
                for row in rows:
                    chunk = get_chunk(row["id"])
                    if not add(chunk, "gold_window"):
                        raise ValueError("gold_window_unsafe_or_exceeds_limit")
                    overlap = min(chunk["end_line"], gold["end_line"]) - max(chunk["start_line"], gold["start_line"]) + 1
                    if overlap >= min(3, gold["end_line"] - gold["start_line"] + 1):
                        matching.append(chunk["id"])
                if not matching:
                    raise ValueError("gold_evidence_missing_from_source_index")
                snapshot = source.db.execute("SELECT content FROM files WHERE path=?", (gold["path"],)).fetchone()
                lines = snapshot["content"].splitlines(keepends=True)
                if (gold["end_line"] > len(lines)
                        or digest("".join(lines[gold["start_line"]-1:gold["end_line"]])) != gold["evidence_sha256"]):
                    raise ValueError("gold_evidence_hash_mismatch")
                gold_manifest.append({"case_id": case["id"], "path": gold["path"], "source_sha256": gold["source_sha256"],
                                      "start_line": gold["start_line"], "end_line": gold["end_line"],
                                      "matching_chunk_ids": matching})
        log("gold_windows_selected", selected=len(selected), evidence_items=len(gold_manifest))
        lexical = Search(source, SimpleNamespace())
        negative_queries = []
        for case in fixture["queries"]:
            if case["split"] != "development" or not hard_negatives:
                continue
            hits = lexical.bm25(case["query"], hard_negatives, case.get("project"), case.get("extension"))
            negative_queries.append({"case_id": case["id"], "query_sha256": digest(case["query"]), "retrieved": len(hits)})
            for ident, _ in hits:
                if ident in selected and "gold_window" in reasons[ident]:
                    continue
                add(get_chunk(ident), "development_bm25_candidate")
        log("development_candidates_selected", selected=len(selected), queries=negative_queries)

        # Bounded per-project reservoirs keep selection memory independent of
        # corpus text size. Stable chunk-id hashes make row insertion order
        # irrelevant; only development queries enter negative mining.
        reservoirs = defaultdict(list)
        for row in source.db.execute("SELECT c.id,f.project,f.path FROM chunks c JOIN files f ON f.id=c.file_id ORDER BY c.id"):
            if row["id"] in selected or excluded_relative_path(row["path"]) or sensitive_content(row["path"]):
                continue
            entry = (-int(digest(f"{seed}\0{row['id']}"), 16), row["id"])
            heap = reservoirs[row["project"]]
            if len(heap) < max_chunks:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
        pools = {project: [cid for _, cid in sorted(heap, reverse=True)] for project, heap in reservoirs.items()}
        projects = sorted(pools)
        random.Random(seed).shuffle(projects)
        while len(selected) < max_chunks and any(pools.values()):
            for project in projects:
                while pools[project] and len(selected) < max_chunks:
                    if add(get_chunk(pools[project].pop(0)), "seeded_project_sample"):
                        break
        if not selected:
            raise ValueError("empty_benchmark")
        log("selection_complete", selected=len(selected), rejected=dict(rejected),
            projects=dict(Counter(c["project"] for c in selected.values())))

        configured_root = root or source.meta_get("root") or Path.home() / "AiCode"
        database = output / "benchmark.sqlite3"
        store = Store(database, root=configured_root)
        database.chmod(0o600)
        grouped = defaultdict(list)
        for chunk in selected.values():
            grouped[chunk["file_id"]].append(chunk)
        for file_id, chunks in grouped.items():
            record = dict(source.db.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone())
            record["chunk_count"] = len(chunks)
            store.replace_file(record, sorted(chunks, key=lambda c: c["ordinal"]))
        selection = [{"chunk_id": c["id"], "path": c["path"], "source_sha256": c["source_sha256"],
                      "content_sha256": digest(c["content"]), "embedding_sha256": digest(c["embedding_text"]),
                      "start_line": c["start_line"], "end_line": c["end_line"], "reasons": sorted(reasons[c["id"]])}
                     for c in sorted(selected.values(), key=lambda c: c["id"])]
        selection_hash = digest(json.dumps(selection, sort_keys=True, ensure_ascii=False))
        fixture_path = output / "fixture.json"
        write_json(fixture_path, fixture)
        manifest = {"version": 1, "created_at": utc(), "status": "built", "benchmark_subset": True,
                    "whole_ai_code_quality": False, "production_index_modified": False,
                    "source_database": diagnostic_path(str(source.path)), "source_counts": counts,
                    "database": str(database), "fixture": str(fixture_path), "chunks": len(selection), "files": len(grouped),
                    "max_chunks": max_chunks, "seed": seed, "window_lines": window_lines,
                    "corpus_selection_sha256": selection_hash, "gold": gold_manifest, "selection": selection,
                    "negative_strategy": {"split": "development", "bm25_top_k": hard_negatives,
                                          "human_judged_negative": False, "project_stratified_seeded_fill": True},
                    "limits": ["Gold windows are deliberately included; coverage is controlled rather than naturally sampled.",
                               "BM25 candidate distractors are not human-labelled irrelevant passages.",
                               "Results cannot establish whole-AiCode retrieval quality."]}
        store.meta_set("benchmark", {k: manifest[k] for k in ("benchmark_subset", "whole_ai_code_quality", "corpus_selection_sha256", "chunks")})
        store.meta_set("fixture_sha256", digest(json.dumps(fixture, sort_keys=True)))
        write_json(output / "manifest.json", manifest)
    finally:
        source.close()

    if encode:
        encoder = encoder or RemoteEncoder(model_id=MODEL, revision=REVISION,
                                           endpoint_file=endpoint_file or ROOT / "data/autodl/endpoint.json",
                                           key_file=key_file or ROOT / "data/autodl/inference.key", timeout=45)
        if isinstance(encoder, RemoteEncoder):
            encoder._url = None
        job = store.job_create("embed")
        store.job_update(job["id"], status="running", phase="embed", vectors_total=len(selected),
                         message="独立 benchmark 子集编码；不代表全量索引完成")
        completed = 0
        try:
            pending = sorted(selected.values(), key=lambda c: c["id"])
            while pending:
                batch, chars = [], 0
                for chunk in pending[:16]:
                    size = len(chunk["embedding_text"])
                    if batch and chars + size > 32768:
                        break
                    batch.append(chunk)
                    chars += size
                started = time.perf_counter()
                audit = {"phase": "embed", "items": len(batch), "characters": chars, "started_at": utc(),
                         "input_hashes": [{"chunk_id": c["id"], "sha256": digest(c["embedding_text"])} for c in batch]}
                try:
                    values = np.asarray(encoder.docs([""] * len(batch), [c["embedding_text"] for c in batch], bs=16), dtype=np.float32)
                    if values.shape != (len(batch), DIMENSIONS) or not np.isfinite(values).all():
                        raise RemoteEncoderError("benchmark 向量校验失败")
                    store.vector_put_many_with_audit(list(zip([c["id"] for c in batch], values)), MODEL, REVISION,
                        job_id=job["id"], batch={**audit, "status": "success", "http_status": 200,
                                               "elapsed_ms": (time.perf_counter()-started)*1000, "error_code": None,
                                               "boot_id": getattr(encoder, "boot_id", None), "finished_at": utc()})
                except Exception as exc:
                    store.batch_record(job["id"], **{**audit, "status": "failed", "http_status": getattr(exc, "http_status", None),
                                                    "elapsed_ms": (time.perf_counter()-started)*1000, "error_code": type(exc).__name__,
                                                    "boot_id": getattr(encoder, "boot_id", None), "finished_at": utc()})
                    raise
                completed += len(batch)
                pending = pending[len(batch):]
                store.job_update(job["id"], vectors_done=completed, message=f"benchmark 已编码 {completed}/{len(selected)}")
                log("encoding_batch", items=len(batch), characters=chars, completed=completed,
                    input_sha256=digest(json.dumps(audit["input_hashes"], sort_keys=True)))
            store.job_update(job["id"], status="completed", phase="completed", finished_at=utc(), message="benchmark 子集编码完成")
            manifest.update(status="encoded", vectors=completed, encoding_job_id=job["id"])
        except Exception as exc:
            store.job_update(job["id"], status="failed", phase="failed", finished_at=utc(), errors=1, message=safe_error(exc))
            manifest.update(status="encoding_failed", vectors=completed, error_code=type(exc).__name__, encoding_job_id=job["id"])
            write_json(output / "manifest.json", manifest)
            log("encoding_failed", error_code=type(exc).__name__, completed=completed)
            raise
    else:
        manifest["vectors"] = 0
    write_json(output / "manifest.json", manifest)
    log("completed", status=manifest["status"], chunks=manifest["chunks"], vectors=manifest["vectors"],
        corpus_selection_sha256=selection_hash, whole_ai_code_quality=False)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=ROOT / "data/code-search/code-search.sqlite3")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/code_search_eval.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--max-chunks", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--window-lines", type=int, default=8)
    parser.add_argument("--hard-negatives", type=int, default=20)
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--endpoint-file", type=Path, default=ROOT / "data/autodl/endpoint.json")
    parser.add_argument("--key-file", type=Path, default=ROOT / "data/autodl/inference.key")
    args = parser.parse_args(argv)
    try:
        value = build_benchmark(args.source_db, args.fixture, args.output_dir, root=args.root,
                                max_chunks=args.max_chunks, seed=args.seed, window_lines=args.window_lines,
                                hard_negatives=args.hard_negatives, encode=args.encode,
                                endpoint_file=args.endpoint_file, key_file=args.key_file)
        print(json.dumps({k: value[k] for k in ("status", "database", "fixture", "chunks", "vectors", "corpus_selection_sha256", "whole_ai_code_quality")}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_code": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
