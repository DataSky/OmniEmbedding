#!/usr/bin/env python3
"""基于人工标注源文评测检索；默认仅使用本地只读快照与离线查询向量。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import threading
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omniembed.code_search.search import Search
from omniembed.code_search.store import DIMENSIONS, MODEL, REVISION, Store, tokens
from omniembed.code_search.ingest import excluded_relative_path, sensitive_content


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def validate_fixture(fixture):
    """Validate before logging, freezing inputs, waiting or calling an encoder.

    Extra annotation fields are retained, so inspect their strings as well as
    the fields consumed by scoring. Errors deliberately contain no input data.
    Source existence and readable evidence remain per-case diagnostics.
    """
    if not isinstance(fixture, dict) or not isinstance(fixture.get("queries"), list):
        raise ValueError("评测题集必须包含 queries 列表")
    try:
        json.dumps(fixture, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise ValueError("评测题集必须为有效 JSON 数据") from None
    cases = fixture["queries"]
    if not 1 <= len(cases) <= 100:
        raise ValueError("单次专项评测需包含 1—100 个问题")
    if "scope" in fixture and (not isinstance(fixture["scope"], str) or len(fixture["scope"]) > 4000):
        raise ValueError("评测范围说明无效")
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("评测问题格式无效")
        identifier, query = case.get("id"), case.get("query")
        if (not isinstance(identifier, str) or not 1 <= len(identifier) <= 128
                or not identifier.strip() or identifier in identifiers
                or any(ord(char) < 32 or ord(char) == 127 for char in identifier)):
            raise ValueError("评测问题标识无效或重复")
        identifiers.add(identifier)
        if not isinstance(query, str) or not 1 <= len(query) <= 4000 or not query.strip():
            raise ValueError("评测问题长度不符合范围")
        split = case.get("split", "development")
        if not isinstance(split, str) or split not in {"development", "holdout"}:
            raise ValueError("评测问题分组无效")
        gold = case.get("gold", [])
        if not isinstance(gold, list) or len(gold) > 100:
            raise ValueError("评测证据列表无效")
        for evidence in gold:
            if not isinstance(evidence, dict):
                raise ValueError("评测证据格式无效")
            path = evidence.get("path")
            first, last = evidence.get("start_line"), evidence.get("end_line")
            if (not isinstance(path, str) or not 1 <= len(path) <= 4096 or "\0" in path
                    or type(first) is not int or type(last) is not int or not 1 <= first <= last
                    or any(not isinstance(evidence.get(key), str)
                           or not re.fullmatch(r"[a-f0-9]{64}", evidence[key])
                           for key in ("source_sha256", "evidence_sha256"))):
                raise ValueError("评测证据路径、行号或摘要无效")
    pending = [fixture]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("评测元数据字段无效")
            pending.extend(value)
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and sensitive_content(value):
            raise ValueError("评测元数据含疑似敏感信息，未生成检索日志。")
    return fixture


class ReadOnlyStore(Store):
    """Use one pinned SQLite read transaction; never initialize or write tables."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only=ON")
        self.db.execute("BEGIN")

    @contextmanager
    def connect(self):
        yield self.db

    def add_trace(self, *args, **kwargs):
        pass  # Evaluation writes its own metadata-only JSONL outside the DB.

    def close(self):
        self.db.rollback()
        self.db.close()


class OfflineQueries:
    def __init__(self, fixture, query_vectors):
        self.values = {}
        if query_vectors is None:
            return
        value = json.loads(Path(query_vectors).read_text()) if isinstance(query_vectors, (str, Path)) else query_vectors
        if not isinstance(value.get("query_sha256"), dict):
            raise ValueError("离线向量缺少 query_sha256 绑定；旧格式不能用于评测，请重新生成。")
        if (value.get("model") != MODEL or value.get("revision") != REVISION
                or value.get("dimensions") != DIMENSIONS or not isinstance(value.get("queries"), dict)
                or not isinstance(value.get("query_sha256"), dict)):
            raise ValueError("查询向量的模型、版本或维度不符")
        for case in fixture["queries"]:
            if case["id"] not in value["queries"]:
                continue
            if value["query_sha256"].get(case["id"]) != digest(case["query"]):
                raise ValueError("离线查询向量未绑定当前问题文本")
            vector = np.asarray(value["queries"][case["id"]], dtype=np.float32)
            if (vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1, atol=.05)):
                raise ValueError("离线查询向量校验失败")
            self.values[case["query"]] = vector / np.linalg.norm(vector)

    def query(self, texts, bs=1):
        if any(text not in self.values for text in texts):
            raise ValueError("没有该问题的离线查询向量")
        return np.stack([self.values[text] for text in texts])


def freeze_database(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("快照输出不能覆盖输入数据库；固定快照请用 --frozen")
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src, sqlite3.connect(output) as dest:
        src.backup(dest)
    output.chmod(0o600)
    return output


def snapshot_metadata(store):
    db = store.db
    counts = {table: db.execute("SELECT count(*) FROM " + table).fetchone()[0] for table in ("files", "chunks", "vectors")}
    chunk_hash = hashlib.sha256()
    for row in db.execute("SELECT id FROM chunks ORDER BY id"):
        chunk_hash.update((row[0] + "\n").encode())
    vector_hash = hashlib.sha256()
    for row in db.execute("SELECT chunk_id,model,revision,dimensions,created_at,vector FROM vectors ORDER BY chunk_id"):
        vector_hash.update(json.dumps(list(row)[:-1], ensure_ascii=False).encode())
        vector_hash.update(row["vector"])
    file_hash, content_hash = hashlib.sha256(), hashlib.sha256()
    file_columns = {r[1] for r in db.execute("PRAGMA table_info(files)")}
    for row in db.execute("SELECT id,path,sha256,status,content" + (",policy" if "policy" in file_columns else "") + " FROM files ORDER BY id"):
        record = dict(row)
        record["content_sha256"] = digest(record.pop("content"))
        file_hash.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode())
    for row in db.execute("SELECT id,file_id,start_line,end_line,embedding_text FROM chunks ORDER BY id"):
        content_hash.update(json.dumps(list(row)[:-1], ensure_ascii=False).encode())
        content_hash.update(row["embedding_text"].encode())
    policies = {r[0]: r[1] for r in db.execute("SELECT key,value FROM meta WHERE key IN ('tokenizer','chunk_policy')")}
    metadata = {**counts, "chunk_set_sha256": chunk_hash.hexdigest(), "vector_set_sha256": vector_hash.hexdigest(),
                "files_sha256": file_hash.hexdigest(), "embedding_inputs_sha256": content_hash.hexdigest(), "policies": policies}
    metadata["id"] = digest(json.dumps(metadata, sort_keys=True))[:20]
    metadata["path"] = str(store.path)
    return metadata


def read_source_safely(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts or excluded_relative_path(path):
        raise ValueError("证据路径不符合源目录读取规则")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, directory_flags)
    try:
        for component in path.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        final = os.open(path.parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), dir_fd=descriptor)
        with os.fdopen(final, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
                raise ValueError("证据文件类型或大小不符合规则")
            raw = source.read(1024 * 1024 + 1)
            after = os.fstat(source.fileno())
            if len(raw) > 1024 * 1024 or before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise ValueError("证据文件在读取中变化")
            return raw
    finally:
        os.close(descriptor)


def source_evidence(root, case):
    evidence = []
    for gold in case.get("gold", []):
        check = {k: gold[k] for k in ("path", "start_line", "end_line", "source_sha256", "evidence_sha256")}
        check["valid"] = False
        try:
            raw = read_source_safely(root, gold["path"])
            lines = raw.decode("utf-8-sig").splitlines(keepends=True)
            first, last = gold["start_line"], gold["end_line"]
            check["valid"] = (1 <= first <= last <= len(lines) and digest(raw) == gold["source_sha256"]
                              and digest("".join(lines[first - 1:last])) == gold["evidence_sha256"])
        except (OSError, ValueError, UnicodeError):
            check["error_code"] = "source_unavailable_or_invalid"
        evidence.append(check)
    return evidence


def gold_coverage(store, case):
    evidence = []
    for gold in case.get("gold", []):
        rows = store.db.execute("""SELECT c.id,v.chunk_id FROM chunks c JOIN files f ON f.id=c.file_id
            LEFT JOIN vectors v ON v.chunk_id=c.id AND v.model=? AND v.revision=? AND v.dimensions=?
            WHERE f.path=? AND f.sha256=? AND c.start_line<=? AND c.end_line>=?""",
            (MODEL, REVISION, DIMENSIONS, gold["path"], gold["source_sha256"], gold["end_line"], gold["start_line"])).fetchall()
        evidence.append({"path": gold["path"], "start_line": gold["start_line"], "end_line": gold["end_line"],
                         "indexed": bool(rows), "vectorized": any(r["chunk_id"] is not None for r in rows),
                         "indexed_chunks": len(rows), "vectorized_chunks": sum(r["chunk_id"] is not None for r in rows)})
    return {"gold": len(evidence), "indexed_gold": sum(e["indexed"] for e in evidence),
            "vectorized_gold": sum(e["vectorized"] for e in evidence), "evidence": evidence}


def passage_matches(row, gold):
    overlap = max(0, min(row["end_line"], gold["end_line"]) - max(row["start_line"], gold["start_line"]) + 1)
    return (row["path"] == gold["path"] and row.get("source_sha256") == gold["source_sha256"]
            and overlap >= min(3, gold["end_line"] - gold["start_line"] + 1))


def score_results(rows, gold, ks):
    result = {}
    if not gold:
        return {"no_answer_correct": float(not rows), "returned_results": len(rows)}
    for k in ks:
        result[f"recall@{k}"] = sum(any(passage_matches(row, g) for row in rows[:k]) for g in gold) / len(gold)
    ranks = [rank for rank, row in enumerate(rows, 1) if any(passage_matches(row, g) for g in gold)]
    result["mrr"] = 1 / ranks[0] if ranks else 0.
    file_rows, seen = [], set()
    for row in rows:
        if row["path"] not in seen:
            seen.add(row["path"])
            file_rows.append(row)
    files = {g["path"] for g in gold}
    for k in ks:
        result[f"file_recall@{k}"] = len(files & {row["path"] for row in file_rows[:k]}) / len(files)
    result["duplicate_file_fraction"] = 1 - len(file_rows) / len(rows) if rows else 0.
    return result


def deduplicated(rows, key):
    result, seen = [], set()
    for row in rows:
        if row[key] not in seen:
            seen.add(row[key])
            result.append(row)
    return result


def summarize(cases, modes, ks):
    summary = {}
    for mode in modes:
        summary[mode] = {}
        for split in ("development", "holdout"):
            rows = [case for case in cases if case["mode"] == mode and case["split"] == split]
            diagnostic = [r for r in rows if r.get("metrics") is not None]
            measured = [r for r in diagnostic if r["status"] == "ok"]
            answered = [r for r in measured if not r["unanswerable"]]
            no_answer = [r for r in measured if r["unanswerable"]]
            gold = sum(r["coverage"]["gold"] for r in rows)
            value = {"cases": len(rows), "scorable": len(measured), "skipped": len(rows)-len(measured),
                     "answerable_cases": len(answered), "degraded_cases": sum(r["status"] == "degraded" for r in rows),
                     "incomplete_semantic_cases": sum(r["status"] == "incomplete_semantic_coverage" for r in rows),
                     "execution_error_cases": sum(r["status"] == "execution_error" for r in rows),
                     "effective_mode_counts": dict(Counter(r.get("effective_mode") for r in diagnostic)),
                     "indexed_gold_fraction": sum(r["coverage"]["indexed_gold"] for r in rows) / gold if gold else None,
                     "vector_gold_fraction": sum(r["coverage"]["vectorized_gold"] for r in rows) / gold if gold else None,
                     "no_answer_accuracy": sum(r["metrics"]["no_answer_correct"] for r in no_answer)/len(no_answer) if no_answer else None}
            for metric in ["mrr", "duplicate_file_fraction", *[f"recall@{k}" for k in ks], *[f"file_recall@{k}" for k in ks]]:
                value[metric] = sum(r["metrics"][metric] for r in answered)/len(answered) if answered else None
            partial = [r for r in diagnostic if r["status"] == "incomplete_semantic_coverage" and not r["unanswerable"]]
            value["partial_semantic_diagnostics"] = {"cases": len(partial), "mrr": sum(r["metrics"]["mrr"] for r in partial)/len(partial) if partial else None,
                                                       **{f"recall@{k}": sum(r["metrics"][f"recall@{k}"] for r in partial)/len(partial) if partial else None for k in ks}}
            summary[mode][split] = value
    return summary


def run_evaluation(db, root, fixture, output_dir, label="baseline", modes=("bm25", "vector", "hybrid"),
                   query_vectors=None, search_parameters=None, frozen=False, search_class=None, splits=None):
    """Return the same dictionary written to output_dir/summary.json.

    query_vectors: {model,revision,dimensions:768,queries:{case_id:[floats]},
                    query_sha256:{case_id:sha256(query_utf8)}}.
    frozen=True trusts an already frozen caller-owned SQLite backup; default
    creates a private backup first. No path ever starts or contacts AutoDL.
    """
    fixture = json.loads(Path(fixture).read_text()) if isinstance(fixture, (str, Path)) else fixture
    validate_fixture(fixture)
    root, output = Path(root).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    modes = modes.split(",") if isinstance(modes, str) else list(modes)
    if any(mode not in {"bm25", "vector", "hybrid"} for mode in modes):
        raise ValueError("不支持的检索模式")
    splits = splits.split(",") if isinstance(splits, str) else list(splits) if splits is not None else None
    if splits is not None and (not splits or any(split not in {"development", "holdout"} for split in splits)):
        raise ValueError("不支持的评测分组")
    params = {"candidates": 100, "vector_weight": .6, "ks": [1, 5, 10], **(search_parameters or {})}
    ks = sorted(set(params["ks"]))
    if not ks or not 1 <= min(ks) <= max(ks) <= 100 or not max(ks) <= params["candidates"] <= 1000:
        raise ValueError("评测召回数量超出范围")
    encoder = OfflineQueries(fixture, query_vectors)
    snapshot_path = Path(db).resolve() if frozen else freeze_database(db, output / "snapshot.sqlite3")
    store = ReadOnlyStore(snapshot_path)
    klass = search_class or Search
    search = klass(store, SimpleNamespace(enc=encoder, enc_lock=threading.Lock()))
    stage_path, summary_path = output / "stages.jsonl", output / "summary.json"
    cases = []
    algorithm_path = Path(inspect.getsourcefile(klass) or ROOT / "omniembed/code_search/search.py")
    algorithm_sha = digest(algorithm_path.read_bytes())
    try:
        snapshot = snapshot_metadata(store)
        with stage_path.open("w", encoding="utf-8") as log:
            stage_path.chmod(0o600)

            def emit(stage, **fields):
                log.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "stage": stage,
                                      "label": label, "snapshot_id": snapshot["id"], **fields}, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()

            emit("snapshot", snapshot=snapshot, algorithm_sha256=algorithm_sha)
            for case in fixture["queries"]:
                if splits is not None and case.get("split", "development") not in splits:
                    continue
                base = {"case_id": case["id"], "query": case["query"], "split": case.get("split", "development")}
                evidence = source_evidence(root, case)
                emit("source_evidence", **base, evidence=evidence)
                emit("tokenize", **base, terms=tokens(case["query"]))
                coverage = gold_coverage(store, case)
                emit("coverage", **base, **coverage)
                for mode in modes:
                    record = {"id": case["id"], "query": case["query"], "split": base["split"], "mode": mode,
                              "unanswerable": not case.get("gold"), "coverage": coverage, "metrics": None}
                    if any(not check["valid"] for check in evidence):
                        record["status"] = "invalid_gold_evidence"
                    elif mode != "bm25" and case["query"] not in encoder.values:
                        record["status"] = "missing_offline_query_vector"
                    else:
                        try:
                            result = search.run(case["query"], mode=mode, limit=max(ks), candidates=params["candidates"],
                                                vector_weight=params["vector_weight"], project=case.get("project"), extension=case.get("extension"))
                        except ValueError:
                            record.update(status="execution_error", error_code="retrieval_validation_error")
                            emit("execution_error", **base, mode=mode, error_code=record["error_code"])
                            emit("metrics", **base, mode=mode, status=record["status"], effective_mode=None, metrics=None)
                            cases.append(record)
                            continue
                        safe_rows = []
                        for rank, row in enumerate(result["results"], 1):
                            file = store.file_by_path(row["path"])
                            safe_rows.append({k: row.get(k) for k in ("id", "path", "start_line", "end_line", "score", "vector_score", "bm25_score")}
                                             | {"rank": rank, "content_sha256": digest(row.get("content", "")), "source_sha256": file.get("sha256") if file else None})
                        emit("candidates", **base, mode=mode, routes=result.get("routes", {}))
                        for row in safe_rows:
                            row["gold_matches"] = [index for index, gold in enumerate(case.get("gold", [])) if passage_matches(row, gold)]
                        scope = store.coverage(case.get("project"), case.get("extension"))
                        eligible = result.get("semantic_evaluation_eligible", scope["chunks"] > 0 and scope["vectors"] == scope["chunks"])
                        status = "ok"
                        if result["effective_mode"] != mode or (mode == "hybrid" and params["vector_weight"] in (0., 1.)):
                            status = "degraded"
                        elif mode in {"vector", "hybrid"} and not eligible:
                            status = "incomplete_semantic_coverage"
                        record.update(status=status, effective_mode=result["effective_mode"], semantic_evaluation_eligible=bool(eligible),
                                      warning=result.get("warning"), metrics=score_results(safe_rows, case.get("gold", []), ks))
                        emit("ranks", **base, mode=mode, results=safe_rows,
                             file_deduplicated=deduplicated(safe_rows, "path"), content_deduplicated=deduplicated(safe_rows, "content_sha256"))
                    emit("metrics", **base, mode=mode, status=record["status"], effective_mode=record.get("effective_mode"), metrics=record["metrics"])
                    cases.append(record)
            value = {"label": label, "snapshot": snapshot, "parameters": params,
                     "runtime":{"python":sys.version.split()[0], "numpy":np.__version__},
                     "algorithm_sha256": algorithm_sha,
                     "fixture_sha256": digest(json.dumps(fixture, ensure_ascii=False, sort_keys=True)),
                     "scope": fixture.get("scope"), "splits": splits, "summary": summarize(cases, modes, ks), "cases": cases,
                     "artifacts": {"summary": str(summary_path), "stages": str(stage_path), "snapshot": str(snapshot_path)}}
            summary_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            summary_path.chmod(0o600)
            emit("complete", summary=value["summary"])
            return value
    finally:
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data/code-search/code-search.sqlite3")
    parser.add_argument("--root", type=Path, default=Path.home() / "AiCode")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/code_search_eval.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--modes", default="bm25,vector,hybrid")
    parser.add_argument("--query-vectors", type=Path, help="含model/revision/dimensions/queries/query_sha256的离线JSON；按case id绑定，缺失时语义模式跳过")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--vector-weight", type=float, default=.6)
    parser.add_argument("--frozen", action="store_true", help="输入已为调用者冻结的快照，避免重复备份")
    parser.add_argument("--splits", help="只运行development或holdout；调参先development，选定后再验收holdout")
    args = parser.parse_args(argv)
    result = run_evaluation(args.db, args.root, args.fixture, args.output_dir, args.label, args.modes, args.query_vectors,
                            {"candidates": args.candidates, "vector_weight": args.vector_weight}, args.frozen, splits=args.splits)
    print(json.dumps({"snapshot_id": result["snapshot"]["id"], "summary": result["summary"], "artifacts": result["artifacts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
