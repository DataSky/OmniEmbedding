"""Isolated regressions for ingestion, legacy SQLite and execution auditing.

No test reads the user's corpus or contacts a real inference endpoint.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omniembed.code_search import ingest
from omniembed.code_search.search import Search
from omniembed.code_search.store import DIMENSIONS, MODEL, REVISION, Store
from test_code_search import FakeCodeEncoder, add_file, axis


@pytest.fixture
def ingestion_engine(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    store = Store(tmp_path / "regression.sqlite3", root=root)
    encoder = FakeCodeEncoder()
    engine = ingest.Engine(root=root, store=store, encoder=encoder,
                           endpoint_file=tmp_path / "absent.json", key_file=tmp_path / "absent.key")
    yield root, store, encoder, engine
    engine.shutdown()


def run_job(store, engine, action="sync"):
    job = store.job_create(action)
    engine._run(job["id"], action)
    return store.job_get(job["id"])


@pytest.mark.parametrize("directory", [".env", ".env.production", ".data", "result.xcresult", "Product.app",
                                      "Product.dSYM", "ModuleCache.noindex", "DerivedData", "playwright-report", "test-results"])
def test_generated_and_environment_directories_are_pruned_before_descendant_reads(tmp_path, directory):
    root = tmp_path / "tree"
    target = root / "project" / directory
    target.mkdir(parents=True)
    (target / "never-read.py").write_text("ordinary source hidden inside excluded tree", encoding="utf-8")
    (root / "visible.py").write_text("visible source", encoding="utf-8")
    rows = list(ingest.scan(root))
    assert any(r["file"]["path"] == "visible.py" and r["reason"] is None for r in rows)
    assert not any(r["file"]["path"].endswith("never-read.py") for r in rows)
    assert any(r["is_dir"] and r["file"]["path"] == "project/" + directory and r["reason"] for r in rows)


@pytest.mark.parametrize("field", ["AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "aws_access_key_id"])
def test_literal_aws_credentials_are_never_uploaded(ingestion_engine, field):
    root, store, encoder, engine = ingestion_engine
    value = "AKIA" + "N" * 16 if field == "aws_access_key_id" else "N" * 40
    (root / "unsafe.py").write_text(f'{field} = "' + value + '"\n', encoding="utf-8")
    (root / "safe.py").write_text("ordinarysource\n", encoding="utf-8")
    job = run_job(store, engine)
    assert job["status"] != "failed"
    assert store.stats()["vectors"] == 1
    assert store.file_by_path("unsafe.py")["content"] == ""
    assert all(field not in text for _, texts, _ in encoder.doc_calls for text in texts)


@pytest.mark.parametrize("url", [
    "https://x-access-token:${GH_TOKEN}@github.com/org/repo",
    "postgresql://user:${DB_PASSWORD}@localhost/db",
    "https://user:${{secrets.TOKEN}}@example.org/repo",
])
def test_template_url_credentials_are_local_exemptions_only(url):
    assert ingest.sensitive_content(url) is False
    assert ingest.sensitive_content(url + '\nAPI_KEY="sk-proj-' + "R" * 40 + '"') is True


@pytest.mark.parametrize("url", ["https://user:long-literal-password@example.org/repo",
                                 "postgresql://user:literalpass123@localhost/db"])
def test_literal_url_passwords_remain_blocked(url):
    assert ingest.sensitive_content(url) is True


def test_actual_embedding_header_is_checked_and_does_not_poison_other_items(ingestion_engine):
    root, store, encoder, engine = ingestion_engine
    unsafe_name = "Bearer " + "Z" * 24 + ".py"
    (root / unsafe_name).write_text("harmlesssource\n", encoding="utf-8")
    (root / "safe.py").write_text("visiblecontent\n", encoding="utf-8")
    job = run_job(store, engine)
    assert job["status"] != "failed"
    assert store.stats()["vectors"] == 1
    with store.connect() as db:
        row = db.execute("SELECT * FROM files WHERE id=?", (hashlib.sha256(unsafe_name.encode()).hexdigest(),)).fetchone()
    assert row and row["chunk_count"] == 0
    assert unsafe_name not in row["path"]  # Diagnostics must not disclose the filename's credential.
    assert all("Bearer " not in text for _, texts, _ in encoder.doc_calls for text in texts)


def test_old_generated_snapshot_rows_are_never_uploaded_by_embed_only_job(ingestion_engine):
    root, store, encoder, engine = ingestion_engine
    (root / "safe.py").write_text("validsource\n", encoding="utf-8")
    (root / ".data").mkdir()
    (root / ".data" / "old.py").write_text("oldgeneratedsource\n", encoding="utf-8")
    add_file(store, "safe.py", "validsource\n")
    add_file(store, ".data/old.py", "oldgeneratedsource\n")
    job = run_job(store, engine, "embed")
    assert job["status"] != "failed"
    assert store.stats()["vectors"] == 1
    assert all(".data/" not in text for _, texts, _ in encoder.doc_calls for text in texts)


@pytest.mark.parametrize("invalid", ["", "Bearer " + "S" * 24])
def test_bad_persistent_embedding_item_isolated_while_other_items_complete(ingestion_engine, invalid):
    _, store, encoder, engine = ingestion_engine
    add_file(store, "safe.py", "safechunk")
    _, bad = add_file(store, "bad.py", "badchunk")
    with store.connect() as db:
        db.execute("UPDATE chunks SET embedding_text=? WHERE id=?", (invalid, bad))
    job = run_job(store, engine, "embed")
    assert job["status"] != "failed"
    assert store.stats()["vectors"] == 1
    assert all(text and "Bearer " not in text for _, texts, _ in encoder.doc_calls for text in texts)
    assert job["errors"] >= 1 or job["skipped"] >= 1


def test_sync_encodes_bounded_batches_before_scan_finishes_then_drains_all(ingestion_engine, monkeypatch):
    root, store, encoder, engine = ingestion_engine
    for index in range(140):
        (root / f"source{index:03d}.py").write_text(f"ordinarysource{index}\n", encoding="utf-8")
    original_scan, original_docs = ingest.scan, encoder.docs
    progress = {"seen": 0, "finished": False}
    calls = []

    def observed_scan(*args, **kwargs):
        for entry in original_scan(*args, **kwargs):
            if not entry["is_dir"]:
                progress["seen"] += 1
            yield entry
        progress["finished"] = True

    def docs(titles, texts, bs=16):
        calls.append({**progress, "count": len(texts), "chars": sum(map(len, texts))})
        return original_docs(titles, texts, bs)

    monkeypatch.setattr(ingest, "scan", observed_scan)
    monkeypatch.setattr(encoder, "docs", docs)
    job = run_job(store, engine)
    assert calls and any(not call["finished"] for call in calls)
    assert calls[0]["seen"] < 140
    assert all(1 <= call["count"] <= 16 and call["chars"] <= 32768 for call in calls)
    assert job["status"] == "completed"
    assert store.stats()["chunks"] == store.stats()["vectors"] == 140
    assert store.pending_chunks() == []


def test_unreadable_directory_preserves_vectors_without_reencoding_confirmed_files(ingestion_engine, monkeypatch):
    root, store, encoder, engine = ingestion_engine
    (root / "readable.py").write_text("readablecontent\n", encoding="utf-8")
    hidden = root / "unreadable"
    hidden.mkdir()
    (hidden / "preserved.py").write_text("preservedcontent\n", encoding="utf-8")
    first = run_job(store, engine)
    assert first["status"] == "completed"
    before_calls = len(encoder.doc_calls)
    original = list(ingest.scan(root))
    readable = next(r for r in original if r["file"]["path"] == "readable.py")
    denied = {"file": ingest._file_record("unreadable", status="error", error="没有读取文件或数据库的权限"),
              "chunks": [], "reason": "read_error", "is_dir": True}
    # A global checkpoint may be absent after an earlier interrupted scan.
    # Confirmed per-file policy must still prevent redundant embeddings.
    store.meta_set("chunk_policy", "")
    monkeypatch.setattr(ingest, "scan", lambda *args, **kwargs: iter([readable, denied]))
    second = run_job(store, engine)
    assert second["status"] == "partial"
    assert second["details"]["prune_deferred"] is True
    assert store.stats()["vectors"] == 2
    assert len(encoder.doc_calls) == before_calls
    assert store.file_by_path("unreadable/preserved.py")["chunk_count"] == 1


@pytest.mark.parametrize("terminal", ["paused", "completed"])
def test_cancel_does_not_overwrite_a_job_already_in_terminal_state(ingestion_engine, terminal):
    _, store, _, engine = ingestion_engine
    job = store.job_create("sync")
    store.job_update(job["id"], status=terminal, phase=terminal)
    engine._job_id = job["id"]
    # Simulate the worker's short window between final DB commit and return.
    engine._thread = SimpleNamespace(is_alive=lambda: True)
    try:
        engine.cancel_job()
        assert store.job_get(job["id"])["status"] == terminal
    finally:
        engine._thread = None


def test_error_samples_are_not_displaced_by_many_exclusion_samples(ingestion_engine, monkeypatch):
    _, store, _, engine = ingestion_engine
    entries = [{"file": ingest._file_record(f"ignored{i}", error="excluded_directory"), "chunks": [],
                "reason": "excluded_directory", "is_dir": True} for i in range(90)]
    entries.append({"file": ingest._file_record("read-denied.py", status="error", error="没有读取文件或数据库的权限"),
                    "chunks": [], "reason": "read_error", "is_dir": False})
    monkeypatch.setattr(ingest, "scan", lambda *args, **kwargs: iter(entries))
    job = run_job(store, engine, "scan")
    assert job["status"] == "partial"
    assert job["errors"] == 1
    assert any(item["path"] == "read-denied.py" for item in job["details"]["error_samples"])


def test_fts_insert_has_no_empty_delete_and_updates_use_rowid_lookups(tmp_path, monkeypatch):
    store = Store(tmp_path / "fts.sqlite3")
    add_file(store, "existing.py", "olduniquekey")
    original_connect = store.connect
    statements = []

    @contextmanager
    def observed_connect():
        with original_connect() as db:
            db.set_trace_callback(statements.append)
            yield db

    monkeypatch.setattr(store, "connect", observed_connect)
    add_file(store, "new.py", "newuniquekey")
    deletes = [sql for sql in statements if sql.strip().lower().startswith("delete from chunk_fts")]
    assert deletes == []
    statements.clear()
    add_file(store, "existing.py", "replacementuniquekey")
    deletes = [sql for sql in statements if sql.strip().lower().startswith("delete from chunk_fts")]
    assert deletes and all("rowid" in sql.lower() for sql in deletes)
    assert store.prune_files(["existing.py"]) == 1
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM chunk_fts").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH 'olduniquekey'").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH 'replacementuniquekey'").fetchone()[0] == 1


def test_legacy_schema_migration_preserves_vectors_and_repairs_fts_identity(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    file_id, chunk_id = "legacy-file", "legacy-chunk"
    vector = axis().astype("<f4").tobytes()
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE files(id TEXT PRIMARY KEY,path TEXT UNIQUE NOT NULL,project TEXT NOT NULL,
                extension TEXT,size INTEGER,mtime_ns INTEGER,sha256 TEXT,status TEXT,error TEXT,
                chunk_count INTEGER DEFAULT 0,updated_at TEXT,content TEXT NOT NULL DEFAULT '');
            CREATE TABLE chunks(id TEXT PRIMARY KEY,file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                ordinal INTEGER,start_line INTEGER,end_line INTEGER,content TEXT NOT NULL,
                embedding_text TEXT NOT NULL,sha256 TEXT,UNIQUE(file_id,ordinal));
            CREATE TABLE vectors(chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                model TEXT NOT NULL,revision TEXT NOT NULL,dimensions INTEGER NOT NULL,vector BLOB NOT NULL,created_at TEXT);
            CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id UNINDEXED,path,body,tokenize='unicode61');
        """)
        db.execute("INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (file_id,"legacy.py","legacy",".py",13,1,"legacy-hash","indexed",None,1,"before","legacyneedle"))
        db.execute("INSERT INTO chunks(rowid,id,file_id,ordinal,start_line,end_line,content,embedding_text,sha256) VALUES (?,?,?,?,?,?,?,?,?)",
                   (42,chunk_id,file_id,0,1,1,"legacyneedle","File: legacy.py\nLines: 1-1\nlegacyneedle","chunk-hash"))
        db.execute("INSERT INTO vectors VALUES (?,?,?,?,?,?)", (chunk_id,MODEL,REVISION,DIMENSIONS,vector,"before"))
        db.execute("INSERT INTO chunk_fts(rowid,chunk_id,path,body) VALUES (7,?,?,?)", (chunk_id,"legacy py","legacyneedle"))
    store = Store(path)
    assert store.stats()["vectors"] == 1
    with store.connect() as db:
        assert db.execute("SELECT vector FROM vectors WHERE chunk_id=?", (chunk_id,)).fetchone()[0] == vector
        assert db.execute("SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH 'legacyneedle'").fetchone()[0] == 1
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    assert Store(path).stats()["vectors"] == 1  # Migration is idempotent.
    assert store.prune_files([]) == 1
    assert store.stats()["vectors"] == store.stats()["chunks"] == 0
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM chunk_fts").fetchone()[0] == 0


def audit_batch(chunk_id, text="hashed-input-only"):
    return dict(status="success", phase="embed", items=1, characters=len(text), http_status=200,
                elapsed_ms=12.5, error_code=None, boot_id="fake-audited-boot",
                input_hashes=[{"chunk_id": chunk_id, "sha256": hashlib.sha256(text.encode()).hexdigest()}])


def test_successful_vector_and_batch_audit_commit_together(ingestion_engine):
    _, store, _, _ = ingestion_engine
    _, chunk_id = add_file(store, "audit.py", "auditcontent")
    job = store.job_create("embed")
    batch = store.vector_put_many_with_audit([(chunk_id, axis())], MODEL, REVISION,
                                             job_id=job["id"], batch=audit_batch(chunk_id))
    assert batch["status"] == "success" and batch["job_id"] == job["id"]
    assert batch["items"] == 1 and batch["http_status"] == 200
    assert store.stats()["vectors"] == 1
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM batches WHERE job_id=? AND status='success'", (job["id"],)).fetchone()[0] == 1


def test_failed_success_audit_rolls_back_vector_commit(ingestion_engine):
    _, store, _, _ = ingestion_engine
    _, chunk_id = add_file(store, "audit.py", "auditcontent")
    job = store.job_create("embed")
    with store.connect() as db:
        db.executescript("CREATE TRIGGER reject_audit BEFORE INSERT ON batches WHEN NEW.status='success' "
                         "BEGIN SELECT RAISE(ABORT,'audit transaction rejected'); END;")
    with pytest.raises(sqlite3.IntegrityError):
        store.vector_put_many_with_audit([(chunk_id, axis())], MODEL, REVISION,
                                         job_id=job["id"], batch=audit_batch(chunk_id))
    assert store.stats()["vectors"] == 0
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM batches").fetchone()[0] == 0


def test_failed_vector_insert_cannot_leave_a_successful_audit(ingestion_engine):
    _, store, _, _ = ingestion_engine
    job = store.job_create("embed")
    with pytest.raises(sqlite3.IntegrityError):
        store.vector_put_many_with_audit([("missing-chunk", axis())], MODEL, REVISION,
                                         job_id=job["id"], batch=audit_batch("missing-chunk"))
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM batches").fetchone()[0] == 0
    assert store.stats()["vectors"] == 0


def test_remote_failure_audit_does_not_persist_raw_exception_or_input(ingestion_engine):
    root, store, encoder, engine = ingestion_engine
    content = "uniquesourcedatathatmustnotappearinaudit\n"
    (root / "safe.py").write_text(content, encoding="utf-8")
    encoder.failure = RuntimeError("PRIVATE_REMOTE_RESPONSE sk-proj-" + "X" * 40)
    job = run_job(store, engine)
    assert job["status"] == "failed"
    assert store.stats()["vectors"] == 0
    with store.connect() as db:
        batches = [dict(r) for r in db.execute("SELECT * FROM batches WHERE job_id=?", (job["id"],))]
        errors = [dict(r) for r in db.execute("SELECT * FROM job_errors WHERE job_id=?", (job["id"],))]
    assert batches and any(row["status"] == "failed" for row in batches)
    payload = json.dumps({"job": job, "batches": batches, "errors": errors})
    assert "PRIVATE_REMOTE_RESPONSE" not in payload and "sk-proj-" not in payload
    assert content.strip() not in payload
    assert all("input_text" not in row and "source_text" not in row for row in batches)


def test_batch_hashes_identify_exact_embedding_input_including_header(ingestion_engine):
    root, store, encoder, engine = ingestion_engine
    (root / "source.py").write_text("uniquebody\n", encoding="utf-8")
    job = run_job(store, engine)
    detail = store.job_detail(job["id"])
    successful = [batch for batch in detail["batches"] if batch["status"] == "success"]
    assert len(successful) == 1
    batch = successful[0]
    with store.connect() as db:
        chunks = {row["id"]: dict(row) for row in db.execute("SELECT * FROM chunks")}
    assert batch["items"] == len(batch["input_hashes"]) == 1
    for entry in batch["input_hashes"]:
        chunk = chunks[entry["chunk_id"]]
        assert entry["sha256"] == hashlib.sha256(chunk["embedding_text"].encode()).hexdigest()
        assert entry["sha256"] != hashlib.sha256(chunk["content"].encode()).hexdigest()
    assert batch["characters"] == sum(len(text) for _, texts, _ in encoder.doc_calls for text in texts)


def test_permission_error_defers_old_unconfirmed_snapshots_instead_of_uploading_them(ingestion_engine, monkeypatch):
    root, store, encoder, engine = ingestion_engine
    (root / "confirmed.py").write_text("confirmedsource\n", encoding="utf-8")
    add_file(store, "unreadable/old.py", "notconfirmedthisscan\n")
    confirmed = next(entry for entry in ingest.scan(root) if entry["file"]["path"] == "confirmed.py")
    denied = {"file": ingest._file_record("unreadable", status="error", error="没有读取文件或数据库的权限"),
              "chunks": [], "reason": "read_error", "is_dir": True}
    monkeypatch.setattr(ingest, "scan", lambda *args, **kwargs: iter([confirmed, denied]))
    job = run_job(store, engine)
    assert job["status"] == "partial"
    assert store.stats()["vectors"] == 1
    assert job["details"]["deferred_unconfirmed"] >= 1
    assert all("notconfirmedthisscan" not in text for _, texts, _ in encoder.doc_calls for text in texts)
    assert any(chunk["path"] == "unreadable/old.py" for chunk in store.pending_chunks())


@pytest.fixture
def audit_api(ingestion_engine):
    from omniembed.code_search.app import create_app

    root, store, _, engine = ingestion_engine
    with TestClient(create_app(root=root, engine=engine), base_url="http://127.0.0.1:8790") as client:
        yield client, store


def test_job_detail_batches_and_errors_are_job_scoped(audit_api):
    client, store = audit_api
    first, second = store.job_create("embed"), store.job_create("embed")
    store.batch_record(first["id"], **audit_batch("first-chunk"))
    store.batch_record(second["id"], **audit_batch("second-chunk"))
    store.error_record(first["id"], "blocked/file.py", "read_error", is_dir=False)
    detail = client.get("/api/code/jobs/" + first["id"])
    assert detail.status_code == 200
    result = detail.json()
    assert result["job"]["id"] == first["id"]
    assert len(result["batches"]) == 1 and result["batches"][0]["job_id"] == first["id"]
    assert len(result["errors"]) == 1 and result["errors"][0]["path"] == "blocked/file.py"
    batches = client.get("/api/code/batches", params={"job_id": first["id"], "limit": 1})
    assert batches.status_code == 200
    assert len(batches.json()["items"]) == 1
    assert batches.json()["items"][0]["job_id"] == first["id"]
    assert client.get("/api/code/jobs/missing").status_code == 404


def test_exclusion_browser_filters_and_never_returns_source_content(audit_api):
    client, store = audit_api
    file_id, _ = add_file(store, "project/private.py", "notreturned", project="project")
    record = store.file_by_path("project/private.py")
    store.replace_file({**record, "status": "skipped", "error": "sensitive_content", "content": ""}, [])
    add_file(store, "project/visible.py", "visible", project="project")
    response = client.get("/api/code/exclusions", params={"reason": "sensitive_content", "project": "project", "q": "private"})
    assert response.status_code == 200
    value = response.json()
    assert value["total"] == 1 and value["items"][0]["id"] == file_id
    assert value["items"][0]["error"] == "sensitive_content"
    assert "content" not in value["items"][0]
    assert any(row["reason"] == "sensitive_content" and row["count"] >= 1 for row in value["reasons"])
    assert client.get("/api/code/exclusions", params={"q": "%' OR 1=1 --"}).json()["total"] == 0


@pytest.mark.parametrize("route,params", [("/api/code/batches", {"limit": 0}),
                                          ("/api/code/batches", {"limit": 1001}),
                                          ("/api/code/exclusions", {"limit": 1001}),
                                          ("/api/code/exclusions", {"offset": -1}),
                                          ("/api/code/exclusions", {"q": "q" * 4097})])
def test_audit_api_query_bounds_and_same_origin_protection(audit_api, route, params):
    client, _ = audit_api
    assert client.get(route, params=params).status_code == 422
    assert client.get(route, headers={"Origin": "https://evil.example"}).status_code == 403


def test_overview_scan_summary_is_a_structured_object(audit_api):
    client, store = audit_api
    store.meta_set("scan_summary", {"files": 3, "skip_reasons": {"too_large": 1}})
    value = client.get("/api/code/overview").json()["scan_summary"]
    assert isinstance(value, dict) and value["files"] == 3
    assert value["skip_reasons"]["too_large"] == 1
