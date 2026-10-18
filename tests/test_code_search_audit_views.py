"""The white-box detail must remain coherent while ingestion replaces files."""
from contextlib import contextmanager
from pathlib import Path
import json
import threading

from fastapi.testclient import TestClient
import pytest

from test_code_search import add_file
from omniembed.code_search.store import Store


def inject_after_file_read(store, monkeypatch, callback):
    original_connect = store.connect
    pending = [callback]

    class Cursor:
        def __init__(self, cursor):
            self.cursor = cursor

        def fetchone(self):
            row = self.cursor.fetchone()
            if pending:
                pending.pop()()
            return row

    class Connection:
        def __init__(self, db):
            self.db = db

        def execute(self, query, *args):
            cursor = self.db.execute(query, *args)
            return Cursor(cursor) if query.startswith("SELECT * FROM files WHERE id=") else cursor

        def __getattr__(self, name):
            return getattr(self.db, name)

    @contextmanager
    def connect():
        with original_connect() as db:
            yield Connection(db)

    monkeypatch.setattr(store, "connect", connect)


def test_file_detail_has_one_snapshot_during_concurrent_replacement(tmp_path, monkeypatch):
    store = Store(tmp_path/"snapshot.sqlite3")
    fid, _ = add_file(store, "source.py", "old-source-line")
    inject_after_file_read(store, monkeypatch, lambda:add_file(store, "source.py", "new-source-line"))
    detail = store.file_detail(fid)
    assert detail["content"] == "old-source-line"
    assert detail["chunks"][0]["content"] == "old-source-line"
    assert store.file_detail(fid)["content"] == "new-source-line"


def test_file_detail_is_complete_during_concurrent_pruning(tmp_path, monkeypatch):
    store = Store(tmp_path/"snapshot.sqlite3")
    fid, _ = add_file(store, "source.py", "preserved-view")
    inject_after_file_read(store, monkeypatch, lambda:store.prune_files([]))
    detail = store.file_detail(fid)
    assert detail["content"] == detail["chunks"][0]["content"] == "preserved-view"
    assert store.file_detail(fid) is None


def test_evaluation_api_only_exposes_enumerated_local_runs(tmp_path):
    from test_code_search import FakeCodeEncoder
    from omniembed.code_search.app import create_app
    from omniembed.code_search.ingest import Engine

    root, data = tmp_path/"source", tmp_path/"data"
    root.mkdir()
    store = Store(data/"index.sqlite3", root=root)
    engine = Engine(root, store, encoder=FakeCodeEncoder())
    run = data/"evaluations/research-example/trial-00"
    run.mkdir(parents=True)
    (run/"summary.json").write_text(json.dumps({"summary":{"bm25":{"mrr":.5}}, "cases":[], "scope":"fixture"}))
    (run/"stages.jsonl").write_text(json.dumps({"stage":"complete", "time":"2026-10-18T00:00:00Z"})+"\n")
    outside = tmp_path/"outside"
    outside.mkdir()
    (outside/"summary.json").write_text('{"scope":"must-not-read"}')
    (data/"evaluations/research-external").symlink_to(outside, target_is_directory=True)
    with TestClient(create_app(root=root, data_dir=data, engine=engine), base_url="http://127.0.0.1:8790") as client:
        listing = client.get("/api/code/evaluations").json()
        assert {r["id"] for r in listing["items"]} == {"research-example__trial-00"}
        detail = client.get("/api/code/evaluations/research-example__trial-00")
        assert detail.status_code == 200 and detail.json()["events"][0]["stage"] == "complete"
        assert client.get("/api/code/evaluations/research-external").status_code == 404
        assert client.get("/api/code/evaluations/missing").status_code == 404
        assert client.get("/api/code/evaluations", headers={"Origin":"https://external.example"}).status_code == 403


def test_mismatched_vector_dimensions_are_pending_in_all_views(tmp_path):
    from test_code_search import axis

    store = Store(tmp_path/"space.sqlite3")
    fid, cid = add_file(store, "source.py", "space-check")
    store.vector_put(cid, axis())
    with store.connect() as db:
        db.execute("UPDATE vectors SET dimensions=128")
    assert store.stats()["vectors"] == 0
    assert store.list_files()["items"][0]["vector_count"] == 0
    assert store.file_detail(fid)["chunks"][0]["vector_status"] == "pending"


def test_evaluation_root_cannot_redirect_reads_through_a_symlink(tmp_path):
    from test_code_search import FakeCodeEncoder
    from omniembed.code_search.app import create_app
    from omniembed.code_search.ingest import Engine

    root, data, outside = tmp_path/"source", tmp_path/"data", tmp_path/"external/run"
    root.mkdir()
    store = Store(data/"index.sqlite3", root=root)
    outside.mkdir(parents=True)
    (outside/"summary.json").write_text('{"scope":"external-private-material"}')
    (outside.parent/"latest.json").write_text('{"scope":"external-private-material"}')
    (data/"evaluations").symlink_to(outside.parent, target_is_directory=True)
    engine = Engine(root, store, encoder=FakeCodeEncoder())
    with TestClient(create_app(root=root, data_dir=data, engine=engine), base_url="http://127.0.0.1:8790") as client:
        assert client.get("/api/code/evaluations").json() == {"items":[], "latest":{}}
        assert client.get("/api/code/evaluations/run").status_code == 404


def commit_writer_after_page_count(store, monkeypatch, callback):
    """Commit on another thread/connection between the page's count and rows."""
    original_connect = store.connect
    pending = [callback]

    class Cursor:
        def __init__(self, cursor):
            self.cursor = cursor

        def fetchone(self):
            row = self.cursor.fetchone()
            if pending:
                action, failures = pending.pop(), []

                def write():
                    try:
                        action()
                    except Exception as error:
                        failures.append(error)

                worker = threading.Thread(target=write, daemon=True)
                worker.start()
                worker.join(timeout=5)
                assert not worker.is_alive(), "WAL writer should commit while a page snapshot is open"
                if failures:
                    raise failures[0]
            return row

    class Connection:
        def __init__(self, db):
            self.db = db

        def execute(self, query, *args):
            cursor = self.db.execute(query, *args)
            return Cursor(cursor) if query.startswith("SELECT count(*) FROM files WHERE ") else cursor

        def __getattr__(self, name):
            return getattr(self.db, name)

    @contextmanager
    def connect():
        with original_connect() as db:
            yield Connection(db)

    monkeypatch.setattr(store, "connect", connect)


@pytest.mark.parametrize("browser", ["files", "exclusions"])
@pytest.mark.parametrize("mutation", ["add", "delete"])
def test_paginated_views_keep_count_rows_and_reasons_in_one_snapshot(tmp_path, monkeypatch, browser, mutation):
    store = Store(tmp_path/"pages.sqlite3")

    def insert(path):
        ident, _ = add_file(store, path, "page evidence", project="scope")
        if browser == "exclusions":
            record = store.file_by_path(path)
            store.replace_file({**record, "status": "skipped", "error": "too_large", "content": ""}, [])
        return ident

    original = insert("scope/original.py")
    writer = (lambda: insert("scope/added.py")) if mutation == "add" else (lambda: store.prune_files([]))
    commit_writer_after_page_count(store, monkeypatch, writer)
    read = store.list_files if browser == "files" else store.exclusions
    page = read(project="scope")
    assert page["total"] == 1
    assert [r["id"] for r in page["items"]] == [original]
    if browser == "exclusions":
        assert page["reasons"] == [{"reason": "too_large", "count": 1}]
    updated = read(project="scope")
    assert updated["total"] == len(updated["items"]) == (2 if mutation == "add" else 0)
    if browser == "exclusions":
        assert updated["reasons"] == ([{"reason": "too_large", "count": 2}] if mutation == "add" else [])
