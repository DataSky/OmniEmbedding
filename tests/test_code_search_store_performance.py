"""Index-only metadata reads, exact coverage and legacy-snapshot compatibility."""
from contextlib import contextmanager
import sqlite3

import pytest

from omniembed.code_search.store import DIMENSIONS, MODEL, REVISION, Store
from test_code_search import add_file, axis


def captured_selects(store, monkeypatch, operation):
    original_connect = store.connect
    statements = []

    @contextmanager
    def connect():
        with original_connect() as db:
            db.set_trace_callback(statements.append)
            try:
                yield db
            finally:
                db.set_trace_callback(None)

    with monkeypatch.context() as patch:
        patch.setattr(store, "connect", connect)
        result = operation()
    with original_connect() as db:
        plans = [(sql, [r[3] for r in db.execute("EXPLAIN QUERY PLAN " + sql)])
                 for sql in statements if sql.lstrip().upper().startswith("SELECT ")]
    return result, plans


@pytest.mark.parametrize("scope", [{}, {"project": "Alpha"}, {"extension": ".PY"},
                                    {"project": "Alpha", "extension": ".py"}])
def test_stats_and_coverage_use_covering_indexes_for_large_payload_metadata(tmp_path, monkeypatch, scope):
    store = Store(tmp_path/"metadata.sqlite3")
    for index in range(24):
        project = "Alpha" if index % 2 else "Beta"
        suffix = ".md" if index % 3 else ".py"
        add_file(store, f"{project}/file{index}{suffix}", "payload evidence " * 1200,
                 project=project, vector=axis() if index % 4 else None)
    stats, stats_plans = captured_selects(store, monkeypatch, store.stats)
    coverage, coverage_plans = captured_selects(store, monkeypatch, lambda: store.coverage(**scope))
    assert stats["files"] == stats["chunks"] == 24 and stats["vectors"] == 18
    expected = [i for i in range(24) if (not scope.get("project") or i % 2)
                and (not scope.get("extension") or i % 3 == 0)]
    assert coverage == {"chunks": len(expected), "vectors": sum(i % 4 != 0 for i in expected)}
    if not scope:
        assert len(coverage_plans) == 2
        assert all("JOIN" not in sql.upper() for sql, _ in coverage_plans)
    for _, details in stats_plans + coverage_plans:
        table_access = [d for d in details if d.startswith(("SCAN ", "SEARCH "))]
        assert table_access and all("USING COVERING INDEX" in d for d in table_access), details


def test_unfiltered_coverage_counts_share_a_snapshot_during_committed_writer(tmp_path, monkeypatch):
    store = Store(tmp_path/"concurrent.sqlite3")
    add_file(store, "original.py", "pending original")
    original_connect, pending = store.connect, [True]

    class Cursor:
        def __init__(self, cursor):
            self.cursor = cursor

        def fetchone(self):
            row = self.cursor.fetchone()
            if pending:
                pending.pop()
                add_file(store, "committed.py", "new vectorized file", vector=axis())
            return row

    class Connection:
        def __init__(self, db):
            self.db = db

        def execute(self, sql, *args):
            cursor = self.db.execute(sql, *args)
            return Cursor(cursor) if sql == "SELECT count(*) FROM chunks" else cursor

        def __getattr__(self, name):
            return getattr(self.db, name)

    @contextmanager
    def connect():
        with original_connect() as db:
            yield Connection(db)

    monkeypatch.setattr(store, "connect", connect)
    assert store.coverage() == {"chunks": 1, "vectors": 0}
    assert store.coverage() == {"chunks": 2, "vectors": 1}


def test_writer_foreign_keys_and_cascades_keep_fast_coverage_exact(tmp_path):
    store = Store(tmp_path/"constraints.sqlite3")
    fid, _ = add_file(store, "source.py", "valid source", vector=axis())
    with store.connect() as db:
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO vectors VALUES (?,?,?,?,?,?)", ("missing", MODEL, REVISION, DIMENSIONS, axis().tobytes(), "test"))
        db.execute("DELETE FROM files WHERE id=?", (fid,))
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    assert store.coverage() == {"chunks": 0, "vectors": 0}


def test_legacy_orphans_cannot_turn_pending_chunks_into_false_complete_coverage(tmp_path):
    path = tmp_path/"legacy.sqlite3"
    store = Store(path)
    add_file(store, "pending.py", "one valid pending chunk")
    # Simulate an older/external writer which disabled foreign-key enforcement.
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO vectors VALUES (?,?,?,?,?,?)", ("orphan-vector", MODEL, REVISION, DIMENSIONS, axis().tobytes(), "test"))
        db.execute("INSERT INTO chunks(id,file_id,ordinal,start_line,end_line,content,embedding_text) VALUES (?,?,?,?,?,?,?)",
                   ("orphan-chunk", "missing-file", 0, 1, 1, "unlinked", "unlinked"))
        db.execute("INSERT INTO vectors VALUES (?,?,?,?,?,?)", ("orphan-chunk", MODEL, REVISION, DIMENSIONS, axis().tobytes(), "test"))
    reopened = Store(path)
    assert reopened._coverage_integrity_verified is False
    assert reopened.coverage() == {"chunks": 1, "vectors": 0}


@pytest.mark.parametrize("invalid_id", [None, "", "   ", 7])
@pytest.mark.parametrize("writer", ["single", "many", "audited"])
def test_vector_writers_reject_invalid_ids_before_any_commit(tmp_path, invalid_id, writer):
    store = Store(tmp_path/"invalid-identifiers.sqlite3")
    add_file(store, "pending.py", "valid pending chunk")
    job = store.job_create("embed")
    with pytest.raises(ValueError, match="分块标识必须为非空字符串"):
        if writer == "single":
            store.vector_put(invalid_id, axis())
        elif writer == "many":
            store.vector_put_many([(invalid_id, axis())])
        else:
            store.vector_put_many_with_audit([(invalid_id, axis())], job_id=job["id"], batch={
                "status": "success", "items": 1, "characters": 1,
                "input_hashes": [{"chunk_id": "safe-identifier", "sha256": "a" * 64}],
            })
    assert store.coverage() == store.coverage(project="demo") == {"chunks": 1, "vectors": 0}
    assert store.list_batches(job["id"]) == []


def test_initial_legacy_null_vector_falls_back_despite_passing_sqlite_fk_check(tmp_path):
    path = tmp_path/"legacy-null.sqlite3"
    store = Store(path)
    add_file(store, "pending.py", "valid pending chunk")
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("INSERT INTO vectors VALUES (?,?,?,?,?,?)", (None, MODEL, REVISION, DIMENSIONS, axis().tobytes(), "test"))
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    reopened = Store(path)
    assert reopened._coverage_integrity_verified is False
    assert reopened.coverage() == reopened.coverage(project="demo") == {"chunks": 1, "vectors": 0}


def test_legacy_snapshot_without_new_indexes_remains_read_only_and_exact(tmp_path):
    from scripts.evaluate_code_search import ReadOnlyStore

    path = tmp_path/"frozen.sqlite3"
    store = Store(path)
    _, cid = add_file(store, "Alpha/source.py", "frozen source", project="Alpha")
    with store.connect() as db:
        for name in ("files_summary", "chunks_coverage", "vectors_coverage"):
            db.execute("DROP INDEX " + name)
    frozen = ReadOnlyStore(path)
    try:
        assert frozen.coverage() == frozen.coverage(project="Alpha") == {"chunks": 1, "vectors": 0}
        store.vector_put(cid, axis())
        assert frozen.coverage() == {"chunks": 1, "vectors": 0}
        assert frozen.db.execute("PRAGMA query_only").fetchone()[0] == 1
        assert frozen.db.in_transaction
    finally:
        frozen.close()
    assert store.coverage() == {"chunks": 1, "vectors": 1}


def test_index_migration_preserves_source_vectors_and_is_idempotent(tmp_path):
    path = tmp_path/"migration.sqlite3"
    store = Store(path)
    fid, cid = add_file(store, "source.py", "source must not change", vector=axis())
    with store.connect() as db:
        original = tuple(db.execute("SELECT model,revision,dimensions,vector,created_at FROM vectors WHERE chunk_id=?", (cid,)).fetchone())
        for name in ("files_summary", "chunks_coverage", "vectors_coverage"):
            db.execute("DROP INDEX " + name)
    for _ in range(2):
        migrated = Store(path)
        assert migrated.file_detail(fid)["content"] == "source must not change"
        assert migrated.coverage() == {"chunks": 1, "vectors": 1}
        with migrated.connect() as db:
            assert tuple(db.execute("SELECT model,revision,dimensions,vector,created_at FROM vectors WHERE chunk_id=?", (cid,)).fetchone()) == original
            names = {r[1] for table in ("files", "chunks", "vectors") for r in db.execute("PRAGMA index_list("+table+")")}
            assert {"files_summary", "chunks_coverage", "vectors_coverage"} <= names
