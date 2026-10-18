"""Fixed benchmark building never mutates production or uses a real GPU."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omniembed.code_search.ingest import CHUNK_POLICY, _file_record, chunk_text
from omniembed.code_search.store import DIMENSIONS, Store
from scripts import build_code_search_benchmark as benchmark


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def add(store, root, path, text):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text)
    record = _file_record(path, file.stat(), status="indexed")
    record.update(content=text, sha256=sha(text), policy=CHUNK_POLICY)
    chunks = chunk_text(text, path)
    store.replace_file(record, chunks)
    return record, chunks


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    store = Store(tmp_path / "production.sqlite3", root=root)
    sentinel = "PRIVATE_SOURCE_BODY_DO_NOT_LOG_e54bdb"
    text = "architecture owner\ncontroller pipeline\nSQLite boundary\n" + sentinel + "\n"
    gold_file, gold_chunks = add(store, root, "project-a/main.md", text)
    for index in range(30):
        project = ("project-a", "project-b", "project-c")[index % 3]
        body = f"ordinary distractor record {index}\n"
        if index < 3:
            body += "architecture alternate design\n"
        add(store, root, f"{project}/f{index:02d}.md", body)
    add(store, root, "project-a/.build/never.swift", "architecture generated library\n")
    add(store, root, "project-b/.data/history.txt", "architecture private runtime\n")
    add(store, root, "project-c/unsafe.py", 'AWS_SECRET_ACCESS_KEY="' + "N" * 40 + '"\n')
    gold = {"path": gold_file["path"], "start_line": 1, "end_line": 3,
            "source_sha256": gold_file["sha256"], "evidence_sha256": sha("".join(text.splitlines(keepends=True)[:3]))}
    fixture = {"version": 1, "queries": [
        {"id": "development-question", "split": "development", "query": "architecture", "gold": [deepcopy(gold)]},
        {"id": "holdout-question", "split": "holdout", "query": "controller pipeline", "gold": [deepcopy(gold)]},
    ]}
    return SimpleNamespace(root=root, store=store, fixture=fixture, gold_file=gold_file, gold_chunks=gold_chunks,
                           sentinel=sentinel, output=tmp_path / "benchmark")


def build(source, **kwargs):
    return benchmark.build_benchmark(source.store.path, source.fixture, source.output, max_chunks=12, **kwargs)


def test_default_build_preserves_gold_identity_and_creates_independent_subset(source):
    before = source.store.stats()
    manifest = build(source)
    selected = Store(manifest["database"])
    assert manifest["status"] == "built" and manifest["vectors"] == 0
    assert manifest["chunks"] == 12 and manifest["whole_ai_code_quality"] is False
    assert manifest["benchmark_subset"] is True and manifest["production_index_modified"] is False
    assert source.store.stats()["chunks"] == before["chunks"]
    assert source.store.stats()["vectors"] == before["vectors"] == 0
    detail = selected.chunk_detail(source.gold_chunks[0]["id"])
    for field in ("id", "sha256", "ordinal", "start_line", "end_line", "content", "embedding_text"):
        assert detail["chunk"][field] == source.gold_chunks[0][field]
    assert detail["file"]["sha256"] == source.gold_file["sha256"]
    assert detail["file"]["chunk_count"] == 1
    assert len(manifest["gold"]) == 2
    assert all(item["matching_chunk_ids"] for item in manifest["gold"])
    assert source.output.stat().st_mode & 0o777 == 0o700
    assert Path(manifest["database"]).stat().st_mode & 0o777 == 0o600


def test_sampling_is_deterministic_project_stratified_and_excludes_old_artifacts(source):
    first = build(source, seed=17, hard_negatives=1)
    source.output = source.output.with_name("benchmark-second")
    second = build(source, seed=17, hard_negatives=1)
    assert first["corpus_selection_sha256"] == second["corpus_selection_sha256"]
    paths = [item["path"] for item in first["selection"]]
    assert {Path(path).parts[0] for path in paths} == {"project-a", "project-b", "project-c"}
    assert not any("/.build/" in path or "/.data/" in path or path.endswith("unsafe.py") for path in paths)
    assert first["negative_strategy"]["split"] == "development"
    assert first["negative_strategy"]["human_judged_negative"] is False
    events = [json.loads(line) for line in (Path(first["database"]).parent / "events.jsonl").read_text().splitlines()]
    negative = next(event for event in events if event["stage"] == "development_candidates_selected")
    assert {query["case_id"] for query in negative["queries"]} == {"development-question"}


def test_no_output_is_overwritten(source):
    source.output.mkdir()
    marker = source.output / "keep.txt"
    marker.write_text("existing artifact")
    with pytest.raises(FileExistsError):
        build(source)
    assert marker.read_text() == "existing artifact"


@pytest.mark.parametrize("change", ["missing", "bad_hash"])
def test_missing_or_invalid_gold_cannot_be_replaced_by_random_fill(source, change):
    fixture = deepcopy(source.fixture)
    if change == "missing":
        fixture["queries"][0]["gold"][0]["source_sha256"] = "0" * 64
    else:
        fixture["queries"][0]["gold"][0]["evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        benchmark.build_benchmark(source.store.path, fixture, source.output, max_chunks=12)
    assert not (source.output / "benchmark.sqlite3").exists()


def test_encoding_sends_only_original_chunks_and_atomically_audits_every_input(source):
    class MockEncoder:
        boot_id = "benchmark-test-boot"

        def __init__(self):
            self.calls = []

        def docs(self, titles, texts, bs=16):
            assert not any(titles)
            assert bs <= 16 and len(texts) <= 16 and sum(map(len, texts)) <= 32768
            self.calls.append(list(texts))
            vectors = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
            vectors[:, 0] = 1
            return vectors

    encoder = MockEncoder()
    manifest = build(source, encode=True, encoder=encoder)
    store = Store(manifest["database"])
    assert manifest["status"] == "encoded"
    assert store.stats()["chunks"] == store.stats()["vectors"] == 12
    with source.store.connect() as db:
        original_texts = {row[0] for row in db.execute("SELECT embedding_text FROM chunks")}
    assert all(text in original_texts for call in encoder.calls for text in call)
    with store.connect() as db:
        batches = list(db.execute("SELECT * FROM batches"))
        assert sum(batch["items"] for batch in batches) == 12
        for batch in batches:
            assert batch["status"] == "success"
            for item in json.loads(batch["input_hashes"]):
                text = db.execute("SELECT embedding_text FROM chunks WHERE id=?", (item["chunk_id"],)).fetchone()[0]
                assert item["sha256"] == sha(text)
    assert source.store.stats()["vectors"] == 0
    events = (source.output / "events.jsonl").read_text()
    manifest_text = (source.output / "manifest.json").read_text()
    assert source.sentinel not in events + manifest_text
    assert "AWS_SECRET_ACCESS_KEY" not in events + manifest_text


def test_remote_failure_is_private_and_keeps_production_untouched(source):
    marker = "PRIVATE_REMOTE_FAILURE_NOT_FOR_LOGS"

    class Failing:
        boot_id = "failed-benchmark-test"

        def docs(self, *args, **kwargs):
            raise RuntimeError(marker)

    with pytest.raises(RuntimeError):
        build(source, encode=True, encoder=Failing())
    manifest = json.loads((source.output / "manifest.json").read_text())
    assert manifest["status"] == "encoding_failed" and manifest["vectors"] == 0
    store = Store(manifest["database"])
    batches = store.list_batches(manifest["encoding_job_id"])
    assert len(batches) == 1 and batches[0]["status"] == "failed"
    assert marker not in (source.output / "events.jsonl").read_text() + (source.output / "manifest.json").read_text()
    assert source.store.stats()["vectors"] == 0
