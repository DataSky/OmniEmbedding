"""Execute research against isolated real SQLite snapshots, never a real GPU."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from omniembed.code_search.ingest import CHUNK_POLICY, _file_record, chunk_text
from omniembed.code_search.store import DIMENSIONS, MODEL, REVISION, Store
from scripts import code_search_research as research
from scripts import evaluate_code_search as evaluation


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def axis(index=0):
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    vector[index] = 1
    return vector


def write_document(store, root, path, content, vector=0):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    record = _file_record(path, target.stat(), status="indexed")
    record.update(content=content, sha256=digest(target.read_bytes()), policy=CHUNK_POLICY)
    chunks = chunk_text(content, path)
    store.replace_file(record, chunks)
    if vector is not None:
        store.vector_put_many([(c["id"], axis(vector)) for c in chunks])
    return record, chunks


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    store = Store(tmp_path / "live.sqlite3", root=root)
    body_marker = "SOURCE_BODY_MUST_NOT_APPEAR_IN_RESEARCH_LOGS_8cb6941b"
    content = "OmniVoice architecture controller\naudio pipeline ownership\nSQLite recovery boundary\n" + body_marker + "\n"
    record, chunks = write_document(store, root, "project/architecture.md", content)
    write_document(store, root, "other/decoy.md", "unrelated gardening flowers\n", vector=1)
    gold = {"path": record["path"], "start_line": 1, "end_line": 3,
            "source_sha256": record["sha256"],
            "evidence_sha256": digest("".join(content.splitlines(keepends=True)[:3]))}
    fixture = {"version": 1, "scope": "temporary integration corpus", "queries": [
        {"id": "dev", "query": "OmniVoice architecture", "split": "development", "gold": [deepcopy(gold)]},
        {"id": "hold", "query": "audio pipeline", "split": "holdout", "gold": [deepcopy(gold)]},
        {"id": "none_dev", "query": "nonexistent quasar policy", "split": "development", "gold": []},
        {"id": "none_hold", "query": "nonexistent neutron owner", "split": "holdout", "gold": []},
    ]}
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    vectors = {"model": MODEL, "revision": REVISION, "dimensions": DIMENSIONS,
               "query_sha256": {case["id"]: digest(case["query"]) for case in fixture["queries"]},
               "queries": {case["id"]: axis().tolist() for case in fixture["queries"]}}
    return SimpleNamespace(root=root, store=store, fixture=fixture, fixture_path=fixture_path,
                           vectors=vectors, target=record, chunks=chunks, body_marker=body_marker,
                           output=tmp_path / "evaluations")


def evaluate(corpus, *, fixture=None, vectors=None, label="run", **kwargs):
    return evaluation.run_evaluation(corpus.store.path, corpus.root,
                                     fixture if fixture is not None else corpus.fixture,
                                     corpus.output / label, label=label,
                                     query_vectors=corpus.vectors if vectors is None else vectors, **kwargs)


def test_evaluation_executes_real_routes_and_keeps_source_out_of_logs(corpus):
    value = evaluate(corpus)
    assert len(value["cases"]) == len(corpus.fixture["queries"]) * 3
    assert value["snapshot"]["chunks"] == value["snapshot"]["vectors"] == 2
    for mode in ("bm25", "vector", "hybrid"):
        for split in ("development", "holdout"):
            summary = value["summary"][mode][split]
            assert summary["cases"] == summary["scorable"] == 2
            assert summary["indexed_gold_fraction"] == summary["vector_gold_fraction"] == 1
    assert all(case["status"] == "ok" for case in value["cases"])
    for name in ("stages.jsonl", "summary.json"):
        path = corpus.output / "run" / name
        assert corpus.body_marker not in path.read_text()
        assert path.stat().st_mode & 0o777 == 0o600
    with corpus.store.connect() as db:
        assert db.execute("SELECT count(*) FROM traces").fetchone()[0] == 0


def test_frozen_backup_is_reused_after_live_database_changes(corpus):
    first = evaluate(corpus, label="first")
    frozen = Path(first["artifacts"]["snapshot"])
    write_document(corpus.store, corpus.root, "later/new.md", "later new irrelevant source\n", vector=1)
    second = evaluation.run_evaluation(frozen, corpus.root, corpus.fixture, corpus.output / "second",
                                       query_vectors=corpus.vectors, frozen=True)
    assert corpus.store.stats()["chunks"] == 3
    assert first["snapshot"]["id"] == second["snapshot"]["id"]
    assert first["summary"] == second["summary"]
    assert second["snapshot"]["chunks"] == 2


@pytest.mark.parametrize("field,value", [("model", "different-model"), ("revision", "different-revision"),
                                         ("dimensions", 128)])
def test_query_vector_model_metadata_is_verified_before_scoring(corpus, field, value):
    vectors = deepcopy(corpus.vectors)
    vectors[field] = value
    with pytest.raises(ValueError):
        evaluate(corpus, vectors=vectors)


@pytest.mark.parametrize("bad_vector", [[1.0], [0.0] * DIMENSIONS, [float("nan")] * DIMENSIONS])
def test_query_vector_shape_norm_and_finiteness_are_verified(corpus, bad_vector):
    vectors = deepcopy(corpus.vectors)
    vectors["queries"]["hold"] = bad_vector
    with pytest.raises(ValueError):
        evaluate(corpus, vectors=vectors)


def test_query_vector_cannot_be_reused_after_same_id_question_changes(corpus):
    fixture = deepcopy(corpus.fixture)
    fixture["queries"][0]["query"] = "a different question keeping the previous case id"
    with pytest.raises(ValueError):
        evaluate(corpus, fixture=fixture)


def test_legacy_query_vectors_without_query_hash_binding_are_rejected(corpus):
    vectors = deepcopy(corpus.vectors)
    vectors.pop("query_sha256")
    with pytest.raises(ValueError):
        evaluate(corpus, vectors=vectors)


def test_missing_query_vector_stays_in_case_count_and_blocks_gate(corpus):
    vectors = deepcopy(corpus.vectors)
    vectors["queries"].pop("hold")
    value = evaluate(corpus, vectors=vectors)
    case = next(c for c in value["cases"] if c["id"] == "hold" and c["mode"] == "hybrid")
    assert case["status"] == "missing_offline_query_vector" and case["metrics"] is None
    summary = value["summary"]["hybrid"]["holdout"]
    assert summary["cases"] == 2 and summary["scorable"] == 1 and summary["skipped"] == 1
    gate = research.acceptance_gate(value, value)
    assert not gate["accepted"] and "unscored_or_invalid_cases" in gate["reasons"]


def test_missing_indexed_gold_is_visible_and_blocks_gate(corpus):
    corpus.store.prune_files(["other/decoy.md"])
    value = evaluate(corpus)
    assert len(value["cases"]) == 12
    for split in ("development", "holdout"):
        summary = value["summary"]["hybrid"][split]
        assert summary["cases"] == summary["scorable"] == 2
        assert summary["indexed_gold_fraction"] == summary["vector_gold_fraction"] == 0
    gate = research.acceptance_gate(value, value)
    assert not gate["accepted"]
    assert "holdout:incomplete_indexed_gold_fraction" in gate["reasons"]


def test_unvectorized_gold_is_not_removed_from_coverage_denominator(corpus):
    with corpus.store.connect() as db:
        db.execute("DELETE FROM vectors WHERE chunk_id=?", (corpus.chunks[0]["id"],))
    value = evaluate(corpus)
    summary = value["summary"]["hybrid"]["holdout"]
    assert summary["indexed_gold_fraction"] == 1 and summary["vector_gold_fraction"] == 0
    assert not research.acceptance_gate(value, value)["accepted"]


def test_invalid_live_source_evidence_stays_visible_and_blocks_gate(corpus):
    (corpus.root / corpus.target["path"]).write_text("changed source invalidates original evidence\n")
    value = evaluate(corpus)
    invalid = [c for c in value["cases"] if c["id"] in {"dev", "hold"}]
    assert len(invalid) == 6 and all(c["status"] == "invalid_gold_evidence" for c in invalid)
    assert value["summary"]["hybrid"]["holdout"]["cases"] == 2
    assert value["summary"]["hybrid"]["holdout"]["skipped"] == 1
    assert not research.acceptance_gate(value, value)["accepted"]


def test_retrieval_error_cannot_become_a_correct_no_answer_case(corpus, monkeypatch):
    original = evaluation.Search.run

    def fail_no_answer(self, query, **kwargs):
        if query.startswith("nonexistent"):
            raise ValueError("unavailable retrieval must not be scored as an answer")
        return original(self, query, **kwargs)

    monkeypatch.setattr(evaluation.Search, "run", fail_no_answer)
    value = evaluate(corpus)
    failed = [c for c in value["cases"] if c["id"].startswith("none_")]
    assert failed and all(c["status"] != "ok" and c["metrics"] is None for c in failed)
    assert not research.acceptance_gate(value, value)["accepted"]


def test_bm25_equivalent_zero_vector_weight_cannot_count_as_hybrid(corpus):
    value = evaluate(corpus, search_parameters={"vector_weight": 0.0})
    cases = [c for c in value["cases"] if c["mode"] == "hybrid"]
    assert cases and all(c["status"] != "ok" for c in cases)
    assert not research.acceptance_gate(value, value)["accepted"]


def research_args(corpus, *extra):
    return ["--db", str(corpus.store.path), "--root", str(corpus.root),
            "--fixture", str(corpus.fixture_path), "--output-dir", str(corpus.output), *extra]


def test_wrapper_runs_real_frozen_trials_with_fake_autodl_and_private_logs(corpus, monkeypatch, capsys):
    # Match normal script import semantics without spawning a network process.
    monkeypatch.setitem(sys.modules, "evaluate_code_search", evaluation)
    import omniembed.remote_encoder as remote
    calls = []

    class FakeRemote:
        boot_id = "test-research-boot"

        def __init__(self, **kwargs):
            assert kwargs["model_id"] == MODEL and kwargs["revision"] == REVISION
            self._url = "legacy-address-must-be-cleared"

        def query(self, texts, bs=16):
            assert self._url is None and bs <= 16
            calls.append(list(texts))
            return np.stack([axis() for _ in texts])

    monkeypatch.setattr(remote, "RemoteEncoder", FakeRemote)
    original = evaluation.run_evaluation
    snapshots = []

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        snapshots.append(result["snapshot"]["id"])
        if len(snapshots) == 1:
            write_document(corpus.store, corpus.root, "later/new.md", "later irrelevant source\n", vector=1)
        return result

    monkeypatch.setattr(evaluation, "run_evaluation", observed)
    assert research.main(research_args(corpus, "--encode-queries", "--wait-for-index", "--max-trials", "3")) == 0
    latest = json.loads((corpus.output / "latest.json").read_text())
    directory = Path(latest["path"])
    assert len(snapshots) >= 3 and len(set(snapshots)) == 1
    assert len(calls) == 1 and len(calls[0]) == 4
    assert latest["gate"]["selection_split"] == "development"
    assert latest["gate"]["acceptance_split"] == "holdout"
    assert latest["snapshot_id"] == snapshots[0]
    trials = [json.loads(p.read_text()) for p in sorted(directory.glob("trial-*/summary.json"))]
    expected = research.select_trial(trials)
    assert latest["selected"] == expected["label"]
    text_artifacts = list(directory.rglob("*.json")) + list(directory.rglob("*.jsonl"))
    assert corpus.body_marker not in capsys.readouterr().out
    assert all(corpus.body_marker not in p.read_text() for p in text_artifacts)
    assert directory.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in text_artifacts)
    with corpus.store.connect() as db:
        assert db.execute("SELECT count(*) FROM traces").fetchone()[0] == 0


def test_wrapper_freezes_fixture_and_scores_holdout_only_after_selection(corpus, monkeypatch):
    monkeypatch.setitem(sys.modules, "evaluate_code_search", evaluation)
    import omniembed.remote_encoder as remote
    original_evaluate = evaluation.run_evaluation
    original_select = research.select_trial
    observations = []
    selected = [False]

    class FakeRemote:
        boot_id = "fixture-freeze-test"

        def __init__(self, **kwargs):
            self._url = None

        def query(self, texts, bs=16):
            return np.stack([axis() for _ in texts])

    def select(*args, **kwargs):
        selected[0] = True
        return original_select(*args, **kwargs)

    def observe(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        splits = {case["split"] for case in result["cases"]}
        observations.append({"fixture_sha256": result["fixture_sha256"], "splits": splits,
                             "selected_already": selected[0], "cases": result["cases"]})
        if len(observations) == 1:
            changed = deepcopy(corpus.fixture)
            changed["queries"][0]["query"] = "changed input after the initial trial"
            corpus.fixture_path.write_text(json.dumps(changed))
        return result

    monkeypatch.setattr(remote, "RemoteEncoder", FakeRemote)
    monkeypatch.setattr(research, "select_trial", select)
    monkeypatch.setattr(evaluation, "run_evaluation", observe)
    assert research.main(research_args(corpus, "--encode-queries", "--max-trials", "2")) == 0
    assert len({row["fixture_sha256"] for row in observations}) == 1
    assert any("holdout" in row["splits"] for row in observations)
    assert all(row["selected_already"] for row in observations if "holdout" in row["splits"])
    assert all(row["splits"] == {"development"} for row in observations if not row["selected_already"])
    assert all(case["query"] != "changed input after the initial trial"
               for row in observations for case in row["cases"])


def test_offline_evaluation_rejects_credential_query_before_writing_logs(corpus):
    fixture = deepcopy(corpus.fixture)
    credential = "Bearer " + "R" * 28
    fixture["queries"][0]["query"] = credential
    with pytest.raises(ValueError):
        evaluation.run_evaluation(corpus.store.path, corpus.root, fixture, corpus.output / "sensitive",
                                  modes=("bm25",), query_vectors=None)
    text_outputs = list(corpus.output.rglob("*.json")) + list(corpus.output.rglob("*.jsonl"))
    assert all(credential not in p.read_text() for p in text_outputs)


@pytest.mark.parametrize("directory", [".build", ".swiftpm", "Pods", ".svelte-kit", ".vite"])
def test_generated_dependency_trees_are_excluded_but_outputs_remains_source(corpus, directory):
    from omniembed.code_search.ingest import excluded_relative_path, scan
    hidden = corpus.root / "project" / directory
    hidden.mkdir(parents=True)
    (hidden / "unwanted.swift").write_text("never index this generated source\n")
    outputs = corpus.root / "project" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "acceptance.md").write_text("valid acceptance evidence\n")
    rows = list(scan(corpus.root))
    assert excluded_relative_path(f"project/{directory}/unwanted.swift")
    assert not any(row["file"]["path"].endswith("unwanted.swift") for row in rows)
    assert any(row["file"]["path"] == "project/outputs/acceptance.md" and row["reason"] is None for row in rows)


def test_wrapper_waits_for_complete_coverage_instead_of_running_partial_trial(corpus, monkeypatch):
    with corpus.store.connect() as db:
        db.execute("DELETE FROM vectors WHERE chunk_id=?", (corpus.chunks[0]["id"],))
    moments = iter((0.0, 2.0, 3.0))
    monkeypatch.setattr(research, "time", SimpleNamespace(monotonic=lambda: next(moments),
                                                        perf_counter=time.perf_counter, sleep=lambda _: None))
    assert research.main(research_args(corpus, "--wait-for-index", "--max-wait-seconds", "1")) == 1
    failures = list(corpus.output.glob("research-*/failure.json"))
    assert len(failures) == 1 and json.loads(failures[0].read_text())["error_code"] == "TimeoutError"
    assert list(corpus.output.glob("research-*/trial-*/summary.json")) == []


def test_wrapper_remote_error_does_not_leak_response_or_credential(corpus, monkeypatch, capsys):
    import omniembed.remote_encoder as remote
    marker = "PRIVATE_REMOTE_DIAGNOSTIC_THAT_MUST_NOT_BE_PERSISTED"
    credential = "sk-proj-" + "R" * 40

    class FailingRemote:
        def __init__(self, **kwargs):
            self._url = None

        def query(self, *args, **kwargs):
            raise RuntimeError(marker + " " + credential)

    monkeypatch.setattr(remote, "RemoteEncoder", FailingRemote)
    assert research.main(research_args(corpus, "--encode-queries", "--max-trials", "1")) == 1
    payload = capsys.readouterr().out
    payload += "".join(p.read_text() for p in corpus.output.rglob("*.json"))
    payload += "".join(p.read_text() for p in corpus.output.rglob("*.jsonl"))
    assert marker not in payload and credential not in payload and corpus.body_marker not in payload


@pytest.mark.parametrize("invalid", [
    "sensitive_id", "sensitive_query", "sensitive_scope", "sensitive_gold_path",
    "sensitive_annotation", "sensitive_metadata_key", "too_many", "long_query",
    "empty_queries", "duplicate_id", "invalid_id", "blank_query", "invalid_split",
    "invalid_split_type", "invalid_gold_hash", "invalid_gold_lines", "invalid_gold_type",
    "nonfinite_metadata",
])
def test_wrapper_rejects_invalid_fixture_before_any_side_effect(corpus, monkeypatch, capsys, invalid):
    import omniembed.remote_encoder as remote

    fixture = deepcopy(corpus.fixture)
    credential = "Bearer " + "R" * 28  # Synthetic, never a runtime credential.
    first = fixture["queries"][0]
    if invalid == "sensitive_id":
        first["id"] = credential
    elif invalid == "sensitive_query":
        first["query"] = credential
    elif invalid == "sensitive_scope":
        fixture["scope"] = credential
    elif invalid == "sensitive_gold_path":
        first["gold"][0]["path"] = credential
    elif invalid == "sensitive_annotation":
        first["annotation"] = {"nested": [credential]}
    elif invalid == "sensitive_metadata_key":
        fixture[credential] = "ordinary annotation"
    elif invalid == "too_many":
        fixture["queries"] = [{**deepcopy(first), "id": f"question-{i}"} for i in range(101)]
    elif invalid == "long_query":
        first["query"] = "Q" * 4001
    elif invalid == "empty_queries":
        fixture["queries"] = []
    elif invalid == "duplicate_id":
        fixture["queries"][1]["id"] = first["id"]
    elif invalid == "invalid_id":
        first["id"] = None
    elif invalid == "blank_query":
        first["query"] = "   "
    elif invalid == "invalid_split":
        first["split"] = "unknown"
    elif invalid == "invalid_split_type":
        first["split"] = []
    elif invalid == "invalid_gold_hash":
        first["gold"][0]["source_sha256"] = "not-a-sha256"
    elif invalid == "invalid_gold_lines":
        first["gold"][0]["start_line"] = True
    elif invalid == "invalid_gold_type":
        first["gold"] = "invalid evidence"
    elif invalid == "nonfinite_metadata":
        fixture["annotation"] = float("nan")
    corpus.fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    calls = {"encoder": 0, "index": 0}

    def forbidden_index(*args, **kwargs):
        calls["index"] += 1
        raise AssertionError("preflight must precede index polling")

    class ForbiddenRemote:
        def __init__(self, **kwargs):
            calls["encoder"] += 1
            raise AssertionError("preflight must precede encoder construction")

    monkeypatch.setattr(research, "index_state", forbidden_index)
    monkeypatch.setattr(remote, "RemoteEncoder", ForbiddenRemote)
    assert research.main(research_args(corpus, "--encode-queries", "--wait-for-index")) == 1
    captured = capsys.readouterr()
    assert credential not in captured.out + captured.err
    assert calls == {"encoder": 0, "index": 0}
    # Even an input copy or failure log would persist the rejected metadata.
    assert not corpus.output.exists()
    event = json.loads(captured.out)
    assert event == {"stage": "failed", "status": "failed", "error_code": "ValueError"}
