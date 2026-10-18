"""Evaluation scores retrieval against source passages and never calls a GPU."""
from __future__ import annotations

import json

import pytest

from scripts.evaluate_code_search import deduplicated, digest, run_evaluation, score_results
from scripts import evaluate_code_search as evaluator
from omniembed.code_search.store import DIMENSIONS, MODEL, REVISION, Store
from test_code_search import add_file, axis


@pytest.fixture
def evaluation_fixture(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    text = "OmniVoice architecture\nVoice handles audio\nPi runs background work\n"
    source = root / "architecture.md"
    source.write_text(text, encoding="utf-8")
    store = Store(tmp_path / "eval.sqlite3", root=root)
    add_file(store, "architecture.md", text, vector=axis())
    add_file(store, "database.md", "database storage\n", vector=axis(1))
    fixture = {"scope": "test source-grounded retrieval", "queries": [{
        "id": "architecture", "query": "OmniVoice architecture", "split": "development",
        "gold": [{"path": "architecture.md", "start_line": 1, "end_line": 3,
                  "source_sha256": digest(source.read_bytes()), "evidence_sha256": digest(text)}],
    }]}
    vectors = {"model": MODEL, "revision": REVISION, "dimensions": DIMENSIONS,
               "queries": {"architecture": axis().tolist()}, "query_sha256": {"architecture": digest("OmniVoice architecture")}}
    return root, store, fixture, vectors


def test_all_modes_evaluate_source_passages_on_private_backup_without_mutating_input(evaluation_fixture, tmp_path):
    root, store, fixture, vectors = evaluation_fixture
    before = store.stats()
    value = run_evaluation(store.path, root, fixture, tmp_path / "output", query_vectors=vectors)
    for mode in ("bm25", "vector", "hybrid"):
        metrics = value["summary"][mode]["development"]
        assert metrics["scorable"] == 1 and metrics["mrr"] == 1
        assert metrics["recall@1"] == 1 and metrics["indexed_gold_fraction"] == 1
    assert store.stats()["vectors"] == before["vectors"]
    assert store.traces()["items"] == []
    assert value["snapshot"]["path"] != str(store.path)
    stages = [json.loads(line) for line in (tmp_path / "output/stages.jsonl").read_text().splitlines()]
    assert {"snapshot", "source_evidence", "tokenize", "coverage", "candidates", "ranks", "metrics", "complete"} <= {row["stage"] for row in stages}
    assert not any("Voice handles audio" in json.dumps(row) for row in stages)
    rank = next(row for row in stages if row["stage"] == "ranks")
    assert rank["results"][0]["gold_matches"] == [0]


def test_missing_query_vectors_are_explicit_and_never_disguised_as_semantic(evaluation_fixture, tmp_path):
    root, store, fixture, _ = evaluation_fixture
    value = run_evaluation(store.path, root, fixture, tmp_path / "missing", frozen=True)
    assert value["summary"]["bm25"]["development"]["mrr"] == 1
    for mode in ("vector", "hybrid"):
        assert value["summary"][mode]["development"]["scorable"] == 0
        assert value["summary"][mode]["development"]["skipped"] == 1
        case = next(c for c in value["cases"] if c["mode"] == mode)
        assert case["status"] == "missing_offline_query_vector" and case["metrics"] is None


def test_invalid_gold_hash_is_not_scored_as_retrieval_failure(evaluation_fixture, tmp_path):
    root, store, fixture, _ = evaluation_fixture
    (root / "architecture.md").write_text("changed source", encoding="utf-8")
    value = run_evaluation(store.path, root, fixture, tmp_path / "stale", modes=["bm25"], frozen=True)
    assert value["cases"][0]["status"] == "invalid_gold_evidence"
    assert value["summary"]["bm25"]["development"]["scorable"] == 0


def test_missing_indexed_gold_remains_in_recall_denominator(evaluation_fixture, tmp_path):
    root, store, fixture, _ = evaluation_fixture
    store.prune_files(["database.md"])
    value = run_evaluation(store.path, root, fixture, tmp_path / "unindexed", modes=["bm25"], frozen=True)
    metrics = value["summary"]["bm25"]["development"]
    assert metrics["scorable"] == 1 and metrics["mrr"] == 0 and metrics["recall@10"] == 0
    assert metrics["indexed_gold_fraction"] == 0


def test_log_dedup_preserves_first_rank_and_no_answer_is_separate():
    rows = [{"path": "a", "rank": 1}, {"path": "b", "rank": 2}, {"path": "a", "rank": 3}]
    assert deduplicated(rows, "path") == rows[:2]
    assert score_results([], [], [1, 5])["no_answer_correct"] == 1
    assert score_results(rows, [], [1, 5])["no_answer_correct"] == 0


def test_frozen_parameter_sweeps_share_same_logical_snapshot(evaluation_fixture, tmp_path):
    root, store, fixture, _ = evaluation_fixture
    first = run_evaluation(store.path, root, fixture, tmp_path / "one", modes=["bm25"], frozen=True)
    second = run_evaluation(store.path, root, fixture, tmp_path / "two", modes=["bm25"], frozen=True,
                            search_parameters={"candidates": 30, "vector_weight": .4})
    assert first["snapshot"]["id"] == second["snapshot"]["id"]


def test_execution_failure_is_not_scored_or_labeled_success(evaluation_fixture, tmp_path, monkeypatch):
    root, store, fixture, _ = evaluation_fixture

    def unavailable(*args, **kwargs):
        raise ValueError("PRIVATE_REMOTE_ERROR_MUST_NOT_BE_LOGGED")

    monkeypatch.setattr(evaluator.Search, "run", unavailable)
    value = run_evaluation(store.path, root, fixture, tmp_path / "failed", modes=["bm25"], frozen=True)
    assert value["cases"][0]["status"] == "execution_error"
    assert value["cases"][0]["metrics"] is None
    assert value["summary"]["bm25"]["development"]["scorable"] == 0
    assert "PRIVATE_REMOTE_ERROR" not in (tmp_path / "failed/stages.jsonl").read_text()


@pytest.mark.parametrize("kind", ["file_link", "directory_link", "parent_escape", "environment"])
def test_source_evidence_rejects_symlinks_escapes_and_excluded_paths(evaluation_fixture, tmp_path, kind):
    root, store, fixture, _ = evaluation_fixture
    if kind == "file_link":
        (root / "linked.md").symlink_to(root / "architecture.md")
        relative = "linked.md"
    elif kind == "directory_link":
        (root / "alias").symlink_to(root, target_is_directory=True)
        relative = "alias/architecture.md"
    elif kind == "parent_escape":
        (tmp_path / "escaped.md").write_bytes((root / "architecture.md").read_bytes())
        relative = "../escaped.md"
    else:
        (root / ".env").write_bytes((root / "architecture.md").read_bytes())
        relative = ".env"
    fixture["queries"][0]["gold"][0]["path"] = relative
    value = run_evaluation(store.path, root, fixture, tmp_path / "unsafe", modes=["bm25"], frozen=True)
    assert value["cases"][0]["status"] == "invalid_gold_evidence"


def test_snapshot_identity_detects_in_place_content_changes_with_same_chunk_ids(evaluation_fixture, tmp_path):
    root, store, fixture, _ = evaluation_fixture
    first = run_evaluation(store.path, root, fixture, tmp_path / "first", modes=["bm25"], frozen=True)
    with store.connect() as db:
        db.execute("UPDATE chunks SET embedding_text=embedding_text || ' changed' WHERE id=(SELECT id FROM chunks LIMIT 1)")
    second = run_evaluation(store.path, root, fixture, tmp_path / "second", modes=["bm25"], frozen=True)
    assert first["snapshot"]["chunk_set_sha256"] == second["snapshot"]["chunk_set_sha256"]
    assert first["snapshot"]["id"] != second["snapshot"]["id"]


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_partial_semantic_scores_are_diagnostic_not_gate_eligible(evaluation_fixture, tmp_path, mode):
    root, store, fixture, vectors = evaluation_fixture
    with store.connect() as db:
        db.execute("DELETE FROM vectors WHERE chunk_id IN (SELECT c.id FROM chunks c JOIN files f ON f.id=c.file_id WHERE f.path='database.md')")
    value = run_evaluation(store.path, root, fixture, tmp_path / mode, modes=[mode], query_vectors=vectors, frozen=True)
    assert value["cases"][0]["status"] == ("degraded" if mode == "hybrid" else "incomplete_semantic_coverage")
    assert value["cases"][0]["metrics"]["mrr"] == 1
    summary = value["summary"][mode]["development"]
    assert summary["scorable"] == 0 and summary["mrr"] is None
    assert summary["incomplete_semantic_cases"] == (0 if mode == "hybrid" else 1)
    assert summary["degraded_cases"] == (1 if mode == "hybrid" else 0)
    assert summary["partial_semantic_diagnostics"]["mrr"] == (None if mode == "hybrid" else 1)


@pytest.mark.parametrize("weight", [0., 1.])
def test_endpoint_hybrid_weights_are_single_route_diagnostics(evaluation_fixture, tmp_path, weight):
    root, store, fixture, vectors = evaluation_fixture
    value = run_evaluation(store.path, root, fixture, tmp_path / "endpoint", modes=["hybrid"], query_vectors=vectors,
                            search_parameters={"vector_weight": weight}, frozen=True)
    assert value["cases"][0]["status"] == "degraded"
    assert value["summary"]["hybrid"]["development"]["scorable"] == 0
    assert value["summary"]["hybrid"]["development"]["degraded_cases"] == 1


def test_caller_can_inject_archived_search_class_for_fixed_snapshot_comparison(evaluation_fixture, tmp_path):
    root, store, fixture, _ = evaluation_fixture

    class ArchivedSearch(evaluator.Search):
        pass

    value = run_evaluation(store.path, root, fixture, tmp_path / "archived", modes=["bm25"], frozen=True,
                            search_class=ArchivedSearch)
    assert value["summary"]["bm25"]["development"]["mrr"] == 1
    assert value["algorithm_sha256"] == digest(__import__("pathlib").Path(__file__).read_bytes())


@pytest.mark.parametrize("binding", [None, {"architecture": "incorrect-query-hash"}])
def test_offline_query_vectors_require_current_text_hash_binding(evaluation_fixture, tmp_path, binding):
    root, store, fixture, vectors = evaluation_fixture
    if binding is None:
        vectors.pop("query_sha256")
    else:
        vectors["query_sha256"] = binding
    with pytest.raises(ValueError):
        run_evaluation(store.path, root, fixture, tmp_path / "mismatched", query_vectors=vectors, frozen=True)


def test_development_only_trials_do_not_execute_or_log_holdout_cases(evaluation_fixture, tmp_path):
    root, store, fixture, _ = evaluation_fixture
    fixture["queries"].append({"id": "untouched_holdout", "query": "hidden holdout query", "split": "holdout", "gold": []})
    value = run_evaluation(store.path, root, fixture, tmp_path / "devonly", modes=["bm25"], frozen=True, splits=("development",))
    assert value["summary"]["bm25"]["development"]["cases"] == 1
    assert value["summary"]["bm25"]["holdout"]["cases"] == 0
    assert all(case["split"] == "development" for case in value["cases"])
    assert "hidden holdout query" not in (tmp_path / "devonly/stages.jsonl").read_text()


@pytest.mark.parametrize("field", ["id", "query", "scope", "path", "annotation"])
def test_fixture_preflight_rejects_sensitive_metadata_before_output_or_backup(evaluation_fixture, tmp_path,
                                                                           monkeypatch, field):
    root, store, fixture, _ = evaluation_fixture
    credential = "Bearer " + "S" * 28  # Deliberately synthetic fixture data.
    case = fixture["queries"][0]
    if field in {"id", "query"}:
        case[field] = credential
    elif field == "scope":
        fixture["scope"] = credential
    elif field == "path":
        case["gold"][0]["path"] = credential
    else:
        fixture["annotations"] = [{"note": credential}]
    backups = []
    monkeypatch.setattr(evaluator, "freeze_database", lambda *args: backups.append(args))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="敏感信息") as caught:
        run_evaluation(store.path, root, fixture, output, modes=("bm25",))
    assert credential not in str(caught.value)
    assert not output.exists() and backups == []


def test_fixture_validation_preserves_valid_annotations_queries_and_gold():
    from copy import deepcopy
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures/code_search_eval.json").read_text())
    original = deepcopy(fixture)
    before = digest(json.dumps(fixture, ensure_ascii=False, sort_keys=True))
    assert evaluator.validate_fixture(fixture) is fixture
    assert fixture == original
    assert digest(json.dumps(fixture, ensure_ascii=False, sort_keys=True)) == before
    # Optional fields keep their existing interpretation without being added.
    minimal = {"queries": [{"id": "中文标识:一", "query": "普通问题"}]}
    assert evaluator.validate_fixture(minimal) is minimal
    assert set(minimal["queries"][0]) == {"id", "query"}
