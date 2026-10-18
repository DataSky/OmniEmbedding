#!/usr/bin/env python3
"""Bounded retrieval research: freeze data, evaluate, select on development,
then check holdout once. Every stage is persisted locally; no source rewriting.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omniembed.code_search.store import DIMENSIONS, MODEL, REVISION
from scripts.evaluate_code_search import validate_fixture


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def index_state(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30) as db:
        db.execute("BEGIN")
        chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        vectors = db.execute("SELECT count(*) FROM vectors WHERE model=? AND revision=? AND dimensions=?",
                             (MODEL, REVISION, DIMENSIONS)).fetchone()[0]
        job = db.execute("SELECT id,status,phase FROM jobs ORDER BY rowid DESC LIMIT 1").fetchone()
    return dict(chunks=chunks, vectors=vectors, job=dict(zip(("id", "status", "phase"), job)) if job else None)


def select_trial(trials, mode="hybrid"):
    """Holdout metrics are deliberately excluded from selection."""
    def key(trial):
        m = trial["summary"].get(mode, {}).get("development", {})
        return (float(m.get("mrr") or 0), float(m.get("recall@10") or 0),
                -trial["parameters"]["candidates"])
    return max(trials, key=key)


def acceptance_gate(baseline, selected, mode="hybrid", *, min_holdout_mrr=.5, min_holdout_recall5=.7):
    reasons = []
    for split in ("development", "holdout"):
        before = baseline["summary"].get(mode, {}).get(split, {})
        after = selected["summary"].get(mode, {}).get(split, {})
        if not after.get("cases") or after.get("mrr") is None:
            reasons.append(split + ":no_scorable_cases")
        if after.get("scorable", after.get("cases")) != after.get("cases"):
            reasons.append(split + ":unscored_cases")
        if any(after.get(key, 0) for key in ("degraded_cases", "incomplete_semantic_cases", "execution_error_cases")):
            reasons.append(split + ":unreliable_route_execution")
        for key in ("indexed_gold_fraction", *(["vector_gold_fraction"] if mode != "bm25" else [])):
            if (after.get(key) or 0) < 1:
                reasons.append(split + ":incomplete_" + key)
        for key in ("mrr", "recall@5", "recall@10"):
            if before.get(key) is not None and (after.get(key) or 0) + 1e-9 < before[key]:
                reasons.append(split + ":regression_" + key)
        if before.get("no_answer_accuracy") is not None and after.get("no_answer_accuracy") is not None:
            if after["no_answer_accuracy"] < before["no_answer_accuracy"]:
                reasons.append(split + ":no_answer_regression")
        if mode == "hybrid" and after.get("effective_mode_counts", {}).get("bm25", 0):
            reasons.append(split + ":semantic_route_degraded")
    invalid = [c for c in selected.get("cases", []) if c.get("mode") == mode and c.get("status") != "ok"]
    if invalid:
        reasons.append("unscored_or_invalid_cases")
    holdout = selected["summary"].get(mode, {}).get("holdout", {})
    if (holdout.get("mrr") or 0) < min_holdout_mrr:
        reasons.append("holdout:below_minimum_mrr")
    if (holdout.get("recall@5") or 0) < min_holdout_recall5:
        reasons.append("holdout:below_minimum_recall5")
    if holdout.get("no_answer_accuracy") is not None and holdout["no_answer_accuracy"] < 1:
        reasons.append("holdout:no_answer_failures")
    return {"accepted":not reasons, "reasons":sorted(set(reasons)), "mode":mode,
            "selection_split":"development", "acceptance_split":"holdout",
            "thresholds":{"holdout_mrr":min_holdout_mrr, "holdout_recall@5":min_holdout_recall5, "no_answer_accuracy":1}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT/"data/code-search/code-search.sqlite3")
    parser.add_argument("--root", type=Path, default=Path.home()/"AiCode")
    parser.add_argument("--fixture", type=Path, default=ROOT/"tests/fixtures/code_search_eval.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/code-search/evaluations")
    parser.add_argument("--query-vectors", type=Path)
    parser.add_argument("--encode-queries", action="store_true", help="仅将评测查询发送至已配置AutoDL，不重编码文档")
    parser.add_argument("--endpoint-file", type=Path, default=ROOT/"data/autodl/endpoint.json")
    parser.add_argument("--key-file", type=Path, default=ROOT/"data/autodl/inference.key")
    parser.add_argument("--wait-for-index", action="store_true")
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--min-holdout-mrr", type=float, default=.5)
    parser.add_argument("--min-holdout-recall5", type=float, default=.7)
    args = parser.parse_args(argv)
    if not 1 <= args.max_trials <= 10 or not 1 <= args.max_wait_seconds <= 86400:
        parser.error("max-trials需为1—10；max-wait-seconds需为1—86400")
    if not 0 <= args.min_holdout_mrr <= 1 or not 0 <= args.min_holdout_recall5 <= 1:
        parser.error("质量门槛需在0—1之间")
    try:
        fixture_value = json.loads(args.fixture.read_text(encoding="utf-8"))
        validate_fixture(fixture_value)
    except Exception as exc:
        # Reject before making a run directory, copying the fixture, polling
        # the index or sending any query. Never echo an invalid input value.
        print(json.dumps(dict(stage="failed", status="failed", error_code=type(exc).__name__)), flush=True)
        return 1
    run_id = datetime.now(timezone.utc).strftime("research-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    output = args.output_dir.resolve()/run_id
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    events = output/"events.jsonl"

    def log(stage, **fields):
        event = dict(at=utc(), stage=stage, **fields)
        with events.open("a", encoding="utf-8") as handle:
            os.chmod(events, 0o600)
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        print(json.dumps(event, ensure_ascii=False), flush=True)

    try:
        frozen_fixture = output/"fixture.json"
        write_json(frozen_fixture, fixture_value)
        frozen_query_vectors = None
        if args.query_vectors:
            frozen_query_vectors = output/"query-vectors.json"
            write_json(frozen_query_vectors, json.loads(args.query_vectors.read_text(encoding="utf-8")))
        log("started", run_id=run_id, max_trials=args.max_trials,
            fixture_sha256=hashlib.sha256(frozen_fixture.read_bytes()).hexdigest(),
            boundary="固定语料、有限参数试验、开发集选参、留出集验收、不自动改源码或模型")
        deadline = time.monotonic() + args.max_wait_seconds
        previous = None
        while True:
            state = index_state(args.db)
            if state != previous:
                log("index_progress", **state)
                previous = state
            running = state["job"] and state["job"]["status"] in {"queued", "running", "cancelling"}
            if not args.wait_for_index or (not running and state["chunks"] > 0 and state["vectors"] == state["chunks"]):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("index_not_ready")
            time.sleep(10)
        snapshot = output/"snapshot.sqlite3"
        started = time.perf_counter()
        with sqlite3.connect(args.db.resolve().as_uri()+"?mode=ro", uri=True, timeout=30) as source, sqlite3.connect(snapshot) as target:
            source.backup(target)
        os.chmod(snapshot, 0o600)
        with sqlite3.connect(snapshot) as db:
            integrity = dict(foreign_key_errors=len(db.execute("PRAGMA foreign_key_check").fetchall()),
                             fts_rows=db.execute("SELECT count(*) FROM chunk_fts").fetchone()[0],
                             chunks=db.execute("SELECT count(*) FROM chunks").fetchone()[0])
            benchmark_row = db.execute("SELECT value FROM meta WHERE key='benchmark'").fetchone()
            benchmark_metadata = json.loads(benchmark_row[0]) if benchmark_row else {}
        log("snapshot_integrity", **integrity, elapsed_ms=round((time.perf_counter()-started)*1000, 2))
        if integrity["foreign_key_errors"] or integrity["fts_rows"] != integrity["chunks"]:
            raise ValueError("snapshot_integrity_failed")
        algorithm_path = output/"search.py"
        algorithm_path.write_bytes((ROOT/"omniembed/code_search/search.py").read_bytes())
        algorithm_path.chmod(0o600)
        module_name = "omniembed.code_search.frozen_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(module_name, algorithm_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        log("algorithm_frozen", algorithm_sha256=hashlib.sha256(algorithm_path.read_bytes()).hexdigest())
        vectors_path = frozen_query_vectors
        if args.encode_queries:
            from omniembed.remote_encoder import RemoteEncoder
            cases = fixture_value["queries"]
            texts = [case["query"] for case in cases]
            encoder = RemoteEncoder(model_id=MODEL, revision=REVISION, endpoint_file=args.endpoint_file,
                                    key_file=args.key_file, timeout=45)
            encoder._url = None
            started = time.perf_counter()
            encoded = encoder.query(texts, bs=16)
            vectors_path = output/"query-vectors.json"
            write_json(vectors_path, dict(model=MODEL, revision=REVISION, dimensions=DIMENSIONS,
                       queries={case["id"]:vector.tolist() for case, vector in zip(cases, encoded)},
                       query_sha256={case["id"]:hashlib.sha256(case["query"].encode()).hexdigest() for case in cases}))
            log("query_encoding", count=len(texts), characters=sum(map(len, texts)),
                hashes=[{"id":case["id"], "sha256":hashlib.sha256(case["query"].encode()).hexdigest()} for case in cases],
                boot_id=encoder.boot_id, elapsed_ms=round((time.perf_counter()-started)*1000,2), http_status=200)
        from evaluate_code_search import run_evaluation
        query_vectors = json.loads(vectors_path.read_text()) if vectors_path else None
        modes = ("bm25", "vector", "hybrid") if query_vectors else ("bm25",)
        configurations = [(100, .6), (100, .35), (100, .8), (200, .35), (200, .6), (200, .8)][:args.max_trials]
        trials = []
        for number, (candidates, weight) in enumerate(configurations):
            label = f"trial-{number:02d}"
            parameters = dict(candidates=candidates, vector_weight=weight)
            log("trial_started", label=label, parameters=parameters)
            trial = run_evaluation(db=snapshot, root=args.root, fixture=frozen_fixture,
                                   output_dir=output/label, label=label, modes=modes,
                                   query_vectors=query_vectors, search_parameters=parameters, frozen=True,
                                   splits=("development",), search_class=module.Search)
            trials.append(trial)
            log("trial_completed", label=label, snapshot_id=trial["snapshot"]["id"],
                development={mode:trial["summary"].get(mode, {}).get("development") for mode in modes})
            # BM25 has no parameter varied by this grid. Avoid identical runs.
            if not query_vectors:
                break
        ids = {trial["snapshot"]["id"] for trial in trials}
        if len(ids) != 1:
            raise ValueError("trial_snapshot_drift")
        fixture_ids = {trial["fixture_sha256"] for trial in trials}
        if len(fixture_ids) != 1:
            raise ValueError("trial_fixture_drift")
        if len({trial["algorithm_sha256"] for trial in trials}) != 1:
            raise ValueError("trial_algorithm_drift")
        mode = "hybrid" if query_vectors else "bm25"
        selected = select_trial(trials, mode)
        # Inspect holdout only after selection. No rejected candidate is tuned
        # against holdout, and the selected configuration is never changed here.
        acceptance = {}
        for trial in (trials[0], selected):
            if trial["label"] in acceptance:
                continue
            label = trial["label"] + "-holdout"
            checked = run_evaluation(db=snapshot, root=args.root, fixture=frozen_fixture,
                                     output_dir=output/label, label=label, modes=modes,
                                     query_vectors=query_vectors, search_parameters=trial["parameters"],
                                     frozen=True, splits=("holdout",), search_class=module.Search)
            if checked["snapshot"]["id"] not in ids or checked["fixture_sha256"] not in fixture_ids:
                raise ValueError("holdout_snapshot_or_fixture_drift")
            combined = deepcopy(trial)
            combined["cases"].extend(checked["cases"])
            for route in modes:
                combined["summary"][route]["holdout"] = checked["summary"][route]["holdout"]
            acceptance[trial["label"]] = combined
        baseline, selected = acceptance[trials[0]["label"]], acceptance[selected["label"]]
        gate = acceptance_gate(baseline, selected, mode, min_holdout_mrr=args.min_holdout_mrr,
                               min_holdout_recall5=args.min_holdout_recall5)
        log("holdout_gate", selected_label=selected["label"], **gate,
            holdout=selected["summary"].get(mode, {}).get("holdout"))
        recommendation = dict(run_id=run_id, created_at=utc(), status="accepted" if gate["accepted"] else "blocked",
                              snapshot_id=next(iter(ids)), selected=selected["label"], parameters=selected["parameters"],
                              snapshot=selected["snapshot"], scope=fixture_value.get("scope"),
                              benchmark_subset=bool(benchmark_metadata.get("benchmark_subset")), whole_ai_code_quality=False,
                              fixture_sha256=next(iter(fixture_ids)), gate=gate, baseline=baseline["summary"],
                              selected_metrics=selected["summary"], cases=selected["cases"],
                              next_steps=["保留失败用例并扩大独立标注集", "先检查缺失语料与编码覆盖，再比较算法", "代码/分块/模型变更另建版本试验，不能混用向量空间"])
        write_json(output/"recommendation.json", recommendation)
        write_json(args.output_dir.resolve()/"latest.json", {**recommendation, "path":str(output)})
        log("completed", status=recommendation["status"], output=str(output), trials=len(trials),
            automatic_source_changes=False, production_parameters_changed=False)
        return 0
    except Exception as exc:
        # Exceptions from remote endpoints or source parsers may echo user data.
        log("failed", status="failed", error_code=type(exc).__name__)
        write_json(output/"failure.json", dict(run_id=run_id, status="failed", created_at=utc(), error_code=type(exc).__name__))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
