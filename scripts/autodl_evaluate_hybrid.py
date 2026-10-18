#!/usr/bin/env python3
"""Exploratory CPU-only RRF diagnostic from existing 768-dimensional caches.

No API access, embedding generation, credentials, BM25, or test-set tuning.
RRF is fixed at k=60, with dense top100 and positive-score char-TFIDF top100.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

# Set limits before importing numerical libraries; enforce again at runtime.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"
sys.dont_write_bytecode = True

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_info, threadpool_limits

from autodl_analyze_quality_deltas import (
    bootstrap_means, file_hash, positive_map, read_json, read_jsonl, source_components,
)

MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
SEED, REPLICATES, RRF_K, WINDOW = 20261018, 1000, 60, 100
DATASETS = ("scifact", "belebele-zho_Hans", "belebele-eng_Latn", "video-text-zh-subset")
METHODS = ("dense768", "char_tfidf", "rrf")
METRICS = ("nDCG@10", "MRR@10", "strictRecall@10", "Hit@1", "Hit@10")
RRF_SOURCE = "https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion"


def input_fingerprint(dataset, role):
    digest = hashlib.sha256((dataset / (role + ".jsonl")).read_bytes())
    digest.update((MODEL + REVISION + role + "deployed-prefix-v2").encode())
    return digest.hexdigest()


def top_indices(scores, positive_only=False):
    if not np.isfinite(scores).all():
        raise ValueError("non-finite retrieval scores")
    order = np.argsort(-scores, kind="stable")
    if positive_only:
        order = order[scores[order] > 0]
    return order[:WINDOW].tolist()


def fuse(dense, lexical):
    scores = {}
    for ranked in (dense, lexical):
        for rank, index in enumerate(ranked, 1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (RRF_K + rank)
    # Corpus-file order breaks all component/fusion score ties reproducibly.
    return sorted(scores, key=lambda index: (-scores[index], index)), scores


def measure(ranked, corpus_ids, gold):
    positions = [(rank, corpus_ids[index]) for rank, index in enumerate(ranked, 1)
                 if corpus_ids[index] in gold]
    first = positions[0][0] if positions else None
    dcg = sum((2 ** gold[cid] - 1) / math.log2(rank + 1)
              for rank, cid in positions if rank <= 10)
    ideal = sum((2 ** score - 1) / math.log2(rank + 1)
                for rank, score in enumerate(sorted(gold.values(), reverse=True)[:10], 1))
    hits = sum(rank <= 10 for rank, _ in positions)
    return {"nDCG@10": dcg / ideal,
            "MRR@10": 1 / first if first is not None and first <= 10 else 0.0,
            "strictRecall@10": hits / len(gold),
            "Hit@1": float(first == 1),
            "Hit@10": float(first is not None and first <= 10),
            "first_positive_rank_in_returned_window": first,
            "returned_candidates": len(ranked),
            "top10_ids": [corpus_ids[index] for index in ranked[:10]]}


def check_edges():
    ranked, scores = fuse([0, 1], [1, 2])
    assert ranked == [1, 0, 2] and math.isclose(scores[1], 1 / 62 + 1 / 61)
    assert top_indices(np.zeros(5), True) == []
    only_dense, _ = fuse([3, 1], [])
    assert only_dense == [3, 1]
    values = measure([0, 1], [str(i) for i in range(12)], {str(i): 1 for i in range(12)})
    assert math.isclose(values["strictRecall@10"], 2 / 12)


def input_paths(root, name):
    dataset, quality = root / "datasets" / name, root / "quality" / name
    files = {"corpus": dataset / "corpus.jsonl", "queries": dataset / "queries.jsonl",
             "qrels": dataset / "qrels.jsonl", "manifest": dataset / "manifest.json",
             "saved_dense_metrics": quality / "per-query-768.json",
             "saved_lexical_metrics": quality / "lexical-per-query.json",
             "summary": quality / "summary.json"}
    for role in ("queries", "corpus"):
        files[role + "_vectors"] = quality / (role + ".npy")
        files[role + "_state"] = quality / (role + "-state.json")
    return files


def reusable_dataset(root, name, previous, current):
    for key in ("schema_version", "dimension", "model", "model_revision", "parameters",
                "rrf_formula", "metrics", "bootstrap_seed", "bootstrap_replicates", "ci_method"):
        if previous.get(key) != current.get(key):
            return None
    old = next((item for item in previous.get("datasets", []) if item.get("dataset") == name), None)
    if old is None or old.get("status") != "complete":
        return None
    paths = input_paths(root, name)
    if set(paths) != set(old.get("input_files", {})):
        return None
    for key, path in paths.items():
        if not path.is_file() or file_hash(path) != old["input_files"][key]["sha256"]:
            return None
    old["cache_reused"] = True
    old["cache_reuse_validation"] = "Fixed parameters, model/revision, bootstrap protocol and every current input file SHA256 match the completed result."
    return old


def evaluate(root, name):
    started = time.monotonic()
    dataset = root / "datasets" / name
    files = input_paths(root, name)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing completed inputs: " + ", ".join(missing))
    corpus, queries, qrels = (read_jsonl(files[role]) for role in ("corpus", "queries", "qrels"))
    corpus_ids, query_ids = ([str(item["id"]) for item in rows] for rows in (corpus, queries))
    if not corpus or not queries or len(set(corpus_ids)) != len(corpus_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("empty input or duplicate IDs")
    if any("image_path" in item or "frame_paths" in item for item in corpus):
        raise ValueError("hybrid text diagnostic does not accept media candidates")
    relevant = positive_map(qrels)
    if set(relevant) != set(query_ids) or any(not set(relevant[qid]) <= set(corpus_ids) for qid in query_ids):
        raise ValueError("invalid query/corpus relevance endpoints")
    summary = read_json(files["summary"])
    if summary["model"] != MODEL or summary["model_revision"] != REVISION:
        raise ValueError("model identity mismatch")
    arrays = {}
    for role, rows in (("queries", queries), ("corpus", corpus)):
        state = read_json(files[role + "_state"])
        if (state["fingerprint"] != input_fingerprint(dataset, role)
                or state["completed"] != state["total"] or state["completed"] != len(rows)
                or state["api_boot_id"] != summary["boot_id"]):
            raise ValueError("incomplete or misaligned cache: " + role)
        values = np.asarray(np.load(files[role + "_vectors"], allow_pickle=False), dtype=np.float32)
        if values.shape != (len(rows), 768) or not np.isfinite(values).all():
            raise ValueError("invalid cached vector shape/values")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or not np.allclose(norms, 1, atol=.001):
            raise ValueError("cached vectors are not unit vectors")
        arrays[role] = values / norms
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=100_000,
                            sublinear_tf=True, dtype=np.float32)
    documents = [item.get("title", "") + " " + item.get("text", "")[:8192] for item in corpus]
    document_matrix = tfidf.fit_transform(documents)
    query_matrix = tfidf.transform([item["text"][:8192] for item in queries])
    per_query, matrices, union_coverage = [], {method: [] for method in METHODS}, []
    saved_dense = {row["query_id"]: row for row in read_json(files["saved_dense_metrics"])["per_query"]}
    saved_lexical = {row["query_id"]: row for row in read_json(files["saved_lexical_metrics"])["per_query"]}
    if set(saved_dense) != set(query_ids) or set(saved_lexical) != set(query_ids):
        raise ValueError("saved metric IDs do not match the caches")
    dense_differences, lexical_gate_differences = [], []
    for beginning in range(0, len(queries), 32):
        end = min(beginning + 32, len(queries))
        # Some macOS BLAS builds emit spurious float warnings; validate actual scores.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            dense_scores = arrays["queries"][beginning:end] @ arrays["corpus"].T
        lexical_scores = (query_matrix[beginning:end] @ document_matrix.T).toarray()
        if not np.isfinite(dense_scores).all() or not np.isfinite(lexical_scores).all():
            raise ValueError("non-finite computed score block")
        for offset, qid in enumerate(query_ids[beginning:end]):
            dense = top_indices(dense_scores[offset])
            lexical = top_indices(lexical_scores[offset], True)
            rrf, rrf_scores = fuse(dense, lexical)
            rankings = {"dense768": dense, "char_tfidf": lexical, "rrf": rrf}
            record = {"query_id": qid, "positive_count": len(relevant[qid]),
                      "gold_ids": sorted(relevant[qid]), "methods": {}}
            for method, order in rankings.items():
                row = measure(order, corpus_ids, relevant[qid])
                record["methods"][method] = row
                matrices[method].append([row[metric] for metric in METRICS])
            dense_row = record["methods"]["dense768"]
            lexical_row = record["methods"]["char_tfidf"]
            dense_differences.append(max(abs(dense_row[metric] - saved_dense[qid][
                "Recall@10" if metric == "strictRecall@10" else metric]) for metric in METRICS))
            lexical_gate_differences.append(max(abs(lexical_row[metric] - saved_lexical[qid][
                "Recall@10" if metric == "strictRecall@10" else metric]) for metric in METRICS))
            covered = sum(corpus_ids[index] in relevant[qid] for index in rrf) / len(relevant[qid])
            union_coverage.append(covered)
            record["candidate_union_strict_positive_coverage"] = covered
            dense_ranks = {index: rank for rank, index in enumerate(dense, 1)}
            lexical_ranks = {index: rank for rank, index in enumerate(lexical, 1)}
            record["rrf_top10_contributions"] = [{"corpus_id": corpus_ids[index],
                "rrf_rank": rank, "rrf_score": rrf_scores[index],
                "dense_rank": dense_ranks.get(index), "char_tfidf_rank": lexical_ranks.get(index)}
                for rank, index in enumerate(rrf[:10], 1)]
            per_query.append(record)
    # Top-10 dense metrics must reproduce the fixed original cache evaluation.
    if max(dense_differences) > 1e-7:
        raise ValueError("recomputed dense ranking differs from saved per-query evaluation")
    matrices = {key: np.asarray(value, dtype=np.float64) for key, value in matrices.items()}
    comparisons = {"rrf_minus_dense768": matrices["rrf"] - matrices["dense768"],
                   "rrf_minus_char_tfidf": matrices["rrf"] - matrices["char_tfidf"]}
    combined_keys = list(METHODS) + list(comparisons)
    combined = np.concatenate([matrices[key] if key in matrices else comparisons[key]
                               for key in combined_keys], axis=1)
    groups = source_components(query_ids, qrels, corpus_ids)
    query_ci = bootstrap_means(combined, seed=SEED, replicates=REPLICATES)
    cluster_ci = (query_ci.copy() if len(groups) == len(query_ids) else
                  bootstrap_means(combined, groups, SEED, REPLICATES) if len(groups) > 1 else None)
    result = {"dataset": name, "status": "complete", "dimension": 768,
        "counts": {"queries": len(queries), "corpus": len(corpus), "qrels": len(qrels),
                   "positive_source_clusters": len(groups), "largest_cluster_queries": max(map(len, groups))},
        "input_files": {key: {"path": str(path), "sha256": file_hash(path)} for key, path in files.items()},
        "methods": {}, "paired_deltas": {}, "per_query": per_query,
        "candidate_union_mean_strict_positive_coverage": float(np.mean(union_coverage)),
        "validation": {"complete_aligned_768d_cache": True, "qrels_integrity": True,
            "dense_matches_saved_per_query_metrics": True,
            "maximum_dense_metric_difference": max(dense_differences),
            "char_queries_differing_from_original_zero_tie_baseline": sum(v > 1e-7 for v in lexical_gate_differences),
            "char_difference_note": "This diagnostic excludes zero-score lexical candidates; the original char sanity baseline ranked even zero-score ties. Parameters otherwise match exactly."},
        "elapsed_cpu_wall_s": round(time.monotonic() - started, 3)}
    for key_index, key in enumerate(combined_keys):
        column = key_index * len(METRICS)
        if key in matrices:
            result["methods"][key] = {"metrics": {metric: float(matrices[key][:, i].mean()) for i, metric in enumerate(METRICS)},
                "query_bootstrap_ci95": {metric: query_ci[:, column + i].tolist() for i, metric in enumerate(METRICS)},
                "positive_source_cluster_ci95": {metric: cluster_ci[:, column + i].tolist() if cluster_ci is not None else None
                                                  for i, metric in enumerate(METRICS)}}
        else:
            reference = "dense768" if key == "rrf_minus_dense768" else "char_tfidf"
            result["paired_deltas"][key] = {metric: {
                "reference_mean": float(matrices[reference][:, i].mean()),
                "fusion_mean": float(matrices["rrf"][:, i].mean()),
                "delta_pp": float(comparisons[key][:, i].mean() * 100),
                "paired_query_bootstrap_ci95_pp": (query_ci[:, column + i] * 100).tolist(),
                "paired_positive_source_cluster_ci95_pp": (cluster_ci[:, column + i] * 100).tolist() if cluster_ci is not None else None}
                for i, metric in enumerate(METRICS)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/autodl-evaluation-20261018"))
    parser.add_argument("--allow-incomplete", action="store_true", help="save partial results and permit missing caches")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS),
                        help="score only these tasks; retain other completed tasks after full cache validation")
    parser.add_argument("--reuse-completed", action="store_true",
                        help="also reuse selected completed tasks when parameters and all input hashes match")
    args = parser.parse_args()
    root = args.root.resolve()
    check_edges()
    result = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dimension": 768, "model": MODEL, "model_revision": REVISION,
        "parameters": {"rrf_k": RRF_K, "rank_window_size": WINDOW,
            "rank_starts_at": 1, "equal_weights": True, "char_positive_scores_only": True,
            "char_tfidf": {"analyzer": "char", "ngram_range": [2, 4], "max_features": 100000,
                "sublinear_tf": True, "dtype": "float32", "norm": "l2", "fit_on": "corpus_only"},
            "tie_break": "original corpus JSONL row order", "max_cpu_threads": 4},
        "rrf_formula": "score(d) = sum(1 / (60 + one_based_rank(d))) over retrievers returning d",
        "formula_source": RRF_SOURCE, "metrics": list(METRICS),
        "strict_recall_definition": "macro mean of retrieved qrels-positive documents / ALL qrels-positive documents per query; denominator is not capped at 10 or candidate union size",
        "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
        "ci_method": "paired percentile query bootstrap and positive-source connected-component cluster bootstrap; differences formed before shared resampling",
        "ci_scope": "Pointwise conditional on these fixed corpora. No multiple-testing correction, training-contamination control or independent confirmation.",
        "interpretation": "Exploratory diagnostic motivated by errors already inspected on these public test tasks; k and windows fixed before this run, with no test-set parameter search. Char TF-IDF is not BM25; no improvement is guaranteed.",
        "requested_datasets": args.datasets, "reused_datasets": [], "datasets": [], "errors": []}
    output = root / "hybrid-results.json"
    previous = read_json(output) if output.is_file() else {}
    invalid = False
    with threadpool_limits(limits=4):
        result["cpu_threadpools"] = threadpool_info()
        for name in DATASETS:
            try:
                if name not in args.datasets or args.reuse_completed:
                    cached = reusable_dataset(root, name, previous, result)
                    if cached is not None:
                        result["datasets"].append(cached)
                        result["reused_datasets"].append(name)
                        print(json.dumps({"dataset": name, "status": "complete", "cache_reused": True}), flush=True)
                        continue
                if name not in args.datasets:
                    raise FileNotFoundError("task not selected and no matching completed cache: " + name)
                dataset = evaluate(root, name)
                result["datasets"].append(dataset)
                print(json.dumps({"dataset": name, "status": "complete", "nDCG@10": {
                    method: dataset["methods"][method]["metrics"]["nDCG@10"] for method in METHODS}}, ensure_ascii=False), flush=True)
            except (OSError, ValueError, KeyError, TypeError) as error:
                missing = isinstance(error, FileNotFoundError)
                record = {"dataset": name, "status": "incomplete" if missing else "error", "error": str(error)}
                result["datasets"].append(record)
                result["errors"].append(record)
                invalid |= not missing
                print(json.dumps(record, ensure_ascii=False), flush=True)
    result["status"] = "complete" if not result["errors"] else "incomplete"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "status": result["status"], "errors": len(result["errors"])}, ensure_ascii=False))
    return 2 if invalid or (result["errors"] and not args.allow_incomplete) else 0


if __name__ == "__main__":
    raise SystemExit(main())
