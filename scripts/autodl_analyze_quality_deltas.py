#!/usr/bin/env python3
"""CPU-only paired dimension analysis; never opens an API or a credential file.

Example:
  python scripts/autodl_analyze_quality_deltas.py --root data/autodl-evaluation-20261018

Reads quality/*/per-query-{768,512,256,128}.json and dataset qrels. Differences
are signed absolute percentage points: shortened dimension minus 768. Every
bootstrap replicate resamples the same queries, or whole positive-source
components, for both dimensions. Intervals are pointwise, conditional on the
fixed candidate gallery, and do not establish equivalence by themselves.

If queries_1ka.jsonl/qrels_1ka.jsonl exist, also score the strict smaller 1K-A
video protocol from cached vectors. Dedicated queries_1ka.npy and
caption-documents-1ka.npy take precedence. Reuse of expanded-caption vectors is
allowed only for byte-exact text matches; case/punctuation/whitespace fallback
and diagonal relevance labels are deliberately absent.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

DIMENSIONS = (768, 512, 256, 128)
METRICS = ("nDCG@10", "MRR@10", "Hit@1", "Hit@10", "Recall@10")
SEED = 20261018
REPLICATES = 1000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def positive_map(qrels):
    result = defaultdict(dict)
    for record in qrels:
        score = float(record["score"])
        if not math.isfinite(score):
            raise ValueError("non-finite qrel")
        if score > 0:
            qid, cid = str(record["query_id"]), str(record["corpus_id"])
            result[qid][cid] = max(result[qid].get(cid, 0.0), score)
    return dict(result)


def source_components(query_ids, qrels, corpus_ids=None):
    """Connect all positive documents of a query; group dependent queries."""
    relevant = positive_map(qrels)
    gallery = set(map(str, corpus_ids)) if corpus_ids is not None else None
    parents = {}

    def root(cid):
        parents.setdefault(cid, cid)
        while parents[cid] != cid:
            parents[cid] = parents[parents[cid]]
            cid = parents[cid]
        return cid

    for qid in query_ids:
        gold = sorted(relevant.get(str(qid), {}))
        if not gold or (gallery is not None and not set(gold) <= gallery):
            raise ValueError("query lacks valid positive corpus items: " + str(qid))
        first = root(gold[0])
        for cid in gold[1:]:
            left, right = root(first), root(cid)
            parents[max(left, right)] = min(left, right)
            first = root(left)
    groups = defaultdict(list)
    for index, qid in enumerate(query_ids):
        groups[root(sorted(relevant[str(qid)])[0])].append(index)
    return [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]


def bootstrap_means(values, groups=None, seed=SEED, replicates=REPLICATES):
    """Bound scratch memory; cluster means retain query-macro weighting."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires a finite nonempty query-by-metric matrix")
    rng = np.random.default_rng(seed)
    if groups is None:
        totals = values
        sizes = np.ones(len(values), dtype=np.int64)
    else:
        if not groups:
            raise ValueError("bootstrap has no source components")
        sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
        totals = np.asarray([values[group].sum(axis=0) for group in groups], dtype=np.float64)
    units = len(totals)
    # Advanced indexing makes a copy; keep it around 16 MiB per iteration.
    batch_size = max(1, min(25, (16 * 1024**2) // max(1, units * values.shape[1] * 8)))
    boot = np.empty((replicates, values.shape[1]), dtype=np.float64)
    for beginning in range(0, replicates, batch_size):
        count = min(batch_size, replicates - beginning)
        sampled = rng.integers(0, units, size=(count, units))
        boot[beginning:beginning + count] = (
            totals[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)[:, None])
    return np.quantile(boot, [.025, .975], axis=0)


def paired_dimension_deltas(by_dimension, query_ids, qrels, corpus_ids,
                            seed=SEED, replicates=REPLICATES):
    reference = np.asarray(by_dimension[768], dtype=np.float64)
    if reference.shape != (len(query_ids), len(METRICS)):
        raise ValueError("unexpected reference metric matrix shape")
    matrices = []
    for dimension in DIMENSIONS:
        current = np.asarray(by_dimension[dimension], dtype=np.float64)
        if current.shape != reference.shape or not np.isfinite(current).all():
            raise ValueError("dimension query metric matrices do not align")
        matrices.append(current)
    # Paired differences are formed BEFORE resampling, including shared-source
    # aggregation. This is not a difference between independently bootstrapped CIs.
    differences = np.stack([matrix - reference for matrix in matrices], axis=1)
    flat = differences.reshape(len(query_ids), -1)
    groups = source_components(query_ids, qrels, corpus_ids)
    query_ci = bootstrap_means(flat, seed=seed, replicates=replicates)
    if len(groups) == len(query_ids):
        cluster_ci = query_ci.copy()
    else:
        cluster_ci = bootstrap_means(flat, groups, seed, replicates) if len(groups) > 1 else None
    deltas = {}
    for dim_index, dimension in enumerate(DIMENSIONS):
        metrics = {}
        for metric_index, metric in enumerate(METRICS):
            column = dim_index * len(METRICS) + metric_index
            metrics[metric] = {
                "reference_mean": float(reference[:, metric_index].mean()),
                "dimension_mean": float(matrices[dim_index][:, metric_index].mean()),
                "delta_pp": float(flat[:, column].mean() * 100),
                "paired_query_bootstrap_ci95_pp": [float(value * 100) for value in query_ci[:, column]],
                "paired_positive_source_cluster_ci95_pp": (
                    [float(value * 100) for value in cluster_ci[:, column]]
                    if cluster_ci is not None else None),
            }
        deltas[str(dimension)] = metrics
    return {"query_count": len(query_ids), "positive_source_clusters": len(groups),
            "largest_source_cluster_queries": max(map(len, groups)),
            "cluster_ci_estimable": len(groups) > 1,
            "cluster_equals_query_units": len(groups) == len(query_ids),
            "reference_dimension": 768, "dimensions": deltas}


def load_metric_rows(path, expected_ids):
    value = read_json(path)
    rows = value.get("per_query")
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing per-query metrics: " + str(path))
    lookup = {}
    for row in rows:
        qid = str(row["query_id"])
        if qid in lookup:
            raise ValueError("duplicate per-query ID: " + qid)
        values = np.asarray([row[metric] for metric in METRICS], dtype=np.float64)
        if not np.isfinite(values).all() or (values < -1e-12).any() or (values > 1 + 1e-12).any():
            raise ValueError("invalid per-query metric value")
        lookup[qid] = values
    if set(lookup) != set(expected_ids):
        raise ValueError("per-query IDs differ from dataset query IDs: " + str(path))
    matrix = np.asarray([lookup[qid] for qid in expected_ids])
    for index, metric in enumerate(METRICS):
        if metric in value.get("metrics", {}) and not math.isclose(
                float(matrix[:, index].mean()), float(value["metrics"][metric]), abs_tol=1e-9):
            raise ValueError("aggregate metric disagrees with per-query rows: " + metric)
    return matrix


def dataset_deltas(dataset, quality, include_reverse=True):
    queries = read_jsonl(dataset / "queries.jsonl")
    corpus = read_jsonl(dataset / "corpus.jsonl")
    qrels = read_jsonl(dataset / "qrels.jsonl")
    if not queries or not corpus or not qrels:
        raise ValueError("dataset corpus, queries and qrels must all be nonempty")
    query_ids = sorted(str(row["id"]) for row in queries)
    corpus_ids = sorted(str(row["id"]) for row in corpus)
    if len(set(query_ids)) != len(queries) or len(set(corpus_ids)) != len(corpus):
        raise ValueError("duplicate dataset IDs")
    sources = {name: file_hash(dataset / name) for name in
               ("corpus.jsonl", "queries.jsonl", "qrels.jsonl")}
    directions = [("forward", "", query_ids, corpus_ids, qrels)]
    media = "image_path" in corpus[0] or "frame_paths" in corpus[0]
    if include_reverse and media:
        reverse_qrels = [{"query_id": row["corpus_id"], "corpus_id": row["query_id"], "score": row["score"]}
                         for row in qrels]
        directions.append(("media_to_text", "reverse-", corpus_ids, query_ids, reverse_qrels))
    result = {"dataset": dataset.name, "status": "complete", "dataset_file_sha256": sources,
              "directions": {}}
    for direction, prefix, ids, gallery_ids, labels in directions:
        paths = {dimension: quality / f"{prefix}per-query-{dimension}.json" for dimension in DIMENSIONS}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError("missing completed metric file: " + str(path))
        matrices = {dimension: load_metric_rows(path, ids) for dimension, path in paths.items()}
        result["directions"][direction] = {
            **paired_dimension_deltas(matrices, ids, labels, gallery_ids),
            "per_query_file_sha256": {str(dimension): file_hash(path) for dimension, path in paths.items()}}
    return result


def merge_exact_caption_qrels(queries, qrels):
    """Deduplicate identical text, union its labels, retain first vector row."""
    by_text, by_id, canonical, indices = {}, {}, [], []
    for index, query in enumerate(queries):
        qid, text = str(query["id"]), query["text"]
        if not isinstance(text, str) or not text:
            raise ValueError("empty official caption")
        if qid in by_id:
            raise ValueError("duplicate source query ID")
        if text not in by_text:
            by_text[text] = qid
            canonical.append({"id": qid, "text": text})
            indices.append(index)
        by_id[qid] = by_text[text]
    merged = {}
    for row in qrels:
        original = str(row["query_id"])
        if original not in by_id:
            raise ValueError("qrels_1ka contains an unknown query")
        pair = (by_id[original], str(row["corpus_id"]))
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError("non-finite official qrel")
        if score > merged.get(pair, float("-inf")):
            merged[pair] = score
    labels = [{"query_id": qid, "corpus_id": cid, "score": score}
              for (qid, cid), score in sorted(merged.items())]
    return canonical, np.asarray(indices, dtype=np.int64), labels


def load_vectors(path, expected_rows):
    state_path = path.with_name(path.stem + "-state.json")
    if state_path.exists():
        state = read_json(state_path)
        if state.get("completed") != expected_rows or state.get("total", expected_rows) != expected_rows:
            raise ValueError("cached vectors are incomplete: " + str(path))
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (expected_rows, 768):
        raise ValueError("cached vector shape does not match its query file: " + str(path))
    for beginning in range(0, expected_rows, 1024):
        block = np.asarray(vectors[beginning:beginning + 1024], dtype=np.float32)
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=.01):
            raise ValueError("invalid cached unit vectors: " + str(path))
    return vectors


def checked_encoding_state(dataset, path):
    # Importing this pure fingerprint helper performs no API or credential I/O.
    from autodl_evaluate_quality import fingerprint
    state_path = path.with_name(path.stem + "-state.json")
    state = read_json(state_path)
    if state.get("fingerprint") != fingerprint(dataset, path.stem):
        raise ValueError("cached encoding input fingerprint differs: " + str(path))
    if not state.get("api_boot_id"):
        raise ValueError("cached encoding has no boot identity: " + str(path))
    return state


def caption_vectors(dataset, quality, original_queries, dedicated_name, expanded_name):
    dedicated = quality / dedicated_name
    if dedicated.exists():
        checked_encoding_state(dataset, dedicated)
        return load_vectors(dedicated, len(original_queries)), {
            "method": "dedicated_exact_official_caption_encoding", "file": str(dedicated.resolve()),
            "sha256": file_hash(dedicated), "rows": len(original_queries)}
    expanded = read_jsonl(dataset / "queries.jsonl")
    lookup = {}
    for index, query in enumerate(expanded):
        lookup.setdefault(query["text"], index)
    missing = [query for query in original_queries if query["text"] not in lookup]
    if missing:
        raise ValueError(f"{dedicated_name} missing; {len(missing)} official captions have no byte-exact "
                         "expanded-caption match. Encode the official texts separately; substitution is forbidden.")
    source_path = quality / expanded_name
    checked_encoding_state(dataset, source_path)
    source = load_vectors(source_path, len(expanded))
    indices = np.asarray([lookup[query["text"]] for query in original_queries], dtype=np.int64)
    return np.asarray(source[indices]), {
        "method": "expanded_caption_byte_exact_text_lookup", "file": str(source_path.resolve()),
        "sha256": file_hash(source_path), "rows": len(original_queries),
        "matched_expanded_row_indices": indices.tolist()}


def normalized(vectors, dimension):
    values = np.asarray(vectors[:, :dimension], dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("invalid truncated vectors")
    return values / norms


def retrieval_rows(scores, query_ids, corpus_ids, qrels):
    """Exact ranking and the same binary/graded metrics as the main evaluator."""
    relevant = positive_map(qrels)
    gallery = list(map(str, corpus_ids))
    if scores.shape != (len(query_ids), len(gallery)) or len(set(gallery)) != len(gallery):
        raise ValueError("invalid exact retrieval score shape")
    rows = []
    for index, qid in enumerate(map(str, query_ids)):
        gold = relevant.get(qid, {})
        if not gold or not set(gold) <= set(gallery):
            raise ValueError("official query lacks valid positives")
        if not np.isfinite(scores[index]).all():
            raise ValueError("non-finite cosine scores")
        order = np.argsort(-scores[index], kind="stable")
        ranked = [gallery[item] for item in order]
        positions = [rank + 1 for rank, cid in enumerate(ranked) if cid in gold]
        first = positions[0]
        dcg = sum((2 ** gold[cid] - 1) / math.log2(rank + 2)
                  for rank, cid in enumerate(ranked[:10]) if cid in gold)
        ideal = sorted(gold.values(), reverse=True)[:10]
        idcg = sum((2 ** score - 1) / math.log2(rank + 2) for rank, score in enumerate(ideal))
        row = {"query_id": qid, "nDCG@10": dcg / idcg,
               "MRR@10": 1 / first if first <= 10 else 0.0,
               "positive_count": len(gold), "first_positive_rank": first}
        for k in (1, 5, 10, 100):
            matches = sum(position <= k for position in positions)
            row[f"Hit@{k}"] = float(matches > 0)
            row[f"Recall@{k}"] = matches / len(gold)
        rows.append(row)
    return rows


def absolute_metrics(rows, ids, qrels, gallery_ids):
    names = [name for name in rows[0] if name not in
             {"query_id", "positive_count", "first_positive_rank"}]
    values = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
    query_ci = bootstrap_means(values)
    groups = source_components(ids, qrels, gallery_ids)
    if len(groups) == len(ids):
        cluster_ci = query_ci.copy()
    else:
        cluster_ci = bootstrap_means(values, groups) if len(groups) > 1 else None
    return {"queries": len(ids), "corpus": len(gallery_ids),
            "positive_source_clusters": len(groups),
            "metrics": {name: float(values[:, index].mean()) for index, name in enumerate(names)},
            "query_bootstrap_ci95": {name: query_ci[:, index].tolist() for index, name in enumerate(names)},
            "positive_source_cluster_ci95": {
                name: cluster_ci[:, index].tolist() if cluster_ci is not None else None
                for index, name in enumerate(names)}, "per_query": rows}


def video_1ka_supplement(dataset, quality):
    original = read_jsonl(dataset / "queries_1ka.jsonl")
    original_qrels = read_jsonl(dataset / "qrels_1ka.jsonl")
    corpus = read_jsonl(dataset / "corpus.jsonl")
    if not corpus or not all("frame_paths" in item for item in corpus):
        raise ValueError("1K-A supplement requires a real-video frame corpus")
    queries, indices, qrels = merge_exact_caption_qrels(original, original_qrels)
    query_ids = [str(query["id"]) for query in queries]
    corpus_ids = [str(item["id"]) for item in corpus]
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("duplicate video gallery IDs")
    query_raw, query_source = caption_vectors(dataset, quality, original, "queries_1ka.npy", "queries.npy")
    document_raw, document_source = caption_vectors(dataset, quality, original,
                                                   "caption-documents-1ka.npy", "caption-documents.npy")
    queries_vector = np.asarray(query_raw[indices])
    documents_vector = np.asarray(document_raw[indices])
    corpus_path = quality / "corpus.npy"
    states = [checked_encoding_state(dataset, path) for path in
              (corpus_path, Path(query_source["file"]), Path(document_source["file"]))]
    if len({state["api_boot_id"] for state in states}) != 1:
        raise ValueError("official-caption roles came from different service boots")
    media_vector = load_vectors(corpus_path, len(corpus))
    reversed_qrels = [{"query_id": row["corpus_id"], "corpus_id": row["query_id"], "score": row["score"]}
                     for row in qrels]
    manifest = read_json(dataset / "manifest.json")
    result = {"dataset": dataset.name, "status": "complete",
              "protocol": "strict official single-caption 1K-A labels on a fixed smaller video gallery",
              "is_full_standard_1ka_benchmark": False,
              "original_standard_candidate_count": manifest.get("original_candidate_count", 1000),
              "candidate_count": len(corpus_ids), "source_query_rows": len(original),
              "unique_exact_caption_queries": len(query_ids),
              "duplicate_caption_rows_merged": len(original) - len(query_ids),
              "qrels_after_exact_caption_merge": len(qrels),
              "qrels_policy": "Union supplied qrels_1ka by exact text; no diagonal labels or approximate caption matching.",
              "query_vectors": query_source, "caption_document_vectors": document_source,
              "api_boot_id": states[0]["api_boot_id"],
              "cache_validation": "Exact input fingerprints, completed shapes/unit norms and one shared service boot checked for all roles.",
              "corpus_vector_sha256": file_hash(corpus_path),
              "source_file_sha256": {name: file_hash(dataset / name) for name in
                                     ("corpus.jsonl", "queries_1ka.jsonl", "qrels_1ka.jsonl")},
              "dimensions": {}}
    forward_matrices, reverse_matrices = {}, {}
    for dimension in DIMENSIONS:
        gallery = normalized(media_vector, dimension)
        forward = retrieval_rows(normalized(queries_vector, dimension) @ gallery.T,
                                 query_ids, corpus_ids, qrels)
        reverse = retrieval_rows(gallery @ normalized(documents_vector, dimension).T,
                                 corpus_ids, query_ids, reversed_qrels)
        result["dimensions"][str(dimension)] = {
            "text_to_video": absolute_metrics(forward, query_ids, qrels, corpus_ids),
            "video_to_text": absolute_metrics(reverse, corpus_ids, reversed_qrels, query_ids)}
        forward_matrices[dimension] = np.asarray([[row[metric] for metric in METRICS] for row in forward])
        reverse_matrices[dimension] = np.asarray([[row[metric] for metric in METRICS] for row in reverse])
    result["paired_dimension_deltas"] = {
        "text_to_video": paired_dimension_deltas(forward_matrices, query_ids, qrels, corpus_ids),
        "video_to_text": paired_dimension_deltas(reverse_matrices, corpus_ids, reversed_qrels, query_ids)}
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--datasets-root", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--skip-video-1ka", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="write explicit incomplete statuses rather than return a nonzero exit code")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    quality_root = args.quality_root or args.root / "quality"
    datasets_root = args.datasets_root or args.root / "datasets"
    output = args.output or args.root / "quality-dimension-deltas.json"
    selected = args.datasets or sorted(path.name for path in quality_root.iterdir()
                                      if path.is_dir() and (datasets_root / path.name / "qrels.jsonl").exists())
    if not selected:
        raise SystemExit("no quality dataset directories found")
    result = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
              "reference_dimension": 768, "metrics": list(METRICS),
              "delta_units": "signed absolute percentage points: dimension minus 768",
              "ci_method": "paired percentile bootstrap, resampling queries or entire positive-source components",
              "ci_scope": "Pointwise 95% intervals conditional on the fixed candidate gallery; no multiple-testing adjustment or equivalence claim.",
              "retrieval_R_at_K_note": "Traditional video/image retrieval R@K usually corresponds to Hit@K here; Recall@K is fraction of all positives retrieved.",
              "numpy_version": np.__version__, "selected_datasets": selected,
              "datasets": [], "video_1ka_supplements": [], "errors": []}
    for name in selected:
        dataset, quality = datasets_root / name, quality_root / name
        try:
            result["datasets"].append(dataset_deltas(dataset, quality, not args.forward_only))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["datasets"].append({"dataset": name, "status": "incomplete", "error": str(error)})
            result["errors"].append({"dataset": name, "stage": "paired_deltas", "error": str(error)})
        if not args.skip_video_1ka and (dataset / "queries_1ka.jsonl").exists():
            try:
                result["video_1ka_supplements"].append(video_1ka_supplement(dataset, quality))
            except (OSError, ValueError, KeyError, TypeError) as error:
                result["video_1ka_supplements"].append({"dataset": name, "status": "incomplete", "error": str(error)})
                result["errors"].append({"dataset": name, "stage": "strict_video_1ka", "error": str(error)})
    result["status"] = "complete" if not result["errors"] else "incomplete"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output.resolve()), "status": result["status"],
                      "datasets": len(result["datasets"]), "video_1ka_supplements": len(result["video_1ka_supplements"]),
                      "errors": result["errors"]}, ensure_ascii=False))
    return 0 if not result["errors"] or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
