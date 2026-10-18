#!/usr/bin/env python3
"""Reproducible retrieval evaluation against the deployed, authenticated API.

Dataset preparation is separate. No training, ANN index, or production data is used.
Raw vectors stay in ignored data/; credentials never enter results or request logs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import httpx
import numpy as np

REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MODEL = "google/embeddinggemma-2"
DIMS = [768, 512, 256, 128]


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def normalize(vectors, dim):
    vectors = np.asarray(vectors[:, :dim], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def relevance_map(qrels):
    result = {}
    for item in qrels:
        score = float(item["score"])
        if score > 0:
            entries = result.setdefault(str(item["query_id"]), {})
            cid = str(item["corpus_id"])
            entries[cid] = max(score, entries.get(cid, 0))
    return result


def retrieval_metrics(scores, query_ids, corpus_ids, qrels, seed=20261018):
    """Exact cosine ranking, graded nDCG, macro recall, and query bootstrap CI."""
    relevant = relevance_map(qrels)
    corpus_ids = [str(x) for x in corpus_ids]
    assert len(corpus_ids) == len(set(corpus_ids)), "duplicate corpus IDs"
    corpus_set = set(corpus_ids)
    corpus_index = {cid: index for index, cid in enumerate(corpus_ids)}
    rows, cases = [], []
    for index, qid in enumerate(map(str, query_ids)):
        gold = relevant.get(qid, {})
        assert gold and set(gold) <= corpus_set, "missing query relevance or positive corpus"
        row_scores = np.asarray(scores[index])
        assert np.isfinite(row_scores).all(), "non-finite ranking scores"
        order = np.argsort(-row_scores, kind="stable")
        ranked = [corpus_ids[j] for j in order[:100]]
        positions = np.flatnonzero(np.isin(order, [corpus_index[cid] for cid in gold])) + 1
        first = int(positions[0])
        dcg = sum((2 ** gold[cid] - 1) / math.log2(rank + 2)
                  for rank, cid in enumerate(ranked[:10]) if cid in gold)
        ideal = sorted(gold.values(), reverse=True)[:10]
        idcg = sum((2 ** value - 1) / math.log2(rank + 2)
                   for rank, value in enumerate(ideal))
        row = {"query_id": qid, "nDCG@10": dcg / idcg,
               "MRR@10": 1 / first if first <= 10 else 0.0,
               "positive_count": len(gold), "first_positive_rank": first}
        for k in [1, 5, 10, 100]:
            matches = int(np.count_nonzero(positions <= k))
            row[f"Recall@{k}"] = matches / len(gold)
            row[f"Hit@{k}"] = float(matches > 0)
        rows.append(row)
        if first > 1 and len(cases) < 15:
            cases.append({"query_id": qid, "gold": list(gold),
                          "first_positive_rank": first, "top10": ranked[:10]})
    names = [key for key in rows[0] if key not in
             {"query_id", "positive_count", "first_positive_rank"}]
    values = np.asarray([[row[name] for name in names] for row in rows])
    rng = np.random.default_rng(seed)
    means = values.mean(axis=0)
    bootstrap = np.empty((1000, len(names)))
    for beginning in range(0, 1000, 25):
        sample_indices = rng.integers(0, len(rows), size=(25, len(rows)))
        bootstrap[beginning:beginning + 25] = values[sample_indices].mean(axis=1)
    bounds = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    result = {"queries": len(rows), "corpus": len(corpus_ids),
            "metrics": {name: float(means[i]) for i, name in enumerate(names)},
            "query_bootstrap_ci95": {name: [float(bounds[0, i]), float(bounds[1, i])]
                                     for i, name in enumerate(names)},
            "bootstrap_replicates": 1000, "bootstrap_seed": seed,
            "per_query": rows, "failure_cases": cases}
    # Captions/questions sharing a source media/passsage are dependent samples.
    # Connect corpus items when one query has multiple positives, then resample
    # whole positive-source components. This complements the conditional query CI.
    parents = {cid: cid for cid in corpus_ids}

    def root(cid):
        while parents[cid] != cid:
            parents[cid] = parents[parents[cid]]
            cid = parents[cid]
        return cid

    for qid in map(str, query_ids):
        positive_ids = list(relevant[qid])
        for cid in positive_ids[1:]:
            parents[root(cid)] = root(positive_ids[0])
    groups = {}
    for index, qid in enumerate(map(str, query_ids)):
        groups.setdefault(root(next(iter(relevant[qid]))), []).append(index)
    if len(groups) < len(rows):
        sizes = np.array([len(indices) for indices in groups.values()])
        totals = np.array([values[indices].sum(axis=0) for indices in groups.values()])
        cluster_bootstrap = np.empty((1000, len(names)))
        for beginning in range(0, 1000, 25):
            indices = rng.integers(0, len(groups), size=(25, len(groups)))
            cluster_bootstrap[beginning:beginning + 25] = (
                totals[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None])
        intervals = np.quantile(cluster_bootstrap, [0.025, 0.975], axis=0)
        result["positive_source_clusters"] = len(groups)
        result["positive_source_cluster_ci95"] = {
            name: [float(intervals[0, i]), float(intervals[1, i])]
            for i, name in enumerate(names)}
    return result


class Service:
    def __init__(self, endpoint, key_file, output):
        self.endpoint = endpoint.rstrip("/")
        key = Path(key_file).read_text().strip()
        self.client = httpx.Client(timeout=120, trust_env=False, follow_redirects=False,
                                   headers={"Authorization": "Bearer " + key})
        self.log = (output / "encoding-requests.jsonl").open("a", buffering=1)
        ready = self.client.get(self.endpoint + "/health/ready")
        ready.raise_for_status()
        identity = ready.json()
        assert identity["ready"] and identity["model"] == MODEL
        assert identity["model_revision"] == REVISION
        self.boot_id = identity["boot_id"]
        self.dtype, self.device = identity["dtype"], identity["device"]

    def close(self):
        self.client.close()
        self.log.close()

    def encode_request(self, path, rows, *, payload=None, files=None, label=""):
        for attempt in range(6):
            started = time.perf_counter()
            request = self.client.build_request("POST", self.endpoint + path,
                                                json=payload, files=files)
            response = self.client.send(request)
            elapsed = time.perf_counter() - started
            record = {"time_utc": datetime.now(timezone.utc).isoformat(),
                      "label": label, "status": response.status_code, "rows": rows,
                      "attempt": attempt + 1, "elapsed_s": elapsed,
                      "response_bytes": len(response.content)}
            self.log.write(json.dumps(record) + "\n")
            if response.status_code == 429 and attempt < 5:
                time.sleep(2 + attempt * 0.1)
                continue
            response.raise_for_status()
            data = response.json()
            assert data["model"] == MODEL and data["model_revision"] == REVISION
            assert data["boot_id"] == self.boot_id, "service restarted during quality encoding"
            assert data["dimensions"] == 768
            assert [entry["index"] for entry in data["data"]] == list(range(rows))
            vectors = np.asarray([entry["embedding"] for entry in data["data"]], dtype=np.float32)
            assert vectors.shape == (rows, 768) and np.isfinite(vectors).all()
            assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.001)
            return vectors
        raise RuntimeError("request retries exhausted")


def resolve_file(dataset, value):
    path = Path(value)
    return path if path.is_absolute() or path.exists() else dataset / path


def fingerprint(dataset, role):
    digest = hashlib.sha256()
    source_role = {"caption-documents": "queries",
                   "caption-documents-1ka": "queries_1ka"}.get(role, role)
    # Hash actual encoder inputs, not timestamps or incidental manifest prose.
    digest.update((dataset / (source_role + ".jsonl")).read_bytes())
    digest.update((MODEL + REVISION + role + "deployed-prefix-v2").encode())
    for item in read_jsonl(dataset / (source_role + ".jsonl")):
        assets = ([item["image_path"]] if "image_path" in item else item.get("frame_paths", []))
        for asset in assets:
            digest.update(str(asset).encode())
            digest.update(hashlib.sha256(resolve_file(dataset, asset).read_bytes()).digest())
    return digest.hexdigest()


def encode_items(service, dataset, output, role, items, batch_size):
    vector_path, state_path = output / (role + ".npy"), output / (role + "-state.json")
    signature = fingerprint(dataset, role)
    state = {"fingerprint": signature, "completed": 0, "total": len(items),
             "truncated_texts": 0, "api_boot_id": service.boot_id,
             "dtype": service.dtype, "device": service.device}
    assert vector_path.exists() == state_path.exists(), "vector/checkpoint files must exist as a pair"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        assert previous["fingerprint"] == signature, "dataset or model changed; use new output folder"
        assert previous["api_boot_id"] == service.boot_id, "service restarted; use a new encoding output folder"
        assert previous.get("dtype", service.dtype) == service.dtype, "service precision changed"
        state = previous
        state.update(dtype=service.dtype, device=service.device)
    assert 0 <= state["completed"] <= len(items) and state["total"] == len(items)
    mode = "r+" if vector_path.exists() else "w+"
    vectors = np.lib.format.open_memmap(vector_path, mode=mode, dtype=np.float32,
                                        shape=(len(items), 768))
    assert vectors.shape == (len(items), 768) and vectors.dtype == np.float32
    cursor = state["completed"]
    if cursor:
        assert np.isfinite(vectors[:cursor]).all(), "cached vectors contain invalid values"
        assert np.allclose(np.linalg.norm(vectors[:cursor], axis=1), 1, atol=0.001)
    save_json(state_path, state)
    is_document = role in {"corpus", "caption-documents", "caption-documents-1ka"}
    while cursor < len(items):
        item = items[cursor]
        if "image_path" in item:
            image = resolve_file(dataset, item["image_path"])
            encoded = service.encode_request("/v1/embeddings/image?dimensions=768", 1,
                files={"file": (image.name, image.read_bytes(),
                                 "image/png" if image.suffix.lower() == ".png" else "image/jpeg")},
                label=f"{dataset.name}/{role}/{cursor}")
            end = cursor + 1
        elif "frame_paths" in item:
            frames = [resolve_file(dataset, value) for value in item["frame_paths"]]
            assert 1 <= len(frames) <= 8
            encoded = service.encode_request("/v1/embeddings/video?dimensions=768", 1,
                files=[("files", (frame.name, frame.read_bytes(), "image/jpeg")) for frame in frames],
                label=f"{dataset.name}/{role}/{cursor}")
            end = cursor + 1
        else:
            texts, titles, total = [], [], 0
            end = cursor
            while end < len(items) and end - cursor < batch_size:
                original = items[end].get("text", "")
                title = items[end].get("title", "")[:1024] if is_document else ""
                text = original[:8192]
                length = len(text) + len(title)
                if texts and total + length > 32768:
                    break
                texts.append(text)
                titles.append(title)
                state["truncated_texts"] += int(len(original) > 8192)
                total += length
                end += 1
            payload = {"input": texts, "input_type": "document" if is_document else "query",
                       "dimensions": 768}
            if is_document:
                payload["titles"] = titles
            encoded = service.encode_request("/v1/embeddings", len(texts), payload=payload,
                                               label=f"{dataset.name}/{role}/{cursor}:{end}")
        vectors[cursor:end] = encoded
        vectors.flush()
        cursor = end
        state["completed"] = cursor
        save_json(state_path, state)
        if cursor % 160 == 0 or cursor == len(items):
            print(f"{dataset.name} {role}: {cursor}/{len(items)}", flush=True)
    return np.asarray(vectors)


def evaluate_dataset(args, dataset, service):
    output = args.output / dataset.name
    output.mkdir(parents=True, exist_ok=True)
    corpus = read_jsonl(dataset / "corpus.jsonl")
    queries = read_jsonl(dataset / "queries.jsonl")
    qrels = read_jsonl(dataset / "qrels.jsonl")
    assert corpus and queries and qrels
    assert len({str(x["id"]) for x in queries}) == len(queries)
    multimodal = "image_path" in corpus[0] or "frame_paths" in corpus[0]
    if args.metrics_only:
        boot_ids = set()
        for role, items in [("corpus", corpus), ("queries", queries)]:
            state = json.loads((output / (role + "-state.json")).read_text())
            assert state["fingerprint"] == fingerprint(dataset, role)
            assert state["completed"] == len(items), "encoding is incomplete"
            boot_ids.add(state["api_boot_id"])
        document_vectors = np.load(output / "corpus.npy", mmap_mode="r")
        query_vectors = np.load(output / "queries.npy", mmap_mode="r")
        if multimodal:
            state = json.loads((output / "caption-documents-state.json").read_text())
            assert state["fingerprint"] == fingerprint(dataset, "caption-documents")
            assert state["completed"] == len(queries)
            boot_ids.add(state["api_boot_id"])
            caption_document_vectors = np.load(output / "caption-documents.npy", mmap_mode="r")
        boot_id = state["api_boot_id"]
        assert len(boot_ids) == 1, "cached roles came from different service boots"
        cached = [(document_vectors, len(corpus)), (query_vectors, len(queries))]
        if multimodal:
            cached.append((caption_document_vectors, len(queries)))
        for vectors, expected_rows in cached:
            assert vectors.shape == (expected_rows, 768), "unexpected cached vector shape"
            assert vectors.dtype == np.float32, "unexpected cached vector precision"
            for beginning in range(0, expected_rows, 1024):
                block = vectors[beginning:beginning + 1024]
                assert np.isfinite(block).all(), "cached vectors contain non-finite values"
                assert np.allclose(np.linalg.norm(block, axis=1), 1, atol=0.001), "cached vectors are not normalized"
    else:
        document_vectors = encode_items(service, dataset, output, "corpus", corpus, args.batch_size)
        query_vectors = encode_items(service, dataset, output, "queries", queries, args.batch_size)
        if multimodal:
            caption_document_vectors = encode_items(service, dataset, output,
                                                     "caption-documents", queries, args.batch_size)
        if (dataset / "queries_1ka.jsonl").exists():
            official_queries = read_jsonl(dataset / "queries_1ka.jsonl")
            encode_items(service, dataset, output, "queries_1ka", official_queries, args.batch_size)
            encode_items(service, dataset, output, "caption-documents-1ka", official_queries, args.batch_size)
        boot_id = service.boot_id
    summary = {"dataset": dataset.name, "model": MODEL, "model_revision": REVISION,
               "boot_id": boot_id, "prefix_configuration": "deployed_search_query_and_document",
               "reverse_text_candidates_use_document_prefix": bool(multimodal),
               "manifest": json.loads((dataset / "manifest.json").read_text()), "dimensions": {}}
    corpus_ids = [str(x["id"]) for x in corpus]
    query_ids = [str(x["id"]) for x in queries]
    if args.encode_only:
        summary["status"] = "encoded_not_scored"
        summary["counts"] = {"corpus": len(corpus), "queries": len(queries)}
        save_json(output / "encoding-summary.json", summary)
        print(f"{dataset.name}: all vectors encoded; scoring deferred", flush=True)
        return summary
    for dimension in DIMS:
        scores = normalize(query_vectors, dimension) @ normalize(document_vectors, dimension).T
        metrics = retrieval_metrics(scores, query_ids, corpus_ids, qrels)
        save_json(output / f"per-query-{dimension}.json", metrics)
        summary["dimensions"][str(dimension)] = {k: v for k, v in metrics.items()
                                                  if k not in {"per_query", "failure_cases"}}
        if multimodal:
            reversed_qrels = [{"query_id": x["corpus_id"], "corpus_id": x["query_id"],
                               "score": x["score"]} for x in qrels]
            reverse_scores = normalize(document_vectors, dimension) @ normalize(caption_document_vectors, dimension).T
            reverse = retrieval_metrics(reverse_scores, corpus_ids, query_ids, reversed_qrels)
            save_json(output / f"reverse-per-query-{dimension}.json", reverse)
            summary["dimensions"][str(dimension)]["media_to_text"] = {
                k: v for k, v in reverse.items() if k not in {"per_query", "failure_cases"}}
    if not multimodal:
        # A lexical sanity baseline, explicitly char TF-IDF rather than BM25.
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=100_000,
                                sublinear_tf=True, dtype=np.float32)
        documents = [x.get("title", "") + " " + x.get("text", "")[:8192] for x in corpus]
        matrix = tfidf.fit_transform(documents)
        lexical_scores = (tfidf.transform([x["text"][:8192] for x in queries]) @ matrix.T).toarray()
        baseline = retrieval_metrics(lexical_scores, query_ids, corpus_ids, qrels)
        save_json(output / "lexical-per-query.json", baseline)
        summary["lexical_char_tfidf_baseline"] = {k: v for k, v in baseline.items()
                                                   if k not in {"per_query", "failure_cases"}}
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(output / "summary.json", summary)
    print(json.dumps({"dataset": dataset.name, "768": summary["dimensions"]["768"]["metrics"]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint-file", type=Path, default=Path("data/autodl/endpoint.json"))
    parser.add_argument("--key-file", type=Path, default=Path("data/autodl/inference.key"))
    parser.add_argument("--batch-size", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--metrics-only", action="store_true",
                        help="recompute metrics from completed cached vectors without API access")
    parser.add_argument("--encode-only", action="store_true",
                        help="encode and checkpoint vectors; defer CPU metrics to a separate run")
    args = parser.parse_args()
    if args.metrics_only and args.encode_only:
        parser.error("metrics-only and encode-only are mutually exclusive")
    args.output.mkdir(parents=True, exist_ok=True)
    service = None
    if not args.metrics_only:
        endpoint = json.loads(args.endpoint_file.read_text())
        service = Service(endpoint["url"], args.key_file, args.output)
    try:
        paths = ([args.datasets_root / name for name in args.datasets] if args.datasets else
                 sorted(path for path in args.datasets_root.iterdir()
                        if path.is_dir() and (path / "manifest.json").exists()))
        summaries = [evaluate_dataset(args, path, service) for path in paths]
        save_json(args.output / "summary.json", summaries)
    finally:
        if service is not None:
            service.close()


if __name__ == "__main__":
    main()
