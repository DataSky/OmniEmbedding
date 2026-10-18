#!/usr/bin/env python3
"""Prepare pinned, auditable public datasets for an embedding service evaluation.

This script never calls an inference service or reads service credentials.
Requires polars (or duckdb/pyarrow) and Pillow. Raw downloads and selection manifests are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/autodl-evaluation-20261018/datasets"
SEED = 20261018


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class Preparer:
    def __init__(self, output: Path):
        self.output = output
        self.downloads = output / ".downloads"
        self.downloads.mkdir(parents=True, exist_ok=True)

    def download(self, repo: str, revision: str, filename: str):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
        target = self.downloads / repo.replace("/", "--") / revision / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            partial = target.with_suffix(target.suffix + ".part")
            for attempt in range(4):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "OmniEmbedding-evaluation/1.0"})
                    with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                    partial.replace(target)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        raw = target.read_bytes()
        return target, {"repo": repo, "revision": revision, "file": filename,
                        "url": url, "sha256": digest(raw), "bytes": len(raw)}

    @staticmethod
    def parquet(path: Path):
        # Polars Python rows avoid a NumPy ABI dependency in older local Arrow installs.
        try:
            import polars as pl
        except ImportError:
            pass
        else:
            return pl.read_parquet(path).to_dicts()
        try:
            import duckdb
        except ImportError:
            pass
        else:
            with duckdb.connect() as connection:
                result = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
                names = [column[0] for column in result.description]
                return [dict(zip(names, row)) for row in result.fetchall()]
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()

    def finish(self, name, corpus, queries, qrels, sources, **metadata):
        folder = self.output / name
        folder.mkdir(parents=True, exist_ok=True)
        corpus_ids = [str(row["id"]) for row in corpus]
        query_ids = [str(row["id"]) for row in queries]
        if len(set(corpus_ids)) != len(corpus_ids) or len(set(query_ids)) != len(query_ids):
            raise ValueError(f"Duplicate corpus/query IDs in {name}")
        cids, qids = set(corpus_ids), set(query_ids)
        pairs = set()
        covered = set()
        for row in qrels:
            key = (str(row["query_id"]), str(row["corpus_id"]))
            if key in pairs:
                raise ValueError(f"Duplicate relevance judgement {key}")
            pairs.add(key)
            if key[0] not in qids or key[1] not in cids:
                raise ValueError(f"Missing relevance endpoint: {key}")
            if row["score"] > 0:
                covered.add(key[0])
        if covered != qids:
            raise ValueError(f"Queries without positive documents: {sorted(qids - covered)[:10]}")
        for filename, rows in [("corpus.jsonl", corpus), ("queries.jsonl", queries), ("qrels.jsonl", qrels)]:
            write_jsonl(folder / filename, rows)
        write_json(folder / "selected_ids.json", {"corpus": corpus_ids, "queries": query_ids})
        files = {p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
                 for p in [*folder.glob("*.jsonl"), folder / "selected_ids.json"]}
        manifest = {"name": name, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seed": SEED, "counts": {"corpus": len(corpus), "queries": len(queries), "qrels": len(qrels)},
                    "validation": {"unique_corpus_ids": True, "unique_query_ids": True,
                                   "qrels_endpoints_present": True, "every_query_has_positive": True},
                    "sources": sources, "files": files, **metadata}
        write_json(folder / "manifest.json", manifest)
        print(json.dumps({"prepared": name, **manifest["counts"]}), flush=True)

    def scifact(self):
        repo, revision = "BeIR/scifact", "b3b5335604bf5ee3c4447671af975ea25143d4f5"
        cp, cs = self.download(repo, revision, "corpus/corpus-00000-of-00001.parquet")
        qp, qs = self.download(repo, revision, "queries/queries-00000-of-00001.parquet")
        rp, rs = self.download("BeIR/scifact-qrels", "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029", "test.tsv")
        qrels = []
        for line in rp.read_text().splitlines():
            cols = line.split("\t")
            if not cols or cols[0] == "query-id":
                continue
            qrels.append({"query_id": cols[0], "corpus_id": cols[1], "score": int(cols[2])})
        qids = {r["query_id"] for r in qrels}
        corpus = [{"id": str(r["_id"]), "text": r["text"], "title": r.get("title", "")} for r in self.parquet(cp)]
        queries = [{"id": str(r["_id"]), "text": r["text"]} for r in self.parquet(qp) if str(r["_id"]) in qids]
        self.finish("scifact", corpus, queries, qrels, [cs, qs, rs], language="en", split="test",
                    license="CC-BY-SA-4.0", selection="Full 5,183-document corpus and all test qrels / queries.",
                    limitations=["Public benchmark overlap with model training cannot be ruled out.",
                                 "Current deployed service uses its production SearchQuery prompt; official FactChecking prompt is a different setting."])

    def belebele(self, language):
        revision = "979a211276faa22f671e69d096634193567cfd05"
        path, source = self.download("mteb/belebele", revision, f"data/{language}.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        # Match MTEB's passage-by-link and unique-question conversion; stable hashes
        # replace the upstream nondeterministic set enumeration for reproducibility.
        passages = {}
        question_map, judgement_map = {}, {}
        for i, row in enumerate(rows):
            passage = row["flores_passage"]
            cid = "p-" + digest(row["link"].encode())[:20]
            if cid in passages and passages[cid]["text"] != passage:
                raise ValueError("Passage hash collision")
            passages[cid] = {"id": cid, "text": passage, "title": ""}
            qid = "q-" + digest(row["question"].encode())[:20]
            question_map[qid] = {"id": qid, "text": row["question"]}
            judgement_map[(qid, cid)] = {"query_id": qid, "corpus_id": cid, "score": 1}
        queries, qrels = list(question_map.values()), list(judgement_map.values())
        self.finish("belebele-" + language, list(passages.values()), queries, qrels, [source],
                    language=language, split="test", license="CC-BY-SA-4.0",
                    selection="Full language test split, MTEB question-to-passage conversion: passages grouped by link; unique question strings; all associated passage qrels. No answer options or answer labels enter embeddings.",
                    conversion_source="https://github.com/embeddings-benchmark/mteb/blob/1.39.4/mteb/tasks/Retrieval/multilingual/BelebeleRetrieval.py",
                    limitations=["Small candidate pool makes this a language diagnostic, not large-scale retrieval evidence.",
                                 "This explicit conversion is question-only retrieval, not multiple-choice reading comprehension accuracy.",
                                 "Public benchmark overlap with model training cannot be ruled out."])

    def video_text(self):
        repo, rev = "C-MTEB/VideoRetrieval", "58c2597a5943a2ba48f4668c3b90d796283c5639"
        cp, cs = self.download(repo, rev, "data/corpus-00000-of-00001-3f9eee0a3ef9663e.parquet")
        qp, qs = self.download(repo, rev, "data/queries-00000-of-00001-718fd9848565c08f.parquet")
        rp, rs = self.download("C-MTEB/VideoRetrieval-qrels", "faa71382b6a29cf1778d1f436b963e75cb5b927c", "data/dev-00000-of-00001-06b3de608f2202bb.parquet")
        corpus_all, queries_all, qrels_all = self.parquet(cp), self.parquet(qp), self.parquet(rp)
        positive_qids = {str(r["qid"]) for r in qrels_all if r["score"] > 0}
        rng = random.Random(SEED)
        eligible = sorted([str(r["id"]) for r in queries_all if str(r["id"]) in positive_qids])
        qids = set(rng.sample(eligible, min(500, len(eligible))))
        # Preserve EVERY judged document (including zero judgements, if any) for the selected queries.
        qrels = [{"query_id": str(r["qid"]), "corpus_id": str(r["pid"]), "score": int(r["score"])}
                 for r in qrels_all if str(r["qid"]) in qids]
        required = {r["corpus_id"] for r in qrels}
        corpus_map = {str(r["id"]): r for r in corpus_all}
        if not required <= set(corpus_map):
            raise ValueError("Missing original positive/judged documents")
        remaining = sorted(set(corpus_map) - required)
        target_size = min(20000, len(corpus_map))
        if len(required) > target_size:
            raise ValueError("Cannot preserve all judged documents inside requested candidate size")
        cids = required | set(rng.sample(remaining, target_size - len(required)))
        corpus = [{"id": str(r["id"]), "text": r["text"], "title": ""} for r in corpus_all if str(r["id"]) in cids]
        queries = [{"id": str(r["id"]), "text": r["text"]} for r in queries_all if str(r["id"]) in qids]
        self.finish("video-text-zh-subset", corpus, queries, qrels, [cs, qs, rs],
                    language="zh", split="dev", license="Not declared in source dataset card; original rights apply.",
                    selection=f"Python random.Random({SEED}); sorted eligible IDs then sample 500 queries; retain all their judged documents, then random remaining corpus IDs to 20,000.",
                    full_counts={"corpus": len(corpus_all), "queries": len(queries_all), "qrels": len(qrels_all)},
                    selected_query_positive_judgements_preserved=True,
                    limitations=["A fixed reduced-corpus diagnostic, not the full C-MTEB benchmark; scores are not leaderboard-comparable.",
                                 "Despite its name, this source contains TEXT descriptions/titles, not video frames; it does not assess video understanding.",
                                 "Unjudged candidates are treated as nonrelevant; original qrels can be incomplete.",
                                 "Public benchmark overlap with model training cannot be ruled out."])

    def coco(self):
        from PIL import Image
        repo, revision = "mteb/coco-modality-equivalence", "8f0d56ee81cb60167d956f2eff75f1a719e4787b"
        cp, cs = self.download(repo, revision, "t2i-corpus/test-00000-of-00001.parquet")
        qp, qs = self.download(repo, revision, "t2i-queries/test-00000-of-00001.parquet")
        rp, rs = self.download(repo, revision, "t2i-qrels/test-00000-of-00001.parquet")
        folder = self.output / "coco-modality-120"
        image_folder = folder / "images"
        image_folder.mkdir(parents=True, exist_ok=True)
        corpus = []
        image_hashes = []
        for row in self.parquet(cp):
            cid = str(row["id"])
            if not cid.isdigit():
                raise ValueError("Unexpected image ID")
            raw = row["image"]["bytes"]
            if not raw:
                raise ValueError("Source does not contain image bytes")
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                width, height = image.size
                if min(width, height) < 1:
                    raise ValueError("Empty image")
                suffix = ".png" if image.format == "PNG" else ".jpg"
            image_path = image_folder / (cid + suffix)
            image_path.write_bytes(raw)
            if digest(image_path.read_bytes()) != digest(raw):
                raise ValueError("Stored image bytes differ from source row")
            corpus.append({"id": cid, "image_path": str(image_path.relative_to(folder)), "width": width, "height": height})
            image_hashes.append({"id": cid, "sha256": digest(raw), "bytes": len(raw), "path": str(image_path.relative_to(folder))})
        queries = [{"id": str(r["id"]), "text": r["text"]} for r in self.parquet(qp)]
        qrels = [{"query_id": str(r["query-id"]), "corpus_id": str(r["corpus-id"]), "score": int(r["score"])} for r in self.parquet(rp)]
        write_jsonl(folder / "image_hashes.jsonl", image_hashes)
        self.finish("coco-modality-120", corpus, queries, qrels, [cs, qs, rs],
                    language="en", split="test", modalities=["text", "image"],
                    license="MTEB task metadata: CC-BY-SA-4.0; COCO annotations: CC-BY-4.0; images retain their individual Flickr rights.",
                    license_sources=["https://docs.mteb.org/overview/available_tasks/retrieval/", "https://cocodataset.org/#termsofuse"],
                    selection="Full pinned MTEB COCOModalEquivT2IRetrieval test split. Image-to-text can use the transposed qrels and the same embeddings.",
                    limitations=["Only 120 candidates and one caption per image; this is not the standard COCO 1K/5K five-caption retrieval protocol.",
                                 "No labels, image filenames, or metadata should be included in model inputs.",
                                 "Public benchmark overlap with model training cannot be ruled out."])

    def coco_5k(self):
        """Full pinned 5K-image MTEB task, preserving multi-positive qrels."""
        from PIL import Image
        repo, revision = "mteb/mbeir_mscoco_task0", "8dfc4d250094f4e241bcb93211d6c5fbc99bb0bd"
        cp, cs = self.download(repo, revision, "corpus-00000-of-00001.parquet")
        qp, qs = self.download(repo, revision, "query-00000-of-00001.parquet")
        rp, rs = self.download(repo, revision, "qrels-00000-of-00001.parquet")
        queries_all, qrels_all = self.parquet(qp), self.parquet(rp)
        query_map = {str(r["id"]): r for r in queries_all}
        if len(query_map) != len(queries_all):
            raise ValueError("Duplicate query IDs in original COCO 5K task")
        queries = [{"id": qid, "text": row["text"]} for qid, row in query_map.items()]
        pair_map = {}
        for row in qrels_all:
            qid, cid = str(row["query-id"]), str(row["corpus-id"])
            key = qid, cid
            pair_map[key] = {"query_id": qid, "corpus_id": cid,
                             "score": max(int(row["score"]), pair_map.get(key, {}).get("score", 0))}
        folder = self.output / "coco5k"
        image_folder = folder / "images"
        image_folder.mkdir(parents=True, exist_ok=True)
        corpus, image_hashes = [], []
        for row in self.parquet(cp):
            cid = str(row["id"])
            raw = row["image"]["bytes"]
            if not raw:
                raise ValueError("Source does not contain image bytes")
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                width, height = image.size
                suffix = ".png" if image.format == "PNG" else ".jpg"
            image_path = image_folder / (digest(cid.encode())[:24] + suffix)
            image_path.write_bytes(raw)
            if digest(image_path.read_bytes()) != digest(raw):
                raise ValueError("Stored image bytes differ from source row")
            corpus.append({"id": cid, "image_path": str(image_path.relative_to(folder)), "width": width, "height": height})
            image_hashes.append({"id": cid, "sha256": digest(raw), "bytes": len(raw), "path": str(image_path.relative_to(folder)),
                                 "stored_width": width, "stored_height": height})
        write_jsonl(folder / "image_hashes.jsonl", image_hashes)
        if len(pair_map) != len(qrels_all):
            raise ValueError("Duplicate original COCO qrel pairs require explicit review")
        positive_counts = {}
        for row in pair_map.values():
            if row["score"] > 0:
                positive_counts[row["query_id"]] = positive_counts.get(row["query_id"], 0) + 1
        self.finish("coco5k", corpus, queries, list(pair_map.values()), [cs, qs, rs],
                    language="en", split="test", modalities=["text", "image"],
                    license="MTEB task metadata: CC-BY-SA-4.0; COCO annotations: CC-BY-4.0; images retain their individual Flickr rights.",
                    license_sources=["https://docs.mteb.org/overview/available_tasks/retrieval/", "https://cocodataset.org/#termsofuse"],
                    selection="Full pinned MTEB/M-BEIR MSCOCO task: all 5,000 gallery images, all 24,809 caption queries, and every original qrel. No sampling.",
                    full_counts={"corpus": len(corpus), "queries": len(queries_all), "qrels": len(qrels_all)},
                    selected_query_positive_judgements_preserved=True,
                    byte_validation={"images_decoded": len(corpus), "all_output_hashes_equal_source_row_bytes": True,
                                     "distinct_image_sha256": len({r["sha256"] for r in image_hashes}),
                                     "stored_width_min": min(r["width"] for r in corpus),
                                     "stored_width_max": max(r["width"] for r in corpus),
                                     "stored_height_min": min(r["height"] for r in corpus),
                                     "stored_height_max": max(r["height"] for r in corpus)},
                    multi_positive_validation={"queries_with_multiple_positives": sum(n > 1 for n in positive_counts.values()),
                                               "max_positives_per_query": max(positive_counts.values()),
                                               "all_original_judgements_preserved": True},
                    limitations=["Full pinned MTEB/M-BEIR task; task-specific query deduplication and multi-positive labels differ from some conventional COCO protocols.",
                                 "Images use the pinned benchmark's stored resolution; do not compare directly with original-resolution reports.",
                                 "Public benchmark overlap with model training cannot be ruled out."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=["scifact", "belebele-zh", "belebele-en", "video-text", "coco", "coco-5k"],
                        default=["scifact", "belebele-zh", "belebele-en", "video-text", "coco", "coco-5k"])
    args = parser.parse_args()
    preparer = Preparer(args.output.resolve())
    for task in args.tasks:
        if task == "scifact": preparer.scifact()
        elif task == "belebele-zh": preparer.belebele("zho_Hans")
        elif task == "belebele-en": preparer.belebele("eng_Latn")
        elif task == "video-text": preparer.video_text()
        elif task == "coco": preparer.coco()
        elif task == "coco-5k": preparer.coco_5k()


if __name__ == "__main__":
    main()
