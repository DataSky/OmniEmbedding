"""Auditable BM25 and exact cosine retrieval with weighted reciprocal ranks."""
from __future__ import annotations

import time
import uuid
import re
import hashlib
from collections import OrderedDict
import threading

import numpy as np

from .store import DIMENSIONS, MODEL, REVISION, RRF_K, tokens


# Small, disclosed software vocabulary for cross-language lexical queries.
# This is independent of corpus/project names and does not invoke an LLM.
SOFTWARE_TERMS = {"架构": ("architecture",), "运行时": ("runtime",),
                  "模块": ("module", "modules"), "初始化": ("initialization", "initialize", "init"),
                  "修改": ("变更", "更改", "update", "edit")}
CJK_CONNECTORS = re.compile(r"分别|哪里|哪个|什么|如何|怎么|是否|需要|可以|进行|以及|或者|一个|我们|你们|他们|和|与|的|在|吗|呢|了")
ENGLISH_CONNECTORS = {"how", "to", "the", "and", "or", "of", "for", "is", "a", "an", "in", "on", "does", "do"}
GENERATED_PARTS = {".build", ".data", "node_modules", "build", "dist", "deriveddata", "pods", "__pycache__"}


class StoredVectorError(ValueError):
    """Invalid persistent vectors, distinct from remote query-encoding errors."""


class LexicalHits(list):
    """Keep per-request query planning and reranking evidence thread-local."""

    def __init__(self, rows, strategy, expression, features=None, duplicates=None):
        super().__init__(rows)
        self.strategy, self.expression = strategy, expression
        self.features, self.duplicates = features or {}, duplicates or {}


def query_units(query):
    identifiers, primary, expanded, topics, phrases = [], [], [], [], []
    quote = lambda value: '"' + value + '"'
    values = re.findall(r"[\u3400-\u9fff]+|[A-Za-z0-9_]+", re.sub(r"\{[^}]{0,256}\}", " ", query))[:32]
    for value in values:
        if "\u3400" <= value[0] <= "\u9fff":
            for phrase in CJK_CONNECTORS.split(value):
                if not phrase or phrase in {"年", "月", "日", "旧"}:
                    continue
                phrases.append(phrase)
                # Retain recognized multi-character software terms as units;
                # split the remaining text into overlapping Chinese bigrams.
                remaining = phrase
                for term, aliases in SOFTWARE_TERMS.items():
                    if term in remaining:
                        topics.append({"term": term, "aliases": aliases})
                        remaining = remaining.replace(term, " ")
                for part in remaining.split():
                    grams = list(dict.fromkeys(part[i:i+2] for i in range(len(part)-1))) if len(part) > 1 else [part]
                    topics.extend({"term": gram, "aliases": ()} for gram in grams[:32])
        else:
            if len(values) > 1 and value.lower() in ENGLISH_CONNECTORS:
                continue
            # A Chinese date's month/day fragments are not mandatory symbols.
            if value.isdigit() and len(value) < 4 and re.search(r"\d{4}年", query):
                continue
            literal = quote(value.lower())
            components = list(dict.fromkeys(tokens(value)[1:]))
            group = "(" + literal + " OR (" + " AND ".join(map(quote, components)) + "))" if components else literal
            identifiers.append(value.lower()); primary.append(literal); expanded.append(group)
    unique = {}
    for unit in topics:
        unique.setdefault(unit["term"], unit)
    return identifiers, primary, expanded, list(unique.values())[:48], phrases


def topic_expression(unit):
    options = []
    for term in [unit["term"], *unit["aliases"]]:
        grams = [term[i:i+2] for i in range(len(term)-1)] if len(term) > 1 and "\u3400" <= term[0] <= "\u9fff" else [term]
        options.append("(" + " AND ".join('"'+gram+'"' for gram in grams) + ")")
    return "(" + " OR ".join(options) + ")"


def lexical_plan(query):
    """Keep identifiers ANDed with topic evidence, never project-only fallback."""
    _, primary, expanded, topics, _ = query_units(query)
    topic_groups = [topic_expression(unit) for unit in topics]
    strict = [*primary, *topic_groups]
    proposals = []
    if strict:
        proposals.append(("all_query_groups", " AND ".join(strict)))
        proposals.append(("identifier_components", " AND ".join([*expanded, *topic_groups])))
    if topic_groups:
        proposals.append(("anchor_topic_bigrams", " AND ".join([*expanded, "(" + " OR ".join(topic_groups) + ")"])))
    seen, result = set(), []
    for strategy, expression in proposals:
        if expression and expression not in seen:
            seen.add(expression); result.append((strategy, expression))
    return result


def lexical_features(row, query, identifiers, topics, phrases):
    body, path = row["content"].lower(), row["path"].lower()
    body_tokens, path_tokens = set(tokens(row["content"])), set(tokens(row["path"]))
    raw_headings = "\n".join(line for line in row["content"].splitlines() if line.lstrip().startswith("#"))
    headings, heading_tokens = raw_headings.lower(), set(tokens(raw_headings))
    matched, body_hits, heading_hits, positions = [], 0, 0, []
    for unit in topics:
        variants = [unit["term"], *unit["aliases"]]
        present = lambda text, token_set: any(v in text if "\u3400" <= v[0] <= "\u9fff" else v in token_set for v in variants)
        if present(body, body_tokens) or present(path, path_tokens):
            matched.append(unit["term"])
        if present(body, body_tokens):
            body_hits += sum(min(body.count(v), 3) for v in variants)
            for variant in variants:
                positions.extend((match.start(), match.end(), unit["term"]) for match in list(re.finditer(re.escape(variant), body))[:64])
        if present(headings, heading_tokens):
            heading_hits += 1
    coverage = len(matched) / len(topics) if topics else 1.
    # A long intent cannot be supported by one incidental common bigram.
    supported = not topics or (len(matched) >= min(2, len(topics)) and coverage >= .2)
    parts = path.split("/")
    historical = bool(re.search(r"历史|早期|旧|过去|退役|\d{4}年\d{1,2}月", query))
    project_exact = row["project"].lower() in identifiers
    path_exact = any(part in identifiers or part.rsplit(".", 1)[0] in identifiers for part in parts[:-1])
    prior = 1.6 if project_exact else 1.25 if path_exact else 1.
    reasons = ["exact_project" if project_exact else "exact_path_component"] if project_exact or path_exact else []
    generated = any(part in GENERATED_PARTS and part not in query.lower() for part in parts)
    if generated:
        prior *= .08; reasons.append("generated_path_penalty")
    if any(part in {"archive", "archived", "retired", "superseded"} or "retired-design" in part for part in parts) and not historical:
        prior *= .35; reasons.append("historical_path_penalty")
    if "current" in parts and not historical:
        prior *= 2.; reasons.append("current_document")
    if re.search(r"保留设计分析|仅供历史|已被.{0,30}取代|不再维护|已废弃", row["source_header"]):
        prior *= .25; reasons.append("superseded_document_status")
    if "handoff" in parts and historical:
        prior *= 1.25; reasons.append("historical_handoff")
    if topics and row["extension"] in {".md", ".rst", ".txt"}:
        prior *= 1.25; reasons.append("natural_language_document")
    if parts[-1] in {"readme.md", "readme.rst"}:
        prior *= 1.2; reasons.append("readme_entrypoint")
    prior /= 1 + .025 * max(0, len(parts)-4)
    coverage_boost = (.2 + .8 * coverage) ** 2
    # Dense evidence in a matching section is stronger than scattered words in
    # release summaries or proposal lists. Preserve independent raw BM25 below.
    positions.sort()
    needed = len({point[2] for point in positions})
    counts, covered, left, span = {}, 0, 0, len(body) + 1
    for _, end, term in positions:
        counts[term] = counts.get(term, 0) + 1
        if counts[term] == 1:
            covered += 1
        while needed and covered == needed:
            span = min(span, end - positions[left][0])
            old = positions[left][2]; counts[old] -= 1; left += 1
            if counts[old] == 0:
                covered -= 1
    evidence_boost = (1 + .15 * min(body_hits, 5) + .25 * min(heading_hits, 3)) * (1 + .25 * max(0., 1 - span / 240) if positions else 1.)
    return dict(topic_coverage=round(coverage, 6), matched_topic_terms=matched,
                topic_terms=[unit["term"] for unit in topics], topic_supported=supported,
                path_prior=round(prior, 6), path_prior_reasons=reasons,
                coverage_boost=round(coverage_boost, 6), evidence_boost=round(evidence_boost, 6),
                heading_topic_hits=heading_hits, proximity_span=span if positions else None,
                exact_project=project_exact, generated_path=generated, historical_intent=historical)


def filters(project, extension):
    where, args = [], []
    if project:
        where.append("f.project=?"); args.append(project)
    if extension:
        where.append("f.extension=?"); args.append(extension.lower())
    return (" AND " + " AND ".join(where) if where else ""), args


class Search:
    def __init__(self, store, engine):
        self.store, self.engine = store, engine
        self._query_cache = OrderedDict()
        self._cache_lock = threading.Lock()

    def query_vector(self, query):
        key = (MODEL, REVISION, query)
        with self._cache_lock:
            cached = self._query_cache.get(key)
            if cached and time.monotonic()-cached[0] < 600:
                self._query_cache.move_to_end(key)
                return cached[1]
        if not self.engine.enc_lock.acquire(timeout=10):
            raise ValueError("AutoDL 当前繁忙，请稍后重试")
        try:
            array = np.asarray(self.engine.enc.query([query], bs=1), dtype=np.float32)
        finally:
            self.engine.enc_lock.release()
        if array.shape != (1, DIMENSIONS) or not np.isfinite(array).all():
            raise ValueError("查询向量校验失败")
        norm = np.sqrt(np.einsum("i,i->", array[0], array[0], dtype=np.float64, optimize=False))
        if not np.isfinite(norm) or norm < 1e-9:
            raise ValueError("查询向量校验失败")
        vector = array[0] / norm
        with self._cache_lock:
            self._query_cache[key] = (time.monotonic(), vector)
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > 128:
                self._query_cache.popitem(last=False)
        return vector

    def bm25(self, query, candidates, project=None, extension=None):
        plan = lexical_plan(query)
        if not plan:
            return LexicalHits([], "empty_query", "")
        identifiers, _, _, topics, phrases = query_units(query)
        condition, args = filters(project, extension)
        pool_limit = min(2000, max(300, candidates * 8))
        pool, used = {}, []
        with self.store.connect() as db:
            for strategy, expression in plan:
                rows = db.execute("""SELECT c.id,c.content,c.sha256,c.start_line,c.end_line,
                    f.path,f.project,f.extension,substr(f.content,1,800) source_header,bm25(chunk_fts,0.0,4.0,1.0) rank
                    FROM chunk_fts JOIN chunks c ON c.id=chunk_fts.chunk_id
                    JOIN files f ON f.id=c.file_id WHERE chunk_fts MATCH ?""" + condition +
                    " ORDER BY rank,c.id LIMIT ?", [expression, *args, pool_limit]).fetchall()
                if rows:
                    used.append(strategy)
                for row in rows:
                    pool[row["id"]] = row
        ranked, features = [], {}
        for ident, row in pool.items():
            feature = lexical_features(row, query, identifiers, topics, phrases)
            if not feature["topic_supported"]:
                continue
            base = -float(row["rank"])
            value = base * feature["path_prior"] * feature["coverage_boost"] * feature["evidence_boost"]
            feature.update(raw_bm25=base, lexical_score=value)
            features[ident] = feature
            ranked.append((ident, value))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        # Collapse copied passages across branches while retaining provenance.
        unique, by_hash, duplicates = [], {}, {}
        for ident, value in ranked:
            row = pool[ident]
            content = row["content"].strip()
            digest = hashlib.sha256(content.encode()).hexdigest() if len(content) >= 64 else ident
            if digest in by_hash:
                representative = by_hash[digest]
                duplicates.setdefault(representative, []).append(dict(id=ident,path=row["path"],start_line=row["start_line"],end_line=row["end_line"]))
                continue
            by_hash[digest] = ident; unique.append((ident, value))
        strategy = "anchor_topic_coverage" if topics else (used[0] if used else plan[-1][0])
        hits = LexicalHits(unique[:candidates], strategy, plan[-1][1], features, duplicates)
        hits.pool_limit, hits.pool_candidates = pool_limit, len(pool)
        return hits

    def vector(self, query, candidates, project=None, extension=None):
        query_vector = self.query_vector(query)
        condition, args = filters(project, extension)
        top = []
        # Read bounded batches directly from SQLite: no second vector database,
        # no unbounded in-memory copy of the entire code corpus.
        with self.store.connect() as db:
            cursor = db.execute("""SELECT v.chunk_id,v.vector FROM vectors v
                JOIN chunks c ON c.id=v.chunk_id JOIN files f ON f.id=c.file_id
                WHERE v.model=? AND v.revision=? AND v.dimensions=?""" + condition,
                [MODEL, REVISION, DIMENSIONS, *args])
            while rows := cursor.fetchmany(2048):
                if any(len(row["vector"]) != DIMENSIONS * 4 for row in rows):
                    raise StoredVectorError("本地向量字节长度与维度不符")
                matrix = np.frombuffer(b"".join(r["vector"] for r in rows), dtype="<f4").reshape(-1, DIMENSIONS)
                if not np.isfinite(matrix).all():
                    raise StoredVectorError("本地向量包含非有限值")
                norms = np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64, optimize=False)
                if not np.isfinite(norms).all() or not np.allclose(norms, 1., atol=.002):
                    raise StoredVectorError("本地向量未归一化")
                # Some macOS NumPy/BLAS builds emit false divide/overflow/invalid
                # warnings for finite unit-vector matmul (crosschecked against
                # einsum). Accumulate independently in float64, without BLAS;
                # reject actual invalid inputs/scores rather than hiding them.
                scores = np.einsum("ij,j->i", matrix, query_vector, dtype=np.float64, optimize=False)
                if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.00001):
                    raise StoredVectorError("本地余弦得分校验失败")
                count = min(candidates, len(rows))
                # Include a deterministic tie break at the candidate boundary;
                # duplicated code often produces exactly identical cosines.
                indices = np.lexsort((np.array([r["chunk_id"] for r in rows]), -scores))[:count]
                top.extend((rows[int(i)]["chunk_id"], float(np.clip(scores[i], -1, 1))) for i in indices)
                top = sorted(top, key=lambda item:(-item[1], item[0]))[:candidates]
        return top

    def run(self, query, *, mode="hybrid", limit=10, candidates=100, vector_weight=.6, project=None, extension=None):
        started = time.perf_counter()
        if not query.strip():
            raise ValueError("请输入查询内容")
        if mode not in {"hybrid", "vector", "bm25"} or not 1 <= limit <= 100 or not limit <= candidates <= 1000 or not 0 <= vector_weight <= 1:
            raise ValueError("检索参数超出范围")
        sparse, dense, warning = [], [], None
        timings = {}
        coverage = None
        effective = mode
        use_bm = mode == "bm25" or (mode == "hybrid" and vector_weight < 1)
        use_vector = mode == "vector" or (mode == "hybrid" and vector_weight > 0)
        if use_bm:
            mark = time.perf_counter()
            sparse = self.bm25(query, candidates, project, extension)
            timings["bm25_ms"] = round((time.perf_counter()-mark)*1000, 2)
        if use_vector:
            coverage = self.store.coverage(project, extension)
            if coverage["vectors"] == 0:
                warning = "尚无已完成向量；本次仅使用 BM25。请先完成向量化任务。"
                if mode == "vector" or not use_bm:
                    raise ValueError("尚无已完成向量，请先完成向量化任务。")
                effective = "bm25"
            elif mode == "hybrid" and use_bm and coverage["vectors"] < coverage["chunks"]:
                warning = (f"当前范围仅 {coverage['vectors']:,}/{coverage['chunks']:,} 个分块已生成向量，覆盖不完整；"
                           "本次自动仅使用 BM25 词项路，覆盖完整后恢复混合检索。可显式选择语义向量查看已有子集。")
                effective = "bm25"
            else:
                mark = time.perf_counter()
                try:
                    dense = self.vector(query, candidates, project, extension)
                except StoredVectorError:
                    if mode == "vector" or not use_bm:
                        raise ValueError("本地向量数据校验失败，请检查索引；未使用损坏向量。") from None
                    warning = "本地向量数据校验失败，本次已明确降级为 BM25；请检查索引。"
                    effective = "bm25"
                except Exception:
                    if mode == "vector" or not use_bm:
                        raise ValueError("AutoDL 查询编码暂不可用，请查看服务状态或使用 BM25。") from None
                    warning = "AutoDL 查询编码失败，本次已明确降级为 BM25；现有向量仍保留。"
                    effective = "bm25"
                timings["vector_ms"] = round((time.perf_counter()-mark)*1000, 2)
            if effective != "bm25" and coverage["vectors"] < coverage["chunks"]:
                warning = f"当前范围仅 {coverage['vectors']:,}/{coverage['chunks']:,} 个分块已生成向量；语义召回覆盖尚不完整。"
        dense_map = {ident:(rank, score) for rank,(ident,score) in enumerate(dense, 1)}
        sparse_map = {ident:(rank, score) for rank,(ident,score) in enumerate(sparse, 1)}
        vw = vector_weight if effective == "hybrid" else (1. if effective == "vector" else 0.)
        bw = 1-vector_weight if effective == "hybrid" else (1. if effective == "bm25" else 0.)
        ranked = []
        for ident in dense_map.keys() | sparse_map.keys():
            vr, vs = dense_map.get(ident, (None, None))
            br, bs = sparse_map.get(ident, (None, None))
            vc = vw/(RRF_K+vr) if vr else 0.
            bc = bw/(RRF_K+br) if br else 0.
            score = vc+bc if effective == "hybrid" else (vs if effective == "vector" else bs)
            if effective == "vector":
                vc, bc = score, 0.
            elif effective == "bm25":
                vc, bc = 0., score
            ranked.append(dict(id=ident, score=score, vector_score=vs, bm25_score=bs,
                               vector_rank=vr, bm25_rank=br, vector_contribution=vc, bm25_contribution=bc))
        ranked.sort(key=lambda r:(-r["score"], r["id"]))
        # Retain up to three informative passages before additional passages
        # from the same file. A one-passage cap can hide the actual answer even
        # after finding the right document. Preserve every route's raw ranks.
        metadata = {}
        with self.store.connect() as db:
            for begin in range(0, len(ranked), 400):
                identifiers = [item["id"] for item in ranked[begin:begin+400]]
                marks = ",".join("?" for _ in identifiers)
                for row in db.execute("SELECT c.id,c.file_id,c.sha256,length(c.content) content_length,c.start_line,c.end_line,f.path FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.id IN (" + marks + ")", identifiers):
                    metadata[row["id"]] = dict(row)
        selected, deferred, seen_files, seen_hashes, duplicate_sources = [], [], {}, {}, {}
        for raw_rank, item in enumerate(ranked, 1):
            meta = metadata.get(item["id"])
            if meta is None:
                continue
            file_id = meta["file_id"]
            digest = meta["sha256"] if meta["sha256"] and meta["content_length"] >= 64 else item["id"]
            if digest in seen_hashes:
                representative = seen_hashes[digest]
                duplicate_sources.setdefault(representative, []).append({k:meta[k] for k in ("id", "path", "start_line", "end_line")})
                continue
            seen_hashes[digest] = item["id"]
            annotated = {**item, "raw_rank": raw_rank}
            if seen_files.get(file_id, 0) >= 3:
                deferred.append(annotated)
            else:
                seen_files[file_id] = seen_files.get(file_id, 0) + 1; selected.append(annotated)
        selected = (selected + deferred)[:limit]
        results = []
        query_terms = set(tokens(query))
        with self.store.connect() as db:
            for item in selected:
                row = db.execute("SELECT c.id,c.file_id,c.start_line,c.end_line,c.content,f.path,f.project,f.extension "
                                 "FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.id=?", (item["id"],)).fetchone()
                if row:
                    record = {**dict(row), **item}
                    record["matched_terms"] = sorted(query_terms & set(tokens(row["content"] + " " + row["path"])))[:30]
                    if item["id"] in getattr(sparse, "features", {}):
                        record["bm25_features"] = sparse.features[item["id"]]
                        record["bm25_raw_score"] = sparse.features[item["id"]]["raw_bm25"]
                        record["raw_bm25_score"] = sparse.features[item["id"]]["raw_bm25"]
                        record["lexical_score"] = sparse.features[item["id"]]["lexical_score"]
                        record["duplicate_sources"] = sparse.duplicates.get(item["id"], [])
                    record["duplicate_sources"] = record.get("duplicate_sources", []) + duplicate_sources.get(item["id"], [])
                    results.append(record)
        lexical_strategy = getattr(sparse, "strategy", None)
        if lexical_strategy in {"identifier_anchor", "partial_query_groups"} and sparse:
            note = "未找到同时覆盖全部查询组的片段，关键词召回已放宽条件；请核对结果原文。"
            warning = (warning + " " + note) if warning else note
        if any(row.get("bm25_features", {}).get("topic_coverage", 1.) < 1. for row in results):
            note = "部分结果仅覆盖部分主题词；已保留标识符约束并公开词项覆盖和排序特征，请核对原文。"
            warning = (warning + " " + note) if warning else note
        result = dict(query=query, mode=mode, effective_mode=effective, warning=warning,
                      elapsed_ms=round((time.perf_counter()-started)*1000, 2), total_candidates=len(ranked), results=results,
                      routes={"vector":[{"id":ident,"rank":rank,"score":score} for rank,(ident,score) in enumerate(dense,1)],
                              "bm25":[{"id":ident,"rank":rank,"score":score,
                                        **({"raw_bm25_score":sparse.features[ident]["raw_bm25"],
                                            "lexical_score":score,"features":sparse.features[ident]}
                                           if ident in getattr(sparse,"features",{}) else {})}
                                       for rank,(ident,score) in enumerate(sparse,1)]},
                      explanation=dict(formula="vector_weight / (60 + vector_rank) + bm25_weight / (60 + bm25_rank)" if effective == "hybrid" else ("cosine(query, chunk)" if effective == "vector" else "lexical_score = -FTS5_bm25(path=4,body=1) * path_prior * coverage_boost * evidence_boost"),
                          rrf_k=RRF_K, vector_weight=vw, bm25_weight=bw, candidates=candidates,
                          absent_rank_contribution=0, score_is_probability=False, timings=timings,
                          tokenizer="identifiers + camelCase + 中文字与双字", scope=dict(project=project, extension=extension), coverage=coverage,
                          lexical_query=dict(strategy=lexical_strategy, expression=getattr(sparse, "expression", None),
                                             route="BM25 candidate retrieval + explainable lexical reranking",
                                             path_weight=4, body_weight=1, expanded_components="AND within one identifier",
                                             topic_rule="identifiers AND (topic bigrams OR disclosed software aliases)",
                                             software_aliases=SOFTWARE_TERMS, internal_pool_limit=getattr(sparse, "pool_limit", None),
                                             pooled_candidates=getattr(sparse, "pool_candidates", None), content_hash_deduplication=True),
                          result_selection="up_to_three_distinct_passages_per_file_then_additional_chunks",
                          raw_rank_is_before_file_diversity=True,
                          semantic_evaluation_eligible=bool(use_vector and coverage and coverage["chunks"] > 0 and coverage["vectors"] == coverage["chunks"] and effective != "bm25")),
                      trace_id=uuid.uuid4().hex)
        self.store.add_trace(result, dict(limit=limit, candidates=candidates, vector_weight=vector_weight, project=project, extension=extension))
        return result
