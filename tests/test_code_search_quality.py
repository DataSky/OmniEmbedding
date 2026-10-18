"""Retrieval quality regressions with independent sources and fake vectors.

No test reads the user's corpus, calls a GPU, or depends on model similarity.
"""
from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace

import pytest
import numpy as np

from omniembed.code_search.search import Search, lexical_plan
from omniembed.code_search.store import Store
from test_code_search import FakeCodeEncoder, add_file, axis


@pytest.fixture
def quality_search(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    store = Store(tmp_path / "quality.sqlite3", root=root)
    encoder = FakeCodeEncoder()
    return store, Search(store, SimpleNamespace(enc=encoder, enc_lock=threading.Lock())), encoder


@pytest.mark.parametrize("name,other", [("OmniVoice", "OmniSQL"), ("NeonAudio", "NeonServer")])
def test_complete_named_entity_and_topic_beats_shared_camel_component(quality_search, name, other):
    store, search, encoder = quality_search
    _, target = add_file(store, name + "/README.md", name + " 当前架构与实施边界，组件关系和入口。", project=name)
    add_file(store, other + "/architecture.md", "架构 架构 总体架构和技术栈 " * 30, project=other)
    add_file(store, "RiverVoice/design.md", "语音架构和架构图 " * 30, project="RiverVoice")
    result = search.run(name + " 架构", mode="bm25")
    assert [row["id"] for row in result["results"]] == [target]
    assert result["explanation"]["lexical_query"]["strategy"] == "anchor_topic_coverage"
    assert encoder.calls == []


def test_identifier_component_fallback_keeps_all_components_required(quality_search):
    store, search, _ = quality_search
    _, target = add_file(store, "audio/parser.py", "def parse_audio_frame(data): return data")
    add_file(store, "audio/render.py", "parse audio waveforms and render each image")
    result = search.run("parseAudioFrame", mode="bm25")
    assert [row["id"] for row in result["results"]] == [target]
    assert result["explanation"]["lexical_query"]["strategy"] == "identifier_components"


def test_chinese_bigram_does_not_match_unrelated_single_characters(quality_search):
    store, search, _ = quality_search
    _, target = add_file(store, "design.md", "当前系统架构与职责边界")
    add_file(store, "drums.md", "架子鼓演奏和节奏练习 " * 20)
    add_file(store, "build.md", "构建输出与构建缓存 " * 20)
    result = search.run("架构", mode="bm25")
    assert [row["id"] for row in result["results"]] == [target]


def test_missing_topic_never_falls_back_to_project_only_results(quality_search):
    store, search, _ = quality_search
    _, target = add_file(store, "NeonAudio/README.md", "NeonAudio handles speech input.", project="NeonAudio")
    add_file(store, "unrelated.md", "量子纠缠相关研究 " * 10)
    result = search.run("NeonAudio 量子纠缠", mode="bm25")
    assert result["results"] == []


def test_long_chinese_query_can_relax_to_bigrams_without_an_fts_error(quality_search):
    store, search, _ = quality_search
    _, target = add_file(store, "database.md", "SQLite 数据库连接与初始化")
    result = search.run("如何配置数据库", mode="bm25")
    assert result["results"][0]["id"] == target
    assert result["explanation"]["lexical_query"]["strategy"] == "anchor_topic_coverage"
    assert "部分主题词" in result["warning"]


def add_multi_chunk_file(store, count=2):
    content = "\n".join(f"architecture section {index}" for index in range(count)) + "\n"
    ident = "multi-file"
    record = dict(id=ident, path="docs/architecture.md", project="demo", extension=".md",
                  size=len(content), mtime_ns=1, sha256=hashlib.sha256(content.encode()).hexdigest(),
                  status="indexed", content=content)
    chunks = [dict(id=f"multi-{index}", ordinal=index, start_line=index+1, end_line=index+1,
                   content=line, embedding_text=line, sha256=hashlib.sha256(line.encode()).hexdigest())
              for index, line in enumerate(content.splitlines())]
    store.replace_file(record, chunks)
    return [row["id"] for row in chunks]


def test_three_informative_passages_survive_before_additional_file_diversity(quality_search, monkeypatch):
    store, search, _ = quality_search
    first, second, third, fourth = add_multi_chunk_file(store, 4)
    _, other = add_file(store, "another.md", "another architecture")
    monkeypatch.setattr(search, "bm25", lambda *args: [(first, 9.), (second, 8.), (third, 7.), (fourth, 6.), (other, 2.)])
    result = search.run("architecture", mode="bm25", limit=5, candidates=5)
    assert [row["id"] for row in result["results"]] == [first, second, third, other, fourth]
    assert [row["raw_rank"] for row in result["results"]] == [1, 2, 3, 5, 4]
    assert [row["bm25_score"] for row in result["results"]] == [9., 8., 7., 2., 6.]
    assert [row["id"] for row in result["routes"]["bm25"]] == [first, second, third, fourth, other]
    assert result["explanation"]["result_selection"] == "up_to_three_distinct_passages_per_file_then_additional_chunks"


def test_same_file_keeps_other_passages_if_no_alternative_files_exist(quality_search):
    store, search, _ = quality_search
    identifiers = add_multi_chunk_file(store)
    result = search.run("architecture", mode="bm25")
    assert {row["id"] for row in result["results"]} == set(identifiers)


def test_incomplete_hybrid_skips_query_encoding_until_scope_is_fully_vectorized(quality_search):
    store, search, encoder = quality_search
    add_file(store, "ready.md", "architecture", vector=axis())
    add_file(store, "pending.md", "architecture")
    partial = search.run("architecture", mode="hybrid")
    assert partial["effective_mode"] == "bm25"
    assert partial["routes"]["vector"] == []
    assert encoder.calls == []
    assert partial["explanation"]["semantic_evaluation_eligible"] is False
    assert "不完整" in partial["warning"]
    assert "自动仅使用 BM25" in partial["warning"]
    explicit = search.run("architecture", mode="vector")
    assert explicit["effective_mode"] == "vector"
    assert explicit["routes"]["vector"]
    assert len(encoder.calls) == 1
    assert explicit["explanation"]["semantic_evaluation_eligible"] is False
    vector_weight_one = search.run("architecture", mode="hybrid", vector_weight=1.)
    assert vector_weight_one["routes"]["vector"]
    assert vector_weight_one["effective_mode"] == "hybrid"
    for row in store.pending_chunks():
        store.vector_put(row["id"], axis())
    complete = search.run("architecture", mode="hybrid")
    assert complete["effective_mode"] == "hybrid"
    assert len(complete["routes"]["vector"]) == 2
    assert complete["explanation"]["semantic_evaluation_eligible"] is True


def test_query_planner_is_bounded_and_never_inserts_user_fts_operators():
    plan = lexical_plan('NeonAudio" NEAR(架构, *, -1) OR path:private\\key')
    assert plan
    assert all("*" not in expression and "path:" not in expression and "\\" not in expression
               for _, expression in plan)
    assert len(lexical_plan("架构" * 2000)[0][1]) < 400


def test_disclosed_software_terms_recall_english_architecture_without_project_specific_rules(quality_search):
    store, search, _ = quality_search
    _, target = add_file(store, "OceanIDE/docs/ARCHITECTURE.md", "Runtime architecture and module organization.", project="OceanIDE")
    add_file(store, "OceanIDE/compiler.py", "ordinary source implementation", project="OceanIDE")
    result = search.run("OceanIDE 运行时架构和模块组织", mode="bm25")
    assert result["results"][0]["id"] == target
    assert "architecture" in result["explanation"]["lexical_query"]["software_aliases"]["架构"]


def test_camel_case_source_aliases_preserve_body_path_and_heading_components(quality_search):
    store, search, _ = quality_search
    _, heading = add_file(store, "OceanIDE/overview.md", "# SoftwareArchitecture\n\nRuntime notes.", project="OceanIDE")
    _, path = add_file(store, "OceanIDE/SoftwareArchitecture.py", "ordinary implementation", project="OceanIDE")
    result = search.run("OceanIDE 架构", mode="bm25")
    by_id = {row["id"]: row for row in result["results"]}
    assert set(by_id) == {heading, path}
    assert by_id[heading]["bm25_features"]["heading_topic_hits"] == 1
    assert by_id[path]["bm25_features"]["topic_coverage"] == 1.


def test_content_duplicates_keep_canonical_project_and_expose_other_sources(quality_search):
    store, search, _ = quality_search
    content = "# 当前架构\nNeonAudio 运行时架构、模块组织和接口边界。\n" * 4
    _, target = add_file(store, "NeonAudio/docs/current/design.md", content, project="NeonAudio")
    add_file(store, "NeonAudio-copy/docs/current/design.md", content, project="NeonAudio-copy")
    result = search.run("NeonAudio 架构", mode="bm25")
    assert [row["id"] for row in result["results"]] == [target]
    assert result["results"][0]["duplicate_sources"][0]["path"].startswith("NeonAudio-copy/")


def test_raw_bm25_and_all_reranking_factors_reproduce_lexical_score(quality_search):
    store, search, _ = quality_search
    add_file(store, "NeonAudio/docs/current/architecture.md", "# 当前架构\n运行时架构与模块边界", project="NeonAudio")
    row = search.run("NeonAudio 架构", mode="bm25")["results"][0]
    features = row["bm25_features"]
    assert row["raw_bm25_score"] == features["raw_bm25"]
    assert row["lexical_score"] == pytest.approx(row["raw_bm25_score"] * features["path_prior"] * features["coverage_boost"] * features["evidence_boost"])
    assert row["bm25_score"] == row["lexical_score"]


@pytest.mark.parametrize("bad", ["nan", "not_unit", "wrong_bytes"])
def test_invalid_stored_vectors_are_distinct_from_remote_transport_failures(quality_search, bad):
    store, search, _ = quality_search
    _, ident = add_file(store, "source.py", "needle", vector=axis())
    value = axis(); value[0] = np.nan if bad == "nan" else 2.
    blob = b"short" if bad == "wrong_bytes" else value.astype("<f4").tobytes()
    with store.connect() as db:
        db.execute("UPDATE vectors SET vector=? WHERE chunk_id=?", (blob, ident))
    with pytest.raises(ValueError, match="本地向量"):
        search.run("needle", mode="vector")
    fallback = search.run("needle", mode="hybrid")
    assert fallback["effective_mode"] == "bm25"
    assert "本地向量" in fallback["warning"]


def test_unit_vector_scores_use_independent_finite_accumulation(quality_search):
    store, search, _ = quality_search
    add_file(store, "positive.py", "needle", vector=axis())
    add_file(store, "negative.py", "needle", vector=axis(sign=-1))
    with np.errstate(all="raise"):
        result = search.run("needle", mode="vector")
    assert [row["vector_score"] for row in result["results"]] == pytest.approx([1., -1.])
    assert all(np.isfinite(row["vector_score"]) for row in result["results"])


def test_vector_results_collapse_long_duplicate_passages_with_provenance(quality_search):
    store, search, _ = quality_search
    content = "The same architecture passage copied into a different branch.\n" * 3
    add_file(store, "original.md", content, vector=axis())
    add_file(store, "branch-copy.md", content, vector=axis())
    result = search.run("architecture", mode="vector")
    assert len(result["routes"]["vector"]) == 2
    assert len(result["results"]) == 1
    assert len(result["results"][0]["duplicate_sources"]) == 1
