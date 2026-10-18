"""Text/media boundary regressions; no model, torch runtime, GPU or network."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest

import omniembed.encoder as encoder_module
from omniembed.encoder import (
    Encoder, QUERY_PREFIX, TEXT_INPUT_POLICY, doc_text, literal_text, processor_input_reason,
)
from omniembed.health import InferenceError, InferenceInputError


PROCESSOR_MODULE = "transformers.models.embedding_gemma2.processing_embedding_gemma2"
MISSING_MEDIA = (
    ("image", "images", "processor_image_without_media"),
    ("audio", "audio inputs", "processor_audio_without_media"),
    ("video", "videos", "processor_video_without_media"),
)


def missing_message(kind, noun, count=1):
    return f"Found {count} <|{kind}|> tokens in the text but no {noun} were passed."


def processor_exception(message, *, module=PROCESSOR_MODULE, function="validate_inputs", exception_type=ValueError):
    namespace = {"__name__": module, "message": message, "error_type": exception_type}
    exec(compile(f"def {function}():\n    raise error_type(message)\n", "synthetic_processor.py", "exec"), namespace)
    try:
        namespace[function]()
    except Exception as error:
        return error
    raise AssertionError("synthetic processor did not raise")


@pytest.mark.parametrize("kind", ["image", "audio", "video"])
def test_exact_literals_get_visible_ascii_separation(kind):
    marker = f"<|{kind}|>"
    original = f"File: sample/{marker}.md\nsource {marker}{marker}\n"
    changed = literal_text(original)
    assert changed == f"File: sample/< |{kind}| >.md\nsource < |{kind}| >< |{kind}| >\n"
    assert literal_text(changed) == changed
    assert "\u200b" not in changed and marker not in changed
    assert original.count(marker) == 3
    assert TEXT_INPUT_POLICY == Encoder.text_input_policy == "media-literals-space-v1"


@pytest.mark.parametrize("text", [
    "", "ordinary source\n  exact whitespace \r\n", "中文查询和代码 user_id",
    "<image> <start_of_image> <start_of_turn> <end_of_turn>",
    "<|Image|> <|IMAGE|> < |image| > <|image> <image|>",
])
def test_ordinary_and_other_spellings_are_unchanged(text):
    assert literal_text(text).encode("utf-8") == text.encode("utf-8")


@pytest.fixture
def recording_encoder(monkeypatch):
    encoder = Encoder(model_id="mock/model", device="cpu", revision="fixed")
    calls, tokenized = [], []

    def tokenize(text):
        tokenized.append(text)
        return {"input_ids": list(range(len(text)))}

    encoder._m = SimpleNamespace(tokenizer=tokenize)

    def encode(items, bs, lock_timeout=None):
        calls.append((items, bs, lock_timeout))
        result = np.zeros((len(items), 768), dtype=np.float32)
        result[:, 0] = 1
        return result

    monkeypatch.setattr(encoder, "_enc", encode)
    return encoder, calls, tokenized


def test_query_transforms_after_task_prefix_and_preserves_callers(recording_encoder):
    encoder, calls, _ = recording_encoder
    texts = ["查找 <|image|> 和 <|audio|>", "plain\n"]
    original = list(texts)
    result = encoder.query(texts, bs=3)
    assert calls == [([QUERY_PREFIX + "查找 < |image| > 和 < |audio| >", QUERY_PREFIX + "plain\n"], 3, None)]
    assert texts == original and result.shape == (2, 768)


def test_docs_transform_title_body_and_title_only_fallback(recording_encoder):
    encoder, calls, _ = recording_encoder
    titles = [" title <|image|> ", "", "<|video|>"]
    texts = [" body <|audio|> ", "<|image|>", ""]
    original = (list(titles), list(texts))
    encoder.docs(titles, texts, bs=4)
    assert calls[0] == ([
        "title: title < |image| > | text: body < |audio| >",
        "title: none | text: < |image| >",
        "title: < |video| > | text: < |video| >",
    ], 4, None)
    assert (titles, texts) == original
    # The display/source formatting helper continues to represent raw text.
    assert doc_text("", "<|image|>") == "title: none | text: <|image|>"


def test_texts_transforms_prefix_before_token_budgeting(recording_encoder):
    encoder, calls, tokenized = recording_encoder
    texts = ["<|audio|>", "plain"]
    result = encoder.texts(texts, prefix="<|image|> ", bs=2, lock_timeout=.25)
    expected = ["< |image| > < |audio| >", "< |image| > plain"]
    assert tokenized == expected
    assert calls == [([expected[1], expected[0]], 2, .25)]
    assert texts == ["<|audio|>", "plain"] and result.shape == (2, 768)


def test_texts_handles_marker_spanning_prefix_boundary(recording_encoder):
    encoder, calls, tokenized = recording_encoder
    encoder.texts(["image|>"], prefix="<|", bs=1)
    assert tokenized == ["< |image| >"]
    assert calls[0][0] == ["< |image| >"]


def test_all_plain_text_routes_keep_their_original_inputs(recording_encoder):
    encoder, calls, tokenized = recording_encoder
    encoder.query(["ordinary\n"], bs=1)
    encoder.docs([" title "], [" body \n"], bs=1)
    encoder.texts(["ordinary\n"], prefix="prefix: ", bs=1)
    assert [call[0] for call in calls] == [
        ["task: search result | query: ordinary\n"],
        ["title: title | text: body"],
        ["prefix: ordinary\n"],
    ]
    assert tokenized == ["prefix: ordinary\n"]


def test_image_and_video_routes_pass_objects_through(recording_encoder):
    encoder, calls, _ = recording_encoder
    image = Image.new("RGB", (2, 2))
    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    encoder.images([image], bs=2)
    encoder.videos([frames], bs=1)
    assert calls[0][0][0] is image
    assert calls[0][1] == 2
    assert calls[1][0][0]["video"] is frames
    assert calls[1][1] == 1


@pytest.mark.parametrize("kind,noun,reason", MISSING_MEDIA)
@pytest.mark.parametrize("count", [1, 23])
def test_only_known_processor_missing_media_messages_are_classified(kind, noun, reason, count):
    error = processor_exception(missing_message(kind, noun, count))
    assert processor_input_reason(error) == reason


@pytest.mark.parametrize("variation", [
    "outside_processor", "other_function", "no_traceback", "subclass", "suffix",
    "prefix", "newline", "zero", "unknown_message", "non_string_arg",
])
def test_similar_and_arbitrary_value_errors_are_not_input_errors(variation):
    message = missing_message("image", "images")
    if variation == "outside_processor":
        error = processor_exception(message, module="unrelated.module")
    elif variation == "other_function":
        error = processor_exception(message, function="model_forward")
    elif variation == "no_traceback":
        error = ValueError(message)
    elif variation == "subclass":
        error = processor_exception(message, exception_type=type("CustomValueError", (ValueError,), {}))
    elif variation == "suffix":
        error = processor_exception(message + " synthetic-private-detail")
    elif variation == "prefix":
        error = processor_exception("synthetic-private-detail " + message)
    elif variation == "newline":
        error = processor_exception(message + "\n")
    elif variation == "zero":
        error = processor_exception(missing_message("image", "images", 0))
    elif variation == "unknown_message":
        error = processor_exception("unexpected model shape; synthetic-private-detail")
    else:
        error = processor_exception({"synthetic-private-detail": "not a processor string"})
    assert processor_input_reason(error) is None


@pytest.fixture
def runtime_stub(monkeypatch):
    class FakeOutOfMemoryError(RuntimeError):
        pass

    cache_clears = []
    torch = SimpleNamespace(OutOfMemoryError=FakeOutOfMemoryError, inference_mode=nullcontext,
                            cuda=SimpleNamespace(empty_cache=lambda: cache_clears.append(1)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(encoder_module, "HEALTH", SimpleNamespace(inference=lambda timeout: nullcontext()))
    return torch, cache_clears


def failing_encoder(monkeypatch, error):
    encoder = Encoder(model_id="mock/model", device="cpu")

    def encode(*args, **kwargs):
        raise error

    monkeypatch.setattr(encoder, "_model", lambda: SimpleNamespace(encode=encode))
    return encoder


@pytest.mark.parametrize("kind,noun,reason", MISSING_MEDIA)
def test_encoding_raises_fixed_safe_input_error(runtime_stub, monkeypatch, kind, noun, reason):
    original = processor_exception(missing_message(kind, noun))
    encoder = failing_encoder(monkeypatch, original)
    with pytest.raises(InferenceInputError) as caught:
        encoder.query(["ordinary input"])
    assert caught.value.category == "InferenceInputError"
    assert caught.value.reason_code == reason
    assert str(caught.value) == reason
    assert str(original) not in str(caught.value)
    assert caught.value.__suppress_context__
    assert runtime_stub[1] == []


def test_oom_keeps_existing_failure_protection(runtime_stub, monkeypatch):
    torch, clears = runtime_stub
    encoder = failing_encoder(monkeypatch, torch.OutOfMemoryError("synthetic-private-detail"))
    with pytest.raises(InferenceError) as caught:
        encoder.query(["ordinary input"])
    assert not isinstance(caught.value, InferenceInputError)
    assert caught.value.category == "InferenceOOM" and clears == [1]


def test_unknown_model_errors_keep_existing_failure_protection(runtime_stub, monkeypatch):
    original = processor_exception("unknown processor failure; synthetic-private-detail")
    encoder = failing_encoder(monkeypatch, original)
    with pytest.raises(InferenceError) as caught:
        encoder.query(["ordinary input"])
    assert not isinstance(caught.value, InferenceInputError)
    assert caught.value.category == "InferenceError"
    assert caught.value.__cause__ is original and runtime_stub[1] == []
