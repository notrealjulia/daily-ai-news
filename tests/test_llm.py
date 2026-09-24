"""Tests for the provider adapters in ainews.llm.

These use the real OpenAI and ElevenLabs SDKs, wired to a fake HTTP transport. So the
request that is checked is exactly what the SDK would put on the wire, and responses
go through the SDK's real parsing. No network is involved and no real key is used.
"""

import json
import shutil
import subprocess

import openai
import pytest
from elevenlabs.client import ElevenLabs

from ainews import llm

try:  # openai 3.x is built on httpx2 (the same library under a new name); 2.x used httpx
    import httpx2 as httpx
except ImportError:
    import httpx

import httpx as elevenlabs_httpx  # the ElevenLabs SDK always uses real httpx, unlike openai above

KEY = "sk-test-key-for-the-fake-transport"
SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["category", "summary"],
    "additionalProperties": False,
}
ANSWER = {"category": "Research", "summary": "A paper introduces a method."}


def response_body(
    text=None, *, status="completed", refusal=None, incomplete_reason=None, error=None
) -> dict:
    """A Responses API reply. `text` becomes the assistant's output_text."""
    content = []
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    if text is not None:
        content.append({"type": "output_text", "text": text, "annotations": []})
    output = (
        [{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": content}]
        if content
        else []
    )
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "some-model",
        "output": output,
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
        "error": error,
    }


def make_llm(handler, **kwargs) -> llm.OpenAIStructuredLLM:
    """The adapter, using a real SDK client whose HTTP goes to `handler`."""
    client = openai.OpenAI(
        api_key=KEY,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,  # otherwise the SDK would pause and retry 429/5xx replies
    )
    return llm.OpenAIStructuredLLM("some-model", api_key=KEY, client=client, **kwargs)


def answering(payload: dict, *, status_code=200):
    return lambda request: httpx.Response(status_code, json=payload)


def generate(model: llm.OpenAIStructuredLLM) -> dict:
    return model.generate(
        instructions="be helpful", input_text="the article", schema_name="my_schema", schema=SCHEMA
    )


# --- the request -------------------------------------------------------------


def test_the_request_uses_the_responses_api_with_a_strict_json_schema():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["method"] = request.url.path, request.method
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=response_body(json.dumps(ANSWER)))

    result = generate(make_llm(handler))

    assert result == ANSWER
    assert (seen["method"], seen["path"]) == ("POST", "/v1/responses")
    assert seen["auth"] == f"Bearer {KEY}"
    body = seen["body"]
    assert body["model"] == "some-model"
    assert body["instructions"] == "be helpful" and body["input"] == "the article"
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "my_schema",
        "strict": True,
        "schema": SCHEMA,
    }
    assert body["reasoning"] == {"effort": llm.DEFAULT_REASONING_EFFORT}
    assert body["max_output_tokens"] == llm.DEFAULT_MAX_OUTPUT_TOKENS
    assert body["store"] is False
    assert "temperature" not in body  # reasoning models don't take one


# --- things that go wrong at the API -----------------------------------------


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (429, "RateLimitError (HTTP 429)"),
        (401, "AuthenticationError (HTTP 401)"),
    ],
)
def test_http_errors_become_llm_errors(status_code, expected):
    payload = {"error": {"message": "the server said no", "type": "x", "code": None}}
    model = make_llm(answering(payload, status_code=status_code))

    with pytest.raises(llm.LLMError, match="the server said no") as caught:
        generate(model)

    assert expected in str(caught.value)


def test_connection_failures_and_timeouts_become_llm_errors():
    def unreachable(request):
        raise httpx.ConnectError("no route to host")

    def slow(request):
        raise httpx.ReadTimeout("took too long")

    with pytest.raises(llm.LLMError, match="APIConnectionError"):
        generate(make_llm(unreachable))
    with pytest.raises(llm.LLMError, match="APITimeoutError"):
        generate(make_llm(slow))


def test_the_api_key_never_appears_in_an_error_message():
    payload = {"error": {"message": f"Incorrect API key provided: {KEY}", "type": "x", "code": None}}

    with pytest.raises(llm.LLMError) as caught:
        generate(make_llm(answering(payload, status_code=401)))

    assert KEY not in str(caught.value) and "***" in str(caught.value)


# --- responses that are not a usable answer ----------------------------------


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            response_body(text=None, status="incomplete", incomplete_reason="max_output_tokens"),
            r"cut off \(max_output_tokens\)",
        ),
        (
            response_body(status="failed", error={"code": "server_error", "message": "it broke"}),
            "the response failed: it broke",
        ),
        (response_body(refusal="I can't help with that"), "the model refused: I can't help with that"),
        (response_body(text=None), "contained no text"),
        (response_body("this is not json"), "not valid JSON"),
        (response_body(json.dumps(["a", "list"])), "expected a JSON object"),
    ],
    ids=["cut-off", "failed", "refusal", "no-text", "not-json", "not-an-object"],
)
def test_responses_that_are_not_a_usable_answer_are_llm_errors(body, message):
    with pytest.raises(llm.LLMError, match=message):
        generate(make_llm(answering(body)))


# --- construction and the API key --------------------------------------------


def test_creating_without_any_key_explains_what_to_do(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(llm.LLMConfigError, match=r"OPENAI_API_KEY is not set.*\.env"):
        llm.create()


def test_the_key_can_come_from_the_env_file_without_entering_the_environment(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-the-env-file\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(llm, "ENV_PATH", env_file)

    model = llm.create("some-model")

    assert model._client.api_key == "sk-from-the-env-file"
    assert model.model == "some-model"
    import os

    assert "OPENAI_API_KEY" not in os.environ  # read directly, not loaded into the process


def test_the_environment_takes_precedence_over_the_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-the-env-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-environment")
    monkeypatch.setattr(llm, "ENV_PATH", env_file)

    assert llm.create()._client.api_key == "sk-from-the-environment"


# --- text to speech (ainews.narrate) ------------------------------------------


def test_split_into_chunks_only_cuts_at_sentence_boundaries():
    text = "One. Two. Three. Four. Five. Six. Seven. Eight."

    chunks = llm._split_into_chunks(text, 4)

    assert len(chunks) == 4
    for chunk in chunks:
        assert chunk.strip().endswith(".")  # never cuts a sentence in half
    # nothing invented, nothing dropped
    assert " ".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_split_into_chunks_balances_roughly_evenly():
    text = "One. Two. Three. Four. Five. Six. Seven. Eight."

    chunks = llm._split_into_chunks(text, 4)

    target = len(text) / 4
    assert all(abs(len(c) - target) < target for c in chunks)


def test_split_into_chunks_degrades_gracefully_with_fewer_sentences_than_chunks():
    text = "Only two sentences here. Here is the second one."

    chunks = llm._split_into_chunks(text, 4)

    assert len(chunks) == 2  # never invents empty chunks to hit the requested count


def test_split_into_chunks_falls_back_to_whole_text_with_no_sentence_boundaries():
    text = "no punctuation at all just words"

    assert llm._split_into_chunks(text, 4) == [text]


def make_tts(handler, **kwargs) -> llm.ElevenLabsTextToSpeech:
    client = ElevenLabs(
        api_key=KEY,
        httpx_client=elevenlabs_httpx.Client(transport=elevenlabs_httpx.MockTransport(handler)),
    )
    return llm.ElevenLabsTextToSpeech("some-tts-model", "some-voice-id", api_key=KEY, client=client, **kwargs)


def fake_ffmpeg_passthrough(monkeypatch):
    """Replace the ffmpeg-calling helpers with simple fakes that don't need ffmpeg, for
    tests that only care about synthesize()'s chunking/request wiring. The ffmpeg
    helpers themselves get their own dedicated (skippable) tests further down."""

    def fake_measure(path):
        return {"input_i": "-20", "input_tp": "-2", "input_lra": "5", "input_thresh": "-30", "target_offset": "1"}

    def fake_normalize(src, dst, stats):
        dst.write_bytes(src.read_bytes())  # identity: just needs to produce a file

    def fake_concat(paths, dst):
        dst.write_bytes(b"".join(p.read_bytes() for p in paths))

    monkeypatch.setattr(llm, "_measure_loudness", fake_measure)
    monkeypatch.setattr(llm, "_normalize_loudness", fake_normalize)
    monkeypatch.setattr(llm, "_concat_mp3s", fake_concat)


def test_the_tts_request_and_the_returned_audio_bytes(monkeypatch):
    fake_ffmpeg_passthrough(monkeypatch)
    seen = []

    def handler(request: elevenlabs_httpx.Request) -> elevenlabs_httpx.Response:
        seen.append(
            {
                "path": request.url.path,
                "method": request.method,
                "auth": request.headers["xi-api-key"],
                "params": dict(request.url.params),
                "body": json.loads(request.content),
            }
        )
        return elevenlabs_httpx.Response(200, content=f"audio-{len(seen)}".encode())

    text = "One thing happens. Then another thing happens. And a third thing. Finally a fourth thing."
    audio = make_tts(handler).synthesize(text)

    assert len(seen) == 4  # split into 4 sentence-boundary chunks, one ElevenLabs call each
    for call in seen:
        assert (call["method"], call["path"]) == ("POST", "/v1/text-to-speech/some-voice-id")
        assert call["auth"] == KEY
        assert call["params"]["output_format"] == llm.TTS_OUTPUT_FORMAT
        assert call["body"]["model_id"] == "some-tts-model"
        # No voice_settings override: uses the voice's own account-configured settings.
        assert call["body"]["voice_settings"] is None
    # every chunk's text is a piece of the original - nothing invented, nothing dropped
    rejoined = " ".join(call["body"]["text"] for call in seen)
    assert rejoined.replace(" ", "") == text.replace(" ", "")
    # the fake concat step joins each chunk's (fake, passed-through) audio in order
    assert audio == b"".join(f"audio-{i}".encode() for i in range(1, 5))


def test_tts_defaults_are_the_documented_model_and_voice():
    assert (llm.DEFAULT_TTS_MODEL, llm.DEFAULT_TTS_VOICE) == ("eleven_multilingual_v2", "1KW5b0DZhKA18MyNj4Kb")


def test_tts_http_errors_become_llm_errors():
    def handler(request: elevenlabs_httpx.Request) -> elevenlabs_httpx.Response:
        return elevenlabs_httpx.Response(429, json={"detail": {"message": "the server said no"}})

    with pytest.raises(llm.LLMError, match="429"):
        make_tts(handler).synthesize("text")


def test_tts_connection_failures_become_llm_errors():
    def unreachable(request):
        raise elevenlabs_httpx.ConnectError("no route to host")

    with pytest.raises(llm.LLMError, match="ConnectError"):
        make_tts(unreachable).synthesize("text")


def test_the_tts_api_key_never_appears_in_an_error_message():
    def handler(request: elevenlabs_httpx.Request) -> elevenlabs_httpx.Response:
        return elevenlabs_httpx.Response(401, json={"detail": f"Incorrect API key provided: {KEY}"})

    with pytest.raises(llm.LLMError) as caught:
        make_tts(handler).synthesize("text")

    assert KEY not in str(caught.value) and "***" in str(caught.value)


def test_creating_tts_without_any_key_explains_what_to_do(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY")

    with pytest.raises(llm.LLMConfigError, match=r"ELEVENLABS_API_KEY is not set.*\.env"):
        llm.create_tts()


# --- loudness normalization (real ffmpeg, skipped if it isn't installed) -----------

NO_FFMPEG = shutil.which("ffmpeg") is None


@pytest.fixture(scope="module")
def tone_mp3_path(tmp_path_factory):
    """A tiny, genuinely decodable MP3 with real (if quiet) signal - loudnorm needs
    actual measurable audio, not fake bytes, and rejects pure digital silence (-inf
    LUFS is out of its accepted range)."""
    path = tmp_path_factory.mktemp("ffmpeg-fixture") / "tone.mp3"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "1",
            "-af", "volume=0.1", "-c:a", "libmp3lame", "-b:a", "128k", str(path),
        ],  # fmt: skip
        check=True,
    )
    return path


@pytest.mark.skipif(NO_FFMPEG, reason="ffmpeg not installed")
def test_measure_loudness_returns_the_expected_stats(tone_mp3_path):
    stats = llm._measure_loudness(tone_mp3_path)

    assert {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"} <= stats.keys()


@pytest.mark.skipif(NO_FFMPEG, reason="ffmpeg not installed")
def test_normalize_loudness_produces_a_file(tmp_path, tone_mp3_path):
    stats = llm._measure_loudness(tone_mp3_path)
    out = tmp_path / "normalized.mp3"

    llm._normalize_loudness(tone_mp3_path, out, stats)

    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(NO_FFMPEG, reason="ffmpeg not installed")
def test_concat_mp3s_combines_files_in_order(tmp_path, tone_mp3_path):
    out = tmp_path / "combined.mp3"

    llm._concat_mp3s([tone_mp3_path, tone_mp3_path], out)

    assert out.exists()
    assert out.stat().st_size > tone_mp3_path.stat().st_size  # roughly double the one input


def test_only_llm_py_talks_to_the_provider_sdks():
    # The provider boundary: swapping providers must only ever mean editing llm.py.
    # Read the code as code (not text), so no way of writing an import can slip past.
    import ast
    from pathlib import Path

    def imported_modules(path):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                yield from (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield node.module

    package = Path(llm.__file__).parent
    offenders = [
        path.name
        for path in package.glob("*.py")
        if path.name != "llm.py"
        and any(module.split(".")[0] in ("openai", "elevenlabs") for module in imported_modules(path))
    ]
    assert offenders == []


def test_tests_can_never_see_the_real_key():
    # The autouse fixture in conftest.py replaces it; this guards the guard.
    assert llm.create()._client.api_key == "sk-test-not-a-real-key"
    assert not llm.ENV_PATH.exists()
