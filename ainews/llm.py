"""The only module that knows which provider backs each capability: OpenAI for
structured generation, ElevenLabs for text-to-speech.

Everything else talks to one of two small interfaces below. `StructuredLLM`: give it
instructions, an input and a JSON schema, get back a dict that matches the schema.
`TextToSpeech`: give it text, get back MP3 bytes. Swapping providers means writing
another class with the same methods here.

The OpenAI implementation uses the Responses API with Structured Outputs
(`text.format` of type `json_schema`, `strict: true`), which OpenAI's current
documentation recommends for structured output. The ElevenLabs implementation uses
its text-to-speech `convert` endpoint with no `voice_settings` override, i.e. the
voice's own account-configured settings - split into sentence-boundary chunks,
loudness-normalized with ffmpeg and concatenated, since ElevenLabs' volume drifts down
over a long narration (confirmed with a local A/B listening test). The previous
OpenAI-based TTS implementation is kept for reference in Deprecated/openai_tts.py;
nothing here imports it.
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

import httpx
import openai
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError as ElevenLabsApiError

from ainews.defaults import DEFAULT_MODEL

# Reasoning models spend hidden "reasoning tokens" (billed as output) before answering.
# Classifying and summarizing one article needs little of that, and the API default
# (medium) would be wasted cost.
DEFAULT_REASONING_EFFORT = "low"
# The cap covers reasoning tokens as well as the answer, so leave generous headroom.
# You are only billed for tokens actually used.
DEFAULT_MAX_OUTPUT_TOKENS = 8000
REQUEST_TIMEOUT_SECONDS = 90

# Text-to-speech defaults (ainews.narrate is the only current caller). This voice/model
# pairing, with no voice_settings override (the voice's own account-configured
# settings), was chosen after comparing voice-setting variants and approved for
# production use.
DEFAULT_TTS_MODEL = "eleven_multilingual_v2"
DEFAULT_TTS_VOICE = "1KW5b0DZhKA18MyNj4Kb"
TTS_OUTPUT_FORMAT = "mp3_44100_128"

# ElevenLabs' output volume drifts down over a long narration (confirmed with a local
# A/B listening test comparing a chunked+normalized take against a single long call).
# Splitting into sentence-boundary chunks, synthesizing each separately, and loudness-
# normalizing each to the same target before concatenating keeps the volume consistent.
# Requires ffmpeg on PATH (see README's "Running" prerequisites).
NARRATION_CHUNKS = 4
LOUDNORM_TARGET_I = -16  # integrated loudness, LUFS
LOUDNORM_TARGET_TP = -1.5  # true peak, dBTP
LOUDNORM_TARGET_LRA = 11  # loudness range, LU

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
API_KEY_VARIABLE = "OPENAI_API_KEY"
ELEVENLABS_API_KEY_VARIABLE = "ELEVENLABS_API_KEY"


class LLMError(Exception):
    """One request failed (network, API error, refusal, truncated or unusable output).

    Failures of this kind are expected now and then; the article is simply tried again
    on a later run.
    """


class LLMConfigError(Exception):
    """The LLM cannot be used at all as configured (for example, no API key)."""


class StructuredLLM(Protocol):
    """What the enrichment stage needs from any LLM provider."""

    model: str  # recorded with every result

    def generate(
        self, *, instructions: str, input_text: str, schema_name: str, schema: dict
    ) -> dict:
        """Return a dict matching `schema`, or raise LLMError."""
        ...


class TextToSpeech(Protocol):
    """What the narrate stage needs from any text-to-speech provider."""

    def synthesize(self, text: str) -> bytes:
        """Return audio bytes (MP3) for `text`, or raise LLMError."""
        ...


class OpenAIStructuredLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        client: openai.OpenAI | None = None,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        """`client` lets tests supply an SDK client wired to a fake transport."""
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self._secret = api_key
        if client is None:
            api_key = api_key or os.environ.get(API_KEY_VARIABLE)
            if not api_key:
                raise LLMConfigError(
                    f"{API_KEY_VARIABLE} is not set. Put it in the .env file in the project "
                    "folder (see .env.example) or set it in the environment."
                )
            self._secret = api_key
            client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        self._client = client

    def generate(
        self, *, instructions: str, input_text: str, schema_name: str, schema: dict
    ) -> dict:
        request: dict = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            "max_output_tokens": self.max_output_tokens,
            "store": False,  # don't keep these requests on OpenAI's side
        }
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}

        try:
            response = self._client.responses.create(**request)
        except openai.APIError as e:  # connection, timeout and HTTP status errors alike
            raise LLMError(self._redact(_describe_api_error(e))) from e
        return self._structured_result(response)

    def _structured_result(self, response) -> dict:
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else "unknown"
            raise LLMError(f"the response was cut off ({reason})")
        if response.status == "failed":
            detail = response.error.message if response.error else "no detail given"
            raise LLMError(self._redact(f"the response failed: {detail}"))
        if response.status not in (None, "completed"):
            raise LLMError(f"unexpected response status {response.status!r}")

        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "refusal":
                        raise LLMError(f"the model refused: {part.refusal}")

        text = response.output_text
        if not text:
            raise LLMError("the response contained no text")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"the response was not valid JSON ({e})") from e
        if not isinstance(data, dict):
            raise LLMError(f"expected a JSON object, got {type(data).__name__}")
        return data

    def _redact(self, message: str) -> str:
        """Never let the API key leak into an error message we print."""
        return message.replace(self._secret, "***") if self._secret else message


class ElevenLabsTextToSpeech:
    def __init__(
        self,
        model: str = DEFAULT_TTS_MODEL,
        voice: str = DEFAULT_TTS_VOICE,
        *,
        api_key: str | None = None,
        client: ElevenLabs | None = None,
    ) -> None:
        """`client` lets tests supply an SDK client wired to a fake transport."""
        self.model = model
        self.voice = voice
        self._secret = api_key
        if client is None:
            api_key = api_key or os.environ.get(ELEVENLABS_API_KEY_VARIABLE)
            if not api_key:
                raise LLMConfigError(
                    f"{ELEVENLABS_API_KEY_VARIABLE} is not set. Put it in the .env file in the "
                    "project folder (see .env.example) or set it in the environment."
                )
            self._secret = api_key
            client = ElevenLabs(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        self._client = client

    def synthesize(self, text: str) -> bytes:
        """Split `text` into sentence-boundary chunks, synthesize each with ElevenLabs,
        loudness-normalize each to a common target, then concatenate - see
        NARRATION_CHUNKS above for why. Requires ffmpeg on PATH."""
        chunk_texts = _split_into_chunks(text, NARRATION_CHUNKS)
        with tempfile.TemporaryDirectory(prefix="ainews-tts-") as tmp:
            tmp_path = Path(tmp)
            normalized_paths = []
            for i, chunk_text in enumerate(chunk_texts, start=1):
                raw_path = tmp_path / f"chunk-{i}.mp3"
                raw_path.write_bytes(self._synthesize_chunk(chunk_text))
                stats = _measure_loudness(raw_path)
                normalized_path = tmp_path / f"chunk-{i}-normalized.mp3"
                _normalize_loudness(raw_path, normalized_path, stats)
                normalized_paths.append(normalized_path)
            combined_path = tmp_path / "combined.mp3"
            _concat_mp3s(normalized_paths, combined_path)
            return combined_path.read_bytes()

    def _synthesize_chunk(self, text: str) -> bytes:
        try:
            chunks = self._client.text_to_speech.convert(
                self.voice,
                text=text,
                model_id=self.model,
                output_format=TTS_OUTPUT_FORMAT,
                voice_settings=None,  # the voice's own account-configured settings
            )
            return b"".join(chunks)
        except (ElevenLabsApiError, httpx.HTTPError) as e:  # API errors and connection/timeout alike
            raise LLMError(self._redact(_describe_elevenlabs_error(e))) from e

    def _redact(self, message: str) -> str:
        """Never let the API key leak into an error message we print."""
        return message.replace(self._secret, "***") if self._secret else message


# Cuts only after ., ! or ? followed by whitespace. Good enough for narration scripts
# (plain prose, no abbreviations like "Dr." in practice) without a real sentence
# tokenizer.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_into_chunks(text: str, n: int) -> list[str]:
    """Split `text` into at most `n` chunks, only at sentence boundaries, each chunk's
    character count as close as possible to an even split. Never cuts a sentence in
    half. Falls back to fewer, non-empty chunks if there are fewer than `n` sentences,
    or to the whole text as one chunk if there's no sentence-ending punctuation at all."""
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
    if not sentences:
        return [text]
    n = min(n, len(sentences))

    total_len = sum(len(s) for s in sentences)
    target = total_len / n

    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for i, sentence in enumerate(sentences):
        current.append(sentence)
        current_len += len(sentence) + 1  # +1 for the joining space
        remaining_after = len(sentences) - (i + 1)
        chunks_left_to_fill = n - len(chunks) - 1
        reached_target = current_len >= target
        # If closing any later would leave too few sentences for the remaining
        # chunks, closing now isn't just preferred, it's the only option left.
        must_close_now = remaining_after <= chunks_left_to_fill
        if len(chunks) < n - 1 and (reached_target or must_close_now):
            chunks.append(current)
            current = []
            current_len = 0
    chunks.append(current)  # whatever's left becomes the last chunk

    return [" ".join(c) for c in chunks]


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    """Run ffmpeg, raising LLMConfigError if it isn't installed at all (a genuine
    stage-level problem) and never a raw subprocess/OS exception."""
    try:
        return subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-y", *args],
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as e:
        raise LLMConfigError(
            "ffmpeg is not installed or not on PATH. It's required to normalize "
            "narration audio (e.g. `apt-get install ffmpeg`, `brew install ffmpeg`)."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"ffmpeg timed out after {REQUEST_TIMEOUT_SECONDS}s") from e


def _measure_loudness(path: Path) -> dict:
    """Run ffmpeg's loudnorm filter in measurement-only mode and return its stats."""
    result = _run_ffmpeg([
        "-i", str(path),
        "-af", f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={LOUDNORM_TARGET_LRA}:print_format=json",
        "-f", "null", "-",
    ])  # fmt: skip
    if result.returncode != 0:
        raise LLMError(f"ffmpeg loudness measurement failed: {result.stderr[-500:]}")
    start, end = result.stderr.find("{"), result.stderr.rfind("}")
    if start == -1 or end == -1:
        raise LLMError("ffmpeg loudness measurement produced no readable stats")
    try:
        return json.loads(result.stderr[start : end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"ffmpeg loudness stats were not valid JSON ({e})") from e


def _normalize_loudness(src: Path, dst: Path, stats: dict) -> None:
    """Apply loudnorm using the stats `_measure_loudness` already measured for `src`
    (two-pass: far more accurate than loudnorm's single-pass dynamic mode)."""
    loudnorm = (
        f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={LOUDNORM_TARGET_LRA}:"
        f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
        f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:linear=true:print_format=summary"
    )
    result = _run_ffmpeg([
        "-i", str(src), "-af", loudnorm,
        "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(dst),
    ])  # fmt: skip
    if result.returncode != 0:
        raise LLMError(f"ffmpeg loudness normalization failed: {result.stderr[-500:]}")


def _concat_mp3s(paths: list[Path], dst: Path) -> None:
    """Concatenate already-encoded MP3s in order via ffmpeg's concat demuxer (stream
    copy, no re-encoding) - not raw byte concatenation, which can click at the seams
    between separately-encoded files."""
    list_file = dst.with_suffix(".txt")
    list_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in paths), encoding="utf-8")
    result = _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(dst)])
    if result.returncode != 0:
        raise LLMError(f"ffmpeg concatenation failed: {result.stderr[-500:]}")


def _describe_api_error(e: openai.APIError) -> str:
    if isinstance(e, openai.APIStatusError):
        return f"{type(e).__name__} (HTTP {e.status_code}): {e.message}"
    return f"{type(e).__name__}: {e.message}"


def _describe_elevenlabs_error(e: Exception) -> str:
    if isinstance(e, ElevenLabsApiError):
        return f"{type(e).__name__} (HTTP {e.status_code}): {e.body}"
    return f"{type(e).__name__}: {e}"


def create(model: str = DEFAULT_MODEL) -> OpenAIStructuredLLM:
    """Build the LLM the commands use.

    The key comes from the environment, or failing that from the .env file. The .env
    file is read directly rather than loaded into the environment, so nothing else in
    the process can see the key by accident.
    """
    api_key = os.environ.get(API_KEY_VARIABLE) or dotenv_values(ENV_PATH).get(API_KEY_VARIABLE)
    return OpenAIStructuredLLM(model, api_key=api_key)


def create_tts(model: str = DEFAULT_TTS_MODEL, voice: str = DEFAULT_TTS_VOICE) -> ElevenLabsTextToSpeech:
    """Build the text-to-speech provider the narrate command uses.

    The key comes from the environment, or failing that from the .env file, same as
    create() but under ELEVENLABS_API_KEY.
    """
    api_key = os.environ.get(ELEVENLABS_API_KEY_VARIABLE) or dotenv_values(ENV_PATH).get(ELEVENLABS_API_KEY_VARIABLE)
    return ElevenLabsTextToSpeech(model, voice, api_key=api_key)
