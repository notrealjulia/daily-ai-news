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
voice's own account-configured settings. The previous OpenAI-based TTS implementation
is kept for reference in Deprecated/openai_tts.py; nothing here imports it.
"""

import json
import os
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
