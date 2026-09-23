"""Deprecated: the original OpenAI-based text-to-speech implementation.

Production narration (ainews.narrate) now uses ElevenLabs instead - see
ElevenLabsTextToSpeech and create_tts() in ainews/llm.py.

This module is kept only for reference (or as a fallback if ElevenLabs ever needs to
be dropped); nothing in ainews/ imports it. It reuses ainews.llm's generic pieces
(LLMError, LLMConfigError, the OPENAI_API_KEY lookup, and the OpenAI error-describing
helper) since those still back OpenAIStructuredLLM there and aren't specific to TTS.

Uses OpenAI's Audio API (`audio.speech`), model "gpt-4o-mini-tts", voice "alloy".
"""

import os

import openai
from dotenv import dotenv_values

from ainews.llm import (
    API_KEY_VARIABLE,
    ENV_PATH,
    REQUEST_TIMEOUT_SECONDS,
    LLMConfigError,
    LLMError,
    _describe_api_error,
)

DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "alloy"


class OpenAITextToSpeech:
    def __init__(
        self,
        model: str = DEFAULT_TTS_MODEL,
        voice: str = DEFAULT_TTS_VOICE,
        *,
        api_key: str | None = None,
        client: openai.OpenAI | None = None,
    ) -> None:
        """`client` lets tests supply an SDK client wired to a fake transport."""
        self.model = model
        self.voice = voice
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

    def synthesize(self, text: str) -> bytes:
        try:
            response = self._client.audio.speech.create(
                model=self.model, voice=self.voice, input=text, response_format="mp3"
            )
        except openai.APIError as e:  # connection, timeout and HTTP status errors alike
            raise LLMError(self._redact(_describe_api_error(e))) from e
        return response.content

    def _redact(self, message: str) -> str:
        """Never let the API key leak into an error message we print."""
        return message.replace(self._secret, "***") if self._secret else message


def create_tts(model: str = DEFAULT_TTS_MODEL, voice: str = DEFAULT_TTS_VOICE) -> OpenAITextToSpeech:
    """Build this deprecated provider. Same key lookup as ainews.llm.create_tts()."""
    api_key = os.environ.get(API_KEY_VARIABLE) or dotenv_values(ENV_PATH).get(API_KEY_VARIABLE)
    return OpenAITextToSpeech(model, voice, api_key=api_key)
