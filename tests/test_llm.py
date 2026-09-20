"""Tests for the OpenAI adapter in ainews.llm.

These use the real OpenAI SDK, wired to a fake HTTP transport. So the request that is
checked is exactly what the SDK would put on the wire, and responses go through the
SDK's real parsing. No network is involved and no real key is used.
"""

import json

import openai
import pytest

from ainews import llm

try:  # openai 3.x is built on httpx2 (the same library under a new name); 2.x used httpx
    import httpx2 as httpx
except ImportError:
    import httpx

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


def test_only_llm_py_talks_to_the_openai_sdk():
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
        and any(module.split(".")[0] == "openai" for module in imported_modules(path))
    ]
    assert offenders == []


def test_tests_can_never_see_the_real_key():
    # The autouse fixture in conftest.py replaces it; this guards the guard.
    assert llm.create()._client.api_key == "sk-test-not-a-real-key"
    assert not llm.ENV_PATH.exists()
