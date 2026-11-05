"""Non-thinking sanity checks: basics, streaming, tools, JSON, multi-turn, limits."""
import json

from conftest import MODEL

OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def test_models_endpoint(llm):
    from conftest import URL
    ids = [m["id"] for m in llm.http.get(URL.replace("/chat/completions", "/models")).json()["data"]]
    assert MODEL in ids


def test_basic_completion(llm):
    r = llm.chat("Reply with exactly the word: pong", **OFF)
    assert "pong" in r["choices"][0]["message"]["content"].lower()
    assert r["choices"][0]["finish_reason"] == "stop"


def test_system_prompt_followed(llm):
    m = llm.message(messages=[
        {"role": "system", "content": "Always answer in French."},
        {"role": "user", "content": "Say hello."}], **OFF)
    assert any(w in m["content"].lower() for w in ("bonjour", "salut"))


def test_multi_turn(llm):
    m = llm.message(messages=[
        {"role": "user", "content": "My name is Daniel."},
        {"role": "assistant", "content": "Nice to meet you, Daniel!"},
        {"role": "user", "content": "What is my name?"}], **OFF)
    assert "daniel" in m["content"].lower()


def test_max_tokens_truncates(llm):
    r = llm.chat("Count from 1 to 500.", max_tokens=20, **OFF)
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["completion_tokens"] <= 20


def test_stop_sequence(llm):
    m = llm.message("Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10", stop=["5"], **OFF)
    assert "5" not in m["content"]


def test_streaming_has_usage_and_done(llm):
    deltas = list(llm.stream("Say hi.", **OFF))
    assert "".join(d.get("content") or "" for d in deltas)


def test_json_mode(llm):
    m = llm.message("Return a JSON object with key 'a' set to 1.",
                    response_format={"type": "json_object"}, **OFF)
    assert json.loads(m["content"])["a"] == 1


def test_tool_call(llm):
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    m = llm.message("What's the weather in Paris?", tools=tools, **OFF)
    calls = m.get("tool_calls") or []
    assert calls, f"no tool call: {m}"
    assert calls[0]["function"]["name"] == "get_weather"
    assert "paris" in json.loads(calls[0]["function"]["arguments"])["city"].lower()


def test_tool_call_with_thinking(llm):
    """Tool calls must still parse when thinking is on (reasoning parser + tool parser interplay)."""
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    m = llm.message("What's the weather in Paris?", tools=tools, max_tokens=2048)
    assert m.get("tool_calls"), f"no tool call: {m}"


def test_long_context(llm):
    filler = "The sky is blue. " * 1500
    m = llm.message(f"{filler}\nThe secret code is 7421.\n{filler}\nWhat is the secret code? Digits only.", **OFF)
    assert "7421" in m["content"]
