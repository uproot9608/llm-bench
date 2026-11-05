"""Sampling and request parameters: temperature, seed, top_p, n, penalties, logprobs, stop."""
import json

OFF = {"chat_template_kwargs": {"enable_thinking": False}}
STORY = "Write one sentence about the sea."


def text(llm, prompt=STORY, **kw):
    return llm.message(prompt, **{**OFF, **kw})["content"]


def test_high_temperature_varies(llm):
    outs = {text(llm, "Name a random animal, one word.", temperature=1.5, max_tokens=8) for _ in range(8)}
    assert len(outs) > 1, outs


def test_different_seeds_differ(llm):
    outs = {text(llm, temperature=1.2, seed=s, max_tokens=60) for s in range(1, 5)}
    assert len(outs) > 1


def test_top_p_and_top_k_accepted(llm):
    assert text(llm, top_p=0.5, max_tokens=30)
    assert text(llm, top_k=5, max_tokens=30)


def test_top_k_one_is_greedy(llm):
    outs = {text(llm, "What is 2+2? Just the number.", temperature=1.5, top_k=1, max_tokens=10) for _ in range(5)}
    assert len(outs) == 1


def test_min_p_accepted(llm):
    assert text(llm, min_p=0.1, max_tokens=30)


def test_n_choices(llm):
    r = llm.chat(STORY, n=3, temperature=1.0, max_tokens=30, **OFF)
    assert [c["index"] for c in r["choices"]] == [0, 1, 2]
    assert all(c["message"]["content"] for c in r["choices"])


def test_max_completion_tokens(llm):
    r = llm.chat("Count from 1 to 500.", max_tokens=None, max_completion_tokens=15, **OFF)
    assert r["usage"]["completion_tokens"] <= 15


def test_multiple_stop_sequences(llm):
    out = text(llm, "Count: 1 2 3 4 5 6 7 8 9", stop=["4", "7"], temperature=0)
    assert "4" not in out and "7" not in out


def test_frequency_penalty_reduces_repetition(llm):
    p = "Repeat the word 'echo' 40 times separated by spaces."
    base = text(llm, p, temperature=0, max_tokens=120).lower().count("echo")
    pen = text(llm, p, temperature=0, max_tokens=120, frequency_penalty=2.0).lower().count("echo")
    assert pen < base, (base, pen)


def test_presence_penalty_accepted(llm):
    assert text(llm, presence_penalty=1.0, max_tokens=30)


def test_logprobs(llm):
    r = llm.chat("Say hi.", logprobs=True, top_logprobs=3, max_tokens=5, **OFF)
    lp = r["choices"][0]["logprobs"]["content"]
    assert lp and len(lp[0]["top_logprobs"]) == 3
    assert lp[0]["logprob"] <= 0


def test_logit_bias(llm):
    # banning is model/tokenizer specific; just require the parameter is accepted
    assert text(llm, logit_bias={"1": -100}, max_tokens=20)


def test_unknown_params_ignored(llm):
    assert text(llm, some_future_param=True, max_tokens=20)


def test_response_format_json_schema(llm):
    schema = {"type": "object", "required": ["name", "age"], "additionalProperties": False,
              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    out = text(llm, "Make up a person.", max_tokens=100,
               response_format={"type": "json_schema",
                                "json_schema": {"name": "p", "schema": schema, "strict": True}})
    d = json.loads(out)
    assert set(d) == {"name", "age"} and isinstance(d["age"], int)


def test_regex_constrained(llm):
    out = text(llm, "Give a 3-digit number.", max_tokens=10, extra_body={"regex": r"\d{3}"})
    # extra_body is an OpenAI-SDK construct; raw API takes `regex` top-level in sglang
    out = text(llm, "Give a 3-digit number.", max_tokens=10, regex=r"[0-9]{3}")
    assert out.strip().isdigit() and len(out.strip()) == 3


def test_json_schema_with_thinking_on(llm):
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    m = llm.message("Return a=5.", max_tokens=2048,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "x", "schema": schema}})
    assert json.loads(m["content"])["a"] == 5
