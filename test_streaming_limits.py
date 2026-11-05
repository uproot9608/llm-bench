"""Streaming protocol, usage reporting, limits, and API robustness."""
import json
from concurrent.futures import ThreadPoolExecutor

import httpx

from conftest import MODEL, URL

OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def raw_stream(llm, **extra):
    body = {"model": MODEL, "stream": True, "max_tokens": 30,
            "messages": [{"role": "user", "content": "Say hi."}], **OFF, **extra}
    with llm.http.stream("POST", URL, json=body) as r:
        r.raise_for_status()
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    return lines


def test_stream_ends_with_done(llm):
    assert raw_stream(llm)[-1] == "data: [DONE]"


def test_stream_include_usage(llm):
    chunks = [json.loads(l[6:]) for l in raw_stream(llm, stream_options={"include_usage": True})[:-1]]
    usage = [c["usage"] for c in chunks if c.get("usage")]
    assert usage and usage[-1]["completion_tokens"] > 0


def test_stream_first_role_and_single_finish(llm):
    chunks = [json.loads(l[6:]) for l in raw_stream(llm)[:-1]]
    chunks = [c for c in chunks if c["choices"]]
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    finishes = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"][0].get("finish_reason")]
    assert len(finishes) == 1


def test_stream_matches_non_stream(llm):
    kw = dict(temperature=0, max_tokens=40, **OFF)
    streamed = "".join(d.get("content") or "" for d in llm.stream("Name three colors.", **kw))
    plain = llm.message("Name three colors.", **kw)["content"]
    assert streamed == plain


def test_usage_counts_consistent(llm):
    u = llm.chat("Say hi.", max_tokens=20, **OFF)["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_response_shape(llm):
    r = llm.chat("Say hi.", max_tokens=10, **OFF)
    assert r["model"] and r["object"] == "chat.completion" and r["id"]


def test_over_context_prompt_rejected_cleanly(llm):
    r = llm.http.post(URL, json={"model": MODEL, "max_tokens": 10,
                                 "messages": [{"role": "user", "content": "word " * 40000}]})
    assert r.status_code == 400, r.status_code


def test_max_tokens_beyond_context(llm):
    r = llm.http.post(URL, json={"model": MODEL, "max_tokens": 10_000_000, "temperature": 0,
                                 "messages": [{"role": "user", "content": "hi"}], **OFF})
    assert r.status_code in (200, 400), r.status_code
    assert r.status_code != 500


def test_malformed_requests_are_4xx(llm):
    for body in [{"model": MODEL},
                 {"model": MODEL, "messages": []},
                 {"model": MODEL, "messages": [{"role": "nope", "content": "x"}]},
                 {"model": MODEL, "messages": "hi"}]:
        r = llm.http.post(URL, json=body)
        assert 400 <= r.status_code < 500, (body, r.status_code, r.text[:200])


def test_content_as_parts(llm):
    m = llm.message(messages=[{"role": "user", "content": [{"type": "text", "text": "Reply with exactly the word: pong"}]}], **OFF)
    assert "pong" in m["content"].lower()


def test_message_name_field(llm):
    m = llm.message(messages=[{"role": "user", "name": "daniel", "content": "Reply with exactly the word: pong"}], **OFF)
    assert m["content"]


def test_system_message_midconversation(llm):
    r = llm.http.post(URL, json={"model": MODEL, "max_tokens": 20, "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi again"}], **OFF})
    assert r.status_code < 500


def test_client_disconnect_frees_slot(llm):
    body = {"model": MODEL, "stream": True, "max_tokens": 2000, **OFF,
            "messages": [{"role": "user", "content": "Write a very long story."}]}
    with httpx.stream("POST", URL, json=body, timeout=30) as r:
        next(r.iter_lines())  # read one line, then drop the connection
    assert llm.message("Reply: ok", max_tokens=10, **OFF)["content"]


def test_parallel_requests_all_correct(llm):
    def ask(i):
        return i, llm.message(f"What is {i}+{i}? Just the number.", temperature=0, max_tokens=20, **OFF)["content"]

    with ThreadPoolExecutor(16) as ex:
        for i, out in ex.map(ask, range(2, 18)):
            assert str(i + i) in out, (i, out)


def test_queueing_beyond_max_running(llm):
    def ask(i):
        return llm.message(f"What is {i}+1? Just the number.", temperature=0, max_tokens=20, **OFF)["content"]

    with ThreadPoolExecutor(24) as ex:
        outs = list(ex.map(ask, range(24)))
    assert all(str(i + 1) in o for i, o in zip(range(24), outs))


def test_prefix_cache_speeds_second_request(llm):
    import time
    prompt = "Filler sentence number one. " * 800 + "\nSay ok."
    t = []
    for _ in range(2):
        s = time.perf_counter()
        llm.chat(prompt, max_tokens=1, temperature=0, **OFF)
        t.append(time.perf_counter() - s)
    assert t[1] < t[0], t


def test_needle_at_depths(llm):
    filler = "The grass is green. " * 600
    for pos in (0.1, 0.5, 0.9):
        i = int(len(filler) * pos)
        doc = filler[:i] + " The vault code is 8264. " + filler[i:]
        out = llm.message(f"{doc}\nWhat is the vault code? Digits only.", max_tokens=20, **OFF)["content"]
        assert "8264" in out, pos


def test_health_endpoint(llm):
    base = URL.split("/v1/")[0]
    assert httpx.get(base + "/health", timeout=10).status_code == 200


def test_legacy_completions_endpoint(llm):
    r = httpx.post(URL.replace("/chat/completions", "/completions"),
                   json={"model": MODEL, "prompt": "1+1=", "max_tokens": 5, "temperature": 0}, timeout=60)
    assert r.status_code == 200 and r.json()["choices"][0]["text"] is not None
