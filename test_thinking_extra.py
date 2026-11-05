"""More thinking on/off scenarios: stability, truncation, multi-turn, mixed load."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import reasoning_of

OFF = {"chat_template_kwargs": {"enable_thinking": False}}
ON = {"chat_template_kwargs": {"enable_thinking": True}}
Q = "What is 17*23? Answer with just the number."


def test_explicit_enable_thinking_true(llm):
    m = llm.message(**ON)
    assert reasoning_of(m) and "391" in (m.get("content") or "")


def test_reasoning_effort_values_accepted(llm):
    """Other effort values must not error (Open WebUI offers low/medium/high)."""
    for effort in ("low", "medium", "high"):
        m = llm.message(reasoning_effort=effort)
        assert "391" in (m.get("content") or "") or reasoning_of(m), effort


def test_no_think_stable_over_runs(llm):
    """Parser bugs can be intermittent: repeat the off-request."""
    for i in range(10):
        m = llm.message(**OFF)
        assert not reasoning_of(m), f"run {i}: {reasoning_of(m)[:100]!r}"
        assert "391" in (m.get("content") or ""), f"run {i}: {m}"


def test_think_truncated_by_max_tokens(llm):
    """Budget ends mid-thought: must not crash, no answer expected, no think tags in content."""
    r = llm.chat("Prove that sqrt(2) is irrational, in detail.", max_tokens=30, **ON)
    ch = r["choices"][0]
    assert ch["finish_reason"] == "length"
    assert "<think>" not in (ch["message"].get("content") or "")


def test_multi_turn_with_reasoning_in_history(llm):
    """Sending back reasoning_content in the assistant turn must be accepted."""
    first = llm.message(**ON)
    m = llm.message(messages=[
        {"role": "user", "content": Q},
        {"role": "assistant", "content": first["content"],
         "reasoning_content": reasoning_of(first)},
        {"role": "user", "content": "Now multiply that by 2. Just the number."}], **OFF)
    assert "782" in (m.get("content") or "")


def test_toggle_between_requests(llm):
    """No state leaks between on/off requests."""
    for extra, expect_reasoning in [(ON, True), (OFF, False), (ON, True), (OFF, False)]:
        m = llm.message(**extra)
        assert bool(reasoning_of(m)) == expect_reasoning, (extra, m)


def test_mixed_think_modes_concurrently(llm):
    def run(i):
        extra = ON if i % 2 == 0 else OFF
        return i, llm.message(**extra)

    with ThreadPoolExecutor(8) as ex:
        for i, m in ex.map(run, range(8)):
            assert bool(reasoning_of(m)) == (i % 2 == 0), (i, m)
            assert "391" in (m.get("content") or ""), (i, m)


def test_soft_switch_no_think(llm):
    m = llm.message(Q + " /no_think")
    # informational for Qwen3.6: template may ignore soft switches; only require a valid answer
    assert "391" in (m.get("content") or "") or "391" in reasoning_of(m)


def test_off_with_system_prompt(llm):
    m = llm.message(messages=[
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": Q}], **OFF)
    assert not reasoning_of(m) and "391" in m["content"]


def test_think_on_tokens_exceed_off(llm):
    on = llm.chat(**ON)["usage"]["completion_tokens"]
    off = llm.chat(**OFF)["usage"]["completion_tokens"]
    assert on > off * 3, (on, off)


def test_stream_reasoning_precedes_content(llm):
    seen_content = False
    for d in llm.stream(Q, **ON):
        if d.get("content"):
            seen_content = True
        elif d.get("reasoning_content") or d.get("reasoning"):
            assert not seen_content, "reasoning delta arrived after content started"
