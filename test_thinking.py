"""Does the server honour 'no thinking' requests, and are <think> blocks parsed correctly?"""
import pytest

from conftest import NO_THINK_VARIANTS, reasoning_of


def test_default_thinks_and_splits_reasoning(llm):
    """Baseline: thinking is on by default, reasoning lands in its own field, answer in content."""
    m = llm.message()
    assert reasoning_of(m), f"expected reasoning with defaults: {m}"
    assert "391" in (m.get("content") or ""), f"answer missing from content: {m}"


@pytest.mark.parametrize("extra", NO_THINK_VARIANTS.values(), ids=NO_THINK_VARIANTS.keys())
def test_no_think_has_no_reasoning(llm, extra):
    m = llm.message(**extra)
    assert not reasoning_of(m), f"reasoning still present: {reasoning_of(m)[:200]!r}"


@pytest.mark.parametrize("extra", NO_THINK_VARIANTS.values(), ids=NO_THINK_VARIANTS.keys())
def test_no_think_answer_is_in_content(llm, extra):
    """Catches the parser bug where the whole answer ends up in reasoning_content and content is empty."""
    m = llm.message(**extra)
    assert "391" in (m.get("content") or ""), f"answer not in content: {m}"


@pytest.mark.parametrize("extra", NO_THINK_VARIANTS.values(), ids=NO_THINK_VARIANTS.keys())
def test_no_think_content_has_no_think_tags(llm, extra):
    content = llm.message(**extra).get("content") or ""
    assert "<think>" not in content and "</think>" not in content


@pytest.mark.parametrize("extra", NO_THINK_VARIANTS.values(), ids=NO_THINK_VARIANTS.keys())
def test_no_think_streaming(llm, extra):
    """Open WebUI streams; check no reasoning deltas and some content deltas."""
    reasoning, content = "", ""
    for d in llm.stream("What is 17*23? Answer with just the number.", **extra):
        reasoning += d.get("reasoning_content") or d.get("reasoning") or ""
        content += d.get("content") or ""
    assert not reasoning, f"streamed reasoning: {reasoning[:200]!r}"
    assert "391" in content, f"content was {content!r}"


def test_no_think_uses_few_tokens(llm):
    r = llm.chat("Say hi.", reasoning_effort="none", chat_template_kwargs={"enable_thinking": False})
    assert r["usage"]["completion_tokens"] < 50, r["usage"]


def test_thinking_streaming_separates_fields(llm):
    reasoning, content = "", ""
    for d in llm.stream("What is 17*23? Answer with just the number."):
        reasoning += d.get("reasoning_content") or d.get("reasoning") or ""
        content += d.get("content") or ""
    assert reasoning and "391" in content
    assert "</think>" not in content
