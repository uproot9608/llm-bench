"""Tool calling: selection, args, tool_choice, results, streaming, thinking on/off."""
import json

import pytest

OFF = {"chat_template_kwargs": {"enable_thinking": False}}
ON = {"chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 4096}
MODES = pytest.mark.parametrize("mode", [OFF, ON], ids=["think-off", "think-on"])


def fn(name, desc, props, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required or list(props)}}}


WEATHER = fn("get_weather", "Get current weather for a city", {"city": {"type": "string"}})
TIME = fn("get_time", "Get current time in a timezone", {"tz": {"type": "string"}})
CALC = fn("add", "Add two integers", {"a": {"type": "integer"}, "b": {"type": "integer"}})


def calls_of(m):
    assert m.get("tool_calls"), f"no tool call: {m}"
    return m["tool_calls"]


@MODES
def test_single_tool(llm, mode):
    c = calls_of(llm.message("Weather in Paris?", tools=[WEATHER], **mode))[0]
    assert c["function"]["name"] == "get_weather"
    assert "paris" in json.loads(c["function"]["arguments"])["city"].lower()


@MODES
def test_picks_correct_tool(llm, mode):
    c = calls_of(llm.message("What time is it in Tokyo?", tools=[WEATHER, TIME, CALC], **mode))[0]
    assert c["function"]["name"] == "get_time"


@MODES
def test_integer_args_typed(llm, mode):
    c = calls_of(llm.message("Add 17 and 25 using the tool.", tools=[CALC], **mode))[0]
    args = json.loads(c["function"]["arguments"])
    assert args == {"a": 17, "b": 25} or args == {"a": 25, "b": 17}
    assert all(isinstance(v, int) for v in args.values())


def test_parallel_calls(llm):
    cs = calls_of(llm.message("Get the weather in Paris and in Berlin.", tools=[WEATHER], **OFF))
    cities = {json.loads(c["function"]["arguments"])["city"].lower() for c in cs}
    assert {"paris", "berlin"} <= cities


def test_no_tool_needed(llm):
    m = llm.message("Say hello.", tools=[WEATHER, TIME, CALC], **OFF)
    assert not m.get("tool_calls") and m.get("content")


def test_tool_choice_none(llm):
    m = llm.message("Weather in Paris?", tools=[WEATHER], tool_choice="none", **OFF)
    assert not m.get("tool_calls")


def test_tool_choice_required(llm):
    m = llm.message("Hello.", tools=[WEATHER], tool_choice="required", **OFF)
    assert m.get("tool_calls")


def test_tool_choice_named(llm):
    m = llm.message("Hello.", tools=[WEATHER, TIME],
                    tool_choice={"type": "function", "function": {"name": "get_time"}}, **OFF)
    assert calls_of(m)[0]["function"]["name"] == "get_time"


@MODES
def test_tool_result_roundtrip(llm, mode):
    first = llm.message("Weather in Paris?", tools=[WEATHER], **mode)
    c = calls_of(first)[0]
    m = llm.message(messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [c]},
        {"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"temp_c": 23, "sky": "sunny"})},
    ], tools=[WEATHER], **mode)
    assert "23" in (m.get("content") or "")
    assert not m.get("tool_calls")


def test_special_chars_in_args(llm):
    note = fn("save_note", "Save a note", {"text": {"type": "string"}})
    c = calls_of(llm.message(
        'Save this note exactly: He said "hi" <b>& bye</b>\nline2 \\ ü 你好', tools=[note], **OFF))[0]
    text = json.loads(c["function"]["arguments"])["text"]
    assert '"hi"' in text and "&" in text and "你好" in text


def test_nested_and_array_args(llm):
    t = {"type": "function", "function": {"name": "create_order", "description": "Create an order",
         "parameters": {"type": "object", "required": ["items", "express"], "properties": {
             "express": {"type": "boolean"},
             "items": {"type": "array", "items": {"type": "object", "required": ["sku", "qty"],
                       "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}}}}}}}}
    c = calls_of(llm.message("Order 2 of sku A1 and 5 of sku B2, express shipping.", tools=[t], **OFF))[0]
    args = json.loads(c["function"]["arguments"])
    assert args["express"] is True
    assert {(i["sku"], i["qty"]) for i in args["items"]} == {("A1", 2), ("B2", 5)}


def test_enum_arg(llm):
    t = fn("set_mode", "Set mode", {"mode": {"type": "string", "enum": ["fast", "slow", "auto"]}})
    c = calls_of(llm.message("Set the mode to slow.", tools=[t], **OFF))[0]
    assert json.loads(c["function"]["arguments"])["mode"] == "slow"


@MODES
def test_streaming_tool_call(llm, mode):
    name, args = "", ""
    for d in llm.stream("Weather in Paris?", tools=[WEATHER], **mode):
        assert "<tool_call>" not in (d.get("content") or ""), "raw tool markup leaked into content"
        for tc in d.get("tool_calls") or []:
            name += tc["function"].get("name") or ""
            args += tc["function"].get("arguments") or ""
    assert name == "get_weather"
    assert "paris" in json.loads(args)["city"].lower()


def test_tool_content_has_no_markup(llm):
    m = llm.message("Weather in Paris?", tools=[WEATHER], **OFF)
    assert "<tool_call>" not in (m.get("content") or "")
