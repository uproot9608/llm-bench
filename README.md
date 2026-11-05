# LLM Endpoint Tests

A pytest suite that checks an OpenAI-compatible chat completions server (built for sglang serving Qwen models). Its main purpose is to verify that **thinking / reasoning is controlled and parsed correctly**. It also covers general behaviour: streaming, tool calls, JSON mode and so on. A single `llm-bench` command runs the tests, a concurrency load benchmark, or both.

## Setup

Dependencies are managed with uv. `httpx` is a project dependency and `pytest` is a dev dependency.

```
uv sync
```

(They were added with `uv add httpx` and `uv add --dev pytest`.)

## Running

### One entry point: `llm-bench`

`uv sync` installs the `llm-bench` console command, which runs the tests, the benchmark, or both:

```
uv run llm-bench                  # tests, then the load benchmark
uv run llm-bench --tests          # tests only
uv run llm-bench --benchmark      # benchmark only
```

Anything after the flags is forwarded, so you can combine them with the native options:

```
uv run llm-bench --tests -k tool_call            # extra args go to pytest
uv run llm-bench --benchmark --levels 1 4 16     # extra args go to the benchmark
uv run llm-bench --tests --model Qwen/Qwen3-8B   # override endpoint/model
```

### Directly with pytest

```
uv run pytest                          # everything
uv run pytest test_thinking.py         # thinking tests only
uv run pytest -k tool_call -v          # filter by name
```

### Configuration

The target comes from `LLM_URL` and `LLM_MODEL`, read from the environment or from `.env` in the project root (see `.env.example`). Real environment variables win over `.env`.

```
LLM_URL=http://localhost:8000/v1/chat/completions LLM_MODEL=Qwen/Qwen3-8B uv run llm-bench
```

If `/v1/models` is unreachable, the whole run aborts with a clear message.

## Layout

| File | Purpose |
|---|---|
| `src/llm_bench/cli.py` | The `llm-bench` entry point: runs the tests, the benchmark, or both |
| `src/llm_bench/benchmark.py` | Concurrency load benchmark (TTFT, tok/s, parallelism) |
| `src/llm_bench/settings.py` | Loads `.env` and the environment (shared with the tests and benchmark) |
| `conftest.py` | The small HTTP `Client` (plain and streaming), the `llm` fixture, and the list of "no thinking" request variants |
| `test_thinking.py` | Thinking on/off behaviour and reasoning parsing |
| `test_general.py` | General endpoint sanity checks |
| `test_thinking_extra.py` | More thinking on/off scenarios: stability, truncation, history, mixed load |
| `test_tools.py` | Tool calling with thinking on and off |
| `test_sampling.py` | Sampling and request parameters |
| `test_streaming_limits.py` | Streaming protocol, limits, robustness, concurrency |

## What is tested

### Thinking (`test_thinking.py`)

There are three ways a client can ask for no thinking, and each is tested separately (`NO_THINK_VARIANTS` in `conftest.py`):

- `reasoning_effort=none`. This is what Open WebUI's "reasoning effort" setting sends.
- `chat_template_kwargs.enable_thinking=false`. This is what Qwen's chat template reads.
- Both together.

| Test | Checks |
|---|---|
| `test_default_thinks_and_splits_reasoning` | By default, reasoning lands in `reasoning_content` and the answer in `content` |
| `test_no_think_has_no_reasoning` | With thinking off, `reasoning_content` is empty |
| `test_no_think_answer_is_in_content` | With thinking off, the answer is in `content` and not misrouted into `reasoning_content` |
| `test_no_think_content_has_no_think_tags` | No raw `<think>` or `</think>` leaks into `content` |
| `test_no_think_streaming` | Same as above for streamed deltas, which is what Open WebUI uses |
| `test_no_think_uses_few_tokens` | Thinking really is skipped, so a short reply uses under 50 tokens |
| `test_thinking_streaming_separates_fields` | Streamed reasoning and content are split correctly when thinking is on |

### General (`test_general.py`)

Most of these run with thinking disabled so they isolate one feature.

| Test | Checks |
|---|---|
| `test_models_endpoint` | `/v1/models` lists the configured model |
| `test_basic_completion` | Basic reply and `finish_reason == "stop"` |
| `test_system_prompt_followed` | System prompt is respected |
| `test_multi_turn` | Conversation history is used |
| `test_max_tokens_truncates` | `max_tokens` is honoured, with `finish_reason == "length"` |
| `test_stop_sequence` | `stop` strings cut the output |
| `test_streaming_has_usage_and_done` | Streaming yields content |
| `test_json_mode` | `response_format: json_object` returns valid JSON |
| `test_tool_call` | Function call is parsed into `tool_calls` |
| `test_tool_call_with_thinking` | Same, with thinking on, which tests the reasoning parser and tool parser together |
| `test_long_context` | A fact buried in about 6k tokens of filler is recalled |

### More thinking scenarios (`test_thinking_extra.py`)

| Test | Checks |
|---|---|
| `test_explicit_enable_thinking_true` | `enable_thinking=true` thinks and splits fields |
| `test_reasoning_effort_values_accepted` | `low`, `medium` and `high` don't error |
| `test_no_think_stable_over_runs` | 10 repeated no-think requests, to catch intermittent parser bugs |
| `test_think_truncated_by_max_tokens` | Budget runs out mid-think: `finish_reason=length`, no `<think>` in content |
| `test_multi_turn_with_reasoning_in_history` | `reasoning_content` sent back in an assistant turn is accepted |
| `test_toggle_between_requests` | On/off/on/off with no state leaking between requests |
| `test_mixed_think_modes_concurrently` | Thinking and non-thinking requests in one batch stay correctly separated |
| `test_soft_switch_no_think` | `/no_think` in the prompt still returns a valid answer |
| `test_off_with_system_prompt` | Thinking off works with a system prompt |
| `test_think_on_tokens_exceed_off` | Thinking on uses clearly more tokens than off |
| `test_stream_reasoning_precedes_content` | In a stream, reasoning deltas never arrive after content starts |

### Tools (`test_tools.py`)

Most tests run with thinking both off and on (`think-off` and `think-on` ids).

| Test | Checks |
|---|---|
| `test_single_tool` / `test_picks_correct_tool` | Correct function and arguments, chosen from 1 or 3 tools |
| `test_integer_args_typed` | Integer args come back as ints |
| `test_parallel_calls` | Two calls in one reply |
| `test_no_tool_needed` | No tool call when none is needed |
| `test_tool_choice_none` / `_required` / `_named` | All three `tool_choice` modes |
| `test_tool_result_roundtrip` | A `tool` role result is used in the final answer |
| `test_special_chars_in_args` | Quotes, `<`, `&`, newlines, backslashes and unicode survive in arguments |
| `test_nested_and_array_args` | Arrays of objects and booleans |
| `test_enum_arg` | Enum values are respected |
| `test_streaming_tool_call` | Streamed argument deltas reassemble into valid JSON |
| `test_tool_content_has_no_markup` | No raw `<tool_call>` in `content` |

### Sampling and parameters (`test_sampling.py`)

| Test | Checks |
|---|---|
| `test_high_temperature_varies` | High temperature gives varied output |
| `test_different_seeds_differ` | Different seeds give different output |
| `test_top_p_and_top_k_accepted` / `test_top_k_one_is_greedy` / `test_min_p_accepted` | Truncation sampling parameters |
| `test_n_choices` | `n>1` returns several choices |
| `test_max_completion_tokens` | The newer token-limit field is honoured |
| `test_multiple_stop_sequences` | Several stop strings |
| `test_frequency_penalty_reduces_repetition` / `test_presence_penalty_accepted` | Penalties |
| `test_logprobs` / `test_logit_bias` | Logprob output and bias parameter |
| `test_unknown_params_ignored` | Unknown fields don't cause an error |
| `test_response_format_json_schema` / `test_json_schema_with_thinking_on` | Schema-constrained output |
| `test_regex_constrained` | sglang's `regex` parameter |

### Streaming and limits (`test_streaming_limits.py`)

| Test | Checks |
|---|---|
| `test_stream_ends_with_done` / `_include_usage` / `_first_role_and_single_finish` | Stream protocol |
| `test_stream_matches_non_stream` | Same text streamed and non-streamed at `temperature=0` |
| `test_usage_counts_consistent` / `test_response_shape` | Response metadata |
| `test_over_context_prompt_rejected_cleanly` | A prompt past the context length returns 400 |
| `test_max_tokens_beyond_context` / `test_malformed_requests_are_4xx` | Errors are never 500 |
| `test_content_as_parts` / `test_message_name_field` / `test_system_message_midconversation` | Client compatibility, which matters for Open WebUI |
| `test_client_disconnect_frees_slot` | The server keeps working after an aborted stream |
| `test_parallel_requests_all_correct` / `test_queueing_beyond_max_running` | 16 and 24 parallel requests all return correct answers |
| `test_prefix_cache_speeds_second_request` | Repeated long prefix is faster |
| `test_needle_at_depths` | Fact recall at 10%, 50% and 90% depth |
| `test_health_endpoint` / `test_legacy_completions_endpoint` | `/health` and `/v1/completions` |

## Interpreting failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `test_no_think_*` fails with `streamed reasoning: '391'` or an empty `content` | The reasoning parser assumes output always starts inside a think block, so with thinking off the whole answer lands in `reasoning_content` | Check the sglang `--reasoning-parser` setting and how your sglang version handles `enable_thinking=false` for this model's template |
| `test_tool_call*` fails and `content` holds `<tool_call><function=...>` text | No tool-call parser is configured | Launch sglang with `--tool-call-parser qwen3_coder` (Qwen3.6 uses the XML-style format) |
| `test_default_thinks_and_splits_reasoning` fails | The reasoning parser is missing, or the reasoning is inline in `content` | Launch with `--reasoning-parser qwen3` |
| `test_max_tokens_truncates` or `test_long_context` fails with an HTTP 400 | Context window is too small (this server reports `max_model_len` 16384) | Raise the server's context length, or shrink the test input |
| `test_prefix_cache_speeds_second_request` fails | Radix cache disabled, or the machine is noisy | Check that `--disable-radix-cache` isn't set; re-run |
| `test_needle_at_depths` or `test_over_context_prompt_rejected_cleanly` fails | Context length changed | Adjust the filler size in the test to fit `--context-length` |
| Everything fails at startup | Server unreachable | Check `LLM_URL` and that the server is up |

Tests that fail only because `content` is empty (JSON mode, multi-turn, system prompt, long context) usually share the thinking-parser root cause. Fix that first and re-run.

## Possible fixes for known failures

Suggestions only, none applied to the compose files:

- **Thinking shows in Open WebUI despite effort `none`:** keep `--reasoning-parser qwen3`. Then either set `enable_thinking=false` through a custom parameter or a function/filter in Open WebUI, or create a second model preset with that default. The Qwen template reads `enable_thinking`, not `reasoning_effort`.
- **Raw `<tool_call>` text or `tool_calls: null`:** make sure `--tool-call-parser qwen3_coder` is present.
- **Context-length tests fail after raising it:** raising `--context-length` means revisiting `--max-total-tokens` and `--mem-fraction-static`.

## Adding tests

Use the `llm` fixture:

```python
def test_something(llm):
    m = llm.message("prompt", chat_template_kwargs={"enable_thinking": False})
    assert "expected" in m["content"]
```

- `llm.chat(...)` returns the full response JSON.
- `llm.message(...)` returns the first choice's message.
- `llm.stream(...)` yields the streaming `delta` dicts.
- Extra keyword arguments are merged into the request body.
- `reasoning_of(msg)` returns the reasoning text whether the server uses `reasoning_content` or `reasoning`.

## Quick manual check with curl

```
curl -s http://10.0.90.101:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.6-35B-A3B-NVFP4","reasoning_effort":"none",
       "chat_template_kwargs":{"enable_thinking":false},
       "messages":[{"role":"user","content":"What is 17*23?"}]}' | jq '.choices[0].message'
```

With a correct setup, `content` holds the answer and `reasoning_content` is null.
