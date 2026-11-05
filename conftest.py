import httpx
import pytest

from llm_bench.settings import MODEL, URL

# Every way a client may ask for "no thinking". Open WebUI's "reasoning effort"
# field sends `reasoning_effort`; Qwen's chat template reads `enable_thinking`.
NO_THINK_VARIANTS = {
    "reasoning_effort=none": {"reasoning_effort": "none"},
    "enable_thinking=false": {"chat_template_kwargs": {"enable_thinking": False}},
    "both": {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}},
}


class Client:
    def __init__(self):
        self.http = httpx.Client(timeout=120)

    def chat(self, content="What is 17*23? Answer with just the number.", *, messages=None, **extra):
        body = {
            "model": MODEL,
            "max_tokens": 1024,
            "temperature": 0,
            "messages": messages or [{"role": "user", "content": content}],
            **extra,
        }
        r = self.http.post(URL, json=body)
        r.raise_for_status()
        return r.json()

    def message(self, *a, **kw):
        return self.chat(*a, **kw)["choices"][0]["message"]

    def stream(self, content, **extra):
        """Yield delta dicts from a streaming request."""
        import json

        body = {"model": MODEL, "max_tokens": 1024, "temperature": 0, "stream": True,
                "messages": [{"role": "user", "content": content}], **extra}
        with self.http.stream("POST", URL, json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    if chunk["choices"]:
                        yield chunk["choices"][0]["delta"]


def reasoning_of(msg):
    """Reasoning text regardless of field name used by the server."""
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


@pytest.fixture(scope="session")
def llm():
    c = Client()
    try:
        c.http.get(URL.replace("/chat/completions", "/models"), timeout=5).raise_for_status()
    except httpx.HTTPError as e:
        pytest.exit(f"Server not reachable at {URL}: {e}")
    return c
