"""Shared configuration: values come from the environment, then .env, then defaults."""
import os
from pathlib import Path

from dotenv import load_dotenv

# The editable install (uv sync) keeps this file in the repo; .env lives at the root.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

URL = os.environ.get("LLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("LLM_MODEL", "default")

BENCH_LEVELS = [int(x) for x in os.environ.get("BENCH_LEVELS", "1 2 4 8").split()]
BENCH_MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "256"))
BENCH_TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "120"))
BENCH_PROMPT = os.environ.get("BENCH_PROMPT", "Write a short story about a robot learning to paint.")
BENCH_THINKING = os.environ.get("BENCH_THINKING", "false").lower() in ("1", "true", "yes")
