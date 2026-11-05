"""Single entry point: run the pytest suite, the load benchmark, or both.

    uv run llm-bench                  # tests, then benchmark
    uv run llm-bench --tests          # tests only
    uv run llm-bench --benchmark      # benchmark only
    uv run llm-bench --tests -k tool  # extra arguments go to pytest
    uv run llm-bench --benchmark --levels 1 4 16   # extra arguments go to the benchmark

Endpoint and model come from the environment / .env (LLM_URL, LLM_MODEL);
--url and --model override them for this run.
"""
import argparse
import os
from pathlib import Path

# src/llm_bench/cli.py -> repo root (the editable install keeps the package in the repo).
ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", action="store_true", help="run only the pytest suite")
    p.add_argument("--benchmark", action="store_true", help="run only the benchmark")
    p.add_argument("--url", help="override LLM_URL")
    p.add_argument("--model", help="override LLM_MODEL")
    args, extra = p.parse_known_args()

    # Must be set before settings is imported; real env vars beat .env.
    if args.url:
        os.environ["LLM_URL"] = args.url
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    both = not (args.tests or args.benchmark)
    code = 0

    if args.tests or both:
        import pytest
        print("=== tests ===", flush=True)
        code = int(pytest.main([str(ROOT), "-q", *(extra if args.tests else [])]))

    if args.benchmark or both:
        from . import benchmark
        print("\n=== benchmark ===", flush=True)
        benchmark.main(extra if args.benchmark else [])

    return code
