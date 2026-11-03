#!/usr/bin/env python3
"""
Minimal load generator for an llm-d (or any OpenAI-compatible) endpoint.

Deliberately stdlib-only. Its job is to prove the madengine <-> llm-d contract
end to end — endpoint reachable, model served, results parsed into perf.csv —
not to be a production benchmark harness. For real numbers, point the model's
run.sh at guidellm, inference-perf or `vllm bench serve` instead.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def _post(url: str, payload: dict, timeout: float) -> dict:
    """POST JSON and return the decoded response."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def check_model_served(endpoint: str, model: str, timeout: float) -> None:
    """Fail early and loudly if the gateway is not serving the expected model."""
    url = f"{endpoint.rstrip('/')}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            served = [m["id"] for m in json.loads(response.read().decode())["data"]]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        sys.exit(f"ERROR: could not read {url}: {e}")

    print(f"Endpoint serves: {', '.join(served) or '(nothing)'}")
    if model not in served:
        sys.exit(
            f"ERROR: model '{model}' is not served by {endpoint}. "
            f"Available: {', '.join(served) or '(nothing)'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", required=True, help="Base URL of the llm-d gateway"
    )
    parser.add_argument("--model", required=True, help="Model name to request")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--prompt",
        default="Summarize the benefits of disaggregated prefill and decode serving.",
    )
    args = parser.parse_args()

    check_model_served(args.endpoint, args.model, args.timeout)

    url = f"{args.endpoint.rstrip('/')}/v1/completions"
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }

    def one_request(index: int):
        # Vary the prompt so prefix-cache-aware routing has something to chew on
        # while still keeping every request the same shape.
        body = dict(payload, prompt=f"[{index}] {args.prompt}")
        started = time.perf_counter()
        try:
            response = _post(url, body, args.timeout)
        except (urllib.error.URLError, OSError, ValueError) as e:
            return None, 0, str(e)
        elapsed = time.perf_counter() - started
        completion_tokens = (response.get("usage") or {}).get("completion_tokens", 0)
        return elapsed, completion_tokens, None

    print(
        f"Sending {args.num_requests} requests at concurrency {args.concurrency} "
        f"to {url}"
    )
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one_request, range(args.num_requests)))
    wall_elapsed = time.perf_counter() - wall_start

    latencies = [r[0] for r in results if r[0] is not None]
    output_tokens = sum(r[1] for r in results)
    failures = [r[2] for r in results if r[2] is not None]

    for message in failures[:5]:
        print(f"  request failed: {message}", file=sys.stderr)
    if failures:
        print(f"{len(failures)}/{args.num_requests} requests failed", file=sys.stderr)

    if not latencies:
        print(
            "ERROR: every request failed; no performance number to report",
            file=sys.stderr,
        )
        return 1

    throughput = len(latencies) / wall_elapsed
    print("=" * 72)
    print(f"Successful requests : {len(latencies)}/{args.num_requests}")
    print(f"Wall time           : {wall_elapsed:.2f} s")
    print(f"Mean latency        : {statistics.mean(latencies):.3f} s")
    print(f"P99 latency         : {max(latencies):.3f} s")
    print(f"Output tokens       : {output_tokens}")
    print(f"Output tok/s        : {output_tokens / wall_elapsed:.2f}")
    print("=" * 72)

    # The line madengine scrapes (PERFORMANCE_LOG_PATTERN in deployment/base.py).
    print(f"performance: {throughput:.4f} requests_per_second")

    # Any request failing means the reported number does not describe the run
    # that was asked for; do not let it reach perf.csv as a clean result.
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
