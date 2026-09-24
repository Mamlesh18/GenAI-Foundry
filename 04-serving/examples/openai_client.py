"""
One client, any server -- and the two numbers that describe serving speed.

vLLM, SGLang, TGI, Ollama and the cloud providers all expose the same
OpenAI-compatible HTTP API. So the same client talks to all of them; only the
URL changes. This script uses nothing but the Python standard library, so you
can see the raw HTTP underneath the SDKs.

It does three things:
  1. asks the server what it is  (GET /v1/models)
  2. sends a normal request      (POST /v1/chat/completions)
  3. sends a STREAMING request and measures
       TTFT  time to first token  -- how long prefill took
       TPOT  time per output token -- how fast decode runs

Run against anything:

    python mock_server.py                                    # no GPU needed
    python openai_client.py

    vllm serve Qwen/Qwen2.5-1.5B-Instruct                     # real, port 8000
    python openai_client.py --model Qwen/Qwen2.5-1.5B-Instruct

    python -m sglang.launch_server --model-path <model>        # real, port 30000
    python openai_client.py --base-url http://127.0.0.1:30000/v1 --model <model>

    python openai_client.py --base-url https://api.groq.com/openai/v1 \
        --model llama-3.3-70b-versatile --api-key $GROQ_API_KEY

Note the URLs say 127.0.0.1, not "localhost". On many machines "localhost"
resolves to IPv6 first, and the failover to IPv4 adds ~2 seconds to EVERY
request -- which would ruin the latency numbers this script reports.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://localhost:8000/v1"


def request_json(url, api_key=None, payload=None, stream=False):
    """POST if payload is given, otherwise GET. Returns the raw response object
    when streaming, or parsed JSON when not."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    response = urllib.request.urlopen(req, timeout=120)
    return response if stream else json.loads(response.read())


def friendly_error(exc, base_url):
    print(f"\n  Could not reach {base_url}\n  {type(exc).__name__}: {exc}\n")
    print("  Checks:")
    print("    - is a server running?  python mock_server.py")
    print("    - right port?  vLLM defaults to 8000, SGLang to 30000")
    print("    - cloud endpoint? pass --api-key")
    sys.exit(1)


def show_models(base_url, api_key):
    print("=" * 72)
    print("1. WHAT IS THIS SERVER RUNNING?    GET /v1/models")
    print("=" * 72)
    data = request_json(f"{base_url}/models", api_key)
    models = [m["id"] for m in data.get("data", [])]
    for model in models[:10]:
        print(f"  {model}")
    if len(models) > 10:
        print(f"  ... and {len(models) - 10} more")
    print("\n  A serving engine usually holds ONE model in GPU memory. Cloud endpoints")
    print("  list many because they are a router in front of many servers.")
    return models


def normal_request(base_url, api_key, model, prompt):
    print("\n" + "=" * 72)
    print("2. A NORMAL REQUEST    POST /v1/chat/completions")
    print("=" * 72)
    start = time.perf_counter()
    result = request_json(f"{base_url}/chat/completions", api_key, {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
        "temperature": 0.7,
    })
    elapsed = time.perf_counter() - start

    print(f"  {result['choices'][0]['message']['content'].strip()}\n")
    usage = result.get("usage") or {}
    print(f"  total time     : {elapsed:.2f}s")
    print(f"  prompt tokens  : {usage.get('prompt_tokens', '?')}")
    print(f"  output tokens  : {usage.get('completion_tokens', '?')}")
    print(f"  finish_reason  : {result['choices'][0].get('finish_reason')}")
    print("\n  You waited for the whole answer before seeing anything. That is why")
    print("  every chat UI streams instead.")


def streaming_request(base_url, api_key, model, prompt):
    print("\n" + "=" * 72)
    print("3. A STREAMING REQUEST -- and the numbers that matter")
    print("=" * 72)

    start = time.perf_counter()
    first_token_at, token_times, text = None, [], []

    response = request_json(f"{base_url}/chat/completions", api_key, {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
        "temperature": 0.7,
        "stream": True,
    }, stream=True)

    print()
    for raw_line in response:
        line = raw_line.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        piece = chunk["choices"][0].get("delta", {}).get("content")
        if not piece:
            continue
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now - start
        token_times.append(now)
        text.append(piece)
        print(piece, end="", flush=True)

    total = time.perf_counter() - start
    n = len(token_times)
    print("\n")
    if n < 2 or first_token_at is None:
        print("  Too few chunks to measure.")
        return

    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    tpot = sum(gaps) / len(gaps)
    print(f"  TTFT  time to first token : {first_token_at * 1000:7.0f} ms   <- prefill: reading your prompt")
    print(f"  TPOT  time per output token: {tpot * 1000:7.0f} ms   <- decode: one forward pass per token")
    print(f"  chunks received            : {n}")
    print(f"  total time                 : {total:7.2f} s")
    print(f"\n  check: TTFT + TPOT x {n - 1} = "
          f"{(first_token_at + tpot * (n - 1)):.2f}s  vs measured {total:.2f}s")
    print(
        "\n  TTFT is what makes an app feel responsive; TPOT is the reading speed.\n"
        "  A long prompt raises TTFT. A long answer raises total time, not TTFT.\n"
        "  On a busy server both rise as the batch grows -- that is the throughput\n"
        "  vs latency trade-off from ../../03-inference/02-batching/."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=None, help="defaults to the first model the server lists")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--prompt", default="In two sentences, what does an LLM serving engine do?")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"\n  talking to: {base_url}\n")

    try:
        models = show_models(base_url, args.api_key)
        model = args.model or (models[0] if models else None)
        if not model:
            sys.exit("  Server listed no models -- pass --model explicitly.")
        print(f"\n  using model: {model}")
        normal_request(base_url, args.api_key, model, args.prompt)
        streaming_request(base_url, args.api_key, model, args.prompt)
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        friendly_error(exc, base_url)
    except urllib.error.HTTPError as exc:               # pragma: no cover
        sys.exit(f"  HTTP {exc.code}: {exc.read().decode()[:300]}")


if __name__ == "__main__":
    main()
