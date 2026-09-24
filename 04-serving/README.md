# 04 · Serving

> **In one sentence:** serving is the system that sits between your users and the model — it
> accepts requests over HTTP, decides which ones run in which forward pass, manages GPU memory,
> and streams tokens back.

[03 · Inference](../03-inference/) explained the *techniques* (KV cache, batching, quantization,
paging, speculation). This track is about the **software that implements all of them for you** —
and how to run it.

---

## 1. Why not just load the model in a web app?

The obvious approach is a Flask app that calls `model.generate()`. Here is what breaks, roughly in
the order you will discover it:

| You do this | What happens | What an engine does instead |
|---|---|---|
| One request at a time | GPU is idle most of the time; one user's long answer blocks everyone | Continuous batching |
| Pre-allocate memory per request | Runs out of memory with 3 users | Paged KV cache |
| Return the full answer at the end | Users stare at a spinner for 10 seconds | Token streaming |
| No queue | The 50th simultaneous request crashes the process | Admission control + queueing |
| Recompute the system prompt every time | Pay for the same 2,000 tokens on every call | Prefix caching |
| Restart to change anything | Model load takes minutes; requests fail | Health checks, rolling deploys |

A serving engine is all of that, already built, tested and optimised. **You configure it; you do
not write it.**

---

## 2. The architecture every engine shares

Different engines use different names, but the shape is the same:

```
   many users
       |  HTTP  (OpenAI-compatible JSON)
       v
 +--------------------------------------------------------------+
 |  ROUTER / LOAD BALANCER        (only once you have replicas)  |
 |  picks the least busy server, retries, rate limits            |
 +--------------------------------------------------------------+
       |
       v
 +--------------------------------------------------------------+
 |  API SERVER                                                   |
 |  validates JSON, applies the chat template, TOKENIZES,        |
 |  streams results back as they appear                          |
 +--------------------------------------------------------------+
       |  queue of waiting requests
       v
 +--------------------------------------------------------------+
 |  SCHEDULER            <-- the brain                           |
 |  every step: who runs now? admit, preempt, or make them wait  |
 |  budget: max sequences, max tokens per step, free KV memory   |
 +--------------------------------------------------------------+
       |                                  ^
       v                                  | free/allocate
 +----------------------------+   +----------------------------+
 |  MODEL EXECUTOR            |   |  KV CACHE MANAGER          |
 |  runs one forward pass for |   |  block pool, block tables, |
 |  the whole batch           |   |  prefix cache, eviction    |
 +----------------------------+   +----------------------------+
       |
       v
 +--------------------------------------------------------------+
 |  WORKERS -- one process per GPU                               |
 |  holds a shard of the weights, runs the model, talks to the   |
 |  other GPUs when the model is split across them               |
 +--------------------------------------------------------------+
       |
       v
     GPUs
```

| Component | Job | Covered in |
|---|---|---|
| Router | spread load across replicas | [03 · Distributed](03-distributed-inference/) |
| API server | HTTP, tokenization, streaming | this page |
| **Scheduler** | which requests run in each step | [03-inference/02 · Batching](../03-inference/02-batching/) |
| **KV cache manager** | GPU memory for conversations | [03-inference/01](../03-inference/01-kv-cache/), [04](../03-inference/04-paged-attention/) |
| Executor / workers | the actual forward pass | [03 · Distributed](03-distributed-inference/) |

---

## 3. The life of one request

1. **Arrives** as JSON at `POST /v1/chat/completions`.
2. **Validated and templated** — messages become one token sequence using the model's chat template.
3. **Tokenized** — text becomes token ids.
4. **Queued** — the request waits for the scheduler.
5. **Admitted** — the scheduler checks there are free KV blocks and a free slot in the batch.
6. **Prefill** — the prompt goes through the model in one pass, filling the KV cache; the first
   token appears. *(This is your TTFT.)*
7. **Decode loop** — on every step this request rides along with everyone else's in one batched
   forward pass, producing one token each. *(This is your TPOT.)*
8. **Streamed** — each token is detokenized and sent as a server-sent event.
9. **Finished** — an end token or `max_tokens` stops it; its KV blocks return to the pool
   immediately so a waiting request can start.

Steps 5–9 are happening for dozens of requests simultaneously, at different stages. That is
continuous batching.

---

## 4. The OpenAI-compatible API

Almost every engine copies OpenAI's HTTP API. That is why you can swap vLLM for SGLang for a cloud
provider and change only a URL.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | the main one: messages in, message (or token stream) out |
| `POST /v1/completions` | older, plain-text-in/out |
| `POST /v1/embeddings` | vectors, when serving an embedding model |
| `GET /v1/models` | which model is loaded |
| `GET /health` | is the server alive? (used by Kubernetes and load balancers) |
| `GET /metrics` | Prometheus metrics: queue depth, KV usage, latencies |

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-1.5B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

Set `"stream": true` and the reply arrives as a series of `data: {...}` lines instead of one JSON
object.

---

## 5. Choosing an engine

| Engine | Runs on | Best for | Trade-off |
|---|---|---|---|
| **[vLLM](01-vllm/)** | NVIDIA, AMD, others | the default for GPU serving; huge model coverage | needs a GPU and some tuning |
| **[SGLang](02-sglang/)** | NVIDIA, AMD, others | heavy shared prefixes (agents, multi-turn), structured output | smaller community than vLLM |
| **TensorRT-LLM** | NVIDIA only | maximum speed on NVIDIA hardware | compile step, steeper learning curve |
| **Ollama / llama.cpp** | laptops, Macs, CPUs | local development, single user | not built for many concurrent users |

**A simple rule:** learning or developing locally → Ollama. Serving real traffic on a GPU → vLLM.
Workloads where requests share long prefixes → try SGLang and measure. Squeezing an NVIDIA
deployment → TensorRT-LLM.

---

## 6. Modules

| # | Module | What you will learn |
|---|---|---|
| 01 | [vLLM](01-vllm/) | Its process architecture, the engine loop, and the flags that matter |
| 02 | [SGLang](02-sglang/) | Its three-process design and RadixAttention prefix caching |
| 03 | [Distributed Inference](03-distributed-inference/) | Splitting a model across GPUs and machines: TP, PP, DP, EP |

---

## 7. Before you call it production

| Concern | What to do |
|---|---|
| **Startup time** | Loading a 70B model takes minutes. Kubernetes readiness probes must allow for it, or traffic arrives before the model is ready. |
| **Health** | Point liveness/readiness probes at `/health`. Note it only says the process is alive — not that the queue is healthy. |
| **Metrics** | Scrape `/metrics`. Watch queue depth and KV-cache usage, not just GPU utilisation. |
| **Limits** | Cap `max_tokens` and context length per request; one huge request can starve the batch. |
| **Timeouts** | Set client and server timeouts. A queued request can wait a long time under load. |
| **Autoscaling** | Scale on queue depth and KV-cache usage — GPU utilisation looks "busy" even when idle-waiting on memory. |
| **Cost** | GPUs are billed by the hour, not the token. An idle replica costs the same as a busy one. |

---

## 8. Examples

Two small files, no GPU needed:

```bash
cd examples
python mock_server.py        # terminal 1: a fake LLM server that speaks the OpenAI API
python openai_client.py      # terminal 2: talk to it, and measure TTFT and TPOT
```

| File | What it is for |
|---|---|
| [`examples/mock_server.py`](examples/mock_server.py) | A ~120-line server implementing `/health`, `/v1/models` and `/v1/chat/completions` (streaming included), with realistic delays. Shows exactly what an "OpenAI-compatible" server *is*. |
| [`examples/openai_client.py`](examples/openai_client.py) | One client, standard library only. Works unchanged against the mock, vLLM, SGLang or a cloud endpoint, and measures time to first token and time per output token. |

Point the same client at a real server later:

```bash
python openai_client.py --base-url http://127.0.0.1:30000/v1        # SGLang
python openai_client.py --base-url https://api.groq.com/openai/v1 \
    --model llama-3.3-70b-versatile --api-key $GROQ_API_KEY         # a cloud provider
```

---

## 9. Glossary

| Term | Plain meaning |
|---|---|
| **Engine** | the program that loads the model and runs requests (vLLM, SGLang, ...) |
| **Scheduler** | decides which requests run in each forward pass |
| **Worker** | one process driving one GPU |
| **Replica** | a complete copy of the server, for more throughput |
| **Router / gateway** | spreads requests across replicas |
| **OpenAI-compatible** | speaks the same HTTP API as OpenAI, so clients are interchangeable |
| **SSE (server-sent events)** | the `data: {...}` streaming format |
| **Preemption** | pausing a running request to free memory, and resuming it later |
| **Readiness probe** | a health check that decides whether a server should receive traffic |

---

## 10. Exercises

1. **Read the protocol.** Run the mock server and client, then run the client with
   `--prompt "..."`. Add a `print` in the client to dump one raw SSE line. What are the fields?
2. **Break it deliberately.** Stop the mock server and run the client. Then run it against a wrong
   port. Are the error messages good enough to debug from?
3. **Change the physics.** In `mock_server.py`, set `TIME_TO_FIRST_TOKEN` to 3 s and
   `TIME_PER_OUTPUT_TOKEN` to 0.005 s. Which feels worse to use, and which metric shows it?
4. **Map a real server.** Open the vLLM or SGLang docs and match each component in the diagram
   above to their name for it.
5. **Plan a deployment.** For a chatbot with 200 concurrent users, 8K context and a 7B model: which
   engine, how many GPUs, what would you monitor? Use
   [the planner](03-distributed-inference/examples/gpu_planner.py) for the hardware part.

## 11. Projects

- **Beginner — A router.** Put a small load balancer in front of two mock servers, sending each
  request to the one with fewer in flight. *Test:* with one server artificially slowed down, most
  requests should go to the fast one; confirm no request is lost.
- **Intermediate — A metrics dashboard.** Add a `/metrics` endpoint to the mock server (queue
  depth, requests running, token counts) and graph it while running a load test. *Test:* queue
  depth should rise when you exceed capacity, and recover afterwards.
- **Advanced — Serve a real model.** Run vLLM on a rented GPU, put it behind a health check, and
  load-test it. *Test:* report p50/p95 TTFT and TPOT at three request rates, and identify which
  metric warns you first that the server is overloaded.

---

## 12. Resources

**Start here**
- [vLLM quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) — install, serve, and call a model in about ten minutes.
- [SGLang documentation](https://docs.sglang.io/) — install, launch, and feature list.
- [Ollama](https://ollama.com) — the easiest local serving, good for developing against.

**Go deeper**
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — the best single article on how an engine actually works.
- [Life of an inference request (vLLM V1)](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1) — a request traced through every component.
- [vLLM on Kubernetes: deploy, scale and monitor](https://scaleops.com/blog/vllm-kubernetes/) — the production side.
- [Monitoring vLLM in production](https://akrisanov.com/vllm-metrics/) — which metrics matter and what to alert on.

---

**Previous track:** [03 · Inference](../03-inference/) · **Next track:** [05 · Evaluation](../05-evaluation/)
