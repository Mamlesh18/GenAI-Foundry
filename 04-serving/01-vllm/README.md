# 01 · vLLM

> **In one sentence:** vLLM is an open-source serving engine that loads a model onto your GPUs and
> exposes it as an OpenAI-compatible API, with continuous batching and paged KV cache built in.

It is the default choice for serving open models on GPUs, and the project that introduced
[PagedAttention](../../03-inference/04-paged-attention/).

---

## 1. Architecture: three kinds of process

The most important thing to understand about vLLM is that it is **not one program**. It splits into
separate processes that talk over ZeroMQ sockets, so that Python work (HTTP, tokenizing) never
blocks the GPU.

```
   HTTP requests
        |
        v
 +-------------------------------------------------------------+
 |  API SERVER PROCESS                                          |
 |                                                              |
 |   FastAPI  ->  validate, apply chat template, TOKENIZE       |
 |                                                              |
 |   ...and on the way back: DETOKENIZE, stream SSE to client   |
 +-------------------------------------------------------------+
        |   ZeroMQ sockets (serialized requests / outputs)
        v
 +-------------------------------------------------------------+
 |  ENGINE CORE PROCESS            "the busy loop"              |
 |                                                              |
 |   +-------------------+    +------------------------------+  |
 |   |    SCHEDULER      |<-->|     KV CACHE MANAGER         |  |
 |   | which requests    |    | block pool, block tables,    |  |
 |   | run this step,    |    | prefix cache, eviction       |  |
 |   | and how many      |    +------------------------------+  |
 |   | tokens each gets  |                                      |
 |   +-------------------+                                      |
 |            |                                                 |
 |            v                                                 |
 |   +-------------------------------------------------------+  |
 |   |  EXECUTOR                                             |  |
 |   |  UniProcExecutor  = 1 GPU                             |  |
 |   |  MultiProcExecutor = many GPUs (tensor parallel)      |  |
 |   +-------------------------------------------------------+  |
 +-------------------------------------------------------------+
        |
        v
 +-------------------------------------------------------------+
 |  WORKER PROCESS (one per GPU)                                |
 |    ModelRunner -> prepares tensors, runs the forward pass,   |
 |                   holds the KV cache and the sampler         |
 |    Model       -> the actual torch.nn.Module                 |
 +-------------------------------------------------------------+
```

| Component | What it does |
|---|---|
| **API server process** | HTTP, input validation, tokenization, detokenization, streaming |
| **EngineCore process** | Runs the scheduler and KV cache manager, drives model execution. One per data-parallel rank |
| **Scheduler** | Each step, chooses which requests run and how many tokens each may process |
| **KV cache manager** | Owns the block pool: allocates blocks, matches cached prefixes, evicts |
| **Executor** | `UniProcExecutor` for a single GPU, `MultiProcExecutor` to coordinate several |
| **Worker + ModelRunner** | One per GPU: loads its shard of the weights, runs forward passes |

---

## 2. The busy loop

EngineCore repeats three steps forever:

```
while True:
    1. SCHEDULE    pick requests, give each a token budget
    2. FORWARD     one batched pass over all of them
    3. POSTPROCESS sample tokens, mark finished requests, send outputs back
```

Two details make this powerful:

**A unified token budget.** The scheduler does not treat "prefill requests" and "decode requests"
as separate categories. It keeps something like `{request_id: number_of_tokens}` and allocates from
a fixed budget per step. A brand-new request with a 5,000-token prompt can take 2,000 tokens this
step and the rest next step (that is **chunked prefill**), while thirty ongoing conversations take
one token each. This single mechanism gives chunked prefill, prefix caching and speculative
decoding without special cases.

**Nothing is padded.** Requests of different lengths are concatenated into one flat tensor, and the
attention kernels keep them separate. No wasted compute on padding.

---

## 3. Two ways to use it

**Offline** — a Python script, for batch jobs and experiments:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
outputs = llm.generate(["Hello, my name is", "The capital of France is"],
                       SamplingParams(temperature=0.8, max_tokens=64))
for out in outputs:
    print(out.outputs[0].text)
```

vLLM batches those prompts for you — the more you hand it at once, the better the GPU is used.

**Online** — a server, for applications:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct        # listens on port 8000
```

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)
print(response.choices[0].message.content)
```

The client in [`../examples/openai_client.py`](../examples/openai_client.py) works against this
server unchanged — try it with `--base-url http://127.0.0.1:8000/v1`.

---

## 4. The flags that matter

Each one maps to a concept from [03 · Inference](../../03-inference/):

| Flag | Default-ish | What it controls | Concept |
|---|---|---|---|
| `--gpu-memory-utilization` | 0.9+ | Share of the GPU vLLM may use. Whatever the weights don't take becomes KV cache | [KV cache](../../03-inference/01-kv-cache/) |
| `--max-model-len` | from model config | Longest context allowed. Lower it if you get "not enough KV cache" at startup | [KV cache](../../03-inference/01-kv-cache/) |
| `--kv-cache-dtype fp8` | `auto` | Store the KV cache in 8 bits: roughly twice as many users | [KV cache](../../03-inference/01-kv-cache/) |
| `--max-num-seqs` | — | Maximum requests in one batch | [Batching](../../03-inference/02-batching/) |
| `--max-num-batched-tokens` | — | Maximum tokens processed per step | [Batching](../../03-inference/02-batching/) |
| `--enable-chunked-prefill` | — | Split long prompts so they don't stall ongoing decodes | [Batching](../../03-inference/02-batching/) |
| `--enable-prefix-caching` | — | Reuse KV blocks for shared prefixes | [Paged attention](../../03-inference/04-paged-attention/) |
| `--block-size` | 16 | Tokens per KV block | [Paged attention](../../03-inference/04-paged-attention/) |
| `--quantization` | — | Load a quantized checkpoint (AWQ, GPTQ, FP8, ...) | [Quantization](../../03-inference/03-quantization/) |
| `--speculative-config` | — | Draft model or method for speculative decoding | [Speculative decoding](../../03-inference/05-speculative-decoding/) |
| `--tensor-parallel-size` | 1 | Split the model across GPUs | [Distributed](../03-distributed-inference/) |

Defaults change between releases — check `vllm serve --help` for your version.

---

## 5. Watching it run

Two endpoints you should wire up on day one:

```bash
curl http://127.0.0.1:8000/health      # 200 = process alive (does NOT mean "healthy queue")
curl http://127.0.0.1:8000/metrics     # Prometheus text format
```

The metrics worth graphing:

| Metric | Tells you |
|---|---|
| `vllm:num_requests_running` | how many are in the batch right now |
| `vllm:num_requests_waiting` | **queue depth — the first sign of overload** |
| `vllm:gpu_cache_usage_perc` | how full the KV cache is; near 100% means preemption is coming |
| `vllm:time_to_first_token_seconds` | TTFT histogram |
| `vllm:e2e_request_latency_seconds` | full request latency histogram |

**Autoscale on queue depth and KV-cache usage**, not GPU utilisation — a GPU waiting on memory
still looks busy.

---

## 6. When it goes wrong

| Symptom | Likely cause | Try |
|---|---|---|
| CUDA out of memory at startup | weights + KV cache don't fit | lower `--gpu-memory-utilization`, lower `--max-model-len`, use a quantized model, or add GPUs |
| "The model's max seq len is larger than the KV cache can hold" | context too long for the memory left | lower `--max-model-len` or free memory |
| Very slow first token under load | long prompts blocking the batch | `--enable-chunked-prefill`, cap prompt length |
| Throughput lower than expected | batch too small | raise `--max-num-seqs` / `--max-num-batched-tokens` (watch latency) |
| Requests occasionally restart | preemption: KV cache full | more memory, shorter `--max-model-len`, or fewer concurrent requests |
| Kubernetes kills the pod on startup | model load exceeds the probe delay | raise the readiness probe's initial delay |

---

## 7. Exercises

*(1–4 need no GPU.)*

1. **Trace a request.** Using the diagram in section 1, write down every component a request passes
   through, in order, from HTTP arrival to the first streamed token.
2. **Predict the failure.** A 7B fp16 model on a 24 GB GPU, `--max-model-len 32768`, 50 concurrent
   users. Using the [KV cache calculator](../../03-inference/01-kv-cache/examples/kv_cache_calculator.py),
   will it start? Will it serve 50 users? Which flag would you change first?
3. **Read the flags.** Find three flags in `vllm serve --help` (or the engine arguments docs) that
   are not in the table above, and say which module of track 03 each one relates to.
4. **Design the alerts.** Which of the metrics in section 5 would you page a human for, at what
   threshold, and why not the others?
5. **Run it.** On a free Colab GPU or a rented one, serve a small model and point
   `openai_client.py` at it. Compare its TTFT and TPOT against the mock server's.
6. **Tune it.** Serve the same model with `--max-num-seqs 4` and `--max-num-seqs 64`, and load-test
   both. What happens to throughput, and to p95 latency?

---

## 8. Projects to build and test

### Beginner — Serve a model and document it
Serve a small model, then write a one-page runbook: the exact command, memory used, startup time,
measured TTFT/TPOT, and the three things most likely to break.

**How to test it:** hand the runbook to someone else and have them reproduce your setup without
asking you anything.

### Intermediate — Find the capacity limit
Load-test one vLLM server at rising request rates.

**How to test it:** plot throughput, p95 TTFT and `vllm:num_requests_waiting` against request rate
on one chart. Identify the rate where queue depth starts growing without bound — that is your
capacity, and it will be lower than the rate where throughput peaks.

### Advanced — Prefix caching in anger
Take a workload with a long shared system prompt and measure it with and without
`--enable-prefix-caching`.

**How to test it:** report TTFT and throughput for both, and vary the shared prefix length. Explain
the shape of the curve using [module 04](../../03-inference/04-paged-attention/).

---

## 9. Resources

**Start here**
- [vLLM quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) — install, serve, call.
- [Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/) — the official version of section 1.
- [Life of an inference request (vLLM V1)](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1) — a request traced end to end.

**Go deeper**
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — long, excellent, and worth the time.
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — why the engine is split into processes.
- [Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/) — every flag.
- [Parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) — multi-GPU, covered in [module 03](../03-distributed-inference/).

**Production**
- [vLLM on Kubernetes](https://scaleops.com/blog/vllm-kubernetes/) · [Monitoring vLLM](https://akrisanov.com/vllm-metrics/)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — the paper behind the engine.

---

**Track:** [04 · Serving](../) · **Next:** [02 · SGLang](../02-sglang/)
