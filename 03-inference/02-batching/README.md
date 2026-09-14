# 02 · Batching

> **In one sentence:** batching runs many users' requests through the model in the same forward
> pass — and *continuous* batching lets requests join and leave that batch after every single
> token, so no GPU time is wasted waiting.

This is the biggest single reason a hosted LLM costs cents instead of dollars.

---

## 1. Why batching works at all

Remember from the [track introduction](../) that decode is **memory-bound**: for each token the GPU
spends most of its time waiting for weights to arrive from memory, while its arithmetic units sit
largely idle.

So what happens if you process 16 users' next tokens in the same pass instead of 1? The weights
still have to be read **once** — the same big cost — and the extra arithmetic for 15 more users
uses compute that was idle anyway. You get roughly 16× the tokens for a small increase in time per
step.

From the simulator's cost model (step time = 1 + 0.02 × batch size):

| Batch size | Time per step | Tokens per unit time |
|---|---|---|
| 1 | 1.02 | 1.0 |
| 8 | 1.16 | 6.9 |
| 16 | 1.32 | 12.1 |
| 64 | 2.28 | 28.1 |

This holds on real hardware until the batch becomes **compute-bound** or the **KV cache runs out of
memory** ([module 01](../01-kv-cache/)). Past that point, per-user latency climbs.

---

## 2. Static batching — the obvious way, and its flaw

**Analogy:** a tour bus. It waits until it has a group, then drives everyone around, and nobody
gets off until the *longest* tour stop is finished. Someone who wanted a five-minute ride sits on
the bus for three hours.

Static batching collects a group of requests, pads them to the same length, and runs until the
longest one finishes. Answers have wildly different lengths ("Yes." vs a 2,000-word essay), so:

- short requests finish early and then **occupy a slot doing nothing** (padding)
- new requests **cannot join** until the whole batch is done

The picture from [`batching_simulator.py`](examples/batching_simulator.py) — four slots, requests
A–D arrive first, E and F one step later (`.` = a slot wasted on a finished request):

```
STATIC                          CONTINUOUS
  step    0123456789              step    01234567
  slot 0  AA......EE              slot 0  AAEE
  slot 1  BBBBBBBBFF              slot 1  BBBBBBBB
  slot 2  CCC.....                slot 2  CCCFF
  slot 3  DDDDD...                slot 3  DDDDD
  14 wasted slot-steps            0 wasted slot-steps
  E, F finish at step 10          E finishes at 4, F at 5
```

---

## 3. Continuous batching — the fix

**Analogy:** a hop-on, hop-off bus. The moment someone gets off, the next person in the queue gets
on.

Continuous batching (introduced as **iteration-level scheduling** in the
[Orca paper](https://www.usenix.org/conference/osdi22/presentation/yu), and called **in-flight
batching** in TensorRT-LLM) makes the scheduling decision **after every decode step**:

```
every step:
  1. remove requests that just produced their end token
  2. fill the free slots from the waiting queue
  3. run one forward pass for everyone in the batch
```

The batch is always as full as the traffic allows, and nobody waits for a stranger's long answer.

### What the simulator measures

400 chat-like requests (80% short answers, 20% long), 16 slots, the same arrival pattern for both:

| Policy | Tokens per unit time | Median latency | 95th-percentile latency | Compute wasted on padding |
|---|---|---|---|---|
| Static | 2.73 | 5,191 | 9,667 | 77% |
| **Continuous** | **9.35** | **58** | **538** | **0%** |

At this traffic level static batching cannot keep up at all: every batch runs as long as its
longest request, so its queue grows faster than it drains. Continuous batching handles the same
traffic with the same 16 slots, and requests barely wait. The
[Anyscale benchmark](https://www.anyscale.com/blog/continuous-batching-llm-inference) found up to
**23× throughput** over naive batching on real hardware.

---

## 4. The next problem: prefill gets in the way

Continuous batching mixes requests at different stages. When a new request joins, its **prefill**
must run — and a 10,000-token prompt is a heavy, compute-bound pass. Every user currently decoding
has to wait for it, so their token stream visibly stutters.

Two solutions, both now standard:

| Technique | Idea | In vLLM |
|---|---|---|
| **Chunked prefill** | split a long prompt into chunks and mix one chunk per step with the ongoing decodes, so no single step is huge | `--enable-chunked-prefill`, sized by `--max-num-batched-tokens` |
| **Prefill/decode disaggregation** | run prefill and decode on *different* GPUs, and hand the KV cache over | used in large deployments; see [DistServe](https://arxiv.org/abs/2401.09670) |

[Sarathi-Serve](https://arxiv.org/abs/2403.02310) is the paper that made chunked prefill the norm.

---

## 5. The throughput vs latency trade-off

Batching is not a free lunch forever. As the batch grows:

```
bigger batch  -->  more tokens per second for the SERVER  (cheaper per token)
              -->  a bit slower per token for EACH USER   (higher TPOT)
              -->  eventually: out of KV memory, requests queue or get preempted
```

You choose a point on that curve. A chat product wants low latency and accepts smaller batches. An
overnight job summarising a million documents wants maximum throughput and doesn't care about
latency — which is why providers offer cheaper **batch APIs** that return results hours later.

**Goodput** — requests per second that meet your latency targets — is often the metric that
matters: raw throughput that breaks your latency promise is not useful capacity.

---

## 6. Where you will meet it

**vLLM** — continuous batching is always on; you tune its limits:
```bash
vllm serve <model> \
  --max-num-seqs 64 \               # at most 64 requests in a batch
  --max-num-batched-tokens 8192 \   # at most 8192 tokens processed per step
  --enable-chunked-prefill          # don't let long prompts stall decoding
```

**Offline with vLLM** — hand it a whole list and it batches for you:
```python
from vllm import LLM, SamplingParams
llm = LLM(model="<model>")
outputs = llm.generate(list_of_1000_prompts, SamplingParams(max_tokens=200))
```

---

## 7. Common misconceptions

- **"Bigger batch is always better."** Only for throughput, and only until memory runs out.
  Per-user latency always rises a little, and sharply once you hit limits.
- **"Batching changes the model's answers."** It doesn't. Each request attends only to its own
  tokens. (Tiny floating-point differences between batch sizes can occasionally flip a
  near-tied token — one reason `temperature=0` is not perfectly reproducible on servers.)
- **"Batching is only for offline jobs."** Continuous batching is exactly how *interactive* chat
  services serve thousands of users at once.

---

## 8. Examples

```bash
python examples/batching_simulator.py      # no dependencies, runs instantly
```

[`batching_simulator.py`](examples/batching_simulator.py) contains:
1. the six-request timeline above, for both policies
2. a 400-request simulation comparing throughput, p50/p95 latency and wasted compute
3. how throughput and step time scale with batch size

Its assumptions are stated at the top of the file — change them and rerun.

---

## 9. Exercises

1. **Read the picture.** In the timeline, count the wasted slot-steps for static batching by hand
   and check the script's number.
2. **Find the breaking point.** Lower `ARRIVAL_RATE` step by step until static batching keeps up.
   How close does its throughput get to continuous batching? What stays worse?
3. **Remove the variance.** Make every output exactly 100 tokens. How much of static batching's
   problem disappears? What does that tell you about *why* it fails on chat traffic?
4. **Make batching expensive.** Raise `PER_SEQ` from 0.02 to 0.5 (a compute-bound model). Does
   continuous batching still win? By how much?
5. **Add a memory limit.** Give each request a KV cost of its length and cap total memory. What
   happens to continuous batching when the queue has more work than memory allows?
6. **Add prefill.** Give each request a prefill step costing `prompt_tokens / 100` and add a few
   10,000-token prompts. Measure the stutter for decoding requests, then implement chunked prefill.

---

## 10. Projects to build and test

### Beginner — Batch an offline job
Summarise 500 short documents with a local model (Ollama or vLLM), first one at a time, then with
batched requests.

**How to test it:** report total wall-clock time and tokens per second for both; outputs should be
equivalent in quality (spot-check 20).

### Intermediate — A continuous-batching scheduler
Write a scheduler around a real small model (Hugging Face `transformers`, CPU is fine) that admits
new requests between decode steps.

**How to test it:** send requests with varying lengths from several threads; every response must
match what the model produces when run alone, and short requests must not wait for long ones.

### Advanced — Load-test a real server
Run vLLM on a GPU and load-test it at increasing request rates with realistic prompt and output
lengths.

**How to test it:** plot throughput and p95 TTFT/TPOT against request rate for three values of
`--max-num-seqs`, and identify the setting that maximises goodput for a target of "p95 TPOT under
50 ms".

---

## 11. Resources

**Start here — beginner friendly**
- [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) — Anyscale. The article that made this widely understood; clear diagrams.
- [LLM Batching: Static vs Continuous and Why It Matters for Throughput](https://blog.premai.io/llm-batching-static-vs-continuous-and-why-it-matters-for-throughput/) — Prem AI.
- [LLM Inference: Continuous Batching and PagedAttention](https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/) — Insu Jang. Connects this module to module 04.

**Go deeper**
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) — Databricks, on batch size, latency and memory-bandwidth utilisation.
- [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/) — every batching knob, documented.
- [Understand LLM latency and throughput metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — Anyscale docs.

**Papers**
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022) — iteration-level scheduling
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) — chunked prefill
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670)

---

**Previous:** [01 · KV Cache](../01-kv-cache/) · **Next:** [03 · Quantization](../03-quantization/)
