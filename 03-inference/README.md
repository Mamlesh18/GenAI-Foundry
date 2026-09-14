# 03 · Inference

Training builds a model once. **Inference** is everything that happens every single time someone
uses it: a prompt goes in, text comes out. It runs millions of times a day, so it is where most of
the cost, the waiting and the engineering actually live.

This track answers one practical question: **why is generating text slow and expensive, and what
do real systems do about it?** No GPU needed: every module has a small script that runs on a
laptop CPU and shows the idea working.

---

## 1. The life of one request

When you send "Explain black holes" to a chatbot, this happens:

```
 "Explain black holes"
        |
   [ tokenize ]                 text -> token ids              (see 01-foundations/05)
        |
   [ PREFILL ]                  read the WHOLE prompt in one parallel pass
        |                       and store every token's keys & values (the KV cache)
        |                       ---> produces the FIRST output token
        v
   [ DECODE ] <----+            take the newest token, run ONE forward pass,
        |          |            read the KV cache, produce the next token
        +----------+            repeat until an end token or max_tokens
        |
   [ detokenize ]               token ids -> text, streamed back to you
        |
 "A black hole is a region..."
```

The two phases behave completely differently, and almost every trick in this track targets one of
them:

| | **Prefill** | **Decode** |
|---|---|---|
| What it processes | all prompt tokens, at once | one new token per pass |
| Speed limited by | **compute** (arithmetic) | **memory bandwidth** (moving weights and cache) |
| GPU busy? | very | mostly waiting for memory |
| Sets which metric | time to first token | time per output token |
| Grows with | prompt length | answer length |

The script in module 01 measures this on a CPU: pushing 512 tokens through a small model in one
prefill pass took **49 ms**; pushing the same 512 tokens one at a time took **2,239 ms** — about
46× longer for the same work.

### Why decode is "memory-bound"

To produce each new token, the model reads essentially all of its weights once. A 16 GB model
means moving roughly 16 GB of data from memory to the processor *per token*. Modern GPUs can
multiply far faster than their memory can deliver bytes, so the arithmetic sits idle while waiting.

That one fact explains most of this track:

- **Quantization** makes the weights smaller, so less to move per token → faster
- **Batching** serves many users in the same pass, using the idle compute for free
- **Speculative decoding** checks several tokens per pass instead of producing one
- **KV cache + paged attention** avoid recomputing, and fit more users into memory

Horace He's [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) is the classic
explanation of compute-bound vs memory-bound vs overhead-bound work.

---

## 2. How we measure it

| Metric | What it means | Users feel it as |
|---|---|---|
| **TTFT** — time to first token | from sending the request to the first token arriving | "is it responding?" |
| **TPOT** — time per output token (also *ITL*, inter-token latency) | average gap between tokens while it streams | reading speed |
| **Latency** | total time for the full answer ≈ TTFT + TPOT × output tokens | "how long did that take?" |
| **Throughput** | output tokens per second across **all** users | the server's cost per token |
| **Goodput** | requests per second that meet your latency targets | throughput that actually counts |

A worked example: TTFT 0.4 s, TPOT 25 ms, 400-token answer →
latency ≈ 0.4 + 0.025 × 400 = **10.4 s**. Streaming makes that feel fast, because reading starts
after 0.4 s.

**The central trade-off:** throughput and per-user latency pull in opposite directions. Bigger
batches serve more users per GPU (cheaper) but each user's tokens arrive a little slower. Every
serving decision is a position on that curve.

> **Measure it yourself now.** Run
> [`01-foundations/02-llms/examples/03_streaming_and_params.py`](../01-foundations/02-llms/examples/03_streaming_and_params.py) —
> it prints time-to-first-token and total time for a real API call.

---

## 3. The modules

| # | Module | The problem it solves | What the script shows |
|---|---|---|---|
| 01 | [KV Cache](01-kv-cache/) | Without it, every token re-reads the whole conversation | Same output, 6.9× faster at 512 tokens; the memory bill for real models |
| 02 | [Batching](02-batching/) | One user at a time wastes most of the GPU | Continuous vs static batching: 3.4× throughput, ~90× lower median latency |
| 03 | [Quantization](03-quantization/) | Models are too big to fit, and too slow to read | Why 4-bit needs small groups; why activations are harder than weights |
| 04 | [Paged Attention](04-paged-attention/) | The KV cache wastes most of its reserved memory | Block tables, fragmentation, prefix sharing |
| 05 | [Speculative Decoding](05-speculative-decoding/) | One pass of the big model produces only one token | A lossless 2–4× speedup, verified statistically |

**Suggested order:** 01 → 02 → 04 → 03 → 05. Paged attention builds directly on the KV cache and
batching; quantization and speculative decoding stand on their own.

---

## 4. How it all fits together in a real server

You rarely implement these yourself. You pick a serving engine and turn knobs. Here is where each
module shows up in [vLLM](https://docs.vllm.ai/en/latest/configuration/engine_args/), the most
widely used open-source engine:

| Module | vLLM flag | What it controls |
|---|---|---|
| KV cache | `--gpu-memory-utilization` | share of GPU memory vLLM takes; what the weights don't use becomes KV cache |
| KV cache | `--max-model-len` | the longest context a request may use |
| KV cache | `--kv-cache-dtype fp8` | store the cache in 8 bits: about twice the users |
| Batching | `--max-num-seqs` | how many requests run in one batch |
| Batching | `--max-num-batched-tokens` | how many tokens are processed per step |
| Batching | `--enable-chunked-prefill` | split long prompts so they don't stall everyone else |
| Paged attention | `--block-size` | tokens per KV block |
| Paged attention | `--enable-prefix-caching` | reuse cached blocks for repeated prefixes (system prompts) |
| Quantization | `--quantization` | which weight format to load (AWQ, GPTQ, FP8, ...) |
| Speculative decoding | `--speculative-config` | draft model or method, and tokens to speculate |

Defaults change between versions — always check `vllm serve --help` for yours.

### The engines you will meet

| Engine | Best for |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | GPU serving for many users; the default choice |
| [SGLang](https://github.com/sgl-project/sglang) | GPU serving, especially heavy prefix reuse and structured output |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | squeezing maximum speed from NVIDIA hardware |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | CPUs, Macs, consumer GPUs; GGUF models |
| [Ollama](https://ollama.com) | the easiest way to run a model on your own laptop |

Serving them properly is covered in [04 · Serving](../04-serving/).

---

## 5. Glossary

| Term | Plain meaning |
|---|---|
| **Forward pass** | running the model once over some tokens |
| **Autoregressive** | each output token becomes part of the input for the next one |
| **Prefill** | the first pass, over the whole prompt |
| **Decode** | every pass after that, one new token each |
| **KV cache** | stored keys and values of past tokens, so they are not recomputed |
| **Batch** | several requests processed in the same forward pass |
| **Memory bandwidth** | how many GB per second memory can deliver to the processor |
| **Quantization** | storing numbers in fewer bits (e.g. 16 → 4) |
| **Block / page** | a fixed-size chunk of KV-cache memory |
| **Draft model** | a small, fast model that guesses tokens for a big one to check |
| **p50 / p95 latency** | the median / the 95th-percentile request's latency — the slow tail matters |

---

## 6. Exercises for the whole track

1. **Feel prefill vs decode.** Using any API, send a 50-token prompt and a 5,000-token prompt,
   each asking for a 20-token answer. Compare TTFT. Then send a short prompt asking for a
   1,000-token answer and compare total latency. Which knob moved which metric?
2. **Do the arithmetic.** An 8B model in fp16 on a GPU with 1,000 GB/s memory bandwidth: what is
   the upper bound on single-user tokens per second? What changes at 4-bit?
3. **Read a real config.** Open `config.json` for any model on Hugging Face and calculate its KV
   cache per token with the module 01 calculator.
4. **Map the flags.** Run `vllm serve --help` (or read the engine arguments page) and find five
   flags not listed above. For each, say which module's idea it controls.

## 7. Projects

- **Beginner — Latency dashboard.** Call two API providers 50 times each with the same prompt,
  record TTFT, TPOT and total latency, and plot p50 and p95. *Test:* your latency should roughly
  equal TTFT + TPOT × tokens for every request; investigate any that don't.
- **Intermediate — Local benchmark.** Run the same model with Ollama at two quantization levels
  and measure tokens per second and answer quality on 20 fixed prompts. *Test:* compare your speed
  against the memory-bandwidth rule from module 03 and explain the gap.
- **Advanced — Serve and load-test.** Run vLLM on a rented or free GPU, then load-test it at
  increasing request rates. *Test:* plot throughput and p95 latency against request rate, find the
  point where latency climbs sharply, and show how `--max-num-seqs` moves it.

---

## 8. Resources

**Start here — beginner friendly**
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) — Databricks. The clearest introduction to TTFT, TPOT, throughput and why decode is memory-bound.
- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) — NVIDIA. One article touring every technique in this track.
- [Optimizing your LLM in production](https://huggingface.co/blog/optimize-llm) — Hugging Face. Precision, FlashAttention, KV cache and GQA with runnable code.
- [LLM inference latency: TTFT, tokens per second, and what to measure](https://clickhouse.com/resources/engineering/llm-inference-latency) — ClickHouse. Practical metrics guide.

**Go deeper**
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) — Horace He. Compute vs memory vs overhead; the mental model behind everything here.
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) — kipply. The back-of-envelope maths for memory and speed.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — Lilian Weng. A survey of quantization, pruning, sparsity and distillation.
- [Understand LLM latency and throughput metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — Anyscale docs.

**Papers that shaped modern serving**
- [Orca: iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022) — continuous batching
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310) — chunked prefill and the throughput–latency trade-off
- [DistServe](https://arxiv.org/abs/2401.09670) — running prefill and decode on separate GPUs; where "goodput" comes from

---

**Previous track:** [02 · Training](../02-training/) · **Next track:** [04 · Serving](../04-serving/)
