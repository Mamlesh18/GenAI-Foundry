# 02 · SGLang

> **In one sentence:** SGLang is a serving engine like vLLM, built around one distinctive idea —
> it keeps every cached prefix in a **radix tree**, so requests that start with the same text reuse
> each other's work automatically.

If your traffic has long shared prefixes — one system prompt for thousands of chats, agents that
resend their tool definitions, multi-turn conversations — this is the engine to measure.

---

## 1. Architecture: three processes in a line

Where [vLLM](../01-vllm/) splits into an API server process and an engine core process, SGLang
splits the work into three roles connected by ZeroMQ:

```
   HTTP requests
        |
        v
 +---------------------------------------------------------------+
 |  MAIN PROCESS                                                  |
 |    FastAPI HTTP server                                         |
 |    TOKENIZER MANAGER -- turns text into token ids, tracks      |
 |                         each request, streams results out      |
 +---------------------------------------------------------------+
        |   ZeroMQ
        v
 +---------------------------------------------------------------+
 |  SCHEDULER  (subprocess; one per tensor-parallel worker)       |
 |                                                                |
 |    +---------------------+    +-----------------------------+  |
 |    | batch scheduling    |<-->|  RADIX CACHE (prefix tree)  |  |
 |    | continuous batching |    |  longest-prefix match,      |  |
 |    | cache-aware order   |    |  LRU eviction               |  |
 |    +---------------------+    +-----------------------------+  |
 |                |                                               |
 |                v                                               |
 |    model forward pass on the GPU(s)                            |
 +---------------------------------------------------------------+
        |   ZeroMQ (output token ids)
        v
 +---------------------------------------------------------------+
 |  DETOKENIZER MANAGER (subprocess)                              |
 |    token ids -> text, incrementally, then back to the          |
 |    tokenizer manager, which streams it to the client           |
 +---------------------------------------------------------------+
```

| Process | Job |
|---|---|
| **Tokenizer manager** | Tokenizes incoming requests, keeps per-request state, returns results to clients |
| **Scheduler** | The engine: forms batches, consults the radix cache, runs the model |
| **Detokenizer manager** | Converts output token ids back to text (incremental decoding is fiddly — hence its own process) |

Splitting tokenization and detokenization into separate processes keeps that CPU work off the
critical path of the GPU loop — the same motivation behind vLLM's multi-process design.

---

## 2. RadixAttention: the distinctive part

Both engines cache prefixes. The difference is the **data structure**.

vLLM hashes fixed-size blocks and looks them up. SGLang keeps a **radix tree** (a trie where
single-child chains are merged) whose keys are token sequences and whose values are the cached KV
blocks:

```
                      [ "You are a helpful assistant. Tools: ..." ]   <- 1,200 tokens, stored ONCE
                       /                    |                     \
        [ "Book a flight" ]      [ "Cancel my order" ]     [ "What is my balance" ]
           /          \
   [ "...to Delhi" ]  [ "...to Chennai" ]
```

When a request arrives, SGLang walks the tree to find the **longest matching prefix**, reuses those
KV blocks, and computes only the new part. Benefits:

- **Shorter TTFT** — the shared prefix is not re-prefilled.
- **More memory for everyone** — one copy of the shared prefix instead of hundreds.
- **It is automatic** — no cache keys to manage in your application.

Two supporting pieces make it work in practice:

- **LRU eviction** — when memory runs short, least-recently-used leaves are evicted first, so
  popular prefixes (your system prompt) survive.
- **Cache-aware scheduling** — requests that share a prefix are preferentially scheduled together,
  which raises the hit rate instead of leaving it to luck.

**Where this shines:** agents (same tool definitions every call), multi-turn chat (the conversation
so far is a prefix of the next turn), few-shot prompting, evaluating many prompts against one long
instruction. **Where it doesn't:** every request genuinely different, e.g. one-off document
summarisation.

> **Practical tip, and it applies to every engine:** put the stable part of your prompt (system
> instructions, tools, examples) at the **start**, and the variable part at the **end**. Only a
> matching *prefix* can be reused.

---

## 3. The other thing it is known for

SGLang began as a *language* for structured LLM programs, not only a server. That heritage shows up
as strong support for **constrained decoding**: forcing output to match a JSON schema, a regular
expression or a grammar, by masking out tokens that would break the format during sampling.

Compared with asking politely for JSON in the prompt
([03 · Structured Outputs](../../skills/03-structured-outputs/)), this makes malformed output
*impossible* rather than unlikely, which matters when the output feeds code.

---

## 4. Running it

```bash
# launch the server (default port 30000)
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct

# the same client from the track examples talks to it unchanged
python ../examples/openai_client.py --base-url http://127.0.0.1:30000/v1
```

Installation options (pip, source, Docker, and per-hardware notes) are in the
[install guide](https://docs.sglang.io/).

| Flag | Meaning |
|---|---|
| `--tp` (`--tensor-parallel-size`) | Split the model across GPUs — see [module 03](../03-distributed-inference/) |
| `--dp` (`--data-parallel-size`) | Run several replicas; use the SGLang router/gateway in front |
| `--mem-fraction-static` | Fraction of GPU memory for weights + KV pool. **Lower this first when you hit OOM** |
| `--chunked-prefill-size` | Tokens per prefill chunk; `-1` disables chunked prefill |
| `--disable-radix-cache` | Turn the prefix cache off — useful to measure what it is worth |
| `--nnodes`, `--node-rank`, `--dist-init-addr` | Multi-node setup |
| `--port` | Default `30000` |

---

## 5. vLLM or SGLang?

They have converged a lot: both do continuous batching, paged memory, prefix caching, quantization,
speculative decoding and multi-GPU serving.

| Situation | Leaning |
|---|---|
| General serving, widest model and hardware coverage, biggest community | **vLLM** |
| Long shared prefixes: agents, multi-turn chat, few-shot evaluation | **SGLang** |
| Heavy structured output (JSON schema, grammars) | **SGLang** |
| You want the most documentation and examples to learn from | **vLLM** |

**The honest answer:** benchmark both on *your* traffic with *your* model. Published benchmarks
change with every release and rarely match your workload. Measure TTFT, TPOT and throughput at your
real request rate and prefix-sharing pattern.

---

## 6. Exercises

*(1–3 need no GPU.)*

1. **Draw your own tree.** Take three prompts your application would send. Draw the radix tree they
   would form. How many tokens are shared, and how many would be recomputed per request without
   the cache?
2. **Estimate the win.** 500 requests/hour, a 1,500-token shared system prompt, 100-token questions.
   How many prefill tokens per hour does prefix caching save?
3. **Reorder a prompt.** Find a prompt template in your own project that puts variable content
   early. Rewrite it so the stable part comes first, and count how many more tokens become
   cacheable.
4. **Measure the cache.** Run SGLang with and without `--disable-radix-cache` on a workload with a
   shared prefix. Compare TTFT. Then rerun with all-different prompts — does the gap disappear?
5. **Compare engines.** Serve the same model with vLLM and SGLang, and run the same load test
   against both. Report TTFT, TPOT and throughput, and say which you would deploy and why.

---

## 7. Projects to build and test

### Beginner — Radix tree from scratch
Implement insertion and longest-prefix lookup over token lists, plus LRU eviction with a memory cap.

**How to test it:** insert 1,000 synthetic prompts sharing several system prompts; assert that
lookups return the longest match, that total stored tokens are far below the naive sum, and that
eviction never removes a node with children still in use.

### Intermediate — Cache hit-rate simulator
Replay a realistic request trace (agent-style, with a fixed prefix) through your tree and report
hit rate, tokens saved and memory used as the cache size varies.

**How to test it:** hit rate should rise with cache size and fall as prompts become more varied;
plot both curves and explain the shape.

### Advanced — Prefix-aware routing
With two replicas, route each request to the one most likely to already hold its prefix, instead of
round-robin.

**How to test it:** measure cache hit rate and TTFT against round-robin on the same trace. The win
should grow with prefix length — quantify it, and note the load-imbalance risk your router creates.

---

## 8. Resources

**Start here**
- [SGLang documentation](https://docs.sglang.io/) — install, quickstart, feature list.
- [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) — the original announcement, with clear diagrams of the radix tree.
- [Server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) — every flag, with defaults.

**Go deeper**
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) — the paper; RadixAttention is section 3.
- [SGLang code walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme.md) — a community tour of the processes in section 1.
- [SGLang production deployment guide](https://www.spheron.network/blog/sglang-production-deployment-guide/) — practical multi-turn and RadixAttention notes.

**Related in this repo**
- [03-inference/04 · Paged Attention](../../03-inference/04-paged-attention/) — prefix sharing and copy-on-write, with a runnable simulation
- [skills/03 · Structured Outputs](../../skills/03-structured-outputs/) — why constrained decoding matters

---

**Previous:** [01 · vLLM](../01-vllm/) · **Next:** [03 · Distributed Inference](../03-distributed-inference/)
