# 04 · Paged Attention

> **In one sentence:** PagedAttention stores the KV cache in small fixed-size blocks that can sit
> anywhere in GPU memory, handed out only as a response grows, so almost no memory is wasted and
> far more requests fit on the same GPU.

It is the idea that launched [vLLM](https://vllm.ai/blog/2023-06-20-vllm), and it borrows a trick
operating systems have used since the 1960s.

---

## 1. The problem

From [module 01](../01-kv-cache/): every request needs KV-cache memory that grows with each token.
From [module 02](../02-batching/): the more requests fit in memory at once, the bigger the batch and
the higher the throughput. So **KV-cache memory is the bottleneck on throughput**.

And there is a nasty complication: **nobody knows how long an answer will be.** "What's 2+2?" might
get 3 tokens; "write an essay" might get 2,000.

Older serving systems handled that by reserving one **contiguous** chunk of memory per request,
sized for the worst case. That wastes memory three ways:

| Waste | What happens |
|---|---|
| **Over-reservation** | memory reserved for 2,000 tokens, answer uses 50; the rest sits empty for the whole request |
| **Internal fragmentation** | leftover space inside an allocation that no one else can use |
| **External fragmentation** | plenty of free memory in total, but split into holes too small for the next request |

The PagedAttention paper measured existing systems wasting **60–80%** of their KV-cache memory.

### An analogy

A restaurant that seats every party at a 12-person table "in case more friends turn up" — most
seats stay empty, and a new party of 8 gets turned away even though there are 20 empty seats
scattered around the room.

The fix: small 4-seat tables anywhere in the room, a seating chart saying which tables belong to
which party, and a new table added only when someone else actually arrives.

---

## 2. How PagedAttention works

Operating systems solved this exact problem with **virtual memory**: split memory into fixed-size
**pages**, give each program a **page table** mapping its addresses to physical pages that can be
anywhere. PagedAttention does the same for the KV cache:

```
GPU KV memory, carved into blocks (here 4 tokens each; vLLM defaults to 16)

 physical:  [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ][ 8 ][ 9 ][10 ][11 ]
              .   A1   B0    .   B1    .    .    .    .   A0   A2    .

block tables:
  chat-A (11 tokens):  logical 0 -> physical 9    [####]
                       logical 1 -> physical 1    [####]
                       logical 2 -> physical 10   [###-]    <- only waste: 1 slot
  chat-B ( 6 tokens):  logical 0 -> physical 2    [####]
                       logical 1 -> physical 4    [##--]    <- only waste: 2 slots
```

(This is real output from [`paged_kv_simulator.py`](examples/paged_kv_simulator.py).)

The rules:

1. **Allocate on demand.** A request gets a new block only when its current block fills up.
2. **Blocks need not be adjacent.** Any free block will do, so external fragmentation cannot happen.
3. **Free on finish.** When a request ends, its blocks go straight back to the pool.
4. **Waste is bounded.** At most one partly-filled block per request.

### Does attention still give the right answer?

Yes — exactly. Attention needs each token's key and value; it does not care where in memory they
live. A special attention kernel reads through the block table. The script checks this by computing
attention over scattered blocks and over ordinary contiguous tensors: the maximum difference is
**0.0**.

---

## 3. What it buys you

### Capacity — how many requests fit?

A 40,000-token KV budget, prompts of 100–1,500 tokens, answers of 20–1,000 tokens, measured once
every admitted request has produced its full answer:

| Strategy | Requests that fit | KV actually used | Wasted |
|---|---|---|---|
| Reserve the model's max length (4,096) | 9 | 13,012 | 65% |
| Reserve prompt + max new tokens (1,024) | 21 | 28,228 | 27% |
| **Paged, blocks of 16** | **29** | **38,847** | **1%** |

**3.2× more concurrent requests** from the same memory.

### A live simulation — requests arriving and leaving

600 requests, one arriving every 4 decode steps:

| Allocator | Steps to finish | Average batch | KV memory in use | Fragmentation stalls | Preemptions |
|---|---|---|---|---|---|
| Contiguous, first-fit | 17,134 | 18.0 | 51% | 16,045 | 0 |
| **Paged (vLLM-style)** | **9,520** | **32.9** | **93%** | **0** | 465 |

- **Fragmentation stalls** are steps where the next request could not start *even though enough
  memory was free in total* — it was just in pieces. Paging makes this impossible.
- **Preemptions** are the price paging pays instead. Because it admits requests optimistically,
  memory can run out mid-generation; the scheduler then evicts the newest request and recomputes
  it later. Occasional recomputation is far cheaper than leaving memory idle.

Paging runs 1.8× more requests at once and finishes the same work in 56% of the steps.

---

## 4. Bonus: sharing memory between requests

Because blocks are referenced through tables, **two requests can point at the same physical block.**
That opens up free savings:

### Prefix sharing

Most apps send the same long system prompt with every request. From the script — 100 chats sharing
a 1,210-token system prompt, each adding 50 user tokens and a 200-token answer:

| | Blocks |
|---|---|
| Without sharing | 100 × 92 = **9,200** |
| With sharing | 75 shared + 100 × 17 private = **1,775** |
| **Saved** | **81%** |

The 75 full blocks of the system prompt are stored **once**, with a reference count of 100. The last
10 prompt tokens sit in a partly-filled block; when a request writes its own token there, it gets a
private copy — **copy-on-write**, exactly as in an operating system.

Shared blocks also skip **prefill** — their keys and values are already computed — so time to first
token drops as well. vLLM calls this **automatic prefix caching**; SGLang's version is
**RadixAttention**.

> **Practical tip:** put stable content (system prompt, tool definitions, few-shot examples) at the
> **start** of your prompt and the changing part (the user's question) at the **end**. Only a
> matching prefix can be reused.

### Parallel sampling and beam search

Asking for 5 alternative answers to one prompt? All 5 share the prompt's blocks and only diverge
once they start generating. The same applies to the many candidate sequences in beam search.

---

## 5. Where you will meet it

**vLLM** — paging is always on:
```bash
vllm serve <model> \
  --gpu-memory-utilization 0.90 \   # memory vLLM may use; what the weights leave becomes KV blocks
  --block-size 16 \                 # tokens per block
  --enable-prefix-caching           # share blocks between requests with the same prefix
```
When vLLM starts, it logs how much KV-cache capacity it has — that number is the block pool.

**SGLang** uses paged KV memory plus RadixAttention, which organises cached prefixes as a tree so
that many requests with partly-overlapping prompts (agents, multi-turn chat) share as much as
possible.

---

## 6. Common misconceptions

- **"PagedAttention makes attention faster."** Not directly — reading through a block table adds a
  little overhead. The win is **memory**, which allows much **larger batches**, which is where the
  throughput comes from.
- **"It changes the model's output."** No. The script shows a difference of exactly 0.0.
- **"Bigger blocks are better."** Bigger blocks mean fewer table lookups but more waste in each
  request's last block, and coarser prefix sharing. Small blocks (16 tokens) are the usual balance.
- **"Prefix caching remembers my conversation."** Only while the blocks are still in memory. It is
  an optimisation, not storage; cached blocks are evicted when memory is needed.

---

## 7. Examples

```bash
pip install torch
python examples/paged_kv_simulator.py       # a few seconds on CPU
```

[`paged_kv_simulator.py`](examples/paged_kv_simulator.py):
1. builds a real paged KV cache and prints its block tables
2. proves attention over scattered blocks matches contiguous attention exactly
3. compares how many requests fit under three reservation strategies
4. runs a live simulation with arrivals, departures, fragmentation and preemption
5. calculates the savings from prefix sharing with copy-on-write

---

## 8. Exercises

1. **Draw it.** Using the block table printed by the script, draw the physical pool and label which
   request owns each block. Then free chat-A and redraw.
2. **Change the block size.** Set `PAGE` to 1, 16, 64 and 256 in section 3. Plot requests that fit
   against block size. Why is 1 not the best choice in a real system? (Hint: the table itself costs
   memory and lookups.)
3. **Break it.** In `PagedKVCache.read`, shuffle the order of `self.tables[seq]` before gathering.
   What happens to the attention output, and what does that tell you about the block table's job?
4. **Stress preemption.** Halve `POOL_TOKENS` in the live simulation. How do preemptions and total
   steps change? At what point does paging's optimistic admission start to hurt?
5. **Prefix sharing arithmetic.** How much would 100 chats save if each had a *different*
   1,210-token system prompt? If the system prompt were exactly 1,200 tokens?
6. **Prompt design.** Take a prompt template from your own project and reorder it so the maximum
   amount is a stable prefix. How many tokens become cacheable?

---

## 9. Projects to build and test

### Beginner — Block allocator with a visualiser
Build a block allocator (allocate, append token, free) and draw the pool as a grid in the terminal
after every operation.

**How to test it:** random operations for 10,000 steps — no block may ever be owned by two
sequences, no block may leak, and every sequence's tokens must read back in order.

### Intermediate — Copy-on-write prefix cache
Add reference counts and copy-on-write to the allocator, plus a lookup that finds the longest cached
prefix of a new prompt (hash each full block's tokens together with its prefix).

**How to test it:** 100 prompts sharing various prefixes — memory used must match your hand
calculation, and modifying one sequence's tokens must never change another's.

### Advanced — Measure prefix caching on a real server
Serve a model with vLLM with and without `--enable-prefix-caching`, and send requests that share a
long system prompt.

**How to test it:** report TTFT and throughput for both at several request rates. The cached
version's TTFT gain should grow with the length of the shared prefix — plot that relationship.

---

## 10. Resources

**Start here — beginner friendly**
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm) — the original vLLM blog post, with animations of blocks being allocated.
- [How PagedAttention resolves memory waste of LLM systems](https://developers.redhat.com/articles/2025/07/24/how-pagedattention-resolves-memory-waste-llm-systems) — Red Hat Developer.
- [Introduction to vLLM and PagedAttention](https://www.runpod.io/blog/introduction-to-vllm-and-pagedattention) — RunPod.

**Go deeper**
- [Paged Attention from First Principles: A View Inside vLLM](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/) — a detailed walk through the internals.
- [LLM Inference: Continuous Batching and PagedAttention](https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/) — Insu Jang. How this module and module 02 work together.
- [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) — vLLM docs.
- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) — NVIDIA's PagedAttention section.

**Papers**
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — the vLLM paper; sections 3–4 are very readable
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) — RadixAttention

---

**Previous:** [03 · Quantization](../03-quantization/) · **Next:** [05 · Speculative Decoding](../05-speculative-decoding/)
