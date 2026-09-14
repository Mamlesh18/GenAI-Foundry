# 01 · KV Cache

> **In one sentence:** the KV cache stores the attention keys and values of every token already
> processed, so each new token only computes its own instead of recomputing the entire
> conversation.

It is the single most important inference optimisation. Every LLM you have used runs with one.

---

## 1. The problem, with an analogy

Imagine writing a story one word at a time, with a rule: before writing each new word, you must
re-read the whole story from the beginning. Word 10 costs 10 words of reading. Word 1,000 costs
1,000. The total work grows with the *square* of the length.

That is exactly what a language model does without a cache. Generation is **autoregressive**:
each new token is appended to the input and the model runs again. And inside that run, attention
(from [01-foundations/01](../../01-foundations/01-attention-is-all-you-need/)) needs a **key** and
**value** vector for every earlier token.

But the keys and values of earlier tokens **never change**. Token 5's key is the same whether we
are generating token 6 or token 600. Recomputing them is pure waste.

**The fix:** keep a notebook. Write down each token's key and value the first time you compute
them, and just read the notebook afterwards. That notebook is the KV cache.

---

## 2. How it works, step by step

```
PREFILL  -- prompt: "The cat sat"
  compute K,V for "The", "cat", "sat"   ->  cache = [The, cat, sat]
  produce first new token: "on"

DECODE step 1 -- input is ONLY "on"
  compute K,V for "on"                  ->  cache = [The, cat, sat, on]
  query of "on" attends over all 4 cached K,V
  produce: "the"

DECODE step 2 -- input is ONLY "the"
  compute K,V for "the"                 ->  cache = [The, cat, sat, on, the]
  produce: "mat"
```

Each decode step pushes **one** token through the model instead of the whole sequence.

### Why keys and values, but not queries?

The query asks *"what should I, the new token, pay attention to?"* Only the newest token needs to
ask that — past tokens already asked their questions when they were generated. Keys and values
answer *"what do I contain?"*, and every future token needs to read those. So K and V are cached;
Q is thrown away after each step.

### What the cache is not

- **It does not change the output.** Same tokens, same probabilities — only faster. Our script
  checks this: 512 generated tokens identical both ways, with the largest logit difference
  **1.7 × 10⁻⁶** (floating-point noise).
- **It does not speed up prefill.** Prefill *builds* the cache. The cache speeds up decode.
- **It is not remembered between API calls.** Each request builds its own. (Reusing cache across
  requests with the same prefix is a separate trick — see [04 · Paged Attention](../04-paged-attention/).)

---

## 3. The numbers

From [`kv_cache_from_scratch.py`](examples/kv_cache_from_scratch.py) — a small 4-layer model,
32-token prompt, on a laptop CPU:

| New tokens | Positions processed, no cache | With cache | Time, no cache | With cache | Speedup |
|---|---|---|---|---|---|
| 64 | 4,064 | 95 | 0.76 s | 0.34 s | 2.2× |
| 256 | 40,832 | 287 | 5.60 s | 1.24 s | 4.5× |
| 512 | 147,200 | 543 | 14.86 s | 2.15 s | **6.9×** |

The work without a cache grows **quadratically**; with a cache, **linearly**. The longer the
answer, the bigger the win. (Your timings will differ; the positions column will not.)

---

## 4. The price: memory

Nothing is free. The cache must sit in GPU memory for the whole request.

```
KV cache bytes per token = 2 × layers × kv_heads × head_dim × bytes per number
                           ^
                           one key + one value
```

**Worked example — Llama 3.1 8B** (32 layers, 8 KV heads, head dim 128, fp16 = 2 bytes):

```
2 × 32 × 8 × 128 × 2 = 131,072 bytes  ≈ 128 KB per token
× 32,000 tokens      ≈ 3.9 GB for ONE long conversation
```

From [`kv_cache_calculator.py`](examples/kv_cache_calculator.py):

| Model | Per token | 4K tokens | 32K tokens | 128K tokens |
|---|---|---|---|---|
| Qwen2.5 7B | 56 KB | 0.21 GB | 1.71 GB | 6.84 GB |
| Llama 3.1 8B | 128 KB | 0.49 GB | 3.91 GB | 15.62 GB |
| Llama 3.1 70B | 320 KB | 1.22 GB | 9.77 GB | 39.06 GB |

Multiply by the number of users served at once and the cache — not the weights — becomes the thing
that decides how many people one GPU can serve. The calculator estimates that for an 8B model on an
80 GB GPU: about **116 users at 4K tokens**, or **14 at 32K**.

---

## 5. Making the cache smaller

Because memory is the limit, a whole family of techniques exists just to shrink the cache.

| Technique | Idea | Saving | Used by |
|---|---|---|---|
| **GQA** — grouped-query attention | several query heads share one set of K/V | 4–8× | Llama 3, Mistral, Qwen, most modern models |
| **MQA** — multi-query attention | *all* query heads share one K/V | up to 32× | some older models; quality cost |
| **MLA** — multi-head latent attention | store a compressed latent, rebuild K/V | large | DeepSeek-V2 and later |
| **KV cache quantization** | store K/V in 8 or even 2 bits | 2–8× | vLLM `--kv-cache-dtype fp8`, KIVI |
| **Sliding window / attention sinks** | keep only recent tokens (plus the first few) | bounded size | Mistral's sliding window, StreamingLLM |
| **Offloading** | keep cache in CPU RAM, move layers in as needed | GPU memory | Hugging Face `cache_implementation="offloaded"` |
| **Paged attention** | don't waste reserved memory | 2–4× more users | vLLM, SGLang — [module 04](../04-paged-attention/) |

GQA is the one to understand first. For the Llama 3.1 8B shape at 32K tokens:

| Attention type | KV heads | Cache |
|---|---|---|
| MHA (every head has its own K/V) | 32 | 15.6 GB |
| **GQA (Llama 3's choice)** | **8** | **3.9 GB** |
| MQA (one shared K/V) | 1 | 0.5 GB |

Same 32 query heads in all three. GQA keeps most of MHA's quality at a quarter of the memory, which
is why nearly every recent model uses it.

---

## 6. Where you will meet it

**Hugging Face `transformers`** — on by default:
```python
model.generate(**inputs, max_new_tokens=100)                     # uses a DynamicCache
model.generate(**inputs, max_new_tokens=100, use_cache=False)    # turn it off, feel the pain
model.generate(**inputs, cache_implementation="static")          # fixed size, works with torch.compile
model.generate(**inputs, cache_implementation="offloaded")       # low GPU memory
```

**vLLM** — the cache is sized from whatever memory the weights leave:
```bash
vllm serve <model> --gpu-memory-utilization 0.90 --max-model-len 8192 --kv-cache-dtype fp8
```

---

## 7. Examples

```bash
pip install torch
python examples/kv_cache_from_scratch.py     # ~40 s on CPU
python examples/kv_cache_calculator.py       # instant, no dependencies
```

| Script | Shows |
|---|---|
| [`kv_cache_from_scratch.py`](examples/kv_cache_from_scratch.py) | A tiny transformer generating with and without a cache: identical output, quadratic vs linear work, cache size, and prefill vs decode timing |
| [`kv_cache_calculator.py`](examples/kv_cache_calculator.py) | Cache size for real models, MHA vs GQA vs MQA, and how many users fit on a GPU |

---

## 8. Exercises

1. **Prove it's lossless.** In `kv_cache_from_scratch.py`, switch greedy `argmax` to sampling with
   a fixed seed for both paths. Are the sequences still identical? Why must they be?
2. **Break the cache on purpose.** Cache the *queries* too and try to use them. What goes wrong,
   and what does that tell you about why only K and V are stored?
3. **Off-by-one hunt.** Change `start_pos` in the decode step to `seq.shape[1]`. Run it. The
   outputs diverge — explain exactly why position matters.
4. **Add a model.** Pick any model on Hugging Face, read its `config.json`, add it to the
   calculator, and work out the cache for 50 users at 8K tokens each.
5. **GQA trade-off.** If a 70B model used MHA instead of GQA, how many 32K-token users would fit in
   the KV budget of two 80 GB GPUs? Compare with the calculator's GQA number.
6. **Watch decode slow down.** Time a single decode step at cache lengths 64, 512 and 2,048 in the
   from-scratch script. Why does each step get a little slower even though only one token goes in?

---

## 9. Projects to build and test

### Beginner — KV cache calculator web page
Turn the calculator into a small web page: pick a model, context length, users, GPU and cache dtype.

**How to test it:** your per-token numbers must match the formula by hand for three models, and
your "tokens that fit" estimate should land close to the KV-cache capacity vLLM reports in its
startup logs for the same model, GPU and `--gpu-memory-utilization`. Explain any gap.

### Intermediate — Add a cache to nanoGPT
Take [nanoGPT](https://github.com/karpathy/nanoGPT)'s `generate()` (which has no cache) and add one.

**How to test it:** generated tokens must be identical with and without the cache for 5 prompts;
report the speedup at 100, 500 and 1,000 tokens and check it grows with length.

### Advanced — Sliding-window cache with attention sinks
Implement a cache that keeps the first 4 tokens plus the most recent N, as in StreamingLLM.

**How to test it:** generate 10× beyond N tokens without memory growing, and compare perplexity on
long text against a plain sliding window that drops the first tokens — the sinks version should
stay stable where plain windowing degrades.

---

## 10. Resources

**Start here — beginner friendly**
- [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) — Sebastian Raschka. The best from-scratch walkthrough, with readable code.
- [KV Caching in LLMs: A Guide for Developers](https://machinelearningmastery.com/kv-caching-in-llms-a-guide-for-developers/) — MachineLearningMastery.
- [Cache strategies](https://huggingface.co/docs/transformers/en/kv_cache) — Hugging Face docs: dynamic, static, offloaded and quantized caches with code.

**Go deeper**
- [Optimizing your LLM in production — Key-Value Cache section](https://huggingface.co/blog/optimize-llm) — Hugging Face, including MQA and GQA.
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) — kipply. The memory maths behind the formula.
- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) — NVIDIA, KV cache and attention variants.

**Papers**
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — multi-query attention
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — grouped-query attention
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — multi-head latent attention
- [KIVI: 2-bit KV cache quantization](https://arxiv.org/abs/2402.02750)
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) — StreamingLLM

---

**Track:** [03 · Inference](../) · **Next:** [02 · Batching](../02-batching/)
