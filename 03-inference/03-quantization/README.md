# 03 · Quantization

> **In one sentence:** quantization stores a model's numbers with fewer bits — 16-bit floats
> become 8-bit or 4-bit integers — so the model takes less memory and, because decoding is limited
> by memory speed, generates text faster.

It is the reason a model that "needs" a datacentre GPU can run on your laptop.

---

## 1. The idea, with an analogy

A shop's price list says ₹249.37, ₹1,099.82, ₹18.05. Round everything to the nearest rupee and
the list is shorter to write, easy to read, and almost nobody notices the difference. Round to the
nearest ₹100 and ₹18.05 becomes ₹0 — now you notice.

Quantization is rounding for model weights. The questions are always the same: **how coarse can
the rounding be before the model notices, and how do we round cleverly?**

---

## 2. Why it matters for inference

**It makes models fit.** An 8B model in fp16 is 16 GB — too big for most consumer GPUs. At 4-bit
it is about 4 GB.

**It makes generation faster.** From the [track introduction](../): each generated token reads
roughly every weight once, so single-user decode speed is capped by

```
tokens per second  ≲  memory bandwidth (GB/s)  ÷  model size (GB)
```

From [`weight_quantization_demo.py`](examples/weight_quantization_demo.py), upper bounds for an 8B
model with one user:

| Format | Size | PC, DDR5 dual-ch (~90 GB/s) | Apple M2 Ultra (800 GB/s) | RTX 4090 (1,008 GB/s) | A100 80GB (1,935 GB/s) |
|---|---|---|---|---|---|
| fp16 | 16.0 GB | 6 tok/s | 50 tok/s | 63 tok/s | 121 tok/s |
| int8 / FP8 | 8.0 GB | 11 tok/s | 100 tok/s | 126 tok/s | 242 tok/s |
| int4, group 128 | 4.2 GB | 21 tok/s | 188 tok/s | 237 tok/s | 455 tok/s |

Real speeds are lower (compute, KV-cache reads, overhead) — but the *pattern* holds: halve the
bytes, nearly double the speed. It also explains why a Mac with fast unified memory runs local
models well, and a PC running on ordinary system RAM does not.

---

## 3. How quantization works — the basic recipe

Take a group of weights, find the largest magnitude, and map the range onto integers:

```
weights:   [ 0.12, -0.50,  0.33,  0.05 ]

int8 range is -127..127
scale  = max |w| / 127 = 0.50 / 127 = 0.003937

store  = round(w / scale) = [ 30, -127,  84,  13 ]      <- 1 byte each
use    = store × scale    = [ 0.1181, -0.5000, 0.3307, 0.0512 ]
```

That is **symmetric absmax, round-to-nearest (RTN)** quantization. You store the small integers
plus one scale; you multiply back when you need the numbers.

### Granularity: how many scales?

Here is the catch. One huge weight makes the scale huge, and then every normal weight rounds to
almost nothing. Real LLM weight matrices *do* contain a few unusually large values. The fix is to
keep more scales:

| Granularity | One scale per | Extra cost |
|---|---|---|
| per-tensor | whole matrix | none |
| per-channel | row (output channel) | tiny |
| per-group | block of e.g. 128 weights | ~0.125 bits per weight |

What the demo measures on a layer with 0.1% outlier weights (error in the layer's *output*):

| Format | Per-tensor | Per-channel | Per-group (128) |
|---|---|---|---|
| int8 | 12.75% | 3.52% | **1.42%** |
| int4 | 85.60% | 47.93% | **20.77%** |

Two lessons:
1. **Granularity decides whether 4-bit works at all.** One scale at int4 destroys the layer.
   Every practical 4-bit format uses small groups.
2. **Plain rounding is not enough for 4-bit.** Even with groups, naive RTN leaves ~21% error here.
   Closing that gap is exactly what GPTQ and AWQ were invented for (section 5).

---

## 4. Weights vs activations — reading the names

Formats are named **W**_bits_**A**_bits_: what precision the **W**eights and the **A**ctivations
(the numbers flowing between layers) use.

| Name | Weights | Activations | Typical use |
|---|---|---|---|
| W16A16 / fp16, bf16 | 16-bit | 16-bit | baseline |
| **W4A16** | 4-bit | 16-bit | GPTQ, AWQ, GGUF Q4 — the most common local/serving format |
| W8A16 | 8-bit | 16-bit | safe, simple memory saving |
| **W8A8** / **FP8** | 8-bit | 8-bit | fast serving on GPUs with native 8-bit maths |

**Weight-only** (W4A16, W8A16) saves memory and speeds up memory-bound decode, but the maths still
runs in 16-bit. **Weight + activation** (W8A8, FP8) can also speed up the arithmetic itself — but is
much harder, because of outliers.

### The activation outlier problem

Large models develop a handful of hidden-state features that are consistently enormous. Quantize
activations with one scale and those few features crush everything else. The demo reproduces it:

| Method | Output error |
|---|---|
| W8A16 — weights int8, activations fp16 | 0.79% |
| W8A8 — naive, activations int8 | **9.95%** (13× worse) |
| **LLM.int8()** — keep the outlier features in fp16 | 0.28% |
| **SmoothQuant** — move the outliers from activations into weights, then W8A8 | 1.67% |

SmoothQuant's trick is neat: `X · Wᵀ = (X / s) · (W · s)ᵀ` for any per-feature scale `s`. Divide the
spiky activations by `s`, multiply the weights by `s`, and the maths is unchanged — but now both are
easy to quantize.

---

## 5. The methods you'll actually meet

| Method | What it does | When to use |
|---|---|---|
| **RTN** (round-to-nearest) | the basic recipe above | 8-bit; baseline for comparison |
| **GPTQ** | quantizes weights one at a time and **adjusts the remaining weights to compensate** for each rounding error, using a small calibration set | 4-bit GPU serving |
| **AWQ** | finds the ~1% of weights that matter most (by looking at activations) and **scales them up before rounding** so they survive | 4-bit GPU serving; often slightly better than GPTQ |
| **GGUF** (llama.cpp) | a **file format**, not one algorithm: K-quants like `Q4_K_M`, `Q5_K_M`, `Q8_0`, mixing bit-widths across layers | CPUs, Macs, consumer GPUs, Ollama |
| **bitsandbytes** | quantizes on the fly when loading in Python (8-bit, 4-bit NF4) | quick experiments; QLoRA training |
| **FP8** | 8-bit floating point, weights and activations | fast serving on newer NVIDIA/AMD GPUs |
| **KV-cache quantization** | quantize the cache, not the weights | many more users per GPU — see [module 01](../01-kv-cache/) |

### Decoding a GGUF file name

`Llama-3.1-8B-Instruct-Q4_K_M.gguf`:
- `Q4` — about 4 bits per weight
- `K` — "K-quant", llama.cpp's block-wise scheme with per-block scales
- `M` — medium: some important tensors kept at higher precision (`S` = small, `L` = large)

`Q4_K_M` is the popular default balance; `Q5_K_M` if you have the memory; `Q8_0` is near-lossless;
below `Q4` quality drops noticeably.

### Which should I use?

| Situation | Choose |
|---|---|
| Laptop, Mac, CPU, one user | **GGUF `Q4_K_M`** or `Q5_K_M` with llama.cpp or Ollama |
| GPU server, many users | **AWQ or GPTQ int4**, or **FP8** on newer GPUs, served with vLLM |
| Quick Python experiment | **bitsandbytes** 8-bit or 4-bit |
| Fine-tuning a big model on a small GPU | **QLoRA (NF4)** — see [02-training/04-qlora](../../02-training/04-qlora/) |

---

## 6. Quality: always check

Averages hide failures. A quantized model can match the original on general chat and fall apart on
arithmetic, code, long context or less-common languages — usually the hardest tasks degrade
first. Rules of thumb:

- **8-bit** is close to lossless for most models.
- **4-bit with a good method** (AWQ, GPTQ, GGUF K-quants) is usually a small, acceptable loss.
- **3-bit and below** degrades noticeably; bigger models tolerate it better than small ones.
- A **larger model at 4-bit** often beats a **smaller model at 16-bit** with the same memory.

Always evaluate on *your* task with *your* prompts ([05 · Evaluation](../../05-evaluation/)).

---

## 7. Try it

**Ollama** — models are 4-bit GGUF by default:
```bash
ollama run llama3.1:8b
```

**llama.cpp** — measure speed for different quantizations on your machine:
```bash
llama-bench -m model-Q4_K_M.gguf
llama-bench -m model-Q8_0.gguf
```

**vLLM** — serve a pre-quantized model (AWQ, GPTQ and FP8 checkpoints are published on Hugging Face):
```bash
vllm serve <an-AWQ-quantized-model> --quantization awq
```

**Transformers + bitsandbytes** — quantize while loading:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
    "<model>", quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)
```

---

## 8. Examples

```bash
pip install torch
python examples/weight_quantization_demo.py      # a few seconds on CPU
```

[`weight_quantization_demo.py`](examples/weight_quantization_demo.py) measures:
1. int8 and int4 at three granularities, on a layer with outlier weights
2. the activation-outlier problem, and how LLM.int8() and SmoothQuant fix it
3. model size and the upper-bound decode speed on four kinds of hardware

---

## 9. Exercises

1. **Quantize by hand.** Take `[0.9, -0.2, 0.05, -0.7]`, quantize to int4 symmetric (−8..7) with one
   scale, dequantize, and compute the error of each value. Which value suffers most, and why?
2. **Remove the outliers.** In the demo, delete the line that multiplies outlier weights by 20.
   How much do int4 per-tensor and per-group errors change? What does that tell you about where
   quantization error really comes from?
3. **Group size sweep.** Try groups of 32, 64, 128, 256. Plot output error against bits per weight.
   Where is the sweet spot?
4. **A mini-GPTQ.** Quantize the weights of a row one at a time; after rounding each weight, spread
   its rounding error onto the not-yet-quantized weights in that row (a very simplified version of
   GPTQ's compensation). Does int4 per-group error drop below plain RTN?
5. **SmoothQuant's alpha.** Try `alpha` = 0, 0.25, 0.5, 0.75, 1.0. Plot error. Why is neither
   extreme best?
6. **Check the speed rule.** Look up your own computer's memory bandwidth, run a Q4_K_M model with
   `llama-bench`, and compare against bandwidth ÷ file size.

---

## 10. Projects to build and test

### Beginner — Local quantization comparison
Run one model in Ollama or llama.cpp at `Q4_K_M`, `Q5_K_M` and `Q8_0`.

**How to test it:** measure tokens per second with `llama-bench`, and score 20 fixed questions
(include arithmetic and code). Produce a table of size, speed and correct answers, and say which you
would pick for your laptop and why.

### Intermediate — RTN vs GPTQ vs AWQ on a small model
Quantize a small model (0.5–1.5B) three ways to 4-bit and measure the damage.

**How to test it:** compare perplexity on a held-out text *and* accuracy on a small task benchmark.
If perplexity and task accuracy disagree about which method is best, report that — it is a real
finding, not a bug.

### Advanced — Implement GPTQ for one layer
Implement GPTQ's column-by-column quantization with Hessian-based error compensation for a single
linear layer.

**How to test it:** on a real layer with real calibration activations, your int4 output error must
be clearly lower than round-to-nearest at the same group size, and close to the reference
implementation's.

---

## 11. Resources

**Start here — beginner friendly**
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — Maarten Grootendorst. 50+ diagrams; the best first read.
- [LLM Quantization Methods: GPTQ, AWQ, GGUF](https://cast.ai/blog/demystifying-quantizations-llms/) — Cast AI. A clear comparison of the formats.
- [LLM Quantization Guide: GGUF vs AWQ vs GPTQ vs bitsandbytes](https://www.premai.io/blog/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — Prem AI. Practical format chooser.

**Go deeper**
- [The Complete Guide to LLM Quantization with vLLM](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — JarvisLabs, with benchmarks.
- [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/) — supported formats and hardware compatibility.
- [Optimizing your LLM in production — Lower Precision](https://huggingface.co/blog/optimize-llm) — Hugging Face.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the GGUF ecosystem, `llama-quantize` and `llama-bench`.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — Lilian Weng's quantization section.

**Papers**
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — the outlier features discovery
- [SmoothQuant](https://arxiv.org/abs/2211.10438) — migrating difficulty from activations to weights
- [GPTQ](https://arxiv.org/abs/2210.17323) — accurate post-training 4-bit quantization
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)

**Related in this repo**
- [02-training/04-qlora](../../02-training/04-qlora/) — NF4 and 4-bit for fine-tuning

---

**Previous:** [02 · Batching](../02-batching/) · **Next:** [04 · Paged Attention](../04-paged-attention/)
