# 04 · QLoRA

LoRA made fine-tuning cheap in *trainable parameters*. But you still had to hold the frozen base
model in memory at 16-bit — 14GB for a 7B model, before anything else.

**QLoRA quantizes that frozen base to 4-bit.** A 7B model drops to roughly 6GB, which fits on a
free Colab T4. That is the whole idea, and it is the reason an individual can fine-tune a serious
model at all.

---

## 1. What this is

```
LoRA    :  frozen base in 16-bit  +  trainable adapters in 16-bit   ->  ~16 GB for 7B
QLoRA   :  frozen base in  4-bit  +  trainable adapters in 16-bit   ->  ~6 GB for 7B
                             ^
                    this is the only change
```

The adapters stay at full precision — they are tiny, and you are training them, so precision
matters there. The base model is frozen and never updated, so it can tolerate aggressive
compression.

### The mechanics

During the forward pass, 4-bit weights are **dequantized back to 16-bit on the fly**, block by
block, used for the matmul, and discarded. You never materialise the full 16-bit model. You pay a
little compute to save a lot of memory — QLoRA typically runs ~20–40% slower than LoRA, which is
almost always a trade worth making when the alternative is not running at all.

### Three ideas from the paper

**1. NF4 (4-bit NormalFloat).** Neural network weights are roughly normally distributed. Plain
INT4 spaces its 16 levels evenly, which wastes most of them on the sparse tails. NF4 places its 16
levels so that *each holds an equal number of weights* under a normal distribution —
information-theoretically optimal for normally distributed data, and measurably better than INT4 at
the same bit width.

**2. Double quantization.** Quantization needs a scale constant per block (typically 64 weights).
Those constants are themselves stored in 32-bit — which costs ~0.5 bits per parameter. So quantize
the constants too. Saves ~0.37 bits per parameter; on a 70B model, several GB for free.

**3. Paged optimizers.** Gradient checkpointing causes memory spikes that OOM at the worst moment.
Paged optimizer states spill to CPU RAM automatically when VRAM runs short, the way an OS pages
memory. Turns a crash into a slowdown.

### What it costs you in quality

The paper's headline finding: **QLoRA matches 16-bit LoRA performance.** Not "close to" — matches,
across their evaluations. This is the unusual case where the memory saving is close to free.

Caveats worth holding: it is slower, very low-bit quantization (2–3 bit) does degrade noticeably,
and results vary by model and task. Measure on your own task rather than trusting the headline.

---

## 2. What actually fits

| Model | Full FT | LoRA (16-bit) | **QLoRA (4-bit)** |
|---|---|---|---|
| 7B | ~120 GB | ~16 GB | **~6 GB** |
| 14B | ~240 GB | ~32 GB | **~12 GB** |
| 70B | ~1200 GB | ~160 GB | **~48 GB** |

*Order of magnitude; actual usage depends on sequence length, batch size and optimizer.*

Read that 7B row against the hardware you can get free: a Colab T4 has 16GB. QLoRA puts a 7B
fine-tune comfortably inside it, with room for a reasonable batch size and sequence length.

---

## 3. Choosing your quantization

LLaMA-Factory supports several backends. They are not interchangeable:

| Method | When to use | Note |
|---|---|---|
| **bitsandbytes (NF4)** | **Training.** This is QLoRA. | On-the-fly quantization, no calibration step |
| **GPTQ** | Serving an already-trained model | Needs a calibration pass; faster inference |
| **AWQ** | Serving | Activation-aware; often better quality than GPTQ |
| **HQQ / EETQ** | Alternatives worth benchmarking | Fast, no calibration |

**The rule: bitsandbytes for training, GPTQ/AWQ for serving.** In LLaMA-Factory,
`quantization_bit: 4` with `quantization_method: bnb` is the QLoRA path.

### Merging order matters

A mistake that quietly costs quality:

```
WRONG:  quantize base -> train adapter -> merge adapter into the 4-bit model
RIGHT:  quantize base -> train adapter -> merge into the FULL-PRECISION base -> quantize result
```

Merging into a quantized model compounds quantization error into your adapter's contribution. Load
the original 16-bit weights to merge, then quantize the merged output if you want a small model to
serve.

---

## 4. Examples

### See quantization happen

```bash
pip install torch
python examples/quantization_demo.py
```

[`quantization_demo.py`](examples/quantization_demo.py) implements 4-bit block-wise quantization
from scratch in PyTorch and measures what it actually costs. It:

- quantizes a real weight-shaped tensor to 4 bits and back, reporting the reconstruction error
- compares **INT4 (evenly spaced) against NF4 (normal-optimal)** on the same data, showing why NF4
  wins on normally distributed weights
- shows how **block size** trades memory against accuracy
- measures the end-to-end effect on a matmul's output, which is what actually matters
- computes real memory numbers for 7B/13B/70B at each bit width

No `bitsandbytes` needed — it runs on CPU so you can read every line.

### Real QLoRA with LLaMA-Factory

```bash
pip install bitsandbytes
llamafactory-cli train examples/llamafactory_qlora.yaml
```

[`llamafactory_qlora.yaml`](examples/llamafactory_qlora.yaml) is tuned to fit a free Colab T4.

---

## 5. Exercises

1. **Measure the error.** Run `quantization_demo.py`. What relative error does 4-bit introduce?
   Is it large enough to worry about, given the paper's claim that quality is preserved?
2. **NF4 vs INT4.** Compare the two on normally distributed weights. Now compare them on
   *uniformly* distributed data. NF4's advantage should shrink or vanish — explain why.
3. **Block size sweep.** Try blocks of 16, 64, 256, 1024. Plot error against memory overhead. Why
   is 64 the common default?
4. **Where error goes.** Quantize only the attention projections, then only the FFN, and measure
   the effect on output. Which is more sensitive?
5. **Fit a real model.** On a free Colab T4, work out the largest model you can QLoRA at sequence
   length 2048. Then actually run it and see whether your arithmetic held.
6. **The real comparison.** Fine-tune the same task with LoRA (16-bit) and QLoRA (4-bit). Compare
   final quality, peak memory and wall-clock time. Was the paper's "matches 16-bit" claim true for
   *your* task?
7. **Break the merge order.** Merge an adapter into a 4-bit base, and into the full-precision base.
   Compare outputs. Quantify what the wrong order costs.

---

## 6. Projects to build and test

### Beginner — Quantization error explorer
Extend the demo: support 2, 3, 4 and 8 bits, and plot error against bit width for real weights
pulled from a pretrained model.

**How to test it:** find the bit width where error climbs sharply. Compare with published results —
it is usually around 3 bits, and knowing where the cliff is tells you which claims to distrust.

### Intermediate — QLoRA on a free GPU
Fine-tune a 7B model on Colab or Kaggle, end to end, and publish the notebook.

**How to test it:** report peak VRAM, wall-clock time, and a before/after quality comparison on a
held-out set. Confirm it fits with room to spare, then find the sequence length at which it OOMs.

### Intermediate — Memory profiler
Instrument a training run and break peak memory down by component: base weights, adapters,
gradients, optimizer states, activations.

**How to test it:** your numbers should sum to roughly the peak your GPU reports. Then change batch
size and sequence length and confirm the right components scale.

### Advanced — Quantization bake-off for serving
Take one fine-tuned model and serve it in fp16, GPTQ-4bit, AWQ-4bit and bnb-4bit.

**How to test it:** measure quality on a real evaluation set, plus tokens/second and memory for
each. Produce the table you would actually use to make a deployment decision — including the cases
where fp16 is the right answer.

---

## 7. Resources

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — the paper. Clear, and the NF4 derivation is worth reading properly.
- [Making LLMs even more accessible](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — Hugging Face's introduction, with working code.
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — the library that implements it
- [Unsloth](https://github.com/unslothai/unsloth) — QLoRA ~2× faster with less memory; free Colab notebooks that work out of the box. The fastest path from zero to a trained adapter.
- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [LLaMA-Factory QLoRA examples](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples/train_qlora)
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — Maarten Grootendorst. Excellent diagrams; read this if the maths has not clicked.
- See also [03 · Quantization](../../03-inference/03-quantization/) for quantization at *inference* time, which is a related but distinct problem.

---

**Previous:** [03 · LoRA](../03-lora/) · **Next:** [05 · Preference Tuning](../05-preference-tuning/)
