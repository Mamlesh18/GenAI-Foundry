# 03 · LoRA

**Low-Rank Adaptation.** Train 0.1–1% of the parameters, get most of the quality, and produce an
adapter measured in megabytes instead of gigabytes.

This is the technique that moved fine-tuning from "you need a datacentre" to "you need a free
Colab notebook."

---

## 1. What this is

Full fine-tuning updates every weight matrix `W`. LoRA freezes `W` entirely and learns a small
**correction** beside it:

```
              frozen                    trainable
                |                           |
  output  =  W · x       +       (B · A) · x  · (alpha / r)
             ^                    ^     ^
        d x k, frozen        d x r   r x k,  r is tiny (4-64)
```

Instead of learning a `d × k` update, you learn two thin matrices whose product has the same shape.
For a 4096×4096 layer at rank 8:

| | Parameters |
|---|---|
| Full update `ΔW` | 4096 × 4096 = **16,777,216** |
| LoRA `A` + `B` | (8 × 4096) + (4096 × 8) = **65,536** |
| **Ratio** | **0.39%** — a 256× reduction |

### Why a low rank is enough

The insight from the paper: the *update* a model needs during fine-tuning has low **intrinsic
rank**. You are not teaching it language from scratch — it already knows that. You are nudging its
behaviour, and a nudge is a low-dimensional thing. The full `d × k` matrix has far more degrees of
freedom than the adjustment actually requires.

### Initialisation is not arbitrary

```python
A ~ N(0, sigma)     # random
B = 0               # ZEROS
```

`B` starts at zero, so `BA = 0`, so at step 0 **the model is exactly the original model**. Training
starts from a known-good state and departs from it gradually. If both were random, you would begin
by injecting noise into a working model, and the first steps would be spent undoing that.

### The alpha/rank scaling

The output is scaled by `alpha / r`. This exists so that changing the rank does not force you to
retune the learning rate — raising `r` raises the magnitude of `BA`, and dividing by `r`
compensates.

Convention is `alpha = 2 × r`. This is convention, not law; some setups use `alpha = r`. What
matters is that you know which one your framework applies, because it changes the effective
learning rate.

### Merging: zero inference cost

After training you can fold the adapter back in:

```
W_merged = W + (alpha / r) · B · A
```

This is exact — no approximation. The merged model has *identical* architecture and parameter count
to the original, so inference has **zero** LoRA overhead. Keep the adapter separate only if you
want to swap between several at request time.

---

## 2. Why it matters

| | Full fine-tuning | LoRA |
|---|---|---|
| Trainable parameters (7B) | 7,000M | ~5–20M |
| GPU memory | ~120 GB | ~16 GB |
| Checkpoint size | ~14 GB | **~20 MB** |
| Catastrophic forgetting | severe | much milder |
| Swap between tasks | reload 14 GB | load 20 MB |
| Quality vs full FT | baseline | ~95–100% on most tasks |

That checkpoint row is quietly transformative. You can keep a hundred task-specific adapters for
the storage cost of one full model, and serve them all against a single base model held in memory
once.

**The forgetting advantage is real too.** The base weights are frozen, so whatever the model knew,
it still knows. The adapter can only add a low-rank correction — it is structurally incapable of
overwriting general capability the way full fine-tuning can.

---

## 3. Choosing the hyperparameters

Only three matter, and the first two matter most.

### `lora_target` — which layers get adapters

The original paper adapted only the attention `Q` and `V` projections. Current practice is **all
linear layers** (attention *and* the feed-forward network), which consistently works better for a
given parameter budget. In LLaMA-Factory: `lora_target: all`.

### `lora_rank` (r) — capacity

| Rank | Use for |
|---|---|
| 4–8 | Style, tone, output format. Most tasks genuinely land here. |
| 16–32 | Harder tasks, more diverse data |
| 64–128 | Learning genuinely new domain behaviour, large datasets |
| 256+ | Rarely justified; you are approaching full fine-tuning's cost without its benefits |

**Start at 8.** People reach for 64 assuming more is better and mostly buy slower training and more
overfitting. If rank 8 underfits, raise it — but check that your *data* is not the limitation first,
because it usually is.

### `lora_alpha` — scaling

Set `alpha = 2 × rank` and stop thinking about it until you have a specific reason not to.

### Learning rate

LoRA tolerates a learning rate **10–100× higher** than full fine-tuning. `1e-4` to `3e-4` is the
normal range, versus `1e-5` for full. Using a full-fine-tuning learning rate with LoRA is a common
cause of "it did not learn anything".

---

## 4. Variants worth knowing

| Variant | What it changes | When |
|---|---|---|
| **QLoRA** | 4-bit frozen base model | Memory-bound. See [04 · QLoRA](../04-qlora/). |
| **DoRA** | Decomposes into magnitude + direction | Small consistent gain, slightly slower |
| **rsLoRA** | Scales by `alpha/√r` instead of `alpha/r` | Makes high ranks actually behave better |
| **LoRA+** | Different learning rates for A and B | Faster convergence, cheap to try |
| **PiSSA** | Initialises from the principal singular values of W | Faster convergence than zero-init B |

All are one-flag changes in PEFT and LLaMA-Factory. None of them will rescue a bad dataset.

---

## 5. Examples

### From scratch — build LoRA in 100 lines

```bash
pip install torch
python examples/lora_from_scratch.py
```

[`lora_from_scratch.py`](examples/lora_from_scratch.py) implements a `LoRALinear` layer in pure
PyTorch and then *proves* the three claims above numerically:

- at initialisation the LoRA model output is **bit-identical** to the base model
- only `A` and `B` receive gradients; the frozen `W` receives none
- merging `W + (alpha/r)·B·A` reproduces the adapted model's output to within float precision
- the parameter counts match the table above

It trains a small model on a task, shows the base model failing at it and the adapted model
succeeding, and reports what fraction of parameters were actually trained.

### Real LoRA with LLaMA-Factory

```bash
llamafactory-cli train examples/llamafactory_lora.yaml
```

[`llamafactory_lora.yaml`](examples/llamafactory_lora.yaml) is annotated with what each knob does
and which ones are worth touching.

### Or with PEFT directly

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,505,600 || all params: 1,548,222,464 || trainable%: 0.2910
```

---

## 6. Exercises

1. **Verify zero-init.** Run `lora_from_scratch.py`. Confirm the base and LoRA outputs are
   identical at step 0. Now change `B` to random init and rerun — what happens to the first few
   training steps, and why?
2. **Count the parameters.** For `d_model=4096`, 32 layers, adapting Q/K/V/O at rank 16: how many
   trainable parameters? What percentage of a 7B model? Check against `print_trainable_parameters`.
3. **Prove the merge.** Confirm merged and unmerged outputs match to `1e-5`. Then try merging with
   the wrong `alpha/r` scaling and measure how far the output drifts.
4. **Sweep the rank.** Train the same task at rank 1, 4, 16, 64. Plot final loss against rank. Find
   where the curve flattens — that is your task's intrinsic rank, and it is usually lower than
   people expect.
5. **Target ablation.** Adapt attention only, then FFN only, then both, at a *matched* parameter
   budget. Which wins?
6. **Learning rate.** Train LoRA at `1e-5` (a full-fine-tuning rate) and at `2e-4`. Compare.
   This is the mistake behind most "LoRA did not work for me" reports.
7. **Test forgetting.** Fine-tune narrowly with LoRA and with full fine-tuning at matched quality
   on the target task. Then test both on unrelated capabilities. Quantify the difference.

---

## 7. Projects to build and test

### Beginner — LoRA from scratch on a real model
Implement `LoRALinear`, inject it into a small Hugging Face model by wrapping its `nn.Linear`
modules, and train it. No `peft`.

**How to test it:** your trainable parameter count must match PEFT's for the same config, and
training the same data with both should reach comparable loss. If yours is far off, you have
targeted different modules — find out which.

### Intermediate — Multi-adapter server
One base model in memory, three task-specific adapters, swapped per request.

**How to test it:** confirm each adapter produces its task's behaviour, measure the swap latency,
and confirm total memory is roughly base + 3 small adapters rather than 3 full models.

### Intermediate — Rank/quality frontier
Train the same task across ranks 1→128, plot quality against trainable parameters and training time.

**How to test it:** use a held-out set and a real metric. Publish the curve. Then check whether your
conclusion survives on a *second*, different task — intrinsic rank is task-specific, and a single
curve generalises less than people assume.

### Advanced — Variant bake-off
Compare LoRA, DoRA, rsLoRA and LoRA+ on the same task and budget.

**How to test it:** matched parameter counts and matched training steps, 3 seeds each because the
differences are small enough to be drowned by seed noise. Report means and spread — a single run
showing DoRA winning by 0.5% is not evidence of anything.

---

## 8. Resources

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — the paper. Short and readable.
- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — Sebastian Raschka. Hundreds of experiments, real conclusions about rank, alpha and targets. The most useful practical page on LoRA anywhere.
- [Code LoRA From Scratch](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch) — build it yourself, guided.
- [PEFT documentation](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [LLaMA-Factory LoRA examples](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples/train_lora)
- [DoRA](https://arxiv.org/abs/2402.09353) · [rsLoRA](https://arxiv.org/abs/2312.03732) · [LoRA+](https://arxiv.org/abs/2402.12354) · [PiSSA](https://arxiv.org/abs/2404.02948)
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) — an honest look at where LoRA genuinely trails full fine-tuning, and where its resistance to forgetting wins.

---

**Previous:** [02 · Fine-Tuning](../02-fine-tuning/) · **Next:** [04 · QLoRA](../04-qlora/)
