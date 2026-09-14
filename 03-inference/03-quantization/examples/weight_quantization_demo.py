"""
Quantization for inference -- what it does, where it breaks, and why the
famous methods exist.

Quantization stores numbers with fewer bits: 16-bit floats become 8-bit or
4-bit integers plus a "scale" that maps them back. Less memory, and -- because
decoding is limited by memory speed -- faster generation.

This script measures, in plain PyTorch on CPU:

  1. the basic recipe, and how GRANULARITY (one scale per tensor / per row /
     per group) decides whether 4-bit works at all
  2. why quantizing ACTIVATIONS is much harder than quantizing weights, and
     how LLM.int8() and SmoothQuant get around it
  3. what it buys you: model size and a back-of-envelope decode speed

Training-time 4-bit (NF4 / QLoRA) is covered in ../../../02-training/04-qlora/.

    pip install torch
    python weight_quantization_demo.py
"""

import torch

torch.manual_seed(0)


def quantize(x, bits, granularity, group_size=128):
    """Symmetric absmax, round-to-nearest quantization, returned already
    de-quantized so we can measure the error. The integer codes are round(x/scale)."""
    qmax = 2 ** (bits - 1) - 1                       # int8 -> 127, int4 -> 7

    def q(t, scale):
        scale = scale.clamp(min=1e-12)
        return (t / scale).round().clamp(-qmax - 1, qmax) * scale

    if granularity == "per-tensor":                  # one scale for everything
        return q(x, x.abs().max() / qmax)
    if granularity == "per-channel":                 # one scale per output row
        return q(x, x.abs().amax(dim=1, keepdim=True) / qmax)
    if granularity == "per-group":                   # one scale per `group_size` weights
        rows, cols = x.shape
        g = x.reshape(rows, cols // group_size, group_size)
        return q(g, g.abs().amax(dim=2, keepdim=True) / qmax).reshape(rows, cols)
    raise ValueError(granularity)


def rel_err(exact, approx):
    return ((exact - approx).norm() / exact.norm()).item()


def section(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------------------
# 1. Weights: granularity is everything
# ---------------------------------------------------------------------------

def demo_weights():
    section("1. QUANTIZING WEIGHTS -- how many scales you keep decides everything")

    W = torch.randn(1024, 1024) * 0.02
    # Real LLM weight matrices contain a small number of unusually large values.
    flat = W.view(-1)
    idx = torch.randperm(flat.numel())[:1000]        # 0.1% of weights
    flat[idx] = flat[idx] * 20

    X = torch.randn(256, 1024)                       # a batch of inputs to the layer
    Y = X @ W.T

    print("  A 1024x1024 layer with 0.1% outlier weights, plain round-to-nearest.")
    print("  'output error' = how far the layer's OUTPUT moves -- the number that matters.\n")
    print(f"  {'format':<8} {'granularity':<24} {'bits/weight':>12} {'output error':>13}")
    print("  " + "-" * 60)

    overhead = {"per-tensor": 0.0, "per-channel": 16 / 1024, "per-group": 16 / 128}
    label = {"per-tensor": "per-tensor (1 scale)", "per-channel": "per-channel (1024)",
             "per-group": "per-group (g=128)"}
    e = {}
    for bits in (8, 4):
        for gran in ("per-tensor", "per-channel", "per-group"):
            e[(bits, gran)] = rel_err(Y, X @ quantize(W, bits, gran).T)
            print(f"  {'int' + str(bits):<8} {label[gran]:<24} {bits + overhead[gran]:>12.3f}"
                  f" {e[(bits, gran)]:>12.2%}")

    print(
        f"\n  int8 with ONE scale for the whole matrix already loses {e[(8, 'per-tensor')]:.1%}: the few\n"
        "  outliers stretch the scale, so ordinary weights use only a handful of the\n"
        f"  255 levels. One scale per group of 128 brings that down to {e[(8, 'per-group')]:.1%}.\n\n"
        "  int4 has only 16 levels, so it is far less forgiving: "
        f"{e[(4, 'per-tensor')]:.0%} error with one\n"
        f"  scale, {e[(4, 'per-channel')]:.0%} per row, {e[(4, 'per-group')]:.1%} per group. Small groups are why every\n"
        "  practical 4-bit format -- GPTQ, AWQ, GGUF's K-quants -- quantizes in blocks.\n\n"
        f"  But {e[(4, 'per-group')]:.0%} is still a lot, and that is the honest point of this table.\n"
        "  Plain rounding is the naive baseline. GPTQ closes much of the remaining gap\n"
        "  by nudging the not-yet-quantized weights to compensate for each rounding\n"
        "  error; AWQ by scaling up the weights that matter most to the output before\n"
        "  rounding them. That remaining error is the reason those methods exist."
    )


# ---------------------------------------------------------------------------
# 2. Activations: the hard part
# ---------------------------------------------------------------------------

def demo_activations():
    section("2. QUANTIZING ACTIVATIONS -- the outlier problem")

    W = torch.randn(1024, 1024) * 0.02
    X = torch.randn(256, 1024)
    # LLM.int8() found that large models develop a few hidden-state FEATURES
    # that are consistently huge -- across nearly every token.
    outlier_features = torch.tensor([7, 100, 333, 512, 700, 901])
    X[:, outlier_features] *= 60
    Y = X @ W.T

    print("  Hidden states with 6 outlier features (x60), as seen in real LLMs.\n")
    print(f"  {'method':<46} {'output error':>13}")
    print("  " + "-" * 60)

    # (a) weight-only: weights int8, activations stay 16-bit
    w8a16 = rel_err(Y, X @ quantize(W, 8, "per-channel").T)
    print(f"  {'W8A16  weights int8, activations fp16':<46} {w8a16:>12.2%}")

    # (b) naive W8A8: quantize activations with one scale too
    w8a8 = rel_err(Y, quantize(X, 8, "per-tensor") @ quantize(W, 8, "per-channel").T)
    print(f"  {'W8A8   naive, activations per-tensor int8':<46} {w8a8:>12.2%}")

    # (c) LLM.int8(): keep outlier feature columns in fp16, int8 for the rest
    is_outlier = X.abs().amax(dim=0) > 6.0
    normal = ~is_outlier
    y_out = X[:, is_outlier] @ W[:, is_outlier].T
    y_norm = quantize(X[:, normal], 8, "per-tensor") @ quantize(W[:, normal], 8, "per-channel").T
    llm_int8 = rel_err(Y, y_out + y_norm)
    print(f"  {'LLM.int8()  outlier features kept in fp16':<46} {llm_int8:>12.2%}")

    # (d) SmoothQuant: move the difficulty from activations into weights.
    #     X @ W.T == (X / s) @ (W * s).T exactly, for any per-feature scale s.
    alpha = 0.5
    s = X.abs().amax(dim=0) ** alpha / W.abs().amax(dim=0) ** (1 - alpha)
    smooth = rel_err(Y, quantize(X / s, 8, "per-tensor") @ quantize(W * s, 8, "per-channel").T)
    print(f"  {'SmoothQuant (alpha=0.5), then W8A8':<46} {smooth:>12.2%}")

    print(
        f"\n  Weight-only int8 barely notices the outliers ({w8a16:.2%}). Quantizing the\n"
        f"  activations naively is {w8a8 / w8a16:.0f}x worse ({w8a8:.1%}): six huge features set the\n"
        "  scale, and every other feature is crushed onto a handful of integer levels.\n\n"
        f"  LLM.int8() keeps those few features in fp16 ({llm_int8:.2%}) -- accurate, but\n"
        "  the split makes the matmul slower. SmoothQuant divides the activations by a\n"
        "  per-feature scale and multiplies the weights by the same scale, which leaves\n"
        f"  the maths unchanged but moves the outliers into the weights ({smooth:.2%}).\n\n"
        "  This is why most local inference uses WEIGHT-ONLY quantization (W4A16,\n"
        "  W8A16): easy and robust. Weight+activation formats (W8A8, FP8) are faster on\n"
        "  hardware with native low-bit kernels, but need tricks like these for quality."
    )


# ---------------------------------------------------------------------------
# 3. What it buys you
# ---------------------------------------------------------------------------

def demo_payoff():
    section("3. WHAT IT BUYS YOU -- size and decode speed (8B model)")

    params = 8.0e9
    formats = [("fp16", 16), ("int8 / FP8", 8), ("int4, group 128", 4.25)]
    # Memory bandwidth in GB/s (manufacturer specs, approximate).
    hardware = [("PC, DDR5 dual-ch", 89.6), ("Apple M2 Ultra", 800),
                ("RTX 4090", 1008), ("A100 80GB", 1935)]

    print("  Upper-bound decode speed for ONE user = memory bandwidth / model size\n")
    print(f"  {'format':<16} {'size':>8} " + "".join(f"{h:>18}" for h, _ in hardware))
    print("  " + "-" * (26 + 18 * len(hardware)))
    for name, bits in formats:
        size_gb = params * bits / 8 / 1e9
        speeds = "".join(f"{bw / size_gb:>12.0f} tok/s" for _, bw in hardware)
        print(f"  {name:<16} {size_gb:>5.1f} GB {speeds}")

    print(
        "\n  Every generated token reads (roughly) every weight once, so the speed\n"
        "  limit is how fast memory delivers bytes, not how fast the chip multiplies.\n"
        "  Halve the bytes and you can nearly double the tokens per second.\n\n"
        "  These are UPPER BOUNDS for batch size 1 -- real numbers are lower (compute,\n"
        "  KV cache reads, kernel overhead). And a 16 GB fp16 8B model does not fit on\n"
        "  many consumer GPUs at all, which is the other half of the story."
    )


def main():
    demo_weights()
    demo_activations()
    demo_payoff()

    section("WHICH FORMAT SHOULD I USE?")
    print(
        "  Laptop / Mac / CPU, one user   -> GGUF (Q4_K_M or Q5_K_M) with llama.cpp or Ollama\n"
        "  GPU server, many users         -> AWQ or GPTQ int4, or FP8 on newer GPUs, with vLLM\n"
        "  Quick experiment in Python     -> bitsandbytes (8-bit or 4-bit loading)\n"
        "  Fine-tuning                    -> QLoRA NF4, see ../../../02-training/04-qlora/\n\n"
        "  Whatever you choose: evaluate the quantized model on YOUR task. Averages\n"
        "  hide the cases -- maths, code, rare languages -- that degrade first."
    )


if __name__ == "__main__":
    main()
