"""
4-bit quantization from scratch -- and what it actually costs you.

QLoRA's only change to LoRA is storing the FROZEN base model in 4 bits. This
script implements that compression in plain PyTorch and measures the damage:

  1. block-wise quantization: 16-bit -> 4-bit -> 16-bit, and the error
  2. INT4 (evenly spaced levels) vs NF4 (normal-optimal levels)
  3. how block size trades memory against accuracy
  4. the end-to-end effect on a matmul, which is what actually matters
  5. real memory numbers for 7B / 13B / 70B

No bitsandbytes required -- it runs on CPU so you can read every line.

    pip install torch
    python quantization_demo.py
"""

import torch

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# The NF4 data type
# ---------------------------------------------------------------------------
# 16 levels, placed so that each holds an equal probability mass under a normal
# distribution. Evenly spaced levels (INT4) waste most of their resolution on
# the sparse tails; NF4 concentrates it where the weights actually are.
# These are the constants from the QLoRA paper.

NF4_LEVELS = torch.tensor([
    -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0000,
     0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0000,
])

INT4_LEVELS = torch.linspace(-1, 1, 16)


def quantize_blockwise(w, levels, block_size=64):
    """Quantize a tensor to the given 16 levels, one scale per block.

    Per-block scaling is what makes 4 bits usable at all: a single scale for a
    whole 4096x4096 matrix would be dominated by its largest outlier.
    """
    flat = w.flatten()
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.view(-1, block_size)

    # One scale per block: the largest magnitude in that block.
    scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    normalised = blocks / scales

    # Snap each value to its nearest level -> this is the 4-bit index (0..15).
    idx = (normalised.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    return idx, scales, pad, w.shape


def dequantize_blockwise(idx, scales, pad, shape, levels):
    """Reconstruct 16-bit weights. In QLoRA this happens on the fly during the
    forward pass, block by block, and the result is discarded immediately."""
    blocks = levels[idx] * scales
    flat = blocks.flatten()
    if pad:
        flat = flat[:-pad]
    return flat.view(shape)


def roundtrip(w, levels, block_size=64):
    return dequantize_blockwise(*quantize_blockwise(w, levels, block_size), levels)


def rel_error(original, reconstructed):
    return ((original - reconstructed).norm() / original.norm()).item()


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_roundtrip():
    print("=" * 74)
    print("1. QUANTIZE AND BACK")
    print("=" * 74)

    # Real transformer weights are approximately normally distributed.
    w = torch.randn(2048, 2048) * 0.02
    recovered = roundtrip(w, NF4_LEVELS)

    print(f"  tensor            : {tuple(w.shape)}")
    print(f"  original (fp16)   : {w.numel() * 2 / 1e6:.1f} MB")
    print(f"  quantized (nf4)   : {w.numel() * 0.5 / 1e6:.1f} MB + scales")
    print(f"  compression       : ~4x\n")
    print(f"  first 6 original  : {[f'{v:+.4f}' for v in w.flatten()[:6].tolist()]}")
    print(f"  first 6 recovered : {[f'{v:+.4f}' for v in recovered.flatten()[:6].tolist()]}")
    print(f"\n  relative error    : {rel_error(w, recovered):.4%}")
    print("  Each weight is now one of only 16 values per block. The error is the")
    print("  price; the QLoRA paper's finding is that a FROZEN model tolerates it.")


def demo_nf4_vs_int4():
    print("\n" + "=" * 74)
    print("2. NF4 vs INT4 -- why the level placement matters")
    print("=" * 74)

    normal = torch.randn(4096, 512) * 0.02          # like real weights
    uniform = (torch.rand(4096, 512) - 0.5) * 0.08  # deliberately not normal

    print(f"  {'data':>22} {'INT4 error':>12} {'NF4 error':>12} {'NF4 better by':>15}")
    print("  " + "-" * 64)
    for name, data in (("normal (real weights)", normal), ("uniform", uniform)):
        e_int = rel_error(data, roundtrip(data, INT4_LEVELS))
        e_nf4 = rel_error(data, roundtrip(data, NF4_LEVELS))
        print(f"  {name:>22} {e_int:>11.3%} {e_nf4:>11.3%} {e_int / e_nf4:>14.2f}x")

    print(
        "\n  NF4 places its 16 levels so each holds equal probability mass under a\n"
        "  normal distribution -- dense near zero, sparse in the tails, matching\n"
        "  where weights actually live. On uniform data that advantage disappears,\n"
        "  because there the evenly spaced INT4 levels are already well matched.\n"
        "  NF4 is not universally better; it is better for the distribution it was\n"
        "  designed for, which happens to be how neural network weights look."
    )


def demo_block_size():
    print("\n" + "=" * 74)
    print("3. BLOCK SIZE -- memory vs accuracy")
    print("=" * 74)

    w = torch.randn(2048, 2048) * 0.02
    # A few large outliers, which real weight matrices do have.
    w.flatten()[torch.randint(0, w.numel(), (50,))] *= 25

    print(f"  {'block size':>11} {'error':>10} {'scale overhead':>16} {'bits/weight':>13}")
    print("  " + "-" * 54)
    for bs in (16, 64, 256, 1024, w.numel()):
        err = rel_error(w, roundtrip(w, NF4_LEVELS, block_size=bs))
        overhead_bits = 32 / bs                  # one fp32 scale per block
        label = "whole tensor" if bs == w.numel() else str(bs)
        print(f"  {label:>11} {err:>9.3%} {overhead_bits:>14.3f} b {4 + overhead_bits:>11.3f}")

    print(
        "\n  Smaller blocks isolate outliers, so error falls -- but each block needs\n"
        "  its own scale constant, so memory rises. One scale for the whole tensor\n"
        "  is catastrophic: a single outlier sets the scale and everything else\n"
        "  collapses toward zero.\n"
        "  64 is the common default. Double quantization then compresses the scale\n"
        "  constants themselves, recovering most of that 0.5 bits/weight overhead."
    )


def demo_end_to_end():
    print("\n" + "=" * 74)
    print("4. WHAT IT DOES TO AN ACTUAL FORWARD PASS")
    print("=" * 74)

    w = torch.randn(1024, 1024) * 0.02
    x = torch.randn(64, 1024)

    exact = x @ w.T
    approx = x @ roundtrip(w, NF4_LEVELS).T

    print(f"  weight error on its own      : {rel_error(w, roundtrip(w, NF4_LEVELS)):.4%}")
    print(f"  resulting output error       : {rel_error(exact, approx):.4%}")
    print(f"  cosine similarity of outputs : {torch.nn.functional.cosine_similarity(
        exact.flatten(), approx.flatten(), dim=0).item():.6f}")
    print(
        "\n  This is the number that matters -- not the weight error, but its effect\n"
        "  on what the layer computes. Errors are partly random and partly cancel\n"
        "  across a 1024-term dot product, so the output degrades less than the\n"
        "  weights do. Stack many layers and it accumulates, which is why 2-3 bit\n"
        "  quantization degrades sharply while 4-bit largely holds up."
    )


def demo_memory():
    print("\n" + "=" * 74)
    print("5. WHAT ACTUALLY FITS")
    print("=" * 74)
    print(f"  {'model':>8} {'fp32':>9} {'fp16':>9} {'int8':>9} {'4-bit':>9}")
    print("  " + "-" * 48)
    for name, n in (("7B", 7e9), ("13B", 13e9), ("70B", 70e9)):
        row = "  ".join(f"{n * b / 1e9:>7.1f}GB" for b in (4, 2, 1, 0.5))
        print(f"  {name:>8}  {row}")

    print(
        "\n  Weights only -- add optimizer states, gradients and activations for the\n"
        "  real training figure (see the table in ../../README.md).\n\n"
        "  The point: a free Colab T4 has 16GB. At fp16 a 7B model's weights alone\n"
        "  eat 14GB of it and you cannot train. At 4-bit they take 3.5GB, leaving\n"
        "  room for adapters, activations and a usable batch size.\n"
        "  That single row is why QLoRA exists."
    )


def main():
    demo_roundtrip()
    demo_nf4_vs_int4()
    demo_block_size()
    demo_end_to_end()
    demo_memory()

    print("\n" + "=" * 74)
    print("SO WHAT IS QLoRA?")
    print("=" * 74)
    print(
        "  Exactly LoRA, with the frozen base model stored like this instead of in\n"
        "  16-bit. The adapters stay full precision -- they are tiny and they are\n"
        "  what you are training. The base is frozen and never updated, so it can\n"
        "  absorb the compression.\n\n"
        "  Forward pass: dequantize a block, use it, throw it away. You trade ~20-40%\n"
        "  speed for ~4x less memory, which is the right trade whenever the\n"
        "  alternative is not being able to run at all.\n\n"
        "  In production this is bitsandbytes, and in LLaMA-Factory it is two lines:\n"
        "      quantization_bit: 4\n"
        "      quantization_method: bnb"
    )


if __name__ == "__main__":
    main()
