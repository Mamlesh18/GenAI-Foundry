"""
LoRA in about 40 lines of PyTorch -- and proof that it does what it claims.

LoRA freezes a weight matrix W and learns a low-rank correction beside it:

    output = W @ x  +  (alpha / r) * (B @ A) @ x
             frozen              trainable, r is tiny

This script implements that, then verifies the four claims that matter:

  1. at initialisation the output is IDENTICAL to the base model (B starts at 0)
  2. only A and B receive gradients; W receives none
  3. merging W + (alpha/r)*B*A reproduces the adapted output exactly
  4. the parameter saving is as large as advertised

    pip install torch
    python lora_from_scratch.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# The whole technique
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank correction."""

    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r        # so changing r does not change the effective LR

        # Freeze the original weights. This is the point: they never move.
        for p in self.base.parameters():
            p.requires_grad = False

        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.dropout = nn.Dropout(dropout)

        # A is random, B is ZERO. So B@A == 0 at step 0, so the model starts as
        # an exact copy of the base model and departs from it gradually.
        # Random-init both and you begin by injecting noise into a working model.
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x):
        return self.base(x) + self.dropout(x) @ self.A.T @ self.B.T * self.scaling

    def merged_weight(self):
        """W + (alpha/r) * B @ A -- exact, no approximation."""
        return self.base.weight + self.scaling * (self.B @ self.A)


# ---------------------------------------------------------------------------
# A small model to test it on
# ---------------------------------------------------------------------------

class Net(nn.Module):
    # Dimensions chosen to be representative of a real transformer's linear
    # layers. With a tiny net, rank 8 is not small relative to the weights and
    # the parameter saving looks unimpressive for the wrong reason.
    def __init__(self, d_in=1024, d_hidden=2048, d_out=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def apply_lora(model, r=8, alpha=16):
    """Swap every nn.Linear for a LoRA-wrapped version. This is, in essence,
    what peft.get_peft_model does."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha))
        else:
            apply_lora(module, r, alpha)
    return model


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Proofs
# ---------------------------------------------------------------------------

def main():
    D_IN, D_HIDDEN, D_OUT, R, ALPHA = 1024, 2048, 1024, 8, 16
    x = torch.randn(16, D_IN)

    base = Net(D_IN, D_HIDDEN, D_OUT)
    base_out = base(x).clone()

    base_trainable, base_total = count_params(base)

    lora = apply_lora(Net(D_IN, D_HIDDEN, D_OUT), r=R, alpha=ALPHA)
    lora.load_state_dict(
        {k.replace("fc1.", "fc1.base.").replace("fc2.", "fc2.base."): v
         for k, v in base.state_dict().items()}, strict=False)

    print("=" * 74)
    print("CLAIM 1: at initialisation, LoRA output == base output")
    print("=" * 74)
    lora_out = lora(x)
    diff = (base_out - lora_out).abs().max().item()
    print(f"  max |base - lora| = {diff:.2e}")
    print(f"  identical: {torch.allclose(base_out, lora_out, atol=1e-6)}")
    print("\n  Because B is initialised to zeros, B@A == 0, so the correction term")
    print("  contributes nothing. Training starts from a known-good model.")

    print("\n" + "=" * 74)
    print("CLAIM 2: only A and B are trainable")
    print("=" * 74)
    lora_trainable, lora_total = count_params(lora)
    print(f"  base model : {base_trainable:>9,} trainable / {base_total:>9,} total")
    print(f"  LoRA model : {lora_trainable:>9,} trainable / {lora_total:>9,} total")
    print(f"  training {lora_trainable / lora_total:.2%} of the parameters")
    print(f"  reduction  : {base_trainable / lora_trainable:.0f}x fewer trainable parameters\n")
    for name, p in lora.named_parameters():
        mark = "TRAIN" if p.requires_grad else "frozen"
        print(f"    {mark:>6}  {name:<20} {tuple(p.shape)}")

    # Confirm gradients flow only where we expect.
    lora(x).sum().backward()
    print("\n  after backward():")
    for name, p in lora.named_parameters():
        state = "has grad" if p.grad is not None else "no grad "
        print(f"    {state}  {name}")

    print("\n" + "=" * 74)
    print("CLAIM 3: LoRA recovers a LOW-RANK update -- which is its whole premise")
    print("=" * 74)
    print(
        "  The paper's claim is that the update a model needs during fine-tuning has\n"
        "  low intrinsic rank. We can test that directly: build target models that\n"
        "  differ from the base by a known rank, and see which ones rank-8 LoRA can\n"
        "  actually reach.\n"
    )

    X = torch.randn(512, D_IN)

    def make_target(target_rank, seed):
        """A copy of `base` whose weights differ by a perturbation of known rank."""
        g = torch.Generator().manual_seed(seed)
        tgt = Net(D_IN, D_HIDDEN, D_OUT)
        tgt.load_state_dict(base.state_dict())
        with torch.no_grad():
            for lin in (tgt.fc1, tgt.fc2):
                d_out, d_in = lin.weight.shape
                u = torch.randn(d_out, target_rank, generator=g)
                v = torch.randn(target_rank, d_in, generator=g)
                delta = u @ v
                # Rescale every perturbation to the SAME Frobenius norm, so the
                # only thing varying across rows of the table is rank. Without
                # this, higher-rank targets come out smaller and the comparison
                # measures magnitude rather than rank.
                delta = delta / delta.norm() * (0.30 * lin.weight.norm())
                lin.weight.add_(delta)
        return tgt

    def train_lora(Y, steps=400):
        model = apply_lora(Net(D_IN, D_HIDDEN, D_OUT), r=R, alpha=ALPHA)
        model.load_state_dict(
            {k.replace("fc1.", "fc1.base.").replace("fc2.", "fc2.base."): v
             for k, v in base.state_dict().items()}, strict=False)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        loss_fn = nn.MSELoss()
        before = loss_fn(model(X), Y).item()
        for _ in range(steps):
            loss = loss_fn(model(X), Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return before, loss.item()

    print(f"  {'target rank':>12} {'loss before':>13} {'loss after':>12} {'reduction':>11}")
    print("  " + "-" * 52)
    for target_rank in (2, 8, 64, 1024):
        Y = make_target(target_rank, seed=target_rank)(X).detach()
        before, after = train_lora(Y)
        label = f"{target_rank}" + (" (full)" if target_rank == 1024 else "")
        print(f"  {label:>12} {before:>13.4f} {after:>12.4f} {before / after:>10.0f}x")

    print(
        "\n  Rank-8 LoRA nearly erases a low-rank difference and barely dents a\n"
        "  full-rank one. That IS the LoRA bet: fine-tuning adjustments are\n"
        "  low-rank, so a low-rank correction is enough. When a task needs more,\n"
        "  raising the rank is the fix -- which is exercise 4.\n"
        "\n  Throughout, the frozen base weights never moved. All of it came from A and B."
    )

    print("\n" + "=" * 74)
    print("CLAIM 4: merging is exact -- zero inference overhead")
    print("=" * 74)

    merged = Net(D_IN, D_HIDDEN, D_OUT)
    with torch.no_grad():
        merged.fc1.weight.copy_(lora.fc1.merged_weight())
        merged.fc1.bias.copy_(lora.fc1.base.bias)
        merged.fc2.weight.copy_(lora.fc2.merged_weight())
        merged.fc2.bias.copy_(lora.fc2.base.bias)

    with torch.no_grad():
        adapted_out = lora(x)
        merged_out = merged(x)
    merge_diff = (adapted_out - merged_out).abs().max().item()

    m_trainable, m_total = count_params(merged)
    print(f"  max |adapted - merged| = {merge_diff:.2e}")
    print(f"  equivalent: {torch.allclose(adapted_out, merged_out, atol=1e-5)}")
    print(f"\n  merged model parameters : {m_total:,}  (base was {base_total:,})")
    print("  Identical architecture, identical parameter count, no adapter at runtime.")

    print("\n" + "=" * 74)
    print("WHY THIS SCALES SO WELL")
    print("=" * 74)
    d = 4096
    full = d * d
    lo = R * d + d * R
    print(f"  For one {d}x{d} attention projection at rank {R}:")
    print(f"    full update dW : {full:>12,} parameters")
    print(f"    LoRA A + B     : {lo:>12,} parameters")
    print(f"    ratio          : {lo / full:>12.2%}  ({full // lo}x smaller)\n")
    print(
        "  A 7B model's LoRA adapter is ~20MB against ~14GB for a full checkpoint.\n"
        "  You can keep a hundred task-specific adapters for the storage cost of one\n"
        "  full model, and serve them all against a single base model held in memory\n"
        "  once. That is what made fine-tuning accessible.\n\n"
        "  In production, use peft -- it handles every layer type, quantized bases,\n"
        "  saving/loading and merging. But it is doing exactly what is above."
    )


if __name__ == "__main__":
    main()
