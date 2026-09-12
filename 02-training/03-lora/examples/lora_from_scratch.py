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
    print("CLAIM 3: training works, and the base model alone cannot do the task")
    print("=" * 74)

    # A task the randomly-initialised base model has no reason to solve.
    torch.manual_seed(1)
    target_fn = nn.Linear(D_IN, D_OUT)
    for p in target_fn.parameters():
        p.requires_grad = False

    X = torch.randn(512, D_IN)
    Y = target_fn(X)

    opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=1e-2)
    loss_fn = nn.MSELoss()

    start_loss = loss_fn(lora(X), Y).item()
    for step in range(400):
        loss = loss_fn(lora(X), Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    end_loss = loss.item()

    base_loss = loss_fn(base(X), Y).item()
    print(f"  base model loss (frozen, untrained) : {base_loss:.4f}")
    print(f"  LoRA loss before training           : {start_loss:.4f}")
    print(f"  LoRA loss after 400 steps           : {end_loss:.4f}")
    print(f"  improvement                         : {start_loss / end_loss:.1f}x")
    print("\n  The frozen base weights never moved. All of that came from A and B.")

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
