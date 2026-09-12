"""
Pretraining, from random weights, on a CPU, in about a minute.

This is a toy in size only. The algorithm is exactly what trains a 70B model:
  - a decoder-only transformer with a causal mask
  - predict token t+1 from tokens 0..t, at every position at once
  - cross-entropy loss, backprop, AdamW

Watch the loss start near ln(vocab_size) -- a model guessing uniformly -- and
fall as it learns which characters follow which.

    pip install torch
    python tiny_pretrain.py
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Config -- deliberately tiny so this runs on a laptop CPU
# ---------------------------------------------------------------------------
BLOCK_SIZE = 64      # context length: how many characters the model can look back
D_MODEL = 128        # embedding dimension
N_HEADS = 4
N_LAYERS = 3
BATCH_SIZE = 32
STEPS = 1500
LR = 3e-3


# ---------------------------------------------------------------------------
# Data -- character-level. A real run uses a BPE tokenizer over trillions of
# tokens; the only thing that changes is scale.
# ---------------------------------------------------------------------------
TEXT = """
The transformer is a neural network architecture based entirely on attention.
Attention lets every token look at every other token and decide what matters.
The model learns by predicting the next token in a sequence, over and over.
There are no labels in pretraining. The text itself is the label, because the
next token is always sitting right there in the data. This is called
self-supervised learning, and it is what makes training on the whole internet
possible. A model that predicts the next token well has learned grammar, facts,
style, and a surprising amount of reasoning, purely as a side effect of that
one simple objective repeated trillions of times.
""" * 40

chars = sorted(set(TEXT))
VOCAB_SIZE = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


def get_batch(source):
    """Sample BATCH_SIZE random windows, and the same windows shifted by one.

    This shift is the whole training signal: y[i] is the token that follows x[i].
    """
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.n_heads = N_HEADS
        self.d_head = D_MODEL // N_HEADS

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(D_MODEL, dim=2)
        # (B, T, C) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # is_causal=True applies the mask that stops position i seeing i+1.
        # Without it the model would trivially copy the answer and loss -> 0.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    """Pre-LN transformer block: normalise, sublayer, add residual."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # tokens exchange information
        x = x + self.ffn(self.ln2(x))    # each token is processed independently
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))

        if targets is None:
            return logits, None

        # Cross-entropy over every position at once. One forward pass on a
        # T-token window produces T training signals -- that parallelism is
        # exactly what the causal mask buys you.
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, source, iters=20):
    model.eval()
    losses = [model(*get_batch(source))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def sample(model, n=180):
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, n)[0].tolist()
    return "".join(itos[i] for i in out).replace("\n", " ")


def main():
    model = TinyGPT()
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("=" * 72)
    print("PRETRAINING A TRANSFORMER FROM RANDOM WEIGHTS")
    print("=" * 72)
    print(f"vocabulary      : {VOCAB_SIZE} characters")
    print(f"training tokens : {len(train_data):,}")
    print(f"parameters      : {n_params:,}")
    print(f"\nA model guessing uniformly scores ln({VOCAB_SIZE}) = {math.log(VOCAB_SIZE):.3f}")
    print("That is where an untrained model starts. Watch it fall.\n")

    print(f"{'step':>6} {'train':>8} {'val':>8} {'perplexity':>11}")
    print("-" * 36)

    start_time = time.time()
    for step in range(STEPS + 1):
        if step % 250 == 0:
            tr, va = estimate_loss(model, train_data), estimate_loss(model, val_data)
            print(f"{step:>6} {tr:>8.3f} {va:>8.3f} {math.exp(va):>11.1f}")

        x, y = get_batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"\ntrained in {time.time() - start_time:.1f}s on {torch.get_default_dtype()} CPU")

    print("\n" + "=" * 72)
    print("SAMPLES")
    print("=" * 72)
    print(f"\n{sample(model)}\n")

    print("=" * 72)
    print("WHAT JUST HAPPENED")
    print("=" * 72)
    print(
        f"Loss fell from ~{math.log(VOCAB_SIZE):.2f} (uniform guessing) toward ~1.5.\n"
        "Perplexity is exp(loss): at loss 1.5 the model is effectively choosing\n"
        "between ~4.5 characters instead of the full vocabulary.\n\n"
        "Nothing was labelled. The next character was always sitting in the data,\n"
        "so the text supervised itself. That is the entire trick that makes\n"
        "training on the whole internet possible.\n\n"
        "Scale this up -- BPE tokens instead of characters, 15 trillion tokens\n"
        "instead of 20,000, 32 layers instead of 3, thousands of GPUs for weeks --\n"
        "and you have Llama 3. The loop above does not change."
    )


if __name__ == "__main__":
    main()
