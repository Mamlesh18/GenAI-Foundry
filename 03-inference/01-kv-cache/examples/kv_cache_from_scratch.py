"""
The KV cache, from scratch -- and proof that it changes SPEED, never OUTPUT.

When a model generates text it produces one token at a time. Without a cache,
every new token means re-reading the ENTIRE sequence so far. With a cache, the
keys and values of earlier tokens are stored, and each new token only has to
compute its own.

This script builds a tiny transformer and generates the same text both ways:

  1. the outputs are identical (same tokens, same logits to float precision)
  2. the work done is wildly different -- quadratic vs linear
  3. the price is memory, and we measure exactly how much

It also shows the two phases every LLM request goes through:
  PREFILL -- read the whole prompt in one parallel pass
  DECODE  -- produce new tokens one at a time

    pip install torch
    python kv_cache_from_scratch.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_grad_enabled(False)      # inference only -- no gradients needed

VOCAB = 1000
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
MAX_LEN = 2048
D_HEAD = D_MODEL // N_HEADS


# ---------------------------------------------------------------------------
# A tiny decoder-only transformer whose attention can use a cache
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(D_MODEL, dim=2)
        q, k, v = (t.view(B, T, N_HEADS, D_HEAD).transpose(1, 2) for t in (q, k, v))

        if cache is not None:
            # THE KV CACHE. Keys and values of every earlier token are kept, so
            # the new token(s) only add their own. Queries are never cached:
            # only the newest token asks a question.
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # Causal mask. The T new queries sit at the END of an S-long sequence,
        # so query i may attend to key positions 0 .. (S - T + i).
        S = k.shape[2]
        mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D_MODEL))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL)
        )

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, idx, caches=None, start_pos=0):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(start_pos, start_pos + T))
        for i, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[i])
        return self.head(self.ln(x))


# ---------------------------------------------------------------------------
# Two ways to generate the same text
# ---------------------------------------------------------------------------

def generate_without_cache(model, prompt, n_new):
    """Re-run the model over the WHOLE sequence for every new token."""
    seq, positions, logits_log = prompt.clone(), 0, []
    for _ in range(n_new):
        logits = model(seq)                       # reads every token again
        positions += seq.shape[1]
        last = logits[:, -1, :]
        logits_log.append(last)
        seq = torch.cat([seq, last.argmax(-1, keepdim=True)], dim=1)
    return seq, positions, torch.cat(logits_log)


def generate_with_cache(model, prompt, n_new):
    """Prefill the prompt once, then feed ONE token per step."""
    caches = [{} for _ in range(N_LAYERS)]
    logits = model(prompt, caches, start_pos=0)   # PREFILL: whole prompt, one pass
    positions = prompt.shape[1]
    seq, logits_log = prompt.clone(), []

    for step in range(n_new):
        last = logits[:, -1, :]
        logits_log.append(last)
        nxt = last.argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        if step < n_new - 1:
            # DECODE: only the newest token goes through the model
            logits = model(nxt, caches, start_pos=seq.shape[1] - 1)
            positions += 1
    return seq, positions, torch.cat(logits_log), caches


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def main():
    model = TinyLM().eval()
    prompt = torch.randint(0, VOCAB, (1, 32))
    generate_with_cache(model, prompt, 8)          # warm-up so timings are fair

    print("=" * 76)
    print("1. SAME OUTPUT, VERY DIFFERENT WORK")
    print("=" * 76)
    print("  prompt: 32 tokens. 'positions' = token positions pushed through the model.\n")
    print(f"  {'new tokens':>10} {'positions':>11} {'positions':>11} {'time':>8} {'time':>8}"
          f" {'speedup':>8} {'same':>6}")
    print(f"  {'':>10} {'no cache':>11} {'with cache':>11} {'no cache':>8} {'cache':>8}"
          f" {'':>8} {'text?':>6}")
    print("  " + "-" * 70)

    for n_new in (64, 256, 512):
        (seq_a, pos_a, log_a), t_a = timed(generate_without_cache, model, prompt, n_new)
        (seq_b, pos_b, log_b, _), t_b = timed(generate_with_cache, model, prompt, n_new)
        same = torch.equal(seq_a, seq_b)
        print(f"  {n_new:>10} {pos_a:>11,} {pos_b:>11,} {t_a:>7.2f}s {t_b:>7.2f}s"
              f" {t_a / t_b:>7.1f}x {str(same):>6}")

    max_diff = (log_a - log_b).abs().max().item()
    print(f"\n  largest logit difference (512 tokens): {max_diff:.2e}  -- floating-point noise")
    print(
        "\n  Without a cache the work grows QUADRATICALLY: token 500 re-reads 531\n"
        "  positions just to produce one token. With a cache it grows LINEARLY: one\n"
        "  new position per token. The longer the output, the bigger the gap.\n"
        "  (Attention itself still reads every cached key, so a decode step does get\n"
        "  a little slower as the context grows -- but nothing is recomputed.)"
    )

    print("\n" + "=" * 76)
    print("2. THE PRICE: MEMORY")
    print("=" * 76)
    _, _, _, caches = generate_with_cache(model, prompt, 512)
    cache_bytes = sum(c["k"].numel() * 4 + c["v"].numel() * 4 for c in caches)
    seq_len = caches[0]["k"].shape[2]
    per_token = 2 * N_LAYERS * N_HEADS * D_HEAD * 4
    print(f"  cache tensor shape per layer : {tuple(caches[0]['k'].shape)}"
          "  (batch, heads, tokens, head_dim)")
    print(f"  tokens in cache              : {seq_len}")
    print(f"  total cache size             : {cache_bytes / 1e6:.2f} MB (float32)")
    print(f"  per token                    : {per_token:,} bytes")
    print(f"  formula                      : 2 x layers x heads x head_dim x bytes"
          f" = 2 x {N_LAYERS} x {N_HEADS} x {D_HEAD} x 4")
    print(
        "\n  Tiny here. For Llama 3.1 8B it is ~128 KB per token -- about 1 GB for a\n"
        "  single 8,000-token conversation. Run kv_cache_calculator.py for real models.\n"
        "  That memory bill is what the rest of this track (batching, paged attention,\n"
        "  KV quantization) exists to manage."
    )

    print("\n" + "=" * 76)
    print("3. PREFILL vs DECODE -- the two phases of every request")
    print("=" * 76)
    long_prompt = torch.randint(0, VOCAB, (1, 512))

    caches = [{} for _ in range(N_LAYERS)]
    _, t_prefill = timed(model, long_prompt, caches, 0)

    caches = [{} for _ in range(N_LAYERS)]
    model(long_prompt[:, :1], caches, 0)
    start = time.perf_counter()
    for i in range(1, 512):
        model(long_prompt[:, i:i + 1], caches, i)
    t_decode = time.perf_counter() - start

    print(f"  process 512 tokens in ONE parallel pass (prefill) : {t_prefill * 1000:8.1f} ms")
    print(f"  process 512 tokens ONE AT A TIME (decode)         : {t_decode * 1000:8.1f} ms")
    print(f"  prefill is {t_decode / t_prefill:.0f}x faster for the same tokens")
    print(
        "\n  The prompt is known up front, so all of it goes through in one pass --\n"
        "  that sets TIME TO FIRST TOKEN. Output tokens do not exist yet, so they must\n"
        "  come one per pass -- that sets TIME PER OUTPUT TOKEN. On a GPU the gap is\n"
        "  far larger than on this CPU, because a GPU is built for parallel work.\n"
        "  This is why long prompts are cheap relative to long answers."
    )


if __name__ == "__main__":
    main()
