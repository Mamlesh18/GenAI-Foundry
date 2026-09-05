"""
Scaled dot-product attention and multi-head attention, in pure NumPy.

The point of this file is not efficiency -- it is that you can read every line and see
exactly what the formula from the paper does:

    Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V

Run it:
    pip install numpy
    python self_attention_from_scratch.py
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# 1. The core operation
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Numerically stable softmax: subtract the max before exponentiating."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None, scale=True):
    """
    Q: (n_queries, d_k)
    K: (n_keys,    d_k)
    V: (n_keys,    d_v)

    Returns (output, attention_weights).

    Set scale=False to see why the 1/sqrt(d_k) term exists (exercise 2).
    """
    d_k = Q.shape[-1]

    # Step 1: score every query against every key -> (n_queries, n_keys)
    scores = Q @ K.T

    # Step 2: scale, so softmax does not saturate for large d_k
    if scale:
        scores = scores / np.sqrt(d_k)

    # Step 3: mask BEFORE the softmax, so masked positions get ~zero probability.
    # Masking after softmax would leave the remaining weights not summing to 1.
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    weights = softmax(scores, axis=-1)

    # Step 4: weighted average of the value vectors
    return weights @ V, weights


def causal_mask(n):
    """Lower-triangular True matrix: position i may attend to positions 0..i only."""
    return np.tril(np.ones((n, n), dtype=bool))


# ---------------------------------------------------------------------------
# 2. Multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    """h attention heads in parallel, each of dimension d_model // h."""

    def __init__(self, d_model=64, n_heads=4, rng=np.random):
        assert d_model % n_heads == 0, "d_model must divide evenly among heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Learned projections. In a real model these are trained; here they are random.
        scale = 1.0 / np.sqrt(d_model)
        self.W_q = rng.randn(d_model, d_model) * scale
        self.W_k = rng.randn(d_model, d_model) * scale
        self.W_v = rng.randn(d_model, d_model) * scale
        self.W_o = rng.randn(d_model, d_model) * scale

    def __call__(self, X, mask=None):
        """X: (seq_len, d_model) -> (output, per_head_attention_weights)"""
        n = X.shape[0]

        # Project once, then split into heads.
        Q = (X @ self.W_q).reshape(n, self.n_heads, self.d_head).transpose(1, 0, 2)
        K = (X @ self.W_k).reshape(n, self.n_heads, self.d_head).transpose(1, 0, 2)
        V = (X @ self.W_v).reshape(n, self.n_heads, self.d_head).transpose(1, 0, 2)

        heads, all_weights = [], []
        for h in range(self.n_heads):
            out, w = scaled_dot_product_attention(Q[h], K[h], V[h], mask=mask)
            heads.append(out)
            all_weights.append(w)

        # Concatenate the heads back to d_model, then mix them with W_o.
        concat = np.concatenate(heads, axis=-1)
        return concat @ self.W_o, np.stack(all_weights)


# ---------------------------------------------------------------------------
# 3. Positional encoding (sinusoidal, as in the paper)
# ---------------------------------------------------------------------------

def sinusoidal_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]                     # (seq_len, 1)
    i = np.arange(d_model)[None, :]                       # (1, d_model)
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------

def show_attention(tokens, weights, title):
    print(f"\n{title}")
    print("        " + "".join(f"{t:>10}" for t in tokens))
    for i, tok in enumerate(tokens):
        row = "".join(f"{w:>10.3f}" for w in weights[i])
        print(f"{tok:>8}{row}")


def main():
    tokens = ["the", "animal", "crossed", "the", "street", "because", "it", "was", "tired"]
    n, d_model = len(tokens), 64

    # Fake token embeddings. A real model looks these up from a learned table.
    embeddings = np.random.randn(n, d_model)

    # Without positional encoding, attention cannot tell "dog bites man" from "man bites dog".
    X = embeddings + sinusoidal_positional_encoding(n, d_model)

    print("=" * 74)
    print("SINGLE-HEAD SELF-ATTENTION")
    print("=" * 74)
    print(f"tokens          : {tokens}")
    print(f"input X         : {X.shape}   (seq_len, d_model)")

    W = np.random.randn(d_model, 16) / np.sqrt(d_model)
    Q, K, V = X @ W, X @ W, X @ W
    print(f"Q, K, V         : {Q.shape}   (seq_len, d_k)")

    out, weights = scaled_dot_product_attention(Q, K, V)
    print(f"attention matrix: {weights.shape}   (seq_len, seq_len)")
    print(f"output          : {out.shape}   (seq_len, d_v)")
    print(f"\nevery row sums to 1: {np.allclose(weights.sum(axis=-1), 1.0)}")
    show_attention(tokens, weights, "Bidirectional attention (encoder style):")

    print("\n" + "=" * 74)
    print("CAUSAL MASKING (decoder style -- what GPT-like models use)")
    print("=" * 74)
    mask = causal_mask(n)
    _, causal_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    show_attention(tokens, causal_weights, "Each token attends only to itself and the past:")
    print("\nNote the upper triangle is exactly 0.0 -- token i cannot see token i+1.")

    print("\n" + "=" * 74)
    print("MULTI-HEAD ATTENTION")
    print("=" * 74)
    mha = MultiHeadAttention(d_model=d_model, n_heads=4)
    mha_out, head_weights = mha(X, mask=mask)
    print(f"heads           : {mha.n_heads} x {mha.d_head} dims = {mha.d_model}")
    print(f"per-head weights: {head_weights.shape}   (n_heads, seq_len, seq_len)")
    print(f"output          : {mha_out.shape}   (seq_len, d_model)")
    print("\nEach head learns a different relationship. With random (untrained) weights")
    print("the patterns are meaningless -- train the model and they become interpretable.")

    print("\n" + "=" * 74)
    print("WHY WE DIVIDE BY sqrt(d_k)  (exercise 2)")
    print("=" * 74)
    d_k = 512
    Qb, Kb, Vb = (np.random.randn(4, d_k) for _ in range(3))

    print(f"d_k = {d_k}, so raw dot products have standard deviation ~sqrt(d_k) = {np.sqrt(d_k):.1f}\n")
    for label, use_scale in (("with    /sqrt(d_k)", True), ("without /sqrt(d_k)", False)):
        _, w = scaled_dot_product_attention(Qb, Kb, Vb, scale=use_scale)
        # Entropy measures how spread out the attention is.
        # log(4) = 1.386 nats is a perfectly uniform distribution over 4 keys; 0 is one-hot.
        entropy = float(-np.sum(w[0] * np.log(w[0] + 1e-12)))
        print(f"{label}: row 0 = {w[0]}  entropy = {entropy:.4f} nats")

    print("\nUnscaled, softmax collapses onto a single key: entropy near 0, the distribution")
    print("is effectively one-hot, and softmax's gradient there is ~0 -- the layer stops")
    print("learning. Scaling keeps the scores near unit variance, so gradients survive.")


if __name__ == "__main__":
    main()
