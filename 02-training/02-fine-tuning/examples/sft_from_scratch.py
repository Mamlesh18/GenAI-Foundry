"""
Supervised fine-tuning from scratch -- and what loss masking actually does.

SFT is pretraining with two changes:
  1. the data is (instruction, response) pairs in a chat template
  2. loss is computed ONLY on the response tokens

Change 2 is the one people get wrong, and it is invisible in the loss curve.
This script trains the SAME model twice -- once masked, once not -- and shows
you the difference in what it learns.

Character-level and tiny so it runs on a CPU in under a minute.

    pip install torch
    python sft_from_scratch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

BLOCK_SIZE = 96
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
STEPS = 1200
LR = 3e-3
IGNORE_INDEX = -100     # the label value F.cross_entropy skips


# ---------------------------------------------------------------------------
# 1. The dataset: instruction -> response pairs
# ---------------------------------------------------------------------------

PAIRS = [
    ("what is the capital of france", "the capital of france is paris"),
    ("what is the capital of japan", "the capital of japan is tokyo"),
    ("what is the capital of italy", "the capital of italy is rome"),
    ("what is the capital of spain", "the capital of spain is madrid"),
    ("what is the capital of egypt", "the capital of egypt is cairo"),
    ("what is the capital of india", "the capital of india is delhi"),
    ("what is the capital of kenya", "the capital of kenya is nairobi"),
    ("what is the capital of chile", "the capital of chile is santiago"),
]

# A chat template. Every model family has its own; the principle is identical:
# unambiguous markers for where the user stops and the assistant starts.
USER_TAG = "<u>"
ASSISTANT_TAG = "<a>"
END_TAG = "<e>"


def build_text(instruction, response):
    return f"{USER_TAG}{instruction}{ASSISTANT_TAG}{response}{END_TAG}"


corpus = "".join(build_text(i, r) for i, r in PAIRS)
chars = sorted(set(corpus))
VOCAB_SIZE = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)


def make_example(instruction, response, mask_prompt=True):
    """Return (input_ids, labels).

    labels are input_ids shifted by one. When mask_prompt is True, every label
    position belonging to the prompt is set to IGNORE_INDEX so no loss is
    computed there -- the model is graded only on what it should GENERATE.
    """
    prompt = f"{USER_TAG}{instruction}{ASSISTANT_TAG}"
    full = prompt + response + END_TAG

    ids = encode(full)
    x = ids[:-1]
    y = ids[1:]

    if mask_prompt:
        # Position i of y is the token that follows x[i]. The response starts at
        # index len(prompt), so labels before that predict prompt tokens.
        n_prompt = len(encode(prompt))
        y = [IGNORE_INDEX] * (n_prompt - 1) + y[n_prompt - 1:]

    pad = BLOCK_SIZE - len(x)
    return (
        torch.tensor(x + [0] * pad),
        torch.tensor(y + [IGNORE_INDEX] * pad),   # padding never contributes loss
    )


def make_batch(mask_prompt=True):
    xs, ys = zip(*[make_example(i, r, mask_prompt) for i, r in PAIRS])
    return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------
# 2. A small decoder-only model (same architecture as ../../01-pretraining/)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL)
        )

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D_MODEL, dim=2)
        shape = (B, T, N_HEADS, C // N_HEADS)
        q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).contiguous().view(B, T, C))
        return x + self.ffn(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, idx, labels=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if labels is None:
            return logits, None
        # ignore_index is what makes masking work: these positions are skipped
        # entirely, and do not even contribute to the mean.
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_new=45):
        """Greedy decode until END_TAG is produced or max_new tokens are used.

        END_TAG is several characters long, so the stop check has to look at the
        decoded tail -- comparing a single new token against it never matches.
        """
        idx = torch.tensor([encode(prompt)])
        for _ in range(max_new):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if decode(idx[0].tolist()).endswith(END_TAG):
                break
        return decode(idx[0].tolist())[len(prompt):]


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

def train(mask_prompt):
    model = TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    x, y = make_batch(mask_prompt)
    for _ in range(STEPS):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, loss.item()


def show_masking():
    print("=" * 74)
    print("WHAT LOSS MASKING LOOKS LIKE")
    print("=" * 74)

    instruction, response = PAIRS[0]
    x, y = make_example(instruction, response, mask_prompt=True)
    n = len(encode(f"{USER_TAG}{instruction}{ASSISTANT_TAG}{response}{END_TAG}")) - 1

    print(f"\ninstruction : {instruction!r}")
    print(f"response    : {response!r}")
    print(f"\ntemplate    : {build_text(instruction, response)}")

    print("\nPosition by position (padding trimmed):\n")
    print(f"  {'idx':>4}  {'input':>7}  {'label':>7}   trained on?")
    print("  " + "-" * 42)
    for i in range(n):
        inp = decode([x[i].item()])
        lab = y[i].item()
        lab_s = "IGNORE" if lab == IGNORE_INDEX else repr(decode([lab]))
        trained = "no  <- prompt" if lab == IGNORE_INDEX else "YES <- response"
        print(f"  {i:>4}  {inp!r:>7}  {lab_s:>7}   {trained}")

    n_masked = int((y[:n] == IGNORE_INDEX).sum())
    print(f"\n  {n_masked} of {n} positions masked out.")
    print("  The model is graded ONLY on tokens it should generate.")


def main():
    show_masking()

    print("\n" + "=" * 74)
    print("TRAINING BOTH WAYS")
    print("=" * 74)

    masked_model, masked_loss = train(mask_prompt=True)
    print(f"\n  with masking    : final loss {masked_loss:.4f}")

    unmasked_model, unmasked_loss = train(mask_prompt=False)
    print(f"  without masking : final loss {unmasked_loss:.4f}")
    print("\n  Both losses look fine. The loss curve will NOT tell you which is correct.")

    print("\n" + "=" * 74)
    print("1. ANSWERING -- the job you actually trained for")
    print("=" * 74)
    print("  Both do this well. Masking is not about whether it can answer.\n")
    for instruction, _ in PAIRS[:3]:
        prompt = f"{USER_TAG}{instruction}{ASSISTANT_TAG}"
        print(f"  {instruction}")
        print(f"     masked   : {masked_model.generate(prompt)!r}")
        print(f"     unmasked : {unmasked_model.generate(prompt)!r}")

    print("\n" + "=" * 74)
    print("2. WHAT ELSE THEY LEARNED -- generating from the <u> tag alone")
    print("=" * 74)
    print(
        "  Here we give each model ONLY the user tag and let it continue.\n"
        "  This asks: did it learn to write the USER's half of the conversation?\n"
    )
    for name, model in (("masked  ", masked_model), ("unmasked", unmasked_model)):
        print(f"  {name} : {model.generate(USER_TAG, max_new=60)!r}")

    # Quantify it: how well does each model predict the PROMPT tokens?
    x_um, y_um = make_batch(mask_prompt=False)          # labels on every position
    x_m, y_m = make_batch(mask_prompt=True)             # labels on the response only
    prompt_only = torch.where(y_m == IGNORE_INDEX, y_um, torch.full_like(y_um, IGNORE_INDEX))

    with torch.no_grad():
        masked_prompt_loss = F.cross_entropy(
            masked_model(x_m)[0].view(-1, VOCAB_SIZE),
            prompt_only.reshape(-1), ignore_index=IGNORE_INDEX).item()
        unmasked_prompt_loss = F.cross_entropy(
            unmasked_model(x_um)[0].view(-1, VOCAB_SIZE),
            prompt_only.reshape(-1), ignore_index=IGNORE_INDEX).item()

    print("\n  Loss measured ONLY on prompt tokens (lower = better at writing questions):")
    print(f"     masked   : {masked_prompt_loss:.3f}")
    print(f"     unmasked : {unmasked_prompt_loss:.3f}")

    print("\n" + "=" * 74)
    print("THE POINT")
    print("=" * 74)
    print(
        "Both models answer. The difference is what ELSE they spent capacity on.\n\n"
        "The unmasked model is much better at predicting prompt tokens -- because it\n"
        "was trained to. Every gradient it received on those 34 positions taught it to\n"
        "generate text the user will always supply anyway. That is capacity and\n"
        "training signal spent on a task you will never ask it to do.\n\n"
        "On a toy with 8 examples this is merely wasteful. At real scale, on real data,\n"
        "it is the difference between a model that answers and one that carries on and\n"
        "writes your next question for you.\n\n"
        "In production you get correct masking from:\n"
        "  TRL           -- SFTTrainer with completion-only loss\n"
        "  LLaMA-Factory -- handled automatically by `stage: sft`\n"
        "Both set label positions to -100 exactly as shown at the top of this output."
    )


if __name__ == "__main__":
    main()
