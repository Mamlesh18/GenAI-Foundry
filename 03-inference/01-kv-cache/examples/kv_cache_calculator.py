"""
KV cache memory calculator for real models.

    KV bytes per token = 2 x layers x kv_heads x head_dim x bytes_per_value
                         ^
                         one K and one V

Answers the questions that decide real deployments:
  - how much memory does one conversation's cache need?
  - why did GQA (grouped-query attention) matter so much?
  - how many users fit on one GPU?

No dependencies.   python kv_cache_calculator.py
"""

GiB = 1024 ** 3

# Architecture numbers from each model's published config.json.
MODELS = {
    "Qwen2.5 7B":    dict(params=7.6e9,  layers=28, attn_heads=28, kv_heads=4, head_dim=128),
    "Mistral 7B":    dict(params=7.2e9,  layers=32, attn_heads=32, kv_heads=8, head_dim=128),
    "Llama 3.1 8B":  dict(params=8.0e9,  layers=32, attn_heads=32, kv_heads=8, head_dim=128),
    "Llama 3.1 70B": dict(params=70.6e9, layers=80, attn_heads=64, kv_heads=8, head_dim=128),
}


def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value=2, **_):
    return 2 * layers * kv_heads * head_dim * bytes_per_value


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    section("1. HOW BIG IS THE CACHE?  (fp16 cache, batch size 1)")
    print(f"  {'model':<15} {'per token':>11} {'4K tokens':>11} {'32K tokens':>12} {'128K tokens':>13}")
    print("  " + "-" * 66)
    for name, cfg in MODELS.items():
        b = kv_bytes_per_token(**cfg)
        print(f"  {name:<15} {b / 1024:>8.0f} KB"
              f" {b * 4_000 / GiB:>8.2f} GB {b * 32_000 / GiB:>9.2f} GB {b * 128_000 / GiB:>10.2f} GB")
    print(
        "\n  One long conversation can need more memory than a small model's weights.\n"
        "  (Columns assume the model supports that context length.)"
    )

    section("2. WHY GQA MATTERED  (Llama 3.1 8B shape, 32 attention heads)")
    base = MODELS["Llama 3.1 8B"]
    print(f"  {'attention type':<32} {'KV heads':>9} {'per token':>11} {'32K tokens':>12}")
    print("  " + "-" * 68)
    for label, kv in (("MHA  multi-head (every head)", 32),
                      ("GQA  grouped-query (Llama 3)", 8),
                      ("MQA  multi-query (one shared)", 1)):
        b = kv_bytes_per_token(**{**base, "kv_heads": kv})
        print(f"  {label:<32} {kv:>9} {b / 1024:>8.0f} KB {b * 32_000 / GiB:>9.2f} GB")
    print(
        "\n  All three have 32 QUERY heads. They differ in how many heads get their own\n"
        "  keys and values. GQA shares one K/V set across a group of 4 query heads:\n"
        "  4x less cache, with quality close to full multi-head attention. It is why\n"
        "  long context became affordable, and why nearly every modern model uses it."
    )

    section("3. HOW MANY USERS FIT ON ONE GPU?")
    print(
        "  Serving engines like vLLM claim a fraction of GPU memory (--gpu-memory-utilization;\n"
        "  we use 90% here), load the weights, and give what is left to the KV cache.\n"
        "  (Real engines also reserve some memory for activations, so treat these as\n"
        "  optimistic upper bounds.)\n"
    )
    scenarios = [
        ("Llama 3.1 8B",  24, "fp16"),
        ("Llama 3.1 8B",  80, "fp16"),
        ("Llama 3.1 8B",  80, "fp8"),
        ("Llama 3.1 70B", 80, "fp16"),     # weights alone do not fit -- shown on purpose
        ("Llama 3.1 70B", 160, "fp16"),    # e.g. 2 x 80GB with tensor parallelism
    ]
    print(f"  {'model':<14} {'GPU mem':>8} {'KV dtype':>9} {'weights':>9} {'KV budget':>10}"
          f" {'users @4K':>10} {'users @32K':>11}")
    print("  " + "-" * 77)
    for name, gpu_gb, kv_dtype in scenarios:
        cfg = MODELS[name]
        weights = cfg["params"] * 2 / GiB                      # fp16 weights
        budget = gpu_gb * 0.90 - weights
        bpv = 2 if kv_dtype == "fp16" else 1
        per_token = kv_bytes_per_token(**cfg, bytes_per_value=bpv) / GiB
        if budget <= 0:
            print(f"  {name:<14} {gpu_gb:>6} GB {kv_dtype:>9} {weights:>6.1f} GB"
                  f" {'none':>10} {'--':>10} {'--':>11}   weights do not fit")
            continue
        print(f"  {name:<14} {gpu_gb:>6} GB {kv_dtype:>9} {weights:>6.1f} GB {budget:>7.1f} GB"
              f" {int(budget / (per_token * 4_000)):>10} {int(budget / (per_token * 32_000)):>11}")
    print(
        "\n  Read the rows top to bottom:\n"
        "  - a 24GB card holds an 8B model but only a handful of long conversations\n"
        "  - an FP8 KV cache doubles the users on the same hardware\n"
        "  - a 70B model in fp16 needs more than one 80GB GPU before it serves anyone\n"
        "\n  'Users' here assumes every conversation is at full length at once -- the\n"
        "  worst case. Real traffic is shorter on average, which is exactly the slack\n"
        "  that paged attention (module 04) is designed to exploit."
    )

    section("TRY IT")
    print(
        "  Open any model's config.json on Hugging Face and read:\n"
        "    num_hidden_layers    -> layers\n"
        "    num_key_value_heads  -> kv_heads   (missing? then it equals attention heads)\n"
        "    head_dim             -> head_dim   (missing? hidden_size / num_attention_heads)\n"
        "  Add it to MODELS above and rerun."
    )


if __name__ == "__main__":
    main()
