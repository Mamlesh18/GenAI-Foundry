"""
How many GPUs do I need, and how should I split the model across them?

Distributed serving starts with arithmetic, not code. This planner does the
arithmetic for you and explains each number:

    weights  = parameters x bytes per parameter        (fixed cost)
    KV cache = 2 x layers x kv_heads x head_dim x bytes x tokens x users
    per GPU  = (weights + KV) / number_of_GPUs + working space

Then it suggests a parallelism plan following the standard advice:
one GPU if it fits; tensor parallelism inside a node; pipeline parallelism
across nodes.

No dependencies.   python gpu_planner.py
"""

import math

GiB = 1024 ** 3

# Architecture numbers come from each model's published config.json.
MODELS = {
    "Llama 3.1 8B":   dict(params=8.0e9,  layers=32, attn_heads=32, kv_heads=8,  head_dim=128),
    "Qwen2.5 32B":    dict(params=32.8e9, layers=64, attn_heads=40, kv_heads=8,  head_dim=128),
    "Llama 3.1 70B":  dict(params=70.6e9, layers=80, attn_heads=64, kv_heads=8,  head_dim=128),
    "Llama 3.1 405B": dict(params=406e9,  layers=126, attn_heads=128, kv_heads=8, head_dim=128),
}

GPUS = {
    "RTX 4090 24GB": 24,
    "L40S 48GB": 48,
    "A100 80GB": 80,
    "H100 80GB": 80,
}

MEMORY_UTILISATION = 0.90    # engines leave headroom; vLLM's --gpu-memory-utilization
WORKING_SPACE_GB = 2.0       # activations, CUDA graphs, fragmentation, per GPU


def kv_bytes_per_token(cfg, bytes_per_value=2):
    return 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * bytes_per_value


def plan(model_name, gpu_name, context=8192, users=32, weight_bits=16,
         kv_bits=16, gpus_per_node=8):
    cfg = MODELS[model_name]
    gpu_gb = GPUS[gpu_name]

    weights_gb = cfg["params"] * weight_bits / 8 / GiB
    kv_gb = kv_bytes_per_token(cfg, kv_bits // 8) * context * users / GiB
    usable_per_gpu = gpu_gb * MEMORY_UTILISATION - WORKING_SPACE_GB

    # Tensor parallelism shards BOTH weights and KV cache, so both divide by n.
    # Valid n must divide the attention head count and the KV head count.
    valid = [n for n in (1, 2, 4, 8, 16, 32, 64)
             if cfg["attn_heads"] % n == 0 and cfg["kv_heads"] % max(1, min(n, cfg["kv_heads"])) == 0]
    needed = None
    for n in valid:
        if usable_per_gpu <= 0:
            break
        if (weights_gb + kv_gb) / n <= usable_per_gpu:
            needed = n
            break

    if needed is None:
        return dict(model=model_name, gpu=gpu_name, weights=weights_gb, kv=kv_gb,
                    gpus=None, tp=None, pp=None, note="does not fit in the sizes tried")

    tp = min(needed, gpus_per_node)
    pp = math.ceil(needed / gpus_per_node)
    return dict(model=model_name, gpu=gpu_name, weights=weights_gb, kv=kv_gb,
                gpus=tp * pp, tp=tp, pp=pp,
                per_gpu=(weights_gb + kv_gb) / (tp * pp) + WORKING_SPACE_GB,
                usable=usable_per_gpu, note=None)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    context, users = 8192, 32

    section(f"1. WHAT FITS WHERE?   ({context:,}-token context, {users} concurrent users, fp16)")
    print(f"  {'model':<16} {'GPU':<15} {'weights':>9} {'KV cache':>10} {'GPUs':>6} {'plan':>12}")
    print("  " + "-" * 74)
    for model_name in MODELS:
        for gpu_name in ("RTX 4090 24GB", "A100 80GB"):
            p = plan(model_name, gpu_name, context, users)
            if p["gpus"] is None:
                plan_text, gpus_text = "too big", "--"
            else:
                plan_text = f"TP={p['tp']}" + (f", PP={p['pp']}" if p["pp"] > 1 else "")
                gpus_text = str(p["gpus"])
            print(f"  {model_name:<16} {gpu_name:<15} {p['weights']:>7.1f} GB"
                  f" {p['kv']:>8.1f} GB {gpus_text:>6} {plan_text:>12}")
    print(
        "\n  Two independent costs. WEIGHTS are fixed -- a 70B model in fp16 needs about\n"
        "  132 GB before serving anyone. The KV CACHE grows with context length AND\n"
        "  with the number of users, and it is usually what pushes you to more GPUs."
    )

    section("2. A WORKED EXAMPLE -- Llama 3.1 70B on A100 80GB")
    p = plan("Llama 3.1 70B", "A100 80GB", context, users)
    total = p["weights"] + p["kv"]
    print(f"  weights (fp16)             : {p['weights']:8.1f} GB")
    print(f"  KV cache ({users} users x {context:,}) : {p['kv']:8.1f} GB")
    print(f"  total to hold              : {total:8.1f} GB")
    print(f"  usable per GPU             : {p['usable']:8.1f} GB"
          f"   (80 GB x {MEMORY_UTILISATION:.0%} - {WORKING_SPACE_GB} GB working space)")
    print(f"  minimum GPUs by memory     : {math.ceil(total / p['usable']):8}")
    print(f"  plan                       : TP={p['tp']}, PP={p['pp']}  ->  {p['gpus']} GPUs")
    print(f"  each GPU then holds        : {p['per_gpu']:8.1f} GB")
    print(
        "\n  Tensor parallelism splits every layer, so BOTH the weights and the KV cache\n"
        "  divide by the number of GPUs. All of them work on the same request together."
    )

    section("3. THE KV CACHE IS THE VARIABLE -- Llama 3.1 70B, A100 80GB")
    print(f"  {'context':>9} {'users':>7} {'KV cache':>10} {'total':>9} {'GPUs needed':>12}")
    print("  " + "-" * 52)
    for ctx, n_users in ((4096, 8), (8192, 32), (32768, 32), (32768, 128), (131072, 32)):
        p = plan("Llama 3.1 70B", "A100 80GB", ctx, n_users)
        gpus = p["gpus"] if p["gpus"] else ">64"
        print(f"  {ctx:>9,} {n_users:>7} {p['kv']:>8.1f} GB {p['weights'] + p['kv']:>7.1f} GB"
              f" {str(gpus):>12}")
    print(
        "\n  Same model, same GPU -- the answer changes by 4x depending on how much\n"
        "  context you promise and how many users you serve at once. Decide those\n"
        "  two numbers BEFORE buying or renting hardware.\n"
        "  An fp8 KV cache (--kv-cache-dtype fp8) halves that column."
    )

    section("4. TENSOR PARALLEL SIZE MUST DIVIDE THE HEADS")
    print(f"  {'model':<16} {'attn heads':>11} {'KV heads':>9}   valid TP sizes")
    print("  " + "-" * 60)
    for name, cfg in MODELS.items():
        ok = [n for n in (1, 2, 4, 8, 16) if cfg["attn_heads"] % n == 0]
        warn = " (KV heads limit efficiency past %d)" % cfg["kv_heads"]
        print(f"  {name:<16} {cfg['attn_heads']:>11} {cfg['kv_heads']:>9}   "
              f"{', '.join(map(str, ok))}{warn if max(ok) > cfg['kv_heads'] else ''}")
    print(
        "\n  Tensor parallelism splits attention head-wise, so the head count must divide\n"
        "  evenly by the TP size -- 'TP=6' on a 32-head model is simply rejected.\n"
        "  With grouped-query attention there are only a few KV heads, so beyond that\n"
        "  number the KV cache has to be duplicated across GPUs instead of split."
    )

    section("HOW TO USE THIS")
    print(
        "  1. Pick your context length and concurrent users from the product, not the\n"
        "     hardware. They set the KV cache, which usually sets the GPU count.\n"
        "  2. Prefer ONE GPU if it fits: no communication, no complexity.\n"
        "  3. Then tensor parallelism WITHIN a node (fast NVLink between GPUs).\n"
        "  4. Only then pipeline parallelism ACROSS nodes (slow network between them).\n"
        "  5. Need more throughput, not more memory? Run several REPLICAS behind a\n"
        "     load balancer (data parallelism) instead of a bigger split.\n\n"
        "  Edit MODELS and GPUS at the top for your own hardware. These are estimates:\n"
        "  always confirm against what the engine reports at startup."
    )


if __name__ == "__main__":
    main()
