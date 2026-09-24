"""
How many GPUs do I need, and how should I split the model across them?

Distributed serving starts with arithmetic, not code. This planner does the
arithmetic for you and explains each number:

    weights  = parameters x bytes per parameter        (fixed cost)
    KV cache = 2 x layers x kv_heads x head_dim x bytes x tokens x users
    per GPU  = weights / GPUs + KV / KV shards + working space

Then it suggests a parallelism plan following the standard advice:
one GPU if it fits; tensor parallelism inside a node; pipeline parallelism
across nodes.

No dependencies.   python gpu_planner.py
"""

import math

GiB = 1024 ** 3

# Architecture numbers come from each model's published config.json.
MODELS = {
    "Llama 3.1 8B":   dict(params=8.0e9,  layers=32,  attn_heads=32,  kv_heads=8, head_dim=128),
    "Qwen2.5 32B":    dict(params=32.8e9, layers=64,  attn_heads=40,  kv_heads=8, head_dim=128),
    "Llama 3.1 70B":  dict(params=70.6e9, layers=80,  attn_heads=64,  kv_heads=8, head_dim=128),
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
CANDIDATE_GPU_COUNTS = (1, 2, 4, 8, 16, 32, 64)


def kv_bytes_per_token(cfg, bytes_per_value=2):
    return 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * bytes_per_value


def split(n, cfg, gpus_per_node):
    """Turn a GPU count into a (tensor parallel, pipeline parallel) plan and work
    out how many ways the KV cache actually divides.

    Tensor parallelism splits attention head-wise, so it can only split the KV
    cache across as many GPUs as there are KV heads. With grouped-query
    attention there are few KV heads (often 8), so past that the cache is
    DUPLICATED on each GPU rather than split. Pipeline parallelism always
    divides it, because each stage owns a different set of layers.
    """
    tp = min(n, gpus_per_node)
    pp = math.ceil(n / gpus_per_node)
    kv_shards = min(tp, cfg["kv_heads"]) * pp
    return tp, pp, kv_shards


def plan(model_name, gpu_name, context=8192, users=32, weight_bits=16,
         kv_bits=16, gpus_per_node=8):
    cfg = MODELS[model_name]
    weights_gb = cfg["params"] * weight_bits / 8 / GiB
    kv_gb = kv_bytes_per_token(cfg, kv_bits // 8) * context * users / GiB
    usable_per_gpu = GPUS[gpu_name] * MEMORY_UTILISATION - WORKING_SPACE_GB

    result = dict(model=model_name, gpu=gpu_name, weights=weights_gb, kv=kv_gb,
                  usable=usable_per_gpu, gpus=None, tp=None, pp=None, per_gpu=None)

    for n in CANDIDATE_GPU_COUNTS:
        # Tensor parallelism splits each layer head-wise, so the head count
        # must divide evenly by the tensor-parallel size.
        tp, pp, kv_shards = split(n, cfg, gpus_per_node)
        if cfg["attn_heads"] % tp != 0:
            continue
        per_gpu = weights_gb / n + kv_gb / kv_shards + WORKING_SPACE_GB
        if per_gpu <= usable_per_gpu + WORKING_SPACE_GB and per_gpu <= GPUS[gpu_name] * MEMORY_UTILISATION:
            result.update(gpus=n, tp=tp, pp=pp, per_gpu=per_gpu, kv_shards=kv_shards)
            break
    return result


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
                gpus_text, plan_text = "--", "needs >64"
            else:
                gpus_text = str(p["gpus"])
                plan_text = f"TP={p['tp']}" + (f", PP={p['pp']}" if p["pp"] > 1 else "")
            print(f"  {model_name:<16} {gpu_name:<15} {p['weights']:>7.1f} GB"
                  f" {p['kv']:>8.1f} GB {gpus_text:>6} {plan_text:>12}")
    print(
        "\n  Two independent costs. WEIGHTS are fixed -- a 70B model in fp16 needs about\n"
        "  132 GB before it serves anyone. The KV CACHE grows with context length AND\n"
        "  with the number of users, and it is usually what pushes you to more GPUs."
    )

    section("2. A WORKED EXAMPLE -- Llama 3.1 70B on A100 80GB")
    p = plan("Llama 3.1 70B", "A100 80GB", context, users)
    print(f"  weights (fp16)                  : {p['weights']:8.1f} GB")
    print(f"  KV cache ({users} users, {context:,} tokens) : {p['kv']:8.1f} GB")
    print(f"  total to hold                   : {p['weights'] + p['kv']:8.1f} GB")
    print(f"  usable per GPU                  : {p['usable']:8.1f} GB"
          f"   (80 GB x {MEMORY_UTILISATION:.0%} - {WORKING_SPACE_GB} GB working space)")
    print(f"  plan                            : TP={p['tp']}, PP={p['pp']}  ->  {p['gpus']} GPUs")
    print(f"  weights per GPU                 : {p['weights'] / p['gpus']:8.1f} GB"
          f"   (split {p['gpus']} ways)")
    print(f"  KV cache per GPU                : {p['kv'] / p['kv_shards']:8.1f} GB"
          f"   (split {p['kv_shards']} ways)")
    print(f"  total per GPU                   : {p['per_gpu']:8.1f} GB")
    print(
        "\n  Tensor parallelism splits every layer across the GPUs, so they all work on\n"
        "  the same request together and exchange results after each layer."
    )

    section("3. THE KV CACHE IS THE VARIABLE -- Llama 3.1 70B, A100 80GB")
    print(f"  {'context':>9} {'users':>7} {'KV cache':>11} {'total':>10} {'GPUs':>6} {'plan':>12}")
    print("  " + "-" * 60)
    for ctx, n_users in ((4096, 8), (8192, 32), (32768, 32), (131072, 8), (32768, 128)):
        p = plan("Llama 3.1 70B", "A100 80GB", ctx, n_users)
        gpus_text = str(p["gpus"]) if p["gpus"] else "--"
        plan_text = (f"TP={p['tp']}" + (f", PP={p['pp']}" if p["pp"] > 1 else "")) if p["gpus"] else "needs >64"
        print(f"  {ctx:>9,} {n_users:>7} {p['kv']:>9.1f} GB {p['weights'] + p['kv']:>8.1f} GB"
              f" {gpus_text:>6} {plan_text:>12}")
    print(
        "\n  Same model, same GPU -- the hardware bill changes by 8x depending on how\n"
        "  much context you promise and how many users you serve at once. Decide those\n"
        "  two numbers from the product BEFORE buying or renting hardware.\n"
        "  An fp8 KV cache (--kv-cache-dtype fp8) halves that column."
    )

    section("4. TENSOR PARALLEL SIZE MUST DIVIDE THE HEADS")
    print(f"  {'model':<16} {'attn heads':>11} {'KV heads':>9}   valid TP sizes")
    print("  " + "-" * 62)
    for name, cfg in MODELS.items():
        ok = [n for n in (1, 2, 4, 8, 16) if cfg["attn_heads"] % n == 0]
        print(f"  {name:<16} {cfg['attn_heads']:>11} {cfg['kv_heads']:>9}   {', '.join(map(str, ok))}")
    print(
        "\n  Tensor parallelism splits attention head-wise, so the attention head count\n"
        "  must divide evenly by the TP size -- asking for TP=6 on a 32-head model is\n"
        "  simply rejected. Qwen2.5 32B has 40 heads, so TP=16 is not available.\n\n"
        "  The KV head count matters too: with grouped-query attention there are often\n"
        "  only 8, so beyond TP=8 the KV cache is DUPLICATED on each GPU instead of\n"
        "  split. This planner accounts for that."
    )

    section("HOW TO USE THIS")
    print(
        "  1. Pick your context length and concurrent users from the product, not the\n"
        "     hardware. They set the KV cache, which usually sets the GPU count.\n"
        "  2. Prefer ONE GPU if it fits: no communication, no complexity.\n"
        "  3. Then tensor parallelism WITHIN a node (fast NVLink between GPUs).\n"
        "  4. Only then pipeline parallelism ACROSS nodes (slower network between them).\n"
        "  5. Need more throughput rather than more memory? Run several REPLICAS behind\n"
        "     a load balancer (data parallelism) instead of a bigger split.\n\n"
        "  Edit MODELS and GPUS at the top for your own hardware. These are estimates\n"
        "  that ignore fragmentation and framework overhead: always confirm against the\n"
        "  KV-cache capacity the engine reports when it starts."
    )


if __name__ == "__main__":
    main()
