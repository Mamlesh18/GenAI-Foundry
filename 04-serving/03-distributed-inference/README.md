# 03 · Distributed Inference

> **In one sentence:** when a model will not fit on one GPU — or one GPU cannot serve enough users —
> you split the work across several GPUs, and sometimes several machines. *How* you split it decides
> how much they have to talk to each other, which decides your speed.

---

## 1. Start with the arithmetic

Two costs decide everything (both from [03-inference](../../03-inference/)):

```
weights  = parameters x bytes per parameter       fixed
KV cache = 2 x layers x kv_heads x head_dim x bytes x context x users   variable
```

From [`gpu_planner.py`](examples/gpu_planner.py), at 8K context and 32 concurrent users:

| Model | Weights (fp16) | KV cache | On A100 80GB | On RTX 4090 24GB |
|---|---|---|---|---|
| Llama 3.1 8B | 14.9 GB | 32.0 GB | 1 GPU | 4 GPUs (TP=4) |
| Qwen2.5 32B | 61.1 GB | 64.0 GB | 2 GPUs (TP=2) | 8 GPUs (TP=8) |
| Llama 3.1 70B | 131.5 GB | 80.0 GB | 4 GPUs (TP=4) | 16 GPUs (TP=8, PP=2) |
| Llama 3.1 405B | 756.2 GB | 126.0 GB | 16 GPUs (TP=8, PP=2) | 64 GPUs (TP=8, PP=8) |

Notice the KV cache is often **bigger than the weights**. And it depends on promises you make in
the product, not on the model — same 70B model on the same A100s:

| Context | Users | KV cache | GPUs needed |
|---|---|---|---|
| 4,096 | 8 | 10.0 GB | 4 |
| 8,192 | 32 | 80.0 GB | 4 |
| 32,768 | 32 | 320.0 GB | 8 |
| 32,768 | 128 | 1,280.0 GB | 32 |

**Decide your context length and concurrency before you choose hardware.** They can change the bill
eightfold.

---

## 2. The four ways to split

### Data parallelism (DP) — copy the whole model

```
   requests ---> [ router ]
                  /      \
          GPU 0          GPU 1
        full model     full model      each serves different requests
```

**Splits:** nothing — it replicates. **Communication:** none between GPUs.
**Use when:** the model already fits on one GPU and you just need more throughput.
**Limit:** every GPU needs the whole model in memory.

This is the simplest scaling there is: run N servers, put a load balancer in front.

### Tensor parallelism (TP) — split every layer

```
                 one request
                      |
        +-------------+-------------+
        v                           v
     GPU 0                        GPU 1
  heads 1-16 of every layer   heads 17-32 of every layer
        |                           |
        +----------> all-reduce <---+        after attention AND after the MLP,
                      (every layer)          in EVERY layer
```

**Splits:** each layer's matrices, head-wise. **Communication:** very high — two all-reduces per
layer, so ~160 synchronisations for an 80-layer model, on every forward pass.
**Use when:** the model does not fit on one GPU and your GPUs are in **one machine with NVLink**.
**Limit:** the tensor-parallel size must divide the attention head count. And with grouped-query
attention (often only 8 KV heads), going past TP=8 duplicates the KV cache instead of splitting it.

### Pipeline parallelism (PP) — split the layers

```
   request --> GPU 0 (layers 1-40) --> GPU 1 (layers 41-80) --> output
                        activations passed once per stage boundary
```

**Splits:** the model's layers into sequential stages. **Communication:** low — one handoff per
stage boundary, point-to-point.
**Use when:** you have crossed machine boundaries, where the network is far slower than NVLink.
**Limit:** while GPU 1 works, GPU 0 may be idle — the "bubble". Continuous batching hides much of
this by keeping several requests in flight at different stages.

### Expert parallelism (EP) — split the experts

For Mixture-of-Experts models only: each GPU holds different experts, and tokens are routed to
whichever GPU owns the expert they need. **Communication:** all-to-all, which is demanding — but it
is the only practical way to serve very large MoE models.

### In practice you combine them

```
 Node 1 (8 GPUs, NVLink inside)        Node 2 (8 GPUs, NVLink inside)
 +------------------------------+      +------------------------------+
 |  TP=8: layers 1-40 split     | ---> |  TP=8: layers 41-80 split    |
 +------------------------------+ PP=2 +------------------------------+
        ^                                          |
        |          (replicate the whole thing for more throughput = DP)
```

**The rule of thumb every engine's docs repeat: tensor parallelism *inside* a node, pipeline
parallelism *across* nodes.** TP's constant chatter needs the fast interconnect; PP's occasional
handoffs survive a network.

---

## 3. Why the interconnect decides the design

Very roughly, and this is the whole reason for the rule above:

| Link | Where | Bandwidth, order of magnitude |
|---|---|---|
| **NVLink** | between GPUs in one server | hundreds of GB/s |
| **PCIe** | between GPUs without NVLink | tens of GB/s |
| **Network** (Ethernet / InfiniBand) | between servers | a few to tens of GB/s |

Tensor parallelism synchronises twice per layer. On NVLink that is cheap. Over a network it is a
disaster — the GPUs spend more time waiting than computing. That single fact is why "TP inside the
node, PP across nodes" is the standard answer.

---

## 4. How to decide

```
Does the model fit on ONE GPU, with room for your KV cache?
   YES -> use one GPU. Need more throughput? Add REPLICAS (data parallelism).
   NO  -> Does it fit in ONE NODE (e.g. 8 GPUs)?
            YES -> tensor parallelism, TP = number of GPUs you need
            NO  -> TP = GPUs per node, PP = number of nodes
                   (MoE model? add expert parallelism)
```

Then: **always try to avoid the next step up.** A quantized model
([03-inference/03](../../03-inference/03-quantization/)) or an fp8 KV cache can drop you from four
GPUs to two, which is simpler *and* faster than any clever split.

---

## 5. The commands

**vLLM**
```bash
# 4 GPUs in one machine
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4

# 2 machines with 8 GPUs each
vllm serve <model> --tensor-parallel-size 8 --pipeline-parallel-size 2

# more throughput, model already fits: replicas
vllm serve <model> --data-parallel-size 2
```
For multi-node, vLLM recommends running it on a Ray cluster (`pip install "ray[cgraph]"`), which
lets you launch from a single node; a manual multiprocessing setup is also supported. See
[Parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/).

**SGLang**
```bash
python -m sglang.launch_server --model-path <model> --tp 4          # 4 GPUs, one machine
python -m sglang.launch_server --model-path <model> --dp 2 --tp 2   # 2 replicas x 2 GPUs

# 2 nodes
python -m sglang.launch_server --model-path <model> --tp 16 \
    --nnodes 2 --node-rank 0 --dist-init-addr <head-node>:50000     # and --node-rank 1 on the other
```

---

## 6. One more split: prefill and decode

A newer idea worth knowing. Prefill is compute-bound; decode is memory-bandwidth-bound
([03-inference](../../03-inference/)). Running them on the same GPU means long prompts stall
everyone's token stream.

**Disaggregated serving** puts them on *different* GPUs: prefill machines build the KV cache and
hand it to decode machines. More moving parts (the KV cache must be transferred), but each side can
be tuned and scaled independently. [DistServe](https://arxiv.org/abs/2401.09670) is the paper;
both vLLM and SGLang have implementations.

---

## 7. Traps

| Trap | Why it bites |
|---|---|
| **TP across machines** | The network cannot keep up with two all-reduces per layer. Keep TP inside a node. |
| **TP that doesn't divide the heads** | `--tensor-parallel-size 6` on a 32-head model is rejected. Use 1, 2, 4, 8, ... |
| **TP beyond the KV head count** | With 8 KV heads, TP=16 duplicates the KV cache rather than splitting it — you pay for GPUs and get less benefit than expected. |
| **Expecting linear speedup** | 2 GPUs is not 2× faster. Communication is overhead; TP mainly buys *capacity*, not proportional speed. |
| **Forgetting the KV cache in the plan** | Weights fit in 2 GPUs, then real traffic arrives and memory runs out. Size for weights **plus** cache at your real concurrency. |
| **One giant deployment** | TP=8 for a model that fits in 2 GPUs wastes hardware. Four replicas of TP=2 usually serve more users. |
| **Uneven pipeline stages** | In PP, the slowest stage sets the pace. Split layers evenly. |

---

## 8. Example

```bash
python examples/gpu_planner.py       # no dependencies, instant
```

[`gpu_planner.py`](examples/gpu_planner.py) computes weights and KV cache for several models and
GPUs, suggests a TP/PP plan following the rules above, shows how the answer moves with context
length and concurrency, and lists the valid tensor-parallel sizes for each model. Edit the `MODELS`
and `GPUS` tables at the top for your own hardware.

---

## 9. Exercises

*(All of these are arithmetic — no GPUs required.)*

1. **Plan three deployments.** Use the planner for: (a) 8B, 4K context, 10 users; (b) 70B, 32K
   context, 50 users; (c) 405B, 8K context, 100 users. How many GPUs, and what split?
2. **Halve the bill.** Take deployment (b) and apply 4-bit weights and an fp8 KV cache. Work out the
   new GPU count by hand, then change `weight_bits` and `kv_bits` in the planner to check.
3. **Replicas or a bigger split?** A 13B model fits on one A100. You need 4× the throughput. Compare
   TP=4 on one server against 4 replicas: which serves more users, and why?
4. **Count the chatter.** An 80-layer model at TP=8 does two all-reduces per layer. How many
   synchronisations per generated token? Per 500-token answer? Now explain, in one sentence, why
   this cannot run over Ethernet.
5. **Pick the split.** You have 2 machines × 4 GPUs, NVLink inside each, 25 GbE between them, and a
   model needing 6 GPUs' worth of memory. What TP and PP do you set, and why not TP=6?
6. **Find the bubble.** In a 2-stage pipeline with one request in flight, what fraction of the time
   is each GPU idle? Why does continuous batching mostly fix this?

---

## 10. Projects to build and test

### Beginner — Deployment planner for your own stack
Extend the planner with the GPUs you can actually rent, cloud prices per hour, and a cost-per-
million-tokens estimate.

**How to test it:** check its GPU count against a published deployment guide for one model you can
verify; explain any difference (working space, fragmentation, framework overhead).

### Intermediate — Measure the parallelism tax
On a multi-GPU machine, serve the same model at TP=1, 2 and 4 and load-test each.

**How to test it:** plot throughput and p95 latency against TP size. You should *not* see linear
scaling — quantify the gap and attribute it to communication.

### Advanced — Replicas vs one big split
With 4 GPUs, compare one TP=4 server against 4 single-GPU replicas behind a load balancer, for a
model that fits on one GPU.

**How to test it:** same total hardware, same load test. Report throughput, p50/p95 latency and
failure behaviour when one replica dies. State which you would run in production and why.

---

## 11. Resources

**Start here**
- [Multi-GPU LLM Inference: TP vs PP vs EP](https://www.premai.io/blog/multi-gpu-llm-inference-tp-vs-pp-vs-ep-parallelism-guide-2026/) — Prem AI. A clear, beginner-friendly comparison.
- [Scaling LLM Inference: Data, Pipeline and Tensor Parallelism in vLLM](https://jarvislabs.ai/blog/scaling-llm-inference-dp-pp-tp) — JarvisLabs, with concrete commands.
- [Parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) — vLLM's official guidance and flags.

**Go deeper**
- [SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) — `--tp`, `--dp`, multi-node flags.
- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) — NVIDIA's section on model parallelism.

**Papers**
- [Megatron-LM](https://arxiv.org/abs/1909.08053) — where tensor parallelism for transformers comes from
- [DistServe](https://arxiv.org/abs/2401.09670) — separating prefill and decode across GPUs
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) — continuous batching, which is what hides pipeline bubbles

---

**Previous:** [02 · SGLang](../02-sglang/) · **Track:** [04 · Serving](../) ·
**Next track:** [05 · Evaluation](../../05-evaluation/)
