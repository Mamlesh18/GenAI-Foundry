"""
PagedAttention, explained by building a tiny version of it.

The KV cache (module 01) must live in GPU memory, and nobody knows in advance
how long a response will be. Early servers handled that by reserving one big
CONTIGUOUS chunk per request, sized for the worst case. Most of it sat empty.

vLLM's PagedAttention borrows an idea from operating systems: split memory into
small fixed-size BLOCKS (pages), hand them out only as a sequence grows, and
keep a BLOCK TABLE mapping each sequence's logical positions to physical blocks
that can be anywhere.

This script shows:
  1. what a block table looks like
  2. that attention over scattered blocks gives EXACTLY the same result
  3. how much memory contiguous reservation wastes vs paging
  4. a live simulation: requests arriving and leaving, fragmentation, preemption
  5. prefix sharing: 100 requests with the same system prompt

    pip install torch
    python paged_kv_simulator.py
"""

import math
import random
from collections import deque

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def section(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------------------
# 1 + 2. Block tables, and attention over non-contiguous memory
# ---------------------------------------------------------------------------

BLOCK = 4            # tokens per block (vLLM's default is 16)
N_HEADS, D_HEAD, N_BLOCKS = 2, 8, 12


class PagedKVCache:
    def __init__(self):
        # The physical pool: one big tensor, carved into blocks.
        self.k = torch.zeros(N_BLOCKS, BLOCK, N_HEADS, D_HEAD)
        self.v = torch.zeros(N_BLOCKS, BLOCK, N_HEADS, D_HEAD)
        self.free = list(range(N_BLOCKS))
        random.Random(1).shuffle(self.free)          # blocks come from anywhere
        self.tables, self.lengths = {}, {}

    def append(self, seq, k, v):
        table = self.tables.setdefault(seq, [])
        n = self.lengths.get(seq, 0)
        if n % BLOCK == 0:                           # current block full -> grab a new one
            table.append(self.free.pop())
        block, offset = table[-1], n % BLOCK
        self.k[block, offset], self.v[block, offset] = k, v
        self.lengths[seq] = n + 1

    def read(self, seq):
        """Gather a sequence's K and V through its block table. Real kernels read
        the blocks in place instead of copying; the result is identical."""
        idx = torch.tensor(self.tables[seq])
        n = self.lengths[seq]
        return (self.k[idx].reshape(-1, N_HEADS, D_HEAD)[:n],
                self.v[idx].reshape(-1, N_HEADS, D_HEAD)[:n])


def attend(q, k, v):
    """One query token attending over n cached tokens. q: (H, d), k/v: (n, H, d)."""
    q = q.unsqueeze(1)                               # (H, 1, d)
    k, v = k.transpose(0, 1), v.transpose(0, 1)      # (H, n, d)
    return F.scaled_dot_product_attention(q, k, v).squeeze(1)


def demo_block_table():
    section("1. BLOCK TABLES -- logical positions, physical blocks")
    cache = PagedKVCache()
    reference = {"chat-A": ([], []), "chat-B": ([], [])}

    # Two conversations generating tokens interleaved, as they would on a server.
    order = ["chat-A"] * 11 + ["chat-B"] * 6
    random.Random(2).shuffle(order)
    for seq in order:
        k, v = torch.randn(N_HEADS, D_HEAD), torch.randn(N_HEADS, D_HEAD)
        cache.append(seq, k, v)
        reference[seq][0].append(k)
        reference[seq][1].append(v)

    for seq in ("chat-A", "chat-B"):
        n, table = cache.lengths[seq], cache.tables[seq]
        print(f"  {seq}: {n:>2} tokens -> {len(table)} blocks of {BLOCK}")
        for logical, physical in enumerate(table):
            used = min(BLOCK, n - logical * BLOCK)
            print(f"      logical block {logical} -> physical block {physical:>2}   "
                  f"[{'#' * used}{'-' * (BLOCK - used)}]")
    print(f"  free physical blocks left: {sorted(cache.free)}")
    print(
        "\n  Each sequence looks contiguous to itself (logical blocks 0,1,2...) but its\n"
        "  data is scattered through the pool. Waste is at most one partly-filled block\n"
        "  per sequence ('-' above)."
    )

    section("2. ATTENTION OVER SCATTERED BLOCKS GIVES THE SAME ANSWER")
    for seq in ("chat-A", "chat-B"):
        q = torch.randn(N_HEADS, D_HEAD)
        k_paged, v_paged = cache.read(seq)
        k_ref, v_ref = torch.stack(reference[seq][0]), torch.stack(reference[seq][1])
        out_paged, out_ref = attend(q, k_paged, v_paged), attend(q, k_ref, v_ref)
        diff = (out_paged - out_ref).abs().max().item()
        print(f"  {seq}: max difference paged vs contiguous = {diff:.1e}  "
              f"identical: {torch.allclose(out_paged, out_ref)}")
    print(
        "\n  Attention does not care WHERE keys and values live, only which tokens they\n"
        "  belong to. The block table supplies that. This is why paging costs almost no\n"
        "  accuracy and very little speed."
    )


# ---------------------------------------------------------------------------
# 3. How much memory does reservation waste?
# ---------------------------------------------------------------------------

POOL_TOKENS = 40_000          # KV budget, in tokens, for the whole server
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 1024
PAGE = 16                     # vLLM's default block size


def make_requests(n, seed):
    rng = random.Random(seed)
    return [{"id": i, "prompt": rng.randint(100, 1500), "out": rng.randint(20, 1000)}
            for i in range(n)]


def demo_capacity():
    section(f"3. HOW MANY REQUESTS FIT IN A {POOL_TOKENS:,}-TOKEN KV BUDGET?")
    reqs = make_requests(2000, seed=3)
    print("  Prompts 100-1500 tokens, answers 20-1000 tokens (length unknown in advance).")
    print("  Measured when every admitted request has produced its FULL answer.\n")

    def fill(cost):
        used = reserved = count = 0
        for r in reqs:
            c = cost(r)
            if reserved + c > POOL_TOKENS:
                break
            reserved += c
            used += r["prompt"] + r["out"]
            count += 1
        return count, used, reserved

    strategies = [
        ("reserve max_model_len (4096)", lambda r: MAX_MODEL_LEN),
        ("reserve prompt + max_new (1024)", lambda r: r["prompt"] + MAX_NEW_TOKENS),
        (f"paged, blocks of {PAGE}", lambda r: math.ceil((r["prompt"] + r["out"]) / PAGE) * PAGE),
    ]
    print(f"  {'strategy':<34} {'requests':>9} {'KV really used':>15} {'wasted':>8}")
    print("  " + "-" * 70)
    counts = []
    for name, cost in strategies:
        count, used, reserved = fill(cost)
        counts.append(count)
        print(f"  {name:<34} {count:>9} {used:>15,} {1 - used / reserved:>7.0%}")
    print(
        f"\n  Same memory, {counts[2] / counts[0]:.1f}x more concurrent requests than worst-case reservation.\n"
        "  Contiguous reservation wastes memory on answers that were never that long;\n"
        "  paging wastes only the unfilled tail of each sequence's last block. More\n"
        "  requests in memory at once means bigger batches, which (module 02) means\n"
        "  more throughput from the same GPU."
    )


# ---------------------------------------------------------------------------
# 4. A live simulation
# ---------------------------------------------------------------------------

def simulate_contiguous(reqs, arrival_every):
    """First-fit allocator over one contiguous address space. Each request
    reserves prompt + MAX_NEW_TOKENS up front and frees it when done."""
    free = [(0, POOL_TOKENS)]                         # sorted (start, size) holes

    def alloc(n):
        for i, (start, size) in enumerate(free):
            if size >= n:
                free[i:i + 1] = [] if size == n else [(start + n, size - n)]
                return start
        return None

    def release(start, n):
        free.append((start, n))
        free.sort()
        merged = []
        for s, sz in free:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + sz)
            else:
                merged.append((s, sz))
        free[:] = merged

    pending, waiting, running = deque(reqs), deque(), []
    step = tokens = frag_steps = conc = used_kv = 0
    while pending or waiting or running:
        while pending and pending[0]["id"] * arrival_every <= step:
            waiting.append(pending.popleft())
        while waiting:
            need = waiting[0]["prompt"] + MAX_NEW_TOKENS
            start = alloc(need)
            if start is None:
                if need <= sum(sz for _, sz in free):
                    frag_steps += 1                    # enough memory, just not in one piece
                break
            r = waiting.popleft()
            running.append({"r": r, "start": start, "need": need, "gen": 0})
        step += 1
        conc += len(running)
        tokens += len(running)
        still = []
        for s in running:
            s["gen"] += 1
            used_kv += s["r"]["prompt"] + s["gen"]
            if s["gen"] == s["r"]["out"]:
                release(s["start"], s["need"])
            else:
                still.append(s)
        running = still
    return {"steps": step, "tokens": tokens, "conc": conc / step,
            "util": used_kv / (step * POOL_TOKENS), "frag": frag_steps, "preempt": 0}


def simulate_paged(reqs, arrival_every):
    """Blocks handed out on demand. If the pool runs dry mid-generation, the
    newest request is PREEMPTED: its blocks are freed and it is recomputed later
    (vLLM handles memory pressure the same way: preempt, then recompute later)."""
    total_blocks = POOL_TOKENS // PAGE
    free_blocks = total_blocks
    pending, waiting, running = deque(reqs), deque(), []
    step = tokens = conc = used_kv = preemptions = 0

    blocks_for = lambda n_tokens: math.ceil(n_tokens / PAGE)
    while pending or waiting or running:
        while pending and pending[0]["id"] * arrival_every <= step:
            waiting.append(pending.popleft())
        while waiting:
            need = blocks_for(waiting[0]["prompt"] + 1)
            if need > free_blocks:
                break
            r = waiting.popleft()
            free_blocks -= need
            running.append({"r": r, "blocks": need, "gen": 0})

        # Make sure every running sequence can get the block it needs this step.
        extra = sum(blocks_for(s["r"]["prompt"] + s["gen"] + 1) - s["blocks"] for s in running)
        while extra > free_blocks and running:
            victim = running.pop()                        # newest request
            free_blocks += victim["blocks"]
            waiting.appendleft(victim["r"])
            preemptions += 1
            extra = sum(blocks_for(s["r"]["prompt"] + s["gen"] + 1) - s["blocks"] for s in running)

        step += 1
        conc += len(running)
        tokens += len(running)
        still = []
        for s in running:
            s["gen"] += 1
            want = blocks_for(s["r"]["prompt"] + s["gen"])
            free_blocks -= want - s["blocks"]
            s["blocks"] = want
            used_kv += s["r"]["prompt"] + s["gen"]
            if s["gen"] == s["r"]["out"]:
                free_blocks += s["blocks"]
            else:
                still.append(s)
        running = still
    return {"steps": step, "tokens": tokens, "conc": conc / step,
            "util": used_kv / (step * POOL_TOKENS), "frag": 0, "preempt": preemptions}


def demo_live():
    section("4. LIVE: 600 REQUESTS ARRIVING OVER TIME")
    reqs = make_requests(600, seed=4)
    arrival_every = 4                                 # one new request every 4 steps
    total_out = sum(r["out"] for r in reqs)
    print(f"  One new request every {arrival_every} decode steps. {total_out:,} output tokens to produce.\n")

    rows = [("contiguous first-fit", simulate_contiguous(reqs, arrival_every)),
            ("paged (vLLM-style)", simulate_paged(reqs, arrival_every))]
    print(f"  {'allocator':<22} {'steps to finish':>15} {'avg batch':>10} {'KV memory used':>15}"
          f" {'frag stalls':>12} {'preemptions':>12}")
    print("  " + "-" * 91)
    for name, m in rows:
        print(f"  {name:<22} {m['steps']:>15,} {m['conc']:>10.1f} {m['util']:>14.0%}"
              f" {m['frag']:>12,} {m['preempt']:>12,}")

    c, p = rows[0][1], rows[1][1]
    print(
        f"\n  Paging runs {p['conc'] / c['conc']:.1f}x more requests at once and finishes the same work in\n"
        f"  {p['steps'] / c['steps']:.0%} of the steps.\n\n"
        "  'frag stalls' counts steps where the next request could not start even\n"
        "  though enough memory was free in total -- it was just split into holes too\n"
        "  small to use. That is EXTERNAL fragmentation, and fixed-size blocks make it\n"
        "  impossible. 'preemptions' is the price paging pays instead: it admits\n"
        "  optimistically, and occasionally must evict a request and recompute it."
    )


# ---------------------------------------------------------------------------
# 5. Prefix sharing
# ---------------------------------------------------------------------------

def demo_prefix_sharing():
    section("5. PREFIX SHARING -- 100 chats with the same system prompt")
    n_requests, system_prompt, user_part, answer = 100, 1210, 50, 200

    per_request = system_prompt + user_part + answer
    without = n_requests * math.ceil(per_request / PAGE)

    shared_full = system_prompt // PAGE                # full blocks: stored ONCE
    tail = system_prompt % PAGE                        # partial block: copied per request
    per_request_private = math.ceil((tail + user_part + answer) / PAGE)
    with_sharing = shared_full + n_requests * per_request_private

    print(f"  system prompt {system_prompt} tokens, then {user_part} user tokens and a {answer}-token answer each.\n")
    print(f"  without sharing : {n_requests} x {math.ceil(per_request / PAGE)} blocks"
          f"                  = {without:>6,} blocks")
    print(f"  with sharing    : {shared_full} shared blocks (reference count {n_requests})")
    print(f"                    + {n_requests} x {per_request_private} private blocks"
          f"        = {with_sharing:>6,} blocks")
    print(f"  memory saved    : {1 - with_sharing / without:.0%}")
    print(
        f"\n  The {shared_full} full blocks of the system prompt are identical for every request, so\n"
        f"  one physical copy is shared, with a reference count. The last {tail} prompt tokens\n"
        f"  sit in a PARTLY filled block; the moment a request writes its own token\n"
        "  there, it gets a private copy -- COPY-ON-WRITE, exactly as in an OS.\n\n"
        "  Shared prefixes also skip PREFILL: the cached blocks' K/V are already\n"
        "  computed, so time-to-first-token drops too. vLLM calls this automatic prefix\n"
        "  caching; it matches whole blocks, which is why the tail block is not shared."
    )


def main():
    demo_block_table()
    demo_capacity()
    demo_live()
    demo_prefix_sharing()


if __name__ == "__main__":
    main()
