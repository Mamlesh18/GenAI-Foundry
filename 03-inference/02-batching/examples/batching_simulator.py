"""
Static batching vs continuous batching -- a simulator you can read in one sitting.

A GPU generating text for ONE user is mostly idle: each decode step is limited
by how fast weights can be read from memory, not by arithmetic. So serving
several users in the same step is nearly free -- IF you batch them well.

  STATIC batching      collect a group, run until the LONGEST one finishes,
                       then start the next group. Short requests sit finished
                       in their slot while the GPU pads them.

  CONTINUOUS batching  (a.k.a. iteration-level scheduling, from the Orca paper)
                       after EVERY step, finished requests leave and waiting
                       requests take their slot immediately.

Simplifying assumptions, stated honestly:
  - time is measured in decode steps; a step costs BASE_STEP + PER_SEQ * batch
    (decode is memory-bound, so extra sequences add only a little per step)
  - prefill is ignored, memory is unlimited up to MAX_BATCH slots
  - static batching is treated GENEROUSLY: each response counts as delivered
    the moment its own last token is produced, not when the batch returns

No dependencies.   python batching_simulator.py
"""

import random
import statistics
from collections import deque

MAX_BATCH = 16
N_REQUESTS = 400
ARRIVAL_RATE = 0.10        # requests per unit of time, on average
BASE_STEP = 1.0            # cost of one decode step ...
PER_SEQ = 0.02             # ... plus this much per sequence in the batch
SEED = 0


def step_time(batch_size):
    return BASE_STEP + PER_SEQ * batch_size


# ---------------------------------------------------------------------------
# 1. A picture first: six requests, four slots
# ---------------------------------------------------------------------------

def timeline(policy):
    # (name, arrival step, output tokens)
    requests = [("A", 0, 2), ("B", 0, 8), ("C", 0, 3), ("D", 0, 5), ("E", 1, 2), ("F", 1, 2)]
    slots = [None] * 4                   # each: [name, tokens_left]
    grid = [[] for _ in range(4)]
    pending, waiting, done, step = list(requests), [], {}, 0

    while pending or waiting or any(s is not None for s in slots):
        waiting += [r for r in pending if r[1] <= step]
        pending = [r for r in pending if r[1] > step]

        batch_empty = all(s is None for s in slots)
        if policy == "continuous" or batch_empty:
            for i in range(4):
                if slots[i] is None and waiting:
                    name, _, length = waiting.pop(0)
                    slots[i] = [name, length]

        for i, s in enumerate(slots):
            if s is None:
                grid[i].append(" ")
            elif s[1] == 0:
                grid[i].append(".")      # finished, but still holding the slot
            else:
                grid[i].append(s[0])
                s[1] -= 1
                if s[1] == 0:
                    done[s[0]] = step + 1
                    if policy == "continuous":
                        slots[i] = None

        if policy == "static" and all(s is None or s[1] == 0 for s in slots):
            slots = [None] * 4           # whole batch over -> release every slot
        step += 1

    return grid, done, step


def show_timelines():
    print("=" * 76)
    print("1. SIX REQUESTS, FOUR SLOTS")
    print("=" * 76)
    print("  A,B,C,D arrive at step 0 (lengths 2,8,3,5). E,F arrive at step 1 (lengths 2,2).")
    print("  letter = generating that request   . = slot wasted on a finished request\n")
    results = {}
    for policy in ("static", "continuous"):
        grid, done, steps = timeline(policy)
        results[policy] = done
        wasted = sum(row.count(".") for row in grid)
        print(f"  {policy.upper()}")
        print("    step    " + "".join(str(i % 10) for i in range(steps)))
        for i, row in enumerate(grid):
            print(f"    slot {i}  " + "".join(row))
        print(f"    E done at step {done['E']}, F done at step {done['F']}, "
              f"all done at step {steps}, wasted slot-steps: {wasted}\n")

    s, c = results["static"], results["continuous"]
    print(
        "  Static: E and F wait for B -- the longest request -- before they can even\n"
        "  start. Continuous: they slip into the slots A and C free up, and finish\n"
        f"  {s['E'] - c['E']} and {s['F'] - c['F']} steps sooner, with no wasted slots at all."
    )


# ---------------------------------------------------------------------------
# 2. A realistic workload
# ---------------------------------------------------------------------------

def make_workload(seed=SEED):
    """Chat-like traffic: mostly short answers, some long ones. That length
    variance is exactly what hurts static batching."""
    rng = random.Random(seed)
    t, requests = 0.0, []
    for i in range(N_REQUESTS):
        t += rng.expovariate(ARRIVAL_RATE)
        out_len = rng.randint(10, 60) if rng.random() < 0.8 else rng.randint(200, 500)
        requests.append({"id": i, "arrival": t, "out_len": out_len})
    return requests


def simulate_static(requests):
    pending = deque(requests)
    waiting, finish = [], {}
    now, real_tokens, computed_tokens = 0.0, 0, 0

    while pending or waiting:
        while pending and pending[0]["arrival"] <= now:
            waiting.append(pending.popleft())
        if not waiting:
            now = pending[0]["arrival"]            # idle until the next arrival
            continue

        batch, waiting = waiting[:MAX_BATCH], waiting[MAX_BATCH:]
        longest = max(r["out_len"] for r in batch)
        dt = step_time(len(batch))                 # padded slots still cost compute
        for r in batch:
            finish[r["id"]] = now + r["out_len"] * dt
        real_tokens += sum(r["out_len"] for r in batch)
        computed_tokens += len(batch) * longest
        now += longest * dt

    return finish, real_tokens, computed_tokens, now


def simulate_continuous(requests):
    pending, waiting, running = deque(requests), deque(), []
    finish = {}
    now, tokens = 0.0, 0

    while pending or waiting or running:
        while pending and pending[0]["arrival"] <= now:
            waiting.append(pending.popleft())
        while waiting and len(running) < MAX_BATCH:          # fill every free slot
            r = waiting.popleft()
            running.append([r, r["out_len"]])
        if not running:
            now = pending[0]["arrival"]
            continue

        now += step_time(len(running))
        tokens += len(running)
        still_running = []
        for item in running:
            item[1] -= 1
            if item[1] == 0:
                finish[item[0]["id"]] = now                  # leaves THIS step
            else:
                still_running.append(item)
        running = still_running

    return finish, tokens, tokens, now


def report(name, requests, result):
    finish, real, computed, end = result
    latencies = sorted(finish[r["id"]] - r["arrival"] for r in requests)
    elapsed = end - requests[0]["arrival"]
    return {
        "name": name,
        "throughput": real / elapsed,
        "p50": statistics.median(latencies),
        "p95": latencies[int(0.95 * len(latencies))],
        "wasted": 1 - real / computed,
    }


def show_simulation():
    print("\n" + "=" * 76)
    print(f"2. {N_REQUESTS} REQUESTS, MAX BATCH {MAX_BATCH}")
    print("=" * 76)
    requests = make_workload()
    lengths = [r["out_len"] for r in requests]
    print("  output lengths: 80% short (10-60 tokens), 20% long (200-500 tokens)")
    print(f"  mean {statistics.fmean(lengths):.0f} tokens, "
          f"one arrival every {1 / ARRIVAL_RATE:.0f} time units on average\n")

    rows = [report("static", requests, simulate_static(requests)),
            report("continuous", requests, simulate_continuous(requests))]

    print(f"  {'policy':<12} {'tokens/time':>12} {'p50 latency':>12} {'p95 latency':>12}"
          f" {'compute wasted':>15}")
    print("  " + "-" * 67)
    for r in rows:
        print(f"  {r['name']:<12} {r['throughput']:>12.2f} {r['p50']:>12.0f} {r['p95']:>12.0f}"
              f" {r['wasted']:>14.0%}")

    s, c = rows
    print(
        f"\n  Continuous batching: {c['throughput'] / s['throughput']:.1f}x the throughput, "
        f"{s['p50'] / c['p50']:.0f}x lower median latency,\n"
        f"  and compute wasted on padding falls from {s['wasted']:.0%} to {c['wasted']:.0%}.\n"
    )
    if c["throughput"] > 1.25 * s["throughput"]:
        print(
            "  At this arrival rate static batching cannot keep up. Every batch runs as\n"
            "  long as its longest request, so most slot-time goes to padding and the\n"
            "  queue grows faster than it drains -- that is where the enormous latencies\n"
            "  come from. Continuous batching does the same useful work with the same 16\n"
            "  slots and keeps up, so requests barely wait.\n"
            "\n  Lower ARRIVAL_RATE (exercise 2) until static batching does keep up: the\n"
            "  throughput gap closes, but short requests still wait behind long ones."
        )
    else:
        print(
            "  Both policies keep up with this traffic, so throughput is similar. The\n"
            "  difference is WAITING -- short requests stuck behind long ones -- and the\n"
            "  compute burned on padding. Raise ARRIVAL_RATE and static batching falls\n"
            "  behind on throughput too."
        )
    print(
        "\n  Every modern server uses continuous batching: vLLM, SGLang, TGI, and\n"
        "  TensorRT-LLM (which calls it in-flight batching). In vLLM the knobs are\n"
        "  --max-num-seqs (slots) and --max-num-batched-tokens (tokens per step)."
    )


def show_batch_size_effect():
    print("\n" + "=" * 76)
    print("3. WHY BATCHING IS NEARLY FREE (under this simulator's cost model)")
    print("=" * 76)
    print(f"  {'batch size':>10} {'step time':>10} {'tokens/time':>12} {'time per user token':>20}")
    print("  " + "-" * 56)
    for b in (1, 2, 4, 8, 16, 32, 64):
        dt = step_time(b)
        print(f"  {b:>10} {dt:>10.2f} {b / dt:>12.1f} {dt:>20.2f}")
    print(
        "\n  Throughput climbs almost linearly with batch size while each user's step\n"
        "  time barely grows. That holds on real GPUs until the batch becomes COMPUTE-\n"
        "  bound or runs out of KV-cache memory -- then per-user latency rises. Finding\n"
        "  that knee for your model and hardware is what a load test is for."
    )


def main():
    show_timelines()
    show_simulation()
    show_batch_size_effect()


if __name__ == "__main__":
    main()
