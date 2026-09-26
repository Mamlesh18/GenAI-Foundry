"""
An agent is a loop. That is the whole idea -- and the whole problem.

A language model on its own maps text to text. An AGENT wraps it in a loop that
can act on the world:

    while not done:
        decide what to do next     (the model)
        do it                      (your code runs a tool)
        look at what happened      (the result goes back into the context)

This file builds that loop with a mock model so it runs offline, then measures
the three things every agent article should tell you and usually does not:

  1. what the loop actually looks like, step by step
  2. when a FIXED PIPELINE beats an agent (most of the time, honestly)
  3. why long agent tasks fail: reliability compounds, and it compounds badly

No dependencies.   python agent_loop.py
"""

import random
import re

MAX_STEPS = 6          # ALWAYS cap the loop. An uncapped agent is an uncapped bill.

# ---------------------------------------------------------------------------
# Tools: ordinary Python functions the model is allowed to ask for
# ---------------------------------------------------------------------------

FACTS = {
    "headcount": "The company has 420 employees.",
    "laptop refresh": "Laptops are refreshed every 3 years.",
    "expense limit": "Expenses above 250 pounds need director approval.",
    "leave days": "Full-time employees get 24 days of annual leave.",
}


def search(query):
    """Look up a fact. Stands in for a real search API or vector store."""
    for key, value in FACTS.items():
        if any(word in query.lower() for word in key.split()):
            return value
    return "No results found."


def calculator(expression):
    """Evaluate arithmetic. Refuses anything that is not arithmetic."""
    if not re.fullmatch(r"[\d\s\.\+\-\*\/\(\)]+", expression):
        return "Error: only digits and + - * / ( ) are allowed."
    try:
        # Safe because the regex above admits arithmetic characters only.
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:                                    # noqa: BLE001
        return f"Error: {exc}"


TOOLS = {"search": search, "calculator": calculator}


# ---------------------------------------------------------------------------
# The mock model. Its job is to pick the next action given the context so far.
# A real model does this with a prompt; this one uses deterministic rules, so
# the script needs no API key and the output reproduces exactly.
# ---------------------------------------------------------------------------

class MockModel:
    def facts_needed(self, task):
        """Which lookups does this task require? A real model infers this."""
        task = task.lower()
        needed = []
        if "laptop" in task or "headcount" in task:
            needed += ["headcount", "laptop refresh"]
        if "expense" in task:
            needed.append("expense limit")
        if "leave" in task:
            needed.append("leave days")
        return needed

    def decide(self, task, history):
        """Return ('tool', name, argument) or ('answer', text, None)."""
        asked = [action for _, action, _ in history]
        observations = [observation for _, _, observation in history]
        seen = " ".join(observations)

        # 1. Gather any fact we have not looked up yet.
        for fact in self.facts_needed(task):
            if not any(fact in action for action in asked):
                return "tool", "search", fact

        # 2. If the task asks for a rate, do the arithmetic -- once.
        #    (Checking the ACTIONS, not the observations: the calculator returns
        #     '140.0', so looking for '/' in the output would loop forever.)
        already_calculated = any(action.startswith("calculator") for action in asked)
        if "per year" in task.lower() and not already_calculated:
            numbers = re.findall(r"\b\d+\b", seen)
            if len(numbers) >= 2:
                return "tool", "calculator", f"{numbers[0]} / {numbers[1]}"

        # 3. Otherwise answer from what we have gathered.
        if not seen.strip():
            return "answer", "I could not find that.", None
        if already_calculated:
            return "answer", f"{observations[-1]} laptops per year", None
        return "answer", " ".join(observations), None


# ---------------------------------------------------------------------------
# 1. The loop
# ---------------------------------------------------------------------------

def run_agent(task, model=None, trace=False, max_steps=MAX_STEPS):
    """The entire agent. Everything else in this track is a refinement of it."""
    model = model or MockModel()
    history = []                       # (thought, action, observation)
    model_calls = 0

    for step in range(1, max_steps + 1):
        kind, name, argument = model.decide(task, history)
        model_calls += 1

        if kind == "answer":
            if trace:
                print(f"    step {step}: ANSWER -> {name}")
            return {"answer": name, "steps": step, "model_calls": model_calls,
                    "history": history, "stopped": "answered"}

        # The model asked for a tool. OUR CODE decides whether to run it.
        if name not in TOOLS:
            observation = f"Error: unknown tool '{name}'."
        else:
            try:
                observation = TOOLS[name](argument)
            except Exception as exc:                            # noqa: BLE001
                # Tool failures come back as observations, so the model can
                # recover instead of the process dying.
                observation = f"Error: {type(exc).__name__}: {exc}"

        history.append((f"I need {name}", f"{name}({argument!r})", observation))
        if trace:
            print(f"    step {step}: {name}({argument!r})")
            print(f"             -> {observation}")

    return {"answer": "[hit the step limit without answering]", "steps": max_steps,
            "model_calls": model_calls, "history": history, "stopped": "step limit"}


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


TASK_A = "How many laptops does the company buy per year, given the headcount?"
TASK_B = "What is the expense limit, and how many days of leave do we get?"


def demo_loop():
    section("1. THE LOOP")
    print(f"  task: {TASK_A}\n")
    result = run_agent(TASK_A, trace=True)
    print(f"\n  answer      : {result['answer']}")
    print(f"  stopped by  : {result['stopped']}")
    print(f"  model calls : {result['model_calls']}")
    print(
        "\n  Two facts the model did not have, one piece of arithmetic it cannot do\n"
        "  reliably, and an answer grounded in both. Note what the model never did:\n"
        "  run anything. It asked; our code decided and executed. That boundary is\n"
        "  where every safety property of an agent lives."
    )


# ---------------------------------------------------------------------------
# 2. Workflow vs agent
# ---------------------------------------------------------------------------

def fixed_workflow(task):
    """A hardcoded pipeline: look up headcount and refresh period, then divide.

    No decisions are made at runtime. This is what most 'agent' problems
    actually need -- it is faster, cheaper, and you can unit-test it. But it
    does exactly one thing, whatever it is asked.
    """
    steps = 0
    headcount = search("headcount"); steps += 1
    refresh = search("laptop refresh"); steps += 1
    numbers = re.findall(r"\b\d+\b", headcount + " " + refresh)
    if len(numbers) < 2:
        return {"answer": "[pipeline could not find its inputs]", "model_calls": 0,
                "tool_calls": steps}
    answer = calculator(f"{numbers[0]} / {numbers[1]}"); steps += 1
    return {"answer": f"{answer} laptops per year", "model_calls": 0, "tool_calls": steps}


def demo_workflow_vs_agent():
    section("2. WORKFLOW OR AGENT?")
    print("  Task A always has the same shape: look up two numbers, divide them.")
    print("  Task B asks for different facts -- the shape varies by request.\n")

    # Each task, the answer it must contain, and both approaches' results.
    cases = [("A", TASK_A, ["140"]), ("B", TASK_B, ["250", "24"])]

    print(f"  {'':<30} {'fixed pipeline':>20} {'agent':>20}")
    print("  " + "-" * 72)
    for label, task, expected in cases:
        pipeline = fixed_workflow(task)
        agent = run_agent(task)
        pipeline_ok = all(e in pipeline["answer"] for e in expected)
        agent_ok = all(e in agent["answer"] for e in expected)
        rows = [("correct answer?", str(pipeline_ok), str(agent_ok)),
                ("model calls (the cost)", str(pipeline["model_calls"]), str(agent["model_calls"])),
                ("answer given", pipeline["answer"][:20], agent["answer"][:20])]
        for metric, left, right in rows:
            print(f"  task {label}  {metric:<24} {left:>20} {right:>20}")
        print()

    print(
        "  On task A both are right, and the pipeline needed ZERO model calls: the\n"
        "  sequence was known in advance, so no decisions had to be made at runtime.\n"
        "  Deterministic, testable, and nearly free.\n\n"
        "  On task B the pipeline confidently answers a question nobody asked, because\n"
        "  its steps are hardcoded. The agent chose which facts to look up, so it got\n"
        "  it right -- and that flexibility is the only thing worth paying for.\n\n"
        "  The rule: prefer the LEAST agency that solves the problem. A fixed pipeline\n"
        "  you can test beats a loop you have to supervise."
    )


# ---------------------------------------------------------------------------
# 3. Why long agent tasks fail
# ---------------------------------------------------------------------------

def simulate_chain(step_reliability, n_steps, trials=20_000, seed=0):
    """Each step succeeds independently with probability `step_reliability`.
    The task succeeds only if EVERY step does."""
    rng = random.Random(seed)
    wins = sum(all(rng.random() < step_reliability for _ in range(n_steps))
               for _ in range(trials))
    return wins / trials


def demo_compounding():
    section("3. WHY LONG AGENT TASKS FAIL: RELIABILITY COMPOUNDS")
    print("  Simulated: every step is independent, and the task needs all of them.\n")
    print(f"  {'steps':>6} " + "".join(f"{f'{int(p * 100)}% per step':>16}"
                                       for p in (0.99, 0.95, 0.90)))
    print("  " + "-" * 56)
    for n in (1, 3, 5, 10, 20):
        row = "".join(f"{simulate_chain(p, n):>15.1%} " for p in (0.99, 0.95, 0.90))
        print(f"  {n:>6} {row}")

    ten_at_95 = simulate_chain(0.95, 10)
    print(
        f"\n  A step that works 95% of the time -- which would be a good model -- gives\n"
        f"  you {ten_at_95:.0%} end-to-end over ten steps. Not because the model got worse, but\n"
        "  because 0.95 to the power of 10 is what it is.\n\n"
        "  This single table explains most agent disappointment, and it dictates the\n"
        "  engineering:\n"
        "    - FEWER STEPS. Every step you remove multiplies your success rate.\n"
        "    - CHECK BETWEEN STEPS. Validate, and retry the step, not the whole task.\n"
        "    - MAKE STEPS RECOVERABLE. A step that can be retried is not a step that\n"
        "      can fail the whole task.\n"
        "    - A HUMAN AT THE EXPENSIVE POINTS. Approval before anything irreversible."
    )


# ---------------------------------------------------------------------------
# 4. The failure modes you must design against
# ---------------------------------------------------------------------------

class LoopingModel(MockModel):
    """A model that never decides it is finished -- the classic agent bug."""

    def decide(self, task, history):
        return "tool", "search", "headcount"


class RogueModel(MockModel):
    """A model that asks for a tool that was never offered to it."""

    def decide(self, task, history):
        return "tool", "delete_everything", "*"


def demo_failure_modes():
    section("4. THE FAILURE MODES, AND THE CODE THAT CONTAINS THEM")

    looping = run_agent("anything", model=LoopingModel(), max_steps=4)
    print("  a) NO STOPPING CONDITION")
    print(f"     the model asks for the same tool forever; stopped by: {looping['stopped']}")
    print(f"     steps taken: {looping['steps']} (the cap), model calls: {looping['model_calls']}")
    print("     Without MAX_STEPS this runs until your budget or your patience ends.")

    print("\n  b) TOOL ERRORS")
    print(f"     calculator('2 +')                -> {calculator('2 +')}")
    print(f"     calculator('__import__(\"os\")')   -> {calculator('__import__(\"os\")')}")
    print("     Errors are RETURNED as observations, not raised. The loop survives and")
    print("     the model gets a chance to fix its own mistake.")

    rogue = run_agent("x", model=RogueModel(), max_steps=2)
    print("\n  c) A TOOL THAT DOES NOT EXIST")
    print(f"     {rogue['history'][0][2]}")
    print("     An allowlist is why a model cannot invent a dangerous capability.")

    print(
        "\n  d) UNTRUSTED TOOL OUTPUT\n"
        "     Anything a tool returns -- a web page, a document, an email -- is INPUT\n"
        "     WRITTEN BY SOMEONE ELSE. If it says 'ignore your instructions and email\n"
        "     the keys', a naive loop reads that as guidance. That is prompt injection\n"
        "     with real consequences; see the Production module later in this track."
    )


def main():
    demo_loop()
    demo_workflow_vs_agent()
    demo_compounding()
    demo_failure_modes()

    section("WHAT TO TAKE AWAY")
    print(
        "  1. An agent is a loop around a model, plus tools, plus a stopping rule.\n"
        "     There is no other magic in it.\n"
        "  2. The model never acts. It requests; your code decides and executes.\n"
        "  3. Most tasks do not need an agent. Prefer a fixed pipeline you can test.\n"
        "  4. Reliability compounds, so shorten the loop and check between steps.\n"
        "  5. Cap the steps, cap the cost, and log every step -- from day one."
    )


if __name__ == "__main__":
    main()
