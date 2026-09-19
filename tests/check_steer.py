"""Plumbing checks: run with `uv run python tests/check_steer.py`."""
import torch

from app.steer import Steerer

s = Steerer()
print(f"device={s.device} scale={s.scale:.2f} emotions={len(s.emotions)}")
msgs = [{"role": "user", "content": "Describe the house you grew up in."}]

def greedy(n=40):
    torch.manual_seed(0)
    return "".join(s.stream(msgs, max_new_tokens=n, temperature=0))

s.vec = None
base = greedy()
s.set_alphas({})
assert s.vec is None
assert greedy() == base, "alpha=0 changed the output"
print("ok  alpha=0 is a no-op")

calls = s.calls
s.set_alphas({"sad": 0.6})
sad = greedy()
assert s.calls > calls, "hook never ran"
assert sad != base, "steering had no effect"
print("ok  hook fires and changes output")

push = s.set_alphas({"joyful": 0.5, "excited": 0.5})
assert abs(push - 1.0) < 1e-6 and abs(s.vec.norm().item() - s.scale) < 1e-3
push = s.set_alphas({n: 1.0 for n in s.emotions})
assert push == 1.5
print("ok  push = sum|alpha|, capped")

# live change: flip steering partway through a single generation
s.set_alphas({})
it = s.stream(msgs, max_new_tokens=60, temperature=0)
first = "".join(next(it) for _ in range(20))
s.set_alphas({"afraid": 0.7})
rest = "".join(it)
assert first == base[: len(first)], "prefix should match unsteered"
print("ok  mid-generation change applies to later tokens only")

print("\n--- baseline ---\n" + base)
print("\n--- sad 0.6 ---\n" + sad)
print("\n--- switched to afraid after 20 tokens ---\n" + first + " ‖ " + rest)
