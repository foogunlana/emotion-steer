"""Re-extract Qwen2.5-0.5B vectors and compare with the ones from emotion-concepts.

Run with: PYTHONPATH=. uv run python tests/check_extract.py
"""
import tempfile
from pathlib import Path

import torch

from app import extract
from app.steer import DEFAULT_MODEL, Steerer

ref = torch.load(extract.vectors_path(DEFAULT_MODEL))
with tempfile.TemporaryDirectory() as tmp:
    extract.VECTOR_DIR = Path(tmp)  # write the fresh copy somewhere disposable
    s = Steerer(DEFAULT_MODEL, status=lambda stage, f: print(f"\r{stage} {f or 0:.0%}  ", end=""))
    print()
    new = torch.load(extract.vectors_path(DEFAULT_MODEL))

assert new["emotions"] == ref["emotions"]
L = s.layer
cos = torch.nn.functional.cosine_similarity(new["V"][:, L], ref["V"][:, L], dim=-1)
print(f"layer {L}: cosine new vs reference per emotion: min {cos.min():.4f}, mean {cos.mean():.4f}")
assert cos.min() > 0.99, "extraction does not reproduce the reference vectors"
print("ok  extraction reproduces the reference vectors")
