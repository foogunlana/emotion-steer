"""Emotion vectors for any model, from the bundled story corpus.

Same recipe as emotion-concepts/0-extract-emotion-vectors (arXiv:2604.07729 Part 1):
  1. mean-pool every layer's residual stream over each story, skipping the
     first SKIP tokens (the opening is mostly scene-setting)
  2. per-emotion mean over TRAIN stories, minus the mean of those means
  3. project out the top PCs of the neutral dialogues (to VAR_FRAC variance)

Output format matches data/vectors/*.pt:
  V (n_emotions, n_layers + 1, d_model), grand_mean, emotions, layers, meta
"""

import json
from datetime import date
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
VECTOR_DIR = ROOT / "data" / "vectors"
SKIP = 18
VAR_FRAC = 0.5


def vectors_path(model_name: str) -> Path:
    return VECTOR_DIR / (model_name.replace("/", "--").lower() + ".pt")


def cached_models() -> list[str]:
    out = []
    for p in sorted(VECTOR_DIR.glob("*.pt")):
        try:
            out.append(torch.load(p, map_location="cpu")["meta"]["model"])
        except Exception:
            continue
    return out


def read_corpus() -> list[dict]:
    rows = []
    for p in sorted(CORPUS.glob("*.jsonl")):
        rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


@torch.no_grad()
def pooled_acts(backbone, tok, texts: list[str], device: str, batch_size: int,
                progress: Callable[[int, int], None], done_before: int, total: int) -> torch.Tensor:
    """(n_texts, n_layers + 1, d_model) on CPU, float32. Mask-weighted means."""
    prev_side = tok.padding_side
    tok.padding_side = "right"
    out = []
    try:
        for b in range(0, len(texts), batch_size):
            inputs = tok(texts[b:b + batch_size], padding=True, return_tensors="pt").to(device)
            hs = backbone(**inputs, output_hidden_states=True).hidden_states
            mask = inputs.attention_mask.unsqueeze(-1).clone()
            mask[:, :SKIP] = 0
            denom = mask.sum(1).clamp(min=1)
            out.append(torch.stack([((h * mask).sum(1) / denom).float().cpu() for h in hs], dim=1))
            del hs
            progress(done_before + min(b + batch_size, len(texts)), total)
    finally:
        tok.padding_side = prev_side
    return torch.cat(out)


def extract(model, tok, model_name: str, device: str,
            progress: Callable[[int, int], None] = lambda done, total: None,
            batch_size: int = 8) -> dict:
    rows = read_corpus()
    train = [r for r in rows if r["split"] == "train" and r["emotion"] != "neutral"]
    neutral = [r["text"] for r in rows if r["emotion"] == "neutral"]
    emotions = sorted({r["emotion"] for r in train})
    total = len(train) + len(neutral)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    backbone = model.base_model  # skip the unembedding: logits for the whole batch are huge and unused

    X = pooled_acts(backbone, tok, [r["text"] for r in train], device, batch_size, progress, 0, total)
    N = pooled_acts(backbone, tok, neutral, device, batch_size, progress, len(train), total)

    labels = torch.tensor([emotions.index(r["emotion"]) for r in train])
    n_layers = X.shape[1]
    V = torch.empty(len(emotions), n_layers, X.shape[2])
    grand = torch.empty(n_layers, X.shape[2])
    for l in range(n_layers):
        means = torch.stack([X[labels == k, l].mean(0) for k in range(len(emotions))])
        grand[l] = means.mean(0)
        v = means - grand[l]
        Nc = N[:, l] - N[:, l].mean(0)
        _, S, Vh = torch.linalg.svd(Nc, full_matrices=False)
        ratio = S**2 / (S**2).sum()
        k = int((ratio.cumsum(0) < VAR_FRAC).sum()) + 1
        U = Vh[:k]
        V[:, l] = v - (v @ U.T) @ U

    assert not V.isnan().any(), "NaN in emotion vectors"
    blob = {
        "V": V,
        "grand_mean": grand,
        "emotions": emotions,
        "layers": list(range(n_layers)),
        "meta": {"model": model_name, "skip": SKIP, "var_frac": VAR_FRAC,
                 "n_train": len(train), "n_neutral": len(neutral), "created": date.today().isoformat()},
    }
    path = vectors_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    return blob
