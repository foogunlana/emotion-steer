"""Model, live steering hook, and a token-by-token sampler.

The hook reads `Steerer.vec` on every forward pass, so changing it mid-generation
changes the very next token. Tokens already written keep the steering they were
written under, because their keys/values sit in the KV cache.
"""

from pathlib import Path
from typing import Callable

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from app import extract

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_PUSH = 1.5  # hard cap on total steering, as a fraction of the residual norm
FP32_BELOW = 1e9  # parameters; smaller models run in float32, larger in bfloat16

# Plain text for measuring the typical residual norm, so alpha means
# "fraction of a normal activation" rather than raw units.
NEUTRAL = [
    "The train leaves the station at nine and arrives in the city just before noon.",
    "To reset the router, hold the button on the back for ten seconds until the light blinks.",
    "The library is open from Monday to Saturday and closed on public holidays.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The report covers sales figures for the second quarter across all three regions.",
    "Add the flour and sugar to the bowl, then stir in the milk a little at a time.",
    "The meeting has been moved to the small conference room on the fourth floor.",
    "Most maps place north at the top and east to the right.",
]
# end-of-turn markers across chat templates; whichever exist in the vocab stop generation
END_TOKENS = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>", "<|return|>", "</s>"]
LAYER_PATHS = ["model.layers", "model.language_model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers"]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decoder_layers(model) -> torch.nn.ModuleList:
    for path in LAYER_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"can't find the decoder layers of {type(model).__name__}")


class Steerer:
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 status: Callable[[str, float | None], None] = lambda stage, frac: None):
        self.name = model_name
        self.device = pick_device()

        # already downloaded: skip the Hub round-trips, which cost more than reading the weights
        cached = (ROOT / ".cache" / ("models--" + model_name.replace("/", "--")) / "snapshots").exists()
        try:
            self._load(model_name, cached, status)
        except OSError:
            if not cached:
                raise
            self._load(model_name, False, status)  # an interrupted download: fetch the rest
        self.layers = decoder_layers(self.model)

        path = extract.vectors_path(model_name)
        if path.exists():
            d = torch.load(path, map_location="cpu")
        else:
            status("extracting emotion vectors", 0.0)
            d = extract.extract(self.model, self.tok, model_name, self.device,
                                progress=lambda done, total: status("extracting emotion vectors", done / total))

        # two-thirds deep, as in the paper; hidden_states[L] is the output of layers[L - 1]
        self.layer = round(len(self.layers) * 2 / 3)
        self.emotions: list[str] = d["emotions"]
        V = d["V"][:, self.layer].float()
        self.units = V / V.norm(dim=-1, keepdim=True)  # (n_emotions, d_model), on CPU

        status("measuring activation scale", None)
        self.scale = self.resid_scale(NEUTRAL)

        vocab = self.tok.get_vocab()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.stop_ids = {i for i in [self.tok.eos_token_id, *eos, *(vocab.get(t) for t in END_TOKENS)] if i is not None}

        # vec is written from the server's event loop and read in the generation
        # thread. It stays on CPU; the hook moves it to the device lazily.
        self.vec: torch.Tensor | None = None
        self._dev_vec: torch.Tensor | None = None
        self._dev_src: torch.Tensor | None = None
        self.calls = 0
        self._handle = self.layers[self.layer - 1].register_forward_hook(self._hook)

    def _load(self, model_name: str, local: bool, status):
        hub = dict(cache_dir=ROOT / ".cache", local_files_only=local)
        status("loading weights" if local else "downloading weights", None)
        self.tok = AutoTokenizer.from_pretrained(model_name, **hub)
        config = AutoConfig.from_pretrained(model_name, **hub)
        with torch.device("meta"):  # count parameters without allocating them
            self.n_params = sum(p.numel() for p in AutoModelForCausalLM.from_config(config).parameters())
        dtype = torch.float32 if self.device == "cpu" or self.n_params < FP32_BELOW else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **hub).to(self.device).eval()

    def close(self):
        self._handle.remove()
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    @torch.no_grad()
    def resid_scale(self, texts: list[str]) -> float:
        total, count = 0.0, 0
        for t in texts:
            ids = self.tok(t, return_tensors="pt").input_ids.to(self.device)
            h = self.model.base_model(ids, output_hidden_states=True).hidden_states[self.layer][0]
            norms = h.float().norm(dim=-1)[1:]  # position 0 is an attention sink with a huge norm
            total += norms.sum().item()
            count += norms.numel()
        return total / count

    def set_alphas(self, alphas: dict[str, float]) -> float:
        """Mix emotions into one direction; its size is the sum of |alpha|.

        Returns the push actually applied (after the MAX_PUSH cap).
        """
        a = torch.tensor([float(alphas.get(e, 0.0)) for e in self.emotions])
        push = min(a.abs().sum().item(), MAX_PUSH)
        mix = a @ self.units
        if push < 1e-4 or mix.norm() < 1e-6:
            self.vec = None
            return 0.0
        self.vec = mix / mix.norm() * push * self.scale
        return push

    def _hook(self, module, inputs, output):
        self.calls += 1
        vec = self.vec
        if vec is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if self._dev_src is not vec:
            self._dev_vec, self._dev_src = vec.to(self.device, h.dtype), vec
        if h.shape[1] > 1:
            h[:, 1:] += self._dev_vec  # prefill: leave the attention sink alone
        else:
            h += self._dev_vec
        return output

    def prompt_ids(self, messages: list[dict]) -> torch.Tensor:
        kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        try:
            text = self.tok.apply_chat_template(messages, **kw)
        except Exception:
            # some templates (Gemma) reject a system turn: fold it into the first user turn
            if messages and messages[0]["role"] == "system":
                sys, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
                rest[0]["content"] = f"{sys}\n\n{rest[0]['content']}"
                text = self.tok.apply_chat_template(rest, **kw)
            else:
                raise
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    @torch.no_grad()
    def stream(self, messages: list[dict], max_new_tokens=400, temperature=0.8, top_p=0.9,
               rep_penalty=1.1, rep_window=256, overrun=120):
        """Yield decoded text pieces one token at a time.

        Past max_new_tokens it keeps going to the end of the sentence, up to
        `overrun` more tokens, so chapters don't stop mid-word.
        """
        out = self.model(input_ids=self.prompt_ids(messages), use_cache=True)
        generated: list[int] = []
        emitted = ""
        for step in range(max_new_tokens + overrun):
            logits = out.logits[0, -1].float()
            if generated and rep_penalty != 1.0:
                recent = torch.tensor(sorted(set(generated[-rep_window:])), device=logits.device)
                picked = logits[recent]
                logits[recent] = torch.where(picked > 0, picked / rep_penalty, picked * rep_penalty)
            nxt = sample(logits, temperature, top_p)
            if nxt in self.stop_ids:
                break
            generated.append(nxt)
            # decode the whole reply and emit only the new suffix, so multi-token
            # characters come out whole
            full = self.tok.decode(generated, skip_special_tokens=True)
            if not full.endswith("�"):
                yield full[len(emitted):]
                emitted = full
                if step >= max_new_tokens and full.rstrip().endswith((".", "!", "?", '"', "”")):
                    break
            out = self.model(
                input_ids=torch.tensor([[nxt]], device=self.device),
                past_key_values=out.past_key_values,
                use_cache=True,
            )


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_p, idx = probs.sort(descending=True)
    keep = sorted_p.cumsum(0) - sorted_p < top_p
    sorted_p = sorted_p * keep
    choice = torch.multinomial(sorted_p / sorted_p.sum(), 1)
    return int(idx[choice])
