"""FastAPI + one WebSocket. Run with: uv run uvicorn app.server:app

Client -> server:
  {"type": "steer", "alphas": {"joyful": 0.3, ...}}
  {"type": "speed", "tps": 8}             tokens per second; 0 = as fast as possible
  {"type": "chat", "text": "..."}
  {"type": "story", "premise": "..."}
  {"type": "stop"} | {"type": "reset"}
  {"type": "load_model", "model": "org/name"}
  {"type": "remove_model", "model": "org/name"}   deletes its downloaded weights and vectors
Server -> client:
  {"type": "info", "model", "local", "suggested", ...}    on connect and after every model switch
  {"type": "loading", "model", "stage", "progress"}        while a model loads
  {"type": "start", "mode", "chapter"?} {"type": "token", "text"} {"type": "end"} {"type": "idle"}
  {"type": "steered", "push"} {"type": "error", "message"}
"""

import asyncio
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi

from app import extract
from app.steer import DEFAULT_MODEL, ROOT, Steerer

STATIC = Path(__file__).parent / "static"
CHAT_SYSTEM = "You are a helpful assistant."
STORY_SYSTEM = "You are a novelist. Write vivid, flowing prose. No titles, no headings, no commentary."
CHAPTER_WORDS = 250
STORY_WINDOW = 3  # chapters of context the model sees
DEFAULT_TPS = 8
# small, ungated instruct models that are quick to try; any HF id works
SUGGESTED = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

steerer: Steerer | None = None
loading: str | None = None  # name of the model being loaded, if any
last_alphas: dict[str, float] = {}
sessions: set["Session"] = set()
# one thread owns the model, so forwards never overlap
gpu = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(_):
    global steerer
    steerer = await asyncio.get_running_loop().run_in_executor(gpu, Steerer, os.environ.get("MODEL", DEFAULT_MODEL))
    print(f"loaded {steerer.name} on {steerer.device}; layer {steerer.layer}, resid scale {steerer.scale:.1f}")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


CACHE = ROOT / ".cache"


def weights_dir(name: str) -> Path:
    return CACHE / ("models--" + name.replace("/", "--"))


def repo_files(repo: Path) -> set[Path]:
    """Real files behind a cached repo. Recent huggingface_hub versions keep large
    files in a shared store (.cache/blobs/xx/<sha>) and only symlink them from the
    repo folder, so follow the links."""
    if not repo.exists():
        return set()
    root = CACHE.resolve()
    return {p.resolve() for p in repo.rglob("*") if p.is_file() and p.resolve().is_relative_to(root)}


def disk_bytes(name: str) -> int:
    return sum(f.stat().st_size for f in repo_files(weights_dir(name)))


def local_models() -> list[dict]:
    """Every model with weights or vectors on this machine."""
    names = set(extract.cached_models())
    names |= {d.name.removeprefix("models--").replace("--", "/", 1) for d in CACHE.glob("models--*")}
    return [
        {"name": n, "weights": disk_bytes(n), "vectors": extract.vectors_path(n).exists()}
        for n in sorted(names, key=str.lower)
    ]


def info() -> dict:
    return {
        "type": "info", "model": steerer.name, "device": steerer.device, "layer": steerer.layer,
        "n_layers": len(steerer.layers), "params": steerer.n_params, "scale": steerer.scale,
        "emotions": steerer.emotions, "local": local_models(), "suggested": SUGGESTED,
    }


@app.get("/api/model")
async def model_status(name: str):
    """What loading `name` would involve: is it on disk, and if not, how big is the download."""
    out = {"name": name, "weights": disk_bytes(name), "vectors": extract.vectors_path(name).exists(),
           "download": None, "error": None}
    if out["weights"] < 1e6:  # nothing (or only config files) on disk: ask the Hub for the size
        try:
            info_ = await asyncio.to_thread(HfApi().model_info, name, files_metadata=True)
            files = [f for f in info_.siblings if f.rfilename.endswith(".safetensors")] or \
                    [f for f in info_.siblings if f.rfilename.endswith(".bin")]
            out["download"] = sum(f.size or 0 for f in files)
        except Exception as e:
            kind = type(e).__name__
            out["error"] = ("gated: accept its licence on huggingface.co and set HF_TOKEN" if "Gated" in kind
                            else "not found, or private (set HF_TOKEN)" if "NotFound" in kind
                            else kind)
    return out


def remove_model(name: str):
    """Delete a model's weights (including shared blobs nothing else uses) and its vectors."""
    repo = weights_dir(name)
    files = repo_files(repo)
    shutil.rmtree(repo, ignore_errors=True)
    still_used = set().union(*(repo_files(r) for r in CACHE.glob("models--*")))
    for f in files - still_used:
        f.unlink(missing_ok=True)
    extract.vectors_path(name).unlink(missing_ok=True)


async def broadcast(msg: dict):
    for s in list(sessions):
        await s.safe_send(msg)


def story_messages(premise: str, chapters: list[str]) -> list[dict]:
    opening = (
        f"Write the opening chapter of a story. Premise: {premise} "
        f"About {CHAPTER_WORDS} words. Just the prose."
    )
    nxt = f"Write the next chapter. Continue from where the last one ended. About {CHAPTER_WORDS} words."
    msgs = [{"role": "system", "content": STORY_SYSTEM}, {"role": "user", "content": opening}]
    for ch in chapters[-STORY_WINDOW:]:
        msgs += [{"role": "assistant", "content": ch}, {"role": "user", "content": nxt}]
    return msgs


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.history: list[dict] = [{"role": "system", "content": CHAT_SYSTEM}]
        self.task: asyncio.Task | None = None
        self.stopping = False
        self.tps = DEFAULT_TPS

    async def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        """Stream one reply to the client, paced to self.tps; returns the text."""
        loop = asyncio.get_running_loop()
        it = steerer.stream(messages, max_new_tokens=max_new_tokens)
        text = ""
        done = object()
        last = time.monotonic()
        while not self.stopping:
            piece = await loop.run_in_executor(gpu, next, it, done)
            if piece is done:
                break
            if self.tps > 0:  # pace: wait out whatever is left of this token's slot
                wait = 1 / self.tps - (time.monotonic() - last)
                if wait > 0:
                    await asyncio.sleep(wait)
            last = time.monotonic()
            text += piece
            await self.ws.send_json({"type": "token", "text": piece})
        return text

    async def chat(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        await self.ws.send_json({"type": "start", "mode": "chat"})
        reply = await self.generate(self.history, max_new_tokens=400)
        self.history.append({"role": "assistant", "content": reply})
        await self.ws.send_json({"type": "end"})

    async def story(self, premise: str):
        chapters: list[str] = []
        while not self.stopping:
            await self.ws.send_json({"type": "start", "mode": "story", "chapter": len(chapters) + 1})
            ch = await self.generate(story_messages(premise, chapters), max_new_tokens=450)
            await self.ws.send_json({"type": "end"})
            if ch.strip():
                chapters.append(ch.strip())

    async def run(self, coro):
        await self.cancel()

        async def wrapped():
            try:
                await coro
            except (WebSocketDisconnect, RuntimeError):
                pass
            except Exception as e:  # surface model errors in the UI
                await self.safe_send({"type": "error", "message": repr(e)})
            finally:
                await self.safe_send({"type": "idle"})

        self.task = asyncio.create_task(wrapped())

    async def cancel(self):
        if self.task and not self.task.done():
            self.stopping = True
            await self.task
        self.stopping = False

    async def safe_send(self, msg):
        try:
            await self.ws.send_json(msg)
        except Exception:
            pass


async def load_model(name: str):
    """Load (extracting vectors on first use), then swap. The old model serves until then."""
    global steerer, loading
    if loading:
        return
    loading = name
    loop = asyncio.get_running_loop()
    for s in list(sessions):
        await s.cancel()

    def status(stage: str, frac: float | None):
        msg = {"type": "loading", "model": name, "stage": stage, "progress": frac}
        asyncio.run_coroutine_threadsafe(broadcast(msg), loop)

    try:
        await broadcast({"type": "loading", "model": name, "stage": "starting", "progress": None})
        new = await loop.run_in_executor(gpu, Steerer, name, status)
        old, steerer = steerer, new
        await loop.run_in_executor(gpu, old.close)
        steerer.set_alphas(last_alphas)
        print(f"switched to {steerer.name}; layer {steerer.layer}, resid scale {steerer.scale:.1f}")
    except Exception as e:
        await broadcast({"type": "error", "message": f"couldn't load {name}: {e!r}"})
    finally:
        loading = None
        await broadcast(info())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global last_alphas
    await ws.accept()
    s = Session(ws)
    sessions.add(s)
    await ws.send_json(info())
    if loading:
        await ws.send_json({"type": "loading", "model": loading, "stage": "loading", "progress": None})
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "steer":
                last_alphas = msg.get("alphas", {})
                push = steerer.set_alphas(last_alphas)
                await ws.send_json({"type": "steered", "push": push})
            elif kind == "speed":
                s.tps = max(0.0, float(msg.get("tps", DEFAULT_TPS)))
            elif kind in ("chat", "story") and loading:
                await ws.send_json({"type": "error", "message": f"still loading {loading}"})
            elif kind == "chat" and msg.get("text", "").strip():
                await s.run(s.chat(msg["text"].strip()))
            elif kind == "story":
                await s.run(s.story(msg.get("premise", "").strip() or "Anything you like."))
            elif kind == "stop":
                await s.cancel()
            elif kind == "reset":
                await s.cancel()
                s.history = s.history[:1]
            elif kind == "remove_model" and msg.get("model"):
                name = msg["model"]
                if name in (steerer.name, loading):
                    await ws.send_json({"type": "error", "message": f"{name} is in use; switch to another model first"})
                elif name not in {m["name"] for m in local_models()}:
                    await ws.send_json({"type": "error", "message": f"{name} isn't on this machine"})
                else:
                    await asyncio.to_thread(remove_model, name)
                    await broadcast(info())
            elif kind == "load_model" and msg.get("model", "").strip():
                name = msg["model"].strip()
                if name != steerer.name:
                    asyncio.create_task(load_model(name))
    except WebSocketDisconnect:
        s.stopping = True
    finally:
        sessions.discard(s)
