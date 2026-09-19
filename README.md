# Mood Dial

A visualisation of emotion concepts, inspired by
[*Emotion Concepts and their Function in a Large Language Model*](https://arxiv.org/abs/2604.07729).
Drag a point around an emotion wheel while a small open-weight model writes, and
watch the story change mood as you go. Each word is tinted with the steering
that was active when it was written.

![Mood Dial: a story steered from calm, to lonely, to excited](docs/screenshot.jpg)

It uses emotion vectors extracted the way the paper describes and adds them to
the model's residual stream during generation. For a replication of some of the
paper's results (extracting and validating the vectors, and steering), see
[foogunlana/emotion-concepts](https://github.com/foogunlana/emotion-concepts).

## Install

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```sh
git clone https://github.com/foogunlana/emotion-steer.git
cd emotion-steer
uv sync
```

`uv sync` creates `.venv` and installs PyTorch, Transformers and FastAPI. The
first run downloads the default model (about 1 GB) into `.cache/`.

## Run

```sh
uv run uvicorn app.server:app --port 8765
```

Then open http://127.0.0.1:8765. It starts on `Qwen/Qwen2.5-0.5B-Instruct`
(set `MODEL=org/name` to start on another). It runs on CUDA, Apple Silicon or CPU.

Other Hugging Face models can be loaded from the Model picker. On first load, their
emotion vectors are extracted from the bundled story corpus and cached in
`data/vectors/`.
