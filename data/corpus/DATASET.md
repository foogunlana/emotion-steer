---
license: apache-2.0
language:
  - en
tags:
  - interpretability
  - emotion
  - activations
configs:
  - config_name: stories
    data_files: corpus/*.jsonl
---

# Qwen emotion stories

Corpus for a local replication of Part 1 of [*Emotion Concepts and their Function
in a Large Language Model*](https://arxiv.org/abs/2604.07729).

Stories are written by **Qwen2.5-7B-Instruct**. The model under study is
**Qwen2.5-0.5B-Instruct**, whose activations are extracted over this text. The
paper generates with the same model it studies, but Sonnet 4.5 can write and a
0.5B cannot; two of the three public replications also separate the writer from
the subject.

> **Partial.** 13 of 172 emotions generated (807 stories). The remaining shards are still to run, so
> cross-emotion comparisons and any n-way classification are not
> meaningful yet.

## Configs

| config | rows | source |
|---|---|---|
| `stories` | 807 (595 train / 212 test) | model-generated |

The held-out tests are defined in `config.yaml`, not shipped as separate configs:
6 intensity ladders (36 graded items) and 12 implicit
scenarios.

13 emotions over 100 topics, generated with `Qwen/Qwen2.5-7B-Instruct`.
Split assigned with `test_frac=0.25`, `seed=0`.

```python
from datasets import load_dataset
stories = load_dataset("foogunlana/qwen-emotion-stories", "stories")["train"]
```

## How the corpus was made

Enough to reproduce it. Every script referenced is in `src/scripts/`.

- **Generator** `Qwen/Qwen2.5-7B-Instruct`, bfloat16, one RTX 4090. Not fp16:
  fp16 on Apple MPS spliced junk tokens (`ölüm`, `쌘`) mid-sentence into ~4% of
  stories.
- **Prompts** `src/core/prompts.py`, verbatim from the paper's appendix except
  for a system prompt, a line pinning the language to English, and three lines
  addressing failure modes specific to a small model. Each addition and each
  rejected one is documented with its evidence in that file's docstring.
- **Grid** 12 emotions + `neutral`, × 100 topics from `config.yaml`, × 1 story
  per pair = 1,300 generated. `neutral` is Person/AI dialogue, not narrative —
  it is the baseline the PCA denoising is fitted on, so it is generated under
  identical conditions to the emotion shards.
- **Sampling** `temperature=0.8`, `top_p=0.9`, `max_new_tokens=512`, batch 32.
  Output is trimmed back to the last complete sentence.
- **Seeding** one seed per batch, derived by blake2b from
  `(run_seed, emotion, topic, index)` — keyed on the batch's identity, not its
  position, so adding an emotion or topic does not perturb existing shards.
- **Shards** one `.jsonl` per emotion, written atomically. The shard is the unit
  of resume: a crash costs one emotion, and existing shards are skipped.
- **Filtering** `filter_corpus.py` dropped 493 of 1,300 (37.9%): 489 named an
  emotion, 325 named their own, 4 truncated mid-sentence, 4 held non-Latin
  characters (checks overlap). Nothing was rewritten — a story either survives
  intact or is discarded.
- **Split** by topic, not by row: 25% of topics are held out, the same ones for
  every emotion. A row-wise split leaks, since stories share a premise.

Two commands reproduce it, given a GPU:

```bash
uv run python src/scripts/generate_corpus.py --model Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 --n-per-pair 1 --batch-size 32 \
    --emotions afraid,sad,angry,joyful,disgusted,surprised,calm,desperate,proud,ashamed,lonely,excited
uv run python src/scripts/filter_corpus.py
```

### Known limitations

- Stories are static and interior — a character alone, remembering — where the
  paper's are scenes with events, dialogue and other people acting. Repeated
  prompt changes failed to close this; a few-shot exemplar is the untried lever.
- Filtering costs each emotion a different amount, so shard sizes are unequal
  (`lonely` 41, `neutral` 98). Per-emotion means are therefore computed over
  unequal *n*, and the imbalance tracks how hard each emotion's vocabulary is to
  write around. `filter_corpus.py --balance` trims every shard to the smallest.
- The `emotion-words` filter is stricter than the paper's own standard. Their
  published example names an emotion belonging to another character ("he watched
  the fear flash across her face"); it would not survive this corpus's filter.

## Held-out tests

`intensity` and `implicit` are written by Claude Opus 5 — ideally hand-written,
but the point is that they are not written by the model being tested.

## Fields

`emotion`, `topic`, `index`, `prompt`, `text`, `batch_seed`, `mentions_emotion`, `split`.

A story is identified by `(emotion, topic)`. `text` is the story; `prompt` is the
exact instruction that produced it.

Two fields are vestigial in this release and kept only so the schema matches
what the generator writes:

- `mentions_emotion` is `false` for every row. It flags a story that names its
  own emotion, and every such story was dropped by the filter.
- `index` is always `1`, because this run generated one story per
  (emotion, topic) pair. It exists to identify repeats when `--n-per-pair > 1`.

## Reproducibility

Generation is deterministic given model, dtype, device, torch version, seed,
batch size, and prompt order — batch size included, because one `generate()`
call draws from a single RNG stream, so regrouping prompts redistributes the
draws. `meta.json` records all of it.

`config.yaml` is the full definition: emotions, topics, and both held-out test
sets.
