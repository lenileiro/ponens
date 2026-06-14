# Ponens Package Reference

The supported `thinking` package is manifest-driven. It no longer ships the legacy synthetic
English bank, kinship QA worlds, proof-checking CLI, rule-induction harness, or verbalizer. Keep
new work on the current entry points:

- `thinking.text`: raw reading and semantic text learning with latent concept discovery,
  replay, self-study, reanalysis, and memory-gap training.
- `thinking.multimodal`: generic named-feature/text prefix bridge into a shared decoder, with
  latent slots and concept memory.
- `thinking.vision_understanding`: image concept memory from manifests.
- `thinking.image_*`: image fetch, caption, embed, score, curate, latent-flow train/eval, and
  preference artifacts.
- `thinking.audio`, `thinking.listen`, `thinking.speak`, `thinking.realspeech`, and related audio
  modules for the audio experiments.

## Text

`thinking.text` accepts raw reading corpora through `--reading-data`. This is the language-mastery
path: text chunks are read directly, latent slots are trained with
sequence, factorization, association, graph, neighborhood, transition, cluster, discovery,
reanalysis, memory-gap, and graph-closure insight objectives. No English rules or prompt
templates are embedded in the module, and the old structured fact/QA/import command-line surfaces
are no longer supported.

```bash
python -m thinking.text --selftest

python -m thinking.text --reading-data data/reading.jsonl \
    --steps 40000 --batch 16 --d 256 --layers 4 --heads 8 \
    --latent-concept-slots 4 --reading-memory-size 256 \
    --reading-study-strategy gap --reading-study-probe-n 256 \
    --reading-discovery-w 0.05 --reading-reanalysis-w 0.05 \
    --reading-gap-w 0.05 \
    --out runs/text_raw_reading.json \
    --checkpoint runs/text_raw_reading.pt

python -m thinking.text --reading-data data/new_reading.jsonl \
    --reading-checkpoint runs/text_raw_reading.pt \
    --reading-out-checkpoint runs/text_raw_reading_studied.pt \
    --steps 4000 --batch 16 \
    --reading-study-strategy gap --reading-study-probe-n 256 \
    --reading-discovery-w 0.05 --reading-reanalysis-w 0.05 \
    --reading-gap-w 0.05 \
    --out runs/text_raw_reading_study.json
```

## Multimodal

`thinking.multimodal` receives JSONL rows with optional text, named feature views, and optional
target tokens. The model learns to fuse views into continuous prefixes for one decoder. If a
dataset wants captions, extraction facts, actions, or no decoder target at all, that target choice
lives in the manifest rather than in module code.

```json
{"split":"train",
 "text":["caption","tokens"],
 "views":{"sensor_a":[0.1,0.2],"sensor_b":[0.3,0.4]},
 "target":["extract","concept","x","done","."]}
```

```bash
python -m thinking.multimodal --selftest

python -m thinking.multimodal --manifest data/multimodal.jsonl \
    --steps 400 --batch 32 --dim 96 \
    --latent-concept-slots 8 --latent-concept-memory-size 64 \
    --latent-concept-discovery-w 0.05 \
    --latent-concept-reanalysis-w 0.05 \
    --latent-concept-gap-w 0.05 \
    --out runs/multimodal.json \
    --checkpoint runs/multimodal.pt
```

## Image

The image stack is manifest-first: fetch/caption/score/curate records, embed them, train latent
flows, then run offline image-quality and alignment evals.

```bash
python -m thinking.image_fetch --source text-to-image-2m-512-2m \
    --max-records 1024 --image-dir data/images/web_fetch \
    --manifest data/images/train_web.jsonl --root data/images \
    --report-out runs/image_fetch_report.json

python -m thinking.image_caption --manifest data/images/train_web.jsonl \
    --root data/images --backend hf --model Salesforce/blip-image-captioning-large \
    --mode replace --batch 16 --device cuda \
    --out data/images/train_web_captioned.jsonl \
    --report-out runs/image_caption_report.json

python -m thinking.image_score --manifest data/images/train_web_captioned.jsonl \
    --root data/images --backend stats --image-size 256 \
    --out data/images/train_web_scored.jsonl \
    --report-out runs/image_score_report.json

python -m thinking.image_latent --train --cond-mode text --flow-arch mmdit \
    --image-manifest data/images/train_web_scored.jsonl --image-root data/images \
    --ae-steps 40 --flow-steps 40 \
    --out runs/image_manifest_mmdit.pt
```

## RunPod

The H100 launcher exposes only supported current jobs. It does not default to a hidden legacy
world; select the job explicitly.

```bash
RUNPOD_API_KEY=... python runpod/launch_thinking.py \
    --text-reading --reading-data data/reading.jsonl \
    --text-reading-steps 40000 --go

RUNPOD_API_KEY=... python runpod/launch_thinking.py \
    --image-fetch --image-caption --image-score --image-latent \
    --image-quality-preset web-hf-vae --go

RUNPOD_API_KEY=... python runpod/launch_thinking.py \
    --text-reading --reading-data data/reading.jsonl \
    --multimodal --multimodal-manifest data/multimodal.jsonl --go
```

Operationally, the launcher still tar-syncs the repo, runs under pod-side `timeout`, tees logs to
local pod disk, copies logs back, and terminates the pod in cleanup.

## Removed Legacy Modules

The old synthetic-language modules were removed from the active package:

- legacy JSON language bank
- `thinking.cli`
- `thinking.config`
- `thinking.distill`
- `thinking.kinship`
- `thinking.world`
- `thinking.verify`
- `thinking.flow`
- `thinking.train`
- `thinking.evaluate`
- `thinking.deep_eval`
- `thinking.probes`
- `thinking.induce`
- `thinking.verbalize`
- `thinking.hybrid_vocab`
- standalone exam/write/mind/meaning prototypes
- `datalog.py`

Do not add replacements for these as hard-coded QA/rule/template layers. New data handling should
be expressed through manifests, corpora, feature views, targets, and learned objectives. For text
self-study, prefer `--reading-study-strategy gap` when latent memory is enabled; it ranks raw
chunks by the model's own graph-predicted missing-concept score. Use `--reading-discovery-w` to
activate the broader discovery objective, including graph-closure insight from partial reading
contexts into fuller concept states.
