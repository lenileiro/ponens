# Ponens Package Reference

The supported `thinking` package is manifest-driven. It no longer ships the legacy synthetic
English bank, proof-checking CLI, rule-induction harness, or verbalizer. Keep
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
span completion, context closure, reanalysis, memory-gap, and graph-closure insight
objectives. No English rules or prompt templates are embedded in the module, and legacy dataset
importers are not supported APIs.
Checkpoints carry a bounded raw-reading replay bank selected from the model's own study records,
so continuation can retain earlier concepts without a separate task-specific harness. Optional
`--reading-study-self-teach-w` allocates extra training weight from the model's own eval deficits
across view, context, span, closure, sequence, neighborhood, cluster, FER, and bridge signals.
`--reading-objective-profile mastery` is the default raw-reading posture: it raises only these
schema-free objective weights to practical floors, including self-teach, discovery, gap,
graph, bridge, cluster, and reanalysis losses, then runs a selected-round study loop with
patience and a mastery-score target. In `auto` study mode, each round routes to the
weakest evaluated signal, such as sequence, closure, FER, or discovery, and each
round branches from the best accepted state so rejected self-teaching attempts do not
poison the next attempt. Concept-memory study rounds also report a model-derived
concept-insight delta from weak-signal gains and bridge connectivity, giving the loop
a generic "new connection discovered" acceptance path without English-specific rules.
Checkpoint continuation also turns on replay and retention from the checkpoint's own
reading replay bank when available; replay rows carry model-derived priority and
reasons such as hard-study examples or concept-insight records, and continuation
sampling uses those priorities before falling back to uniform replay. Checkpoints also
carry a compact reading-mastery history, so long training runs preserve each session's
score deltas, accepted update, self-teach signal, replay priority counts, and concept
insight evidence without task-specific labels. Use `manual` for exact low-level
ablations.
Optional `--latent-concept-topk` applies the shared latent-slot sparsity gate used by
multimodal, keeping only the strongest schema-free concept slots per record so reading
runs can encourage slot specialization without task-specific rules.

```bash
python -m thinking.text --selftest

python -m thinking.text --reading-data data/reading.jsonl \
    --steps 40000 --batch 16 --d 256 --layers 4 --heads 8 \
    --latent-concept-slots 4 --latent-concept-topk 2 --reading-memory-size 256 \
    --reading-objective-profile mastery \
    --reading-study-strategy auto --reading-study-probe-n 256 \
    --reading-discovery-w 0.05 --reading-reanalysis-w 0.05 \
    --reading-gap-w 0.05 --reading-span-completion-w 0.05 \
    --out runs/text_raw_reading.json \
    --checkpoint runs/text_raw_reading.pt

python -m thinking.text --reading-data data/new_reading.jsonl \
    --reading-checkpoint runs/text_raw_reading.pt \
    --reading-out-checkpoint runs/text_raw_reading_studied.pt \
    --steps 4000 --batch 16 \
    --reading-objective-profile mastery \
    --reading-study-strategy auto --reading-study-probe-n 256 \
    --reading-discovery-w 0.05 --reading-reanalysis-w 0.05 \
    --reading-gap-w 0.05 --reading-span-completion-w 0.05 \
    --reading-replay-w 0.05 \
    --out runs/text_raw_reading_study.json
```

## Multimodal

`thinking.multimodal` receives JSONL rows with optional text, named feature views, and optional
target tokens. The model learns to fuse views into continuous prefixes for one decoder, and can
train partial views with the shared concept-completion objective via
`--latent-concept-completion-w`. Discovery hard study also includes concept-completion surprise,
so multimodal data can surface records where partial views fail to reconstruct the full latent
state. If a dataset wants captions, extraction facts, actions, or no decoder target at all, that
target choice lives in the manifest rather than in module code.

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
    --latent-concept-completion-w 0.05 \
    --out runs/multimodal.json \
    --checkpoint runs/multimodal.pt
```

## Image

The image stack is manifest-first: fetch/caption/score/curate records, embed them, train latent
flows, then run offline image-quality and alignment evals. Embedding sidecars can carry pooled
text/image vectors plus token-level text and image sequences, so REPA-style visual alignment can
learn from patch-token targets instead of only global image descriptors. High-res runs can cap
`image_embedding_sequence` tokens before REPA caching/training to keep that richer supervision
linear in the selected token budget instead of the source encoder patch count.

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
- `thinking.crossmodal`
- standalone exam/write/mind/meaning prototypes
- `datalog.py`

Do not add replacements for these as hard-coded task, rule, or template layers. New data handling should
be expressed through manifests, corpora, feature views, targets, and learned objectives. For text
self-study, prefer `--reading-study-strategy auto`: with latent memory it resolves to discovery,
and without memory it falls back to closure study. Discovery ranks raw chunks by the model's own
graph-predicted gaps, graph-closure insight, bridges, sequence surprise, and partial-context
closure surprise from prefix/suffix readings into fuller concept states. Enable
`--reading-study-self-teach-w` to let round selection turn those eval deficits into extra
self-supervised objective weight instead of manually picking which surface to emphasize.
