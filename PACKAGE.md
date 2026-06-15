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
`--reading-objective-profile mastery` is the default raw-reading posture: it raises these
schema-free objective weights and study controls to practical floors, including self-teach,
discovery, gap, graph, bridge, cluster, reanalysis, bounded hard-study mining, and periodic
structure refreshes, then runs a selected-round study loop with patience and a mastery-score
target. In `auto` study mode, each round routes to the
weakest evaluated signal, such as sequence, closure, FER, or discovery, and each
round branches from the best accepted state so rejected self-teaching attempts do not
poison the next attempt. Concept-memory study rounds also report a model-derived
concept-insight delta from weak-signal gains and bridge connectivity, giving the loop
a generic "new connection discovered" acceptance path without English-specific rules.
Selected-round study can also accept a guarded representation-insight update when internal
organization improves enough, so a round can be kept for a real "light bulb" reorganization even
when the surface score alone is not decisive.
Checkpoint continuation also turns on replay and retention from the checkpoint's own
reading replay bank when available; replay rows carry model-derived priority and
reasons such as hard-study examples or concept-insight records, and continuation
sampling uses those priorities before falling back to uniform replay. Checkpoints also
carry a compact reading-mastery history, so long training runs preserve each session's
score deltas, accepted update, self-teach signal, replay priority counts, and concept
insight evidence without task-specific labels. The history also stores label-free
representation-progress evidence over FER, bridge, sequence, neighborhood, cluster, span, and
closure signals, plus compact sampled parameter-update evidence, including selected-round
attempts that are rolled back, so runs can verify that reading actually moved weights and changed
internal organization. Checkpoint study feeds prior concept-insight and representation weakness
signals back into self-teach, so previously discovered connections can shape later reading
updates. Each run also writes a compact `learning_event` when an applied update both moved sampled
weights and improved score, representation organization, or concept-connection evidence; later
text and multimodal self-teach priors can reuse that event without task labels. The default
`mastery` profile also owns the practical hard-study defaults: it probes a
bounded candidate set, caps selected hard records, and periodically refreshes neighborhood and
cluster mining instead of requiring those controls to be repeated in every launch command. Triggered
learning events also add replay reasons to the selected study records, so continuation can revisit
chunks tied to actual weight movement and internal reorganization. During checkpoint continuation,
the mastery profile now includes those replay-bank rows in the primary self-supervised study pool
and samples priority rows more often, while acceptance is still scored on the new reading corpus.
Raw-reading vocabularies are capped by default with frequency-based retention
(`--reading-max-vocab`, set `0` to disable), preserving known checkpoint tokens while mapping the
long tail to `<unk>` instead of growing the embedding table for every one-off corpus token.
Use
`manual` for exact low-level
ablations.
Optional `--latent-concept-topk` applies the shared latent-slot sparsity gate used by
multimodal, keeping only the strongest schema-free concept slots per record so reading
runs can encourage slot specialization without task-specific rules.

```bash
python -m thinking.text --selftest

python -m thinking.text --reading-data data/reading.jsonl \
    --steps 40000 --batch 16 --d 256 --layers 4 --heads 8 \
    --latent-concept-slots 4 --latent-concept-topk 2 --reading-memory-size 256 \
    --out runs/text_raw_reading.json \
    --checkpoint runs/text_raw_reading.pt

python -m thinking.text --reading-data data/new_reading.jsonl \
    --reading-checkpoint runs/text_raw_reading.pt \
    --reading-out-checkpoint runs/text_raw_reading_studied.pt \
    --steps 4000 --batch 16 \
    --out runs/text_raw_reading_study.json
```

## Multimodal

`thinking.multimodal` receives JSONL rows with optional text, named feature views, and optional
target tokens. The model learns to fuse views into continuous prefixes for one decoder, and can
train partial views with the shared concept-completion objective via
`--latent-concept-completion-w`. Discovery hard study also includes concept-completion surprise,
so multimodal data can surface records where partial views fail to reconstruct the full latent
state. If a dataset wants captions, extraction facts, actions, or no decoder target at all, that
target choice lives in the manifest rather than in module code. Multimodal runs can warm-start
from `thinking.text` checkpoints, inheriting latent top-k sparsity and reading-mastery history;
`--self-teach-history-prior-w` lets multimodal self-teach reuse prior abstract weaknesses such as
FER, bridge, sequence, and mode-floor deficits without labels. Text checkpoint concept-insight
events also reinforce multimodal bridge/discovery self-teach, so accepted "new connection"
evidence can transfer across modalities. Text checkpoint representation-progress history also
transfers as FER/bridge/sequence self-teach pressure. The warm-start report surfaces whether the
source reading stage changed sampled weights and organization signals, giving multimodal training
evidence that it is inheriting learned parameters rather than only configuration. For target-aware
transfer, `--text-transfer-probe-n` probes the current multimodal manifest before and after a text
checkpoint import; by default it probes 64 records, requires a `0.1` target-score gain, and keeps
`--text-transfer-gate` enabled so harmful imports are rolled back and their reading-history prior
is not trusted for self-teach. The import report also exposes source `learning_event` counts,
top signal, kind, score, and priority-study evidence, so multimodal runs can see whether the text
source learned by moving weights, revisiting event-linked chunks, and reorganizing concepts before
it transferred. Multimodal train reports now write their own generic `learning_event` from applied
weight movement, score gain, bridge insight, and representation reorganization, and multimodal
checkpoints now persist a bounded `multimodal_learning_history` so continuation can reuse multiple
prior learning events and representation-progress summaries for later self-teach. Multimodal
checkpoint import reports expose the source history count, event counts, top signal/kind, and
summary alongside the latest event. Multimodal train reports include the same
bounded sampled parameter-update summary for the current run, including attempted selected rounds
that are later rolled back. Optional `--representation-probe-n` records a before/after
label-free organization report over FER, bridge, and sequence signals, so a run can distinguish
surface task progress from better internal concept structure. `--multimodal-checkpoint`
warm-starts compatible weights from an earlier multimodal run and converts its representation
progress report into a self-teach prior, letting later runs continue from discovered internal
weaknesses instead of restarting from a surface task score.

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
    --representation-probe-n 128 \
    --out runs/multimodal.json \
    --checkpoint runs/multimodal.pt

python -m thinking.multimodal --manifest data/multimodal_next.jsonl \
    --text-checkpoint runs/text_raw_reading_discovery_study_smoke.pt \
    --multimodal-checkpoint runs/multimodal.pt \
    --steps 400 --batch 32 --dim 96 \
    --latent-concept-slots 8 --latent-concept-memory-size 64 \
    --self-teach-w 0.05 --self-teach-history-prior-w 1.0 \
    --representation-probe-n 128 \
    --out runs/multimodal_continued.json \
    --checkpoint runs/multimodal_continued.pt
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
