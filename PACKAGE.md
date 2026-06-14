# Ponens — the `thinking` package reference

A small LM reads natural-language facts, **thinks in a typed trace language**, and a Datalog
checker validates **every emitted line** before it commits. This document is the package
reference: architecture, the trace language, worlds and exams, the CLI, GPU operations, and the
experimental record.

## 1. Architecture

```
                 facts (NL surface)            question (NL)
                        │                          │
                        ▼                          ▼
              ┌───────────────────────────────────────────┐
              │  ScratchpadLM  (pointer head, dense aux)   │   emits one LINE at a time
              └───────────────────────────────────────────┘
                        │ line
                        ▼
              ┌───────────────────────────────────────────┐
              │  GoalChecker (forward, evidence-first)     │   validates LOCALLY:
              │  · check F   → F ∈ stated facts            │   never proves anything itself
              │  · think B so H → B ⊆ known ∧ rule(H←B)    │
              │  · arithmetic heads → builtin VERIFIES     │
              │  · duplicates rejected                     │
              └───────────────────────────────────────────┘
                 valid → commit, H joins `known`              invalid → resample (≤ retry), else
                                                              answer from the verified derivation
```

**Trace language** (every token survives the word tokenizer; premises ALWAYS precede
conclusions — at line level and trace level; violating this is answer-only supervision and does
not train):

```
check <h> <pred> <t> .                          ground a stated fact
think <B1> [and <B2> ...] so <H> .              derive H from known atoms via a rule
answer <relation | value | person> .            only valid when the derivation supports it
fact <h> <pred> <t> . ... done .                reading exercise (extraction)
write <level> <h> <pred> <t> : <sentence> .     writing exercise
compute <y2> minus <y1> : <diff> .              math drill
define <level> <word> : <definition> .          vocabulary lesson
```

**Model** (`scratchpad_model.py`): 4-layer transformer, RoPE, QK-norm with learnable per-head
attention temperature, tied embeddings, and a **pointer/copy head** (output = learned-gate
mixture of the LM softmax and last-block-head-0 attention scattered onto context token ids) —
copying is architectural, and the dense aux supervision trains exactly that head. Recurrence
(looped / HRM `hier` / TRM / mHC hyper-connections) and Ouro-style learned halting are
implemented but **off by default** (ablation: shared-block recurrence loses 0.93-vs-2.1 to
distinct layers on trace corpora at this scale). **Use `d=256` for multi-rule worlds.**
For multimodal bridge runs, `ScratchpadLM.forward(..., prefix=embeddings)` can prepend continuous
feature-view embeddings before the token stream on **non-pointer** models; pointer models reject
prefixes because the copy distribution is defined over discrete context token ids.

**Engine** (`datalog.py`): least-fixpoint closure with provenance (gold traces = linearized
proof trees), entailment oracle, and SLD backward chaining (`prove`/`options`) for goal-directed
worlds with distractors.

**Verified decode** (`flow.py`), in the order things happen:

1. The model emits a greedy line; the checker validates it.
2. A rejected line is **resampled at temp 0.5** from the model's own distribution. Resampling is
   **prefix-pruned**: a trie over all checker-valid lines (every premise order — the checker
   accepts any body permutation, and a stricter trie silently collapses acceptance) aborts an
   attempt at the first token that cannot extend to a valid line. Identical acceptance set to
   full-line propose-then-reject; failure costs ~2 tokens instead of 18, which funds a 4× retry
   budget.
3. If every retry fails, **skip-don't-die**: the model's own rejected line advances the
   *context* (uncommitted to the checker) and decoding continues. This is load-bearing —
   regenerating from the same clean prefix just reproduces the same line until the budget dies
   (removing it collapsed verified k=3 from 0.95 to 0.10; restored in `13c5bc3`).
4. Every rejection is classified (`reject_causes` in eval output: syntax / check-not-edb /
   premise-unknown / no-rule / goal-anchor / duplicate / builtin-reject) — the taxonomy says
   where per-line error actually lives before you optimize it.

Two stronger interventions were built and measured, and both **lose** to rejection sampling on
a gold-trace-trained policy (forced valid-but-foreign lines push generation off-distribution and
the damage compounds): `--mode masked` (masked repair of rejected lines, 0.50) and
`decode="masked-full"` (every token masked to valid prefixes, 0.40) vs verified 0.80–0.95. Kept
as diagnostics. `decode="ranker"` scores the full legal-action frontier and records the oracle
action's rank — measures trace-policy quality in isolation from syntax failures.

**Representation probes** (`thinking probe`) are the FER/UFR guardrail from arXiv:2505.11581:
they are separate from answer accuracy. The probe teacher-forces verified traces, captures
proof-line hidden states, and reports same-rule reuse, cross-depth reuse, depth-index leakage,
and a heuristic `ufr_score`/`verdict`. A run can solve depth-30 rollout while still being
fractured if rule reuse is weak or tied to line position; the GPU learning-curve summary now
keeps those fields beside `deep_eval_*` accuracy.

## 2. Worlds

- **`chain`** — linear `r`-chains, transitive `far` closure. The minimal benchmark; rung 0.
- **`kinship`** — CLUTRR-style families. 16 interacting rules (gendered bases → parent/sibling
  abstractions → grandparents, aunts/uncles, cousins, nephews/nieces via gender witnesses,
  in-laws, recursive `ancestor`). Questions are NL; the **answer is a relation** the flow must
  derive. Queried links are enforced **rule-only** (no stated fact ever connects the pair).
  - **deep regime** (`--deep-depth N`): N-generation spines with the full relation web as
    distractors + a consistent chronology (`born`/`died` years flow down generations).
  - **value queries**: ages at death, relative ages, hypothetical "how old in YEAR" — the
    checker *verifies* the arithmetic (builtins), it never computes answers.
  - **relativity set**: `older_by` (same fact from either frame), `who_older`/`who_younger`
    (direction flips the answer; pair order randomized).
  - **novel relations**: a rule **defined only in the question** ("a relX of someone is the P1
    of a P2 of that person — who is the relX of H?"); the skill is trained on random
    compositions, evaluated on held-out ones. Answers are nouns reached by linking.

**Entity anonymization** (`world.anonymize`, default ON): all persons renamed per-example to
slot tokens `p0..p159`. Non-negotiable — never-trained name embeddings derail the model's
structural circuits even though the pointer copies them correctly.

## 3. Language: the distilled surface bank

`thinking/distill.py` has a frontier teacher (`claude -p`) write the English; every pattern is
mechanically validated (slot discipline, charset, length) and **split train/eval** so held-out
phrasings measure language understanding. Eight education registers (preschool → scholar) form a
**training curriculum** (the ladder is climbed in cumulative mixtures before the full mix), plus
leveled word definitions — which, for relation words, are *the rules stated in English*.

```bash
python -m thinking.cli train ...           # uses surfaces.json automatically when present
python -m thinking.distill --go            # regenerate the bank (frontier calls)
python -m thinking.distill --definitions   # add/refresh leveled definitions
```

`thinking.text` is the separate Text-0 rung for language understanding beyond the local kinship
surface bank. It consumes web-backed semantic records, not next-token corpora: each example pairs
natural language with a canonical meaning target, and the model must decode `extract fact ...`
traces. The first importer is SNLI premise/hypothesis inference; the target is the human label
(`entailment`, `contradiction`, or `neutral`) rendered as a canonical `nli` fact. The MultiNLI
importer uses the same canonical target across broader genres, so the text rung can learn from
more language domains without adding English rules. The HANS importer adds controlled
adversarial NLI examples for lexical-overlap, subsequence, and constituent heuristics; HANS
`non-entailment` maps to SNLI-compatible `neutral` by default so checkpoints can be scored
across both datasets. The evaluator reports teacher-forced fact accuracy,
direct semantic-head fact accuracy, free decoded fact F1/exact match, per-dataset and
per-heuristic buckets, paraphrase groups when the dataset supplies them, explicit counterfactual
records when supplied, and an NLI artifact control that scores hypothesis-only and premise-only
ablations. Large mixed-language runs can cap teacher/semantic eval with `--fact-n`,
`--kind-fact-n`, and `--artifact-n`; sampled reports explicitly mark `sampled: true`.

```bash
python -m thinking.text --selftest
python -m thinking.text --import-snli --snli-zip /private/tmp/snli_1.0.zip \
    --snli-train 20000 --snli-eval 1000 --seed 11 --out data/text_snli.jsonl
python -m thinking.text --import-mnli --mnli-zip /private/tmp/multinli_1.0.zip \
    --mnli-train 20000 --mnli-eval 2000 --seed 19 --out data/text_mnli.jsonl
python -m thinking.text --data data/text_snli.jsonl --steps 1500 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 --free-n 200 --max-new 20 \
    --out runs/text_snli.json --checkpoint runs/text_snli.pt
python -m thinking.text --import-hans --hans-train 6000 --hans-eval 3000 \
    --seed 13 --out data/text_hans.jsonl
python -m thinking.text --data data/text_hans.jsonl --eval-checkpoint runs/text_snli.pt \
    --free-n 200 --max-new 20 --out runs/text_snli_on_hans.json
python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --steps 1500 --batch 64 --d 192 --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 200 --max-new 20 \
    --out runs/text_snli_hans.json --checkpoint runs/text_snli_hans.pt
python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --steps 1200 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 100 --paraphrase-n 30 --counterfactual-n 30 --max-new 32 \
    --out runs/text_snli_hans_extended.json --checkpoint runs/text_snli_hans_extended.pt
python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --steps 1200 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 --balance-by kind \
    --free-n 80 --kind-free-n 5 --paraphrase-n 20 --counterfactual-n 20 --max-new 32 \
    --out runs/text_snli_hans_balanced.json \
    --checkpoint runs/text_snli_hans_balanced.pt
python -m thinking.text --data data/text_snli.jsonl --data data/text_mnli.jsonl \
    --data data/text_hans.jsonl \
    --steps 120 --batch 48 --d 96 --layers 3 --heads 4 --semantic-w 0.75 \
    --balance-by kind --fact-n 240 --kind-fact-n 40 --artifact-n 240 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_snli_mnli_hans_smoke.json \
    --checkpoint runs/text_snli_mnli_hans_smoke.pt
python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_smoke.pt \
    --steps 40 --batch 32 --study-lr 0.0005 --semantic-w 0.75 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_smoke.json
python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_semantic_replay_smoke.pt \
    --steps 80 --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_semantic_replay_smoke.json
python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_error_replay_smoke.pt \
    --steps 40 --study-rounds 2 --study-strategy errors \
    --study-probe-n 512 --study-hard-max 256 \
    --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_error_replay_smoke.json
python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_select_both_replay_smoke.pt \
    --steps 40 --study-rounds 2 --study-strategy errors \
    --study-probe-n 512 --study-hard-max 256 --study-select-best \
    --study-score-metric both --study-retention-w 2.0 \
    --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_select_both_replay_smoke.json
python -m thinking.text --reading-data NEWER-TECHNIQUES.md \
    --steps 4 --batch 16 --d 96 --layers 2 --heads 4 \
    --latent-concept-slots 6 --latent-concept-layers 1 \
    --reading-max-tokens 96 --reading-min-tokens 8 --reading-eval-n 32 \
    --reading-study-strategy graph --reading-study-probe-n 48 \
    --reading-study-hard-max 24 --reading-study-refresh-steps 1 \
    --reading-memory-size 64 --reading-memory-w 0.05 \
    --reading-association-w 0.05 --reading-association-transitive-steps 3 \
    --reading-association-transitive-w 0.25 \
    --reading-composition-w 0.1 --reading-composition-transitive-steps 3 \
    --reading-composition-transitive-w 0.25 \
    --reading-graph-predict-w 0.2 --reading-graph-predict-transitive-steps 3 \
    --reading-graph-predict-transitive-w 0.25 \
    --reading-context-target-w 0.1 --reading-neighborhood-w 0.05 \
    --reading-neighborhood-batch 8 --reading-neighborhood-probe-n 32 \
    --reading-transition-w 0.05 --reading-transition-batch 8 \
    --reading-cluster-w 0.05 --reading-cluster-batch 16 \
    --reading-cluster-probe-n 32 \
    --out runs/text_raw_reading_graph_study_smoke.json \
    --checkpoint runs/text_raw_reading_graph_study_smoke.pt
python -m thinking.multimodal --manifest data/multimodal_manifest.jsonl \
    --steps 3 --batch 4 --dim 96 --layers 1 --heads 4 --eval-n 20 \
    --latent-concept-slots 6 \
    --latent-concept-memory-size 64 --latent-concept-memory-w 0.05 \
    --latent-concept-association-w 0.05 \
    --latent-concept-association-transitive-steps 3 \
    --latent-concept-association-transitive-w 0.25 \
    --latent-concept-composition-w 0.05 \
    --latent-concept-composition-transitive-steps 3 \
    --latent-concept-composition-transitive-w 0.25 \
    --latent-concept-graph-predict-w 0.1 \
    --latent-concept-graph-predict-transitive-steps 3 \
    --latent-concept-graph-predict-transitive-w 0.25 \
    --latent-concept-neighborhood-w 0.05 \
    --latent-concept-transition-w 0.05 --latent-concept-cluster-w 0.05 \
    --text-checkpoint runs/text_raw_reading_graph_study_smoke.pt \
    --out runs/m0_multimodal_graph_study_smoke.json \
    --checkpoint runs/m0_multimodal_graph_study_smoke.pt
```

Current local Text-0 SNLI baseline (20k train / 1k dev, 1.5k steps, d=192, 4 layers, 6 heads,
semantic auxiliary classification weight 0.75, learned attention readout per canonical fact key)
does **not** gate: free decoded exact/F1 is **0.515** on a 200-example free-decode sample,
teacher-forced fact-value accuracy is **0.519**, and the direct semantic head is **0.519**.
The artifact control is now more meaningful: hypothesis-only decoded accuracy is **0.408**
(`full - hypothesis_only = 0.111`), while hypothesis-only semantic-head accuracy is **0.401**
(`semantic full - hypothesis_only = 0.118`). This is useful as a measurement floor and shows the
model is extracting some premise-hypothesis signal, not a language-mastery claim.

HANS now makes that failure explicit. The SNLI checkpoint scored on HANS without retraining gets
only **0.423** teacher-forced fact accuracy, **0.429** semantic-head accuracy, and **0.430** free
decoded F1 on a 200-example sample; hypothesis-only accuracy is higher than full-input accuracy,
so this checkpoint is still shortcut-sensitive. A HANS-only control run (6k train / 3k eval,
1k steps) does gate at **0.911** teacher-forced, **0.909** semantic-head, and **0.905** sampled
free decoded F1, with lexical-overlap the weakest bucket (**0.812** teacher-forced). The mixed
SNLI+HANS run (26k train / 4k eval, 1.5k steps) is the current honest language-rung baseline:
it improves overall semantic-head accuracy to **0.722** and keeps constituent/subsequence HANS
strong, but it still fails the gate (**0.625** sampled free decoded F1, SNLI dev **0.486**
semantic-head, lexical-overlap HANS **0.595** semantic-head). This is progress toward
shortcut-resistant language understanding, not mastery.

The first MultiNLI integration is wired as infrastructure, not as a mastery run. A sample imports
from the official NYU zip into `data/text_mnli.jsonl` across ten genres, and the training/eval path
supports sampled large-set evaluation without adding hard-coded English rules. The useful outcome
is broader semantic data plumbing that can be mixed with SNLI/HANS and raw-reading records.

The reading-task update path is also wired. `--study-checkpoint` loads an existing text
checkpoint, expands token embeddings and semantic heads for new reading-task vocabulary/facts,
evaluates before, fine-tunes on the reading records, saves to a new checkpoint, and evaluates
after. `--decode-w 0` gives a semantic-only study mode for understanding updates that do not train
the canonical trace decoder on the same step. `--study-strategy errors` adds an explicit
self-study loop: each round samples train records, mines examples where the semantic head is wrong,
and trains on those hard examples plus replay. `--study-select-best` evaluates each study round and
can restore the best weights using semantic, teacher-forced, combined, or bottleneck scoring.

Raw reading is now the active reading-task path. `--reading-data` accepts JSON, JSONL, or TXT
chunks and builds schema-free text windows without answer labels or task-specific heads. Training
creates two corrupted views of each window, updates persistent latent concept memory, mines
association/composition graphs, and can choose hard examples by graph-prediction surprise.
`--reading-study-strategy graph` waits until memory and relations exist, then refreshes the study
pool from nonzero graph scores. Selection is driven by retrieval, context, neighborhood, cluster,
and graph-prediction metrics rather than a task-specific answer harness. The old task-specific
reader/import/candidate path has been removed from the code and is no longer supported here.

A local MPS smoke on `NEWER-TECHNIQUES.md` ran real optimizer steps through this path. Graph-study
refresh produced nonzero mean scores at steps 2-4 (**0.320**, **0.341**, **0.290**), final
`graph_predict_loss` was **0.384**, `memory_active` reached **64**, and the association graph held
**698** active edges. `cluster_acc` moved from **0.000** to **1.000** on the sampled probe. This is
real weight movement and graph self-study, not a claim of language mastery.

The upstream multimodal bridge inherits the same latent concept machinery without a built-in
oracle world. `thinking.multimodal` consumes JSONL manifests with data-supplied named feature
views, text tokens, and target token traces. Text-side latent memory, relation graphs, and
graph-prediction surprise are available upstream to multimodal learning.

The architecture now has a generic schema concept head in `thinking.concepts.SchemaConceptHead`.
Instead of adding another task-specific rule path, text records expose `(slot, predicate) -> value`
concepts through learned key queries and learned value embeddings. `TextFactLM` trains this with
`--fact-concept-w` and reports `fact_concept_head`; checkpoint expansion copies learned concept
queries/value embeddings by symbolic schema identity, not by array position.
A concept-only SNLI check with decoder and old semantic-head losses disabled
(`--decode-w 0 --semantic-w 0 --fact-concept-w 1`) reached sampled `fact_concept_head` **0.405**
after 40 local MPS steps. This is not language mastery; it is the model-side insertion point for
understanding-oriented training rather than a harness-specific branch.

The same concept head now has a schema-generic contrastive geometry objective. Text reading runs
can add `--fact-concept-contrast-w` so records sharing a data-supplied concept value pull the same
slot states together while other values for that slot separate; reports expose
`fact_concept_geometry` nearest-same accuracy and same-vs-different cosine margin.

The geometry head also supports learned per-value prototypes. Text runs expose
`--fact-concept-prototype-w` and `--fact-concept-prototype-spread-w`; targets come from
data-supplied factor ids, while the loss only sees key-local prototype classes and an
anti-collapse margin.

## 3b. Image and vision understanding

The old synthetic visual-factor harness has been removed. The active image path is manifest-driven: image rows, captions, embeddings, quality scores, and preference sidecars come from data files, and concept learning uses latent slots plus concept memory rather than a fixed visual grammar. For image generation, `runpod/launch_thinking.py --image-quality-preset web-hf-vae-hq` is the current high-resolution path: it fetches a mixed 1024+512 web manifest with high-res source upweighting, including WebDataset shards with JSON metadata or plain `.txt` caption sidecars, scores prompt-image quality with PickScore plus technical image-health checks, applies stricter alignment/safety/dedupe/caption-hygiene cleaning, trains 1024/multi-aspect HF-VAE latent buckets with a progressive bucket curriculum, uses latent patching so higher-resolution training stays within a bounded MM-DiT token budget, and runs a 12-block/12-head 768-wide MM-DiT instead of the smoke-test transformer defaults. HQ also enables boundary-enforced double-cosine rectified-flow velocity so the model honors the normalized latent endpoint field at noise/data boundaries, and Karras-style timestep placement so high-step samplers spend more resolution near the decoded data end of the trajectory. HQ runs emit sampler-diverse generated prompt candidates, score those candidates, and mine chosen/rejected preference pairs with per-candidate seed/solver/CFG/schedule provenance; when that portable preference artifact and its candidate images exist locally, the next HQ launch auto-uploads and consumes them for latent quality-scorer pretraining plus direct latent-flow preference updates unless `--image-no-auto-preferences` is set.

Use `thinking.vision_understanding` for visual concept learning:

```bash
python -m thinking.vision_understanding --selftest
python -m thinking.vision_understanding --train \
    --manifest data/images/train_web_ppm_smoke.jsonl --root data/images \
    --steps 200 --out runs/vision_understanding.pt \
    --report-out runs/vision_understanding_report.json
```

Use `thinking.image_curate` after fetch/caption/embed/score to keep the training set aligned
with high-quality generation requirements before any GPU run:

```bash
python -m thinking.image_curate --selftest
python -m thinking.image_curate --manifest data/images/train_web_scored.jsonl \
    --root data/images --min-caption-tokens 4 --min-width 256 --min-height 256 \
    --min-caption-unique-ratio 0.25 --max-caption-token-frequency 0.5 \
    --max-caption-token-run 0.5 --max-caption-char-run 0.35 \
    --max-nsfw 0.2 --max-watermark 0.2 --min-image-text-cosine 0.2 \
    --max-image-duplicate-cosine 0.985 --eval-frac 0.02 \
    --out data/images/train_web_curated.jsonl \
    --report-out runs/image_curate_report.json
```

Use `thinking.image_latent` for text-conditioned latent generation on manifest data:

```bash
python -m thinking.image_latent --selftest
python -m thinking.image_latent --train --cond-mode text --flow-arch mmdit \
    --image-manifest data/images/train_web_ppm_smoke.jsonl --image-root data/images \
    --ae-steps 400 --flow-steps 400 --cond-drop 0.1 --cfg-scale 1.5 \
    --cfg-rescale 0.7 --sample-steps 8 --time-sampling logit-normal \
    --time-shift-mode auto \
    --flow-consistency-w 0.05 --flow-endpoint-w 0.1 \
    --flow-noise-coupling sliced_ot --flow-noise-coupling-projections 4 \
    --dit-attn-impl auto \
    --out runs/image_latent_mmdit_text.pt
python -m thinking.image_latent --eval-checkpoint runs/image_latent_mmdit_text.pt \
    --eval-image-manifest data/images/train_web_ppm_smoke.jsonl --eval-image-root data/images \
    --cfg-scales 1.0,1.25,1.5 --cfg-rescales 0.0,0.7 \
    --sample-steps-list 4,8 --eval-seeds 1,2,3 \
    --eval-out runs/image_latent_eval.json
```

RunPod now launches vision understanding and image latent training through those manifest paths:

```bash
RUNPOD_API_KEY=... python runpod/launch_thinking.py \
    --vision-understanding --image-latent --image-latent-arch mmdit \
    --image-manifest data/images/train_web_ppm_smoke.jsonl --image-root data/images \
    --image-cond-mode text --image-cfg-scale 1.5 --image-cfg-rescale 0.7 \
    --image-sample-steps 8 --image-eval-sweep --fast --go
```

The modern image stack still keeps the efficient latent-flow techniques added earlier: DiT/MM-DiT backbones, adaLN-Zero residual-gated DiT/CrossDiT/MM-DiT conditioning, exact auto SDPA attention by default with linear attention only as an explicit approximation, EMA evaluation, manifest latent caches, dual pooled-plus-token text conditioning, crop/flip/pad-aware Fourier geometry conditioning, Fourier timestep embeddings, text/image feature alignment, quality ranking, generated-pair scorer pretraining, direct flow preference tuning, REPA/self-REPA/SRA alignment, endpoint consistency, sliced-OT noise coupling, logit-normal plus SD3-style mode-biased and adaptive loss-tracked timestep sampling, Karras-style sampler timestep placement, soft Min-SNR velocity weighting, guidance distillation, aspect-ratio buckets, triangular middle-window CFG scheduling, adaptive Heun sampling, decoded-pixel dynamic thresholding for high-CFG samples, and finite-sampling guards. The `web-hf-vae` RunPod preset now defaults to SigLIP So400m pooled prompts plus T5-large text-token sequence conditioning, crop/flip/pad-aware geometry conditioning for multi-aspect targets, Fourier flow-time conditioning, multi-aspect `512x512,768x512,512x768` HF-VAE latent buckets, progressive bucket unlocking during flow training, CFG dropout, decoded-pixel dynamic thresholding, an EMA-teacher guided self-distillation pass, Karras/cosine/linear timestep sweeps, Heun/adaptive-Heun/RK4 sampling sweeps, sampler-diverse multi-candidate reranking, and standard CFG plus CFG++ sweeps. These are implemented against manifest captions and embeddings, not against the removed visual fact harness.

## 3c. Multimodal bridge: manifest features into the same trace language

`thinking.multimodal` is now a generic prefix bridge. It receives data-supplied named feature
views and text tokens from a JSONL manifest, fuses those views with learned latent prefix tokens,
and trains one non-pointer `ScratchpadLM` decoder on the target token trace supplied by the record.
If a dataset wants extraction facts, captions, or another trace format, that target sequence lives
in the manifest rather than in module code.

```json
{"split":"train",
 "text":["caption","tokens"],
 "views":{"sensor_a":[0.1,0.2],"sensor_b":[0.3,0.4]},
 "target":["extract","concept","x","done","."]}
```

```bash
python -m thinking.multimodal --manifest data/multimodal_manifest.jsonl \
    --steps 400 --batch 32 --dim 96 --agreement-w 0.1 \
    --latent-concept-slots 8 --latent-concept-memory-size 64 \
    --latent-concept-memory-w 0.05 --latent-concept-association-w 0.05 \
    --latent-concept-graph-predict-w 0.1 \
    --out runs/multimodal.json --checkpoint runs/multimodal.pt
```

The remaining multimodal objectives are schema-free: cross-mode token-distribution agreement,
VICReg-style latent slot alignment, latent slot factorization, persistent latent memory,
self-mined association/composition graphs, graph prediction, neighborhood alignment, transition
alignment, and cluster consolidation. RunPod forwards the same manifest path with
`--multimodal-manifest` and `--multimodal-root`.

## 4. Exams (the report card)

| exam | command fragment | measures |
|---|---|---|
| reasoning | `--mode verified/free --hops ...` | derivation accuracy by depth |
| composition holdout | `--split holdout` | relations never queried in training |
| novel relations | `--split novel` | rules that exist only in the question |
| reading | `--mode extract` | NL → canonical facts, F1 |
| self-grounded reasoning | `--mode self` | reasoning over the model's OWN extraction |
| writing | `--mode write` | express a fact at an education level |
| math | `--mode math` | subtraction on (almost surely) unseen operand pairs |
| vocabulary | `--mode define` | state a word's definition at a level |
| language understanding | `--phrasings eval` | HELD-OUT surface patterns |
| names | `--train-names` | train-pool vs held-out entities (harness discriminator) |

## 5. CLI

```bash
python -m thinking.cli selftest      # the correctness core: checker, parsers, aux alignment,
                                     # bank coverage, gold traces for every query type
python -m thinking.cli train  --world kinship [--simple|--bank|--canon] [--deep-depth N]
                              [--dim 256] [--steps N] [--batch N] [--no-loop] [--neg] --out RUN
python -m thinking.cli eval   RUN [--mode free|verified|masked|path|extract|self|write|math|
                              define] [--split iid|holdout|novel] [--hops 2,3]
                              [--phrasings train|eval] [--level preschool..scholar|mix] [--n N]
python -m thinking.cli demo   RUN [--k N]
python -m thinking.cli induce [--out rules.json]    # discover the rules from raw observations
python -m thinking.cli ablate                       # recurrence ablation harness
python -m thinking.verbalize  [--corpus tinystories[,cosmopedia]] --out V.pt  # expressive output
```

Run directories are self-contained (`config.json`, `model.pt` with auto-captured model config,
`results.json` accumulating tagged eval cells).

Training mixture (configurable fractions): reasoning traces + reading + writing + math drills +
vocabulary + novel-relation problems, with **rolling pool refresh** (the generator is infinite —
fixed pools memorize and collapse) and an example-packed, never-truncate-mid-example batcher.

## 6. GPU operations

```bash
RUNPOD_API_KEY=... python runpod/launch_thinking.py --stair [--bank] [--canon] \
    [--stair-world chain|kinship] [--dim 256] [--train-steps N] --fast --go
```

Tar-over-ssh sync, pod-side `timeout`, always-terminate in `finally`. Operational rules learned
the hard way: logs tee to the pod's **local disk** (`/workspace` is a network volume that stalls
streaming writes); after killing a launcher, **verify pod state via the REST API** (a hung SSH
also blocks the cleanup that would terminate the pod); pre-flight any data-shape change locally
(`build_examples` dry-run) before it bills a pod.

## 7. Experimental record (the staircase)

Validated bottom-up, one variable per run, after a nine-run zero streak taught us not to stack:

| rung | config | result |
|---|---|---|
| 0 | chain, production trainer, train names | 0.95/0.80 verified k=2/4 |
| 0b | + entity anonymization | **1.00 / 1.00** (held-out = train by construction) |
| A | kinship, built-in NL, d=128 | reads + validates, think-grammar fails |
| A-d256 | + capacity | **1.00 / 0.95 free k=3** |
| B | + 8-register bank + curriculum (20k steps) | 0.45 trained / **0.40 held-out phrasings** |
| B6 | + budget (50k steps, fixed-phase curriculum) | 0.45 — UNCHANGED: phase pools memorize |
| B7b | bank, NO curriculum, always-fresh pool | **0.75 trained / 0.90 held-out phrasings** |
| C4 | + deep-20 trees, verified-decode fix | verified k=20 **0.45** (was ~0), free 0.50; shallow starved by 50% deep mix |
| C6d | + contrastive questions (3/tree) + goal anchor | k=2 OFF ZERO (0.10-0.20); verified k=3 **0.75** |
| L (rope) | trained depth<=6, eval 6/10/20/40 | 0.70/0.40/0.17/0.00 — no cliff; per-line error compounds |
| L (nope) | same, no positions | 0.40/0.60/0.00/0.00 — position mode not decisive |
| CONS | contrastive 0.9, 36k steps (2x C6d) | k=2 **0.45-0.55**, verified k=3 **0.95** / free **1.00**; deep dipped (k=20 free 0.45) — trio inflation diluted effective deep share to ~0.20 |
| C5 | deep_frac 0.3 rebalance (18k) | **verified k=3 0.60 > free 0.55**; deep k=20 0.40-0.45; held-out phrasings ≈ trained |
| TR1 | trace-rank objective, depth-16 train -> depth-30 eval | constrained decode **1.00 acc / 1.00 valid**; ranker top1 0.61, top5 0.80, MRR 0.70 over 16 candidates; decode ~4.2 ms/example |
| LC1 | 500/1000-step learning curve, aux vs noaux, eval depth-30 | all arms reached **1.00 acc / 1.00 valid**; aux@1000 hit top1/MRR **1.00/1.00**; noaux solved rollout but had weak rule geometry (FER margin 0.02-0.06 vs aux 0.62-0.68) |

Latest finding: trace-rank improves usable trace execution before we add any hand-coded
domain rules. The model is learning to select legal next actions from a generic frontier
more reliably; the auxiliary rule losses are not needed for this small depth-30 accuracy
check, but they keep the internal rule space separated enough to support scaling.

Root causes found en route (now defaults/tests): unseen-name embeddings → anonymization;
head-first lines → premises-first; fixed-pool epochs → rolling refresh; d=128 → d=256;
mHC zero-init write gate (block got zero gradient); duplicate-line loops → checker dedup;
fixed-phase curricula memorize → always-fresh mixed pools (`--no-curriculum`);
verified-mode drops → skip-don't-die + temp-0.5 resampling.

## 8. Roadmap

Rung C: deep trees at scale. Rung D: the full exam battery in one model. Then: digit-level
arithmetic traces (atomic-token math can't emit unseen differences), the verbalizer span-copy
fix (BPE fragments names), model-proposed rule induction (the enumerate-and-test learner already
hits 0.99 — the model should propose the candidates), and the self-verify flywheel: verified
traces become the next generation's training data.
