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
image/audio embeddings before the token stream on **non-pointer** models; pointer models reject
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
(`entailment`, `contradiction`, or `neutral`) rendered as a canonical `nli` fact. The HANS
importer adds controlled adversarial NLI examples for lexical-overlap, subsequence, and
constituent heuristics; HANS `non-entailment` maps to SNLI-compatible `neutral` by default so
checkpoints can be scored across both datasets. The grounded importer samples transcript-only
descriptions from `thinking.multimodal` and asks that module to render the canonical multimodal
trace; `thinking.text` parses that trace into target facts, so the text rung does not duplicate
or hard-code the image/audio fact schema. The evaluator reports teacher-forced fact accuracy,
direct semantic-head fact accuracy, free decoded fact F1/exact match, per-dataset and
per-heuristic buckets, paraphrase groups when the dataset supplies them, explicit counterfactual
records when supplied, and an NLI artifact control that scores hypothesis-only and premise-only
ablations.

```bash
python -m thinking.text --selftest
python -m thinking.text --import-snli --snli-zip /private/tmp/snli_1.0.zip \
    --snli-train 20000 --snli-eval 1000 --seed 11 --out data/text_snli.jsonl
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
python -m thinking.text --import-grounded --grounded-train 8000 \
    --grounded-eval 1500 --grounded-counterfactual 100 --seed 17 \
    --out data/text_grounded.jsonl
python -m thinking.text --data data/text_grounded.jsonl --steps 600 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 100 --paraphrase-n 50 --counterfactual-n 50 --max-new 32 \
    --out runs/text_grounded.json --checkpoint runs/text_grounded.pt
python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --data data/text_grounded.jsonl --steps 1200 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 100 --paraphrase-n 30 --counterfactual-n 30 --max-new 32 \
    --out runs/text_snli_hans_grounded.json --checkpoint runs/text_snli_hans_grounded.pt
python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --data data/text_grounded.jsonl --steps 1200 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 --balance-by kind \
    --free-n 80 --kind-free-n 5 --paraphrase-n 20 --counterfactual-n 20 --max-new 32 \
    --out runs/text_snli_hans_grounded_balanced.json \
    --checkpoint runs/text_snli_hans_grounded_balanced.pt
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

Grounded transcript language is now a separate text target tied to the existing image/audio
world rather than another NLI label. The grounded-only run (8k train / 1.6k eval, 600 steps)
gates: **0.9895** teacher-forced fact accuracy, **0.9968** semantic-head accuracy, **0.996**
sampled free decoded F1, **0.88** sampled paraphrase consistency, and **0.996** sampled
counterfactual F1. The unified SNLI+HANS+grounded run (34k train / 5.6k eval, 1.2k steps)
improves global teacher-forced/semantic accuracy to **0.874 / 0.894** and keeps grounded
semantic accuracy high (**0.981**), but it still fails the gate: sampled free F1 is **0.742**,
sampled paraphrase consistency is **0.467**, SNLI dev semantic-head is **0.522**, and HANS
lexical-overlap semantic-head remains **0.594**. The next text rung should scale shortcut-
resistant web data and add richer semantic datasets (e.g. MultiNLI/CFQ/GeoQuery/SQuAD-style
answer facts) plus stronger paraphrase and counterfactual splits.

Kind-balanced mixed training is a useful curriculum control, not a rulebook. With
`--balance-by kind`, the mixed SNLI+HANS+grounded run improves global sampled free F1 to
**0.8125** and lifts HANS lexical-overlap semantic-head accuracy to **0.786**, while keeping
grounded semantic accuracy high (**0.975**). It still fails the language gate: SNLI dev
semantic-head accuracy drops to **0.413** and sampled paraphrase consistency is only **0.65**.
This says the next step is broader language supervision and better paraphrase/counterfactual
coverage, not hand-coded English rules.

## 3b. Image grounding: synthetic visual factors first

`thinking/vision.py` is the Image-0/Image-1 rung for applying the FER hypothesis to pixels
without losing checkability. It generates small colored-shape scenes with exact canonical facts
(`fact p0 color red .`, `fact p0 shape circle .`, `fact p0 left_of p1 .`), trains a compact
vision encoder to recover color/shape factors, and reports visual FER/UFR probes:
same-color reuse across shapes, same-shape reuse across colors, color-shape leakage, and a
`ufr_score`/`verdict`. The encoder now has two architecture choices:

- `shared`: one representation with separate color/shape heads, kept as the historical baseline.
- `bottleneck`: explicit color and shape subspaces plus a decorrelation loss, so diagnostics can
  inspect the actual spaces that feed the factor heads.

```bash
python -m thinking.vision --selftest
python -m thinking.vision --train --steps 2000 --batch 64 --dim 64 \
    --out runs/vision_object_encoder.pt
python -m thinking.vision --train --arch bottleneck --steps 2000 --batch 64 --dim 64 \
    --out runs/vision_bottleneck.pt
python -m thinking.image2 --steps 400 --seeds 0,1,2 --out runs/image2_bottleneck.json
python -m thinking.image_flow --train --steps 400 --out runs/image_flow.pt
python -m thinking.image_latent --train --ae-steps 400 --flow-steps 400 \
    --out runs/image_latent_flow.pt
python -m thinking.image_latent --train --flow-arch dit --ae-steps 400 --flow-steps 400 \
    --cond-drop 0.1 --cfg-scale 1.5 --sample-steps 8 --flow-semantic-w 0.25 \
    --out runs/image_latent_dit.pt
python -m thinking.image_latent --train --cond-mode text --flow-arch mmdit \
    --ae-steps 400 --flow-steps 400 --cond-drop 0.1 --cfg-scale 1.5 \
    --sample-steps 8 --flow-semantic-w 0.25 --time-sampling logit-normal \
    --flow-ema-decay 0.999 --ae-intervention-w 0.1 --ae-factor-orth-w 0.05 \
    --semantic-guidance-w 2.0 \
    --out runs/image_latent_mmdit_text.pt
python -m thinking.image_latent --eval-checkpoint runs/image_latent_dit.pt \
    --cfg-scales 1.0,1.25,1.5,2.0 --sample-steps-list 4,8,16 \
    --eval-seeds 1,2,3 --roundtrip-samples 2 --eval-out runs/image_latent_dit_sweep.json
python -m thinking.audio --steps 500 --seeds 0,1,2 --out runs/audio1_fer.json
python -m thinking.multimodal --steps 400 --out runs/m0_multimodal.json
RUNPOD_API_KEY=... python runpod/launch_thinking.py --vision --vision-arch bottleneck \
    --image2 --image-flow --image-latent --image-latent-arch dit \
    --image-cond-drop 0.1 --image-cfg-scale 1.5 --image-sample-steps 8 \
    --image-flow-semantic-w 0.25 --image-eval-sweep \
    --audio --multimodal --fast --go
RUNPOD_API_KEY=... python runpod/launch_thinking.py --image-latent --image-latent-arch mmdit \
    --image-cond-mode text --image-dit-head-width-mult 2 \
    --image-cond-drop 0.1 --image-cfg-scale 1.5 \
    --image-sample-steps 8 --image-flow-semantic-w 0.25 \
    --image-time-sampling logit-normal --image-flow-ema-decay 0.999 \
    --image-latent-normalize channel --image-latent-stat-samples 1024 \
    --image-cfg-interval 0.0,0.8 --image-semantic-guidance-interval 0.0,0.75 \
    --image-ae-intervention-w 0.1 --image-ae-factor-orth-w 0.05 \
    --image-semantic-guidance-w 2.0 --image-semantic-guidance-sweep 0.0,1.0,2.0 \
    --image-sample-methods euler,heun \
    --image-sample-grid \
    --image-eval-sweep --fast --go
```

`thinking.image2` is the head-aware FER experiment: shared factored heads vs explicit
bottleneck vs one joint color×shape classifier on held-out color/shape combinations. It reports
both the old embedding probe and the new factor-space probe, fixing the Image-1 blind spot where
the embedding layer missed where factorization lived.

`thinking.image_flow` is the first generation scaffold. It trains a small fact-conditioned
rectified-flow model in pixel space:

```
canonical visual facts -> condition vector -> velocity field -> image
```

`thinking.image_latent` is the next scale rung:

```
image -> semantic AE latent -> fact-conditioned latent velocity field -> decoder -> image
```

The autoencoder is deliberately semantic, not reconstruction-only: it predicts color/shape facts
from the compressed latent while reconstructing the image. That keeps the latent aligned with the
same factorization requirement as the FER probes and gives us the right insertion point for a
future DiT/MMDiT backbone.

This follows the direction used by modern image systems: latent diffusion reduces generation
cost by operating in compressed spaces (LDM, arXiv:2112.10752), diffusion transformers replace
the U-Net backbone at scale (DiT, arXiv:2212.09748), and current high-resolution systems use
rectified-flow transformer variants (Stable Diffusion 3, arXiv:2403.03206) or linear-attention
efficiency paths (Sana, arXiv:2410.10629). Newer RAE-style work pushes the same point harder:
the autoencoder latent should be semantically rich enough for DiT training, not just a lossy
pixel codec. Our implementation is still synthetic, but the path now matches that shape.

Current finding: the H100 Image-1 baseline reaches **1.00 color / 1.00 shape accuracy** after
2k steps, but the visual factor probe still reports `high_fer_risk` (`ufr_score=0.0`) because
color and shape remain strongly entangled. A two-arm held-out-combo experiment also shows the
factored head generalizes better than a joint color×shape classifier (≈0.30 vs 0.00 holdout),
while the embedding-layer probe alone misses where that factorization lives. So for image, as
for trace reasoning, label accuracy is not enough; probes must inspect the right layer/head.

Image-2 GPU update (H100, 1000 steps, seed 0): the explicit bottleneck keeps **1.00 seen
accuracy**, reaches **0.84 held-out color/shape combo accuracy** vs **0.61** for shared factored
heads and **0.00** for the joint classifier, and cuts factor-space leakage from **0.75 → 0.10**.
The factor-space probe now tracks behavior (`best_factor_space_arm = bottleneck`), while the old
shared embedding probe still reports `ufr_score=0.0` because the concatenated representation is
still entangled. `thinking.image_flow` also trains the first fact-conditioned rectified-flow
scaffold to `velocity_mse ≈ 0.29` after 1000 steps; this proves the generation path is wired, not
that image quality is solved.

A separate tracked Image-2 sweep (`runs/image2_fer.json`, 600 steps × 3 seeds) records the
stronger intervention target: held-out combo accuracy moves **0.65 → 0.97 → 1.00** across
factored, swap, and subspace arms, and the linear-decodability probe ranks the arms in the same
order as behavior. Treat that as the next reproducibility target for the public command path.

Image-3 GPU update (H100, 1000 AE steps + 1000 latent-flow steps): `thinking.image_latent`
with the conv velocity baseline reaches `recon_mse=0.015`, latent color accuracy **1.00**, latent
shape accuracy **0.86**, `latent_velocity_mse=1.05`, and center-target sample MSE **0.044**. This
proves the semantic latent path can preserve canonical factors before moving transport training
off pixels. The module now also has `--flow-arch dit`, a tiny patch-token DiT velocity field over
the same semantic latents. That is the toy-scale version of the architecture we need to scale:
semantic image latents + fact/text conditioning + rectified-flow transformer.

Image-4 GPU update (H100, 1000 AE steps + 1000 DiT-flow steps): the patch-token DiT flow keeps the
same semantic AE quality (`recon_mse=0.015`, color **1.00**, shape **0.86**) and improves the
transport objective vs the conv baseline: `latent_velocity_mse` **1.05 → 0.75** and center-target
sample MSE **0.044 → 0.020**. This makes DiT the default scale direction; the conv flow remains a
small baseline.

The image generator now also has the first generation-faithfulness check: every color×shape
condition is sampled, decoded back to pixels, re-encoded by the semantic AE, and scored for
round-trip color/shape accuracy. The same code path supports classifier-free condition dropout
and guided latent sampling (`--cond-drop`, `--cfg-scale`, `--sample-steps`), so the GPU rung can
measure whether stronger conditioning improves generated-image facts rather than only lowering a
single target-image MSE.

Image-5 GPU update (H100, 1000 AE steps + 1000 DiT-flow steps, `cond_drop=0.1`,
`cfg_scale=1.5`, 8 sample steps): guided latent DiT keeps the earlier reconstruction quality
(`recon_mse=0.015`, color **1.00**, shape **0.86**) and center-target sample MSE remains strong
at **0.0198**. The new round-trip metric is the important finding: generated samples decode back
to the requested color **0.87**, shape **0.67**, and both facts **0.53** across all 15 color×shape
conditions. That means the architecture is now measuring image fact-faithfulness, and the next
image work should attack generated shape fidelity rather than only optimizing latent velocity MSE.

The next image intervention is a REPA-style semantic endpoint loss for the flow: the predicted
clean latent endpoint is read by the frozen semantic AE fact heads and optimized against the
conditioning facts (`--flow-semantic-w`). This is not a hard-coded shape rule; it is a generic
alignment pressure from condition slots to the representation heads already used for evaluation.
It follows the recent lesson that image generation quality depends on semantic representations
inside the denoising/flow model, not only lower transport MSE.

Image-6 GPU update (H100, same 1000+1000 DiT setup, `flow_semantic_w=0.25`): semantic endpoint
alignment fixes the shape-fidelity bottleneck measured by Image-5. Round-trip generated samples
now decode to color **0.87**, shape **1.00**, and both facts **0.87** across all 15 color×shape
conditions, up from **0.87 / 0.67 / 0.53**. The tradeoff is small but real: latent velocity MSE
is **0.78** vs **0.77**, center-target sample MSE moves **0.0198 → 0.0251**, and conditional
sample MSE moves **0.0739 → 0.0848**. For this foundation, the result says representation
alignment is buying semantic faithfulness at a modest pixel-MSE cost.

Image-7 robust checkpoint sweep (`--eval-checkpoint`, no retraining): sampler choice is now
measurable as a Pareto problem without rerunning training. On the Image-6 checkpoint, sweeping
CFG `{1.0,1.25,1.5,2.0}` and steps `{4,8,16}` across eval seeds `{1,2,3}` with two samples per
condition finds the best aggregate fact-faithfulness at `cfg=1.5, steps=4`: color **0.967 ±
0.027**, shape **0.944 ± 0.016**, both facts **0.911 ± 0.031**, and conditional sample MSE
**0.065**. Higher guidance (`cfg=2.0, steps=4`) slightly improves shape mean (**0.944**) but
hurts color and pixel distance (`both=0.900`, MSE **0.095**). The earlier one-seed `cfg=2.0`
winner was therefore too brittle; robust selection points to moderate CFG plus few Euler steps.

Image-8 starts the language-to-image path without hard-coding generator rules: the latent flow can
now take `--cond-mode text`, where a learned prompt encoder maps prompt tokens into the continuous
condition vector consumed by the DiT velocity field. Canonical facts still supervise semantic
endpoint alignment during training, but checkpoint save/load and sampler sweeps now preserve the
prompt vocabulary and text encoder, so inference no longer needs a hand-built fact vector.

Image-9 adds `--flow-arch crossdit`: image latent tokens now cross-attend to prompt/fact condition
tokens instead of receiving only one pooled condition vector. This is still toy-scale, but it moves
the scaffold toward the token-to-token conditioning pattern used by modern rectified-flow T2I
transformers and gives us a concrete place to plug in richer text encoders later.

Image-10 adds paper-aligned timestep sampling for rectified-flow training: `--time-sampling
logit-normal` biases interpolation times toward intermediate noise scales while keeping `uniform`
available as the baseline. The report records the sampled time mean/std in `last_flow`, so GPU
sweeps can compare schedule effects against sampler/CFG effects.

Image-11 adds `--flow-arch mmdit`: a tiny dual-stream multimodal DiT flow with separate image and
condition-token projections, joint attention, and modality-specific feed-forwards. Unlike the
one-way `crossdit` scaffold, prompt/fact tokens can also absorb image-token state inside each block,
matching the bidirectional multimodal token-mixing direction of modern rectified-flow T2I systems.

Image-12 adds condition-adaptive modulation to MM-DiT blocks: the time/text condition vector now
modulates the attention and feed-forward normalizations for both image and condition streams. The
adapter is zero-initialized, so it starts as the previous MM-DiT block and learns stronger
conditioning without destabilizing checkpoint-smoke training.

Image-13 adds AdaLN-style residual gates to the adaptive MM-DiT branches. Attention and feed-forward
updates in both streams now have condition-dependent gates, and old MM-DiT checkpoints without gate
weights are tolerated by the loader with zero-initialized identity-centered gates.

Image-14 adds EMA checkpointing/evaluation for the latent flow and learned text conditioner. Train
reports can evaluate the averaged weights, checkpoints save both raw and EMA states, and
`--eval-checkpoint` prefers EMA when available while keeping `--no-ema-checkpoint` for raw-weight
comparisons.

Image-15 fixes the short-run EMA failure found on the first MMDiT text GPU smoke: a target decay of
0.999 made the averaged checkpoint lag raw weights badly at 400 steps. EMA now uses a generic warmup
schedule that ramps toward the requested decay, records the effective decay in reports/checkpoints,
and keeps `--no-ema-warmup` / `--image-no-ema-warmup` for exact-decay ablations.

Image-16 makes raw-vs-EMA evaluation measured instead of assumed. Train reports and
`--eval-checkpoint` now support `raw`, `ema`, and `auto` weight modes; `auto` evaluates the available
weight candidates and selects by the existing semantic round-trip / MSE sweep score. This prevents a
short-run EMA checkpoint from hiding a faster-learning raw model while still allowing EMA to win when
the metrics support it.

Image-17 adds a FER/UFR latent intervention diagnostic. The probe learns fact-value prototype
directions from encoded examples, edits one requested fact in latent space, decodes/re-reads the
image, and reports target-change accuracy plus collateral stability. It is data-derived rather than
a renderer rule. On the fetched 400-step MMDiT text H100 checkpoint, a 16-sample smoke reports
`latent_intervention_score=0.408`: color directions are strong, while shape edits still disturb color
too often. That gives the next representation-quality target beyond output MSE and round-trip score.

Image-18 adds an optional semantic-AE intervention training loss (`--ae-intervention-w`; RunPod:
`--image-ae-intervention-w`). It uses batch-derived fact prototype deltas, supervises edited target
facts, and penalizes collateral drift after decode/re-encode. A local 60/10-step smoke moved
`latent_intervention_score` from **0.079 -> 0.102** with similar reconstruction MSE, so this is now
the recommended next H100 knob while remaining opt-in for ablations.

Image-19 H100 update (800 AE + 800 MMDiT text-flow steps, `ae_intervention_w=0.1`): the intervention
regularizer is a real representation win. The train report reaches `latent_intervention_score=0.705`
with target image-edit accuracy **0.820** and collateral stability **0.859**, up from the previous
fetched 400-step diagnostic score **0.408**. The checkpoint sweep selects EMA at `cfg=1.5, steps=4`
and reports color **1.00**, shape/both **0.778 ± 0.137**, and conditional sample MSE **0.056**.
Shape edits are still the weak axis, but the image stack now has a generic learned pressure that
improves editable semantic latents rather than only lowering reconstruction or flow MSE.

Image-20 adds a generic latent factor-orthogonality diagnostic and optional training loss
(`--ae-factor-orth-w`; RunPod: `--image-ae-factor-orth-w`). It estimates fact-value prototype
subspaces from the batch and penalizes squared cosine overlap between different fact predicates, so
the pressure is over `FACT_VOCAB` groups rather than color/shape rules. A 60/10-step local smoke
reduced eval `latent_factor_orth_loss` from **0.156 -> 0.131** with comparable reconstruction MSE;
the tiny intervention score did not move, so this is a representation-health knob to combine with
the intervention loss on the next H100 run, not yet a standalone quality claim.

Image-21 adds sampling-time semantic guidance (`--semantic-guidance-w`; RunPod:
`--image-semantic-guidance-w`). It is classifier-style guidance, but the classifier is the learned
semantic AE itself: each Euler step can take a normalized gradient through either latent heads or
decode/re-read heads toward the requested canonical facts. On the fetched H100 MMDiT text checkpoint,
decoded guidance at `w=2.0`, `cfg=1.5`, `steps=4`, seeds `{1,2,3}` moves generated shape/both
round-trip accuracy from **0.800 -> 0.978** while color stays **1.00**; conditional sample MSE moves
from **0.0509 -> 0.0562**. This directly attacks the remaining shape-fidelity gap without adding
renderer rules.

Image-22 makes semantic guidance a measured sweep axis (`--semantic-guidance-weights`; RunPod:
`--image-semantic-guidance-sweep`) rather than a fixed post-hoc knob. On the fetched H100 checkpoint,
the same three-seed sweep over `{0, 1, 2}` selects `sem=2`: both-fact accuracy moves
**0.800 -> 0.933 -> 0.978** as guidance increases, while conditional MSE moves
**0.0509 -> 0.0522 -> 0.0562**. This turns semantic guidance into a Pareto selection problem beside
CFG and sampler steps.

Image-23 makes the rectified-flow ODE solver a measured sweep axis (`--sample-methods`; RunPod:
`--image-sample-methods`) with Euler and Heun methods. On the fetched H100 checkpoint, the joint
`{euler, heun} × {sem=0,1,2}` sweep still selects `euler;sem=2`: Heun reaches the same both-fact
accuracy at each semantic-guidance level but has higher conditional MSE on this straightened toy
flow. That keeps Euler as the current default while giving future less-straight flows a fair solver
comparison.

Image-24 adds dependency-free PPM sample-grid export (`--sample-grid-out`; RunPod:
`--image-sample-grid`). The grid uses the same selected sampler/guidance settings as checkpoint
evaluation, so fact metrics and visual artifacts can be audited together. This is necessary for
image-generation work: the fact heads can say shape is correct, but the artifact makes pixel-level
failures visible.

Image-25 adds a lightweight wide DiT velocity head (`--dit-head-width-mult`; RunPod:
`--image-dit-head-width-mult`) for DiT, Cross-DiT, and MM-DiT flows. The default remains the
old linear head for checkpoint compatibility; setting the multiplier above 1 gives the transformer
a wider projection head for semantically rich latents. This follows the RAE/REPA direction: improve
the model's capacity to operate in representation space without adding renderer-specific rules.

Image-26 adds checkpointed latent normalization (`--latent-normalize none|global|channel`;
RunPod: `--image-latent-normalize`). The flow can now train and sample in normalized AE latent
coordinates while semantic losses, guidance, decoding, and visual grids still operate in raw AE
latent space. This is a generic scale fix for semantically rich/high-dimensional latents: it
stabilizes coordinate scale without hard-coding visual factors or renderer rules, and old
checkpoints default to `none`.

Image-27 adds guidance intervals (`--cfg-interval`, `--semantic-guidance-interval`; RunPod:
`--image-cfg-interval`, `--image-semantic-guidance-interval`). CFG and semantic AE guidance can
now be active over selected rectified-flow time ranges while defaults preserve old always-on
behavior. This follows the REPA/guidance-interval direction: guidance becomes a measured sampler
schedule rather than a hard-coded every-step push, and it stays generic over conditions.

## 3c. Multimodal bridge: image + audio into the same trace language

`thinking.audio` transposes the Image-2 FER setup to sound: pitch × timbre × envelope are rendered
as waveforms and log-spectrograms, with held-out factor combinations and the same factored/swap/
subspace intervention ladder. The target facts use the same h/pred/t convention:
`fact a0 pitch n440 .`, `fact a0 timbre saw .`, `fact a0 env decay .`.

`thinking.multimodal` is M-0: one non-pointer `ScratchpadLM` receives continuous image patch
prefixes, audio spectrogram prefixes, and transcript prefixes from
`data/multimodal_transcripts.json`, then emits one canonical extraction trace:

```
[IMG prefix][AUD prefix][TXT prefix] extract
  fact p0 color red . fact p0 shape circle .
  fact a0 pitch n440 . fact a0 timbre saw . fact a0 env decay . done .
```

This is the multimodal version of the project invariant: every modality grounds into canonical
facts before the checker/reasoner sees it. Image generation stays connected through the same
semantic latent path (`image -> latent -> facts/generation`) rather than becoming a detached
pixel model.

M-0 local update (240 steps, value-token weighted loss, all-mode ablations every step): the
tracked run now gates three paths separately. Teacher-forced full multimodal accuracy is color
**1.00**, shape **1.00**, pitch **0.79**, timbre **0.94**, envelope **1.00**. Text-only held-out
transcript phrasings, with image/audio zeroed, score **0.92 / 0.97 / 0.48 / 0.91 / 0.95**. Free
greedy text-only extraction reaches **0.80 exact trace-fact match** over the small eval sample.
The stronger semantic gate is counterfactual text intervention: edit exactly one described factor
in a held-out transcript, keep image/audio zeroed, and require only the matching canonical fact to
move. Teacher-forced value-position intervention has mean target accuracy **0.92** and collateral
stability **0.93**. The free greedy version, where the model must emit the edited fact trace from
`extract` alone, reaches mean target accuracy **0.95**, collateral stability **0.90**, and exact
all-fact match **0.62**. This does not prove broad natural-language understanding; it proves that
the bridge can learn a configurable transcript surface bank as a sensory input and bind edited
words to canonical facts rather than merely predicting the next caption token.

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
