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
python -m thinking.image_latent --train --cond-mode text --flow-arch dit \
    --ae-steps 400 --flow-steps 400 --cond-drop 0.1 --cfg-scale 1.5 \
    --sample-steps 8 --flow-semantic-w 0.25 --out runs/image_latent_dit_text.pt
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
RUNPOD_API_KEY=... python runpod/launch_thinking.py --image-latent --image-latent-arch dit \
    --image-cond-mode text --image-cond-drop 0.1 --image-cfg-scale 1.5 \
    --image-sample-steps 8 --image-flow-semantic-w 0.25 --image-eval-sweep --fast --go
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
