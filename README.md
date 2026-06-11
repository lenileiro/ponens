# Ponens — tiny models that reason in checkable steps

An in-house research stack betting that **reasoning should be emitted, not latent**: a small
language model *thinks out loud* in a typed trace language, and a Datalog engine **checks every
line as it is decoded** — a proof checker in the loop, never a prover. Wrong steps get rejected
and resampled; conclusions are only accepted when a verified derivation supports them.

```
pearl is the mom of dina .  york is the dad of dina . ...   what is york to fega ?

  check york father dina .                                          [ok]
  think york father dina so york parent dina .                      [ok]
  think york parent dina and dina parent fega so york grandparent fega .   [ok]
  ...
  answer great_grandfather .
```

*Named for* **modus ponens** *— the inference rule (B, B→H ⊢ H) that every line of the
trace language literally executes.*

## Why

Frontier models reason in latent space and hallucinate freely. This project explores the
opposite point in design space at tiny scale (~3–25M params): every inference is **local,
externalized, and machine-verifiable**. The research questions: how much of language + reasoning
+ arithmetic can be learned this way, what generalizes (entities, depths, phrasings, rule
compositions, rules-defined-at-inference), and what the failure modes teach about binding,
memorization, and curriculum.

## What's here

| path | what |
|---|---|
| `thinking/` | the production package — worlds, traces, checker, trainer, exams, rule induction, verbalizer ([docs](PACKAGE.md)) |
| `scratchpad_model.py` | the model: small transformer with a **pointer/copy head**, learnable attention temperature, optional recurrence (looped / HRM / TRM / mHC) and Ouro-style learned halting |
| `datalog.py` | minimal Datalog: least-fixpoint closure with provenance, entailment oracle, proof trees, SLD backward chaining |
| `surfaces.json` | frontier-distilled surface bank: 1,300+ validated English patterns across 8 education registers (preschool → scholar), with held-out splits |
| `runpod/` | H100 launchers (tar-over-ssh, timeout-bounded, always-terminate) |
| `thinking/vision.py`, `thinking/image2.py`, `thinking/image_flow.py`, `thinking/image_latent.py` | Image rungs: synthetic visual factors → canonical facts, head-aware FER probes, pixel flow, and semantic latent flow |
| `thinking/audio.py`, `thinking/multimodal.py` | Audio factors and the M-0 multimodal bridge: image+audio prefixes → one canonical extraction trace |
| `*.md` | research plans and theses (FER bet, reasoning design, training data, validated plan) |

## Headline results (staircase-validated, 2026-06)

- **Chain world**: 1.00 verified accuracy at trained depths (held-out entities, zero resamples).
- **Kinship world** (20 interacting rules, natural-language surface): **verified 0.95 / free
  1.00** at k=3 (d=256, contrastive question training) — the checker now *adds* accuracy over
  free decoding instead of taxing it.
- **Language understanding**: trained on 8 education registers, the model scores within
  **5 points** of its trained-phrasing accuracy on *held-out phrasings it never saw*.
- **Depth generalization**: trained only on derivations ≤6 deep, accuracy decays *smoothly* with
  depth (0.70/0.40/0.17 at k=6/10/20) — **no cliff at the training boundary**. The rules
  transfer; the limiter is constant per-line error compounding, which ordinary training reduces.
- **Rule induction**: a Popper-style generate-test learner recovers the kinship rule system and
  all arithmetic concepts (age = death − birth, hypothetical ages, relativity comparisons) from
  raw (facts, question, answer) observations at **0.99 held-out accuracy** — no hand-coded rules
  shown to the learner.

Hard-won lessons (each cost a debugging arc, all encoded in defaults now): unseen-name
embeddings silently destroy structural processing (→ entity anonymization); conclusion-first
formats are answer-only supervision in disguise (→ premises-first everywhere); fixed example
pools memorize (→ rolling refresh from the infinite generator); capacity thresholds are real
(d=256 for multi-rule worlds); weight-shared recurrence loses to distinct layers at tiny scale;
a rejected line must still advance the *context* — skip-don't-die — or the model stalls
regenerating the same line from the same prefix (removing it collapsed verified accuracy
0.95 → 0.10); constraint-masked decoding *loses* to rejection sampling on a gold-trace policy
(forced valid-but-foreign lines push generation off-distribution), but prefix-pruned resampling
— abort a resample at the first token that can't extend to any valid line — is free: identical
acceptance distribution, 4× the retries for the same budget.

## Quickstart

```bash
uv venv && uv pip install torch numpy tokenizers pandas pyarrow
.venv/bin/python -m thinking.cli selftest          # correctness core, <5s, no GPU
.venv/bin/python -m thinking.cli train --world kinship --simple --dim 256 \
    --steps 8000 --out runs/demo
.venv/bin/python -m thinking.cli eval runs/demo --mode verified
.venv/bin/python -m thinking.cli demo runs/demo --k 3
.venv/bin/python -m thinking.vision --train --steps 200 --out runs/vision_object_encoder.pt
.venv/bin/python -m thinking.image2 --steps 40 --seeds 0 --out runs/image2_smoke.json
.venv/bin/python -m thinking.image_flow --train --steps 40 --out runs/image_flow.pt
.venv/bin/python -m thinking.image_latent --train --ae-steps 40 --flow-steps 40 \
    --out runs/image_latent_flow.pt
.venv/bin/python -m thinking.image_latent --train --flow-arch dit --ae-steps 40 \
    --flow-steps 40 --cond-drop 0.1 --cfg-scale 1.5 --sample-steps 4 \
    --out runs/image_latent_dit.pt
.venv/bin/python -m thinking.multimodal --steps 40 --out runs/m0_multimodal.json
```

GPU runs: `runpod/launch_thinking.py` (see the package docs).
