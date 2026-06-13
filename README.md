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
| `thinking/vision.py`, `thinking/image2.py`, `thinking/image_flow.py`, `thinking/image_data.py`, `thinking/image_embed.py`, `thinking/image_latent.py` | Image rungs: synthetic visual factors → canonical facts, head-aware FER probes, captioned image data, embedding sidecars, pixel flow, and semantic latent flow |
| `thinking/text.py` | Text-0 semantic understanding rung: web-imported English records → canonical facts, with artifact controls |
| `thinking/audio.py`, `thinking/multimodal.py` | Audio factors and the M-0 multimodal bridge: image+audio+transcript prefixes → one canonical extraction trace |
| `thinking/listen.py`, `thinking/speak.py` | speech: **listen** (transcribe real synthesized speech, speaker-invariant), **speak** (emit audio tokens *verified by round-trip* through the frozen listener — the checker, applied to generation) |
| `thinking/crossmodal.py` | cross-modal FER probe: does a concept *heard* align with the same concept *seen*? (retrieval 0.92 — unified, not fractured) |
| `data/multimodal_transcripts.json` | Configurable M-0 language data: template smoke bank plus optional explicit `(text, facts)` examples |
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
    --flow-semantic-w 0.25 \
    --out runs/image_latent_dit.pt
.venv/bin/python -m thinking.image_latent --train --cond-mode text --flow-arch mmdit \
    --ae-steps 40 --flow-steps 40 --cond-drop 0.1 --cfg-scale 1.5 \
    --sample-steps 4 --flow-semantic-w 0.25 --time-sampling logit-normal \
    --flow-consistency-w 0.05 --flow-ema-decay 0.99 \
    --out runs/image_latent_mmdit_text.pt
.venv/bin/python -m thinking.image_embed --manifest data/images/train.jsonl \
    --root data/images --backend hf --model google/siglip-base-patch16-224 \
    --features both --batch 64 --device cuda --out data/images/embeddings.jsonl \
    --report-out runs/image_embed_report.json
.venv/bin/python -m thinking.image_data --manifest data/images/train.jsonl \
    --root data/images --min-side 256 --max-aspect 2.0 \
    --embedding-manifest data/images/embeddings.jsonl --embedding-key image \
    --min-caption-tokens 3 --write-filtered data/images/train_clean.jsonl \
    --report-out runs/image_manifest_report.json
.venv/bin/python -m thinking.image_latent --train --cond-mode text --flow-arch mmdit \
    --image-manifest data/images/train_clean.jsonl --image-root data/images \
    --caption-cond-source auto \
    --size 64 --ae-arch residual --latent-downsample 8 --latent-max-tokens 128 \
    --ae-recon-loss hybrid --ae-grad-w 0.1 --ae-ms-w 0.1 \
    --image-text-align-w 0.1 --flow-text-align-w 0.05 --text-embed-dim 128 \
    --image-feature-align-w 0.1 --flow-feature-align-w 0.05 \
    --image-feature-embed-dim 128 \
    --ae-accum-steps 2 --flow-accum-steps 2 --grad-clip 1.0 \
    --flow-cache-latents --flow-cache-dir runs/image_manifest_cache \
    --flow-cache-shard-size 2048 --flow-cache-batch 32 \
    --ae-steps 40 --flow-steps 40 --sample-grid-out runs/image_manifest_grid.ppm \
    --flow-consistency-w 0.05 --out runs/image_manifest_mmdit.pt
.venv/bin/python -m thinking.image_latent --eval-checkpoint runs/image_manifest_mmdit.pt \
    --eval-image-manifest data/images/train_clean.jsonl --eval-image-root data/images \
    --eval-image-split eval --size 64 --cfg-scales 1.0,1.5 --sample-steps-list 4,8 \
    --eval-seeds 1,2,3 --sample-grid-out runs/image_manifest_eval_grid.ppm \
    --eval-out runs/image_manifest_mmdit_sweep.json
.venv/bin/python -m thinking.image_latent --eval-checkpoint runs/image_latent_dit.pt \
    --cfg-scales 1.0,1.5 --sample-steps-list 4,8 --eval-seeds 1,2,3 \
    --eval-out runs/image_latent_dit_sweep.json
.venv/bin/python -m thinking.text --import-snli --snli-zip /private/tmp/snli_1.0.zip \
    --snli-train 20000 --snli-eval 1000 --seed 11 --out data/text_snli.jsonl
.venv/bin/python -m thinking.text --import-mnli --mnli-zip /private/tmp/multinli_1.0.zip \
    --mnli-train 20000 --mnli-eval 2000 --seed 19 --out data/text_mnli.jsonl
.venv/bin/python -m thinking.text --import-squad --squad-train 5000 --squad-eval 1000 \
    --squad-max-context-tokens 160 --squad-max-question-tokens 40 \
    --squad-max-answer-tokens 8 --seed 23 --out data/text_squad.jsonl
.venv/bin/python -m thinking.text --import-squad --squad-train 800 --squad-eval 200 \
    --squad-max-context-tokens 96 --squad-max-question-tokens 32 \
    --squad-max-answer-tokens 5 --seed 29 --out data/text_squad_smoke.jsonl
.venv/bin/python -m thinking.text --import-squad --squad-choice-only --squad-choice-n 4 \
    --squad-train 1200 --squad-eval 300 --squad-max-context-tokens 128 \
    --squad-max-question-tokens 32 --squad-max-answer-tokens 6 --seed 31 \
    --out data/text_squad_choice_smoke.jsonl
.venv/bin/python -m thinking.text --import-squad --squad-choice-only --squad-choice-n 4 \
    --squad-choice-swap-negatives 1 --squad-train 1800 --squad-eval 450 \
    --squad-max-context-tokens 128 --squad-max-question-tokens 32 \
    --squad-max-answer-tokens 6 --seed 37 \
    --out data/text_squad_choice_neg_smoke.jsonl
.venv/bin/python -m thinking.text --import-squad --squad-choice-only --squad-choice-n 4 \
    --squad-choice-swap-negatives 1 --squad-choice-absent-negatives 1 \
    --squad-train 2400 --squad-eval 600 --squad-max-context-tokens 128 \
    --squad-max-question-tokens 32 --squad-max-answer-tokens 6 --seed 41 \
    --out data/text_squad_choice_absent_neg_smoke.jsonl
.venv/bin/python -m thinking.text --data data/text_snli.jsonl --steps 1500 \
    --batch 64 --d 192 --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 200 --max-new 20 \
    --out runs/text_snli.json --checkpoint runs/text_snli.pt
.venv/bin/python -m thinking.text --import-hans --hans-train 6000 --hans-eval 3000 \
    --seed 13 --out data/text_hans.jsonl
.venv/bin/python -m thinking.text --data data/text_hans.jsonl \
    --eval-checkpoint runs/text_snli.pt --free-n 200 --max-new 20 \
    --out runs/text_snli_on_hans.json
.venv/bin/python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --steps 1500 --batch 64 --d 192 --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 200 --max-new 20 \
    --out runs/text_snli_hans.json --checkpoint runs/text_snli_hans.pt
.venv/bin/python -m thinking.text --import-grounded --grounded-train 8000 \
    --grounded-eval 1500 --grounded-counterfactual 100 --seed 17 \
    --out data/text_grounded.jsonl
.venv/bin/python -m thinking.text --data data/text_grounded.jsonl --steps 600 \
    --batch 64 --d 192 --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 100 --paraphrase-n 50 --counterfactual-n 50 --max-new 32 \
    --out runs/text_grounded.json --checkpoint runs/text_grounded.pt
.venv/bin/python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --data data/text_grounded.jsonl --steps 1200 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 100 --paraphrase-n 30 --counterfactual-n 30 --max-new 32 \
    --out runs/text_snli_hans_grounded.json --checkpoint runs/text_snli_hans_grounded.pt
.venv/bin/python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --data data/text_grounded.jsonl --steps 1200 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 --balance-by kind \
    --free-n 80 --kind-free-n 5 --paraphrase-n 20 --counterfactual-n 20 --max-new 32 \
    --out runs/text_snli_hans_grounded_balanced.json \
    --checkpoint runs/text_snli_hans_grounded_balanced.pt
.venv/bin/python -m thinking.text --data data/text_snli.jsonl --data data/text_mnli.jsonl \
    --data data/text_hans.jsonl --data data/text_grounded.jsonl \
    --steps 120 --batch 48 --d 96 --layers 3 --heads 4 --semantic-w 0.75 \
    --balance-by kind --fact-n 240 --kind-fact-n 40 --artifact-n 240 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_snli_mnli_hans_grounded_smoke.json \
    --checkpoint runs/text_snli_mnli_hans_grounded_smoke.pt
.venv/bin/python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_smoke.pt \
    --steps 40 --batch 32 --study-lr 0.0005 --semantic-w 0.75 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_smoke.json
.venv/bin/python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_semantic_replay_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 80 --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_semantic_replay_smoke.json
.venv/bin/python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_error_replay_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 40 --study-rounds 2 --study-strategy errors \
    --study-probe-n 512 --study-hard-max 256 \
    --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_error_replay_smoke.json
.venv/bin/python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_select_both_replay_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 40 --study-rounds 2 --study-strategy errors \
    --study-probe-n 512 --study-hard-max 256 --study-select-best \
    --study-score-metric both --study-retention-w 2.0 \
    --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_select_both_replay_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_control_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 12 --study-rounds 1 --study-strategy errors \
    --study-probe-n 128 --study-hard-max 96 --study-select-best \
    --study-score-metric both --study-retention-w 2.0 --study-control-w 2.0 \
    --batch 24 --study-lr 0.0005 --decode-w 0.25 --semantic-w 1.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 10 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 36 \
    --out runs/text_study_squad_control_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_choice_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_all_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 80 --study-rounds 1 --study-strategy all --study-select-best \
    --study-score-metric both --study-retention-w 2.0 --study-control-w 2.0 \
    --batch 32 --study-lr 0.0005 --decode-w 0.25 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_all_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_choice_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_neg_guard_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 100 --study-rounds 1 --study-strategy all --study-select-best \
    --study-score-metric both --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0.25 \
    --semantic-w 1.0 --balance-by kind --fact-n 160 --kind-fact-n 30 \
    --artifact-n 160 --free-n -1 --paraphrase-n -1 --counterfactual-n -1 \
    --max-new 24 --out runs/text_study_squad_choice_neg_guard_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_choice_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_seeded_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 100 --study-rounds 1 --study-strategy all --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0.25 \
    --semantic-w 0.5 --choice-w 1.0 --balance-by kind --fact-n 160 \
    --kind-fact-n 30 --artifact-n 160 --free-n -1 --paraphrase-n -1 \
    --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_seeded_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_absent_answerw4_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 100 --study-rounds 1 --study-strategy all --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 4.0 --choice-none-w 1.0 \
    --balance-by kind --fact-n 160 --kind-fact-n 30 --artifact-n 160 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_absent_answerw4_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_evidence_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy all --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 20 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_evidence_2round_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_errors_kindguard_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy errors --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 20 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_errors_kindguard_2round_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_candidate_ctx_pair_controlguard_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy errors --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --choice-pair-w 1.0 --choice-pair-margin 0.0 --balance-by kind \
    --fact-n 80 --kind-fact-n 20 --artifact-n 80 --free-n -1 \
    --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_candidate_ctx_pair_controlguard_2round_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_candidate_ctx_pair_ctrlcontrast_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy errors --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --choice-pair-w 1.0 --choice-pair-margin 0.0 --choice-control-w 0 \
    --choice-control-contrast-w 1.0 --choice-control-margin 0.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 20 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_candidate_ctx_pair_ctrlcontrast_2round_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_contextloc025_pair_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy errors --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --choice-context-w 0.25 --choice-pair-w 1.0 --choice-pair-margin 0.0 \
    --choice-control-w 0 --choice-control-contrast-w 0 --choice-control-margin 0.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 20 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_contextloc025_pair_2round_smoke.json
.venv/bin/python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_answerability_swapctrl_qctx_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy errors --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --choice-answerability-w 0.5 --choice-answerability-control-w 0.5 \
    --choice-context-w 0.25 --choice-question-context-w 0.25 \
    --choice-question-context-contrast-w 0.10 --choice-question-context-margin 0.0 \
    --choice-pair-w 1.0 --choice-pair-margin 0.0 \
    --choice-control-w 0 --choice-control-contrast-w 0 --choice-control-margin 0.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 20 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_answerability_swapctrl_qctx_2round_smoke.json
.venv/bin/python -m thinking.multimodal --steps 240 --eval-n 120 --free-n 20 \
    --counterfactual-n 40 --free-counterfactual-n 20 \
    --out runs/m0_multimodal.json --checkpoint runs/m0_multimodal.pt
```

GPU runs: `runpod/launch_thinking.py` (see the package docs).
