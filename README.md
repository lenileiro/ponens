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
| `thinking/vision_understanding.py`, `thinking/image_data.py`, `thinking/image_caption.py`, `thinking/image_embed.py`, `thinking/image_score.py`, `thinking/image_preferences.py`, `thinking/image_curate.py`, `thinking/image_eval.py`, `thinking/image_latent.py` | Image stack: manifest-driven visual concept learning, captioned image data, recaptioning, embedding/quality/preference sidecars, curation, offline image-quality eval, and text-conditioned latent flow |
| `thinking/text.py` | Text-0 semantic understanding rung: web-imported English records → canonical facts, with artifact controls |
| `thinking/multimodal.py` | Generic manifest-driven multimodal prefix bridge with named feature views, text tokens, targets, latent slots, and concept memory |
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
.venv/bin/python -m thinking.vision_understanding --train \
    --manifest data/images/train_web_ppm_smoke.jsonl --root data/images \
    --steps 40 --out runs/vision_understanding.pt
.venv/bin/python -m thinking.image_latent --train --cond-mode text --flow-arch mmdit \
    --image-manifest data/images/train_web_ppm_smoke.jsonl --image-root data/images \
    --ae-steps 40 --flow-steps 40 --cond-drop 0.1 --cfg-scale 1.5 \
    --sample-steps 4 --time-sampling logit-normal --time-shift-mode auto \
    --flow-consistency-w 0.05 --flow-endpoint-w 0.1 \
    --flow-noise-coupling sliced_ot --flow-noise-coupling-projections 4 \
    --flow-ema-decay 0.99 --flow-checkpoint-blocks \
    --dit-attn-impl auto --dit-mlp swiglu \
    --sample-churn 0.05 --sample-churn-interval 0.0,0.8 \
    --out runs/image_latent_mmdit_text.pt
.venv/bin/python -m thinking.image_fetch --source text-to-image-2m-512-2m \
    --max-records 1024 --image-dir data/images/web_fetch \
    --manifest data/images/train_web.jsonl --root data/images \
    --report-out runs/image_fetch_report.json
# image_fetch accepts WebDataset JSON metadata sidecars and plain .txt caption sidecars.
.venv/bin/python -m thinking.image_caption --manifest data/images/train_web.jsonl \
    --root data/images --backend hf --model Salesforce/blip-image-captioning-large \
    --mode replace --batch 16 --device cuda \
    --out data/images/train_web_captioned.jsonl \
    --report-out runs/image_caption_report.json
.venv/bin/python -m thinking.image_score --manifest data/images/train_web_captioned.jsonl \
    --root data/images --backend stats --image-size 256 \
    --out data/images/train_web_scored.jsonl \
    --sidecar-out data/images/train_web_quality_scores.jsonl \
    --report-out runs/image_score_report.json
.venv/bin/python -m thinking.image_curate --manifest data/images/train_web_scored.jsonl \
    --root data/images --min-caption-tokens 4 --min-width 256 --min-height 256 \
    --min-caption-unique-ratio 0.25 --max-caption-token-frequency 0.5 \
    --max-caption-token-run 0.5 --max-caption-char-run 0.35 \
    --max-nsfw 0.2 --max-watermark 0.2 --min-image-text-cosine 0.2 \
    --max-image-duplicate-cosine 0.985 --eval-frac 0.02 \
    --out data/images/train_web_curated.jsonl \
    --report-out runs/image_curate_report.json
# To use a web/HF preference model, write any JSONL/CSV/TSV sidecar with image + score fields
# and merge it generically instead of hardcoding a reward model into training:
# .venv/bin/python -m thinking.image_score --manifest data/images/train_web_captioned.jsonl \
#     --root data/images --backend ensemble --technical-w 0.3 \
#     --external-sidecar data/images/preference_scores.jsonl \
#     --external-score-field image_reward --external-w 0.7 \
#     --out data/images/train_web_scored.jsonl \
#     --report-out runs/image_score_report.json
# Optional alternate web source: DiffusionDB zip parts. Add --diffusiondb-metadata
# with the official metadata.parquet URL or a local copy to preserve nsfw/size fields.
.venv/bin/python -m thinking.image_fetch --source diffusiondb-2m \
    --max-records 1024 --diffusiondb-start-part 1 --diffusiondb-end-part 1 \
    --image-dir data/images/diffusiondb_fetch \
    --manifest data/images/train_diffusiondb.jsonl --root data/images \
    --report-out runs/image_fetch_diffusiondb_report.json
.venv/bin/python -m thinking.image_embed --manifest data/images/train_web_scored.jsonl \
    --root data/images --backend hf --model google/siglip-base-patch16-224 \
    --features both --text-embed-mode both \
    --text-sequence-model google-t5/t5-base \
    --batch 64 --device cuda \
    --out data/images/embeddings.jsonl \
    --report-out runs/image_embed_report.json
The optional T5 sequence encoder needs `transformers` plus `sentencepiece`; the RunPod preset
installs those when this path is active.
.venv/bin/python -m thinking.image_data --manifest data/images/train_web_scored.jsonl \
    --root data/images --min-side 256 --max-aspect 2.0 \
    --max-nsfw 0.2 --max-watermark 0.5 --min-image-text-cosine 0.15 \
    --max-image-duplicate-cosine 0.985 \
    --embedding-manifest data/images/embeddings.jsonl --embedding-key image \
    --min-caption-tokens 3 --min-caption-unique-ratio 0.25 \
    --max-caption-token-frequency 0.5 --max-caption-token-run 0.5 \
    --max-caption-char-run 0.35 --write-filtered data/images/train_clean.jsonl \
    --report-out runs/image_manifest_report.json
.venv/bin/python -m thinking.image_latent --train --cond-mode text --flow-arch mmdit \
    --image-manifest data/images/train_clean.jsonl --image-root data/images \
    --caption-cond-source auto \
    --size 64 --ae-arch residual --latent-downsample 8 --latent-max-tokens 128 \
    --dit-pos-embed rope2d --dit-attn-impl auto --dit-mlp swiglu \
    --latent-patch-size 2 \
    --ae-recon-loss hybrid --ae-grad-w 0.1 --ae-ms-w 0.1 --ae-fft-w 0.05 \
    --image-text-align-w 0.1 --flow-text-align-w 0.05 --text-embed-dim 128 \
    --image-feature-align-w 0.1 --flow-feature-align-w 0.05 \
    --image-feature-embed-dim 128 \
    --flow-repa-w 0.05 --flow-repa-mode auto \
    --flow-self-repa-w 0.05 --flow-self-repa-mode auto \
    --flow-sra-w 0.05 --flow-sra-mode both --flow-sra-time-gap 0.25 \
    --ae-accum-steps 2 --flow-accum-steps 2 --grad-clip 1.0 \
    --flow-cache-latents --flow-cache-dir runs/image_manifest_cache \
    --flow-cache-shard-size 2048 --flow-cache-batch 32 --flow-cache-dtype bf16 \
    --flow-cache-max-loaded-shards 4 \
    --ae-steps 40 --flow-steps 40 --sample-grid-out runs/image_manifest_grid.ppm \
    --flow-consistency-w 0.05 --flow-endpoint-w 0.1 \
    --flow-distill-steps 20 --flow-guidance-distill-w 0.1 \
    --flow-guidance-distill-cfg-scale 1.5 \
    --flow-noise-coupling sliced_ot --flow-noise-coupling-projections 4 \
    --out runs/image_manifest_mmdit.pt
.venv/bin/python -m thinking.image_latent --eval-checkpoint runs/image_manifest_mmdit.pt \
    --eval-image-manifest data/images/train_clean.jsonl --eval-image-root data/images \
    --eval-image-split eval --size 64 --cfg-scales 1.0,1.5 --sample-steps-list 4,8 \
    --cfg-modes standard,cfgpp --sample-churns 0.0,0.05 --eval-seeds 1,2,3 \
    --eval-generated-samples 16 --eval-generated-candidates-per-prompt 2 \
    --sample-grid-out runs/image_manifest_eval_grid.ppm \
    --sample-manifest-out data/images/generated_captioned.jsonl \
    --eval-out runs/image_manifest_mmdit_sweep.json
.venv/bin/python -m thinking.image_embed --manifest data/images/generated_captioned.jsonl \
    --root data/images --backend hf --model google/siglip-base-patch16-224 \
    --features both --text-embed-mode pooled --batch 64 --device cuda \
    --out data/images/generated_embeddings.jsonl \
    --report-out runs/generated_image_embed_report.json
# Score distribution drift, support coverage, diversity, and image-text alignment offline.
.venv/bin/python -m thinking.image_eval \
    --real-manifest data/images/train_clean.jsonl \
    --generated-manifest data/images/generated_captioned.jsonl \
    --generated-embedding-sidecar data/images/generated_embeddings.jsonl \
    --embedding-key image --max-records 2048 --min-score 0.25 \
    --report-out runs/image_eval_report.json
# Preference-loop artifact: score multiple generated candidates per prompt, then emit
# chosen/rejected pairs for direct flow preference tuning and quality-scorer training.
.venv/bin/python -m thinking.image_score --manifest data/images/generated_captioned.jsonl \
    --root data/images --backend ensemble --technical-w 0.3 \
    --external-sidecar data/images/generated_reward_scores.jsonl \
    --external-score-field reward_score --external-w 0.7 \
    --out data/images/generated_scored.jsonl \
    --report-out runs/generated_score_report.json
.venv/bin/python -m thinking.image_preferences --manifest data/images/generated_scored.jsonl \
    --root data/images --group-by prompt_id,prompt,caption --mode top-bottom \
    --min-score-gap 0.05 --out data/images/generated_preferences.jsonl \
    --report-out runs/generated_preferences_report.json
RUNPOD_API_KEY=... .venv/bin/python runpod/launch_thinking.py \
    --image-quality-preset web-hf-vae
RUNPOD_API_KEY=... .venv/bin/python runpod/launch_thinking.py \
    --image-quality-preset web-hf-vae-hq
# The preset also writes a generated sample manifest, embeds it, and runs image_eval; add
# --image-generated-eval-fail-on-gate plus threshold flags to make quality gates hard. It also
# runs image_score before embedding/cleaning so quality_score metadata reaches sampling,
# duplicate selection, quality-head training, quality-guided prompt sampling, and direct
# chosen/rejected latent-flow preference updates. The `web-hf-vae` preset is the broad
# 512/768 data path; `web-hf-vae-hq` switches to a mixed 1024+512 source manifest with
# high-res source upweighting, PickScore + technical-health quality scoring,
# stricter cleaning, longer text sequence conditioning, 1024/multi-aspect buckets, latent patching
# that keeps MM-DiT token count bounded, a 12-block/12-head 768-wide MM-DiT, more sampling steps,
# Karras-style timestep placement, sampler-diverse candidate reranking, and a
# generated-candidate feedback artifact:
# `*_candidates.jsonl` -> PickScore-scored candidates -> `*_preferences.jsonl` chosen/rejected
# pairs, including sampler/CFG provenance, for the next quality-scorer and direct-flow preference pass. If that portable preference
# artifact and its candidate images exist locally, the next `web-hf-vae-hq` launch auto-uploads
# and consumes it. Prompt candidates now cycle through the configured CFG/sampler/seed
# sweep lists and record per-candidate provenance for preference mining; add
# `--image-no-auto-preferences` to force a first-run/no-feedback profile. Both use SigLIP So400m
# pooled prompts plus T5-large token-sequence conditioning, crop/flip/pad-aware
# geometry conditioning for multi-aspect targets, Fourier flow-time conditioning, decoded-pixel
# dynamic thresholding for high-CFG samples, progressive bucket curriculum, CFG dropout,
# boundary-enforced double-cosine rectified-flow velocity, logit-normal flow times with soft
# Min-SNR velocity weighting, adaptive loss-tracked timestep sampling, EMA-teacher guided
# self-distillation, triangular middle-window CFG scheduling, Karras/cosine/linear
# timestep sweeps, Heun/adaptive-Heun/RK4 sampling sweeps, and standard CFG plus CFG++ sweeps.
# MM-DiT attention defaults to exact auto SDPA on modern PyTorch; use linear only as an
# explicit memory/speed approximation.
.venv/bin/python -m thinking.image_latent --eval-checkpoint runs/image_latent_dit.pt \
    --cfg-scales 1.0,1.5 --sample-steps-list 4,8 --eval-seeds 1,2,3 \
    --eval-out runs/image_latent_dit_sweep.json
.venv/bin/python -m thinking.text --import-snli --snli-zip /private/tmp/snli_1.0.zip \
    --snli-train 20000 --snli-eval 1000 --seed 11 --out data/text_snli.jsonl
.venv/bin/python -m thinking.text --import-mnli --mnli-zip /private/tmp/multinli_1.0.zip \
    --mnli-train 20000 --mnli-eval 2000 --seed 19 --out data/text_mnli.jsonl
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
.venv/bin/python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --steps 1200 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 100 --paraphrase-n 30 --counterfactual-n 30 --max-new 32 \
    --out runs/text_snli_hans_extended.json --checkpoint runs/text_snli_hans_extended.pt
.venv/bin/python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --steps 1200 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 --balance-by kind \
    --free-n 80 --kind-free-n 5 --paraphrase-n 20 --counterfactual-n 20 --max-new 32 \
    --out runs/text_snli_hans_balanced.json \
    --checkpoint runs/text_snli_hans_balanced.pt
.venv/bin/python -m thinking.text --data data/text_snli.jsonl --data data/text_mnli.jsonl \
    --data data/text_hans.jsonl \
    --steps 120 --batch 48 --d 96 --layers 3 --heads 4 --semantic-w 0.75 \
    --balance-by kind --fact-n 240 --kind-fact-n 40 --artifact-n 240 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_snli_mnli_hans_smoke.json \
    --checkpoint runs/text_snli_mnli_hans_smoke.pt
.venv/bin/python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_smoke.pt \
    --steps 40 --batch 32 --study-lr 0.0005 --semantic-w 0.75 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_smoke.json
.venv/bin/python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_semantic_replay_smoke.pt \
    --steps 80 --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_semantic_replay_smoke.json
.venv/bin/python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_error_replay_smoke.pt \
    --steps 40 --study-rounds 2 --study-strategy errors \
    --study-probe-n 512 --study-hard-max 256 \
    --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_error_replay_smoke.json
.venv/bin/python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_select_both_replay_smoke.pt \
    --steps 40 --study-rounds 2 --study-strategy errors \
    --study-probe-n 512 --study-hard-max 256 --study-select-best \
    --study-score-metric both --study-retention-w 2.0 \
    --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_select_both_replay_smoke.json
.venv/bin/python -m thinking.text --reading-data NEWER-TECHNIQUES.md \
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
.venv/bin/python -m thinking.multimodal --manifest data/multimodal_manifest.jsonl \
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

GPU runs: `runpod/launch_thinking.py` (see the package docs).
