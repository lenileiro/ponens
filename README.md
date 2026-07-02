# Ponens — manifest-driven concept learning

An in-house research stack for small models that learn from raw text, images, audio, and named
feature views without a synthetic English bank. The current package centers on data-backed
reading, latent concept discovery, memory-gap training, image generation/evaluation, and a generic
multimodal bridge.

> **Verified-reasoning + typed-language stack (2026-06-20):** a sound proof kernel (Lean's design) +
> set-theoretic category types (Elixir's design) + the LOTA agent language + neuro-symbolic
> proof search, wired end-to-end as *English question → kernel-verified answer*. See
> **[`thinking/STACK.md`](thinking/STACK.md)** for the full map, **[`thinking/VERIFIED_REASONING.md`](thinking/VERIFIED_REASONING.md)**
> for results, and **[`thinking/C2_ROADMAP.md`](thinking/C2_ROADMAP.md)** for the honest C2 gap
> (open-vocab needs a pretrained backbone — proven empirically).

## Why

The current bet is that language mastery should come from reading, latent structure, replay,
reanalysis, self-generated memory gaps, graph-closure insight, partial-context closure surprise,
and cross-modal concept pressure, not from hand-authored task harnesses or templated phrasing
rules. Raw-reading checkpoint continuation reuses the model's own replay bank under the mastery
profile, so new reading can update weights without discarding earlier concepts. Reading and
multimodal checkpoints carry label-free representation-progress signals, letting later self-teach
runs reuse internal FER/bridge/sequence weaknesses rather than only surface scores. Multimodal text
transfer is target-probed before trust: a harmful reading checkpoint can be rolled back before its
history prior drives new training. The default reading profile also owns bounded hard-study
probing, selected hard-record caps, periodic structure refreshes, bounded raw-reading vocabulary,
and source-balanced train/replay sampling, so normal commands do not need low-level study flags,
do not let the longest source dominate, and do not grow an embedding table for every one-off corpus
token. Multimodal training uses the same generic source-balanced default from manifest `source` /
`document` / `dataset` metadata, exposed as `--source-balance-w` locally and
`--multimodal-source-balance-w` on RunPod. Reading checkpoints now expose compact
`learning_event` evidence when an applied update
moved sampled weights and improved score, representation structure, or
concept-connection signals; continuation and multimodal transfer can use that event as self-teach
pressure, and replay banks prioritize the study records attached to those events. Selected study
rounds can also keep a guarded representation reorganization as a model-discovered "light bulb"
moment, instead of depending only on surface score. Checkpoint continuation now folds those
priority replay rows back into the primary self-supervised study pool while still scoring selection
on the new reading task, so the model rereads material tied to real weight movement without
turning replay into labels. Multimodal text-checkpoint transfer also exposes that priority-study
evidence and uses it when converting text learning events into self-teach pressure. Multimodal
training now writes its own generic `learning_event` when applied updates move sampled weights and
improve score, bridge insight, or internal representation organization, and checkpoints preserve a
bounded `multimodal_learning_history` so continuation can reuse a trajectory of discovered
weaknesses rather than only the latest report. The code keeps model inputs manifest-driven:
datasets provide text, feature views, image records, and optional target tokens; model code learns
the representations.

## What's here

| path | what |
|---|---|
| `thinking/` | supported training modules for text, image, audio, and multimodal concept learning ([docs](PACKAGE.md)) |
| `scratchpad_model.py` | the model: small transformer with a **pointer/copy head**, learnable attention temperature, optional recurrence (looped / HRM / TRM / mHC) and Ouro-style learned halting |
| `runpod/` | H100 launchers (tar-over-ssh, timeout-bounded, always-terminate) |
| `thinking/vision_understanding.py`, `thinking/image_data.py`, `thinking/image_caption.py`, `thinking/image_embed.py`, `thinking/image_score.py`, `thinking/image_preferences.py`, `thinking/image_curate.py`, `thinking/image_eval.py`, `thinking/vision_read.py`, `thinking/image_quality_loop.py`, `thinking/image_latent.py` | Image stack: manifest-driven visual concept learning, captioned image data, recaptioning, embedding/quality/preference sidecars, curation, offline image-quality eval, generic vision-read validation, closed-loop generated-image scoring, and text-conditioned latent flow |
| `thinking/text.py` | raw-reading and semantic text learning with latent concept memory, replay, discovery, context closure, reanalysis, graph-closure insight, and memory-gap training |
| `thinking/multimodal.py` | Generic manifest-driven multimodal prefix bridge with named feature views, text tokens, targets, latent slots, concept memory, graph-closure insight, and memory-gap training |
| `thinking/span_pfn_text.py`, `thinking/squadqa.py`, `thinking/squad_reader.py`, `thinking/multitask.py` | Extractive-QA + **in-context span PFN** research track: WordNet-grounded runtime QA, a from-scratch neural reader, one shipped base weight, and TabPFN-style in-context span extraction ([docs](EXTRACTIVE-QA-PFN.md)) |
| `*.md` | research notes and plans; historical synthetic-language docs are no longer package APIs |

## Quickstart

```bash
uv venv && uv pip install torch numpy tokenizers pandas pyarrow
.venv/bin/python -m thinking.text --selftest
.venv/bin/python -m thinking.multimodal --selftest
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
    --features both --text-embed-mode both --image-embed-mode both \
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
    --dit-register-tokens 1 \
    --latent-patch-size 2 \
    --ae-recon-loss hybrid --ae-grad-w 0.1 --ae-ms-w 0.1 --ae-fft-w 0.05 \
    --image-text-align-w 0.1 --flow-text-align-w 0.05 --text-embed-dim 128 \
    --image-feature-align-w 0.1 --flow-feature-align-w 0.05 \
    --image-feature-embed-dim 128 --image-embedding-sequence-max-len 256 \
    --flow-repa-w 0.05 --flow-repa-mode auto \
    --flow-self-repa-w 0.05 --flow-self-repa-mode auto \
    --flow-sra-w 0.05 --flow-sra-mode both --flow-sra-time-gap 0.25 \
    --flow-self-condition --flow-self-condition-p 0.5 \
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
# `--sample-grid-out` writes real PNG/JPEG/WebP files when the path uses those extensions;
# `.ppm` remains the dependency-light raw grid format.
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
# Closed-loop validation: embed real/generated images, run image_eval, build a generic
# vision-read manifest, and optionally train the reader to recover captions from vision.
.venv/bin/python -m thinking.image_quality_loop \
    --real-manifest data/images/train_clean.jsonl --real-root data/images --real-split train \
    --generated-manifest data/images/generated_captioned.jsonl --generated-root data/images \
    --generated-split generated \
    --real-embedded-manifest data/images/embeddings.jsonl \
    --generated-embedded-manifest data/images/generated_embeddings.jsonl \
    --embedding-backend hf --embedding-model google/siglip-base-patch16-224 \
    --embedding-device cuda --embedding-batch 64 --eval-max-records 2048 \
    --min-score 0.25 --vision-read-steps 400 --vision-read-device cuda \
    --vision-read-dim 128 --vision-read-layers 2 --vision-read-heads 4 \
    --vision-read-min-sensor-token-acc 0.05 \
    --work-dir runs/image_quality_loop \
    --report-out runs/image_quality_loop_report.json
# Preference-loop artifact: score multiple generated candidates per prompt, then emit
# chosen/rejected pairs for direct flow preference tuning and quality-scorer training.
# The latent-flow trainer accepts `--flow-preference-loss gap` to use score_gap
# as a pair-specific target margin without a reference copy, or `dpo` to anchor
# pairwise updates against a frozen reference flow.
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
# The preset also writes a generated sample manifest, embeds it, runs image_eval, and runs
# image_quality_loop for generic vision-read caption recovery; add --image-generated-eval-fail-on-gate
# or --image-quality-loop-fail-on-gate plus threshold flags to make quality gates hard. It also
# enables autoencoder reconstruction and reference-reproduction gates by default so the run
# fails if the latent codec collapses or cannot reconstruct shown images before free prompt
# sampling is judged.
# runs image_score before embedding/cleaning so quality_score metadata reaches sampling,
# duplicate selection, quality-head training, quality-guided prompt sampling, and direct
# chosen/rejected latent-flow preference updates. The `web-hf-vae` preset is the broad
# 512/768 data path; `web-hf-vae-hq` switches to a mixed 1024+512 source manifest with
# high-res source upweighting, PickScore + technical-health quality scoring,
# stricter cleaning, longer text sequence conditioning, 1024/multi-aspect buckets, latent patching
# that keeps MM-DiT token count bounded, a 12-block/12-head 768-wide MM-DiT
# with a REG-style image-stream register token, more sampling steps,
# Karras-style timestep placement, sampler-diverse candidate reranking, and a
# generated-candidate feedback artifact:
# `*_candidates.jsonl` -> PickScore-scored candidates -> `*_preferences.jsonl` chosen/rejected
# pairs, including sampler/CFG provenance, for the next quality-scorer and direct-flow preference pass. If that portable preference
# artifact and its candidate images exist locally, the next `web-hf-vae-hq` launch auto-uploads
# and consumes it with score-gap direct flow preference tuning. Prompt candidates
# now cycle through the configured CFG/sampler/seed
# sweep lists and record per-candidate provenance for preference mining; add
# `--image-no-auto-preferences` to force a first-run/no-feedback profile. Both use SigLIP So400m
# pooled prompts plus T5-large token-sequence conditioning, crop/flip/pad-aware
# geometry conditioning for multi-aspect targets, Fourier flow-time conditioning, decoded-pixel
# dynamic thresholding for high-CFG samples, progressive bucket curriculum, CFG dropout,
# boundary-enforced double-cosine rectified-flow velocity, endpoint self-conditioning,
# frequency-domain endpoint
# detail loss, logit-normal flow times with soft Min-SNR velocity weighting,
# SD3-style mode-biased timestep priors inside adaptive loss-tracked timestep sampling,
# early-stopped REPA/SRA representation alignment, EMA-teacher guided self-distillation,
# triangular middle-window CFG scheduling,
# Karras/cosine/linear timestep sweeps,
# adaLN-Zero residual-gated DiT/CrossDiT/MM-DiT blocks,
# Heun/adaptive-Heun/RK4 sampling sweeps, and standard CFG plus CFG++ sweeps.
# MM-DiT attention defaults to exact auto SDPA on modern PyTorch; use linear only as an
# explicit memory/speed approximation.
.venv/bin/python -m thinking.image_latent --eval-checkpoint runs/image_latent_dit.pt \
    --cfg-scales 1.0,1.5 --sample-steps-list 4,8 --eval-seeds 1,2,3 \
    --eval-out runs/image_latent_dit_sweep.json
.venv/bin/python -m thinking.text --reading-data NEWER-TECHNIQUES.md \
    --steps 4 --batch 16 --d 96 --layers 2 --heads 4 \
    --latent-concept-slots 6 --latent-concept-topk 3 --latent-concept-layers 1 \
    --reading-max-tokens 96 --reading-min-tokens 8 --reading-eval-n 32 \
    --reading-memory-size 64 --reading-memory-w 0.05 \
    --reading-association-w 0.05 --reading-association-transitive-steps 3 \
    --reading-association-transitive-w 0.25 \
    --reading-composition-w 0.1 --reading-composition-transitive-steps 3 \
    --reading-composition-transitive-w 0.25 \
    --reading-graph-predict-w 0.2 --reading-graph-predict-transitive-steps 3 \
    --reading-graph-predict-transitive-w 0.25 \
    --reading-discovery-w 0.05 \
    --reading-gap-w 0.05 --reading-gap-transitive-steps 3 \
    --reading-gap-transitive-w 0.25 \
    --reading-context-target-w 0.1 --reading-span-completion-w 0.05 \
    --reading-neighborhood-w 0.05 \
    --reading-neighborhood-batch 8 \
    --reading-transition-w 0.05 --reading-transition-batch 8 \
    --reading-cluster-w 0.05 --reading-cluster-batch 16 \
    --out runs/text_raw_reading_discovery_study_smoke.json \
    --checkpoint runs/text_raw_reading_discovery_study_smoke.pt
.venv/bin/python -m thinking.text --reading-data README.md \
    --reading-checkpoint runs/text_raw_reading_discovery_study_smoke.pt \
    --reading-out-checkpoint runs/text_raw_reading_discovery_study_continued.pt \
    --steps 4 --batch 16 \
    --reading-memory-size 64 --reading-memory-w 0.05 \
    --reading-association-w 0.05 --reading-gap-w 0.05 \
    --out runs/text_raw_reading_discovery_study_continued.json
.venv/bin/python -m thinking.multimodal --manifest data/multimodal_manifest.jsonl \
    --steps 3 --batch 4 --dim 96 --layers 1 --heads 4 --eval-n 20 \
    --source-balance-w 0.5 \
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
    --latent-concept-gap-w 0.05 \
    --latent-concept-gap-transitive-steps 3 \
    --latent-concept-gap-transitive-w 0.25 \
    --latent-concept-neighborhood-w 0.05 \
    --latent-concept-transition-w 0.05 --latent-concept-cluster-w 0.05 \
    --text-checkpoint runs/text_raw_reading_discovery_study_smoke.pt \
    --out runs/m0_multimodal_discovery_study_smoke.json \
    --checkpoint runs/m0_multimodal_discovery_study_smoke.pt

.venv/bin/python -m thinking.multimodal --manifest data/multimodal_manifest.jsonl \
    --multimodal-checkpoint runs/m0_multimodal_discovery_study_smoke.pt \
    --steps 3 --batch 4 --dim 96 --layers 1 --heads 4 --eval-n 20 \
    --latent-concept-slots 6 --latent-concept-memory-size 64 \
    --self-teach-w 0.05 --self-teach-history-prior-w 1.0 \
    --representation-probe-n 20 \
    --out runs/m0_multimodal_discovery_study_continued.json \
    --checkpoint runs/m0_multimodal_discovery_study_continued.pt
```

GPU runs: `runpod/launch_thinking.py` (see the package docs).
