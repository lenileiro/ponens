#!/usr/bin/env python3
"""RunPod H100 runner for the thinking package (Datalog thinking flow, production pipeline).

Default payload: KINSHIP multi-seed (train + negatives + iid/holdout evals + demo) with the looped
default model (mHC + pointer + learned halting). --sweep additionally runs the chain-world
comparison grid (sup x arch x seed). tar-over-ssh; pod-side `timeout` bounds the run; ALWAYS
terminates (try/finally). DEFAULTS to --dry-run.

Auth: export RUNPOD_API_KEY.  Example: RUNPOD_API_KEY=... python runpod/launch_thinking.py --go
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from shlex import quote as shlex_quote

REST = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
REMOTE = "/workspace/fer_relational"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def api(method, path, key, body=None):
    url = REST + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}


def sh(cmd):
    print("  $", cmd)
    return subprocess.run(cmd, shell=True).returncode


def parse_nonnegative_float_csv(raw, name):
    vals = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = float(part)
        except ValueError:
            raise ValueError(f"{name} must be comma-separated numbers")
        if val < 0.0:
            raise ValueError(f"{name} must be non-negative")
        vals.append(val)
    return vals or [0.0]


def parse_unit_interval(raw, name):
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted start,end")
    try:
        start, end = (float(parts[0]), float(parts[1]))
    except ValueError:
        raise ValueError(f"{name} must contain numbers")
    if start < 0.0 or end > 1.0 or start > end:
        raise ValueError(f"{name} must satisfy 0 <= start <= end <= 1")
    return start, end


def parse_source_weight_csv(raw, name):
    text = str(raw or "").strip()
    if not text:
        return {}
    vals = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            raise ValueError(f"{name} entries must be source=weight or source:weight")
        key = key.strip()
        if not key:
            raise ValueError(f"{name} contains an empty source key")
        try:
            val = float(value)
        except ValueError:
            raise ValueError(f"{name} weights must be numbers")
        if val < 0.0:
            raise ValueError(f"{name} weights must be non-negative")
        vals[key] = val
    return vals


def local_path_for_arg(path):
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def remote_path_for_arg(path):
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(REMOTE, path)


def path_with_suffix(path, suffix, default_dir="runs"):
    path = str(path or "").strip()
    if not path:
        return ""
    root, _ext = os.path.splitext(path)
    if not root:
        root = os.path.join(default_dir, "generated")
    return root + suffix


def apply_image_quality_preset(args):
    preset = str(getattr(args, "image_quality_preset", "none") or "none")
    if preset == "none":
        return
    if preset != "web-hf-vae":
        raise ValueError(f"unknown image quality preset {preset!r}")
    args.image_fetch = True
    args.image_fetch_source = "text-to-image-2m-512-2m"
    args.image_score = True
    args.image_score_backend = "stats"
    args.image_score_image_size = max(int(args.image_score_image_size), 512)
    args.image_embed = True
    args.image_embed_backend = "hf"
    args.image_embed_text_mode = "both"
    if not args.image_embed_text_sequence_model:
        args.image_embed_text_sequence_model = "google-t5/t5-base"
    args.image_latent = True
    args.image_cond_mode = "text"
    args.image_caption_cond_source = "auto"
    args.image_fetch_max_records = max(int(args.image_fetch_max_records), 10000)
    args.image_clean_min_side = max(int(args.image_clean_min_side), 512)
    args.image_clean_max_aspect = max(float(args.image_clean_max_aspect), 2.0)
    if args.image_clean_max_nsfw is None:
        args.image_clean_max_nsfw = 0.2
    if args.image_clean_max_watermark is None:
        args.image_clean_max_watermark = 0.5
    if args.image_clean_min_image_text_cosine is None:
        args.image_clean_min_image_text_cosine = 0.15
    if args.image_clean_max_image_duplicate_cosine is None:
        args.image_clean_max_image_duplicate_cosine = 0.985
    args.image_size = "512"
    args.image_size_buckets = "512x512"
    args.image_ae_arch = "hf-vae"
    args.image_latent_arch = "mmdit"
    if not args.image_ae_hf_model:
        args.image_ae_hf_model = "stabilityai/sdxl-vae"
    args.image_latent_downsample = 8
    args.image_latent_patch_size = max(int(args.image_latent_patch_size), 2)
    args.image_latent_max_tokens = max(int(args.image_latent_max_tokens), 2048)
    if int(args.dim or 0) <= 0:
        args.dim = 512
    if int(args.batch) == 16:
        args.batch = 4
    if int(args.train_steps or 0) <= 0:
        args.train_steps = 20000
    args.image_dit_head_width_mult = max(int(args.image_dit_head_width_mult), 2)
    args.image_dit_qk_norm = True
    args.image_dit_attn_impl = "linear"
    args.image_dit_pos_embed = "rope2d"
    args.image_dit_mlp = "swiglu"
    args.image_flow_checkpoint_blocks = True
    args.image_train_precision = "bf16"
    args.image_grad_clip = max(float(args.image_grad_clip), 1.0)
    args.image_flow_cache_latents = True
    args.image_flow_cache_dtype = "bf16"
    args.image_flow_cache_shard_size = max(int(args.image_flow_cache_shard_size), 2048)
    args.image_flow_cache_batch = max(int(args.image_flow_cache_batch), 64)
    args.image_flow_cache_max_loaded_shards = max(int(args.image_flow_cache_max_loaded_shards), 4)
    args.image_text_align_w = max(float(args.image_text_align_w), 0.05)
    args.image_flow_text_align_w = max(float(args.image_flow_text_align_w), 0.05)
    args.image_feature_align_w = max(float(args.image_feature_align_w), 0.05)
    args.image_flow_feature_align_w = max(float(args.image_flow_feature_align_w), 0.05)
    args.image_flow_repa_w = max(float(args.image_flow_repa_w), 0.05)
    args.image_flow_repa_mode = "auto"
    args.image_flow_self_repa_w = max(float(args.image_flow_self_repa_w), 0.05)
    args.image_flow_self_repa_mode = "auto"
    args.image_flow_sra_w = max(float(args.image_flow_sra_w), 0.05)
    args.image_flow_sra_mode = "both"
    args.image_flow_sra_time_gap = float(args.image_flow_sra_time_gap or 0.25)
    args.image_quality_weight = max(float(args.image_quality_weight), 0.5)
    args.image_quality_score_w = max(float(args.image_quality_score_w), 0.02)
    args.image_flow_quality_score_w = max(float(args.image_flow_quality_score_w), 0.01)
    args.image_quality_rank_w = max(float(args.image_quality_rank_w), 0.01)
    args.image_flow_quality_rank_w = max(float(args.image_flow_quality_rank_w), 0.005)
    args.image_quality_rank_margin = max(float(args.image_quality_rank_margin), 0.05)
    if args.image_preference_manifest:
        args.image_quality_score_steps = max(int(args.image_quality_score_steps), 1000)
        args.image_preference_w = max(float(args.image_preference_w), 0.5)
    args.image_flow_consistency_w = max(float(args.image_flow_consistency_w), 0.05)
    args.image_flow_endpoint_w = max(float(args.image_flow_endpoint_w), 0.1)
    args.image_flow_noise_coupling = "sliced_ot"
    args.image_flow_noise_coupling_projections = max(
        int(args.image_flow_noise_coupling_projections), 4)
    args.image_flow_ema_decay = max(float(args.image_flow_ema_decay), 0.999)
    args.image_sample_steps = max(int(args.image_sample_steps), 8)
    args.image_sample_grid = True
    if not args.image_sample_manifest_out:
        args.image_sample_manifest_out = "runs/image_latent_web_hf_vae_generated.jsonl"
    args.image_eval_generated = True
    args.image_prompt_embed_backend = args.image_embed_backend
    args.image_prompt_embed_model = args.image_embed_model
    if not args.image_prompt_embed_text_sequence_model:
        args.image_prompt_embed_text_sequence_model = args.image_embed_text_sequence_model
    if not args.image_sample_prompts:
        args.image_sample_prompts = (
            "a cinematic photo of a hiker crossing snowy mountains; "
            "a stack of pancakes with glossy maple syrup; "
            "a detailed product photo of an orange blue outdoor storage bag"
        )


def upload_path_cmd(local_path, remote_path, ssh):
    local_path = os.path.abspath(local_path)
    if os.path.isdir(local_path):
        return (
            f"COPYFILE_DISABLE=1 tar czf - -C {shlex_quote(local_path)} . "
            f"| {ssh} 'mkdir -p {shlex_quote(remote_path)} && "
            f"tar --no-same-owner -xzf - -C {shlex_quote(remote_path)}'"
        )
    parent = os.path.dirname(local_path) or "."
    name = os.path.basename(local_path)
    remote_dir = os.path.dirname(remote_path) or "."
    return (
        f"COPYFILE_DISABLE=1 tar czf - -C {shlex_quote(parent)} {shlex_quote(name)} "
        f"| {ssh} 'mkdir -p {shlex_quote(remote_dir)} && "
        f"tar --no-same-owner -xzf - -C {shlex_quote(remote_dir)}'"
    )


def payload(args):
    """The pod-side run: kinship multi-seed (+ optional chain sweep), via the package CLI.
    --fast: image-native python (torch preinstalled -- skips the ~5min venv build), 1000 steps,
    3000 examples, trimmed eval (depth-50 verified decoding costs ~75s/example: ~1650 generated
    tokens through the 8-loop model, so n is small there)."""
    PY = ("python3 -u -m thinking.cli" if args.fast
          else "/root/fer-venv/bin/python -u -m thinking.cli")
    trace_rank = (
        (f" --trace-rank-w {args.trace_rank_w}" if args.trace_rank_w else "") +
        (f" --trace-rank-batch {args.trace_rank_batch}" if args.trace_rank_batch else "") +
        (f" --trace-rank-candidates {args.trace_rank_candidates}"
         if args.trace_rank_candidates else "") +
        (f" --trace-rank-states {args.trace_rank_states}" if args.trace_rank_states else "") +
        (f" --trace-dagger-frac {args.trace_dagger_frac}"
         if args.trace_dagger_frac is not None else ""))
    cmds = []
    if args.ablate:
        cmds.append(f"{PY} ablate --steps 800")
    if args.verbalize:
        cmds.append(f"{PY.replace('thinking.cli', 'thinking.verbalize')} "
                    f"--out runs/verbalizer.pt && "
                    f"{PY.replace('thinking.cli', 'thinking.verbalize')} "
                    f"--sample runs/verbalizer.pt")
    if args.lang:                                          # LANG-1: hybrid-vocab fluency model
        VB = PY.replace("thinking.cli", "thinking.verbalize")
        cmds.append(f"{VB} --hybrid --dim {args.dim or 256} --corpus tinystories "
                    f"--corpus-mb {args.lang_mb} --pre-steps {args.train_steps or 40000} "
                    f"--steps {args.lang_ft} --out runs/lang1_fluency.pt && "
                    f"{VB} --sample runs/lang1_fluency.pt")
        return " && ".join(cmds)                           # lang is a COMPLETE payload: without
        #                                                    this return the default kinship
        #                                                    multi-seed run was appended after it
    if (args.vision_understanding or args.vision or args.image2 or args.image_flow
            or args.image_latent or args.image_embed or args.image_fetch
            or args.image_caption or args.image_score or args.audio or args.multimodal):
        effective_image_manifest = args.image_manifest
        if args.image_fetch:
            IFETCH = PY.replace("thinking.cli", "thinking.image_fetch")
            fetch = (f"{IFETCH} --source {args.image_fetch_source} "
                     f"--image-dir {shlex_quote(args.image_fetch_dir)} "
                     f"--manifest {shlex_quote(args.image_fetch_manifest)} "
                     f"--root {shlex_quote(args.image_root)} "
                     f"--max-records {args.image_fetch_max_records} "
                     f"--eval-frac {args.image_fetch_eval_frac} "
                     f"--seed {args.image_fetch_seed} "
                     f"--min-caption-tokens {args.image_clean_min_caption_tokens} "
                     f"--max-caption-tokens {args.image_clean_max_caption_tokens} "
                     f"--report-out {shlex_quote(args.image_fetch_report_out)}")
            if args.image_fetch_source == "diffusiondb-2m":
                fetch += (
                    f" --diffusiondb-start-part {args.image_fetch_diffusiondb_start_part}"
                    f" --diffusiondb-end-part {args.image_fetch_diffusiondb_end_part}")
                if args.image_fetch_diffusiondb_metadata:
                    fetch += (
                        f" --diffusiondb-metadata "
                        f"{shlex_quote(args.image_fetch_diffusiondb_metadata)}")
            if args.image_fetch_overwrite:
                fetch += " --overwrite"
            cmds.append(fetch)
            effective_image_manifest = args.image_fetch_manifest
        if args.image_caption:
            ICAP = PY.replace("thinking.cli", "thinking.image_caption")
            cap = (f"{ICAP} --manifest {shlex_quote(effective_image_manifest)} "
                   f"--root {shlex_quote(args.image_root)} "
                   f"--backend {args.image_caption_backend} "
                   f"--mode {args.image_caption_mode} "
                   f"--batch {args.image_caption_batch} "
                   f"--device {shlex_quote(args.image_caption_device)} "
                   f"--dtype {args.image_caption_dtype} "
                   f"--max-records {args.image_caption_max_records} "
                   f"--max-new-tokens {args.image_caption_max_new_tokens} "
                   f"--num-beams {args.image_caption_num_beams} "
                   f"--out {shlex_quote(args.image_caption_out)} "
                   f"--report-out {shlex_quote(args.image_caption_report_out)}")
            if args.image_caption_model:
                cap += f" --model {shlex_quote(args.image_caption_model)}"
            if args.image_caption_prompt:
                cap += f" --prompt {shlex_quote(args.image_caption_prompt)}"
            if args.image_caption_separator:
                cap += f" --separator {shlex_quote(args.image_caption_separator)}"
            if args.image_caption_trust_remote_code:
                cap += " --trust-remote-code"
            cmds.append(cap)
            effective_image_manifest = args.image_caption_out
        if args.image_score:
            ISCORE = PY.replace("thinking.cli", "thinking.image_score")
            score = (f"{ISCORE} --manifest {shlex_quote(effective_image_manifest)} "
                     f"--root {shlex_quote(args.image_root)} "
                     f"--backend {args.image_score_backend} "
                     f"--max-records {args.image_score_max_records} "
                     f"--image-size {args.image_score_image_size} "
                     f"--technical-w {args.image_score_technical_w} "
                     f"--alignment-w {args.image_score_alignment_w} "
                     f"--external-w {args.image_score_external_w} "
                     f"--out {shlex_quote(args.image_score_out)} "
                     f"--report-out {shlex_quote(args.image_score_report_out)}")
            if args.image_score_split:
                score += f" --split {shlex_quote(args.image_score_split)}"
            if args.image_score_sidecar_out:
                score += f" --sidecar-out {shlex_quote(args.image_score_sidecar_out)}"
            if args.image_score_external_sidecar:
                score += (
                    f" --external-sidecar "
                    f"{shlex_quote(args.image_score_external_sidecar)}")
            if args.image_score_external_root:
                score += f" --external-root {shlex_quote(args.image_score_external_root)}"
            if args.image_score_external_key:
                score += f" --external-key {args.image_score_external_key}"
            if args.image_score_external_score_field:
                score += (
                    f" --external-score-field "
                    f"{shlex_quote(args.image_score_external_score_field)}")
            if args.image_score_external_normalize:
                score += f" --external-normalize {args.image_score_external_normalize}"
            if args.image_score_drop_failed:
                score += " --drop-failed"
            cmds.append(score)
            effective_image_manifest = args.image_score_out
        if args.image_embed:
            IE = PY.replace("thinking.cli", "thinking.image_embed")
            ID = PY.replace("thinking.cli", "thinking.image_data")
            embed = (f"{IE} --manifest {shlex_quote(effective_image_manifest)} "
                     f"--root {shlex_quote(args.image_root)} "
                     f"--backend {args.image_embed_backend} "
                     f"--features {args.image_embed_features} "
                     f"--text-embed-mode {args.image_embed_text_mode} "
                     f"--batch {args.image_embed_batch} "
                     f"--device {args.image_embed_device} "
                     f"--dtype {args.image_embed_dtype} "
                     f"--max-records {args.image_embed_max_records} "
                     f"--out {shlex_quote(args.image_embed_out)} "
                     f"--report-out {shlex_quote(args.image_embed_report_out)}")
            if args.image_embed_model:
                embed += f" --model {shlex_quote(args.image_embed_model)}"
            if args.image_embed_text_sequence_model:
                embed += (
                    f" --text-sequence-model "
                    f"{shlex_quote(args.image_embed_text_sequence_model)}")
            if args.image_embed_text_sequence_max_length:
                embed += (
                    f" --text-sequence-max-length "
                    f"{args.image_embed_text_sequence_max_length}")
            if args.image_embed_no_normalize:
                embed += " --no-normalize"
            if args.image_embed_trust_remote_code:
                embed += " --trust-remote-code"
            clean = (f"{ID} --manifest {shlex_quote(effective_image_manifest)} "
                     f"--root {shlex_quote(args.image_root)} "
                     f"--min-side {args.image_clean_min_side} "
                     f"--max-aspect {args.image_clean_max_aspect} "
                     f"--min-caption-tokens {args.image_clean_min_caption_tokens} "
                     f"--max-caption-tokens {args.image_clean_max_caption_tokens} "
                     f"--embedding-manifest {shlex_quote(args.image_embed_out)} "
                     f"--embedding-root {shlex_quote(args.image_root)} "
                     f"--embedding-key image "
                     f"--write-filtered {shlex_quote(args.image_clean_manifest)} "
                     f"--report-out {shlex_quote(args.image_clean_report_out)}")
            if args.image_min_aesthetic is not None:
                clean += f" --min-aesthetic {args.image_min_aesthetic}"
            if args.image_clean_max_nsfw is not None:
                clean += f" --max-nsfw {args.image_clean_max_nsfw}"
            if args.image_clean_max_watermark is not None:
                clean += f" --max-watermark {args.image_clean_max_watermark}"
            if args.image_clean_min_image_text_cosine is not None:
                clean += (
                    f" --min-image-text-cosine "
                    f"{args.image_clean_min_image_text_cosine}")
            if args.image_clean_max_image_duplicate_cosine is not None:
                clean += (
                    f" --max-image-duplicate-cosine "
                    f"{args.image_clean_max_image_duplicate_cosine}"
                    f" --image-dedupe-lsh-bits {args.image_clean_dedupe_lsh_bits}"
                    f" --image-dedupe-lsh-tables {args.image_clean_dedupe_lsh_tables}"
                    f" --image-dedupe-seed {args.image_clean_dedupe_seed}")
                if args.image_clean_dedupe_keep_first:
                    clean += " --image-dedupe-keep-first"
            if args.image_clean_keep_duplicate_paths:
                clean += " --keep-duplicate-paths"
            if args.image_clean_no_check_images:
                clean += " --no-check-images"
            cmds.extend([embed, clean])
            effective_image_manifest = args.image_clean_manifest
        if args.vision_understanding:
            VU = PY.replace("thinking.cli", "thinking.vision_understanding")
            vu_steps = args.vision_understanding_steps or args.train_steps or 2000
            vu_batch = args.vision_understanding_batch or args.batch
            vu_dim = args.vision_understanding_dim or args.dim or 128
            vu = (f"{VU} --train "
                  f"--manifest {shlex_quote(effective_image_manifest)} "
                  f"--root {shlex_quote(args.image_root)} "
                  f"--split {shlex_quote(args.image_split)} "
                  f"--max-records {args.image_max_records} "
                  f"--steps {vu_steps} --batch {vu_batch} "
                  f"--size {shlex_quote(args.vision_understanding_size)} "
                  f"--patch {args.vision_understanding_patch} "
                  f"--dim {vu_dim} "
                  f"--slots {args.vision_understanding_slots} "
                  f"--heads {args.vision_understanding_heads} "
                  f"--layers {args.vision_understanding_layers} "
                  f"--memory-size {args.vision_understanding_memory_size} "
                  f"--memory-w {args.vision_understanding_memory_w} "
                  f"--association-w {args.vision_understanding_association_w} "
                  f"--composition-w {args.vision_understanding_composition_w} "
                  f"--text-align-w {args.vision_understanding_text_align_w} "
                  f"--image-align-w {args.vision_understanding_image_align_w} "
                  f"--device cuda "
                  f"--out runs/vision_understanding.pt "
                  f"--report-out runs/vision_understanding_report.json")
            cmds.append(vu)
        if args.image_latent:
            IL = PY.replace("thinking.cli", "thinking.image_latent")
            cond_suffix = f"_{args.image_cond_mode}"
            ckpt = f"runs/image_latent_{args.image_latent_arch}{cond_suffix}.pt"
            grid = f"runs/image_latent_{args.image_latent_arch}{cond_suffix}_grid.ppm"
            sample_manifest_args = ""
            if args.image_sample_manifest_out:
                sample_manifest_args += (
                    f" --sample-manifest-out "
                    f"{shlex_quote(args.image_sample_manifest_out)}")
                if args.image_sample_image_dir:
                    sample_manifest_args += (
                        f" --sample-image-dir "
                        f"{shlex_quote(args.image_sample_image_dir)}")
            prompt_grid_args = ""
            if args.image_sample_prompts:
                prompt_grid_args += f" --sample-prompts {shlex_quote(args.image_sample_prompts)}"
                if args.image_sample_negative_prompts:
                    prompt_grid_args += (
                        " --sample-negative-prompts "
                        f"{shlex_quote(args.image_sample_negative_prompts)}")
                prompt_grid_args += (
                    f" --sample-candidates-per-prompt "
                    f"{args.image_sample_candidates_per_prompt}")
                prompt_grid_args += (
                    f" --sample-text-guidance-w {args.image_sample_text_guidance_w} "
                    f"--sample-text-guidance-interval "
                    f"{shlex_quote(args.image_sample_text_guidance_interval)}")
                prompt_grid_args += (
                    f" --sample-feature-guidance-w {args.image_sample_feature_guidance_w} "
                    f"--sample-feature-guidance-interval "
                    f"{shlex_quote(args.image_sample_feature_guidance_interval)}")
                prompt_grid_args += (
                    f" --sample-quality-guidance-w {args.image_sample_quality_guidance_w} "
                    f"--sample-quality-guidance-interval "
                    f"{shlex_quote(args.image_sample_quality_guidance_interval)}")
                prompt_grid_args += f" --prompt-embed-backend {args.image_prompt_embed_backend}"
                prompt_grid_args += f" --prompt-embed-model {shlex_quote(args.image_prompt_embed_model)}"
                if args.image_prompt_embed_text_sequence_model:
                    prompt_grid_args += (
                        f" --prompt-embed-text-sequence-model "
                        f"{shlex_quote(args.image_prompt_embed_text_sequence_model)}")
                if args.image_prompt_embed_text_sequence_max_length:
                    prompt_grid_args += (
                        f" --prompt-embed-text-sequence-max-length "
                        f"{args.image_prompt_embed_text_sequence_max_length}")
                prompt_grid_args += f" --prompt-embed-device {shlex_quote(args.image_prompt_embed_device)}"
                prompt_grid_args += f" --prompt-embed-dtype {args.image_prompt_embed_dtype}"
                prompt_grid_args += f" --prompt-embed-stats-dim {args.image_prompt_embed_stats_dim}"
                if args.image_prompt_embed_no_normalize:
                    prompt_grid_args += " --prompt-embed-no-normalize"
                if args.image_prompt_embed_trust_remote_code:
                    prompt_grid_args += " --prompt-embed-trust-remote-code"
            train = (f"{IL} --train --ae-steps {args.train_steps or 800} "
                     f"--flow-steps {args.train_steps or 800} --batch {args.batch} "
                     f"--ae-accum-steps {args.image_ae_accum_steps} "
                     f"--flow-accum-steps {args.image_flow_accum_steps} "
                     f"--flow-cache-records {args.image_flow_cache_records} "
                     f"--flow-cache-batch {args.image_flow_cache_batch} "
                     f"--flow-cache-shard-size {args.image_flow_cache_shard_size} "
                     f"--flow-cache-dtype {args.image_flow_cache_dtype} "
                     f"--flow-cache-max-loaded-shards {args.image_flow_cache_max_loaded_shards} "
                     f"--train-precision {args.image_train_precision} "
                     f"--grad-clip {args.image_grad_clip} "
                     f"--size {args.image_size} "
                     f"--size-buckets {shlex_quote(args.image_size_buckets)} "
                     f"--hidden {args.dim or 64} --flow-arch {args.image_latent_arch} "
                     f"--dit-head-width-mult {args.image_dit_head_width_mult} "
                     f"--dit-attn-impl {args.image_dit_attn_impl} "
                     f"--dit-pos-embed {args.image_dit_pos_embed} "
                     f"--dit-mlp {args.image_dit_mlp} "
                     f"--ae-arch {args.image_ae_arch} "
                     f"--ae-hf-model {shlex_quote(args.image_ae_hf_model)} "
                     f"--ae-hf-subfolder {shlex_quote(args.image_ae_hf_subfolder)} "
                     f"--ae-hf-scaling-factor {args.image_ae_hf_scaling_factor} "
                     f"--latent-downsample {args.image_latent_downsample} "
                     f"--latent-patch-size {args.image_latent_patch_size} "
                     f"--ae-res-blocks {args.image_ae_res_blocks} "
                     f"--latent-max-tokens {args.image_latent_max_tokens} "
                     f"--ae-recon-loss {args.image_ae_recon_loss} "
                     f"--ae-grad-w {args.image_ae_grad_w} "
                     f"--ae-ms-w {args.image_ae_ms_w} "
                     f"--ae-fft-w {args.image_ae_fft_w} "
                     f"--ae-latent-reg-w {args.image_ae_latent_reg_w} "
                     f"--image-text-align-w {args.image_text_align_w} "
                     f"--flow-text-align-w {args.image_flow_text_align_w} "
                     f"--text-embed-dim {args.image_text_embed_dim} "
                     f"--image-feature-align-w {args.image_feature_align_w} "
                     f"--flow-feature-align-w {args.image_flow_feature_align_w} "
                     f"--image-feature-embed-dim {args.image_feature_embed_dim} "
                     f"--flow-repa-w {args.image_flow_repa_w} "
                     f"--flow-repa-steps {args.image_flow_repa_steps} "
                     f"--flow-repa-embed-dim {args.image_flow_repa_embed_dim} "
                     f"--flow-repa-mode {args.image_flow_repa_mode} "
                     f"--flow-self-repa-w {args.image_flow_self_repa_w} "
                     f"--flow-self-repa-steps {args.image_flow_self_repa_steps} "
                     f"--flow-self-repa-embed-dim {args.image_flow_self_repa_embed_dim} "
                     f"--flow-self-repa-mode {args.image_flow_self_repa_mode} "
                     f"--flow-sra-w {args.image_flow_sra_w} "
                     f"--flow-sra-steps {args.image_flow_sra_steps} "
                     f"--flow-sra-time-gap {args.image_flow_sra_time_gap} "
                     f"--flow-sra-mode {args.image_flow_sra_mode} "
                     f"--cond-mode {args.image_cond_mode} "
                     f"--text-cond-dim {args.image_text_cond_dim} "
                     f"--image-manifest {shlex_quote(effective_image_manifest)} "
                     f"--image-root {shlex_quote(args.image_root)} "
                     f"--image-split {shlex_quote(args.image_split)} "
                     f"--image-max-records {args.image_max_records} "
                     f"--image-quality-weight {args.image_quality_weight} "
                     f"--image-source-weights {shlex_quote(args.image_source_weights)} "
                     f"--image-quality-score-w {args.image_quality_score_w} "
                     f"--flow-quality-score-w {args.image_flow_quality_score_w} "
                     f"--quality-score-steps {args.image_quality_score_steps} "
                     f"--image-quality-rank-w {args.image_quality_rank_w} "
                     f"--quality-score-rank-w {args.image_quality_score_rank_w} "
                     f"--flow-quality-rank-w {args.image_flow_quality_rank_w} "
                     f"--quality-rank-margin {args.image_quality_rank_margin} "
                     f"--image-preference-manifest "
                     f"{shlex_quote(args.image_preference_manifest)} "
                     f"--image-preference-root "
                     f"{shlex_quote(args.image_preference_root)} "
                     f"--image-preference-max-pairs "
                     f"{args.image_preference_max_pairs} "
                     f"--image-preference-w {args.image_preference_w} "
                     f"--caption-vocab-max {args.image_caption_vocab_max} "
                     f"--caption-max-len {args.image_caption_max_len} "
                     f"--caption-cond-source {args.image_caption_cond_source} "
                     f"--image-crop-mode {args.image_crop_mode} "
                     f"--image-hflip-prob {args.image_hflip_prob} "
                     f"--cond-drop {args.image_cond_drop} "
                     f"--cfg-scale {args.image_cfg_scale} "
                     f"--cfg-rescale {args.image_cfg_rescale} "
                     f"--cfg-mode {args.image_cfg_mode} "
                     f"--cfg-interval {shlex_quote(args.image_cfg_interval)} "
                     f"--sample-steps {args.image_sample_steps} "
                     f"--sample-method {args.image_sample_method} "
                     f"--sample-schedule {args.image_sample_schedule} "
                     f"--sample-churn {args.image_sample_churn} "
                     f"--sample-churn-interval "
                     f"{shlex_quote(args.image_sample_churn_interval)} "
                     f"--sample-velocity-clip {args.image_sample_velocity_clip} "
                     f"--sample-latent-clip {args.image_sample_latent_clip} "
                     f"--eval-generated-candidates-per-prompt "
                     f"{args.image_eval_generated_candidates_per_prompt} "
                     f"--semantic-guidance-w {args.image_semantic_guidance_w} "
                     f"--semantic-guidance-mode {args.image_semantic_guidance_mode} "
                     f"--semantic-guidance-interval "
                     f"{shlex_quote(args.image_semantic_guidance_interval)} "
                     f"--roundtrip-samples {args.image_roundtrip_samples} "
                     f"--intervention-samples {args.image_intervention_samples} "
                     f"--ae-intervention-w {args.image_ae_intervention_w} "
                     f"--ae-factor-orth-w {args.image_ae_factor_orth_w} "
                     f"--flow-semantic-w {args.image_flow_semantic_w} "
                     f"--flow-consistency-w {args.image_flow_consistency_w} "
                     f"--flow-endpoint-w {args.image_flow_endpoint_w} "
                     f"--flow-noise-coupling {args.image_flow_noise_coupling} "
                     f"--flow-noise-coupling-projections "
                     f"{args.image_flow_noise_coupling_projections} "
                     f"--flow-distill-steps {args.image_flow_distill_steps} "
                     f"--flow-distill-w {args.image_flow_distill_w} "
                     f"--flow-distill-time-gap {args.image_flow_distill_time_gap} "
                     f"--flow-distill-teacher {args.image_flow_distill_teacher} "
                     f"--flow-guidance-distill-w {args.image_flow_guidance_distill_w} "
                     f"--flow-guidance-distill-cfg-scale "
                     f"{args.image_flow_guidance_distill_cfg_scale} "
                     f"--flow-guidance-distill-cfg-rescale "
                     f"{args.image_flow_guidance_distill_cfg_rescale} "
                     f"--flow-ema-decay {args.image_flow_ema_decay} "
                     f"--ema-eval-mode {args.image_ema_eval_mode} "
                     f"--time-sampling {args.image_time_sampling} "
                     f"--time-logit-mean {args.image_time_logit_mean} "
                     f"--time-logit-std {args.image_time_logit_std} "
                     f"--time-curriculum-frac {args.image_time_curriculum_frac} "
                     f"--time-shift {args.image_time_shift} "
                     f"--time-shift-mode {args.image_time_shift_mode} "
                     f"--time-shift-ref-dim {args.image_time_shift_ref_dim} "
                     f"--time-shift-dim-power {args.image_time_shift_dim_power} "
                     f"--flow-loss-weight {args.image_flow_loss_weight} "
                     f"--flow-loss-weight-gamma {args.image_flow_loss_weight_gamma} "
                     f"--latent-normalize {args.image_latent_normalize} "
                     f"--latent-stat-samples {args.image_latent_stat_samples} "
                     f"--out {ckpt}")
            if args.image_sample_grid:
                train += (f" --sample-grid-out {grid} "
                          f"--sample-grid-samples {args.image_sample_grid_samples}"
                          f"{sample_manifest_args}{prompt_grid_args}")
            if args.image_no_ema_warmup:
                train += " --no-ema-warmup"
            if args.image_dit_qk_norm:
                train += " --dit-qk-norm"
            if args.image_flow_checkpoint_blocks:
                train += " --flow-checkpoint-blocks"
            if args.image_flow_cache_latents:
                train += " --flow-cache-latents"
            if args.image_sample_finite_guard:
                train += " --sample-finite-guard"
            if args.image_no_flow_loss_weight_normalize:
                train += " --no-flow-loss-weight-normalize"
            if args.image_flow_cache_dir:
                train += f" --flow-cache-dir {shlex_quote(args.image_flow_cache_dir)}"
            if args.image_prompt_templates:
                train += f" --prompt-templates {shlex_quote(args.image_prompt_templates)}"
            if args.image_min_aesthetic is not None:
                train += f" --image-min-aesthetic {args.image_min_aesthetic}"
            if args.image_eval_sweep:
                eval_cmd = (f" && {IL} --eval-checkpoint {ckpt} "
                          f"--size {args.image_size} "
                          f"--checkpoint-weight-mode {args.image_checkpoint_weight_mode} "
                          f"--cfg-scales {shlex_quote(args.image_cfg_sweep)} "
                          f"--cfg-rescales {shlex_quote(args.image_cfg_rescale_sweep)} "
                          f"--cfg-mode {args.image_cfg_mode} "
                          f"--cfg-modes "
                          f"{shlex_quote(args.image_cfg_modes or args.image_cfg_mode)} "
                          f"--sample-steps-list {shlex_quote(args.image_sample_steps_sweep)} "
                          f"--cfg-interval {shlex_quote(args.image_cfg_interval)} "
                          f"--sample-method {args.image_sample_method} "
                          f"--sample-methods {shlex_quote(args.image_sample_methods)} "
                          f"--sample-schedule {args.image_sample_schedule} "
                          f"--sample-schedules {shlex_quote(args.image_sample_schedules)} "
                          f"--sample-churn {args.image_sample_churn} "
                          f"--sample-churns {shlex_quote(args.image_sample_churns)} "
                          f"--sample-churn-interval "
                          f"{shlex_quote(args.image_sample_churn_interval)} "
                          f"--sample-velocity-clip {args.image_sample_velocity_clip} "
                          f"--sample-latent-clip {args.image_sample_latent_clip} "
                          f"--eval-seeds {shlex_quote(args.image_eval_seeds)} "
                          f"--eval-out runs/image_latent_{args.image_latent_arch}"
                          f"{cond_suffix}_sweep.json")
                if effective_image_manifest:
                    eval_cmd += (f" --eval-image-manifest "
                                 f"{shlex_quote(effective_image_manifest)} "
                                 f"--eval-image-root {shlex_quote(args.image_root)} "
                                 f"--eval-image-split {shlex_quote(args.image_eval_split)} "
                                 f"--eval-image-max-records {args.image_eval_max_records} "
                                 f"--eval-generated-samples "
                                 f"{args.image_eval_generated_samples} "
                                 f"--eval-generated-candidates-per-prompt "
                                 f"{args.image_eval_generated_candidates_per_prompt} "
                                 f"--eval-text-guidance-weights "
                                 f"{shlex_quote(args.image_eval_text_guidance_sweep)} "
                                 f"--eval-feature-guidance-weights "
                                 f"{shlex_quote(args.image_eval_feature_guidance_sweep)} "
                                 f"--eval-quality-guidance-weights "
                                 f"{shlex_quote(args.image_eval_quality_guidance_sweep)}")
                    if args.image_min_aesthetic is not None:
                        eval_cmd += (
                            f" --eval-image-min-aesthetic {args.image_min_aesthetic}"
                        )
                else:
                    eval_cmd += (
                          f" --semantic-guidance-w {args.image_semantic_guidance_w} "
                          f"--semantic-guidance-weights "
                          f"{shlex_quote(args.image_semantic_guidance_sweep)} "
                          f"--semantic-guidance-mode {args.image_semantic_guidance_mode} "
                          f"--semantic-guidance-interval "
                          f"{shlex_quote(args.image_semantic_guidance_interval)} "
                          f"--roundtrip-samples {args.image_roundtrip_samples} "
                          f"--intervention-samples {args.image_intervention_samples}")
                if args.image_sample_finite_guard:
                    eval_cmd += " --sample-finite-guard"
                if args.image_sample_grid:
                    eval_cmd += (f" --sample-grid-out {grid} "
                                 f"--sample-grid-samples {args.image_sample_grid_samples}"
                                 f"{sample_manifest_args}{prompt_grid_args}")
                train += eval_cmd
            if args.image_eval_generated:
                IEVAL = PY.replace("thinking.cli", "thinking.image_eval")
                IGEN_EMBED = PY.replace("thinking.cli", "thinking.image_embed")
                generated_manifest = args.image_sample_manifest_out
                generated_embed_out = (
                    args.image_generated_embed_out
                    or path_with_suffix(generated_manifest, "_embeddings.jsonl")
                )
                generated_embed_report = (
                    args.image_generated_embed_report_out
                    or path_with_suffix(generated_manifest, "_embed_report.json")
                )
                generated_eval_report = (
                    args.image_generated_eval_report_out
                    or path_with_suffix(generated_manifest, "_eval_report.json")
                )
                real_eval_manifest = (
                    args.image_eval_generated_real_manifest or effective_image_manifest
                )
                generated_embed = (
                    f" && {IGEN_EMBED} --manifest {shlex_quote(generated_manifest)} "
                    f"--backend {args.image_embed_backend} "
                    f"--features both --text-embed-mode pooled "
                    f"--batch {args.image_embed_batch} "
                    f"--device {args.image_embed_device} "
                    f"--dtype {args.image_embed_dtype} "
                    f"--max-records {args.image_generated_eval_max_records} "
                    f"--out {shlex_quote(generated_embed_out)} "
                    f"--report-out {shlex_quote(generated_embed_report)}"
                )
                if args.image_embed_model:
                    generated_embed += f" --model {shlex_quote(args.image_embed_model)}"
                if args.image_embed_no_normalize:
                    generated_embed += " --no-normalize"
                if args.image_embed_trust_remote_code:
                    generated_embed += " --trust-remote-code"
                generated_eval = (
                    f" && {IEVAL} "
                    f"--real-manifest {shlex_quote(real_eval_manifest)} "
                    f"--generated-manifest {shlex_quote(generated_manifest)} "
                    f"--real-root {shlex_quote(args.image_root)} "
                    f"--generated-embedding-sidecar {shlex_quote(generated_embed_out)} "
                    f"--embedding-key image "
                    f"--max-records {args.image_generated_eval_max_records} "
                    f"--min-score {args.image_generated_eval_min_score} "
                    f"--min-support-precision "
                    f"{args.image_generated_eval_min_support_precision} "
                    f"--min-support-recall "
                    f"{args.image_generated_eval_min_support_recall} "
                    f"--report-out {shlex_quote(generated_eval_report)}"
                )
                if args.image_generated_eval_min_image_text_cos is not None:
                    generated_eval += (
                        f" --min-image-text-cos "
                        f"{args.image_generated_eval_min_image_text_cos}")
                if args.image_generated_eval_max_frechet is not None:
                    generated_eval += f" --max-frechet {args.image_generated_eval_max_frechet}"
                if args.image_generated_eval_max_mmd_rbf is not None:
                    generated_eval += f" --max-mmd-rbf {args.image_generated_eval_max_mmd_rbf}"
                if args.image_generated_eval_fail_on_gate:
                    generated_eval += " --fail-on-gate"
                train += generated_embed + generated_eval
            cmds.append(train)
        if args.audio:                                     # AUDIO-1: audio factors -> facts
            AU = PY.replace("thinking.cli", "thinking.audio")
            cmds.append(f"{AU} --steps {args.train_steps or 500} "
                        f"--seeds {shlex_quote(args.seeds)} --out runs/audio1_fer.json")
        if args.multimodal:                                # M-0: image+audio -> canonical facts
            MM = PY.replace("thinking.cli", "thinking.multimodal")
            mm_dim = args.multimodal_dim or args.dim or 96
            mm_cmd = (
                f"{MM} --steps {args.train_steps or 400} "
                f"--batch {args.multimodal_batch} --dim {mm_dim} "
                f"--layers {args.multimodal_layers} --heads {args.multimodal_heads} "
                f"--lr {args.multimodal_lr} --log-every {args.multimodal_log_every} "
                f"--value-w {args.multimodal_value_w} "
                f"--agreement-w {args.multimodal_agreement_w} "
                f"--concept-tokens {args.multimodal_concept_tokens} "
                f"--fusion-layers {args.multimodal_fusion_layers} "
                f"--concept-w {args.multimodal_concept_w} "
                f"--concept-agreement-w {args.multimodal_concept_agreement_w} "
                f"--concept-distill-w {args.multimodal_concept_distill_w} "
                f"--concept-distill-temperature "
                f"{args.multimodal_concept_distill_temperature} "
                f"--concept-rank-distill-w {args.multimodal_concept_rank_distill_w} "
                f"--concept-rank-distill-margin "
                f"{args.multimodal_concept_rank_distill_margin} "
                f"--concept-transfer-w {args.multimodal_concept_transfer_w} "
                f"--concept-transfer-margin "
                f"{args.multimodal_concept_transfer_margin} "
                f"--concept-contrast-w {args.multimodal_concept_contrast_w} "
                f"--concept-contrast-temperature "
                f"{args.multimodal_concept_contrast_temperature} "
                f"--concept-centroid-w {args.multimodal_concept_centroid_w} "
                f"--concept-centroid-temperature "
                f"{args.multimodal_concept_centroid_temperature} "
                f"--concept-centroid-margin {args.multimodal_concept_centroid_margin} "
                f"--concept-prototype-w {args.multimodal_concept_prototype_w} "
                f"--concept-prototype-spread-w "
                f"{args.multimodal_concept_prototype_spread_w} "
                f"--concept-prototype-spread-margin "
                f"{args.multimodal_concept_prototype_spread_margin} "
                f"--concept-state-spread-w {args.multimodal_concept_state_spread_w} "
                f"--concept-state-spread-variance "
                f"{args.multimodal_concept_state_spread_variance} "
                f"--concept-state-spread-margin "
                f"{args.multimodal_concept_state_spread_margin} "
                f"--concept-state-spread-covariance-w "
                f"{args.multimodal_concept_state_spread_covariance_w} "
                f"--latent-concept-slots {args.multimodal_latent_concept_slots} "
                f"--latent-concept-layers {args.multimodal_latent_concept_layers} "
                f"--latent-concept-w {args.multimodal_latent_concept_w} "
                f"--latent-concept-view-dropout "
                f"{args.multimodal_latent_concept_view_dropout} "
                f"--latent-concept-invariance-w "
                f"{args.multimodal_latent_concept_invariance_w} "
                f"--latent-concept-variance-w "
                f"{args.multimodal_latent_concept_variance_w} "
                f"--latent-concept-covariance-w "
                f"{args.multimodal_latent_concept_covariance_w} "
                f"--latent-concept-variance-target "
                f"{args.multimodal_latent_concept_variance_target} "
                f"--img-tokens {args.multimodal_img_tokens} "
                f"--aud-tokens {args.multimodal_aud_tokens} "
                f"--txt-tokens {args.multimodal_txt_tokens} "
                f"--trunk-arch {args.multimodal_trunk_arch} "
                f"--trunk-width {args.multimodal_trunk_width} "
                f"--trunk-depth {args.multimodal_trunk_depth} "
                f"--text-layers {args.multimodal_text_layers} "
                f"--modality-dropout {args.multimodal_dropout} "
                f"--eval-n {args.multimodal_eval_n} --free-n {args.multimodal_free_n} "
                f"--counterfactual-n {args.multimodal_counterfactual_n} "
                f"--free-counterfactual-n {args.multimodal_free_counterfactual_n} "
                f"--out runs/m0_multimodal.json --checkpoint runs/m0_multimodal.pt"
            )
            if args.multimodal_concept_prefix:
                mm_cmd += " --concept-prefix"
            if args.multimodal_surfaces:
                mm_cmd += f" --surfaces {shlex_quote(args.multimodal_surfaces)}"
            cmds.append(mm_cmd)
        return " && ".join(cmds)
    if args.eval_only_run:
        run = shlex_quote(args.eval_only_run)
        depths = ",".join(str(d) for d in args.eval_depths)
        eval_block = max(18432, 144 * max(args.eval_depths))
        eval_cmds = []
        for decode in args.eval_decodes:
            out = os.path.join(args.eval_only_run, f"deep_eval_{decode}.json")
            eval_cmds.append(
                f"{PY} deep-eval {run} --depths {shlex_quote(depths)} "
                f"--n {args.eval_n} --preds ancestor --block {eval_block} "
                f"--decode {shlex_quote(decode)} --out {shlex_quote(out)}")
        return " && ".join(eval_cmds)
    if args.learning_curve:
        eval_block = max(18432, 144 * max(args.eval_depths))
        depths = ",".join(str(d) for d in args.eval_depths)
        curve_cmds = ["rm -rf runs/learn_* runs/learning_curve_summary.json && mkdir -p runs"]
        runs = []
        for arm in args.curve_arms:
            if arm == "aux":
                rw, rcw = args.rule_w, args.rule_contrast_w
            elif arm == "noaux":
                rw, rcw = 0.0, 0.0
            else:
                raise ValueError(f"unknown learning-curve arm: {arm}")
            for steps in args.curve_steps:
                run = f"runs/learn_{arm}_{steps}"
                runs.append(run)
                train = (f"{PY} train --world kinship --simple --canon "
                         f"--deep-depth {args.deep_depth} --deep-preds ancestor "
                         f"--deep-frac {args.deep_frac} --dim {args.dim or 256} "
                         f"--steps {steps} --examples {args.examples} --batch {args.batch} "
                         f"--rule-w {rw} --rule-contrast-w {rcw}{trace_rank} --out {run}")
                evals = []
                for decode in args.eval_decodes:
                    out = f"{run}/deep_eval_{decode}.json"
                    evals.append(
                        f"{PY} deep-eval {run} --depths {depths} --n {args.eval_n} "
                        f"--preds ancestor --block {eval_block} --decode {decode} "
                        f"--out {out}")
                probe = (f"{PY} probe {run} --depths {depths} --n {args.probe_n} "
                         f"--preds ancestor --block {eval_block} --out {run}/fer_probe.json")
                curve_cmds.append(" && ".join([train] + evals + [probe]))
        summary_code = (
            "import json, pathlib\n"
            "rows=[]\n"
            f"runs={runs!r}\n"
            "for run in runs:\n"
            "    p=pathlib.Path(run)\n"
            "    arm, steps = p.name.split('_')[1], int(p.name.split('_')[2])\n"
            "    row={'run':run,'arm':arm,'steps':steps}\n"
            "    for ep in p.glob('deep_eval_*.json'):\n"
            "        data=json.loads(ep.read_text())\n"
            "        dec=data.get('decode', ep.stem.replace('deep_eval_',''))\n"
            "        row[f'{dec}_by_depth']=data.get('by_depth',{})\n"
            "    fp=p/'fer_probe.json'\n"
            "    if fp.exists():\n"
            "        pr=json.loads(fp.read_text())\n"
            "        keys=['same_rule_cos','different_rule_cos','rule_reuse_margin',"
            "'same_rule_cross_depth_cos','cross_depth_reuse_margin','cross_depth_reuse_gap',"
            "'depth_index_leakage','ufr_score','verdict','weak_rules','risk_flags','n_vectors']\n"
            "        row['fer']={k:pr.get(k) for k in keys}\n"
            "    rows.append(row)\n"
            "out=pathlib.Path('runs/learning_curve_summary.json')\n"
            "out.write_text(json.dumps(rows, indent=1))\n"
            "print(out.read_text())\n")
        summary = f"python3 -c {shlex_quote(summary_code)}"
        return " && ".join(curve_cmds + [summary])
    if args.lengen:                                        # RUNG L: train shallow-deep (<=6),
        cmds2 = []                                         # eval FAR deeper -- length-gen arms
        for pos in ("rope", "none"):
            run = f"runs/lengen_{pos}"
            evs = []
            for hop, n in (("6", 10), ("10", 10), ("20", 6), ("40", 4)):
                evs.append(f"{PY} eval {run} --mode verified --split iid --hops {hop} "
                           f"--n {n} --block 6144 --preds ancestor")
            cmds2.append(f"{PY} train --world kinship --simple --bank --no-curriculum "
                         f"--deep-depth 6 --deep-frac 0.4 --pos {pos} --test-names 110 "
                         f"--out {run} --seed 0 --batch 16 "
                         f"--steps {args.train_steps or 15000} --dim 256 && "
                         + " ; ".join(evs))                # evals NON-FATAL: one bad cell
        return " && ".join(cmds2).replace(" && python3 -u -m thinking.cli eval", " ; python3 -u -m thinking.cli eval")
    if args.deep_ancestor_rule_aux:
        run = args.run_name
        eval_block = max(18432, 144 * max(args.eval_depths))
        train = (f"{PY} train --world kinship --simple --canon "
                 f"--deep-depth {args.deep_depth} --deep-preds ancestor "
                 f"--deep-frac {args.deep_frac} --dim {args.dim or 256} "
                 f"--steps {args.train_steps or 8000} --examples {args.examples} "
                 f"--batch {args.batch} --rule-w {args.rule_w} "
                 f"--rule-contrast-w {args.rule_contrast_w}{trace_rank} --out {run}")
        depths = ",".join(str(d) for d in args.eval_depths)
        return " && ".join([
            train,
            f"{PY} deep-eval {run} --depths {depths} --n {args.eval_n} "
            f"--preds ancestor --block {eval_block}",
            f"{PY} probe {run} --depths {depths} --n {args.probe_n} "
            f"--preds ancestor --block {eval_block}",
        ])
    if args.stair:                                         # staircase: minimal world, decisive evals
        run = f"runs/stair_{args.stair_world}"
        canon = ((" --canon" if args.canon else "") + (" --bank" if args.bank else "") + (" --no-curriculum" if args.no_curriculum else ""))
        simple = " --simple" if args.stair_world == "kinship" else ""
        if args.stair_world == "kinship" and args.deep_depth and args.stair:
            stair_df = args.deep_frac if args.deep_frac != 0.6 else 0.3   # 0.6 = non-stair default
            simple += f" --deep-depth {args.deep_depth} --deep-frac {stair_df} --contrastive 0.9"
        sbatch = 8 if (args.deep_depth and args.stair_world == "kinship") else 32
        hops = ("2,3" if not (args.deep_depth and args.stair_world == "kinship")
                else f"2,3,{args.deep_depth // 2},{args.deep_depth}")
        if args.stair_world == "chain":
            hops = "2,4,6"
        # training must succeed (&&); evals are non-fatal and CHEAP-FIRST (free before
        # verified -- verified deep evals can run 14+ min/depth and hit the pod cap)
        evals = "; ".join([
            f"{PY} eval {run} --mode free --split iid --hops {hops} --n 20",
            f"{PY} eval {run} --mode free --split iid --hops {hops} --n 20 --train-names",
            f"{PY} eval {run} --mode free --split iid --hops {hops} --n 20 --phrasings eval",
            f"{PY} eval {run} --mode verified --split iid --hops {hops} --n 20",
            f"{PY} eval {run} --mode verified --split iid --hops {hops} --n 20 --train-names",
            f"{PY} eval {run} --mode verified --split iid --hops {hops} --n 20 --phrasings eval",
            f"{PY} demo {run} --k 2",
        ])
        return (f"{PY} train --world {args.stair_world}{simple}{canon} --out {run} --seed 0 "
                f"--batch {sbatch} --steps {args.train_steps or 4000}"
                + (f" --dim {args.dim}" if args.dim else "")
                + trace_rank
                + " && { " + evals + "; }")
    neg = " --neg" if args.neg else ""
    loops = f" --loops {args.loops}" if args.loops else ""
    noloop = " --no-loop" if args.no_loop else ""
    steps = f" --steps {args.train_steps}" if args.train_steps else (
        " --steps 1000 --examples 3000" if args.fast else "")
    for s in args.seeds.split(","):
        run = f"runs/kin_s{s}"
        cmds.append(f"{PY} train --world kinship --deep-depth {args.deep_depth} --out {run} "
                    f"--seed {s} --batch {args.batch}{steps}{loops}{noloop}{neg}"
                    f"{trace_rank}")
        if args.fast:
            d = args.deep_depth
            cmds += [
                f"{PY} eval {run} --mode verified --split iid --hops 3 --n 20",
                f"{PY} eval {run} --mode free --split iid --hops 3 --n 20",
                f"{PY} eval {run} --mode verified --split iid --hops {d // 2},{d} --n 4",
                f"{PY} eval {run} --mode verified --split holdout --hops 3 --n 20",
                f"{PY} eval {run} --mode extract --split iid --hops 3,{d // 2} --n 12",
                f"{PY} eval {run} --mode self --split iid --hops 3 --n 12",
                f"{PY} eval {run} --mode verified --split iid --hops {d // 2} --n 12 "
                f"--preds older_by,who_older,who_younger",
                f"{PY} eval {run} --mode write --split iid --hops 3 --n 24",
                f"{PY} eval {run} --mode math --split iid --hops 3 --n 40",
                f"{PY} eval {run} --mode verified --split iid --hops 3 --n 20 --phrasings eval",
                f"{PY} eval {run} --mode extract --split iid --hops 3 --n 12 --phrasings eval",
                f"{PY} eval {run} --mode verified --split novel --hops 2 --n 20",
            ]
        else:
            cmds += [
                f"{PY} eval {run} --mode verified --split iid",
                f"{PY} eval {run} --mode free --split iid",
                f"{PY} eval {run} --mode verified --split holdout",
                f"{PY} eval {run} --mode free --split holdout",
            ]
    cmds.append(f"{PY} demo runs/kin_s{args.seeds.split(',')[0]} --k 3")
    if args.sweep:
        cmds.append(f"{PY} sweep --out runs/grid --seeds {args.seeds}")
    return " && ".join(cmds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="fer-thinking")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--deep-depth", type=int, default=50, help="kinship deep-tree depth")
    ap.add_argument("--batch", type=int, default=16,
                    help="H100 batch at block 3200 (8-loop backward graph holds 8 LxL attention "
                         "maps per head -- batch 64 OOMs 80GB)")
    ap.add_argument("--train-steps", type=int, default=0, help="override training steps")
    ap.add_argument("--loops", type=int, default=0, help="latent recursion depth override")
    ap.add_argument("--neg", action="store_true", help="include the negatives fine-tune pass")
    ap.add_argument("--fast", action="store_true", help="<20min: image python, trimmed eval")
    ap.add_argument("--stair", action="store_true",
                    help="staircase rung A: minimal world, train-names + held-out evals")
    ap.add_argument("--canon", action="store_true",
                    help="rung A0: canonical fact surfaces (chain-world conditions)")
    ap.add_argument("--dim", type=int, default=0, help="model width override")
    ap.add_argument("--bank", action="store_true", help="rung B: surface bank + curriculum")
    ap.add_argument("--no-curriculum", action="store_true", dest="no_curriculum")
    ap.add_argument("--stair-world", default="kinship", dest="stair_world",
                    choices=("kinship", "chain"),
                    help="rung 0 = chain (the baseline-validated task on the production trainer)")
    ap.add_argument("--no-loop", action="store_true", dest="no_loop",
                    help="train non-looped (pending the loop-regression ablation verdict)")
    ap.add_argument("--ablate", action="store_true", help="run the loop ablation first")
    ap.add_argument("--verbalize", action="store_true", help="train+sample the verbalizer")
    ap.add_argument("--lang", action="store_true",
                    help="LANG-1: hybrid-vocab fluency pretraining (reasoning-compatible)")
    ap.add_argument("--lang-mb", type=int, default=24, dest="lang_mb")
    ap.add_argument("--lang-ft", type=int, default=6000, dest="lang_ft")
    ap.add_argument("--ref", default="HEAD",
                    help="deploy this git ref (pinned commit); '' = live tree")
    ap.add_argument("--vision", action="store_true",
                    help="removed legacy synthetic image harness; use --vision-understanding")
    ap.add_argument("--vision-arch", default="shared", choices=("shared", "factored", "bottleneck"),
                    help="removed legacy synthetic image harness option")
    ap.add_argument("--image2", action="store_true",
                    help="removed legacy synthetic image harness; use --vision-understanding")
    ap.add_argument("--image-flow", action="store_true", dest="image_flow",
                    help="removed legacy synthetic image harness; use --image-latent with manifest")
    ap.add_argument("--vision-understanding", action="store_true",
                    dest="vision_understanding",
                    help="train manifest-driven visual concept slots and concept memory")
    ap.add_argument("--vision-understanding-steps", type=int, default=0,
                    dest="vision_understanding_steps",
                    help="vision understanding steps; 0 uses --train-steps or 2000")
    ap.add_argument("--vision-understanding-batch", type=int, default=0,
                    dest="vision_understanding_batch",
                    help="vision understanding batch; 0 uses --batch")
    ap.add_argument("--vision-understanding-size", default="128",
                    dest="vision_understanding_size",
                    help="vision understanding image size as SIZE or HxW")
    ap.add_argument("--vision-understanding-patch", type=int, default=8,
                    dest="vision_understanding_patch",
                    help="vision understanding patch size")
    ap.add_argument("--vision-understanding-dim", type=int, default=0,
                    dest="vision_understanding_dim",
                    help="vision understanding width; 0 uses --dim or 128")
    ap.add_argument("--vision-understanding-slots", type=int, default=8,
                    dest="vision_understanding_slots",
                    help="number of latent visual concept slots")
    ap.add_argument("--vision-understanding-heads", type=int, default=4,
                    dest="vision_understanding_heads",
                    help="attention heads for latent visual concept slots")
    ap.add_argument("--vision-understanding-layers", type=int, default=1,
                    dest="vision_understanding_layers",
                    help="latent concept mixer layers")
    ap.add_argument("--vision-understanding-memory-size", type=int, default=1024,
                    dest="vision_understanding_memory_size",
                    help="persistent visual concept memory size")
    ap.add_argument("--vision-understanding-memory-w", type=float, default=0.1,
                    dest="vision_understanding_memory_w",
                    help="prototype memory loss weight")
    ap.add_argument("--vision-understanding-association-w", type=float, default=0.05,
                    dest="vision_understanding_association_w",
                    help="self-mined concept association loss weight")
    ap.add_argument("--vision-understanding-composition-w", type=float, default=0.02,
                    dest="vision_understanding_composition_w",
                    help="self-mined concept composition loss weight")
    ap.add_argument("--vision-understanding-text-align-w", type=float, default=0.0,
                    dest="vision_understanding_text_align_w",
                    help="optional text_embedding alignment weight")
    ap.add_argument("--vision-understanding-image-align-w", type=float, default=0.0,
                    dest="vision_understanding_image_align_w",
                    help="optional image_embedding alignment weight")
    ap.add_argument("--image-latent", action="store_true", dest="image_latent",
                    help="train manifest-conditioned autoencoder + latent image flow")
    ap.add_argument("--image-quality-preset", default="none",
                    choices=("none", "web-hf-vae"),
                    dest="image_quality_preset",
                    help=("apply a coherent high-quality image training profile; "
                          "web-hf-vae fetches web data and trains in a frozen SDXL VAE latent space"))
    ap.add_argument("--image-fetch", action="store_true", dest="image_fetch",
                    help="stream a captioned web image shard into a local manifest on the pod")
    ap.add_argument("--image-fetch-source", default="text-to-image-2m-1024-10k",
                    choices=("text-to-image-2m-1024-10k", "flux-1024-10k",
                             "text-to-image-2m-512-2m", "diffusiondb-2m"),
                    dest="image_fetch_source",
                    help="known source for --image-fetch")
    ap.add_argument("--image-fetch-max-records", type=int, default=1024,
                    dest="image_fetch_max_records",
                    help="max image/caption pairs to fetch on the pod; 0 means whole shard")
    ap.add_argument("--image-fetch-eval-frac", type=float, default=0.02,
                    dest="image_fetch_eval_frac",
                    help="fraction of fetched rows marked split=eval")
    ap.add_argument("--image-fetch-seed", type=int, default=0,
                    dest="image_fetch_seed")
    ap.add_argument("--image-fetch-dir", default="data/images/web_fetch",
                    dest="image_fetch_dir",
                    help="pod-local image directory for fetched web data")
    ap.add_argument("--image-fetch-manifest", default="data/images/web_fetch.jsonl",
                    dest="image_fetch_manifest",
                    help="pod-local manifest written by --image-fetch")
    ap.add_argument("--image-fetch-report-out", default="runs/image_fetch_report.json",
                    dest="image_fetch_report_out",
                    help="pod-local fetch report JSON")
    ap.add_argument("--image-fetch-diffusiondb-start-part", type=int, default=1,
                    dest="image_fetch_diffusiondb_start_part",
                    help="first DiffusionDB zip part fetched when --image-fetch-source diffusiondb-2m")
    ap.add_argument("--image-fetch-diffusiondb-end-part", type=int, default=0,
                    dest="image_fetch_diffusiondb_end_part",
                    help="last DiffusionDB zip part fetched; 0 lets image_fetch use all available parts")
    ap.add_argument("--image-fetch-diffusiondb-metadata", default="",
                    dest="image_fetch_diffusiondb_metadata",
                    help="optional path or URL to DiffusionDB metadata.parquet for nsfw/size fields")
    ap.add_argument("--image-fetch-overwrite", action="store_true",
                    dest="image_fetch_overwrite",
                    help="overwrite fetched image files if present")
    ap.add_argument("--image-caption", action="store_true", dest="image_caption",
                    help="generate improved captions into a new manifest before embedding/training")
    ap.add_argument("--image-caption-backend", default="hf", choices=("stats", "hf"),
                    dest="image_caption_backend",
                    help="captioning backend for --image-caption")
    ap.add_argument("--image-caption-model", default="Salesforce/blip-image-captioning-large",
                    dest="image_caption_model",
                    help="Hugging Face caption model id for --image-caption-backend hf")
    ap.add_argument("--image-caption-mode", default="replace",
                    choices=("replace", "append", "fill-empty", "sidecar"),
                    dest="image_caption_mode",
                    help="how generated captions are written into the captioned manifest")
    ap.add_argument("--image-caption-prompt", default="", dest="image_caption_prompt",
                    help="optional captioning instruction for models that accept text prompts")
    ap.add_argument("--image-caption-separator", default="; ",
                    dest="image_caption_separator",
                    help="separator used by --image-caption-mode append")
    ap.add_argument("--image-caption-batch", type=int, default=8,
                    dest="image_caption_batch",
                    help="caption generation batch size")
    ap.add_argument("--image-caption-device", default="cuda",
                    dest="image_caption_device",
                    help="device used by image caption preprocessing")
    ap.add_argument("--image-caption-dtype", default="auto",
                    choices=("auto", "fp32", "fp16", "bf16"),
                    dest="image_caption_dtype",
                    help="dtype used by Hugging Face image caption preprocessing")
    ap.add_argument("--image-caption-max-records", type=int, default=0,
                    dest="image_caption_max_records",
                    help="cap captioned records for GPU smoke tests; 0 means all")
    ap.add_argument("--image-caption-max-new-tokens", type=int, default=96,
                    dest="image_caption_max_new_tokens",
                    help="generation length for --image-caption")
    ap.add_argument("--image-caption-num-beams", type=int, default=3,
                    dest="image_caption_num_beams",
                    help="beam count for --image-caption")
    ap.add_argument("--image-caption-out", default="runs/image_captioned.jsonl",
                    dest="image_caption_out",
                    help="captioned manifest written by --image-caption")
    ap.add_argument("--image-caption-report-out", default="runs/image_caption_report.json",
                    dest="image_caption_report_out",
                    help="caption generation report JSON")
    ap.add_argument("--image-caption-trust-remote-code", action="store_true",
                    dest="image_caption_trust_remote_code",
                    help="pass trust_remote_code=True to Hugging Face caption loaders")
    ap.add_argument("--image-score", action="store_true", dest="image_score",
                    help=("annotate the active image manifest with generic quality_score "
                          "metadata before embedding/cleaning/training"))
    ap.add_argument("--image-score-backend", default="stats",
                    choices=("stats", "embedding", "external", "ensemble"),
                    dest="image_score_backend",
                    help=("quality source for --image-score: technical stats, existing "
                          "image/text embeddings, external reward sidecar, or ensemble"))
    ap.add_argument("--image-score-split", default="", dest="image_score_split",
                    help="optional split to score; default scores all rows")
    ap.add_argument("--image-score-max-records", type=int, default=0,
                    dest="image_score_max_records",
                    help="cap rows scored by --image-score; 0 means all")
    ap.add_argument("--image-score-image-size", type=int, default=256,
                    dest="image_score_image_size",
                    help="working image size for dependency-free technical quality scoring")
    ap.add_argument("--image-score-technical-w", type=float, default=1.0,
                    dest="image_score_technical_w",
                    help="ensemble weight for technical image quality")
    ap.add_argument("--image-score-alignment-w", type=float, default=0.0,
                    dest="image_score_alignment_w",
                    help="ensemble weight for existing image/text embedding cosine")
    ap.add_argument("--image-score-external-w", type=float, default=0.0,
                    dest="image_score_external_w",
                    help="ensemble weight for an external preference/reward sidecar")
    ap.add_argument("--image-score-external-sidecar", default="",
                    dest="image_score_external_sidecar",
                    help="JSONL/CSV/TSV sidecar with external quality/preference scores")
    ap.add_argument("--image-score-external-root", default="",
                    dest="image_score_external_root",
                    help="base directory for relative external score sidecar paths")
    ap.add_argument("--image-score-external-key", default="image",
                    choices=("image", "basename", "caption"),
                    dest="image_score_external_key",
                    help="join key for --image-score-external-sidecar")
    ap.add_argument("--image-score-external-score-field", default="",
                    dest="image_score_external_score_field",
                    help=("explicit external score field; default searches common score "
                          "columns such as quality_score, reward_score, hps, pickscore"))
    ap.add_argument("--image-score-external-normalize", default="auto",
                    choices=("auto", "minmax", "none"),
                    dest="image_score_external_normalize",
                    help="normalization for external scores before writing quality_score")
    ap.add_argument("--image-score-out", default="runs/image_scored.jsonl",
                    dest="image_score_out",
                    help="scored manifest written by --image-score")
    ap.add_argument("--image-score-sidecar-out", default="",
                    dest="image_score_sidecar_out",
                    help="optional score-only sidecar written by --image-score")
    ap.add_argument("--image-score-report-out", default="runs/image_score_report.json",
                    dest="image_score_report_out",
                    help="JSON report written by --image-score")
    ap.add_argument("--image-score-drop-failed", action="store_true",
                    dest="image_score_drop_failed",
                    help="drop rows that cannot be scored instead of preserving them")
    ap.add_argument("--image-embed", action="store_true", dest="image_embed",
                    help="precompute text/image embedding sidecar and cleaned manifest on pod")
    ap.add_argument("--image-embed-backend", default="hf", choices=("stats", "hf"),
                    dest="image_embed_backend",
                    help="embedding backend for --image-embed")
    ap.add_argument("--image-embed-model", default="google/siglip-base-patch16-224",
                    dest="image_embed_model",
                    help="Hugging Face model id for --image-embed-backend hf")
    ap.add_argument("--image-embed-text-sequence-model", default="",
                    dest="image_embed_text_sequence_model",
                    help=("optional HF text encoder for token-level text_embedding_sequence "
                          "rows; primary --image-embed-model still supplies image and pooled "
                          "text embeddings"))
    ap.add_argument("--image-embed-text-sequence-max-length", type=int, default=0,
                    dest="image_embed_text_sequence_max_length",
                    help="optional tokenizer max_length for --image-embed-text-sequence-model")
    ap.add_argument("--image-embed-features", default="both", choices=("both", "image", "text"),
                    dest="image_embed_features",
                    help="which embedding columns to write before manifest training")
    ap.add_argument("--image-embed-text-mode", default="pooled",
                    choices=("pooled", "tokens", "both"), dest="image_embed_text_mode",
                    help=("write pooled text embeddings, token-level text embedding sequences, "
                          "or both"))
    ap.add_argument("--image-embed-batch", type=int, default=64,
                    dest="image_embed_batch",
                    help="batch size for image embedding preprocessing")
    ap.add_argument("--image-embed-device", default="cuda", dest="image_embed_device",
                    help="device used by image embedding preprocessing")
    ap.add_argument("--image-embed-dtype", default="auto",
                    choices=("auto", "fp32", "fp16", "bf16"), dest="image_embed_dtype",
                    help="dtype used by Hugging Face image embedding preprocessing")
    ap.add_argument("--image-embed-max-records", type=int, default=0,
                    dest="image_embed_max_records",
                    help="cap records for image embedding preprocessing; 0 means all")
    ap.add_argument("--image-embed-out", default="runs/image_embeddings.jsonl",
                    dest="image_embed_out",
                    help="sidecar JSONL written by --image-embed")
    ap.add_argument("--image-embed-report-out", default="runs/image_embed_report.json",
                    dest="image_embed_report_out",
                    help="JSON report written by --image-embed")
    ap.add_argument("--image-embed-no-normalize", action="store_true",
                    dest="image_embed_no_normalize",
                    help="do not L2-normalize image/text embedding rows")
    ap.add_argument("--image-embed-trust-remote-code", action="store_true",
                    dest="image_embed_trust_remote_code",
                    help="pass trust_remote_code=True to Hugging Face embedding loaders")
    ap.add_argument("--image-clean-manifest", default="runs/image_train_clean.jsonl",
                    dest="image_clean_manifest",
                    help="cleaned manifest written after embedding merge")
    ap.add_argument("--image-clean-report-out", default="runs/image_manifest_report.json",
                    dest="image_clean_report_out",
                    help="manifest QA report written after embedding merge")
    ap.add_argument("--image-clean-min-side", type=int, default=0,
                    dest="image_clean_min_side",
                    help="reject manifest images whose smaller side is below this size")
    ap.add_argument("--image-clean-max-aspect", type=float, default=0.0,
                    dest="image_clean_max_aspect",
                    help="reject manifest images wider/taller than this aspect ratio; 0 disables")
    ap.add_argument("--image-clean-max-nsfw", type=float, default=None,
                    dest="image_clean_max_nsfw",
                    help="reject rows with nsfw/image_nsfw/prompt_nsfw above this threshold")
    ap.add_argument("--image-clean-max-watermark", type=float, default=None,
                    dest="image_clean_max_watermark",
                    help="reject rows with watermark/watermark_score above this threshold")
    ap.add_argument("--image-clean-min-image-text-cosine", type=float, default=None,
                    dest="image_clean_min_image_text_cosine",
                    help=("after embedding merge, reject rows whose pooled text/image "
                          "embedding cosine is below this threshold"))
    ap.add_argument("--image-clean-max-image-duplicate-cosine", type=float, default=None,
                    dest="image_clean_max_image_duplicate_cosine",
                    help=("after embedding merge, reject near-duplicate image embeddings "
                          "with cosine at or above this threshold"))
    ap.add_argument("--image-clean-dedupe-lsh-bits", type=int, default=18,
                    dest="image_clean_dedupe_lsh_bits",
                    help="random-hyperplane LSH bits per table for image duplicate cleaning")
    ap.add_argument("--image-clean-dedupe-lsh-tables", type=int, default=4,
                    dest="image_clean_dedupe_lsh_tables",
                    help="number of LSH tables for image duplicate cleaning")
    ap.add_argument("--image-clean-dedupe-seed", type=int, default=0,
                    dest="image_clean_dedupe_seed",
                    help="deterministic LSH seed for image duplicate cleaning")
    ap.add_argument("--image-clean-dedupe-keep-first", action="store_true",
                    dest="image_clean_dedupe_keep_first",
                    help="keep first row among near duplicates instead of preferring quality score")
    ap.add_argument("--image-clean-min-caption-tokens", type=int, default=1,
                    dest="image_clean_min_caption_tokens",
                    help="reject captions shorter than this token count")
    ap.add_argument("--image-clean-max-caption-tokens", type=int, default=0,
                    dest="image_clean_max_caption_tokens",
                    help="reject captions longer than this token count; 0 disables")
    ap.add_argument("--image-clean-no-check-images", action="store_true",
                    dest="image_clean_no_check_images",
                    help="skip image header/decode checks during pod-side manifest cleaning")
    ap.add_argument("--image-clean-keep-duplicate-paths", action="store_true",
                    dest="image_clean_keep_duplicate_paths",
                    help="do not reject duplicate image paths during pod-side manifest cleaning")
    ap.add_argument("--image-size", default="32", dest="image_size",
                    help="image size as SIZE or HxW for latent image train/eval/sample grids")
    ap.add_argument("--image-size-buckets", default="", dest="image_size_buckets",
                    help="optional manifest train buckets, e.g. 128x128,128x192,192x128")
    ap.add_argument("--image-ae-accum-steps", type=int, default=1,
                    dest="image_ae_accum_steps",
                    help="AE gradient accumulation microsteps")
    ap.add_argument("--image-flow-accum-steps", type=int, default=1,
                    dest="image_flow_accum_steps",
                    help="flow gradient accumulation microsteps")
    ap.add_argument("--image-flow-cache-latents", action="store_true",
                    dest="image_flow_cache_latents",
                    help="precompute AE latents for manifest flow training")
    ap.add_argument("--image-flow-cache-records", type=int, default=0,
                    dest="image_flow_cache_records",
                    help="maximum manifest records cached for image flow training; 0 means all")
    ap.add_argument("--image-flow-cache-batch", type=int, default=64,
                    dest="image_flow_cache_batch",
                    help="batch size used while building the image latent cache")
    ap.add_argument("--image-flow-cache-dir", default="",
                    dest="image_flow_cache_dir",
                    help="directory for disk-backed image latent cache; implies cache")
    ap.add_argument("--image-flow-cache-shard-size", type=int, default=1024,
                    dest="image_flow_cache_shard_size",
                    help="records per shard for disk-backed image latent cache")
    ap.add_argument("--image-flow-cache-dtype", default="fp32",
                    choices=("fp32", "bf16", "fp16"), dest="image_flow_cache_dtype",
                    help="dtype used to store cached image AE latents")
    ap.add_argument("--image-flow-cache-max-loaded-shards", type=int, default=0,
                    dest="image_flow_cache_max_loaded_shards",
                    help="LRU disk-cache shards kept loaded in CPU RAM; 0 disables")
    ap.add_argument("--image-train-precision", default="fp32",
                    choices=("fp32", "bf16", "fp16"), dest="image_train_precision",
                    help="latent image training precision; bf16/fp16 AMP runs on CUDA")
    ap.add_argument("--image-grad-clip", type=float, default=0.0, dest="image_grad_clip",
                    help="clip latent image AE/flow gradient norm; 0 disables")
    ap.add_argument("--image-latent-arch", default="conv",
                    choices=("conv", "dit", "crossdit", "mmdit"),
                    dest="image_latent_arch", help="latent velocity architecture")
    ap.add_argument("--image-dit-head-width-mult", type=int, default=1,
                    dest="image_dit_head_width_mult",
                    help="width multiplier for the latent DiT/MM-DiT velocity head")
    ap.add_argument("--image-dit-qk-norm", action="store_true",
                    dest="image_dit_qk_norm",
                    help="enable per-head QK RMSNorm in latent image MM-DiT attention")
    ap.add_argument("--image-dit-attn-impl", default="manual",
                    choices=("manual", "sdpa", "linear", "auto"), dest="image_dit_attn_impl",
                    help="latent image MM-DiT attention implementation")
    ap.add_argument("--image-dit-pos-embed", default="learned",
                    choices=("learned", "sincos2d", "rope2d"), dest="image_dit_pos_embed",
                    help=("latent image DiT positional embedding; rope2d requires "
                          "--image-latent-arch mmdit"))
    ap.add_argument("--image-dit-mlp", default="gelu",
                    choices=("gelu", "swiglu"), dest="image_dit_mlp",
                    help="latent image CrossDiT/MM-DiT feed-forward block")
    ap.add_argument("--image-flow-checkpoint-blocks", action="store_true",
                    dest="image_flow_checkpoint_blocks",
                    help="checkpoint latent image transformer blocks during flow training")
    ap.add_argument("--image-ae-arch", default="semantic",
                    choices=("semantic", "residual", "hf-vae"),
                    dest="image_ae_arch",
                    help="autoencoder architecture for latent image generation")
    ap.add_argument("--image-ae-hf-model", default="",
                    dest="image_ae_hf_model",
                    help="Diffusers AutoencoderKL model id when --image-ae-arch hf-vae")
    ap.add_argument("--image-ae-hf-subfolder", default="",
                    dest="image_ae_hf_subfolder",
                    help="optional Hugging Face subfolder for --image-ae-hf-model")
    ap.add_argument("--image-ae-hf-scaling-factor", type=float, default=0.0,
                    dest="image_ae_hf_scaling_factor",
                    help="override HF VAE latent scaling factor; 0 uses model config")
    ap.add_argument("--image-latent-downsample", type=int, default=4,
                    dest="image_latent_downsample",
                    help="AE spatial compression factor for latent image generation")
    ap.add_argument("--image-latent-patch-size", type=int, default=1,
                    dest="image_latent_patch_size",
                    help="latent image transformer patch size")
    ap.add_argument("--image-ae-res-blocks", type=int, default=1,
                    dest="image_ae_res_blocks",
                    help="residual blocks per AE stage when --image-ae-arch residual")
    ap.add_argument("--image-latent-max-tokens", type=int, default=256,
                    dest="image_latent_max_tokens",
                    help="maximum latent grid tokens for image DiT/MM-DiT flows")
    ap.add_argument("--image-ae-recon-loss", default="mse", choices=("mse", "l1", "hybrid"),
                    dest="image_ae_recon_loss",
                    help="base reconstruction loss for image autoencoder")
    ap.add_argument("--image-ae-grad-w", type=float, default=0.0, dest="image_ae_grad_w",
                    help="edge/gradient reconstruction loss weight")
    ap.add_argument("--image-ae-ms-w", type=float, default=0.0, dest="image_ae_ms_w",
                    help="multi-scale reconstruction loss weight")
    ap.add_argument("--image-ae-fft-w", type=float, default=0.0, dest="image_ae_fft_w",
                    help="frequency-spectrum reconstruction loss weight")
    ap.add_argument("--image-ae-latent-reg-w", type=float, default=0.0,
                    dest="image_ae_latent_reg_w",
                    help="latent L2 regularization weight during AE training")
    ap.add_argument("--image-text-align-w", type=float, default=0.0,
                    dest="image_text_align_w",
                    help="contrastive image-latent/caption alignment weight during AE training")
    ap.add_argument("--image-flow-text-align-w", type=float, default=0.0,
                    dest="image_flow_text_align_w",
                    help="contrastive caption alignment weight on predicted flow endpoints")
    ap.add_argument("--image-text-embed-dim", type=int, default=128,
                    dest="image_text_embed_dim",
                    help="shared image/text embedding width for manifest alignment")
    ap.add_argument("--image-feature-align-w", type=float, default=0.0,
                    dest="image_feature_align_w",
                    help="contrastive AE-latent/external-image-feature alignment weight")
    ap.add_argument("--image-flow-feature-align-w", type=float, default=0.0,
                    dest="image_flow_feature_align_w",
                    help="contrastive external image-feature weight on predicted flow endpoints")
    ap.add_argument("--image-feature-embed-dim", type=int, default=128,
                    dest="image_feature_embed_dim",
                    help="shared embedding width for latent/image-feature alignment")
    ap.add_argument("--image-flow-repa-w", type=float, default=0.0,
                    dest="image_flow_repa_w",
                    help="REPA-style hidden-state/image-feature alignment weight")
    ap.add_argument("--image-flow-repa-steps", type=int, default=0,
                    dest="image_flow_repa_steps",
                    help="limit image REPA alignment to the first N flow steps; 0 means all steps")
    ap.add_argument("--image-flow-repa-embed-dim", type=int, default=128,
                    dest="image_flow_repa_embed_dim",
                    help="shared embedding width for image REPA alignment")
    ap.add_argument("--image-flow-repa-mode", default="pooled",
                    choices=("pooled", "token", "both", "auto"),
                    dest="image_flow_repa_mode",
                    help="image REPA target: pooled feature, token sequence, both, or auto")
    ap.add_argument("--image-flow-self-repa-w", type=float, default=0.0,
                    dest="image_flow_self_repa_w",
                    help=("no-extra-data hidden/clean-latent self-REPA weight "
                          "for image transformer flows"))
    ap.add_argument("--image-flow-self-repa-steps", type=int, default=0,
                    dest="image_flow_self_repa_steps",
                    help="limit image self-REPA to the first N flow steps; 0 means all steps")
    ap.add_argument("--image-flow-self-repa-embed-dim", type=int, default=128,
                    dest="image_flow_self_repa_embed_dim",
                    help="shared embedding width for image self-REPA")
    ap.add_argument("--image-flow-self-repa-mode", default="pooled",
                    choices=("pooled", "token", "both", "auto"),
                    dest="image_flow_self_repa_mode",
                    help="image self-REPA target: pooled clean latent, patch tokens, both, or auto")
    ap.add_argument("--image-flow-sra-w", type=float, default=0.0,
                    dest="image_flow_sra_w",
                    help=("image flow self-representation alignment weight: match noisy "
                          "hidden tokens to detached cleaner-step hidden tokens"))
    ap.add_argument("--image-flow-sra-steps", type=int, default=0,
                    dest="image_flow_sra_steps",
                    help="limit image flow SRA to the first N flow steps; 0 means all steps")
    ap.add_argument("--image-flow-sra-time-gap", type=float, default=0.25,
                    dest="image_flow_sra_time_gap",
                    help="fractional move toward clean data for the detached SRA teacher pass")
    ap.add_argument("--image-flow-sra-mode", default="token",
                    choices=("pooled", "token", "both", "auto"),
                    dest="image_flow_sra_mode",
                    help="image flow SRA target: pooled hidden state, patch tokens, both, or auto")
    ap.add_argument("--image-cond-mode", default="text", choices=("text",),
                    dest="image_cond_mode",
                    help="latent image conditioning source")
    ap.add_argument("--image-text-cond-dim", type=int, default=0, dest="image_text_cond_dim",
                    help="text condition vector width; default uses --dim/hidden")
    ap.add_argument("--image-manifest", default="", dest="image_manifest",
                    help="captioned image JSONL/CSV/TSV manifest for real image training")
    ap.add_argument("--image-root", default="", dest="image_root",
                    help="base directory for relative paths in --image-manifest")
    ap.add_argument("--upload-image-data", action="store_true",
                    dest="upload_image_data",
                    help="upload --image-root and --image-manifest to the pod before running")
    ap.add_argument("--image-split", default="train", dest="image_split",
                    help="manifest split used by latent image training")
    ap.add_argument("--image-min-aesthetic", type=float, default=None,
                    dest="image_min_aesthetic",
                    help="optional minimum aesthetic/quality score for image manifest rows")
    ap.add_argument("--image-quality-weight", type=float, default=0.0,
                    dest="image_quality_weight",
                    help=("sample image manifest rows by normalized aesthetic/score/quality "
                          "metadata; 0 keeps uniform sampling"))
    ap.add_argument("--image-source-weights", default="",
                    dest="image_source_weights",
                    help=("comma-separated source=weight data-mixture controls for image "
                          "manifest rows; '*' sets the default source weight"))
    ap.add_argument("--image-quality-score-w", type=float, default=0.0,
                    dest="image_quality_score_w",
                    help="train a latent aesthetic/quality score head from image manifest metadata")
    ap.add_argument("--image-flow-quality-score-w", type=float, default=0.0,
                    dest="image_flow_quality_score_w",
                    help=("preserve learned quality score on latent flow endpoints; "
                          "requires --image-quality-score-w or --image-quality-score-steps"))
    ap.add_argument("--image-quality-score-steps", type=int, default=0,
                    dest="image_quality_score_steps",
                    help=("standalone latent quality-scorer pretrain steps from cached "
                          "or encoded manifest latents"))
    ap.add_argument("--image-quality-rank-w", type=float, default=0.0,
                    dest="image_quality_rank_w",
                    help="AE-phase pairwise quality preference ranking loss weight")
    ap.add_argument("--image-quality-score-rank-w", type=float, default=0.0,
                    dest="image_quality_score_rank_w",
                    help="standalone scorer pretrain pairwise quality ranking loss weight")
    ap.add_argument("--image-flow-quality-rank-w", type=float, default=0.0,
                    dest="image_flow_quality_rank_w",
                    help="preserve batch quality preference ordering on flow endpoints")
    ap.add_argument("--image-quality-rank-margin", type=float, default=0.0,
                    dest="image_quality_rank_margin",
                    help="minimum scorer-logit margin for quality preference ranking")
    ap.add_argument("--image-preference-manifest", default="",
                    dest="image_preference_manifest",
                    help=("JSONL chosen/rejected image-pair manifest consumed by latent "
                          "quality-scorer pretraining"))
    ap.add_argument("--image-preference-root", default="",
                    dest="image_preference_root",
                    help="base directory for relative paths in --image-preference-manifest")
    ap.add_argument("--image-preference-max-pairs", type=int, default=0,
                    dest="image_preference_max_pairs",
                    help="cap image preference pairs loaded for scorer pretraining; 0 means all")
    ap.add_argument("--image-preference-w", type=float, default=0.0,
                    dest="image_preference_w",
                    help=("chosen/rejected preference loss weight inside "
                          "--image-quality-score-steps"))
    ap.add_argument("--image-max-records", type=int, default=0, dest="image_max_records",
                    help="cap image manifest records for GPU smoke tests; 0 means all")
    ap.add_argument("--image-eval-split", default="eval", dest="image_eval_split",
                    help="manifest split used by --image-eval-sweep for real image eval")
    ap.add_argument("--image-eval-max-records", type=int, default=0,
                    dest="image_eval_max_records",
                    help="cap image eval manifest records for GPU smoke tests; 0 means all")
    ap.add_argument("--image-eval-generated-samples", type=int, default=0,
                    dest="image_eval_generated_samples",
                    help=("generated manifest samples per eval row; 0 uses "
                          "min(batch,sample_steps)"))
    ap.add_argument("--image-eval-generated-candidates-per-prompt", type=int, default=1,
                    dest="image_eval_generated_candidates_per_prompt",
                    help=("manifest eval candidates drawn per caption before learned "
                          "aligner/quality reranking"))
    ap.add_argument("--image-caption-vocab-max", type=int, default=8192,
                    dest="image_caption_vocab_max",
                    help="maximum caption vocabulary size for image manifest training")
    ap.add_argument("--image-caption-max-len", type=int, default=64,
                    dest="image_caption_max_len",
                    help="maximum caption tokens for image manifest training")
    ap.add_argument("--image-caption-cond-source", default="tokens",
                    choices=("tokens", "embedding", "auto"),
                    dest="image_caption_cond_source",
                    help="caption conditioning source for image manifests")
    ap.add_argument("--image-crop-mode", default="center",
                    choices=("center", "random", "none", "pad"), dest="image_crop_mode",
                    help="crop mode for manifest image training")
    ap.add_argument("--image-hflip-prob", type=float, default=0.0,
                    dest="image_hflip_prob",
                    help="random horizontal flip probability for manifest image training")
    ap.add_argument("--image-prompt-templates", default="", dest="image_prompt_templates",
                    help="internal fixture prompt templates; production runs use captions/prompts")
    ap.add_argument("--image-cond-drop", type=float, default=0.0, dest="image_cond_drop",
                    help="condition dropout for classifier-free latent image guidance")
    ap.add_argument("--image-cfg-scale", type=float, default=1.0, dest="image_cfg_scale",
                    help="classifier-free guidance scale for latent image sampling")
    ap.add_argument("--image-cfg-rescale", type=float, default=0.0,
                    dest="image_cfg_rescale",
                    help="latent image CFG rescale; 0 disables")
    ap.add_argument("--image-cfg-mode", default="standard",
                    choices=("standard", "cfgpp"), dest="image_cfg_mode",
                    help="latent image classifier-free guidance variant")
    ap.add_argument("--image-cfg-interval", default="0.0,1.0", dest="image_cfg_interval",
                    help="latent image CFG active interval formatted start,end")
    ap.add_argument("--image-sample-steps", type=int, default=4, dest="image_sample_steps",
                    help="ODE sampling steps for latent image evaluation")
    ap.add_argument("--image-sample-method", default="euler",
                    choices=("euler", "heun", "midpoint", "rk4"),
                    dest="image_sample_method",
                    help="latent image ODE sampler method")
    ap.add_argument("--image-sample-methods", default="euler,heun,midpoint",
                    dest="image_sample_methods",
                    help="comma-separated latent image sampler methods for sweeps")
    ap.add_argument("--image-sample-schedule", default="linear",
                    choices=("linear", "quadratic", "sqrt", "cosine"),
                    dest="image_sample_schedule",
                    help="latent image timestep placement schedule")
    ap.add_argument("--image-sample-schedules", default="linear",
                    dest="image_sample_schedules",
                    help="comma-separated latent image timestep schedules for sweeps")
    ap.add_argument("--image-sample-churn", type=float, default=0.0,
                    dest="image_sample_churn",
                    help="stochastic latent sampler churn; 0 keeps deterministic ODE sampling")
    ap.add_argument("--image-sample-churns", default="0.0",
                    dest="image_sample_churns",
                    help="comma-separated stochastic sampler churn values for sweeps")
    ap.add_argument("--image-sample-churn-interval", default="0.0,0.8",
                    dest="image_sample_churn_interval",
                    help="latent sampler churn active interval formatted start,end")
    ap.add_argument("--image-sample-finite-guard", action="store_true",
                    dest="image_sample_finite_guard",
                    help="replace non-finite latent/velocity sampler values with zeros")
    ap.add_argument("--image-sample-velocity-clip", type=float, default=0.0,
                    dest="image_sample_velocity_clip",
                    help="per-sample latent velocity RMS clip during sampling; 0 disables")
    ap.add_argument("--image-sample-latent-clip", type=float, default=0.0,
                    dest="image_sample_latent_clip",
                    help="absolute latent clamp during sampling; 0 disables")
    ap.add_argument("--image-sample-grid", action="store_true",
                    dest="image_sample_grid",
                    help="save a generated prompt/caption PPM grid for latent image jobs")
    ap.add_argument("--image-sample-grid-samples", type=int, default=1,
                    dest="image_sample_grid_samples",
                    help="generated samples per color/shape condition in the image PPM grid")
    ap.add_argument("--image-sample-manifest-out", default="",
                    dest="image_sample_manifest_out",
                    help=("optional JSONL manifest path for individual generated image "
                          "samples from latent image sample grids"))
    ap.add_argument("--image-sample-image-dir", default="",
                    dest="image_sample_image_dir",
                    help=("optional directory for individual generated PPM samples; "
                          "default is derived from --image-sample-manifest-out"))
    ap.add_argument("--image-eval-generated", action="store_true",
                    dest="image_eval_generated",
                    help=("after a latent image job writes --image-sample-manifest-out, "
                          "embed generated samples and compare them to the reference manifest"))
    ap.add_argument("--image-eval-generated-real-manifest", default="",
                    dest="image_eval_generated_real_manifest",
                    help=("optional reference manifest for --image-eval-generated; "
                          "default uses the cleaned/effective image manifest"))
    ap.add_argument("--image-generated-embed-out", default="",
                    dest="image_generated_embed_out",
                    help=("optional embedding sidecar for generated sample manifest; "
                          "default derives from --image-sample-manifest-out"))
    ap.add_argument("--image-generated-embed-report-out", default="",
                    dest="image_generated_embed_report_out",
                    help=("optional embedding report for generated sample manifest; "
                          "default derives from --image-sample-manifest-out"))
    ap.add_argument("--image-generated-eval-report-out", default="",
                    dest="image_generated_eval_report_out",
                    help=("optional image_eval report for generated sample manifest; "
                          "default derives from --image-sample-manifest-out"))
    ap.add_argument("--image-generated-eval-max-records", type=int, default=2048,
                    dest="image_generated_eval_max_records",
                    help="maximum real/generated records for generated-sample image_eval")
    ap.add_argument("--image-generated-eval-min-score", type=float, default=0.0,
                    dest="image_generated_eval_min_score",
                    help="minimum composite generated-sample image_eval_score")
    ap.add_argument("--image-generated-eval-min-support-precision", type=float, default=0.0,
                    dest="image_generated_eval_min_support_precision",
                    help="minimum generated support precision for generated-sample image_eval")
    ap.add_argument("--image-generated-eval-min-support-recall", type=float, default=0.0,
                    dest="image_generated_eval_min_support_recall",
                    help="minimum generated support recall for generated-sample image_eval")
    ap.add_argument("--image-generated-eval-min-image-text-cos", type=float, default=None,
                    dest="image_generated_eval_min_image_text_cos",
                    help="minimum generated image/text cosine for generated-sample image_eval")
    ap.add_argument("--image-generated-eval-max-frechet", type=float, default=None,
                    dest="image_generated_eval_max_frechet",
                    help="maximum embedding Fréchet distance for generated-sample image_eval")
    ap.add_argument("--image-generated-eval-max-mmd-rbf", type=float, default=None,
                    dest="image_generated_eval_max_mmd_rbf",
                    help="maximum RBF-MMD distance for generated-sample image_eval")
    ap.add_argument("--image-generated-eval-fail-on-gate", action="store_true",
                    dest="image_generated_eval_fail_on_gate",
                    help="fail the RunPod job if configured generated-sample eval gates fail")
    ap.add_argument("--image-sample-prompts", default="", dest="image_sample_prompts",
                    help="semicolon/newline separated prompts for latent image sample grids")
    ap.add_argument("--image-sample-negative-prompts", default="",
                    dest="image_sample_negative_prompts",
                    help=("semicolon/newline separated negative prompts for latent image "
                          "prompt sample grids"))
    ap.add_argument("--image-sample-candidates-per-prompt", type=int, default=1,
                    dest="image_sample_candidates_per_prompt",
                    help=("number of candidates to draw per latent image sample prompt; "
                          "compatible aligners/quality scorers select the best candidate"))
    ap.add_argument("--image-sample-text-guidance-w", type=float, default=0.0,
                    dest="image_sample_text_guidance_w",
                    help=("sampling-time text-image alignment guidance weight for latent "
                          "image sample prompts"))
    ap.add_argument("--image-sample-text-guidance-interval", default="0.0,1.0",
                    dest="image_sample_text_guidance_interval",
                    help=("sample-prompt text guidance active interval over flow time, "
                          "formatted start,end"))
    ap.add_argument("--image-sample-feature-guidance-w", type=float, default=0.0,
                    dest="image_sample_feature_guidance_w",
                    help=("sampling-time external image-feature guidance weight for latent "
                          "image sample prompts"))
    ap.add_argument("--image-sample-feature-guidance-interval", default="0.0,1.0",
                    dest="image_sample_feature_guidance_interval",
                    help=("sample-prompt external feature guidance active interval over flow "
                          "time, formatted start,end"))
    ap.add_argument("--image-sample-quality-guidance-w", type=float, default=0.0,
                    dest="image_sample_quality_guidance_w",
                    help="sampling-time manifest quality guidance weight for latent image prompts")
    ap.add_argument("--image-sample-quality-guidance-interval", default="0.0,1.0",
                    dest="image_sample_quality_guidance_interval",
                    help=("sample-prompt quality guidance active interval over flow time, "
                          "formatted start,end"))
    ap.add_argument("--image-prompt-embed-backend", default="stats",
                    choices=("stats", "hf"), dest="image_prompt_embed_backend",
                    help="text embedding backend for --image-sample-prompts on embedding checkpoints")
    ap.add_argument("--image-prompt-embed-model", default="",
                    dest="image_prompt_embed_model",
                    help="Hugging Face model id for --image-prompt-embed-backend hf")
    ap.add_argument("--image-prompt-embed-text-sequence-model", default="",
                    dest="image_prompt_embed_text_sequence_model",
                    help=("optional HF text encoder for token-level live prompt conditioning; "
                          "should match --image-embed-text-sequence-model for embedding runs"))
    ap.add_argument("--image-prompt-embed-text-sequence-max-length", type=int, default=0,
                    dest="image_prompt_embed_text_sequence_max_length",
                    help="optional tokenizer max_length for --image-prompt-embed-text-sequence-model")
    ap.add_argument("--image-prompt-embed-device", default="cuda",
                    dest="image_prompt_embed_device",
                    help="device for live sample-prompt embedding")
    ap.add_argument("--image-prompt-embed-dtype", default="auto",
                    choices=("auto", "fp32", "fp16", "bf16"),
                    dest="image_prompt_embed_dtype")
    ap.add_argument("--image-prompt-embed-stats-dim", type=int, default=0,
                    dest="image_prompt_embed_stats_dim",
                    help="stats backend prompt embedding width; 0 uses checkpoint input dim")
    ap.add_argument("--image-prompt-embed-no-normalize", action="store_true",
                    dest="image_prompt_embed_no_normalize",
                    help="do not L2-normalize live sample-prompt embeddings")
    ap.add_argument("--image-prompt-embed-trust-remote-code", action="store_true",
                    dest="image_prompt_embed_trust_remote_code",
                    help="pass trust_remote_code=True to Hugging Face prompt embedder")
    ap.add_argument("--image-semantic-guidance-w", type=float, default=0.0,
                    dest="image_semantic_guidance_w",
                    help="sampling-time semantic AE guidance weight for latent images")
    ap.add_argument("--image-semantic-guidance-sweep", default="0.0,1.0,2.0",
                    dest="image_semantic_guidance_sweep",
                    help="comma-separated semantic guidance weights for latent image sweeps")
    ap.add_argument("--image-eval-text-guidance-sweep", default="0.0",
                    dest="image_eval_text_guidance_sweep",
                    help="comma-separated text guidance weights for manifest image eval sweeps")
    ap.add_argument("--image-eval-feature-guidance-sweep", default="0.0",
                    dest="image_eval_feature_guidance_sweep",
                    help="comma-separated feature guidance weights for manifest image eval sweeps")
    ap.add_argument("--image-eval-quality-guidance-sweep", default="0.0",
                    dest="image_eval_quality_guidance_sweep",
                    help="comma-separated quality guidance weights for manifest image eval sweeps")
    ap.add_argument("--image-semantic-guidance-mode", default="decoded",
                    choices=("latent", "decoded"), dest="image_semantic_guidance_mode",
                    help="latent image semantic guidance mode")
    ap.add_argument("--image-semantic-guidance-interval", default="0.0,1.0",
                    dest="image_semantic_guidance_interval",
                    help="semantic guidance active interval formatted start,end")
    ap.add_argument("--image-roundtrip-samples", type=int, default=1,
                    dest="image_roundtrip_samples",
                    help="generated samples per color/shape condition during image eval")
    ap.add_argument("--image-intervention-samples", type=int, default=32,
                    dest="image_intervention_samples",
                    help="samples for latent image fact-intervention diagnostics")
    ap.add_argument("--image-ae-intervention-w", type=float, default=0.0,
                    dest="image_ae_intervention_w",
                    help="semantic AE latent fact-intervention loss weight")
    ap.add_argument("--image-ae-factor-orth-w", type=float, default=0.0,
                    dest="image_ae_factor_orth_w",
                    help="semantic AE cross-factor latent orthogonality loss weight")
    ap.add_argument("--image-flow-semantic-w", type=float, default=0.0,
                    dest="image_flow_semantic_w",
                    help="semantic endpoint alignment weight for latent image flow training")
    ap.add_argument("--image-flow-consistency-w", type=float, default=0.0,
                    dest="image_flow_consistency_w",
                    help="same-path endpoint consistency loss weight for latent image flow")
    ap.add_argument("--image-flow-endpoint-w", type=float, default=0.0,
                    dest="image_flow_endpoint_w",
                    help="direct clean-endpoint latent prediction loss weight for image flow")
    ap.add_argument("--image-flow-noise-coupling", default="random",
                    choices=("random", "sliced_ot"), dest="image_flow_noise_coupling",
                    help="source-noise/data pairing for image flow matching")
    ap.add_argument("--image-flow-noise-coupling-projections", type=int, default=1,
                    dest="image_flow_noise_coupling_projections",
                    help="random projections to try for sliced_ot image flow noise coupling")
    ap.add_argument("--image-flow-distill-steps", type=int, default=0,
                    dest="image_flow_distill_steps",
                    help="post-flow own-model endpoint distillation steps")
    ap.add_argument("--image-flow-distill-w", type=float, default=1.0,
                    dest="image_flow_distill_w",
                    help="loss weight for own-model endpoint distillation")
    ap.add_argument("--image-flow-distill-time-gap", type=float, default=0.25,
                    dest="image_flow_distill_time_gap",
                    help="fractional move toward clean data time for the frozen teacher")
    ap.add_argument("--image-flow-distill-teacher", default="auto",
                    choices=("raw", "ema", "auto"), dest="image_flow_distill_teacher",
                    help="teacher snapshot for own-model endpoint distillation")
    ap.add_argument("--image-flow-guidance-distill-w", type=float, default=0.0,
                    dest="image_flow_guidance_distill_w",
                    help="inside image flow distillation, match a CFG-guided teacher endpoint")
    ap.add_argument("--image-flow-guidance-distill-cfg-scale", type=float, default=1.5,
                    dest="image_flow_guidance_distill_cfg_scale",
                    help="CFG scale for image flow guidance distillation teacher")
    ap.add_argument("--image-flow-guidance-distill-cfg-rescale", type=float, default=0.0,
                    dest="image_flow_guidance_distill_cfg_rescale",
                    help="CFG rescale for image flow guidance distillation teacher")
    ap.add_argument("--image-flow-ema-decay", type=float, default=0.0,
                    dest="image_flow_ema_decay",
                    help="EMA decay for latent image flow/conditioner weights")
    ap.add_argument("--image-ema-eval-mode", default="auto",
                    choices=("raw", "ema", "auto"), dest="image_ema_eval_mode",
                    help="latent image train-report weight mode")
    ap.add_argument("--image-no-ema-warmup", action="store_true",
                    dest="image_no_ema_warmup",
                    help="use exact image EMA decay from the first update")
    ap.add_argument("--image-time-sampling", default="uniform",
                    choices=("uniform", "logit-normal"), dest="image_time_sampling",
                    help="latent image flow timestep distribution")
    ap.add_argument("--image-time-logit-mean", type=float, default=0.0,
                    dest="image_time_logit_mean",
                    help="mean for --image-time-sampling logit-normal")
    ap.add_argument("--image-time-logit-std", type=float, default=1.0,
                    dest="image_time_logit_std",
                    help="stddev for --image-time-sampling logit-normal")
    ap.add_argument("--image-time-curriculum-frac", type=float, default=0.0,
                    dest="image_time_curriculum_frac",
                    help=("fraction of image flow training using --image-time-sampling before "
                          "switching timestep sampling to uniform; 0 disables"))
    ap.add_argument("--image-time-shift", type=float, default=1.0,
                    dest="image_time_shift",
                    help="latent image RF data-time shift; >1 biases training toward noise")
    ap.add_argument("--image-time-shift-mode", default="manual",
                    choices=("manual", "dim"), dest="image_time_shift_mode",
                    help="manual uses --image-time-shift as-is; dim scales it by latent dimension")
    ap.add_argument("--image-time-shift-ref-dim", type=float, default=1024.0,
                    dest="image_time_shift_ref_dim",
                    help="reference latent element count for dimension-aware image time shift")
    ap.add_argument("--image-time-shift-dim-power", type=float, default=0.5,
                    dest="image_time_shift_dim_power",
                    help="power used for dimension-aware image time-shift scaling")
    ap.add_argument("--image-flow-loss-weight", default="none",
                    choices=("none", "min-snr-v", "soft-min-snr-v"),
                    dest="image_flow_loss_weight",
                    help="latent image per-timestep velocity loss weighting")
    ap.add_argument("--image-flow-loss-weight-gamma", type=float, default=5.0,
                    dest="image_flow_loss_weight_gamma",
                    help="gamma for latent image Min-SNR-style velocity weighting")
    ap.add_argument("--image-no-flow-loss-weight-normalize", action="store_true",
                    dest="image_no_flow_loss_weight_normalize",
                    help="do not normalize weighted latent image velocity loss to batch mean 1")
    ap.add_argument("--image-latent-normalize", default="auto",
                    choices=("none", "global", "channel", "auto"),
                    dest="image_latent_normalize",
                    help=("normalize image latents before latent image flow training; auto uses "
                          "channel stats for real-image/external-VAE runs"))
    ap.add_argument("--image-latent-stat-samples", type=int, default=512,
                    dest="image_latent_stat_samples",
                    help="samples used to estimate latent image normalization stats")
    ap.add_argument("--image-eval-sweep", action="store_true", dest="image_eval_sweep",
                    help="after latent training, sweep CFG and sampler steps from the checkpoint")
    ap.add_argument("--image-checkpoint-weight-mode", default="auto",
                    choices=("raw", "ema", "auto"), dest="image_checkpoint_weight_mode",
                    help="latent image checkpoint sweep weight mode")
    ap.add_argument("--image-cfg-sweep", default="1.0,1.25,1.5,2.0",
                    dest="image_cfg_sweep",
                    help="comma-separated CFG scales for --image-eval-sweep")
    ap.add_argument("--image-cfg-rescale-sweep", default="0.0,0.7",
                    dest="image_cfg_rescale_sweep",
                    help="comma-separated latent image CFG rescale values for sweeps")
    ap.add_argument("--image-cfg-modes", default="",
                    dest="image_cfg_modes",
                    help=("comma-separated latent image CFG modes for sweeps; "
                          "default uses --image-cfg-mode"))
    ap.add_argument("--image-sample-steps-sweep", default="4,8,16",
                    dest="image_sample_steps_sweep",
                    help="comma-separated sampler step counts for --image-eval-sweep")
    ap.add_argument("--image-eval-seeds", default="1,2,3", dest="image_eval_seeds",
                    help="comma-separated eval seeds for --image-eval-sweep")
    ap.add_argument("--audio", action="store_true",
                    help="AUDIO-1: train synthetic audio factor FER experiment")
    ap.add_argument("--multimodal", action="store_true",
                    help="M-0: train one prefix-conditioned LM on image+audio extraction")
    ap.add_argument("--multimodal-dim", type=int, default=0, dest="multimodal_dim",
                    help="M-0 decoder width; default uses --dim or 96")
    ap.add_argument("--multimodal-layers", type=int, default=3, dest="multimodal_layers")
    ap.add_argument("--multimodal-heads", type=int, default=4, dest="multimodal_heads")
    ap.add_argument("--multimodal-batch", type=int, default=32, dest="multimodal_batch")
    ap.add_argument("--multimodal-lr", type=float, default=1e-3, dest="multimodal_lr")
    ap.add_argument("--multimodal-log-every", type=int, default=100,
                    dest="multimodal_log_every")
    ap.add_argument("--multimodal-value-w", type=float, default=6.0,
                    dest="multimodal_value_w")
    ap.add_argument("--multimodal-agreement-w", type=float, default=0.0,
                    dest="multimodal_agreement_w",
                    help="M-0 cross-mode factor-value agreement loss weight")
    ap.add_argument("--multimodal-concept-tokens", type=int, default=4,
                    dest="multimodal_concept_tokens")
    ap.add_argument("--multimodal-fusion-layers", type=int, default=1,
                    dest="multimodal_fusion_layers")
    ap.add_argument("--multimodal-concept-prefix", action="store_true",
                    dest="multimodal_concept_prefix",
                    help="prepend M-0 schema concept states to the decoder prefix")
    ap.add_argument("--multimodal-concept-w", type=float, default=0.0,
                    dest="multimodal_concept_w",
                    help="M-0 upstream concept-token factor loss weight")
    ap.add_argument("--multimodal-concept-agreement-w", type=float, default=0.0,
                    dest="multimodal_concept_agreement_w",
                    help="M-0 cross-mode upstream concept agreement weight")
    ap.add_argument("--multimodal-concept-distill-w", type=float, default=0.0,
                    dest="multimodal_concept_distill_w",
                    help="M-0 full-to-partial upstream concept distillation weight")
    ap.add_argument("--multimodal-concept-distill-temperature", type=float, default=1.0,
                    dest="multimodal_concept_distill_temperature")
    ap.add_argument("--multimodal-concept-rank-distill-w", type=float, default=0.0,
                    dest="multimodal_concept_rank_distill_w",
                    help="M-0 full-correct concept rank distillation weight")
    ap.add_argument("--multimodal-concept-rank-distill-margin", type=float, default=0.0,
                    dest="multimodal_concept_rank_distill_margin",
                    help="minimum margin for M-0 full-to-partial concept rank distillation")
    ap.add_argument("--multimodal-concept-transfer-w", type=float, default=0.0,
                    dest="multimodal_concept_transfer_w",
                    help="M-0 correct-detached upstream concept vector transfer weight")
    ap.add_argument("--multimodal-concept-transfer-margin", type=float, default=0.0,
                    dest="multimodal_concept_transfer_margin",
                    help="minimum margin for M-0 upstream concept vector transfer")
    ap.add_argument("--multimodal-concept-contrast-w", type=float, default=0.0,
                    dest="multimodal_concept_contrast_w",
                    help="M-0 same-value concept geometry contrastive loss weight")
    ap.add_argument("--multimodal-concept-contrast-temperature", type=float, default=0.1,
                    dest="multimodal_concept_contrast_temperature",
                    help="temperature for M-0 same-value concept geometry contrastive loss")
    ap.add_argument("--multimodal-concept-centroid-w", type=float, default=0.0,
                    dest="multimodal_concept_centroid_w",
                    help="M-0 batch-discovered concept centroid loss weight")
    ap.add_argument("--multimodal-concept-centroid-temperature", type=float, default=0.1,
                    dest="multimodal_concept_centroid_temperature",
                    help="temperature for M-0 concept centroid loss")
    ap.add_argument("--multimodal-concept-centroid-margin", type=float, default=0.0,
                    dest="multimodal_concept_centroid_margin",
                    help="minimum margin for M-0 concept centroid loss")
    ap.add_argument("--multimodal-concept-prototype-w", type=float, default=0.0,
                    dest="multimodal_concept_prototype_w",
                    help="M-0 concept-geometry prototype classification weight")
    ap.add_argument("--multimodal-concept-prototype-spread-w", type=float, default=0.0,
                    dest="multimodal_concept_prototype_spread_w",
                    help="M-0 same-key concept prototype anti-collapse weight")
    ap.add_argument("--multimodal-concept-prototype-spread-margin", type=float, default=0.2,
                    dest="multimodal_concept_prototype_spread_margin",
                    help="M-0 same-key prototype cosine spread margin")
    ap.add_argument("--multimodal-concept-state-spread-w", type=float, default=0.0,
                    dest="multimodal_concept_state_spread_w",
                    help="M-0 concept geometry state anti-collapse loss weight")
    ap.add_argument("--multimodal-concept-state-spread-variance", type=float, default=0.05,
                    dest="multimodal_concept_state_spread_variance",
                    help="minimum M-0 normalized concept-state std target")
    ap.add_argument("--multimodal-concept-state-spread-margin", type=float, default=0.2,
                    dest="multimodal_concept_state_spread_margin",
                    help="maximum M-0 observed-value centroid cosine before spread penalty")
    ap.add_argument("--multimodal-concept-state-spread-covariance-w", type=float,
                    default=0.05, dest="multimodal_concept_state_spread_covariance_w",
                    help="relative M-0 decorrelation weight inside state spread loss")
    ap.add_argument("--multimodal-latent-concept-slots", type=int, default=0,
                    dest="multimodal_latent_concept_slots",
                    help="schema-free M-0 latent concept slots aligned across modality views")
    ap.add_argument("--multimodal-latent-concept-layers", type=int, default=1,
                    dest="multimodal_latent_concept_layers",
                    help="self-attention layers inside M-0 latent concept slots")
    ap.add_argument("--multimodal-latent-concept-w", type=float, default=0.0,
                    dest="multimodal_latent_concept_w",
                    help="M-0 schema-free latent concept loss weight")
    ap.add_argument("--multimodal-latent-concept-view-dropout", type=float, default=0.1,
                    dest="multimodal_latent_concept_view_dropout",
                    help="feature dropout for M-0 latent concept views")
    ap.add_argument("--multimodal-latent-concept-invariance-w", type=float, default=25.0,
                    dest="multimodal_latent_concept_invariance_w",
                    help="M-0 latent concept invariance term weight")
    ap.add_argument("--multimodal-latent-concept-variance-w", type=float, default=25.0,
                    dest="multimodal_latent_concept_variance_w",
                    help="M-0 latent concept anti-collapse variance term weight")
    ap.add_argument("--multimodal-latent-concept-covariance-w", type=float, default=1.0,
                    dest="multimodal_latent_concept_covariance_w",
                    help="M-0 latent concept decorrelation term weight")
    ap.add_argument("--multimodal-latent-concept-variance-target", type=float, default=1.0,
                    dest="multimodal_latent_concept_variance_target",
                    help="minimum M-0 latent concept per-dimension std target")
    ap.add_argument("--multimodal-img-tokens", type=int, default=4,
                    dest="multimodal_img_tokens")
    ap.add_argument("--multimodal-aud-tokens", type=int, default=8,
                    dest="multimodal_aud_tokens")
    ap.add_argument("--multimodal-txt-tokens", type=int, default=8,
                    dest="multimodal_txt_tokens")
    ap.add_argument("--multimodal-trunk-arch", default="conv", choices=("conv", "residual"),
                    dest="multimodal_trunk_arch",
                    help="M-0 sensory reader trunk architecture")
    ap.add_argument("--multimodal-trunk-width", type=int, default=64,
                    dest="multimodal_trunk_width")
    ap.add_argument("--multimodal-trunk-depth", type=int, default=1,
                    dest="multimodal_trunk_depth")
    ap.add_argument("--multimodal-text-layers", type=int, default=1,
                    dest="multimodal_text_layers")
    ap.add_argument("--multimodal-dropout", type=float, default=0.0,
                    dest="multimodal_dropout",
                    help="drop full-mode modality prefixes during M-0 training")
    ap.add_argument("--multimodal-eval-n", type=int, default=200, dest="multimodal_eval_n")
    ap.add_argument("--multimodal-free-n", type=int, default=40, dest="multimodal_free_n")
    ap.add_argument("--multimodal-counterfactual-n", type=int, default=40,
                    dest="multimodal_counterfactual_n")
    ap.add_argument("--multimodal-free-counterfactual-n", type=int, default=20,
                    dest="multimodal_free_counterfactual_n")
    ap.add_argument("--multimodal-surfaces", default="", dest="multimodal_surfaces",
                    help="optional transcript surface JSON path passed to thinking.multimodal")
    ap.add_argument("--lengen", action="store_true", help="rung L: depth generalization")
    ap.add_argument("--deep-ancestor-rule-aux", action="store_true",
                    help="train the forward ancestor run with rule/action and contrastive losses")
    ap.add_argument("--run-name", default="runs/deep_ancestor_rule_aux")
    ap.add_argument("--eval-only-run", default="",
                    help="skip training; upload this local run dir and run deep-eval only")
    ap.add_argument("--eval-decodes", default="sample,hybrid",
                    help="comma-separated deep-eval decoders for --eval-only-run")
    ap.add_argument("--learning-curve", action="store_true",
                    help="train fresh rule-aux/no-aux runs at several step budgets")
    ap.add_argument("--curve-steps", default="1000,2000,4000",
                    help="comma-separated train step budgets for --learning-curve")
    ap.add_argument("--curve-arms", default="aux,noaux",
                    help="comma-separated arms for --learning-curve: aux,noaux")
    ap.add_argument("--examples", type=int, default=6000)
    ap.add_argument("--deep-frac", type=float, default=0.6, dest="deep_frac")
    ap.add_argument("--rule-w", type=float, default=0.1, dest="rule_w")
    ap.add_argument("--rule-contrast-w", type=float, default=0.05, dest="rule_contrast_w")
    ap.add_argument("--trace-rank-w", type=float, default=0.0, dest="trace_rank_w",
                    help="next verifier-action ranking loss weight")
    ap.add_argument("--trace-rank-batch", type=int, default=0, dest="trace_rank_batch",
                    help="ranking states per optimizer step")
    ap.add_argument("--trace-rank-candidates", type=int, default=0,
                    dest="trace_rank_candidates", help="candidate cap for rank loss/decode")
    ap.add_argument("--trace-rank-states", type=int, default=0, dest="trace_rank_states",
                    help="max support/on-policy steps before a rank target")
    ap.add_argument("--trace-dagger-frac", type=float, default=None, dest="trace_dagger_frac",
                    help="fraction of rank states reached by model-ranked rollout")
    ap.add_argument("--eval-depths", default="4,8,16,30,64",
                    help="comma-separated depths for deep-eval/probe in rule-aux mode")
    ap.add_argument("--eval-n", type=int, default=20)
    ap.add_argument("--probe-n", type=int, default=4)
    ap.add_argument("--sweep", action="store_true", help="also run the chain-world grid")
    ap.add_argument("--max-minutes", type=int, default=150)
    ap.add_argument("--go", action="store_true", help="actually create the pod (spends money)")
    args = ap.parse_args()
    if isinstance(args.eval_depths, str):
        args.eval_depths = [int(x.strip()) for x in args.eval_depths.split(",") if x.strip()]
    if isinstance(args.eval_decodes, str):
        args.eval_decodes = [x.strip() for x in args.eval_decodes.split(",") if x.strip()]
    if isinstance(args.curve_steps, str):
        args.curve_steps = [int(x.strip()) for x in args.curve_steps.split(",") if x.strip()]
    if isinstance(args.curve_arms, str):
        args.curve_arms = [x.strip() for x in args.curve_arms.split(",") if x.strip()]
    try:
        apply_image_quality_preset(args)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    bad_decodes = sorted(set(args.eval_decodes) - {"sample", "hybrid", "constrained", "ranker"})
    if bad_decodes:
        sys.exit(f"ERROR: unsupported --eval-decodes values: {','.join(bad_decodes)}")
    bad_arms = sorted(set(args.curve_arms) - {"aux", "noaux"})
    if bad_arms:
        sys.exit(f"ERROR: unsupported --curve-arms values: {','.join(bad_arms)}")
    legacy_image_flags = [
        name for enabled, name in (
            (args.vision, "--vision"),
            (args.image2, "--image2"),
            (args.image_flow, "--image-flow"),
        ) if enabled
    ]
    if legacy_image_flags:
        sys.exit(
            "ERROR: legacy synthetic image harness flags were removed from GPU launch: "
            + ", ".join(legacy_image_flags)
            + ". Use --vision-understanding and --image-latent with --image-manifest "
            "or --image-fetch.")
    if args.vision_understanding and not (args.image_manifest or args.image_fetch):
        sys.exit("ERROR: --vision-understanding requires --image-manifest or --image-fetch")
    if args.image_latent and not (args.image_manifest or args.image_fetch):
        sys.exit("ERROR: --image-latent requires --image-manifest or --image-fetch")
    if args.image_caption and not (args.image_manifest or args.image_fetch):
        sys.exit("ERROR: --image-caption requires --image-manifest or --image-fetch")
    if args.image_embed and not (args.image_manifest or args.image_fetch):
        sys.exit("ERROR: --image-embed requires --image-manifest or --image-fetch")
    if args.image_embed_batch <= 0:
        sys.exit("ERROR: --image-embed-batch must be positive")
    if args.image_embed_max_records < 0:
        sys.exit("ERROR: --image-embed-max-records must be non-negative")
    if args.image_eval_generated_samples < 0:
        sys.exit("ERROR: --image-eval-generated-samples must be non-negative")
    if args.image_eval_generated_candidates_per_prompt <= 0:
        sys.exit("ERROR: --image-eval-generated-candidates-per-prompt must be positive")
    if args.image_hflip_prob < 0.0 or args.image_hflip_prob > 1.0:
        sys.exit("ERROR: --image-hflip-prob must be in [0, 1]")
    try:
        parse_source_weight_csv(args.image_source_weights, "--image-source-weights")
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    if args.image_cfg_rescale < 0.0 or args.image_cfg_rescale > 1.0:
        sys.exit("ERROR: --image-cfg-rescale must be in [0, 1]")
    try:
        for raw in str(args.image_cfg_rescale_sweep).split(","):
            raw = raw.strip()
            if not raw:
                continue
            val = float(raw)
            if val < 0.0 or val > 1.0:
                raise ValueError(raw)
    except ValueError:
        sys.exit("ERROR: --image-cfg-rescale-sweep values must be in [0, 1]")
    valid_cfg_modes = {"standard", "cfgpp"}
    cfg_modes = {
        raw.strip()
        for raw in str(args.image_cfg_modes or args.image_cfg_mode).split(",")
        if raw.strip()
    }
    bad_cfg_modes = sorted(cfg_modes - valid_cfg_modes)
    if bad_cfg_modes:
        sys.exit(
            "ERROR: unsupported --image-cfg-modes values: "
            + ",".join(bad_cfg_modes)
        )
    try:
        parse_unit_interval(args.image_cfg_interval, "--image-cfg-interval")
        parse_unit_interval(
            args.image_semantic_guidance_interval,
            "--image-semantic-guidance-interval",
        )
        parse_unit_interval(
            args.image_sample_churn_interval,
            "--image-sample-churn-interval",
        )
        parse_unit_interval(
            args.image_sample_text_guidance_interval,
            "--image-sample-text-guidance-interval",
        )
        parse_unit_interval(
            args.image_sample_feature_guidance_interval,
            "--image-sample-feature-guidance-interval",
        )
        parse_unit_interval(
            args.image_sample_quality_guidance_interval,
            "--image-sample-quality-guidance-interval",
        )
        parse_nonnegative_float_csv(
            str(args.image_sample_churn),
            "--image-sample-churn",
        )
        parse_nonnegative_float_csv(
            args.image_sample_churns,
            "--image-sample-churns",
        )
        parse_nonnegative_float_csv(
            str(args.image_sample_velocity_clip),
            "--image-sample-velocity-clip",
        )
        parse_nonnegative_float_csv(
            str(args.image_sample_latent_clip),
            "--image-sample-latent-clip",
        )
        parse_nonnegative_float_csv(
            args.image_eval_text_guidance_sweep,
            "--image-eval-text-guidance-sweep",
        )
        parse_nonnegative_float_csv(
            args.image_eval_feature_guidance_sweep,
            "--image-eval-feature-guidance-sweep",
        )
        parse_nonnegative_float_csv(
            args.image_eval_quality_guidance_sweep,
            "--image-eval-quality-guidance-sweep",
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    valid_sample_schedules = {"linear", "quadratic", "sqrt", "cosine"}
    bad_sample_schedules = sorted(
        {raw.strip() for raw in str(args.image_sample_schedules).split(",") if raw.strip()}
        - valid_sample_schedules
    )
    if bad_sample_schedules:
        sys.exit(
            "ERROR: unsupported --image-sample-schedules values: "
            + ",".join(bad_sample_schedules)
        )
    if args.image_time_shift <= 0.0:
        sys.exit("ERROR: --image-time-shift must be positive")
    if args.image_time_curriculum_frac < 0.0 or args.image_time_curriculum_frac > 1.0:
        sys.exit("ERROR: --image-time-curriculum-frac must be in [0, 1]")
    if args.image_time_shift_ref_dim <= 0.0:
        sys.exit("ERROR: --image-time-shift-ref-dim must be positive")
    if args.image_flow_loss_weight_gamma <= 0.0:
        sys.exit("ERROR: --image-flow-loss-weight-gamma must be positive")
    if args.image_flow_endpoint_w < 0.0:
        sys.exit("ERROR: --image-flow-endpoint-w must be non-negative")
    if args.image_flow_self_repa_w < 0.0:
        sys.exit("ERROR: --image-flow-self-repa-w must be non-negative")
    if args.image_flow_self_repa_steps < 0:
        sys.exit("ERROR: --image-flow-self-repa-steps must be non-negative")
    if args.image_flow_self_repa_embed_dim <= 0:
        sys.exit("ERROR: --image-flow-self-repa-embed-dim must be positive")
    if args.image_flow_sra_w < 0.0:
        sys.exit("ERROR: --image-flow-sra-w must be non-negative")
    if args.image_flow_sra_steps < 0:
        sys.exit("ERROR: --image-flow-sra-steps must be non-negative")
    if args.image_flow_sra_time_gap <= 0.0 or args.image_flow_sra_time_gap > 1.0:
        sys.exit("ERROR: --image-flow-sra-time-gap must be in (0, 1]")
    if args.image_flow_noise_coupling_projections <= 0:
        sys.exit("ERROR: --image-flow-noise-coupling-projections must be positive")
    if args.image_fetch_max_records < 0:
        sys.exit("ERROR: --image-fetch-max-records must be non-negative")
    if args.image_fetch_eval_frac < 0.0 or args.image_fetch_eval_frac >= 1.0:
        sys.exit("ERROR: --image-fetch-eval-frac must be in [0, 1)")
    if args.image_fetch_diffusiondb_start_part <= 0:
        sys.exit("ERROR: --image-fetch-diffusiondb-start-part must be positive")
    if args.image_fetch_diffusiondb_end_part < 0:
        sys.exit("ERROR: --image-fetch-diffusiondb-end-part must be non-negative")
    if args.image_caption and args.image_caption_backend == "hf" and not args.image_caption_model:
        sys.exit("ERROR: --image-caption-backend hf requires --image-caption-model")
    if args.image_caption_batch <= 0:
        sys.exit("ERROR: --image-caption-batch must be positive")
    if args.image_caption_max_records < 0:
        sys.exit("ERROR: --image-caption-max-records must be non-negative")
    if args.image_caption_max_new_tokens <= 0:
        sys.exit("ERROR: --image-caption-max-new-tokens must be positive")
    if args.image_caption_num_beams <= 0:
        sys.exit("ERROR: --image-caption-num-beams must be positive")
    if args.image_score and not (args.image_manifest or args.image_fetch):
        sys.exit("ERROR: --image-score requires --image-manifest or --image-fetch")
    if args.image_score_max_records < 0:
        sys.exit("ERROR: --image-score-max-records must be non-negative")
    if args.image_score_image_size <= 0:
        sys.exit("ERROR: --image-score-image-size must be positive")
    if (args.image_score_technical_w < 0.0 or args.image_score_alignment_w < 0.0
            or args.image_score_external_w < 0.0):
        sys.exit("ERROR: image score weights must be non-negative")
    if (args.image_score_backend == "external"
            or (args.image_score_backend == "ensemble" and args.image_score_external_w > 0.0)):
        if not args.image_score_external_sidecar:
            sys.exit(
                "ERROR: external image scoring requires --image-score-external-sidecar")
    if args.image_embed_text_sequence_max_length < 0:
        sys.exit("ERROR: --image-embed-text-sequence-max-length must be non-negative")
    if args.image_embed_text_sequence_model and args.image_embed_backend != "hf":
        sys.exit("ERROR: --image-embed-text-sequence-model requires --image-embed-backend hf")
    if (args.image_embed_text_sequence_model
            and args.image_embed_text_mode == "pooled"):
        sys.exit("ERROR: --image-embed-text-sequence-model requires --image-embed-text-mode tokens or both")
    if args.image_prompt_embed_text_sequence_max_length < 0:
        sys.exit("ERROR: --image-prompt-embed-text-sequence-max-length must be non-negative")
    if (args.image_prompt_embed_text_sequence_model
            and args.image_prompt_embed_backend != "hf"):
        sys.exit("ERROR: --image-prompt-embed-text-sequence-model requires --image-prompt-embed-backend hf")
    if args.image_clean_max_nsfw is not None and args.image_clean_max_nsfw < 0.0:
        sys.exit("ERROR: --image-clean-max-nsfw must be non-negative")
    if (args.image_clean_max_watermark is not None
            and args.image_clean_max_watermark < 0.0):
        sys.exit("ERROR: --image-clean-max-watermark must be non-negative")
    if (args.image_clean_min_image_text_cosine is not None
            and (args.image_clean_min_image_text_cosine < -1.0
                 or args.image_clean_min_image_text_cosine > 1.0)):
        sys.exit("ERROR: --image-clean-min-image-text-cosine must be in [-1, 1]")
    if (args.image_clean_max_image_duplicate_cosine is not None
            and (args.image_clean_max_image_duplicate_cosine < -1.0
                 or args.image_clean_max_image_duplicate_cosine > 1.0)):
        sys.exit("ERROR: --image-clean-max-image-duplicate-cosine must be in [-1, 1]")
    if args.image_clean_dedupe_lsh_bits <= 0:
        sys.exit("ERROR: --image-clean-dedupe-lsh-bits must be positive")
    if args.image_clean_dedupe_lsh_tables <= 0:
        sys.exit("ERROR: --image-clean-dedupe-lsh-tables must be positive")
    if args.image_flow_distill_steps < 0:
        sys.exit("ERROR: --image-flow-distill-steps must be non-negative")
    if args.image_flow_distill_w < 0.0:
        sys.exit("ERROR: --image-flow-distill-w must be non-negative")
    if args.image_flow_distill_time_gap <= 0.0 or args.image_flow_distill_time_gap > 1.0:
        sys.exit("ERROR: --image-flow-distill-time-gap must be in (0, 1]")
    if args.image_flow_guidance_distill_w < 0.0:
        sys.exit("ERROR: --image-flow-guidance-distill-w must be non-negative")
    if args.image_flow_guidance_distill_cfg_scale < 1.0:
        sys.exit("ERROR: --image-flow-guidance-distill-cfg-scale must be >= 1")
    if (args.image_flow_guidance_distill_cfg_rescale < 0.0
            or args.image_flow_guidance_distill_cfg_rescale > 1.0):
        sys.exit("ERROR: --image-flow-guidance-distill-cfg-rescale must be in [0, 1]")
    if (args.image_flow_distill_steps > 0 and args.image_flow_distill_w > 0.0
            and args.image_flow_distill_teacher == "ema"
            and args.image_flow_ema_decay <= 0.0):
        sys.exit("ERROR: --image-flow-distill-teacher ema requires --image-flow-ema-decay > 0")
    if args.image_sample_prompts and not args.image_sample_grid:
        sys.exit("ERROR: --image-sample-prompts requires --image-sample-grid")
    if args.image_sample_manifest_out and not args.image_sample_grid:
        sys.exit("ERROR: --image-sample-manifest-out requires --image-sample-grid")
    if args.image_sample_image_dir and not args.image_sample_manifest_out:
        sys.exit("ERROR: --image-sample-image-dir requires --image-sample-manifest-out")
    if args.image_eval_generated and not args.image_sample_manifest_out:
        sys.exit("ERROR: --image-eval-generated requires --image-sample-manifest-out")
    if args.image_eval_generated and not args.image_latent:
        sys.exit("ERROR: --image-eval-generated requires --image-latent")
    if args.image_eval_generated and not (
            args.image_eval_generated_real_manifest or args.image_manifest or args.image_fetch):
        sys.exit(
            "ERROR: --image-eval-generated requires --image-eval-generated-real-manifest, "
            "--image-manifest, or --image-fetch")
    if args.image_generated_eval_max_records < 0:
        sys.exit("ERROR: --image-generated-eval-max-records must be non-negative")
    if args.image_generated_eval_min_score < 0.0:
        sys.exit("ERROR: --image-generated-eval-min-score must be non-negative")
    if (args.image_generated_eval_min_support_precision < 0.0
            or args.image_generated_eval_min_support_precision > 1.0):
        sys.exit("ERROR: --image-generated-eval-min-support-precision must be in [0, 1]")
    if (args.image_generated_eval_min_support_recall < 0.0
            or args.image_generated_eval_min_support_recall > 1.0):
        sys.exit("ERROR: --image-generated-eval-min-support-recall must be in [0, 1]")
    if (args.image_generated_eval_min_image_text_cos is not None
            and (args.image_generated_eval_min_image_text_cos < -1.0
                 or args.image_generated_eval_min_image_text_cos > 1.0)):
        sys.exit("ERROR: --image-generated-eval-min-image-text-cos must be in [-1, 1]")
    if (args.image_generated_eval_max_frechet is not None
            and args.image_generated_eval_max_frechet < 0.0):
        sys.exit("ERROR: --image-generated-eval-max-frechet must be non-negative")
    if (args.image_generated_eval_max_mmd_rbf is not None
            and args.image_generated_eval_max_mmd_rbf < 0.0):
        sys.exit("ERROR: --image-generated-eval-max-mmd-rbf must be non-negative")
    if args.image_sample_negative_prompts and not args.image_sample_prompts:
        sys.exit("ERROR: --image-sample-negative-prompts requires --image-sample-prompts")
    if args.image_sample_candidates_per_prompt <= 0:
        sys.exit("ERROR: --image-sample-candidates-per-prompt must be positive")
    if args.image_sample_candidates_per_prompt > 1 and not args.image_sample_prompts:
        sys.exit("ERROR: --image-sample-candidates-per-prompt > 1 requires --image-sample-prompts")
    if args.image_sample_text_guidance_w < 0.0:
        sys.exit("ERROR: --image-sample-text-guidance-w must be non-negative")
    if args.image_sample_text_guidance_w > 0.0 and not args.image_sample_prompts:
        sys.exit("ERROR: --image-sample-text-guidance-w requires --image-sample-prompts")
    if args.image_sample_feature_guidance_w < 0.0:
        sys.exit("ERROR: --image-sample-feature-guidance-w must be non-negative")
    if args.image_sample_feature_guidance_w > 0.0 and not args.image_sample_prompts:
        sys.exit("ERROR: --image-sample-feature-guidance-w requires --image-sample-prompts")
    if args.image_sample_quality_guidance_w < 0.0:
        sys.exit("ERROR: --image-sample-quality-guidance-w must be non-negative")
    if args.image_sample_quality_guidance_w > 0.0 and not args.image_sample_prompts:
        sys.exit("ERROR: --image-sample-quality-guidance-w requires --image-sample-prompts")
    if args.image_quality_score_w < 0.0 or args.image_flow_quality_score_w < 0.0:
        sys.exit("ERROR: image quality score weights must be non-negative")
    if args.image_quality_score_steps < 0:
        sys.exit("ERROR: --image-quality-score-steps must be non-negative")
    if (args.image_quality_rank_w < 0.0 or args.image_quality_score_rank_w < 0.0
            or args.image_flow_quality_rank_w < 0.0):
        sys.exit("ERROR: image quality rank weights must be non-negative")
    if args.image_quality_rank_margin < 0.0:
        sys.exit("ERROR: --image-quality-rank-margin must be non-negative")
    if args.image_quality_score_rank_w > 0.0 and args.image_quality_score_steps <= 0:
        sys.exit("ERROR: --image-quality-score-rank-w requires --image-quality-score-steps > 0")
    if args.image_preference_max_pairs < 0:
        sys.exit("ERROR: --image-preference-max-pairs must be non-negative")
    if args.image_preference_w < 0.0:
        sys.exit("ERROR: --image-preference-w must be non-negative")
    if args.image_preference_w > 0.0 and not args.image_preference_manifest:
        sys.exit("ERROR: --image-preference-w requires --image-preference-manifest")
    if args.image_preference_w > 0.0 and args.image_quality_score_steps <= 0:
        sys.exit("ERROR: --image-preference-w requires --image-quality-score-steps > 0")
    if args.image_preference_manifest and not args.image_manifest and not args.image_fetch:
        sys.exit(
            "ERROR: --image-preference-manifest requires --image-manifest or --image-fetch")
    if ((args.image_flow_quality_score_w > 0.0 or args.image_flow_quality_rank_w > 0.0)
            and args.image_quality_score_w <= 0.0 and args.image_quality_rank_w <= 0.0
            and args.image_quality_score_steps <= 0):
        sys.exit(
            "ERROR: image flow quality losses require --image-quality-score-w/"
            "--image-quality-rank-w > 0 or --image-quality-score-steps > 0")
    if args.image_flow_cache_max_loaded_shards < 0:
        sys.exit("ERROR: --image-flow-cache-max-loaded-shards must be non-negative")
    if (args.image_sample_prompts and args.image_prompt_embed_backend == "hf"
            and not args.image_prompt_embed_model):
        sys.exit("ERROR: --image-prompt-embed-backend hf requires --image-prompt-embed-model")
    if args.image_ae_arch == "hf-vae" and not args.image_ae_hf_model:
        sys.exit("ERROR: --image-ae-arch hf-vae requires --image-ae-hf-model")
    if args.image_dit_pos_embed == "rope2d" and args.image_latent_arch != "mmdit":
        sys.exit("ERROR: --image-dit-pos-embed rope2d requires --image-latent-arch mmdit")
    if args.image_dit_mlp != "gelu" and args.image_latent_arch == "dit":
        sys.exit("ERROR: --image-dit-mlp swiglu requires --image-latent-arch crossdit or mmdit")
    if args.image_latent_patch_size <= 0:
        sys.exit("ERROR: --image-latent-patch-size must be positive")
    if args.image_latent_patch_size != 1 and args.image_latent_arch == "conv":
        sys.exit("ERROR: --image-latent-patch-size requires --image-latent-arch dit/crossdit/mmdit")
    if args.vision_understanding:
        positive = {
            "--vision-understanding-patch": args.vision_understanding_patch,
            "--vision-understanding-slots": args.vision_understanding_slots,
            "--vision-understanding-heads": args.vision_understanding_heads,
            "--vision-understanding-layers": args.vision_understanding_layers,
            "--vision-understanding-memory-size": args.vision_understanding_memory_size,
        }
        if args.vision_understanding_steps < 0:
            sys.exit("ERROR: --vision-understanding-steps must be non-negative")
        if args.vision_understanding_batch < 0:
            sys.exit("ERROR: --vision-understanding-batch must be non-negative")
        if args.vision_understanding_dim < 0:
            sys.exit("ERROR: --vision-understanding-dim must be non-negative")
        for name, value in positive.items():
            if value <= 0:
                sys.exit(f"ERROR: {name} must be positive")
        vu_dim = args.vision_understanding_dim or args.dim or 128
        if vu_dim % args.vision_understanding_heads != 0:
            sys.exit("ERROR: vision understanding width must divide --vision-understanding-heads")
        if (args.vision_understanding_memory_w < 0.0
                or args.vision_understanding_association_w < 0.0
                or args.vision_understanding_composition_w < 0.0
                or args.vision_understanding_text_align_w < 0.0
                or args.vision_understanding_image_align_w < 0.0):
            sys.exit("ERROR: vision understanding loss weights must be non-negative")
    if args.multimodal:
        positive = {
            "--multimodal-layers": args.multimodal_layers,
            "--multimodal-heads": args.multimodal_heads,
            "--multimodal-batch": args.multimodal_batch,
            "--multimodal-log-every": args.multimodal_log_every,
            "--multimodal-img-tokens": args.multimodal_img_tokens,
            "--multimodal-aud-tokens": args.multimodal_aud_tokens,
            "--multimodal-txt-tokens": args.multimodal_txt_tokens,
            "--multimodal-trunk-width": args.multimodal_trunk_width,
            "--multimodal-trunk-depth": args.multimodal_trunk_depth,
            "--multimodal-text-layers": args.multimodal_text_layers,
            "--multimodal-concept-tokens": args.multimodal_concept_tokens,
            "--multimodal-fusion-layers": args.multimodal_fusion_layers,
            "--multimodal-latent-concept-layers": (
                args.multimodal_latent_concept_layers),
            "--multimodal-eval-n": args.multimodal_eval_n,
        }
        for name, value in positive.items():
            if value <= 0:
                sys.exit(f"ERROR: {name} must be positive")
        if args.multimodal_dim < 0:
            sys.exit("ERROR: --multimodal-dim must be non-negative")
        mm_dim = args.multimodal_dim or args.dim or 96
        if mm_dim % args.multimodal_heads != 0:
            sys.exit("ERROR: multimodal width must be divisible by --multimodal-heads")
        if (mm_dim // args.multimodal_heads) % 2 != 0:
            sys.exit("ERROR: multimodal head dimension must be even for rope attention")
        if args.multimodal_lr <= 0.0:
            sys.exit("ERROR: --multimodal-lr must be positive")
        if (args.multimodal_value_w < 0.0 or args.multimodal_agreement_w < 0.0
                or args.multimodal_concept_w < 0.0
                or args.multimodal_concept_agreement_w < 0.0
                or args.multimodal_concept_distill_w < 0.0
                or args.multimodal_concept_rank_distill_w < 0.0
                or args.multimodal_concept_transfer_w < 0.0
                or args.multimodal_concept_contrast_w < 0.0
                or args.multimodal_concept_centroid_w < 0.0
                or args.multimodal_concept_prototype_w < 0.0
                or args.multimodal_concept_prototype_spread_w < 0.0
                or args.multimodal_concept_state_spread_w < 0.0
                or args.multimodal_latent_concept_w < 0.0
                or args.multimodal_latent_concept_invariance_w < 0.0
                or args.multimodal_latent_concept_variance_w < 0.0
                or args.multimodal_latent_concept_covariance_w < 0.0):
            sys.exit("ERROR: multimodal loss weights must be non-negative")
        if args.multimodal_latent_concept_slots < 0:
            sys.exit("ERROR: --multimodal-latent-concept-slots must be non-negative")
        if (args.multimodal_latent_concept_w > 0.0
                and args.multimodal_latent_concept_slots <= 0):
            sys.exit(
                "ERROR: --multimodal-latent-concept-w requires "
                "--multimodal-latent-concept-slots > 0")
        if (args.multimodal_latent_concept_view_dropout < 0.0
                or args.multimodal_latent_concept_view_dropout >= 1.0):
            sys.exit("ERROR: --multimodal-latent-concept-view-dropout must be in [0, 1)")
        if args.multimodal_latent_concept_variance_target < 0.0:
            sys.exit(
                "ERROR: --multimodal-latent-concept-variance-target must be non-negative")
        if args.multimodal_concept_distill_temperature <= 0.0:
            sys.exit("ERROR: --multimodal-concept-distill-temperature must be positive")
        if args.multimodal_concept_contrast_temperature <= 0.0:
            sys.exit("ERROR: --multimodal-concept-contrast-temperature must be positive")
        if args.multimodal_concept_centroid_temperature <= 0.0:
            sys.exit("ERROR: --multimodal-concept-centroid-temperature must be positive")
        if args.multimodal_concept_centroid_margin < 0.0:
            sys.exit("ERROR: --multimodal-concept-centroid-margin must be non-negative")
        if args.multimodal_concept_rank_distill_margin < 0.0:
            sys.exit("ERROR: --multimodal-concept-rank-distill-margin must be non-negative")
        if args.multimodal_concept_transfer_margin < 0.0:
            sys.exit("ERROR: --multimodal-concept-transfer-margin must be non-negative")
        if args.multimodal_concept_prototype_spread_margin < 0.0:
            sys.exit(
                "ERROR: --multimodal-concept-prototype-spread-margin must be non-negative")
        if args.multimodal_concept_state_spread_variance < 0.0:
            sys.exit("ERROR: --multimodal-concept-state-spread-variance must be non-negative")
        if args.multimodal_concept_state_spread_margin < -1.0:
            sys.exit("ERROR: --multimodal-concept-state-spread-margin must be >= -1")
        if args.multimodal_concept_state_spread_covariance_w < 0.0:
            sys.exit(
                "ERROR: --multimodal-concept-state-spread-covariance-w must be non-negative")
        if args.multimodal_dropout < 0.0 or args.multimodal_dropout > 1.0:
            sys.exit("ERROR: --multimodal-dropout must be in [0, 1]")
        if (args.multimodal_free_n < 0 or args.multimodal_counterfactual_n < 0
                or args.multimodal_free_counterfactual_n < 0):
            sys.exit("ERROR: multimodal eval counts must be non-negative")
    if args.upload_image_data:
        if not args.image_manifest:
            sys.exit("ERROR: --upload-image-data requires --image-manifest")
        if os.path.isabs(args.image_root) or os.path.isabs(args.image_manifest):
            sys.exit(
                "ERROR: --upload-image-data expects repo-relative --image-root and "
                "--image-manifest paths; omit the upload flag for pod-mounted absolute paths"
            )
        for label, raw_path, want_dir in (
                ("--image-root", args.image_root, True),
                ("--image-manifest", args.image_manifest, False)):
            if not raw_path:
                sys.exit(f"ERROR: --upload-image-data requires {label}")
            local_path = local_path_for_arg(raw_path)
            if want_dir and not os.path.isdir(local_path):
                sys.exit(f"ERROR: {label} not found or not a directory: {local_path}")
            if not want_dir and not os.path.isfile(local_path):
                sys.exit(f"ERROR: {label} not found or not a file: {local_path}")

    key = os.environ.get("RUNPOD_API_KEY")
    if not key and args.go:
        sys.exit("ERROR: export RUNPOD_API_KEY first (do not hardcode it).")
    pub = os.path.expanduser("~/.ssh/id_rsa.pub")
    pubkey = open(pub).read().strip() if os.path.exists(pub) else ""

    body = {"name": args.name, "imageName": IMAGE, "gpuTypeIds": [args.gpu], "gpuCount": 1,
            "cloudType": args.cloud, "containerDiskInGb": args.disk, "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": pubkey}}
    cap = args.max_minutes * 60
    run = payload(args)
    image_embed_deps = bool(
        (args.image_embed or args.image_eval_generated)
        and args.image_embed_backend == "hf")
    image_caption_deps = bool(args.image_caption and args.image_caption_backend == "hf")
    image_hf_ae_deps = bool(args.image_latent and args.image_ae_arch == "hf-vae")
    image_prompt_embed_deps = bool(
        args.image_latent and args.image_sample_prompts
        and args.image_prompt_embed_backend == "hf")
    image_text_sequence_deps = bool(
        args.image_embed_text_sequence_model
        or (args.image_latent and args.image_sample_prompts
            and args.image_prompt_embed_text_sequence_model))
    if args.fast:                                           # image torch; verbalizer
        fast_pkgs = "numpy tokenizers pandas pyarrow pillow"
        if image_embed_deps or image_prompt_embed_deps or image_caption_deps:
            fast_pkgs += " transformers accelerate"
        if image_text_sequence_deps or image_caption_deps:
            fast_pkgs += " sentencepiece"
        if image_hf_ae_deps:
            fast_pkgs += " diffusers transformers accelerate safetensors"
        setup = f"pip install -q {fast_pkgs}"
    else:
        install_deps = ""
        if image_embed_deps or image_prompt_embed_deps:
            install_deps += "INSTALL_IMAGE_EMBED_DEPS=1 "
        if image_caption_deps:
            install_deps += "INSTALL_IMAGE_CAPTION_DEPS=1 "
        if image_text_sequence_deps:
            install_deps += "INSTALL_IMAGE_TEXT_SEQUENCE_DEPS=1 "
        if image_hf_ae_deps:
            install_deps += "INSTALL_IMAGE_HF_AE_DEPS=1 "
        setup = f"{install_deps}WORKDIR={REMOTE} bash runpod/setup.sh"
    # tee to LOCAL disk: /workspace is a network volume that stalls under streaming writes
    # (see runpod/setup.sh -- it cost us rung B4: training was healthy, only the log froze)
    remote_cmd = (
        f"cd {REMOTE} && rm -f thinking.log /root/thinking.log && "
        f"({setup} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')}) "
        f"2>&1 | tee /root/thinking.log; "
        f"cp /root/thinking.log {REMOTE}/thinking.log 2>/dev/null; true")

    image_jobs = [name for enabled, name in (
        (args.image_score, "image quality scoring preprocess"),
        (args.image_embed, "image embedding preprocess"),
        (args.vision_understanding, "vision understanding concept memory"),
        (args.image_latent,
         f"latent {args.image_latent_arch} {args.image_cond_mode}-conditioned flow"),
        (args.audio, "audio FER arms"),
        (args.multimodal, "multimodal bridge"),
    ) if enabled]
    job = " + ".join(image_jobs) if image_jobs else "kinship multi-seed"
    print("=== PLAN === thinking package on H100: " + job
          + (" + chain grid" if args.sweep and not args.vision_understanding else ""))
    print(f"gpu/cloud : {args.gpu} / {args.cloud}")
    print(f"seeds     : {args.seeds}")
    print(f"sync up   : {HERE}/ -> pod:{REMOTE}")
    if args.upload_image_data:
        print(f"image data: {local_path_for_arg(args.image_root)} -> "
              f"pod:{remote_path_for_arg(args.image_root)}")
        print(f"manifest  : {local_path_for_arg(args.image_manifest)} -> "
              f"pod:{remote_path_for_arg(args.image_manifest)}")
    print(f"fetch     : thinking.log + runs/ (models, config, results) -> {HERE}/")
    print(f"guard     : pod-side timeout {cap}s + always-terminate; SSH-wait cap {args.max_minutes}m")
    if not args.go:
        print("\n[dry-run] nothing created. Re-run with --go to launch (spends money).")
        return

    print("\n=== creating pod ===")
    st, pod = api("POST", "/pods", key, body)
    if st not in (200, 201):
        sys.exit(f"create failed: HTTP {st} {pod}")
    pid = pod.get("id") or pod.get("podId")
    print("pod id:", pid)
    t0 = time.time()
    try:
        ip = port = None
        while time.time() - t0 < args.max_minutes * 60:
            st, p = api("GET", f"/pods/{pid}", key)
            status = p.get("desiredStatus")
            ip = p.get("publicIp")
            port = (p.get("portMappings") or {}).get("22")
            print(f"  status={status} ip={ip} port={port}")
            if status == "RUNNING" and ip and port:
                break
            time.sleep(12)
        if not (ip and port):
            sys.exit("pod never exposed SSH within cap; terminating.")
        time.sleep(25)
        ssh = f"ssh -o StrictHostKeyChecking=no -p {port} root@{ip}"
        if getattr(args, "ref", None):
            # PINNED DEPLOY: ship exactly one committed tree (REBAL2 lesson: tar of the live
            # working dir snapshots parallel mid-edits -> selftest passed locally but the pod
            # ran different, broken code; its whole eval ladder was junk)
            up = (f"git -C {shlex_quote(HERE)} archive --format=tar.gz {shlex_quote(args.ref)} "
                  f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        else:
            up = (f"COPYFILE_DISABLE=1 tar czf - --exclude './.venv*' --exclude '*/__pycache__' "
                  f"--exclude './results_gpu' --exclude '*.zip' --exclude './data' --exclude '*.pt' "
                  f"--exclude '*.log' --exclude './runs' --exclude './experiments' "
                  f"--exclude './tooling' --exclude './artifacts' --exclude './.git' "
                  f"--exclude '*.tgz' -C {HERE} . "
                  f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(up)
        if args.upload_image_data:
            root_local = local_path_for_arg(args.image_root)
            root_remote = remote_path_for_arg(args.image_root)
            manifest_local = local_path_for_arg(args.image_manifest)
            manifest_remote = remote_path_for_arg(args.image_manifest)
            sh(upload_path_cmd(root_local, root_remote, ssh))
            if os.path.abspath(manifest_local) != os.path.abspath(root_local):
                sh(upload_path_cmd(manifest_local, manifest_remote, ssh))
        if args.eval_only_run:
            local_run = os.path.join(HERE, args.eval_only_run)
            if not os.path.isdir(local_run):
                raise FileNotFoundError(f"--eval-only-run not found: {local_run}")
            up_run = (f"COPYFILE_DISABLE=1 tar czf - -C {shlex_quote(HERE)} "
                      f"{shlex_quote(args.eval_only_run)} "
                      f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
            sh(up_run)
        # DETACHED execution: nohup on the pod + short-poll. A dropped SSH pipe killed three
        # healthy runs (B7/C6/L) when it took the cost-guard with it -- never hold a session.
        script = remote_cmd + "; touch /root/DONE\n"
        for _try in range(5):                              # VERIFIED upload (a silent network
            subprocess.run(f"{ssh} 'cat > /root/run.sh'",  # blip once shipped an empty script)
                           shell=True, input=script, text=True, timeout=120)
            ok = subprocess.run(f"{ssh} 'test -s /root/run.sh && echo OK'", shell=True,
                                capture_output=True, text=True, timeout=120)
            if "OK" in (ok.stdout or ""):
                break
            time.sleep(10)
        else:
            raise RuntimeError("run.sh upload failed after 5 attempts")
        sh(f"{ssh} 'rm -f /root/DONE /root/thinking.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < args.max_minutes * 60:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; "
                                   f"tail -1 /root/thinking.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue                                   # network blip: poll again
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("payload complete")
                break
        fetch = (f"{ssh} 'cd {REMOTE} && tar czf - thinking.log runs 2>/dev/null' "
                 f"| tar xzf - -C {HERE}")
        sh(fetch)
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
