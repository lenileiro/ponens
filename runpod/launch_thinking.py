#!/usr/bin/env python3
"""RunPod H100 runner for the current thinking package.

The launcher runs the manifest/raw-data training surfaces that are still supported: raw reading,
image preprocessing/generation/eval, vision understanding, and the generic multimodal bridge.
Pod-side `timeout` bounds the run and cleanup always terminates the pod. Defaults to dry-run.

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
IMAGE_SAMPLE_SCHEDULES = ("linear", "quadratic", "sqrt", "cosine", "karras")
IMAGE_TIME_SAMPLINGS = ("uniform", "logit-normal", "mode", "adaptive")
IMAGE_TIME_ADAPTIVE_PRIORS = ("uniform", "logit-normal", "mode")
IMAGE_DEFAULT_TIME_MODE_SCALE = 1.29
IMAGE_MAX_TIME_MODE_SCALE = 1.75


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


def is_url_arg(path):
    return str(path or "").startswith(("http://", "https://"))


def path_with_suffix(path, suffix, default_dir="runs"):
    path = str(path or "").strip()
    if not path:
        return ""
    root, _ext = os.path.splitext(path)
    if not root:
        root = os.path.join(default_dir, "generated")
    return root + suffix


def generated_preference_path_for_args(args):
    candidates = str(getattr(args, "image_sample_candidates_manifest_out", "") or "").strip()
    if not candidates:
        return ""
    return (
        str(getattr(args, "image_generated_preference_out", "") or "").strip()
        or path_with_suffix(candidates, "_preferences.jsonl")
    )


def _row_pair_paths(row):
    chosen = (
        row.get("chosen_image") or row.get("chosen") or row.get("winner_image")
        or row.get("preferred_image")
    )
    rejected = (
        row.get("rejected_image") or row.get("rejected") or row.get("loser_image")
        or row.get("nonpreferred_image")
    )
    return str(chosen or ""), str(rejected or "")


def local_preference_manifest_is_portable(path, max_rows=64):
    local = local_path_for_arg(path)
    if not local or not os.path.isfile(local):
        return False
    base = os.path.dirname(os.path.abspath(local))
    rows = 0
    with open(local, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return False
            chosen, rejected = _row_pair_paths(row)
            if not chosen or not rejected:
                return False
            for rel in (chosen, rejected):
                if os.path.isabs(rel):
                    return False
                resolved = os.path.abspath(os.path.join(base, rel))
                if not resolved.startswith(base + os.sep) and resolved != base:
                    return False
                if not os.path.isfile(resolved):
                    return False
            rows += 1
            if rows >= int(max_rows):
                break
    return rows > 0


def local_path_remote_destination(local_path):
    local_abs = os.path.abspath(local_path)
    try:
        rel = os.path.relpath(local_abs, HERE)
    except ValueError:
        return local_abs
    if rel.startswith("..") or os.path.isabs(rel):
        return local_abs
    return os.path.join(REMOTE, rel)


def preference_upload_paths(path):
    local = local_path_for_arg(path)
    if not local or not os.path.isfile(local):
        return []
    base = os.path.dirname(os.path.abspath(local))
    uploads = {os.path.abspath(local)}
    with open(local, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for rel in _row_pair_paths(row):
                if not rel or os.path.isabs(rel):
                    continue
                parts = [
                    part for part in rel.replace("\\", "/").split("/")
                    if part and part != "."
                ]
                if not parts or any(part == ".." for part in parts):
                    continue
                top = os.path.join(base, parts[0])
                full = os.path.join(base, *parts)
                uploads.add(os.path.abspath(top if os.path.isdir(top) else full))
    return sorted(uploads)


def reading_upload_paths(args):
    paths = []
    paths.extend(args.reading_data or [])
    paths.extend(args.reading_replay_data or [])
    if args.reading_checkpoint:
        paths.append(args.reading_checkpoint)
    return [path for path in paths if path and not is_url_arg(path)]


def text_reading_cmd(args, py):
    TXT = f"{py} -m thinking.text"
    d = args.text_reading_dim or args.dim or 256
    ckpt = args.reading_out_checkpoint or args.text_reading_checkpoint_out
    report = args.text_reading_out
    cmd = (
        f"{TXT} --steps {args.train_steps or args.text_reading_steps} "
        f"--batch {args.batch} --d {d} --layers {args.text_reading_layers} "
        f"--heads {args.text_reading_heads} --seed {args.text_reading_seed} "
        f"--text-encoder-arch {args.text_reading_encoder_arch} "
        f"--text-encoder-layers {args.text_reading_encoder_layers} "
        f"--latent-concept-layers {args.text_reading_latent_concept_layers} "
        f"--reading-text-field {shlex_quote(args.reading_text_field)} "
        f"--reading-max-tokens {args.reading_max_tokens} "
        f"--reading-min-tokens {args.reading_min_tokens} "
        f"--reading-eval-frac {args.reading_eval_frac} "
        f"--reading-eval-n {args.reading_eval_n} "
        f"--reading-lr {args.reading_lr} "
        f"--reading-token-drop {args.reading_token_drop} "
        f"--reading-token-replace {args.reading_token_replace} "
        f"--reading-feature-dropout {args.reading_feature_dropout} "
        f"--reading-context-target-w {args.reading_context_target_w} "
        f"--reading-context-keep-p {args.reading_context_keep_p} "
        f"--reading-context-target-temperature "
        f"{args.reading_context_target_temperature} "
        f"--reading-sequence-w {args.reading_sequence_w} "
        f"--reading-sequence-batch {args.reading_sequence_batch} "
        f"--reading-sequence-temperature {args.reading_sequence_temperature} "
        f"--reading-factorization-w {args.reading_factorization_w} "
        f"--reading-factorization-variance {args.reading_factorization_variance} "
        f"--reading-factorization-margin {args.reading_factorization_margin} "
        f"--reading-factorization-covariance-w "
        f"{args.reading_factorization_covariance_w} "
        f"--reading-fer-w {args.reading_fer_w} "
        f"--reading-fer-fragmentation-w {args.reading_fer_fragmentation_w} "
        f"--reading-fer-correlation-w {args.reading_fer_correlation_w} "
        f"--reading-fer-balance-w {args.reading_fer_balance_w} "
        f"--reading-memory-w {args.reading_memory_w} "
        f"--reading-memory-size {args.reading_memory_size} "
        f"--reading-memory-temperature {args.reading_memory_temperature} "
        f"--reading-memory-momentum {args.reading_memory_momentum} "
        f"--reading-memory-balance-w {args.reading_memory_balance_w} "
        f"--reading-consolidation-w {args.reading_consolidation_w} "
        f"--reading-consolidation-temperature "
        f"{args.reading_consolidation_temperature} "
        f"--reading-consolidation-balance-w "
        f"{args.reading_consolidation_balance_w} "
        f"--reading-consolidation-anchor-w "
        f"{args.reading_consolidation_anchor_w} "
        f"--reading-consolidation-fer-w {args.reading_consolidation_fer_w} "
        f"--reading-discovery-w {args.reading_discovery_w} "
        f"--reading-discovery-curiosity-w {args.reading_discovery_curiosity_w} "
        f"--reading-discovery-graph-w {args.reading_discovery_graph_w} "
        f"--reading-discovery-cycle-w {args.reading_discovery_cycle_w} "
        f"--reading-discovery-bridge-w {args.reading_discovery_bridge_w} "
        f"--reading-discovery-fer-w {args.reading_discovery_fer_w} "
        f"--reading-reanalysis-w {args.reading_reanalysis_w} "
        f"--reading-reanalysis-graph-w {args.reading_reanalysis_graph_w} "
        f"--reading-reanalysis-cycle-w {args.reading_reanalysis_cycle_w} "
        f"--reading-reanalysis-bridge-w {args.reading_reanalysis_bridge_w} "
        f"--reading-reanalysis-fer-w {args.reading_reanalysis_fer_w} "
        f"--reading-association-w {args.reading_association_w} "
        f"--reading-association-temperature {args.reading_association_temperature} "
        f"--reading-association-decay {args.reading_association_decay} "
        f"--reading-association-target-power "
        f"{args.reading_association_target_power} "
        f"--reading-association-self-loop-w "
        f"{args.reading_association_self_loop_w} "
        f"--reading-association-transitive-steps "
        f"{args.reading_association_transitive_steps} "
        f"--reading-association-transitive-w "
        f"{args.reading_association_transitive_w} "
        f"--reading-composition-w {args.reading_composition_w} "
        f"--reading-composition-temperature {args.reading_composition_temperature} "
        f"--reading-composition-self-loop-w "
        f"{args.reading_composition_self_loop_w} "
        f"--reading-composition-transitive-steps "
        f"{args.reading_composition_transitive_steps} "
        f"--reading-composition-transitive-w "
        f"{args.reading_composition_transitive_w} "
        f"--reading-composition-margin {args.reading_composition_margin} "
        f"--reading-graph-predict-w {args.reading_graph_predict_w} "
        f"--reading-graph-predict-temperature "
        f"{args.reading_graph_predict_temperature} "
        f"--reading-graph-predict-self-loop-w "
        f"{args.reading_graph_predict_self_loop_w} "
        f"--reading-graph-predict-transitive-steps "
        f"{args.reading_graph_predict_transitive_steps} "
        f"--reading-graph-predict-transitive-w "
        f"{args.reading_graph_predict_transitive_w} "
        f"--reading-graph-predict-target-power "
        f"{args.reading_graph_predict_target_power} "
        f"--reading-graph-cycle-w {args.reading_graph_cycle_w} "
        f"--reading-graph-cycle-temperature {args.reading_graph_cycle_temperature} "
        f"--reading-graph-cycle-self-loop-w "
        f"{args.reading_graph_cycle_self_loop_w} "
        f"--reading-graph-cycle-transitive-steps "
        f"{args.reading_graph_cycle_transitive_steps} "
        f"--reading-graph-cycle-transitive-w "
        f"{args.reading_graph_cycle_transitive_w} "
        f"--reading-graph-cycle-target-power "
        f"{args.reading_graph_cycle_target_power} "
        f"--reading-graph-cycle-consistency-w "
        f"{args.reading_graph_cycle_consistency_w} "
        f"--reading-bridge-w {args.reading_bridge_w} "
        f"--reading-neighborhood-w {args.reading_neighborhood_w} "
        f"--reading-neighborhood-batch {args.reading_neighborhood_batch} "
        f"--reading-neighborhood-probe-n {args.reading_neighborhood_probe_n} "
        f"--reading-neighborhood-refresh-steps "
        f"{args.reading_neighborhood_refresh_steps} "
        f"--reading-neighborhood-temperature "
        f"{args.reading_neighborhood_temperature} "
        f"--reading-neighborhood-margin {args.reading_neighborhood_margin} "
        f"--reading-transition-w {args.reading_transition_w} "
        f"--reading-transition-batch {args.reading_transition_batch} "
        f"--reading-transition-temperature {args.reading_transition_temperature} "
        f"--reading-transition-margin {args.reading_transition_margin} "
        f"--reading-cluster-w {args.reading_cluster_w} "
        f"--reading-cluster-batch {args.reading_cluster_batch} "
        f"--reading-cluster-probe-n {args.reading_cluster_probe_n} "
        f"--reading-cluster-refresh-steps {args.reading_cluster_refresh_steps} "
        f"--reading-cluster-temperature {args.reading_cluster_temperature} "
        f"--reading-cluster-margin {args.reading_cluster_margin} "
        f"--reading-cluster-min-size {args.reading_cluster_min_size} "
        f"--reading-study-strategy {args.reading_study_strategy} "
        f"--reading-study-probe-n {args.reading_study_probe_n} "
        f"--reading-study-hard-max {args.reading_study_hard_max} "
        f"--reading-study-refresh-steps {args.reading_study_refresh_steps} "
        f"--reading-study-rounds {args.reading_study_rounds} "
        f"--reading-study-score-metric {args.reading_study_score_metric} "
        f"--reading-study-score-margin-w {args.reading_study_score_margin_w} "
        f"--reading-study-score-min-delta {args.reading_study_score_min_delta} "
        f"--reading-study-score-patience {args.reading_study_score_patience} "
        f"--reading-study-insight-accept-w "
        f"{args.reading_study_insight_accept_w} "
        f"--reading-study-insight-min-delta "
        f"{args.reading_study_insight_min_delta} "
        f"--out {shlex_quote(report)}"
    )
    for source in args.reading_data or []:
        cmd += f" --reading-data {shlex_quote(source)}"
    for source in args.reading_replay_data or []:
        cmd += f" --reading-replay-data {shlex_quote(source)}"
    if args.reading_checkpoint:
        cmd += f" --reading-checkpoint {shlex_quote(args.reading_checkpoint)}"
        cmd += f" --reading-out-checkpoint {shlex_quote(ckpt)}"
    else:
        cmd += f" --checkpoint {shlex_quote(ckpt)}"
    if args.text_reading_latent_concept_slots > 0:
        cmd += f" --latent-concept-slots {args.text_reading_latent_concept_slots}"
    if args.text_reading_latent_concept_prefix:
        cmd += " --latent-concept-prefix"
    if args.text_reading_latent_concept_refine:
        cmd += " --latent-concept-refine"
    cmd += (
        f" --latent-concept-refine-gate-init "
        f"{args.text_reading_latent_concept_refine_gate_init}"
    )
    cmd += (" --reading-study-select-best"
            if args.reading_study_select_best else " --no-reading-study-select-best")
    if args.reading_replay_w:
        cmd += f" --reading-replay-w {args.reading_replay_w}"
    if args.reading_replay_batch:
        cmd += f" --reading-replay-batch {args.reading_replay_batch}"
    if args.reading_replay_retention_w:
        cmd += f" --reading-replay-retention-w {args.reading_replay_retention_w}"
    return cmd


def apply_image_quality_preset(args):
    preset = str(getattr(args, "image_quality_preset", "none") or "none")
    args.image_auto_preference_manifest = False
    if preset == "none":
        return
    if preset not in ("web-hf-vae", "web-hf-vae-hq"):
        raise ValueError(f"unknown image quality preset {preset!r}")
    hq = preset == "web-hf-vae-hq"
    args.image_fetch = True
    args.image_fetch_source = (
        "text-to-image-2m-hq-mix" if hq else "text-to-image-2m-512-2m")
    args.image_score = True
    args.image_score_backend = "ensemble" if hq else "stats"
    args.image_score_image_size = max(int(args.image_score_image_size), 768 if hq else 512)
    if hq:
        if float(args.image_score_technical_w) == 1.0:
            args.image_score_technical_w = 0.25
        args.image_score_pickscore_w = max(float(args.image_score_pickscore_w), 0.75)
        args.image_score_pickscore_batch = max(int(args.image_score_pickscore_batch), 8)
        args.image_score_pickscore_dtype = "auto"
    args.image_embed = True
    args.image_embed_backend = "hf"
    if not args.image_embed_model or args.image_embed_model == "google/siglip-base-patch16-224":
        args.image_embed_model = "google/siglip-so400m-patch14-384"
    args.image_embed_text_mode = "both"
    if (not args.image_embed_text_sequence_model
            or args.image_embed_text_sequence_model == "google-t5/t5-base"):
        args.image_embed_text_sequence_model = "google-t5/t5-large"
    args.image_embed_text_sequence_max_length = max(
        int(args.image_embed_text_sequence_max_length), 192 if hq else 128)
    args.image_latent = True
    args.image_cond_mode = "text"
    args.image_caption_cond_source = "auto"
    args.image_fetch_max_records = max(int(args.image_fetch_max_records), 20000 if hq else 10000)
    args.image_clean_min_caption_tokens = max(
        int(args.image_clean_min_caption_tokens), 4 if hq else 3)
    if hq and float(args.image_clean_min_caption_unique_ratio) <= 0.0:
        args.image_clean_min_caption_unique_ratio = 0.25
    if hq and float(args.image_clean_max_caption_token_frequency) <= 0.0:
        args.image_clean_max_caption_token_frequency = 0.50
    if hq and float(args.image_clean_max_caption_token_run) <= 0.0:
        args.image_clean_max_caption_token_run = 0.50
    if hq and float(args.image_clean_max_caption_char_run) <= 0.0:
        args.image_clean_max_caption_char_run = 0.35
    args.image_clean_min_side = max(int(args.image_clean_min_side), 512)
    args.image_clean_max_aspect = max(float(args.image_clean_max_aspect), 2.0)
    if args.image_clean_max_nsfw is None:
        args.image_clean_max_nsfw = 0.2
    if args.image_clean_max_watermark is None:
        args.image_clean_max_watermark = 0.3 if hq else 0.5
    if args.image_clean_min_image_text_cosine is None:
        args.image_clean_min_image_text_cosine = 0.20 if hq else 0.15
    if args.image_clean_max_image_duplicate_cosine is None:
        args.image_clean_max_image_duplicate_cosine = 0.98 if hq else 0.985
    if str(args.image_size).strip() == "32":
        args.image_size = "1024" if hq else "512"
    if not str(args.image_size_buckets or "").strip():
        args.image_size_buckets = (
            "512x512,768x768,1024x768,768x1024,1024x1024,1344x768,768x1344"
            if hq else "512x512,768x512,512x768")
    if hq and not str(args.image_source_weights or "").strip():
        args.image_source_weights = (
            "text-to-image-2m-1024-10k=2.0,text-to-image-2m-512-2m=1.0,*=1.0")
    args.image_size_curriculum_frac = max(
        float(args.image_size_curriculum_frac), 0.6 if hq else 0.4)
    args.image_ae_arch = "hf-vae"
    args.image_latent_arch = "mmdit"
    args.image_geometry_cond = True
    if not args.image_ae_hf_model:
        args.image_ae_hf_model = "stabilityai/sdxl-vae"
    args.image_latent_downsample = 8
    args.image_latent_patch_size = max(int(args.image_latent_patch_size), 4 if hq else 2)
    args.image_latent_max_tokens = max(int(args.image_latent_max_tokens), 2048)
    if int(args.dim or 0) <= 0:
        args.dim = 768 if hq else 512
    elif hq:
        args.dim = max(int(args.dim), 768)
    if int(args.batch) == 16:
        args.batch = 2 if hq else 4
    if int(args.train_steps or 0) <= 0:
        args.train_steps = 40000 if hq else 20000
    args.image_dit_head_width_mult = max(int(args.image_dit_head_width_mult), 2)
    args.image_dit_depth = max(int(args.image_dit_depth), 12 if hq else 6)
    args.image_dit_heads = max(int(args.image_dit_heads), 12 if hq else 8)
    args.image_dit_qk_norm = True
    args.image_dit_attn_impl = "auto"
    args.image_dit_pos_embed = "rope2d"
    args.image_dit_mlp = "swiglu"
    args.image_flow_time_embed = "fourier"
    args.image_flow_time_embed_dim = max(int(args.image_flow_time_embed_dim), int(args.dim or 0))
    args.image_flow_checkpoint_blocks = True
    args.image_flow_self_condition = True
    args.image_flow_self_condition_p = max(float(args.image_flow_self_condition_p), 0.5)
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
    args.image_cond_drop = max(float(args.image_cond_drop), 0.1)
    args.image_cfg_scale = max(float(args.image_cfg_scale), 2.0 if hq else 1.5)
    args.image_cfg_rescale = max(float(args.image_cfg_rescale), 0.7)
    if hq and str(args.image_cfg_schedule or "constant") == "constant":
        args.image_cfg_schedule = "triangular"
    if hq and str(args.image_flow_boundary_mode or "none") == "none":
        args.image_flow_boundary_mode = "double-cosine"
    args.image_quality_weight = max(float(args.image_quality_weight), 0.5)
    args.image_quality_score_w = max(float(args.image_quality_score_w), 0.02)
    args.image_flow_quality_score_w = max(float(args.image_flow_quality_score_w), 0.01)
    args.image_quality_rank_w = max(float(args.image_quality_rank_w), 0.01)
    args.image_flow_quality_rank_w = max(float(args.image_flow_quality_rank_w), 0.005)
    args.image_quality_rank_margin = max(float(args.image_quality_rank_margin), 0.05)
    args.image_flow_consistency_w = max(float(args.image_flow_consistency_w), 0.05)
    args.image_flow_endpoint_w = max(float(args.image_flow_endpoint_w), 0.1)
    args.image_flow_frequency_w = max(float(args.image_flow_frequency_w), 0.02 if hq else 0.01)
    args.image_flow_straightness_w = max(
        float(args.image_flow_straightness_w), 0.02 if hq else 0.01)
    args.image_flow_multiscale_w = max(float(args.image_flow_multiscale_w), 0.03 if hq else 0.01)
    if not str(args.image_flow_multiscale_scales or "").strip():
        args.image_flow_multiscale_scales = "2,4"
    args.image_flow_noise_coupling = "sliced_ot"
    args.image_flow_noise_coupling_projections = max(
        int(args.image_flow_noise_coupling_projections), 4)
    args.image_flow_ema_decay = max(float(args.image_flow_ema_decay), 0.999)
    args.image_flow_distill_frac = max(float(args.image_flow_distill_frac), 0.1)
    args.image_flow_distill_w = max(float(args.image_flow_distill_w), 1.0)
    args.image_flow_guidance_distill_w = max(float(args.image_flow_guidance_distill_w), 0.1)
    args.image_flow_guidance_distill_cfg_scale = max(
        float(args.image_flow_guidance_distill_cfg_scale), float(args.image_cfg_scale))
    args.image_flow_guidance_distill_cfg_rescale = max(
        float(args.image_flow_guidance_distill_cfg_rescale), float(args.image_cfg_rescale))
    args.image_time_sampling = "adaptive" if hq else "logit-normal"
    if hq:
        args.image_time_adaptive_bins = max(int(args.image_time_adaptive_bins), 32)
        if str(args.image_time_adaptive_prior or "uniform") == "uniform":
            args.image_time_adaptive_prior = "mode"
        args.image_time_adaptive_prior_mix = max(
            float(args.image_time_adaptive_prior_mix), 0.25)
        args.image_time_adaptive_uniform_mix = max(
            float(args.image_time_adaptive_uniform_mix), 0.05)
        args.image_time_adaptive_min_prob = max(
            float(args.image_time_adaptive_min_prob), 0.001)
    args.image_flow_loss_weight = "soft-min-snr-v"
    args.image_sample_method = "heun"
    if (not str(args.image_sample_methods or "").strip()
            or str(args.image_sample_methods).strip() == "euler,heun,midpoint"):
        args.image_sample_methods = "heun,adaptive-heun,rk4" if hq else "heun,midpoint,rk4"
    if not str(args.image_sample_steps_sweep or "").strip() or str(
            args.image_sample_steps_sweep).strip() == "4,8,16":
        args.image_sample_steps_sweep = "16,24,32" if hq else "8,16,24"
    if hq and str(args.image_sample_schedule or "linear").strip() == "linear":
        args.image_sample_schedule = "karras"
    if (not str(args.image_sample_schedules or "").strip()
            or str(args.image_sample_schedules).strip() == "linear"):
        args.image_sample_schedules = (
            "karras,cosine,linear" if hq else str(args.image_sample_schedule))
    if not str(args.image_cfg_modes or "").strip():
        args.image_cfg_modes = "standard,cfgpp"
    if not str(args.image_cfg_schedules or "").strip():
        args.image_cfg_schedules = "triangular,constant" if hq else args.image_cfg_schedule
    args.image_sample_steps = max(int(args.image_sample_steps), 24 if hq else 16)
    if hq and float(args.image_sample_pixel_dynamic_threshold_percentile) <= 0.0:
        args.image_sample_pixel_dynamic_threshold_percentile = 0.995
    args.image_sample_pixel_dynamic_threshold_max = max(
        float(args.image_sample_pixel_dynamic_threshold_max), 1.0)
    args.image_eval_sweep = True
    args.image_eval_generated_candidates_per_prompt = max(
        int(args.image_eval_generated_candidates_per_prompt), 4 if hq else 2)
    args.image_sample_candidates_per_prompt = max(
        int(args.image_sample_candidates_per_prompt), 4 if hq else 2)
    args.image_sample_grid = True
    if not args.image_sample_manifest_out:
        args.image_sample_manifest_out = (
            "runs/image_latent_web_hf_vae_hq_generated.jsonl" if hq
            else "runs/image_latent_web_hf_vae_generated.jsonl")
    if hq and not args.image_sample_candidates_manifest_out:
        args.image_sample_candidates_manifest_out = path_with_suffix(
            args.image_sample_manifest_out, "_candidates.jsonl")
    if (hq and not args.image_preference_manifest
            and not getattr(args, "image_no_auto_preferences", False)):
        generated_preference = generated_preference_path_for_args(args)
        if local_preference_manifest_is_portable(generated_preference):
            args.image_preference_manifest = generated_preference
            args.image_auto_preference_manifest = True
    if args.image_preference_manifest:
        args.image_quality_score_steps = max(int(args.image_quality_score_steps), 1000)
        args.image_preference_w = max(float(args.image_preference_w), 0.5)
        args.image_flow_preference_w = max(float(args.image_flow_preference_w), 0.05)
        args.image_flow_preference_batch = max(int(args.image_flow_preference_batch), 1)
    args.image_eval_generated = True
    args.image_prompt_embed_backend = args.image_embed_backend
    args.image_prompt_embed_model = args.image_embed_model
    if not args.image_prompt_embed_text_sequence_model:
        args.image_prompt_embed_text_sequence_model = args.image_embed_text_sequence_model
    args.image_prompt_embed_text_sequence_max_length = max(
        int(args.image_prompt_embed_text_sequence_max_length),
        int(args.image_embed_text_sequence_max_length))
    if not args.image_sample_prompts:
        if hq:
            args.image_sample_prompts = (
                "a high-resolution editorial photo of a glass greenhouse after rain, "
                "with detailed plants, reflections, and natural morning light; "
                "a sharp studio product photograph of a translucent running shoe on "
                "matte graphite, with visible mesh texture and softbox shadows; "
                "a cinematic wide-angle photograph of a mountain observatory under a "
                "clear star field, with realistic stone, snow, and warm window light"
            )
        else:
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
    """Build the pod-side command for supported manifest/raw-data training jobs."""
    py = "python3 -u" if args.fast else "/root/fer-venv/bin/python -u"

    def mod(name):
        return f"{py} -m {name}"

    cmds = []
    if args.text_reading:
        cmds.append(text_reading_cmd(args, py))
    if args.text_reading and not (
            args.vision_understanding or args.image_latent or args.image_embed
            or args.image_fetch or args.image_caption or args.image_score
            or args.multimodal):
        return " && ".join(cmds)
    if (args.vision_understanding or args.image_latent or args.image_embed
            or args.image_fetch or args.image_caption or args.image_score
        or args.multimodal):
        effective_image_manifest = args.image_manifest
        if args.image_fetch:
            IFETCH = mod("thinking.image_fetch")
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
            ICAP = mod("thinking.image_caption")
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
            ISCORE = mod("thinking.image_score")
            score = (f"{ISCORE} --manifest {shlex_quote(effective_image_manifest)} "
                     f"--root {shlex_quote(args.image_root)} "
                     f"--backend {args.image_score_backend} "
                     f"--max-records {args.image_score_max_records} "
                     f"--image-size {args.image_score_image_size} "
                     f"--technical-w {args.image_score_technical_w} "
                     f"--alignment-w {args.image_score_alignment_w} "
                     f"--external-w {args.image_score_external_w} "
                     f"--pickscore-w {args.image_score_pickscore_w} "
                     f"--pickscore-model "
                     f"{shlex_quote(args.image_score_pickscore_model)} "
                     f"--pickscore-processor "
                     f"{shlex_quote(args.image_score_pickscore_processor)} "
                     f"--pickscore-device {args.image_score_pickscore_device} "
                     f"--pickscore-dtype {args.image_score_pickscore_dtype} "
                     f"--pickscore-batch {args.image_score_pickscore_batch} "
                     f"--pickscore-max-length {args.image_score_pickscore_max_length} "
                     f"--pickscore-normalize "
                     f"{args.image_score_pickscore_normalize} "
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
            IE = mod("thinking.image_embed")
            ID = mod("thinking.image_data")
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
                     f"--min-caption-unique-ratio "
                     f"{args.image_clean_min_caption_unique_ratio} "
                     f"--max-caption-token-frequency "
                     f"{args.image_clean_max_caption_token_frequency} "
                     f"--max-caption-token-run "
                     f"{args.image_clean_max_caption_token_run} "
                     f"--max-caption-char-run "
                     f"{args.image_clean_max_caption_char_run} "
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
            VU = mod("thinking.vision_understanding")
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
            IL = mod("thinking.image_latent")
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
            if args.image_sample_candidates_manifest_out:
                sample_manifest_args += (
                    f" --sample-candidates-manifest-out "
                    f"{shlex_quote(args.image_sample_candidates_manifest_out)}")
                if args.image_sample_candidates_image_dir:
                    sample_manifest_args += (
                        f" --sample-candidates-image-dir "
                        f"{shlex_quote(args.image_sample_candidates_image_dir)}")
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
                if int(args.image_sample_candidates_per_prompt) > 1:
                    prompt_grid_args += (
                        f" --cfg-scales {shlex_quote(args.image_cfg_sweep)} "
                        f"--cfg-rescales {shlex_quote(args.image_cfg_rescale_sweep)} "
                        f"--cfg-modes "
                        f"{shlex_quote(args.image_cfg_modes or args.image_cfg_mode)} "
                        f"--cfg-schedules "
                        f"{shlex_quote(args.image_cfg_schedules or args.image_cfg_schedule)} "
                        f"--sample-steps-list "
                        f"{shlex_quote(args.image_sample_steps_sweep)} "
                        f"--sample-methods {shlex_quote(args.image_sample_methods)} "
                        f"--sample-schedules {shlex_quote(args.image_sample_schedules)} "
                        f"--sample-churns {shlex_quote(args.image_sample_churns)} "
                        f"--eval-seeds {shlex_quote(args.image_eval_seeds)}")
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
                     f"--size-curriculum-frac {args.image_size_curriculum_frac} "
                     f"--hidden {args.dim or 64} --flow-arch {args.image_latent_arch} "
                     f"--dit-depth {args.image_dit_depth} "
                     f"--dit-heads {args.image_dit_heads} "
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
                     f"--flow-preference-w {args.image_flow_preference_w} "
                     f"--flow-preference-margin {args.image_flow_preference_margin} "
                     f"--flow-preference-batch {args.image_flow_preference_batch} "
                     f"--caption-vocab-max {args.image_caption_vocab_max} "
                     f"--caption-max-len {args.image_caption_max_len} "
                     f"--caption-cond-source {args.image_caption_cond_source} "
                     f"--image-crop-mode {args.image_crop_mode} "
                     f"--image-hflip-prob {args.image_hflip_prob} "
                     f"--cond-drop {args.image_cond_drop} "
                     f"--cfg-scale {args.image_cfg_scale} "
                     f"--cfg-rescale {args.image_cfg_rescale} "
                     f"--cfg-mode {args.image_cfg_mode} "
                     f"--cfg-schedule {args.image_cfg_schedule} "
                     f"--flow-boundary-mode {args.image_flow_boundary_mode} "
                     f"--cfg-interval {shlex_quote(args.image_cfg_interval)} "
                     f"--sample-steps {args.image_sample_steps} "
                     f"--sample-method {args.image_sample_method} "
                     f"--sample-schedule {args.image_sample_schedule} "
                     f"--sample-churn {args.image_sample_churn} "
                     f"--sample-churn-interval "
                     f"{shlex_quote(args.image_sample_churn_interval)} "
                     f"--sample-velocity-clip {args.image_sample_velocity_clip} "
                     f"--sample-latent-clip {args.image_sample_latent_clip} "
                     f"--sample-pixel-dynamic-threshold-percentile "
                     f"{args.image_sample_pixel_dynamic_threshold_percentile} "
                     f"--sample-pixel-dynamic-threshold-max "
                     f"{args.image_sample_pixel_dynamic_threshold_max} "
                     f"--eval-generated-candidates-per-prompt "
                     f"{args.image_eval_generated_candidates_per_prompt} "
                     f"--flow-consistency-w {args.image_flow_consistency_w} "
                     f"--flow-endpoint-w {args.image_flow_endpoint_w} "
                     f"--flow-frequency-w {args.image_flow_frequency_w} "
                     f"--flow-straightness-w {args.image_flow_straightness_w} "
                     f"--flow-multiscale-w {args.image_flow_multiscale_w} "
                     f"--flow-multiscale-scales {shlex_quote(args.image_flow_multiscale_scales)} "
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
                     f"--flow-time-embed {args.image_flow_time_embed} "
                     f"--flow-time-embed-dim {args.image_flow_time_embed_dim} "
                     f"--flow-self-condition-p {args.image_flow_self_condition_p} "
                     f"--time-sampling {args.image_time_sampling} "
                     f"--time-logit-mean {args.image_time_logit_mean} "
                     f"--time-logit-std {args.image_time_logit_std} "
                     f"--time-mode-scale {args.image_time_mode_scale} "
                     f"--time-curriculum-frac {args.image_time_curriculum_frac} "
                     f"--time-adaptive-bins {args.image_time_adaptive_bins} "
                     f"--time-adaptive-momentum {args.image_time_adaptive_momentum} "
                     f"--time-adaptive-uniform-mix "
                     f"{args.image_time_adaptive_uniform_mix} "
                     f"--time-adaptive-min-prob {args.image_time_adaptive_min_prob} "
                     f"--time-adaptive-loss-power "
                     f"{args.image_time_adaptive_loss_power} "
                     f"--time-adaptive-prior {args.image_time_adaptive_prior} "
                     f"--time-adaptive-prior-mix "
                     f"{args.image_time_adaptive_prior_mix} "
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
            if args.image_flow_self_condition:
                train += " --flow-self-condition"
            if args.image_geometry_cond:
                train += " --image-geometry-cond"
            if args.image_flow_cache_latents:
                train += " --flow-cache-latents"
            if args.image_sample_finite_guard:
                train += " --sample-finite-guard"
            if args.image_no_flow_loss_weight_normalize:
                train += " --no-flow-loss-weight-normalize"
            if args.image_flow_cache_dir:
                train += f" --flow-cache-dir {shlex_quote(args.image_flow_cache_dir)}"
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
                          f"--cfg-schedule {args.image_cfg_schedule} "
                          f"--cfg-schedules "
                          f"{shlex_quote(args.image_cfg_schedules or args.image_cfg_schedule)} "
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
                          f"--sample-pixel-dynamic-threshold-percentile "
                          f"{args.image_sample_pixel_dynamic_threshold_percentile} "
                          f"--sample-pixel-dynamic-threshold-max "
                          f"{args.image_sample_pixel_dynamic_threshold_max} "
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
                if args.image_sample_finite_guard:
                    eval_cmd += " --sample-finite-guard"
                if args.image_sample_grid:
                    eval_cmd += (f" --sample-grid-out {grid} "
                                 f"--sample-grid-samples {args.image_sample_grid_samples}"
                                 f"{sample_manifest_args}{prompt_grid_args}")
                train += eval_cmd
            if args.image_eval_generated:
                IEVAL = mod("thinking.image_eval")
                IGEN_EMBED = mod("thinking.image_embed")
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
                generated_candidates_manifest = args.image_sample_candidates_manifest_out
                generated_candidates_score_out = (
                    args.image_generated_candidates_score_out
                    or path_with_suffix(
                        generated_candidates_manifest, "_pickscore.jsonl")
                    if generated_candidates_manifest else ""
                )
                generated_candidates_score_report = (
                    args.image_generated_candidates_score_report_out
                    or path_with_suffix(
                        generated_candidates_manifest, "_pickscore_report.json")
                    if generated_candidates_manifest else ""
                )
                generated_preference_out = (
                    args.image_generated_preference_out
                    or path_with_suffix(
                        generated_candidates_manifest, "_preferences.jsonl")
                    if generated_candidates_manifest else ""
                )
                generated_preference_report = (
                    args.image_generated_preference_report_out
                    or path_with_suffix(
                        generated_candidates_manifest, "_preferences_report.json")
                    if generated_candidates_manifest else ""
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
                if generated_candidates_manifest:
                    IGEN_SCORE = mod("thinking.image_score")
                    IPREF = mod("thinking.image_preferences")
                    generated_candidates_root = (
                        os.path.dirname(generated_candidates_manifest) or ".")
                    generated_score = (
                        f" && {IGEN_SCORE} "
                        f"--manifest {shlex_quote(generated_candidates_manifest)} "
                        f"--root {shlex_quote(generated_candidates_root)} "
                        f"--backend {args.image_score_backend} "
                        f"--max-records {args.image_generated_eval_max_records} "
                        f"--image-size {args.image_score_image_size} "
                        f"--technical-w {args.image_score_technical_w} "
                        f"--alignment-w {args.image_score_alignment_w} "
                        f"--external-w {args.image_score_external_w} "
                        f"--pickscore-w {args.image_score_pickscore_w} "
                        f"--pickscore-model "
                        f"{shlex_quote(args.image_score_pickscore_model)} "
                        f"--pickscore-processor "
                        f"{shlex_quote(args.image_score_pickscore_processor)} "
                        f"--pickscore-device {args.image_score_pickscore_device} "
                        f"--pickscore-dtype {args.image_score_pickscore_dtype} "
                        f"--pickscore-batch {args.image_score_pickscore_batch} "
                        f"--pickscore-max-length "
                        f"{args.image_score_pickscore_max_length} "
                        f"--pickscore-normalize "
                        f"{args.image_score_pickscore_normalize} "
                        f"--out {shlex_quote(generated_candidates_score_out)} "
                        f"--report-out "
                        f"{shlex_quote(generated_candidates_score_report)}"
                    )
                    if args.image_score_external_sidecar:
                        generated_score += (
                            f" --external-sidecar "
                            f"{shlex_quote(args.image_score_external_sidecar)}")
                    if args.image_score_external_root:
                        generated_score += (
                            f" --external-root "
                            f"{shlex_quote(args.image_score_external_root)}")
                    if args.image_score_external_key:
                        generated_score += (
                            f" --external-key {args.image_score_external_key}")
                    if args.image_score_external_score_field:
                        generated_score += (
                            f" --external-score-field "
                            f"{shlex_quote(args.image_score_external_score_field)}")
                    if args.image_score_external_normalize:
                        generated_score += (
                            f" --external-normalize "
                            f"{args.image_score_external_normalize}")
                    generated_prefs = (
                        f" && {IPREF} "
                        f"--manifest {shlex_quote(generated_candidates_score_out)} "
                        f"--root {shlex_quote(generated_candidates_root)} "
                        f"--group-by prompt_id --mode top-bottom "
                        f"--min-score-gap "
                        f"{args.image_generated_preference_min_score_gap} "
                        f"--max-pairs-per-group "
                        f"{args.image_generated_preference_max_pairs_per_group} "
                        f"--max-pairs {args.image_generated_preference_max_pairs} "
                        f"--out {shlex_quote(generated_preference_out)} "
                        f"--report-out {shlex_quote(generated_preference_report)}"
                    )
                    train += generated_score + generated_prefs
            cmds.append(train)
        if args.multimodal:
            MM = mod("thinking.multimodal")
            mm_dim = args.multimodal_dim or args.dim or 96
            mm_cmd = (
                f"{MM} --manifest {shlex_quote(args.multimodal_manifest)} "
                f"--steps {args.train_steps or 400} "
                f"--batch {args.multimodal_batch} --dim {mm_dim} "
                f"--layers {args.multimodal_layers} --heads {args.multimodal_heads} "
                f"--lr {args.multimodal_lr} --log-every {args.multimodal_log_every} "
                f"--decode-w {args.multimodal_decode_w} "
                f"--agreement-w {args.multimodal_agreement_w} "
                f"--concept-tokens {args.multimodal_concept_tokens} "
                f"--fusion-layers {args.multimodal_fusion_layers} "
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
                f"--latent-concept-factorization-w "
                f"{args.multimodal_latent_concept_factorization_w} "
                f"--latent-concept-factorization-variance "
                f"{args.multimodal_latent_concept_factorization_variance} "
                f"--latent-concept-factorization-margin "
                f"{args.multimodal_latent_concept_factorization_margin} "
                f"--latent-concept-factorization-covariance-w "
                f"{args.multimodal_latent_concept_factorization_covariance_w} "
                f"--latent-concept-fer-w {args.multimodal_latent_concept_fer_w} "
                f"--latent-concept-fer-fragmentation-w "
                f"{args.multimodal_latent_concept_fer_fragmentation_w} "
                f"--latent-concept-fer-correlation-w "
                f"{args.multimodal_latent_concept_fer_correlation_w} "
                f"--latent-concept-fer-balance-w "
                f"{args.multimodal_latent_concept_fer_balance_w} "
                f"--latent-concept-fer-probe-n "
                f"{args.multimodal_latent_concept_fer_probe_n} "
                f"--latent-concept-fer-hard-max "
                f"{args.multimodal_latent_concept_fer_hard_max} "
                f"--latent-concept-fer-refresh-steps "
                f"{args.multimodal_latent_concept_fer_refresh_steps} "
                f"--latent-concept-discovery-probe-n "
                f"{args.multimodal_latent_concept_discovery_probe_n} "
                f"--latent-concept-discovery-hard-max "
                f"{args.multimodal_latent_concept_discovery_hard_max} "
                f"--latent-concept-discovery-refresh-steps "
                f"{args.multimodal_latent_concept_discovery_refresh_steps} "
                f"--latent-concept-memory-w "
                f"{args.multimodal_latent_concept_memory_w} "
                f"--latent-concept-memory-size "
                f"{args.multimodal_latent_concept_memory_size} "
                f"--latent-concept-memory-temperature "
                f"{args.multimodal_latent_concept_memory_temperature} "
                f"--latent-concept-memory-momentum "
                f"{args.multimodal_latent_concept_memory_momentum} "
                f"--latent-concept-memory-balance-w "
                f"{args.multimodal_latent_concept_memory_balance_w} "
                f"--latent-concept-consolidation-w "
                f"{args.multimodal_latent_concept_consolidation_w} "
                f"--latent-concept-consolidation-temperature "
                f"{args.multimodal_latent_concept_consolidation_temperature} "
                f"--latent-concept-consolidation-balance-w "
                f"{args.multimodal_latent_concept_consolidation_balance_w} "
                f"--latent-concept-consolidation-anchor-w "
                f"{args.multimodal_latent_concept_consolidation_anchor_w} "
                f"--latent-concept-consolidation-fer-w "
                f"{args.multimodal_latent_concept_consolidation_fer_w} "
                f"--latent-concept-discovery-w "
                f"{args.multimodal_latent_concept_discovery_w} "
                f"--latent-concept-discovery-curiosity-w "
                f"{args.multimodal_latent_concept_discovery_curiosity_w} "
                f"--latent-concept-discovery-graph-w "
                f"{args.multimodal_latent_concept_discovery_graph_w} "
                f"--latent-concept-discovery-cycle-w "
                f"{args.multimodal_latent_concept_discovery_cycle_w} "
                f"--latent-concept-discovery-bridge-w "
                f"{args.multimodal_latent_concept_discovery_bridge_w} "
                f"--latent-concept-discovery-fer-w "
                f"{args.multimodal_latent_concept_discovery_fer_w} "
                f"--latent-concept-reanalysis-w "
                f"{args.multimodal_latent_concept_reanalysis_w} "
                f"--latent-concept-reanalysis-graph-w "
                f"{args.multimodal_latent_concept_reanalysis_graph_w} "
                f"--latent-concept-reanalysis-cycle-w "
                f"{args.multimodal_latent_concept_reanalysis_cycle_w} "
                f"--latent-concept-reanalysis-bridge-w "
                f"{args.multimodal_latent_concept_reanalysis_bridge_w} "
                f"--latent-concept-reanalysis-fer-w "
                f"{args.multimodal_latent_concept_reanalysis_fer_w} "
                f"--latent-concept-reanalysis-cycle-consistency-w "
                f"{args.multimodal_latent_concept_reanalysis_cycle_consistency_w} "
                f"--latent-concept-association-w "
                f"{args.multimodal_latent_concept_association_w} "
                f"--latent-concept-association-temperature "
                f"{args.multimodal_latent_concept_association_temperature} "
                f"--latent-concept-association-decay "
                f"{args.multimodal_latent_concept_association_decay} "
                f"--latent-concept-association-target-power "
                f"{args.multimodal_latent_concept_association_target_power} "
                f"--latent-concept-association-self-loop-w "
                f"{args.multimodal_latent_concept_association_self_loop_w} "
                f"--latent-concept-association-transitive-steps "
                f"{args.multimodal_latent_concept_association_transitive_steps} "
                f"--latent-concept-association-transitive-w "
                f"{args.multimodal_latent_concept_association_transitive_w} "
                f"--latent-concept-composition-w "
                f"{args.multimodal_latent_concept_composition_w} "
                f"--latent-concept-composition-temperature "
                f"{args.multimodal_latent_concept_composition_temperature} "
                f"--latent-concept-composition-self-loop-w "
                f"{args.multimodal_latent_concept_composition_self_loop_w} "
                f"--latent-concept-composition-transitive-steps "
                f"{args.multimodal_latent_concept_composition_transitive_steps} "
                f"--latent-concept-composition-transitive-w "
                f"{args.multimodal_latent_concept_composition_transitive_w} "
                f"--latent-concept-composition-margin "
                f"{args.multimodal_latent_concept_composition_margin} "
                f"--latent-concept-graph-predict-w "
                f"{args.multimodal_latent_concept_graph_predict_w} "
                f"--latent-concept-graph-predict-temperature "
                f"{args.multimodal_latent_concept_graph_predict_temperature} "
                f"--latent-concept-graph-predict-self-loop-w "
                f"{args.multimodal_latent_concept_graph_predict_self_loop_w} "
                f"--latent-concept-graph-predict-transitive-steps "
                f"{args.multimodal_latent_concept_graph_predict_transitive_steps} "
                f"--latent-concept-graph-predict-transitive-w "
                f"{args.multimodal_latent_concept_graph_predict_transitive_w} "
                f"--latent-concept-graph-predict-target-power "
                f"{args.multimodal_latent_concept_graph_predict_target_power} "
                f"--latent-concept-bridge-w "
                f"{args.multimodal_latent_concept_bridge_w} "
                f"--latent-concept-sequence-w "
                f"{args.multimodal_latent_concept_sequence_w} "
                f"--latent-concept-sequence-batch "
                f"{args.multimodal_latent_concept_sequence_batch} "
                f"--latent-concept-sequence-temperature "
                f"{args.multimodal_latent_concept_sequence_temperature} "
                f"--latent-concept-neighborhood-w "
                f"{args.multimodal_latent_concept_neighborhood_w} "
                f"--latent-concept-neighborhood-temperature "
                f"{args.multimodal_latent_concept_neighborhood_temperature} "
                f"--latent-concept-neighborhood-margin "
                f"{args.multimodal_latent_concept_neighborhood_margin} "
                f"--latent-concept-transition-w "
                f"{args.multimodal_latent_concept_transition_w} "
                f"--latent-concept-transition-temperature "
                f"{args.multimodal_latent_concept_transition_temperature} "
                f"--latent-concept-transition-margin "
                f"{args.multimodal_latent_concept_transition_margin} "
                f"--latent-concept-cluster-w "
                f"{args.multimodal_latent_concept_cluster_w} "
                f"--latent-concept-cluster-temperature "
                f"{args.multimodal_latent_concept_cluster_temperature} "
                f"--latent-concept-cluster-margin "
                f"{args.multimodal_latent_concept_cluster_margin} "
                f"--latent-concept-cluster-min-size "
                f"{args.multimodal_latent_concept_cluster_min_size} "
                f"--view-tokens {args.multimodal_view_tokens} "
                f"--txt-tokens {args.multimodal_txt_tokens} "
                f"--trunk-arch {args.multimodal_trunk_arch} "
                f"--trunk-width {args.multimodal_trunk_width} "
                f"--trunk-depth {args.multimodal_trunk_depth} "
                f"--text-layers {args.multimodal_text_layers} "
                f"--modality-dropout {args.multimodal_dropout} "
                f"--eval-n {args.multimodal_eval_n} "
                f"--selection-rounds {args.multimodal_selection_rounds} "
                f"--selection-score-metric "
                f"{args.multimodal_selection_score_metric} "
                f"--selection-score-margin-w "
                f"{args.multimodal_selection_score_margin_w} "
                f"--selection-score-min-delta "
                f"{args.multimodal_selection_score_min_delta} "
                f"--selection-score-patience "
                f"{args.multimodal_selection_score_patience} "
                f"--selection-eval-n {args.multimodal_selection_eval_n} "
                f"--selection-insight-accept-w "
                f"{args.multimodal_selection_insight_accept_w} "
                f"--selection-insight-min-delta "
                f"{args.multimodal_selection_insight_min_delta} "
                f"--out runs/m0_multimodal.json --checkpoint runs/m0_multimodal.pt"
            )
            if args.multimodal_root:
                mm_cmd += f" --root {shlex_quote(args.multimodal_root)}"
            if args.multimodal_text_checkpoint:
                mm_cmd += (
                    f" --text-checkpoint "
                    f"{shlex_quote(args.multimodal_text_checkpoint)}")
            mm_cmd += " --select-best" if args.multimodal_select_best else " --no-select-best"
            cmds.append(mm_cmd)
        return " && ".join(cmds)
    if cmds:
        return " && ".join(cmds)
    raise ValueError(
        "no supported job selected; use --text-reading, --image-fetch, --image-caption, "
        "--image-score, --image-embed, --image-latent, --vision-understanding, or --multimodal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="fer-thinking")
    ap.add_argument("--batch", type=int, default=16,
                    help="H100 batch at block 3200 (8-loop backward graph holds 8 LxL attention "
                         "maps per head -- batch 64 OOMs 80GB)")
    ap.add_argument("--train-steps", type=int, default=0, help="override training steps")
    ap.add_argument("--fast", action="store_true",
                    help="use image-native python instead of the managed venv")
    ap.add_argument("--dim", type=int, default=0, help="model width override")
    ap.add_argument("--text-reading", action="store_true", dest="text_reading",
                    help="train thinking.text on raw reading corpora with latent discovery")
    ap.add_argument("--reading-data", action="append",
                    help="raw reading JSON/JSONL/TXT corpus or URL for --text-reading")
    ap.add_argument("--upload-reading-data", action="store_true",
                    dest="upload_reading_data",
                    help="upload repo-relative --reading-data files before running")
    ap.add_argument("--reading-checkpoint", default="", dest="reading_checkpoint",
                    help="optional thinking.text checkpoint to continue with raw reading")
    ap.add_argument("--reading-out-checkpoint", default="",
                    dest="reading_out_checkpoint",
                    help="checkpoint path written by raw-reading continuation")
    ap.add_argument("--text-reading-checkpoint-out",
                    default="runs/text_raw_reading_runpod.pt",
                    dest="text_reading_checkpoint_out",
                    help="checkpoint path written by fresh raw-reading training")
    ap.add_argument("--text-reading-out", default="runs/text_raw_reading_runpod.json",
                    dest="text_reading_out",
                    help="JSON report path written by raw-reading training")
    ap.add_argument("--text-reading-steps", type=int, default=40000,
                    dest="text_reading_steps",
                    help="raw-reading steps used when --train-steps is not set")
    ap.add_argument("--text-reading-dim", type=int, default=0,
                    dest="text_reading_dim",
                    help="raw-reading model width; 0 uses --dim or 256")
    ap.add_argument("--text-reading-layers", type=int, default=4,
                    dest="text_reading_layers")
    ap.add_argument("--text-reading-heads", type=int, default=8,
                    dest="text_reading_heads")
    ap.add_argument("--text-reading-seed", type=int, default=0,
                    dest="text_reading_seed")
    ap.add_argument("--text-reading-encoder-arch", default="transformer",
                    choices=("transformer", "standard", "relational", "abstractor"),
                    dest="text_reading_encoder_arch")
    ap.add_argument("--text-reading-encoder-layers", type=int, default=2,
                    dest="text_reading_encoder_layers")
    ap.add_argument("--text-reading-latent-concept-slots", type=int, default=0,
                    dest="text_reading_latent_concept_slots",
                    help="raw-reading latent concept slots; 0 lets thinking.text choose")
    ap.add_argument("--text-reading-latent-concept-layers", type=int, default=1,
                    dest="text_reading_latent_concept_layers")
    ap.add_argument("--text-reading-latent-concept-prefix", action="store_true",
                    dest="text_reading_latent_concept_prefix")
    ap.add_argument("--text-reading-latent-concept-refine", action="store_true",
                    dest="text_reading_latent_concept_refine")
    ap.add_argument("--text-reading-latent-concept-refine-gate-init",
                    type=float, default=-2.0,
                    dest="text_reading_latent_concept_refine_gate_init")
    ap.add_argument("--reading-replay-data", action="append",
                    dest="reading_replay_data",
                    help="raw reading replay corpus for checkpoint continuation")
    ap.add_argument("--reading-replay-w", type=float, default=0.0,
                    dest="reading_replay_w")
    ap.add_argument("--reading-replay-batch", type=int, default=0,
                    dest="reading_replay_batch")
    ap.add_argument("--reading-replay-retention-w", type=float, default=0.0,
                    dest="reading_replay_retention_w")
    ap.add_argument("--reading-text-field", default="text", dest="reading_text_field")
    ap.add_argument("--reading-max-tokens", type=int, default=128,
                    dest="reading_max_tokens")
    ap.add_argument("--reading-min-tokens", type=int, default=8,
                    dest="reading_min_tokens")
    ap.add_argument("--reading-eval-frac", type=float, default=0.10,
                    dest="reading_eval_frac")
    ap.add_argument("--reading-eval-n", type=int, default=256,
                    dest="reading_eval_n")
    ap.add_argument("--reading-lr", type=float, default=1e-3, dest="reading_lr")
    ap.add_argument("--reading-token-drop", type=float, default=0.15,
                    dest="reading_token_drop")
    ap.add_argument("--reading-token-replace", type=float, default=0.05,
                    dest="reading_token_replace")
    ap.add_argument("--reading-feature-dropout", type=float, default=0.1,
                    dest="reading_feature_dropout")
    ap.add_argument("--reading-context-target-w", type=float, default=0.1,
                    dest="reading_context_target_w")
    ap.add_argument("--reading-context-keep-p", type=float, default=0.5,
                    dest="reading_context_keep_p")
    ap.add_argument("--reading-context-target-temperature", type=float,
                    default=0.1, dest="reading_context_target_temperature")
    ap.add_argument("--reading-sequence-w", type=float, default=0.05,
                    dest="reading_sequence_w")
    ap.add_argument("--reading-sequence-batch", type=int, default=0,
                    dest="reading_sequence_batch")
    ap.add_argument("--reading-sequence-temperature", type=float, default=0.1,
                    dest="reading_sequence_temperature")
    ap.add_argument("--reading-factorization-w", type=float, default=0.05,
                    dest="reading_factorization_w")
    ap.add_argument("--reading-factorization-variance", type=float, default=0.05,
                    dest="reading_factorization_variance")
    ap.add_argument("--reading-factorization-margin", type=float, default=0.2,
                    dest="reading_factorization_margin")
    ap.add_argument("--reading-factorization-covariance-w", type=float,
                    default=0.05, dest="reading_factorization_covariance_w")
    ap.add_argument("--reading-fer-w", type=float, default=0.0,
                    dest="reading_fer_w")
    ap.add_argument("--reading-fer-fragmentation-w", type=float, default=1.0,
                    dest="reading_fer_fragmentation_w")
    ap.add_argument("--reading-fer-correlation-w", type=float, default=1.0,
                    dest="reading_fer_correlation_w")
    ap.add_argument("--reading-fer-balance-w", type=float, default=0.1,
                    dest="reading_fer_balance_w")
    ap.add_argument("--reading-memory-w", type=float, default=0.05,
                    dest="reading_memory_w")
    ap.add_argument("--reading-memory-size", type=int, default=64,
                    dest="reading_memory_size")
    ap.add_argument("--reading-memory-temperature", type=float, default=0.1,
                    dest="reading_memory_temperature")
    ap.add_argument("--reading-memory-momentum", type=float, default=0.95,
                    dest="reading_memory_momentum")
    ap.add_argument("--reading-memory-balance-w", type=float, default=0.01,
                    dest="reading_memory_balance_w")
    ap.add_argument("--reading-consolidation-w", type=float, default=0.0,
                    dest="reading_consolidation_w")
    ap.add_argument("--reading-consolidation-temperature", type=float, default=0.1,
                    dest="reading_consolidation_temperature")
    ap.add_argument("--reading-consolidation-balance-w", type=float, default=0.01,
                    dest="reading_consolidation_balance_w")
    ap.add_argument("--reading-consolidation-anchor-w", type=float, default=1.0,
                    dest="reading_consolidation_anchor_w")
    ap.add_argument("--reading-consolidation-fer-w", type=float, default=0.0,
                    dest="reading_consolidation_fer_w")
    ap.add_argument("--reading-discovery-w", type=float, default=0.0,
                    dest="reading_discovery_w")
    ap.add_argument("--reading-discovery-curiosity-w", type=float, default=1.0,
                    dest="reading_discovery_curiosity_w")
    ap.add_argument("--reading-discovery-graph-w", type=float, default=1.0,
                    dest="reading_discovery_graph_w")
    ap.add_argument("--reading-discovery-cycle-w", type=float, default=1.0,
                    dest="reading_discovery_cycle_w")
    ap.add_argument("--reading-discovery-bridge-w", type=float, default=1.0,
                    dest="reading_discovery_bridge_w")
    ap.add_argument("--reading-discovery-fer-w", type=float, default=0.0,
                    dest="reading_discovery_fer_w")
    ap.add_argument("--reading-reanalysis-w", type=float, default=0.0,
                    dest="reading_reanalysis_w")
    ap.add_argument("--reading-reanalysis-graph-w", type=float, default=1.0,
                    dest="reading_reanalysis_graph_w")
    ap.add_argument("--reading-reanalysis-cycle-w", type=float, default=0.5,
                    dest="reading_reanalysis_cycle_w")
    ap.add_argument("--reading-reanalysis-bridge-w", type=float, default=0.5,
                    dest="reading_reanalysis_bridge_w")
    ap.add_argument("--reading-reanalysis-fer-w", type=float, default=0.0,
                    dest="reading_reanalysis_fer_w")
    ap.add_argument("--reading-association-w", type=float, default=0.05,
                    dest="reading_association_w")
    ap.add_argument("--reading-association-temperature", type=float,
                    default=0.1, dest="reading_association_temperature")
    ap.add_argument("--reading-association-decay", type=float, default=0.99,
                    dest="reading_association_decay")
    ap.add_argument("--reading-association-target-power", type=float,
                    default=1.0, dest="reading_association_target_power")
    ap.add_argument("--reading-association-self-loop-w", type=float,
                    default=0.05, dest="reading_association_self_loop_w")
    ap.add_argument("--reading-association-transitive-steps", type=int,
                    default=2, dest="reading_association_transitive_steps")
    ap.add_argument("--reading-association-transitive-w", type=float,
                    default=0.1, dest="reading_association_transitive_w")
    ap.add_argument("--reading-composition-w", type=float, default=0.0,
                    dest="reading_composition_w")
    ap.add_argument("--reading-composition-temperature", type=float,
                    default=0.1, dest="reading_composition_temperature")
    ap.add_argument("--reading-composition-self-loop-w", type=float,
                    default=0.0, dest="reading_composition_self_loop_w")
    ap.add_argument("--reading-composition-transitive-steps", type=int,
                    default=2, dest="reading_composition_transitive_steps")
    ap.add_argument("--reading-composition-transitive-w", type=float,
                    default=0.1, dest="reading_composition_transitive_w")
    ap.add_argument("--reading-composition-margin", type=float, default=0.0,
                    dest="reading_composition_margin")
    ap.add_argument("--reading-graph-predict-w", type=float, default=0.0,
                    dest="reading_graph_predict_w")
    ap.add_argument("--reading-graph-predict-temperature", type=float,
                    default=0.1, dest="reading_graph_predict_temperature")
    ap.add_argument("--reading-graph-predict-self-loop-w", type=float,
                    default=0.05, dest="reading_graph_predict_self_loop_w")
    ap.add_argument("--reading-graph-predict-transitive-steps", type=int,
                    default=2, dest="reading_graph_predict_transitive_steps")
    ap.add_argument("--reading-graph-predict-transitive-w", type=float,
                    default=0.1, dest="reading_graph_predict_transitive_w")
    ap.add_argument("--reading-graph-predict-target-power", type=float,
                    default=1.0, dest="reading_graph_predict_target_power")
    ap.add_argument("--reading-graph-cycle-w", type=float, default=0.0,
                    dest="reading_graph_cycle_w")
    ap.add_argument("--reading-graph-cycle-temperature", type=float, default=0.1,
                    dest="reading_graph_cycle_temperature")
    ap.add_argument("--reading-graph-cycle-self-loop-w", type=float, default=0.05,
                    dest="reading_graph_cycle_self_loop_w")
    ap.add_argument("--reading-graph-cycle-transitive-steps", type=int,
                    default=2, dest="reading_graph_cycle_transitive_steps")
    ap.add_argument("--reading-graph-cycle-transitive-w", type=float, default=0.1,
                    dest="reading_graph_cycle_transitive_w")
    ap.add_argument("--reading-graph-cycle-target-power", type=float,
                    default=1.0, dest="reading_graph_cycle_target_power")
    ap.add_argument("--reading-graph-cycle-consistency-w", type=float,
                    default=0.5, dest="reading_graph_cycle_consistency_w")
    ap.add_argument("--reading-bridge-w", type=float, default=0.0,
                    dest="reading_bridge_w")
    ap.add_argument("--reading-neighborhood-w", type=float, default=0.0,
                    dest="reading_neighborhood_w")
    ap.add_argument("--reading-neighborhood-batch", type=int, default=0,
                    dest="reading_neighborhood_batch")
    ap.add_argument("--reading-neighborhood-probe-n", type=int, default=0,
                    dest="reading_neighborhood_probe_n")
    ap.add_argument("--reading-neighborhood-refresh-steps", type=int, default=0,
                    dest="reading_neighborhood_refresh_steps")
    ap.add_argument("--reading-neighborhood-temperature", type=float, default=0.1,
                    dest="reading_neighborhood_temperature")
    ap.add_argument("--reading-neighborhood-margin", type=float, default=0.0,
                    dest="reading_neighborhood_margin")
    ap.add_argument("--reading-transition-w", type=float, default=0.05,
                    dest="reading_transition_w")
    ap.add_argument("--reading-transition-batch", type=int, default=0,
                    dest="reading_transition_batch")
    ap.add_argument("--reading-transition-temperature", type=float, default=0.1,
                    dest="reading_transition_temperature")
    ap.add_argument("--reading-transition-margin", type=float, default=0.0,
                    dest="reading_transition_margin")
    ap.add_argument("--reading-cluster-w", type=float, default=0.0,
                    dest="reading_cluster_w")
    ap.add_argument("--reading-cluster-batch", type=int, default=0,
                    dest="reading_cluster_batch")
    ap.add_argument("--reading-cluster-probe-n", type=int, default=0,
                    dest="reading_cluster_probe_n")
    ap.add_argument("--reading-cluster-refresh-steps", type=int, default=0,
                    dest="reading_cluster_refresh_steps")
    ap.add_argument("--reading-cluster-temperature", type=float, default=0.1,
                    dest="reading_cluster_temperature")
    ap.add_argument("--reading-cluster-margin", type=float, default=0.0,
                    dest="reading_cluster_margin")
    ap.add_argument("--reading-cluster-min-size", type=int, default=2,
                    dest="reading_cluster_min_size")
    ap.add_argument("--reading-study-strategy", default="auto",
                    choices=("random", "errors", "fer", "curiosity", "sequence",
                             "graph", "cycle", "discovery", "auto"),
                    dest="reading_study_strategy")
    ap.add_argument("--reading-study-probe-n", type=int, default=0,
                    dest="reading_study_probe_n")
    ap.add_argument("--reading-study-hard-max", type=int, default=0,
                    dest="reading_study_hard_max")
    ap.add_argument("--reading-study-refresh-steps", type=int, default=0,
                    dest="reading_study_refresh_steps")
    ap.add_argument("--reading-study-select-best",
                    action=argparse.BooleanOptionalAction, default=True,
                    dest="reading_study_select_best")
    ap.add_argument("--reading-study-rounds", type=int, default=1,
                    dest="reading_study_rounds")
    ap.add_argument("--reading-study-score-metric", default="mastery",
                    choices=("view", "context", "sequence", "neighborhood",
                             "cluster", "fer", "bridge", "both", "min", "all",
                             "balanced", "mastery"),
                    dest="reading_study_score_metric")
    ap.add_argument("--reading-study-score-margin-w", type=float, default=0.1,
                    dest="reading_study_score_margin_w")
    ap.add_argument("--reading-study-score-min-delta", type=float, default=0.0,
                    dest="reading_study_score_min_delta")
    ap.add_argument("--reading-study-score-patience", type=int, default=0,
                    dest="reading_study_score_patience")
    ap.add_argument("--reading-study-insight-accept-w", "--reading-study-insight-w",
                    type=float, default=0.25,
                    dest="reading_study_insight_accept_w")
    ap.add_argument("--reading-study-insight-min-delta", type=float, default=0.0,
                    dest="reading_study_insight_min_delta")
    ap.add_argument("--ref", default="HEAD",
                    help="deploy this git ref (pinned commit); '' = live tree")
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
                    choices=("none", "web-hf-vae", "web-hf-vae-hq"),
                    dest="image_quality_preset",
                    help=("apply a coherent high-quality image training profile; "
                          "web-hf-vae trains a broad 512/768 web profile; "
                          "web-hf-vae-hq trains a stricter 1024 multi-aspect profile"))
    ap.add_argument("--image-no-auto-preferences", action="store_true",
                    dest="image_no_auto_preferences",
                    help=("do not auto-reuse previous generated candidate preferences "
                          "in the high-quality image preset"))
    ap.add_argument("--image-fetch", action="store_true", dest="image_fetch",
                    help="stream a captioned web image shard into a local manifest on the pod")
    ap.add_argument("--image-fetch-source", default="text-to-image-2m-1024-10k",
                    choices=("text-to-image-2m-1024-10k", "text-to-image-2m-hq-mix",
                             "flux-1024-10k",
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
                    choices=("stats", "embedding", "external", "pickscore", "ensemble"),
                    dest="image_score_backend",
                    help=("quality source for --image-score: technical stats, existing "
                          "image/text embeddings, external reward sidecar, PickScore, "
                          "or ensemble"))
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
    ap.add_argument("--image-score-pickscore-w", type=float, default=0.0,
                    dest="image_score_pickscore_w",
                    help="ensemble weight for PickScore prompt-image reward scoring")
    ap.add_argument("--image-score-pickscore-model",
                    default="yuvalkirstain/PickScore_v1",
                    dest="image_score_pickscore_model",
                    help="Hugging Face model id for PickScore reward scoring")
    ap.add_argument("--image-score-pickscore-processor",
                    default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                    dest="image_score_pickscore_processor",
                    help="Hugging Face processor id for PickScore reward scoring")
    ap.add_argument("--image-score-pickscore-device", default="cuda",
                    dest="image_score_pickscore_device",
                    help="device used by PickScore reward scoring")
    ap.add_argument("--image-score-pickscore-dtype", default="auto",
                    choices=("auto", "fp32", "fp16", "bf16"),
                    dest="image_score_pickscore_dtype",
                    help="dtype used by PickScore reward scoring")
    ap.add_argument("--image-score-pickscore-batch", type=int, default=8,
                    dest="image_score_pickscore_batch",
                    help="batch size for PickScore reward scoring")
    ap.add_argument("--image-score-pickscore-max-length", type=int, default=77,
                    dest="image_score_pickscore_max_length",
                    help="tokenizer max length for PickScore prompt text")
    ap.add_argument("--image-score-pickscore-normalize", default="auto",
                    choices=("auto", "minmax", "none"),
                    dest="image_score_pickscore_normalize",
                    help="normalization for raw PickScore rewards before writing quality_score")
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
                    help="manifest inspection report written after embedding merge")
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
    ap.add_argument("--image-clean-min-caption-unique-ratio", type=float, default=0.0,
                    dest="image_clean_min_caption_unique_ratio",
                    help=("reject cleaned captions below this unique-token ratio; "
                          "0 disables"))
    ap.add_argument("--image-clean-max-caption-token-frequency", type=float, default=0.0,
                    dest="image_clean_max_caption_token_frequency",
                    help=("reject cleaned captions dominated by one token above this "
                          "fraction; 0 disables"))
    ap.add_argument("--image-clean-max-caption-token-run", type=float, default=0.0,
                    dest="image_clean_max_caption_token_run",
                    help=("reject cleaned captions with repeated-token runs above this "
                          "fraction; 0 disables"))
    ap.add_argument("--image-clean-max-caption-char-run", type=float, default=0.0,
                    dest="image_clean_max_caption_char_run",
                    help=("reject cleaned captions with repeated-character runs above "
                          "this fraction; 0 disables"))
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
    ap.add_argument("--image-size-curriculum-frac", type=float, default=0.0,
                    dest="image_size_curriculum_frac",
                    help=("fraction of image flow training that progressively unlocks "
                          "larger size buckets; 0 disables"))
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
    ap.add_argument("--image-dit-depth", type=int, default=3,
                    dest="image_dit_depth",
                    help="number of latent DiT/MM-DiT transformer blocks")
    ap.add_argument("--image-dit-heads", type=int, default=4,
                    dest="image_dit_heads",
                    help="attention heads for latent DiT/MM-DiT transformer blocks")
    ap.add_argument("--image-dit-qk-norm", action="store_true",
                    dest="image_dit_qk_norm",
                    help="enable per-head QK RMSNorm in latent image MM-DiT attention")
    ap.add_argument("--image-dit-attn-impl", default="auto",
                    choices=("manual", "sdpa", "linear", "auto"), dest="image_dit_attn_impl",
                    help=("latent image MM-DiT attention implementation; auto uses exact "
                          "PyTorch SDPA when available, linear is an explicit approximation"))
    ap.add_argument("--image-dit-pos-embed", default="learned",
                    choices=("learned", "sincos2d", "rope2d"), dest="image_dit_pos_embed",
                    help=("latent image DiT positional embedding; rope2d requires "
                          "--image-latent-arch mmdit"))
    ap.add_argument("--image-dit-mlp", default="gelu",
                    choices=("gelu", "swiglu"), dest="image_dit_mlp",
                    help="latent image CrossDiT/MM-DiT feed-forward block")
    ap.add_argument("--image-flow-time-embed", default="scalar",
                    choices=("scalar", "fourier"), dest="image_flow_time_embed",
                    help=("latent image DiT/MM-DiT timestep embedding; scalar keeps "
                          "legacy behavior"))
    ap.add_argument("--image-flow-time-embed-dim", type=int, default=0,
                    dest="image_flow_time_embed_dim",
                    help=("Fourier timestep embedding width for latent image flow; "
                          "0 uses hidden width in the HQ preset"))
    ap.add_argument("--image-flow-checkpoint-blocks", action="store_true",
                    dest="image_flow_checkpoint_blocks",
                    help="checkpoint latent image transformer blocks during flow training")
    ap.add_argument("--image-flow-self-condition", action="store_true",
                    dest="image_flow_self_condition",
                    help="feed previous clean endpoint estimates into latent transformer flows")
    ap.add_argument("--image-flow-self-condition-p", type=float, default=0.5,
                    dest="image_flow_self_condition_p",
                    help="probability of endpoint self-conditioning during image flow training")
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
    ap.add_argument("--image-flow-preference-w", type=float, default=0.0,
                    dest="image_flow_preference_w",
                    help=("direct chosen/rejected preference loss weight on the image "
                          "latent-flow generator"))
    ap.add_argument("--image-flow-preference-margin", type=float, default=0.0,
                    dest="image_flow_preference_margin",
                    help="minimum velocity-loss gap for image direct flow preference pairs")
    ap.add_argument("--image-flow-preference-batch", type=int, default=0,
                    dest="image_flow_preference_batch",
                    help="preference pairs per image flow update; 0 reuses image batch")
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
    ap.add_argument("--image-geometry-cond", action="store_true",
                    dest="image_geometry_cond",
                    help=("append fixed crop/flip/pad-aware geometry conditioning "
                          "to latent image text conditions"))
    ap.add_argument("--image-crop-mode", default="center",
                    choices=("center", "random", "none", "pad"), dest="image_crop_mode",
                    help="crop mode for manifest image training")
    ap.add_argument("--image-hflip-prob", type=float, default=0.0,
                    dest="image_hflip_prob",
                    help="random horizontal flip probability for manifest image training")
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
    ap.add_argument("--image-cfg-schedule", default="constant",
                    choices=("constant", "linear", "cosine", "triangular"),
                    dest="image_cfg_schedule",
                    help="latent image extra-CFG time schedule")
    ap.add_argument("--image-cfg-schedules", default="",
                    dest="image_cfg_schedules",
                    help=("comma-separated latent image CFG schedules for eval sweeps "
                          "and prompt candidates"))
    ap.add_argument("--image-flow-boundary-mode", default="none",
                    choices=("none", "right-linear", "double-linear", "double-cosine"),
                    dest="image_flow_boundary_mode",
                    help=("boundary-conditioned rectified-flow velocity parameterization "
                          "passed to thinking.image_latent"))
    ap.add_argument("--image-cfg-interval", default="0.0,1.0", dest="image_cfg_interval",
                    help="latent image CFG active interval formatted start,end")
    ap.add_argument("--image-sample-steps", type=int, default=4, dest="image_sample_steps",
                    help="ODE sampling steps for latent image evaluation")
    ap.add_argument("--image-sample-method", default="euler",
                    choices=("euler", "heun", "midpoint", "rk4", "adaptive-heun"),
                    dest="image_sample_method",
                    help="latent image ODE sampler method")
    ap.add_argument("--image-sample-methods", default="euler,heun,midpoint",
                    dest="image_sample_methods",
                    help=("comma-separated latent image sampler methods for eval sweeps "
                          "and prompt candidates"))
    ap.add_argument("--image-sample-schedule", default="linear",
                    choices=IMAGE_SAMPLE_SCHEDULES,
                    dest="image_sample_schedule",
                    help="latent image timestep placement schedule")
    ap.add_argument("--image-sample-schedules", default="linear",
                    dest="image_sample_schedules",
                    help=("comma-separated latent image timestep schedules for eval sweeps "
                          "and prompt candidates"))
    ap.add_argument("--image-sample-churn", type=float, default=0.0,
                    dest="image_sample_churn",
                    help="stochastic latent sampler churn; 0 keeps deterministic ODE sampling")
    ap.add_argument("--image-sample-churns", default="0.0",
                    dest="image_sample_churns",
                    help=("comma-separated stochastic sampler churn values for eval sweeps "
                          "and prompt candidates"))
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
    ap.add_argument("--image-sample-pixel-dynamic-threshold-percentile", type=float,
                    default=0.0,
                    dest="image_sample_pixel_dynamic_threshold_percentile",
                    help=("decoded-pixel dynamic threshold percentile in (0,1]; "
                          "0 disables, HQ preset uses 0.995"))
    ap.add_argument("--image-sample-pixel-dynamic-threshold-max", type=float,
                    default=1.0,
                    dest="image_sample_pixel_dynamic_threshold_max",
                    help="decoded-pixel dynamic threshold target max absolute value")
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
    ap.add_argument("--image-sample-candidates-manifest-out", default="",
                    dest="image_sample_candidates_manifest_out",
                    help=("optional JSONL manifest path for all generated prompt candidates "
                          "before internal selection"))
    ap.add_argument("--image-sample-candidates-image-dir", default="",
                    dest="image_sample_candidates_image_dir",
                    help=("optional directory for all generated prompt candidate PPM samples; "
                          "default is derived from --image-sample-candidates-manifest-out"))
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
    ap.add_argument("--image-generated-candidates-score-out", default="",
                    dest="image_generated_candidates_score_out",
                    help=("optional scored manifest for generated prompt candidates; "
                          "default derives from --image-sample-candidates-manifest-out"))
    ap.add_argument("--image-generated-candidates-score-report-out", default="",
                    dest="image_generated_candidates_score_report_out",
                    help=("optional score report for generated prompt candidates; "
                          "default derives from --image-sample-candidates-manifest-out"))
    ap.add_argument("--image-generated-preference-out", default="",
                    dest="image_generated_preference_out",
                    help=("optional chosen/rejected preference pairs mined from generated "
                          "prompt candidates"))
    ap.add_argument("--image-generated-preference-report-out", default="",
                    dest="image_generated_preference_report_out",
                    help="optional report for generated candidate preference mining")
    ap.add_argument("--image-generated-preference-min-score-gap", type=float, default=0.02,
                    dest="image_generated_preference_min_score_gap",
                    help="minimum PickScore/quality gap for generated candidate preferences")
    ap.add_argument("--image-generated-preference-max-pairs-per-group", type=int, default=1,
                    dest="image_generated_preference_max_pairs_per_group",
                    help="maximum generated preference pairs per prompt group; 0 means all")
    ap.add_argument("--image-generated-preference-max-pairs", type=int, default=0,
                    dest="image_generated_preference_max_pairs",
                    help="maximum generated preference pairs total; 0 means all")
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
    ap.add_argument("--image-eval-text-guidance-sweep", default="0.0",
                    dest="image_eval_text_guidance_sweep",
                    help="comma-separated text guidance weights for manifest image eval sweeps")
    ap.add_argument("--image-eval-feature-guidance-sweep", default="0.0",
                    dest="image_eval_feature_guidance_sweep",
                    help="comma-separated feature guidance weights for manifest image eval sweeps")
    ap.add_argument("--image-eval-quality-guidance-sweep", default="0.0",
                    dest="image_eval_quality_guidance_sweep",
                    help="comma-separated quality guidance weights for manifest image eval sweeps")
    ap.add_argument("--image-flow-consistency-w", type=float, default=0.0,
                    dest="image_flow_consistency_w",
                    help="same-path endpoint consistency loss weight for latent image flow")
    ap.add_argument("--image-flow-endpoint-w", type=float, default=0.0,
                    dest="image_flow_endpoint_w",
                    help="direct clean-endpoint latent prediction loss weight for image flow")
    ap.add_argument("--image-flow-frequency-w", type=float, default=0.0,
                    dest="image_flow_frequency_w",
                    help="frequency-domain clean-endpoint latent loss weight for image flow")
    ap.add_argument("--image-flow-straightness-w", type=float, default=0.0,
                    dest="image_flow_straightness_w",
                    help="same-chord velocity straightness loss weight for image flow")
    ap.add_argument("--image-flow-multiscale-w", type=float, default=0.0,
                    dest="image_flow_multiscale_w",
                    help="coarse-to-fine downsampled velocity loss weight for image flow")
    ap.add_argument("--image-flow-multiscale-scales", default="2,4",
                    dest="image_flow_multiscale_scales",
                    help="comma-separated latent downsample scales for image flow multiscale loss")
    ap.add_argument("--image-flow-noise-coupling", default="random",
                    choices=("random", "sliced_ot"), dest="image_flow_noise_coupling",
                    help="source-noise/data pairing for image flow matching")
    ap.add_argument("--image-flow-noise-coupling-projections", type=int, default=1,
                    dest="image_flow_noise_coupling_projections",
                    help="random projections to try for sliced_ot image flow noise coupling")
    ap.add_argument("--image-flow-distill-steps", type=int, default=0,
                    dest="image_flow_distill_steps",
                    help="post-flow own-model endpoint distillation steps")
    ap.add_argument("--image-flow-distill-frac", type=float, default=0.0,
                    dest="image_flow_distill_frac",
                    help=("post-flow distillation steps as a fraction of --train-steps; "
                          "0 uses --image-flow-distill-steps directly"))
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
                    choices=IMAGE_TIME_SAMPLINGS,
                    dest="image_time_sampling",
                    help="latent image flow timestep distribution")
    ap.add_argument("--image-time-logit-mean", type=float, default=0.0,
                    dest="image_time_logit_mean",
                    help="mean for --image-time-sampling logit-normal")
    ap.add_argument("--image-time-logit-std", type=float, default=1.0,
                    dest="image_time_logit_std",
                    help="stddev for --image-time-sampling logit-normal")
    ap.add_argument("--image-time-mode-scale", type=float,
                    default=IMAGE_DEFAULT_TIME_MODE_SCALE,
                    dest="image_time_mode_scale",
                    help=("curvature for --image-time-sampling mode; 0 is uniform, "
                          f"must be < {IMAGE_MAX_TIME_MODE_SCALE:g}"))
    ap.add_argument("--image-time-curriculum-frac", type=float, default=0.0,
                    dest="image_time_curriculum_frac",
                    help=("fraction of image flow training using --image-time-sampling before "
                          "switching timestep sampling to uniform; 0 disables"))
    ap.add_argument("--image-time-adaptive-bins", type=int, default=32,
                    dest="image_time_adaptive_bins",
                    help="loss-tracked bins for adaptive image timestep sampling")
    ap.add_argument("--image-time-adaptive-momentum", type=float, default=0.95,
                    dest="image_time_adaptive_momentum",
                    help="EMA momentum for adaptive image timestep losses")
    ap.add_argument("--image-time-adaptive-uniform-mix", type=float, default=0.05,
                    dest="image_time_adaptive_uniform_mix",
                    help="uniform probability mixed into adaptive image timestep sampling")
    ap.add_argument("--image-time-adaptive-min-prob", type=float, default=0.001,
                    dest="image_time_adaptive_min_prob",
                    help="minimum adaptive image timestep bin probability")
    ap.add_argument("--image-time-adaptive-loss-power", type=float, default=1.0,
                    dest="image_time_adaptive_loss_power",
                    help="power applied to adaptive image timestep loss EMA")
    ap.add_argument("--image-time-adaptive-prior", default="uniform",
                    choices=IMAGE_TIME_ADAPTIVE_PRIORS,
                    dest="image_time_adaptive_prior",
                    help="base timestep prior mixed into adaptive image sampling")
    ap.add_argument("--image-time-adaptive-prior-mix", type=float, default=0.0,
                    dest="image_time_adaptive_prior_mix",
                    help="probability mass taken from --image-time-adaptive-prior")
    ap.add_argument("--image-time-shift", type=float, default=1.0,
                    dest="image_time_shift",
                    help="latent image RF data-time shift; >1 biases training toward noise")
    ap.add_argument("--image-time-shift-mode", default="auto",
                    choices=("manual", "dim", "auto"), dest="image_time_shift_mode",
                    help=("manual uses --image-time-shift as-is; dim scales it by latent "
                          "dimension; auto uses dim only when latent dimension exceeds "
                          "--image-time-shift-ref-dim"))
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
                    help=("comma-separated CFG scales for --image-eval-sweep and "
                          "prompt candidates"))
    ap.add_argument("--image-cfg-rescale-sweep", default="0.0,0.7",
                    dest="image_cfg_rescale_sweep",
                    help=("comma-separated latent image CFG rescale values for eval sweeps "
                          "and prompt candidates"))
    ap.add_argument("--image-cfg-modes", default="",
                    dest="image_cfg_modes",
                    help=("comma-separated latent image CFG modes for eval sweeps/prompt candidates; "
                          "default uses --image-cfg-mode"))
    ap.add_argument("--image-sample-steps-sweep", default="4,8,16",
                    dest="image_sample_steps_sweep",
                    help=("comma-separated sampler step counts for --image-eval-sweep and "
                          "prompt candidates"))
    ap.add_argument("--image-eval-seeds", default="1,2,3", dest="image_eval_seeds",
                    help="comma-separated eval/prompt-candidate seeds")
    ap.add_argument("--multimodal", action="store_true",
                    help="train generic manifest-driven multimodal prefix bridge")
    ap.add_argument("--multimodal-manifest", default="", dest="multimodal_manifest",
                    help="JSONL manifest passed to thinking.multimodal --manifest")
    ap.add_argument("--multimodal-root", default="", dest="multimodal_root",
                    help="optional root for relative multimodal feature paths")
    ap.add_argument("--multimodal-dim", type=int, default=0, dest="multimodal_dim",
                    help="M-0 decoder width; default uses --dim or 96")
    ap.add_argument("--multimodal-layers", type=int, default=3, dest="multimodal_layers")
    ap.add_argument("--multimodal-heads", type=int, default=4, dest="multimodal_heads")
    ap.add_argument("--multimodal-batch", type=int, default=32, dest="multimodal_batch")
    ap.add_argument("--multimodal-lr", type=float, default=1e-3, dest="multimodal_lr")
    ap.add_argument("--multimodal-log-every", type=int, default=100,
                    dest="multimodal_log_every")
    ap.add_argument("--multimodal-decode-w", type=float, default=1.0,
                    dest="multimodal_decode_w",
                    help="M-0 target-token decoder loss weight; set 0 for latent-only runs")
    ap.add_argument("--multimodal-agreement-w", type=float, default=0.0,
                    dest="multimodal_agreement_w",
                    help="cross-mode token-distribution agreement loss weight")
    ap.add_argument("--multimodal-concept-tokens", type=int, default=4,
                    dest="multimodal_concept_tokens")
    ap.add_argument("--multimodal-fusion-layers", type=int, default=1,
                    dest="multimodal_fusion_layers")
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
    ap.add_argument("--multimodal-latent-concept-factorization-w", type=float,
                    default=0.0, dest="multimodal_latent_concept_factorization_w")
    ap.add_argument("--multimodal-latent-concept-factorization-variance", type=float,
                    default=0.05,
                    dest="multimodal_latent_concept_factorization_variance")
    ap.add_argument("--multimodal-latent-concept-factorization-margin", type=float,
                    default=0.2,
                    dest="multimodal_latent_concept_factorization_margin")
    ap.add_argument("--multimodal-latent-concept-factorization-covariance-w",
                    type=float, default=0.05,
                    dest="multimodal_latent_concept_factorization_covariance_w")
    ap.add_argument("--multimodal-latent-concept-fer-w", type=float, default=0.0,
                    dest="multimodal_latent_concept_fer_w")
    ap.add_argument("--multimodal-latent-concept-fer-fragmentation-w", type=float,
                    default=1.0, dest="multimodal_latent_concept_fer_fragmentation_w")
    ap.add_argument("--multimodal-latent-concept-fer-correlation-w", type=float,
                    default=1.0, dest="multimodal_latent_concept_fer_correlation_w")
    ap.add_argument("--multimodal-latent-concept-fer-balance-w", type=float,
                    default=0.1, dest="multimodal_latent_concept_fer_balance_w")
    ap.add_argument("--multimodal-latent-concept-fer-probe-n", type=int, default=0,
                    dest="multimodal_latent_concept_fer_probe_n")
    ap.add_argument("--multimodal-latent-concept-fer-hard-max", type=int, default=0,
                    dest="multimodal_latent_concept_fer_hard_max")
    ap.add_argument("--multimodal-latent-concept-fer-refresh-steps", type=int,
                    default=0, dest="multimodal_latent_concept_fer_refresh_steps")
    ap.add_argument("--multimodal-latent-concept-discovery-probe-n", type=int,
                    default=0, dest="multimodal_latent_concept_discovery_probe_n")
    ap.add_argument("--multimodal-latent-concept-discovery-hard-max", type=int,
                    default=0, dest="multimodal_latent_concept_discovery_hard_max")
    ap.add_argument("--multimodal-latent-concept-discovery-refresh-steps",
                    type=int, default=0,
                    dest="multimodal_latent_concept_discovery_refresh_steps")
    ap.add_argument("--multimodal-latent-concept-memory-w", type=float, default=0.0,
                    dest="multimodal_latent_concept_memory_w")
    ap.add_argument("--multimodal-latent-concept-memory-size", type=int, default=0,
                    dest="multimodal_latent_concept_memory_size")
    ap.add_argument("--multimodal-latent-concept-memory-temperature", type=float,
                    default=0.1, dest="multimodal_latent_concept_memory_temperature")
    ap.add_argument("--multimodal-latent-concept-memory-momentum", type=float,
                    default=0.95, dest="multimodal_latent_concept_memory_momentum")
    ap.add_argument("--multimodal-latent-concept-memory-balance-w", type=float,
                    default=0.01, dest="multimodal_latent_concept_memory_balance_w")
    ap.add_argument("--multimodal-latent-concept-consolidation-w", type=float,
                    default=0.0, dest="multimodal_latent_concept_consolidation_w")
    ap.add_argument("--multimodal-latent-concept-consolidation-temperature",
                    type=float, default=0.1,
                    dest="multimodal_latent_concept_consolidation_temperature")
    ap.add_argument("--multimodal-latent-concept-consolidation-balance-w",
                    type=float, default=0.01,
                    dest="multimodal_latent_concept_consolidation_balance_w")
    ap.add_argument("--multimodal-latent-concept-consolidation-anchor-w",
                    type=float, default=1.0,
                    dest="multimodal_latent_concept_consolidation_anchor_w")
    ap.add_argument("--multimodal-latent-concept-consolidation-fer-w",
                    type=float, default=0.0,
                    dest="multimodal_latent_concept_consolidation_fer_w")
    ap.add_argument("--multimodal-latent-concept-discovery-w", type=float,
                    default=0.0, dest="multimodal_latent_concept_discovery_w")
    ap.add_argument("--multimodal-latent-concept-discovery-curiosity-w",
                    type=float, default=1.0,
                    dest="multimodal_latent_concept_discovery_curiosity_w")
    ap.add_argument("--multimodal-latent-concept-discovery-graph-w",
                    type=float, default=1.0,
                    dest="multimodal_latent_concept_discovery_graph_w")
    ap.add_argument("--multimodal-latent-concept-discovery-cycle-w",
                    type=float, default=1.0,
                    dest="multimodal_latent_concept_discovery_cycle_w")
    ap.add_argument("--multimodal-latent-concept-discovery-bridge-w",
                    type=float, default=1.0,
                    dest="multimodal_latent_concept_discovery_bridge_w")
    ap.add_argument("--multimodal-latent-concept-discovery-fer-w",
                    type=float, default=0.0,
                    dest="multimodal_latent_concept_discovery_fer_w")
    ap.add_argument("--multimodal-latent-concept-reanalysis-w", type=float,
                    default=0.0, dest="multimodal_latent_concept_reanalysis_w")
    ap.add_argument("--multimodal-latent-concept-reanalysis-graph-w",
                    type=float, default=1.0,
                    dest="multimodal_latent_concept_reanalysis_graph_w")
    ap.add_argument("--multimodal-latent-concept-reanalysis-cycle-w",
                    type=float, default=0.5,
                    dest="multimodal_latent_concept_reanalysis_cycle_w")
    ap.add_argument("--multimodal-latent-concept-reanalysis-bridge-w",
                    type=float, default=0.5,
                    dest="multimodal_latent_concept_reanalysis_bridge_w")
    ap.add_argument("--multimodal-latent-concept-reanalysis-fer-w",
                    type=float, default=0.0,
                    dest="multimodal_latent_concept_reanalysis_fer_w")
    ap.add_argument(
        "--multimodal-latent-concept-reanalysis-cycle-consistency-w",
        type=float, default=0.5,
        dest="multimodal_latent_concept_reanalysis_cycle_consistency_w")
    ap.add_argument("--multimodal-latent-concept-association-w", type=float,
                    default=0.0, dest="multimodal_latent_concept_association_w")
    ap.add_argument("--multimodal-latent-concept-association-temperature",
                    type=float, default=0.1,
                    dest="multimodal_latent_concept_association_temperature")
    ap.add_argument("--multimodal-latent-concept-association-decay", type=float,
                    default=0.99, dest="multimodal_latent_concept_association_decay")
    ap.add_argument("--multimodal-latent-concept-association-target-power",
                    type=float, default=1.0,
                    dest="multimodal_latent_concept_association_target_power")
    ap.add_argument("--multimodal-latent-concept-association-self-loop-w",
                    type=float, default=0.05,
                    dest="multimodal_latent_concept_association_self_loop_w")
    ap.add_argument("--multimodal-latent-concept-association-transitive-steps",
                    type=int, default=2,
                    dest="multimodal_latent_concept_association_transitive_steps")
    ap.add_argument("--multimodal-latent-concept-association-transitive-w",
                    type=float, default=0.1,
                    dest="multimodal_latent_concept_association_transitive_w")
    ap.add_argument("--multimodal-latent-concept-composition-w", type=float,
                    default=0.0, dest="multimodal_latent_concept_composition_w")
    ap.add_argument("--multimodal-latent-concept-composition-temperature",
                    type=float, default=0.1,
                    dest="multimodal_latent_concept_composition_temperature")
    ap.add_argument("--multimodal-latent-concept-composition-self-loop-w",
                    type=float, default=0.0,
                    dest="multimodal_latent_concept_composition_self_loop_w")
    ap.add_argument("--multimodal-latent-concept-composition-transitive-steps",
                    type=int, default=2,
                    dest="multimodal_latent_concept_composition_transitive_steps")
    ap.add_argument("--multimodal-latent-concept-composition-transitive-w",
                    type=float, default=0.1,
                    dest="multimodal_latent_concept_composition_transitive_w")
    ap.add_argument("--multimodal-latent-concept-composition-margin", type=float,
                    default=0.0, dest="multimodal_latent_concept_composition_margin")
    ap.add_argument("--multimodal-latent-concept-graph-predict-w", type=float,
                    default=0.0, dest="multimodal_latent_concept_graph_predict_w")
    ap.add_argument("--multimodal-latent-concept-graph-predict-temperature",
                    type=float, default=0.1,
                    dest="multimodal_latent_concept_graph_predict_temperature")
    ap.add_argument("--multimodal-latent-concept-graph-predict-self-loop-w",
                    type=float, default=0.05,
                    dest="multimodal_latent_concept_graph_predict_self_loop_w")
    ap.add_argument("--multimodal-latent-concept-graph-predict-transitive-steps",
                    type=int, default=2,
                    dest="multimodal_latent_concept_graph_predict_transitive_steps")
    ap.add_argument("--multimodal-latent-concept-graph-predict-transitive-w",
                    type=float, default=0.1,
                    dest="multimodal_latent_concept_graph_predict_transitive_w")
    ap.add_argument("--multimodal-latent-concept-graph-predict-target-power",
                    type=float, default=1.0,
                    dest="multimodal_latent_concept_graph_predict_target_power")
    ap.add_argument("--multimodal-latent-concept-bridge-w", type=float, default=0.0,
                    dest="multimodal_latent_concept_bridge_w")
    ap.add_argument("--multimodal-latent-concept-sequence-w", type=float, default=0.0,
                    dest="multimodal_latent_concept_sequence_w")
    ap.add_argument("--multimodal-latent-concept-sequence-batch", type=int,
                    default=0, dest="multimodal_latent_concept_sequence_batch")
    ap.add_argument("--multimodal-latent-concept-sequence-temperature", type=float,
                    default=0.1, dest="multimodal_latent_concept_sequence_temperature")
    ap.add_argument("--multimodal-latent-concept-neighborhood-w", type=float,
                    default=0.0, dest="multimodal_latent_concept_neighborhood_w")
    ap.add_argument("--multimodal-latent-concept-neighborhood-temperature",
                    type=float, default=0.1,
                    dest="multimodal_latent_concept_neighborhood_temperature")
    ap.add_argument("--multimodal-latent-concept-neighborhood-margin", type=float,
                    default=0.0, dest="multimodal_latent_concept_neighborhood_margin")
    ap.add_argument("--multimodal-latent-concept-transition-w", type=float,
                    default=0.0, dest="multimodal_latent_concept_transition_w")
    ap.add_argument("--multimodal-latent-concept-transition-temperature",
                    type=float, default=0.1,
                    dest="multimodal_latent_concept_transition_temperature")
    ap.add_argument("--multimodal-latent-concept-transition-margin", type=float,
                    default=0.0, dest="multimodal_latent_concept_transition_margin")
    ap.add_argument("--multimodal-latent-concept-cluster-w", type=float,
                    default=0.0, dest="multimodal_latent_concept_cluster_w")
    ap.add_argument("--multimodal-latent-concept-cluster-temperature", type=float,
                    default=0.1, dest="multimodal_latent_concept_cluster_temperature")
    ap.add_argument("--multimodal-latent-concept-cluster-margin", type=float,
                    default=0.0, dest="multimodal_latent_concept_cluster_margin")
    ap.add_argument("--multimodal-latent-concept-cluster-min-size", type=int,
                    default=2, dest="multimodal_latent_concept_cluster_min_size")
    ap.add_argument("--multimodal-text-checkpoint", default="",
                    dest="multimodal_text_checkpoint",
                    help="optional thinking.text checkpoint for multimodal warm start")
    ap.add_argument("--multimodal-select-best", action=argparse.BooleanOptionalAction,
                    default=False, dest="multimodal_select_best",
                    help="keep the best self-scored multimodal study round")
    ap.add_argument("--multimodal-selection-rounds", type=int, default=1,
                    dest="multimodal_selection_rounds")
    ap.add_argument("--multimodal-selection-score-metric",
                    choices=("token", "exact", "fer", "bridge", "sequence",
                             "all", "balanced", "mastery"),
                    default="mastery", dest="multimodal_selection_score_metric")
    ap.add_argument("--multimodal-selection-score-margin-w", type=float,
                    default=0.1, dest="multimodal_selection_score_margin_w")
    ap.add_argument("--multimodal-selection-score-min-delta", type=float,
                    default=0.0, dest="multimodal_selection_score_min_delta")
    ap.add_argument("--multimodal-selection-score-patience", type=int,
                    default=0, dest="multimodal_selection_score_patience")
    ap.add_argument("--multimodal-selection-eval-n", type=int, default=200,
                    dest="multimodal_selection_eval_n")
    ap.add_argument("--multimodal-selection-insight-accept-w",
                    "--multimodal-selection-insight-w", type=float, default=0.25,
                    dest="multimodal_selection_insight_accept_w")
    ap.add_argument("--multimodal-selection-insight-min-delta", type=float,
                    default=0.0, dest="multimodal_selection_insight_min_delta")
    ap.add_argument("--multimodal-view-tokens", type=int, default=4,
                    dest="multimodal_view_tokens")
    ap.add_argument("--multimodal-txt-tokens", type=int, default=8,
                    dest="multimodal_txt_tokens")
    ap.add_argument("--multimodal-trunk-arch", default="mlp", choices=("mlp", "residual"),
                    dest="multimodal_trunk_arch",
                    help="multimodal feature reader trunk architecture")
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
    ap.add_argument("--max-minutes", type=int, default=150)
    ap.add_argument("--go", action="store_true", help="actually create the pod (spends money)")
    args = ap.parse_args()
    if args.reading_data and not args.text_reading:
        args.text_reading = True
    try:
        apply_image_quality_preset(args)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    if args.text_reading:
        if not args.reading_data:
            sys.exit("ERROR: --text-reading requires --reading-data")
        if args.batch <= 0:
            sys.exit("ERROR: --batch must be positive for --text-reading")
        if args.train_steps < 0:
            sys.exit("ERROR: --train-steps must be non-negative")
        if args.text_reading_steps <= 0:
            sys.exit("ERROR: --text-reading-steps must be positive")
        text_reading_dim = args.text_reading_dim or args.dim or 256
        if args.text_reading_dim < 0:
            sys.exit("ERROR: --text-reading-dim must be non-negative")
        if text_reading_dim <= 0:
            sys.exit("ERROR: raw-reading width must be positive")
        positive_text = {
            "--text-reading-layers": args.text_reading_layers,
            "--text-reading-heads": args.text_reading_heads,
            "--text-reading-encoder-layers": args.text_reading_encoder_layers,
            "--text-reading-latent-concept-layers": (
                args.text_reading_latent_concept_layers),
            "--reading-max-tokens": args.reading_max_tokens,
            "--reading-min-tokens": args.reading_min_tokens,
            "--reading-association-transitive-steps": (
                args.reading_association_transitive_steps),
            "--reading-composition-transitive-steps": (
                args.reading_composition_transitive_steps),
            "--reading-graph-predict-transitive-steps": (
                args.reading_graph_predict_transitive_steps),
            "--reading-graph-cycle-transitive-steps": (
                args.reading_graph_cycle_transitive_steps),
            "--reading-cluster-min-size": args.reading_cluster_min_size,
            "--reading-study-rounds": args.reading_study_rounds,
        }
        for name, value in positive_text.items():
            if value <= 0:
                sys.exit(f"ERROR: {name} must be positive")
        if args.reading_cluster_min_size < 2:
            sys.exit("ERROR: --reading-cluster-min-size must be at least 2")
        if args.reading_min_tokens > args.reading_max_tokens:
            sys.exit("ERROR: --reading-min-tokens cannot exceed --reading-max-tokens")
        if text_reading_dim % args.text_reading_heads != 0:
            sys.exit("ERROR: raw-reading width must divide --text-reading-heads")
        if (text_reading_dim // args.text_reading_heads) % 2 != 0:
            sys.exit("ERROR: raw-reading head dimension must be even for attention")
        if args.text_reading_latent_concept_slots < 0:
            sys.exit("ERROR: --text-reading-latent-concept-slots must be non-negative")
        text_nonnegative = {
            "--reading-eval-n": args.reading_eval_n,
            "--reading-replay-w": args.reading_replay_w,
            "--reading-replay-batch": args.reading_replay_batch,
            "--reading-replay-retention-w": args.reading_replay_retention_w,
            "--reading-context-target-w": args.reading_context_target_w,
            "--reading-sequence-w": args.reading_sequence_w,
            "--reading-sequence-batch": args.reading_sequence_batch,
            "--reading-factorization-w": args.reading_factorization_w,
            "--reading-factorization-variance": args.reading_factorization_variance,
            "--reading-factorization-margin": args.reading_factorization_margin,
            "--reading-factorization-covariance-w": (
                args.reading_factorization_covariance_w),
            "--reading-fer-w": args.reading_fer_w,
            "--reading-fer-fragmentation-w": args.reading_fer_fragmentation_w,
            "--reading-fer-correlation-w": args.reading_fer_correlation_w,
            "--reading-fer-balance-w": args.reading_fer_balance_w,
            "--reading-memory-w": args.reading_memory_w,
            "--reading-memory-size": args.reading_memory_size,
            "--reading-memory-balance-w": args.reading_memory_balance_w,
            "--reading-consolidation-w": args.reading_consolidation_w,
            "--reading-consolidation-balance-w": (
                args.reading_consolidation_balance_w),
            "--reading-consolidation-anchor-w": args.reading_consolidation_anchor_w,
            "--reading-consolidation-fer-w": args.reading_consolidation_fer_w,
            "--reading-discovery-w": args.reading_discovery_w,
            "--reading-discovery-curiosity-w": args.reading_discovery_curiosity_w,
            "--reading-discovery-graph-w": args.reading_discovery_graph_w,
            "--reading-discovery-cycle-w": args.reading_discovery_cycle_w,
            "--reading-discovery-bridge-w": args.reading_discovery_bridge_w,
            "--reading-discovery-fer-w": args.reading_discovery_fer_w,
            "--reading-reanalysis-w": args.reading_reanalysis_w,
            "--reading-reanalysis-graph-w": args.reading_reanalysis_graph_w,
            "--reading-reanalysis-cycle-w": args.reading_reanalysis_cycle_w,
            "--reading-reanalysis-bridge-w": args.reading_reanalysis_bridge_w,
            "--reading-reanalysis-fer-w": args.reading_reanalysis_fer_w,
            "--reading-association-w": args.reading_association_w,
            "--reading-association-self-loop-w": (
                args.reading_association_self_loop_w),
            "--reading-association-transitive-w": (
                args.reading_association_transitive_w),
            "--reading-composition-w": args.reading_composition_w,
            "--reading-composition-self-loop-w": (
                args.reading_composition_self_loop_w),
            "--reading-composition-transitive-w": (
                args.reading_composition_transitive_w),
            "--reading-composition-margin": args.reading_composition_margin,
            "--reading-graph-predict-w": args.reading_graph_predict_w,
            "--reading-graph-predict-self-loop-w": (
                args.reading_graph_predict_self_loop_w),
            "--reading-graph-predict-transitive-w": (
                args.reading_graph_predict_transitive_w),
            "--reading-graph-cycle-w": args.reading_graph_cycle_w,
            "--reading-graph-cycle-self-loop-w": (
                args.reading_graph_cycle_self_loop_w),
            "--reading-graph-cycle-transitive-w": (
                args.reading_graph_cycle_transitive_w),
            "--reading-graph-cycle-consistency-w": (
                args.reading_graph_cycle_consistency_w),
            "--reading-bridge-w": args.reading_bridge_w,
            "--reading-neighborhood-w": args.reading_neighborhood_w,
            "--reading-neighborhood-batch": args.reading_neighborhood_batch,
            "--reading-neighborhood-probe-n": args.reading_neighborhood_probe_n,
            "--reading-neighborhood-refresh-steps": (
                args.reading_neighborhood_refresh_steps),
            "--reading-neighborhood-margin": args.reading_neighborhood_margin,
            "--reading-transition-w": args.reading_transition_w,
            "--reading-transition-batch": args.reading_transition_batch,
            "--reading-transition-margin": args.reading_transition_margin,
            "--reading-cluster-w": args.reading_cluster_w,
            "--reading-cluster-batch": args.reading_cluster_batch,
            "--reading-cluster-probe-n": args.reading_cluster_probe_n,
            "--reading-cluster-refresh-steps": args.reading_cluster_refresh_steps,
            "--reading-cluster-margin": args.reading_cluster_margin,
            "--reading-study-probe-n": args.reading_study_probe_n,
            "--reading-study-hard-max": args.reading_study_hard_max,
            "--reading-study-refresh-steps": args.reading_study_refresh_steps,
            "--reading-study-score-margin-w": args.reading_study_score_margin_w,
            "--reading-study-score-min-delta": args.reading_study_score_min_delta,
            "--reading-study-score-patience": args.reading_study_score_patience,
            "--reading-study-insight-accept-w": (
                args.reading_study_insight_accept_w),
            "--reading-study-insight-min-delta": (
                args.reading_study_insight_min_delta),
        }
        bad_text_nonnegative = [
            name for name, value in text_nonnegative.items() if value < 0
        ]
        if bad_text_nonnegative:
            sys.exit(
                "ERROR: raw-reading controls must be non-negative: "
                + ", ".join(bad_text_nonnegative))
        if args.reading_lr <= 0.0:
            sys.exit("ERROR: --reading-lr must be positive")
        if args.reading_eval_frac < 0.0 or args.reading_eval_frac >= 1.0:
            sys.exit("ERROR: --reading-eval-frac must be in [0, 1)")
        bounded_text = {
            "--reading-token-drop": args.reading_token_drop,
            "--reading-token-replace": args.reading_token_replace,
            "--reading-context-keep-p": args.reading_context_keep_p,
        }
        for name, value in bounded_text.items():
            if value < 0.0 or value > 1.0:
                sys.exit(f"ERROR: {name} must be in [0, 1]")
        if args.reading_feature_dropout < 0.0 or args.reading_feature_dropout >= 1.0:
            sys.exit("ERROR: --reading-feature-dropout must be in [0, 1)")
        text_temperatures = {
            "--reading-context-target-temperature": (
                args.reading_context_target_temperature),
            "--reading-sequence-temperature": args.reading_sequence_temperature,
            "--reading-memory-temperature": args.reading_memory_temperature,
            "--reading-consolidation-temperature": (
                args.reading_consolidation_temperature),
            "--reading-association-temperature": args.reading_association_temperature,
            "--reading-composition-temperature": args.reading_composition_temperature,
            "--reading-graph-predict-temperature": (
                args.reading_graph_predict_temperature),
            "--reading-graph-cycle-temperature": (
                args.reading_graph_cycle_temperature),
            "--reading-neighborhood-temperature": (
                args.reading_neighborhood_temperature),
            "--reading-transition-temperature": args.reading_transition_temperature,
            "--reading-cluster-temperature": args.reading_cluster_temperature,
        }
        if any(value <= 0.0 for value in text_temperatures.values()):
            sys.exit("ERROR: raw-reading temperatures must be positive")
        if (args.reading_memory_momentum < 0.0
                or args.reading_memory_momentum >= 1.0):
            sys.exit("ERROR: --reading-memory-momentum must be in [0, 1)")
        if args.reading_discovery_w > 0.0 and args.reading_memory_size <= 0:
            sys.exit(
                "ERROR: --reading-discovery-w requires "
                "--reading-memory-size > 0")
        if args.reading_reanalysis_w > 0.0 and args.reading_memory_size <= 0:
            sys.exit(
                "ERROR: --reading-reanalysis-w requires "
                "--reading-memory-size > 0")
        if (args.reading_association_decay < 0.0
                or args.reading_association_decay >= 1.0):
            sys.exit("ERROR: --reading-association-decay must be in [0, 1)")
        if (args.reading_association_target_power <= 0.0
                or args.reading_graph_predict_target_power <= 0.0
                or args.reading_graph_cycle_target_power <= 0.0):
            sys.exit("ERROR: raw-reading graph target powers must be positive")
        if args.reading_sequence_batch == 1:
            sys.exit("ERROR: --reading-sequence-batch must be 0 or at least 2")
        if args.reading_replay_data and not args.reading_checkpoint:
            sys.exit("ERROR: --reading-replay-data requires --reading-checkpoint")
        if args.reading_replay_w > 0.0 and not args.reading_replay_data:
            sys.exit("ERROR: --reading-replay-w requires --reading-replay-data")
        if args.multimodal and not args.multimodal_text_checkpoint:
            args.multimodal_text_checkpoint = (
                args.reading_out_checkpoint or args.text_reading_checkpoint_out)
    if args.image_flow_distill_frac < 0.0 or args.image_flow_distill_frac > 1.0:
        sys.exit("ERROR: --image-flow-distill-frac must be in [0, 1]")
    if args.image_flow_distill_frac > 0.0:
        base_steps = int(args.train_steps or 800)
        distill_steps = max(1, int(float(base_steps) * float(args.image_flow_distill_frac)))
        args.image_flow_distill_steps = max(int(args.image_flow_distill_steps), distill_steps)
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
    valid_cfg_schedules = {"constant", "linear", "cosine", "triangular"}
    cfg_schedules = {
        raw.strip()
        for raw in str(args.image_cfg_schedules or args.image_cfg_schedule).split(",")
        if raw.strip()
    }
    bad_cfg_schedules = sorted(cfg_schedules - valid_cfg_schedules)
    if bad_cfg_schedules:
        sys.exit(
            "ERROR: unsupported --image-cfg-schedules values: "
            + ",".join(bad_cfg_schedules)
        )
    try:
        parse_unit_interval(args.image_cfg_interval, "--image-cfg-interval")
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
            str(args.image_sample_pixel_dynamic_threshold_percentile),
            "--image-sample-pixel-dynamic-threshold-percentile",
        )
        parse_nonnegative_float_csv(
            str(args.image_sample_pixel_dynamic_threshold_max),
            "--image-sample-pixel-dynamic-threshold-max",
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
    valid_sample_schedules = set(IMAGE_SAMPLE_SCHEDULES)
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
    if (args.image_time_mode_scale < 0.0
            or args.image_time_mode_scale >= IMAGE_MAX_TIME_MODE_SCALE):
        sys.exit(
            f"ERROR: --image-time-mode-scale must be in [0, {IMAGE_MAX_TIME_MODE_SCALE:g})")
    if args.image_size_curriculum_frac < 0.0 or args.image_size_curriculum_frac > 1.0:
        sys.exit("ERROR: --image-size-curriculum-frac must be in [0, 1]")
    if args.image_time_curriculum_frac < 0.0 or args.image_time_curriculum_frac > 1.0:
        sys.exit("ERROR: --image-time-curriculum-frac must be in [0, 1]")
    if args.image_time_adaptive_bins <= 1:
        sys.exit("ERROR: --image-time-adaptive-bins must be > 1")
    if (args.image_time_adaptive_momentum < 0.0
            or args.image_time_adaptive_momentum >= 1.0):
        sys.exit("ERROR: --image-time-adaptive-momentum must be in [0, 1)")
    if (args.image_time_adaptive_uniform_mix < 0.0
            or args.image_time_adaptive_uniform_mix > 1.0):
        sys.exit("ERROR: --image-time-adaptive-uniform-mix must be in [0, 1]")
    if (args.image_time_adaptive_min_prob < 0.0
            or args.image_time_adaptive_min_prob >= 1.0):
        sys.exit("ERROR: --image-time-adaptive-min-prob must be in [0, 1)")
    if args.image_time_adaptive_loss_power <= 0.0:
        sys.exit("ERROR: --image-time-adaptive-loss-power must be positive")
    if (args.image_time_adaptive_prior_mix < 0.0
            or args.image_time_adaptive_prior_mix > 1.0):
        sys.exit("ERROR: --image-time-adaptive-prior-mix must be in [0, 1]")
    if args.image_time_shift_ref_dim <= 0.0:
        sys.exit("ERROR: --image-time-shift-ref-dim must be positive")
    if args.image_flow_loss_weight_gamma <= 0.0:
        sys.exit("ERROR: --image-flow-loss-weight-gamma must be positive")
    if args.image_flow_endpoint_w < 0.0:
        sys.exit("ERROR: --image-flow-endpoint-w must be non-negative")
    if args.image_flow_frequency_w < 0.0:
        sys.exit("ERROR: --image-flow-frequency-w must be non-negative")
    if args.image_flow_straightness_w < 0.0:
        sys.exit("ERROR: --image-flow-straightness-w must be non-negative")
    if args.image_flow_multiscale_w < 0.0:
        sys.exit("ERROR: --image-flow-multiscale-w must be non-negative")
    for raw_scale in str(args.image_flow_multiscale_scales or "").split(","):
        raw_scale = raw_scale.strip()
        if not raw_scale:
            continue
        try:
            scale = int(raw_scale)
        except ValueError:
            sys.exit("ERROR: --image-flow-multiscale-scales must contain integers")
        if scale <= 1:
            sys.exit("ERROR: --image-flow-multiscale-scales entries must be > 1")
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
            or args.image_score_external_w < 0.0 or args.image_score_pickscore_w < 0.0):
        sys.exit("ERROR: image score weights must be non-negative")
    if args.image_score_pickscore_batch <= 0:
        sys.exit("ERROR: --image-score-pickscore-batch must be positive")
    if args.image_score_pickscore_max_length <= 0:
        sys.exit("ERROR: --image-score-pickscore-max-length must be positive")
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
    for name in (
            "image_clean_min_caption_unique_ratio",
            "image_clean_max_caption_token_frequency",
            "image_clean_max_caption_token_run",
            "image_clean_max_caption_char_run"):
        val = float(getattr(args, name))
        if val < 0.0 or val > 1.0:
            flag = "--" + name.replace("_", "-")
            sys.exit(f"ERROR: {flag} must be in [0, 1]")
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
    if args.image_sample_candidates_manifest_out and not args.image_sample_grid:
        sys.exit(
            "ERROR: --image-sample-candidates-manifest-out requires --image-sample-grid")
    if args.image_sample_candidates_manifest_out and not args.image_sample_prompts:
        sys.exit(
            "ERROR: --image-sample-candidates-manifest-out requires --image-sample-prompts")
    if (args.image_sample_candidates_manifest_out
            and args.image_sample_candidates_per_prompt <= 1):
        sys.exit(
            "ERROR: --image-sample-candidates-manifest-out requires "
            "--image-sample-candidates-per-prompt > 1")
    if args.image_sample_candidates_image_dir and not args.image_sample_candidates_manifest_out:
        sys.exit(
            "ERROR: --image-sample-candidates-image-dir requires "
            "--image-sample-candidates-manifest-out")
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
    if args.image_generated_preference_min_score_gap < 0.0:
        sys.exit("ERROR: --image-generated-preference-min-score-gap must be non-negative")
    if args.image_generated_preference_max_pairs_per_group < 0:
        sys.exit(
            "ERROR: --image-generated-preference-max-pairs-per-group must be non-negative")
    if args.image_generated_preference_max_pairs < 0:
        sys.exit("ERROR: --image-generated-preference-max-pairs must be non-negative")
    if args.image_sample_negative_prompts and not args.image_sample_prompts:
        sys.exit("ERROR: --image-sample-negative-prompts requires --image-sample-prompts")
    if args.image_sample_candidates_per_prompt <= 0:
        sys.exit("ERROR: --image-sample-candidates-per-prompt must be positive")
    if args.image_sample_candidates_per_prompt > 1 and not args.image_sample_prompts:
        sys.exit("ERROR: --image-sample-candidates-per-prompt > 1 requires --image-sample-prompts")
    if args.image_sample_pixel_dynamic_threshold_percentile > 1.0:
        sys.exit(
            "ERROR: --image-sample-pixel-dynamic-threshold-percentile must be <= 1")
    if args.image_sample_pixel_dynamic_threshold_max <= 0.0:
        sys.exit("ERROR: --image-sample-pixel-dynamic-threshold-max must be positive")
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
    if args.image_flow_preference_w < 0.0:
        sys.exit("ERROR: --image-flow-preference-w must be non-negative")
    if args.image_flow_preference_margin < 0.0:
        sys.exit("ERROR: --image-flow-preference-margin must be non-negative")
    if args.image_flow_preference_batch < 0:
        sys.exit("ERROR: --image-flow-preference-batch must be non-negative")
    if args.image_flow_preference_w > 0.0 and not args.image_preference_manifest:
        sys.exit("ERROR: --image-flow-preference-w requires --image-preference-manifest")
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
    if args.image_dit_depth <= 0:
        sys.exit("ERROR: --image-dit-depth must be positive")
    if args.image_dit_heads <= 0:
        sys.exit("ERROR: --image-dit-heads must be positive")
    if args.image_flow_time_embed_dim < 0:
        sys.exit("ERROR: --image-flow-time-embed-dim must be non-negative")
    if args.image_flow_self_condition and args.image_latent_arch == "conv":
        sys.exit("ERROR: --image-flow-self-condition requires --image-latent-arch dit/crossdit/mmdit")
    if args.image_flow_self_condition_p < 0.0 or args.image_flow_self_condition_p > 1.0:
        sys.exit("ERROR: --image-flow-self-condition-p must be in [0, 1]")
    if args.image_latent_arch in ("dit", "crossdit", "mmdit"):
        hidden = int(args.dim or 64)
        actual_heads = max(1, min(int(args.image_dit_heads), hidden // 16))
        if hidden % actual_heads != 0:
            sys.exit(
                "ERROR: image latent hidden width must be divisible by effective "
                f"attention heads ({hidden} % {actual_heads} != 0)")
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
            "--multimodal-view-tokens": args.multimodal_view_tokens,
            "--multimodal-txt-tokens": args.multimodal_txt_tokens,
            "--multimodal-trunk-width": args.multimodal_trunk_width,
            "--multimodal-trunk-depth": args.multimodal_trunk_depth,
            "--multimodal-text-layers": args.multimodal_text_layers,
            "--multimodal-concept-tokens": args.multimodal_concept_tokens,
            "--multimodal-fusion-layers": args.multimodal_fusion_layers,
            "--multimodal-latent-concept-layers": (
                args.multimodal_latent_concept_layers),
            "--multimodal-latent-concept-association-transitive-steps": (
                args.multimodal_latent_concept_association_transitive_steps),
            "--multimodal-latent-concept-composition-transitive-steps": (
                args.multimodal_latent_concept_composition_transitive_steps),
            "--multimodal-latent-concept-graph-predict-transitive-steps": (
                args.multimodal_latent_concept_graph_predict_transitive_steps),
            "--multimodal-latent-concept-cluster-min-size": (
                args.multimodal_latent_concept_cluster_min_size),
            "--multimodal-selection-rounds": args.multimodal_selection_rounds,
            "--multimodal-eval-n": args.multimodal_eval_n,
        }
        for name, value in positive.items():
            if value <= 0:
                sys.exit(f"ERROR: {name} must be positive")
        if not args.multimodal_manifest:
            sys.exit("ERROR: --multimodal requires --multimodal-manifest")
        if args.multimodal_dim < 0:
            sys.exit("ERROR: --multimodal-dim must be non-negative")
        mm_dim = args.multimodal_dim or args.dim or 96
        if mm_dim % args.multimodal_heads != 0:
            sys.exit("ERROR: multimodal width must be divisible by --multimodal-heads")
        if (mm_dim // args.multimodal_heads) % 2 != 0:
            sys.exit("ERROR: multimodal head dimension must be even for rope attention")
        if args.multimodal_lr <= 0.0:
            sys.exit("ERROR: --multimodal-lr must be positive")
        multimodal_nonnegative = {
            "--multimodal-decode-w": args.multimodal_decode_w,
            "--multimodal-agreement-w": args.multimodal_agreement_w,
            "--multimodal-latent-concept-w": args.multimodal_latent_concept_w,
            "--multimodal-latent-concept-invariance-w": (
                args.multimodal_latent_concept_invariance_w),
            "--multimodal-latent-concept-variance-w": (
                args.multimodal_latent_concept_variance_w),
            "--multimodal-latent-concept-covariance-w": (
                args.multimodal_latent_concept_covariance_w),
            "--multimodal-latent-concept-factorization-w": (
                args.multimodal_latent_concept_factorization_w),
            "--multimodal-latent-concept-factorization-variance": (
                args.multimodal_latent_concept_factorization_variance),
            "--multimodal-latent-concept-factorization-margin": (
                args.multimodal_latent_concept_factorization_margin),
            "--multimodal-latent-concept-factorization-covariance-w": (
                args.multimodal_latent_concept_factorization_covariance_w),
            "--multimodal-latent-concept-fer-w": (
                args.multimodal_latent_concept_fer_w),
            "--multimodal-latent-concept-fer-fragmentation-w": (
                args.multimodal_latent_concept_fer_fragmentation_w),
            "--multimodal-latent-concept-fer-correlation-w": (
                args.multimodal_latent_concept_fer_correlation_w),
            "--multimodal-latent-concept-fer-balance-w": (
                args.multimodal_latent_concept_fer_balance_w),
            "--multimodal-latent-concept-fer-probe-n": (
                args.multimodal_latent_concept_fer_probe_n),
            "--multimodal-latent-concept-fer-hard-max": (
                args.multimodal_latent_concept_fer_hard_max),
            "--multimodal-latent-concept-fer-refresh-steps": (
                args.multimodal_latent_concept_fer_refresh_steps),
            "--multimodal-latent-concept-discovery-probe-n": (
                args.multimodal_latent_concept_discovery_probe_n),
            "--multimodal-latent-concept-discovery-hard-max": (
                args.multimodal_latent_concept_discovery_hard_max),
            "--multimodal-latent-concept-discovery-refresh-steps": (
                args.multimodal_latent_concept_discovery_refresh_steps),
            "--multimodal-latent-concept-memory-w": (
                args.multimodal_latent_concept_memory_w),
            "--multimodal-latent-concept-memory-size": (
                args.multimodal_latent_concept_memory_size),
            "--multimodal-latent-concept-memory-balance-w": (
                args.multimodal_latent_concept_memory_balance_w),
            "--multimodal-latent-concept-consolidation-w": (
                args.multimodal_latent_concept_consolidation_w),
            "--multimodal-latent-concept-consolidation-balance-w": (
                args.multimodal_latent_concept_consolidation_balance_w),
            "--multimodal-latent-concept-consolidation-anchor-w": (
                args.multimodal_latent_concept_consolidation_anchor_w),
            "--multimodal-latent-concept-consolidation-fer-w": (
                args.multimodal_latent_concept_consolidation_fer_w),
            "--multimodal-latent-concept-discovery-w": (
                args.multimodal_latent_concept_discovery_w),
            "--multimodal-latent-concept-discovery-curiosity-w": (
                args.multimodal_latent_concept_discovery_curiosity_w),
            "--multimodal-latent-concept-discovery-graph-w": (
                args.multimodal_latent_concept_discovery_graph_w),
            "--multimodal-latent-concept-discovery-cycle-w": (
                args.multimodal_latent_concept_discovery_cycle_w),
            "--multimodal-latent-concept-discovery-bridge-w": (
                args.multimodal_latent_concept_discovery_bridge_w),
            "--multimodal-latent-concept-discovery-fer-w": (
                args.multimodal_latent_concept_discovery_fer_w),
            "--multimodal-latent-concept-reanalysis-w": (
                args.multimodal_latent_concept_reanalysis_w),
            "--multimodal-latent-concept-reanalysis-graph-w": (
                args.multimodal_latent_concept_reanalysis_graph_w),
            "--multimodal-latent-concept-reanalysis-cycle-w": (
                args.multimodal_latent_concept_reanalysis_cycle_w),
            "--multimodal-latent-concept-reanalysis-bridge-w": (
                args.multimodal_latent_concept_reanalysis_bridge_w),
            "--multimodal-latent-concept-reanalysis-fer-w": (
                args.multimodal_latent_concept_reanalysis_fer_w),
            "--multimodal-latent-concept-reanalysis-cycle-consistency-w": (
                args.multimodal_latent_concept_reanalysis_cycle_consistency_w),
            "--multimodal-latent-concept-association-w": (
                args.multimodal_latent_concept_association_w),
            "--multimodal-latent-concept-association-self-loop-w": (
                args.multimodal_latent_concept_association_self_loop_w),
            "--multimodal-latent-concept-association-transitive-w": (
                args.multimodal_latent_concept_association_transitive_w),
            "--multimodal-latent-concept-composition-w": (
                args.multimodal_latent_concept_composition_w),
            "--multimodal-latent-concept-composition-self-loop-w": (
                args.multimodal_latent_concept_composition_self_loop_w),
            "--multimodal-latent-concept-composition-transitive-w": (
                args.multimodal_latent_concept_composition_transitive_w),
            "--multimodal-latent-concept-composition-margin": (
                args.multimodal_latent_concept_composition_margin),
            "--multimodal-latent-concept-graph-predict-w": (
                args.multimodal_latent_concept_graph_predict_w),
            "--multimodal-latent-concept-graph-predict-self-loop-w": (
                args.multimodal_latent_concept_graph_predict_self_loop_w),
            "--multimodal-latent-concept-graph-predict-transitive-w": (
                args.multimodal_latent_concept_graph_predict_transitive_w),
            "--multimodal-latent-concept-bridge-w": (
                args.multimodal_latent_concept_bridge_w),
            "--multimodal-latent-concept-sequence-w": (
                args.multimodal_latent_concept_sequence_w),
            "--multimodal-latent-concept-sequence-batch": (
                args.multimodal_latent_concept_sequence_batch),
            "--multimodal-latent-concept-neighborhood-w": (
                args.multimodal_latent_concept_neighborhood_w),
            "--multimodal-latent-concept-neighborhood-margin": (
                args.multimodal_latent_concept_neighborhood_margin),
            "--multimodal-latent-concept-transition-w": (
                args.multimodal_latent_concept_transition_w),
            "--multimodal-latent-concept-transition-margin": (
                args.multimodal_latent_concept_transition_margin),
            "--multimodal-latent-concept-cluster-w": (
                args.multimodal_latent_concept_cluster_w),
            "--multimodal-latent-concept-cluster-margin": (
                args.multimodal_latent_concept_cluster_margin),
            "--multimodal-selection-score-margin-w": (
                args.multimodal_selection_score_margin_w),
            "--multimodal-selection-score-min-delta": (
                args.multimodal_selection_score_min_delta),
            "--multimodal-selection-score-patience": (
                args.multimodal_selection_score_patience),
            "--multimodal-selection-eval-n": args.multimodal_selection_eval_n,
            "--multimodal-selection-insight-accept-w": (
                args.multimodal_selection_insight_accept_w),
            "--multimodal-selection-insight-min-delta": (
                args.multimodal_selection_insight_min_delta),
        }
        bad_nonnegative = [
            name for name, value in multimodal_nonnegative.items()
            if value < 0.0
        ]
        if bad_nonnegative:
            bad_flags = ", ".join(bad_nonnegative)
            sys.exit(f"ERROR: multimodal controls must be non-negative: {bad_flags}")
        if args.multimodal_latent_concept_slots < 0:
            sys.exit("ERROR: --multimodal-latent-concept-slots must be non-negative")
        multimodal_latent_weights = [
            args.multimodal_latent_concept_w,
            args.multimodal_latent_concept_factorization_w,
            args.multimodal_latent_concept_fer_w,
            args.multimodal_latent_concept_memory_w,
            args.multimodal_latent_concept_consolidation_w,
            args.multimodal_latent_concept_discovery_w,
            args.multimodal_latent_concept_reanalysis_w,
            args.multimodal_latent_concept_association_w,
            args.multimodal_latent_concept_composition_w,
            args.multimodal_latent_concept_graph_predict_w,
            args.multimodal_latent_concept_bridge_w,
            args.multimodal_latent_concept_sequence_w,
            args.multimodal_latent_concept_neighborhood_w,
            args.multimodal_latent_concept_transition_w,
            args.multimodal_latent_concept_cluster_w,
        ]
        if ((any(w > 0.0 for w in multimodal_latent_weights)
             or args.multimodal_latent_concept_memory_size > 0
             or args.multimodal_latent_concept_fer_hard_max > 0
             or args.multimodal_latent_concept_discovery_hard_max > 0)
                and args.multimodal_latent_concept_slots <= 0):
            sys.exit(
                "ERROR: multimodal latent concept options require "
                "--multimodal-latent-concept-slots > 0")
        if (args.multimodal_latent_concept_view_dropout < 0.0
                or args.multimodal_latent_concept_view_dropout >= 1.0):
            sys.exit("ERROR: --multimodal-latent-concept-view-dropout must be in [0, 1)")
        if args.multimodal_latent_concept_variance_target < 0.0:
            sys.exit(
                "ERROR: --multimodal-latent-concept-variance-target must be non-negative")
        if args.multimodal_latent_concept_sequence_batch == 1:
            sys.exit(
                "ERROR: --multimodal-latent-concept-sequence-batch must be 0 or at least 2")
        if (args.multimodal_latent_concept_discovery_hard_max > 0
                and args.multimodal_latent_concept_memory_size <= 0):
            sys.exit(
                "ERROR: multimodal discovery hard study requires "
                "--multimodal-latent-concept-memory-size > 0")
        multimodal_temperatures = {
            "--multimodal-latent-concept-memory-temperature": (
                args.multimodal_latent_concept_memory_temperature),
            "--multimodal-latent-concept-consolidation-temperature": (
                args.multimodal_latent_concept_consolidation_temperature),
            "--multimodal-latent-concept-association-temperature": (
                args.multimodal_latent_concept_association_temperature),
            "--multimodal-latent-concept-composition-temperature": (
                args.multimodal_latent_concept_composition_temperature),
            "--multimodal-latent-concept-graph-predict-temperature": (
                args.multimodal_latent_concept_graph_predict_temperature),
            "--multimodal-latent-concept-sequence-temperature": (
                args.multimodal_latent_concept_sequence_temperature),
            "--multimodal-latent-concept-neighborhood-temperature": (
                args.multimodal_latent_concept_neighborhood_temperature),
            "--multimodal-latent-concept-transition-temperature": (
                args.multimodal_latent_concept_transition_temperature),
            "--multimodal-latent-concept-cluster-temperature": (
                args.multimodal_latent_concept_cluster_temperature),
        }
        bad_temperature = [
            name for name, value in multimodal_temperatures.items()
            if value <= 0.0
        ]
        if bad_temperature:
            sys.exit("ERROR: multimodal temperatures must be positive")
        if (args.multimodal_latent_concept_memory_momentum < 0.0
                or args.multimodal_latent_concept_memory_momentum >= 1.0):
            sys.exit(
                "ERROR: --multimodal-latent-concept-memory-momentum must be in [0, 1)")
        if (args.multimodal_latent_concept_consolidation_w > 0.0
                and args.multimodal_latent_concept_memory_size <= 0):
            sys.exit(
                "ERROR: --multimodal-latent-concept-consolidation-w requires "
                "--multimodal-latent-concept-memory-size > 0")
        if (args.multimodal_latent_concept_discovery_w > 0.0
                and args.multimodal_latent_concept_memory_size <= 0):
            sys.exit(
                "ERROR: --multimodal-latent-concept-discovery-w requires "
                "--multimodal-latent-concept-memory-size > 0")
        if (args.multimodal_latent_concept_reanalysis_w > 0.0
                and args.multimodal_latent_concept_memory_size <= 0):
            sys.exit(
                "ERROR: --multimodal-latent-concept-reanalysis-w requires "
                "--multimodal-latent-concept-memory-size > 0")
        if (args.multimodal_latent_concept_association_decay < 0.0
                or args.multimodal_latent_concept_association_decay >= 1.0):
            sys.exit(
                "ERROR: --multimodal-latent-concept-association-decay must be in [0, 1)")
        if (args.multimodal_latent_concept_association_target_power <= 0.0
                or args.multimodal_latent_concept_graph_predict_target_power <= 0.0):
            sys.exit("ERROR: multimodal graph target powers must be positive")
        if args.multimodal_dropout < 0.0 or args.multimodal_dropout > 1.0:
            sys.exit("ERROR: --multimodal-dropout must be in [0, 1]")
    if args.upload_reading_data:
        if not args.text_reading:
            sys.exit("ERROR: --upload-reading-data requires --text-reading")
        for raw_path in reading_upload_paths(args):
            if os.path.isabs(raw_path):
                sys.exit(
                    "ERROR: --upload-reading-data expects repo-relative paths; "
                    "omit the upload flag for pod-mounted absolute paths")
            local_path = local_path_for_arg(raw_path)
            if not os.path.isfile(local_path):
                sys.exit(f"ERROR: reading upload path not found: {local_path}")
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
    try:
        run = payload(args)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    image_embed_deps = bool(
        (args.image_embed or args.image_eval_generated)
        and args.image_embed_backend == "hf")
    image_score_deps = bool(args.image_score and (
        args.image_score_backend == "pickscore"
        or (args.image_score_backend == "ensemble" and args.image_score_pickscore_w > 0.0)))
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
        if image_embed_deps or image_prompt_embed_deps or image_caption_deps or image_score_deps:
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
        if image_score_deps:
            install_deps += "INSTALL_IMAGE_SCORE_DEPS=1 "
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
        (args.text_reading, "raw reading latent discovery"),
        (args.image_score, "image quality scoring preprocess"),
        (args.image_embed, "image embedding preprocess"),
        (args.vision_understanding, "vision understanding concept memory"),
        (args.image_latent,
         f"latent {args.image_latent_arch} {args.image_cond_mode}-conditioned flow"),
        (args.multimodal, "multimodal bridge"),
    ) if enabled]
    job = " + ".join(image_jobs) if image_jobs else "no supported job selected"
    print("=== PLAN === thinking package on H100: " + job)
    print(f"gpu/cloud : {args.gpu} / {args.cloud}")
    print(f"sync up   : {HERE}/ -> pod:{REMOTE}")
    reading_uploads = reading_upload_paths(args) if args.upload_reading_data else []
    if reading_uploads:
        print("reading data:")
        for local in reading_uploads:
            print(f"  {local_path_for_arg(local)} -> pod:{remote_path_for_arg(local)}")
    if args.upload_image_data:
        print(f"image data: {local_path_for_arg(args.image_root)} -> "
              f"pod:{remote_path_for_arg(args.image_root)}")
        print(f"manifest  : {local_path_for_arg(args.image_manifest)} -> "
              f"pod:{remote_path_for_arg(args.image_manifest)}")
    preference_uploads = (
        preference_upload_paths(args.image_preference_manifest)
        if getattr(args, "image_auto_preference_manifest", False) else []
    )
    if preference_uploads:
        print("preferences:")
        for local in preference_uploads:
            print(f"  {local} -> pod:{local_path_remote_destination(local)}")
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
        for local in reading_uploads:
            sh(upload_path_cmd(local_path_for_arg(local), remote_path_for_arg(local), ssh))
        for local in preference_uploads:
            sh(upload_path_cmd(local, local_path_remote_destination(local), ssh))
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
