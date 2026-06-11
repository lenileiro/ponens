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
    if (args.vision or args.image2 or args.image_flow or args.image_latent or args.audio
            or args.multimodal):
        VI = PY.replace("thinking.cli", "thinking.vision")
        if args.vision:                                    # IMAGE-0/1: visual factors -> facts
            cmds.append(f"{VI} --train --steps {args.train_steps or 2000} "
                        f"--batch {args.batch} --dim {args.dim or 64} "
                        f"--arch {args.vision_arch} "
                        f"--out runs/vision_object_encoder.pt")
        if args.image2:                                    # IMAGE-2: head-aware FER experiment
            I2 = PY.replace("thinking.cli", "thinking.image2")
            cmds.append(f"{I2} --steps {args.train_steps or 800} "
                        f"--seeds {shlex_quote(args.seeds)} --dim {args.dim or 64} "
                        f"--out runs/image2_bottleneck.json")
        if args.image_flow:                                # first fact-conditioned generator
            IF = PY.replace("thinking.cli", "thinking.image_flow")
            cmds.append(f"{IF} --train --steps {args.train_steps or 800} "
                        f"--batch {args.batch} --dim {args.dim or 64} "
                        f"--out runs/image_flow.pt")
        if args.image_latent:                              # IMAGE-3: latent flow generator
            IL = PY.replace("thinking.cli", "thinking.image_latent")
            ckpt = f"runs/image_latent_{args.image_latent_arch}.pt"
            train = (f"{IL} --train --ae-steps {args.train_steps or 800} "
                     f"--flow-steps {args.train_steps or 800} --batch {args.batch} "
                     f"--hidden {args.dim or 64} --flow-arch {args.image_latent_arch} "
                     f"--cond-drop {args.image_cond_drop} "
                     f"--cfg-scale {args.image_cfg_scale} "
                     f"--sample-steps {args.image_sample_steps} "
                     f"--roundtrip-samples {args.image_roundtrip_samples} "
                     f"--flow-semantic-w {args.image_flow_semantic_w} "
                     f"--out {ckpt}")
            if args.image_eval_sweep:
                train += (f" && {IL} --eval-checkpoint {ckpt} "
                          f"--cfg-scales {shlex_quote(args.image_cfg_sweep)} "
                          f"--sample-steps-list {shlex_quote(args.image_sample_steps_sweep)} "
                          f"--roundtrip-samples {args.image_roundtrip_samples} "
                          f"--eval-out runs/image_latent_{args.image_latent_arch}_sweep.json")
            cmds.append(train)
        if args.audio:                                     # AUDIO-1: audio factors -> facts
            AU = PY.replace("thinking.cli", "thinking.audio")
            cmds.append(f"{AU} --steps {args.train_steps or 500} "
                        f"--seeds {shlex_quote(args.seeds)} --out runs/audio1_fer.json")
        if args.multimodal:                                # M-0: image+audio -> canonical facts
            MM = PY.replace("thinking.cli", "thinking.multimodal")
            cmds.append(f"{MM} --steps {args.train_steps or 400} "
                        f"--out runs/m0_multimodal.json")
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
                    help="rung 0 = chain (the legacy-validated task on the production trainer)")
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
                    help="IMAGE-0/1: train visual factor encoder + FER probe report")
    ap.add_argument("--vision-arch", default="shared", choices=("shared", "factored", "bottleneck"),
                    help="vision encoder architecture for --vision")
    ap.add_argument("--image2", action="store_true",
                    help="IMAGE-2: compare shared, bottleneck, and joint visual FER arms")
    ap.add_argument("--image-flow", action="store_true", dest="image_flow",
                    help="train the tiny fact-conditioned rectified-flow image generator")
    ap.add_argument("--image-latent", action="store_true", dest="image_latent",
                    help="IMAGE-3: train semantic autoencoder + latent fact-conditioned flow")
    ap.add_argument("--image-latent-arch", default="conv", choices=("conv", "dit"),
                    dest="image_latent_arch", help="latent velocity architecture")
    ap.add_argument("--image-cond-drop", type=float, default=0.0, dest="image_cond_drop",
                    help="condition dropout for classifier-free latent image guidance")
    ap.add_argument("--image-cfg-scale", type=float, default=1.0, dest="image_cfg_scale",
                    help="classifier-free guidance scale for latent image sampling")
    ap.add_argument("--image-sample-steps", type=int, default=4, dest="image_sample_steps",
                    help="Euler sampling steps for latent image evaluation")
    ap.add_argument("--image-roundtrip-samples", type=int, default=1,
                    dest="image_roundtrip_samples",
                    help="generated samples per color/shape condition during image eval")
    ap.add_argument("--image-flow-semantic-w", type=float, default=0.0,
                    dest="image_flow_semantic_w",
                    help="semantic endpoint alignment weight for latent image flow training")
    ap.add_argument("--image-eval-sweep", action="store_true", dest="image_eval_sweep",
                    help="after latent training, sweep CFG and sampler steps from the checkpoint")
    ap.add_argument("--image-cfg-sweep", default="1.0,1.25,1.5,2.0",
                    dest="image_cfg_sweep",
                    help="comma-separated CFG scales for --image-eval-sweep")
    ap.add_argument("--image-sample-steps-sweep", default="4,8,16",
                    dest="image_sample_steps_sweep",
                    help="comma-separated sampler step counts for --image-eval-sweep")
    ap.add_argument("--audio", action="store_true",
                    help="AUDIO-1: train synthetic audio factor FER experiment")
    ap.add_argument("--multimodal", action="store_true",
                    help="M-0: train one prefix-conditioned LM on image+audio extraction")
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
    bad_decodes = sorted(set(args.eval_decodes) - {"sample", "hybrid", "constrained", "ranker"})
    if bad_decodes:
        sys.exit(f"ERROR: unsupported --eval-decodes values: {','.join(bad_decodes)}")
    bad_arms = sorted(set(args.curve_arms) - {"aux", "noaux"})
    if bad_arms:
        sys.exit(f"ERROR: unsupported --curve-arms values: {','.join(bad_arms)}")

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
    setup = ("pip install -q numpy tokenizers pandas pyarrow" if args.fast   # image torch; verbalizer
             else f"WORKDIR={REMOTE} bash runpod/setup.sh")                  # deps incl. parquet corpora
    # tee to LOCAL disk: /workspace is a network volume that stalls under streaming writes
    # (see runpod/setup.sh -- it cost us rung B4: training was healthy, only the log froze)
    remote_cmd = (
        f"cd {REMOTE} && rm -f thinking.log /root/thinking.log && "
        f"({setup} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')}) "
        f"2>&1 | tee /root/thinking.log; "
        f"cp /root/thinking.log {REMOTE}/thinking.log 2>/dev/null; true")

    image_jobs = [name for enabled, name in (
        (args.vision, "vision factor encoder"),
        (args.image2, "image2 FER arms"),
        (args.image_flow, "fact-conditioned flow"),
        (args.image_latent, "latent fact-conditioned flow"),
        (args.audio, "audio FER arms"),
        (args.multimodal, "multimodal bridge"),
    ) if enabled]
    job = " + ".join(image_jobs) if image_jobs else "kinship multi-seed"
    print("=== PLAN === thinking package on H100: " + job
          + (" + chain grid" if args.sweep and not args.vision else ""))
    print(f"gpu/cloud : {args.gpu} / {args.cloud}")
    print(f"seeds     : {args.seeds}")
    print(f"sync up   : {HERE}/ -> pod:{REMOTE}")
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
