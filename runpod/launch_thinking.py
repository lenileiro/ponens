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
    cmds = []
    if args.ablate:
        cmds.append(f"{PY} ablate --steps 800")
    if args.verbalize:
        cmds.append(f"{PY.replace('thinking.cli', 'thinking.verbalize')} "
                    f"--out runs/verbalizer.pt && "
                    f"{PY.replace('thinking.cli', 'thinking.verbalize')} "
                    f"--sample runs/verbalizer.pt")
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
    if args.stair:                                         # staircase: minimal world, decisive evals
        run = f"runs/stair_{args.stair_world}"
        canon = ((" --canon" if args.canon else "") + (" --bank" if args.bank else "") + (" --no-curriculum" if args.no_curriculum else ""))
        simple = " --simple" if args.stair_world == "kinship" else ""
        if args.stair_world == "kinship" and args.deep_depth and args.stair:
            simple += f" --deep-depth {args.deep_depth} --deep-frac 0.3"
        sbatch = 8 if (args.deep_depth and args.stair_world == "kinship") else 32
        hops = ("2,3" if not (args.deep_depth and args.stair_world == "kinship")
                else f"2,3,{args.deep_depth // 2},{args.deep_depth}")
        if args.stair_world == "chain":
            hops = "2,4,6"
        return " && ".join([
            f"{PY} train --world {args.stair_world}{simple}{canon} --out {run} --seed 0 "
            f"--batch {sbatch} --steps {args.train_steps or 4000}"
            + (f" --dim {args.dim}" if args.dim else ""),
            f"{PY} eval {run} --mode verified --split iid --hops {hops} --n 20 --train-names",
            f"{PY} eval {run} --mode free --split iid --hops {hops} --n 20 --train-names",
            f"{PY} eval {run} --mode verified --split iid --hops {hops} --n 20",
            f"{PY} eval {run} --mode free --split iid --hops {hops} --n 20",
            f"{PY} eval {run} --mode verified --split iid --hops {hops} --n 20 --phrasings eval",
            f"{PY} eval {run} --mode free --split iid --hops {hops} --n 20 --phrasings eval",
            f"{PY} demo {run} --k 2",
        ])
    neg = " --neg" if args.neg else ""
    loops = f" --loops {args.loops}" if args.loops else ""
    noloop = " --no-loop" if args.no_loop else ""
    steps = f" --steps {args.train_steps}" if args.train_steps else (
        " --steps 1000 --examples 3000" if args.fast else "")
    for s in args.seeds.split(","):
        run = f"runs/kin_s{s}"
        cmds.append(f"{PY} train --world kinship --deep-depth {args.deep_depth} --out {run} "
                    f"--seed {s} --batch {args.batch}{steps}{loops}{noloop}{neg}")
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
    ap.add_argument("--lengen", action="store_true", help="rung L: depth generalization")
    ap.add_argument("--sweep", action="store_true", help="also run the chain-world grid")
    ap.add_argument("--max-minutes", type=int, default=150)
    ap.add_argument("--go", action="store_true", help="actually create the pod (spends money)")
    args = ap.parse_args()

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
    remote_cmd = (f"cd {REMOTE} && {setup} && "
                  f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
                  f'timeout {cap}s bash -c "cd {REMOTE} && ({run}) 2>&1 | tee /root/thinking.log"; '
                  f"cp /root/thinking.log {REMOTE}/thinking.log 2>/dev/null; true")

    print("=== PLAN === thinking package on H100: kinship multi-seed"
          + (" + chain grid" if args.sweep else ""))
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
        up = (f"COPYFILE_DISABLE=1 tar czf - --exclude './.venv*' --exclude '*/__pycache__' "
              f"--exclude './results_gpu' --exclude '*.zip' --exclude './data' --exclude '*.pt' "
              f"--exclude '*.log' --exclude './runs' --exclude './experiments' "
              f"--exclude './tooling' --exclude './artifacts' --exclude '*.tgz' -C {HERE} . "
              f"| {ssh} 'mkdir -p {REMOTE} && tar xzf - -C {REMOTE}'")
        sh(up)
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
        sh(f"{ssh} 'nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
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
