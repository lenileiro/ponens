#!/usr/bin/env python3
"""RunPod H100 runner for the verified-reasoning trainer (thinking/reason_lang_neg.py).

Trains the verified-best NL reasoner (isa/part_of/located_in TRANS + has_prop INHERIT + EXCLUDE
negation-as-failure, proof-supervised, ScratchpadLM pointer head) at scale on a single H100.
Pod-side `timeout` bounds the run and cleanup ALWAYS terminates the pod. Defaults to DRY-RUN:
nothing is created and no money is spent unless `--go` is passed.

Auth: export RUNPOD_API_KEY (never hardcoded).
  Dry-run (default): python runpod/launch_reasoning.py
  Launch (spends $): RUNPOD_API_KEY=... python runpod/launch_reasoning.py --go

Modeled on runpod/launch_thinking.py (same REST endpoint, image, pod create/upload/detached-run/
fetch/terminate pattern).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from shlex import quote as shlex_quote

REST = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
REMOTE = "/workspace/fer_relational"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_REMOTE = "runs/reasoning.json"
HOURLY_USD = 3.29


def api(method, path, key, body=None):
    url = REST + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode()
            try:
                payload = json.loads(txt) if txt.strip() else {}
            except json.JSONDecodeError:
                payload = {"error": txt}
            return r.status, payload
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}
    except urllib.error.URLError as e:
        return 0, {"error": f"network error: {getattr(e, 'reason', e)}"}
    except TimeoutError as e:
        return 0, {"error": f"timeout: {e}"}
    except OSError as e:
        return 0, {"error": f"os error: {e}"}


def sh(cmd):
    print("  $", cmd)
    return subprocess.run(cmd, shell=True).returncode


def build_run(args):
    """The exact `python -m thinking.reason_lang_neg ...` command run on the pod."""
    return (
        f"python -m thinking.reason_lang_neg "
        f"--steps {args.steps} --seeds {args.seeds} --lr {args.lr} "
        f"--dim {args.dim} --layers {args.layers} --heads {args.heads} "
        f"--batch {args.batch} --max-len {args.max_len} "
        f"--which {args.which} --device auto --out {RESULTS_REMOTE}"
    )


def cost_estimate(args):
    """Rough cost estimate (CLEARLY an estimate). Assumes a per-seed steps/sec guess on H100."""
    runs = args.seeds * (2 if args.which == "both" else 1)
    total_steps = runs * args.steps
    est_sec = total_steps / max(1.0, args.steps_per_sec)
    est_sec += runs * args.eval_overhead_sec       # greedy-decode eval after each train run
    est_hr = est_sec / 3600.0
    return est_sec, est_hr, est_hr * HOURLY_USD, runs, total_steps


def main():
    ap = argparse.ArgumentParser()
    # scale defaults for the GPU run (the lead will tune)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=240)
    ap.add_argument("--which", choices=["A", "B", "both"], default="both")
    # cost-estimate knobs (rough guesses, clearly labeled)
    ap.add_argument("--steps-per-sec", type=float, default=25.0,
                    help="rough H100 train steps/sec guess for the cost ESTIMATE")
    ap.add_argument("--eval-overhead-sec", type=float, default=180.0,
                    help="rough per-run greedy-eval overhead (sec) for the cost ESTIMATE")
    # pod knobs
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="fer-reasoning")
    ap.add_argument("--max-minutes", type=int, default=120)
    ap.add_argument("--ref", default="HEAD",
                    help="git ref to ship (pinned tar of one committed tree). Empty to ship working dir.")
    ap.add_argument("--print-payload", action="store_true")
    ap.add_argument("--go", action="store_true", help="actually create the pod (spends money)")
    args = ap.parse_args()

    assert args.dim % args.heads == 0, f"--dim ({args.dim}) must be divisible by --heads ({args.heads})"
    assert (args.dim // args.heads) % 2 == 0, \
        f"head dim ({args.dim // args.heads}) must be even for RoPE"

    key = os.environ.get("RUNPOD_API_KEY")
    if not key and args.go:
        sys.exit("ERROR: export RUNPOD_API_KEY first (do not hardcode it).")
    pub = os.path.expanduser("~/.ssh/id_rsa.pub")
    pubkey = open(pub).read().strip() if os.path.exists(pub) else ""

    body = {"name": args.name, "imageName": IMAGE, "gpuTypeIds": [args.gpu], "gpuCount": 1,
            "cloudType": args.cloud, "containerDiskInGb": args.disk, "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": pubkey}}
    cap = args.max_minutes * 60
    run = build_run(args)
    setup = f"WORKDIR={REMOTE} bash runpod/setup.sh"
    # tee to LOCAL disk: /workspace is a network volume that stalls under streaming writes.
    remote_cmd = (
        f"cd {REMOTE} && rm -f reasoning.log /root/reasoning.log && "
        f"({setup} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"mkdir -p {REMOTE}/runs && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')}) "
        f"2>&1 | tee /root/reasoning.log; "
        f"cp /root/reasoning.log {REMOTE}/reasoning.log 2>/dev/null; true")

    est_sec, est_hr, est_usd, runs, total_steps = cost_estimate(args)
    print("=== PLAN === verified-reasoning trainer (thinking/reason_lang_neg.py) on H100")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (disk {args.disk}GB, name {args.name})")
    print(f"model     : dim={args.dim} layers={args.layers} heads={args.heads} batch={args.batch} "
          f"max_len={args.max_len}")
    print(f"train     : steps={args.steps} seeds={args.seeds} lr={args.lr} which={args.which} "
          f"({runs} train runs, {total_steps} total steps)")
    print(f"sync up   : {HERE}/ -> pod:{REMOTE}  (ref={args.ref or 'working-dir'})")
    print(f"fetch     : reasoning.log + {RESULTS_REMOTE} -> {HERE}/")
    print(f"guard     : pod-side timeout {cap}s + ALWAYS-terminate; SSH-wait cap {args.max_minutes}m")
    print(f"cost EST  : ~{est_hr:.2f} hr * ${HOURLY_USD}/hr = ~${est_usd:.2f}  "
          f"[ESTIMATE ONLY: assumes {args.steps_per_sec:.0f} steps/sec + "
          f"{args.eval_overhead_sec:.0f}s eval/run]")
    print("\n=== EXACT REMOTE COMMAND (under pod-side timeout) ===")
    print(f"  {run}")
    if args.print_payload:
        print("\n=== POD PAYLOAD ===")
        print(remote_cmd)
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
        if args.ref:
            # PINNED DEPLOY: ship exactly one committed tree (avoids mid-edit snapshot drift).
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
        # DETACHED execution: nohup on the pod + short-poll (a dropped SSH pipe must not take the
        # cost-guard with it).
        script = remote_cmd + "; touch /root/DONE\n"
        for _try in range(5):                              # VERIFIED upload (a silent network blip
            subprocess.run(f"{ssh} 'cat > /root/run.sh'",  # once shipped an empty script)
                           shell=True, input=script, text=True, timeout=120)
            ok = subprocess.run(f"{ssh} 'test -s /root/run.sh && echo OK'", shell=True,
                                capture_output=True, text=True, timeout=120)
            if "OK" in (ok.stdout or ""):
                break
            time.sleep(10)
        else:
            raise RuntimeError("run.sh upload failed after 5 attempts")
        sh(f"{ssh} 'rm -f /root/DONE /root/reasoning.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < args.max_minutes * 60:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; "
                                   f"tail -1 /root/reasoning.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue                                   # network blip: poll again
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("run complete")
                break
        fetch = (f"{ssh} 'cd {REMOTE} && tar czf - reasoning.log {RESULTS_REMOTE} 2>/dev/null' "
                 f"| tar xzf - -C {HERE}")
        sh(fetch)
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
