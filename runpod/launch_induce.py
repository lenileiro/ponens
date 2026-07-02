#!/usr/bin/env python3
"""Launch the IN-CONTEXT RULE-INDUCTION scaling run on a RunPod H100 (thinking/induce.py).

The model must FIND THE RULE from a few input->output demos (structured: value rule add-c mod A +
positional reverse/shift) and apply it to a query -- generalizing to HELD-OUT shifts it never trained
on. On CPU the value/positional induction circuit doesn't form in 1-3k steps (identity 1.0, shift ~0.48,
add-c ~chance): the hypothesis is UNDER-TRAINING -- induction heads emerge via a slow phase transition.
This run trains long (heavy steps) + wider/more heads on GPU and prints an accuracy CURVE per chunk, so
we SEE whether the circuit groks.

Safety: DRY-RUN by default unless --go. Pod-side `timeout` bounds it; pod ALWAYS terminated in `finally`.
Auth via env RUNPOD_API_KEY (NEVER hardcoded). No WordNet needed (pure synthetic task).

  Dry run (free): python runpod/launch_induce.py
  Launch ($)    : RUNPOD_API_KEY=... python runpod/launch_induce.py --go
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
REMOTE = "/workspace/llm"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOURLY_USD = 3.29

# (tag, A, L, K, d, heads, T, steps, seeds). Heavy training to reach the induction-head phase transition.
SWEEP = [
    ("induce_l", 12, 6, 6, 256, 8, 12, 60000, 2),
]


def api(method, path, key, body=None):
    url = REST + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, {"error": f"{type(e).__name__}: {getattr(e, 'reason', e)}"}


def sh(cmd):
    print("  $", cmd[:160])
    return subprocess.run(cmd, shell=True).returncode


def build_cmd():
    lines = []
    for (tag, A, L, K, d, h, T, steps, seeds) in SWEEP:
        lines.append(
            f"echo '=== {tag}: A{A} L{L} K{K} d{d} h{h} T{T} steps{steps} seeds{seeds} ==='; "
            f"python -m thinking.induce --A {A} --L {L} --K {K} --d {d} --heads {h} --T {T} "
            f"--steps {steps} --seeds {seeds} --device cuda --out runs/induce_{tag}.json || true")
    return " ; ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=20)
    ap.add_argument("--name", default="induce-probe")
    ap.add_argument("--max-minutes", type=int, default=45)
    ap.add_argument("--go", action="store_true", help="actually create the pod (spends money)")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY")
    if not key and args.go:
        sys.exit("ERROR: export RUNPOD_API_KEY first (do not hardcode it).")
    pub = os.path.expanduser("~/.ssh/id_rsa.pub")
    pubkey = open(pub).read().strip() if os.path.exists(pub) else ""

    cap = args.max_minutes * 60
    run = build_cmd()
    preflight = ("python -c 'import torch; print(\"torch\", torch.__version__, "
                 "\"cuda\", torch.cuda.is_available())'")
    remote_cmd = (
        f"cd {REMOTE} && rm -f induce.log /root/induce.log && "
        f"(({preflight}) && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"mkdir -p {REMOTE}/runs && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')}) "
        f"2>&1 | tee /root/induce.log; cp /root/induce.log {REMOTE}/induce.log 2>/dev/null; true")

    print("=== PLAN === in-context RULE INDUCTION scaling (thinking/induce.py) on H100")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (disk {args.disk}GB, name {args.name})")
    for (tag, A, L, K, d, h, T, steps, seeds) in SWEEP:
        print(f"   [{tag}] A{A} L{L} K{K} d{d} heads{h} T{T} steps{steps} seeds{seeds}")
    print(f"deploy    : working-dir -> pod:{REMOTE} (no WordNet needed); detached run + tee log")
    print(f"guard     : pod-side timeout {cap}s + ALWAYS-terminate; SSH-wait cap {args.max_minutes}m")
    print(f"cost EST  : up to ~{args.max_minutes/60:.2f} hr * ${HOURLY_USD}/hr = "
          f"~${args.max_minutes/60*HOURLY_USD:.2f} (UPPER BOUND = the time cap)")
    if not args.go:
        print("\n[dry-run] nothing created. Re-run with --go to launch (spends money).")
        return

    print("\n=== creating pod ===")
    body = {"name": args.name, "imageName": IMAGE, "gpuTypeIds": [args.gpu], "gpuCount": 1,
            "cloudType": args.cloud, "containerDiskInGb": args.disk, "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": pubkey}}
    st, pod = api("POST", "/pods", key, body)
    if st not in (200, 201):
        sys.exit(f"create failed: HTTP {st} {pod}")
    pid = pod.get("id") or pod.get("podId")
    print("pod id:", pid)
    t0 = time.time()
    try:
        ip = port = None
        while time.time() - t0 < cap:
            st, p = api("GET", f"/pods/{pid}", key)
            ip = p.get("publicIp"); port = (p.get("portMappings") or {}).get("22")
            print(f"  status={p.get('desiredStatus')} ip={ip} port={port}")
            if p.get("desiredStatus") == "RUNNING" and ip and port:
                break
            time.sleep(12)
        if not (ip and port):
            sys.exit("pod never exposed SSH within cap; terminating.")
        time.sleep(25)
        ssh = f"ssh -o StrictHostKeyChecking=no -p {port} root@{ip}"
        up = (f"COPYFILE_DISABLE=1 tar czf - --exclude './.venv*' --exclude '*/__pycache__' "
              f"--exclude './data' --exclude '*.pt' --exclude '*.log' --exclude './runs' "
              f"--exclude './experiments' --exclude './.git' -C {HERE} . "
              f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(up)
        script = remote_cmd + "; touch /root/DONE\n"
        for _try in range(5):
            subprocess.run(f"{ssh} 'cat > /root/run.sh'", shell=True, input=script, text=True, timeout=120)
            ok = subprocess.run(f"{ssh} 'test -s /root/run.sh && echo OK'", shell=True,
                                capture_output=True, text=True, timeout=120)
            if "OK" in (ok.stdout or ""):
                break
            time.sleep(10)
        else:
            raise RuntimeError("run.sh upload failed")
        sh(f"{ssh} 'rm -f /root/DONE /root/induce.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < cap:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; tail -3 /root/induce.log'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            for ln in (r.stdout or "").strip().splitlines()[-3:]:
                print(" ", ln[:120])
            if "DONE" in (r.stdout or ""):
                print("run complete"); break
        tags = " ".join(f"runs/induce_{t[0]}.json" for t in SWEEP)
        sh(f"{ssh} 'cd {REMOTE} && tar czf - induce.log {tags} 2>/dev/null' | tar xzf - -C {HERE}")
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
