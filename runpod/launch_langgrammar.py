#!/usr/bin/env python3
"""Launch the LANGGRAMMAR run on a RunPod H100: position-invariant, grammar-inferred multi-relation reasoning
(thinking/langgrammar.py). The question this answers: with enough scale, does a from-scratch model learn to FORM
RULES NATURALLY -- infer an invented language's word order from the prompt (markers by recurrence, not position),
reason over two relations + inheritance, and PRODUCE the answer in that grammar -- and does it generalize to
(1) word orders never trained and (2) prompts longer than trained (shifted absolute positions)?

Each sweep config writes runs/langgrammar_<tag>.json with in-distribution + the two held-out robustness metrics
(unseen_grammar, random_position). No dataset needed -- episodes are synthetic per-step, so only torch+numpy.

Safety: DRY-RUN by default (nothing created, no spend) unless --go. Pod-side `timeout` bounds the run and the pod
is ALWAYS terminated in `finally`. Auth via env RUNPOD_API_KEY (NEVER hardcoded). Code shipped from the WORKING
DIR (carries uncommitted langgrammar edits).

  Dry run (free): python runpod/launch_langgrammar.py
  Launch ($)    : RUNPOD_API_KEY=... python runpod/launch_langgrammar.py --go
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

# Scaling sweep: (tag, d, layers, heads, steps, batch, seed). head dim (d//heads) must be EVEN for RoPE.
SWEEP = [
    ("s", 256, 6, 8, 40000, 64, 0),     # base scale
    ("m", 384, 8, 8, 80000, 96, 0),     # +width/depth +steps
    ("l", 512, 8, 8, 120000, 128, 0),   # +more (find where the held-out metrics move)
]


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
                return r.status, (json.loads(txt) if txt.strip() else {})
            except json.JSONDecodeError:
                return r.status, {"error": txt}
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}
    except urllib.error.URLError as e:
        return 0, {"error": f"network error: {getattr(e, 'reason', e)}"}
    except (TimeoutError, OSError) as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def sh(cmd):
    print("  $", cmd[:160])
    return subprocess.run(cmd, shell=True).returncode


def build_sweep_cmd():
    """Shell snippet: run each sweep config, each writing runs/langgrammar_<tag>.json. ';' so one config failing
    does not abort the rest; '|| true' keeps the pipeline alive for the tee."""
    lines = []
    for (tag, d, L, h, steps, batch, seed) in SWEEP:
        lines.append(
            f"echo '=== CONFIG {tag}: d{d}/L{L}/h{h} steps{steps} batch{batch} seed{seed} ==='; "
            f"python -m thinking.langgrammar --steps {steps} --d {d} --layers {L} --heads {h} "
            f"--bs {batch} --seed {seed} --device cuda --out runs/langgrammar_{tag}.json || true")
    return " ; ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="langgrammar-run")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--print-payload", action="store_true")
    ap.add_argument("--go", action="store_true", help="actually create the pod (spends money)")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY")
    if not key and args.go:
        sys.exit("ERROR: export RUNPOD_API_KEY first (do not hardcode it).")
    pub = os.path.expanduser("~/.ssh/id_rsa.pub")
    pubkey = open(pub).read().strip() if os.path.exists(pub) else ""

    cap = args.max_minutes * 60
    run = build_sweep_cmd()
    preflight = ("python -c 'import torch,numpy,sys; "
                 "print(\"deps ok torch\",torch.__version__,\"cuda\",torch.cuda.is_available())' "
                 "|| pip install -q numpy")
    remote_cmd = (
        f"cd {REMOTE} && rm -f langgrammar.log /root/lg.log && "
        f"(({preflight}) && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"mkdir -p {REMOTE}/runs && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')}) "
        f"2>&1 | tee /root/lg.log; "
        f"cp /root/lg.log {REMOTE}/langgrammar.log 2>/dev/null; true")

    print("=== PLAN === langgrammar (position-invariant, grammar-inferred reasoning) on H100")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (disk {args.disk}GB, name {args.name})")
    print(f"sweep     : {len(SWEEP)} configs")
    for (tag, d, L, h, steps, batch, seed) in SWEEP:
        print(f"   [{tag}] d{d}/L{L}/h{h} steps{steps} batch{batch} seed{seed}")
    print(f"deploy    : working-dir -> pod:{REMOTE} (uncommitted ok); synthetic data, no upload")
    print(f"fetch     : langgrammar.log + runs/langgrammar_*.json -> {HERE}/")
    print(f"guard     : pod-side timeout {cap}s + ALWAYS-terminate; SSH-wait cap {args.max_minutes}m")
    print(f"cost EST  : up to ~{args.max_minutes/60:.1f} hr * ${HOURLY_USD}/hr = "
          f"~${args.max_minutes/60*HOURLY_USD:.2f} (UPPER BOUND = the time cap)")
    if args.print_payload:
        print("\n=== POD PAYLOAD ===\n" + remote_cmd)
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
              f"--exclude './tooling' --exclude './artifacts' --exclude './.git' "
              f"--exclude '*.tgz' -C {HERE} . "
              f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(up)
        chk = subprocess.run(f"{ssh} 'test -s {REMOTE}/thinking/langgrammar.py && echo CODE_OK'",
                             shell=True, capture_output=True, text=True, timeout=120)
        if "CODE_OK" not in (chk.stdout or ""):
            raise RuntimeError("code upload failed (langgrammar.py missing on pod)")
        print("  code upload verified")
        script = remote_cmd + "; touch /root/DONE\n"
        for _try in range(5):
            subprocess.run(f"{ssh} 'cat > /root/run.sh'", shell=True, input=script,
                           text=True, timeout=120)
            ok = subprocess.run(f"{ssh} 'test -s /root/run.sh && echo OK'", shell=True,
                                capture_output=True, text=True, timeout=120)
            if "OK" in (ok.stdout or ""):
                break
            time.sleep(10)
        else:
            raise RuntimeError("run.sh upload failed after 5 attempts")
        sh(f"{ssh} 'rm -f /root/DONE /root/lg.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < cap:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; "
                                   f"tail -1 /root/lg.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("run complete")
                break
        fetch = (f"{ssh} 'cd {REMOTE} && tar czf - langgrammar.log runs/langgrammar_s.json "
                 f"runs/langgrammar_m.json runs/langgrammar_l.json 2>/dev/null' | tar xzf - -C {HERE}")
        sh(fetch)
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
