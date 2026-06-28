#!/usr/bin/env python3
"""Launch the byte-level SimCSE SEMANTIC encoder (thinking/langbyte.py) on a RunPod H100: contrastive pretrain on
1.6M public tweets (Sentiment140 -- emotion-rich, NOT customer data), then ZERO-SHOT eval on emotions (where the
char-n-gram baseline is ~chance). Question: does a properly-scaled language-agnostic semantic encoder beat the
trivial char-n-gram baseline on a SEMANTIC task?

Honest expectation: uncertain -- from-scratch byte-level contrastive learning is data-hungry; this is the test.

Safety: DRY-RUN by default unless --go. Pod-side timeout + ALWAYS terminate in finally. Env RUNPOD_API_KEY only.
Uploads the corpus (data/sentiment140_text.txt) + eval (kaggle_data/emotions.csv) explicitly (both gitignored).

  Dry run (free): python runpod/launch_langbyte.py
  Launch ($)    : RUNPOD_API_KEY=... python runpod/launch_langbyte.py --go
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
RUN = ("python -u -m thinking.langbyte --corpus data/sentiment140_text.txt "
       "--steps 40000 --d 384 --layers 6 --bs 256 --cap 600000 --device cuda")   # -u: unbuffered; periodic eval logs


def api(method, path, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(REST + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, {"error": str(e)}


def sh(cmd):
    print("  $", cmd[:160]); return subprocess.run(cmd, shell=True).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="langbyte-simcse")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--go", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY")
    if not key and args.go:
        sys.exit("ERROR: export RUNPOD_API_KEY first (do not hardcode it).")
    pub = os.path.expanduser("~/.ssh/id_rsa.pub")
    pubkey = open(pub).read().strip() if os.path.exists(pub) else ""
    cap = args.max_minutes * 60
    preflight = ("python -c 'import torch,numpy;print(\"deps\",torch.__version__,torch.cuda.is_available())' "
                 "|| pip install -q numpy")
    remote_cmd = (
        f"cd {REMOTE} && rm -f lb.log /root/lb.log && "
        f"(({preflight}) && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && {RUN}')}) "
        f"2>&1 | tee /root/lb.log; cp /root/lb.log {REMOTE}/lb.log 2>/dev/null; true")

    print("=== PLAN === langbyte byte-SimCSE semantic encoder on H100")
    print(f"gpu/cloud : {args.gpu} / {args.cloud} (disk {args.disk}GB)")
    print(f"run       : {RUN}")
    print(f"upload    : code (venv/data/kaggle_data excluded) + data/sentiment140_text.txt + kaggle_data/emotions.csv")
    print(f"guard     : pod timeout {cap}s + ALWAYS terminate; cost <= ~${args.max_minutes/60*HOURLY_USD:.2f}")
    if not args.go:
        print("\n[dry-run] nothing created. Re-run with --go to launch (spends money).")
        return

    st, pod = api("POST", "/pods", key, {"name": args.name, "imageName": IMAGE, "gpuTypeIds": [args.gpu],
                  "gpuCount": 1, "cloudType": args.cloud, "containerDiskInGb": args.disk, "ports": ["22/tcp"],
                  "env": {"PUBLIC_KEY": pubkey}})
    if st not in (200, 201):
        sys.exit(f"create failed: HTTP {st} {pod}")
    pid = pod.get("id"); print("pod id:", pid)
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
            sys.exit("pod never exposed SSH; terminating.")
        time.sleep(25)
        ssh = f"ssh -o StrictHostKeyChecking=no -p {port} root@{ip}"
        up = (f"COPYFILE_DISABLE=1 tar czf - --exclude './.venv*' --exclude './venv*' --exclude '*/__pycache__' "
              f"--exclude './data' --exclude './kaggle_data' --exclude '*.pt' --exclude '*.log' --exclude './runs' "
              f"--exclude './experiments' --exclude './.git' --exclude '*.zip' --exclude '*.tgz' -C {HERE} . "
              f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(up)
        data_up = (f"COPYFILE_DISABLE=1 tar czf - -C {HERE} data/sentiment140_text.txt kaggle_data/emotions.csv "
                   f"| {ssh} 'cd {REMOTE} && tar --no-same-owner -xzf -'")
        sh(data_up)
        chk = subprocess.run(f"{ssh} 'test -s {REMOTE}/data/sentiment140_text.txt && "
                             f"test -s {REMOTE}/kaggle_data/emotions.csv && echo DATA_OK'",
                             shell=True, capture_output=True, text=True, timeout=180)
        if "DATA_OK" not in (chk.stdout or ""):
            raise RuntimeError("data upload failed (corpus or emotions missing on pod)")
        print("  data upload verified")
        script = remote_cmd + "; touch /root/DONE\n"
        subprocess.run(f"{ssh} 'cat > /root/run.sh'", shell=True, input=script, text=True, timeout=120)
        sh(f"{ssh} 'rm -f /root/DONE && nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < cap:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; tail -2 /root/lb.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            for ln in (r.stdout or "").strip().splitlines()[-2:]:
                print(" ", ln[:120])
            if "DONE" in (r.stdout or ""):
                print("run complete"); break
        sh(f"{ssh} 'cd {REMOTE} && tar czf - lb.log 2>/dev/null' | tar xzf - -C {HERE}")
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
