#!/usr/bin/env python3
"""Launch the OPEN-VOCAB probe on a RunPod H100: does CHARACTER-level tokenization crack novel-WORD
generalization in the NL<->LOTA bridge, where word-level gave 0.000?

Diagnosed hypothesis (memory: lota-agent-language): a truly-unseen word has an untrained word-level
embedding -> uncopyable; char-level should let it compose from SEEN characters. The CPU char attempt
was undertrained (6k steps, garbled). This trains char-level at scale and compares vs word-level
(control) on the LEXICAL holdout (held-out WORDS). The bridge generates its data in code -> no upload.

HONEST scope: this is the bridge's templated grammar with novel CATEGORY words -- the open-vocab
SUB-problem. Real-English C2 is beyond from-scratch regardless (see thinking/C2_ROADMAP.md).

Safety: DRY-RUN by default (no spend) unless --go. Pod-side `timeout` + ALWAYS-terminate. Auth via env
RUNPOD_API_KEY (never hardcoded). Code shipped from the WORKING DIR (uncommitted ok).

  Dry run (free): python runpod/launch_bridge.py
  Launch ($)    : RUNPOD_API_KEY=... python runpod/launch_bridge.py --go
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

# (tag, extra flags). Both: lexical holdout (unseen WORDS), d384/6L, 20k steps, relabel forces copy.
SWEEP = [
    ("word", "--regime lexical --steps 20000 --d 384 --layers 6 --heads 8 --block 96 --relabel 0.5"),
    ("char", "--regime lexical --char --steps 20000 --d 384 --layers 6 --heads 8 --block 192 --relabel 0.5"),
]


def api(method, path, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(REST + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def sh(cmd):
    print("  $", cmd[:150])
    return subprocess.run(cmd, shell=True).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="c2-openvocab-probe")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--go", action="store_true", help="actually create the pod (spends money)")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY")
    if not key and args.go:
        sys.exit("ERROR: export RUNPOD_API_KEY first (do not hardcode it).")
    pub = os.path.expanduser("~/.ssh/id_rsa.pub")
    pubkey = open(pub).read().strip() if os.path.exists(pub) else ""
    cap = args.max_minutes * 60

    runs = " ; ".join(
        f"echo '=== CONFIG {tag} ==='; python -m thinking.lang_bridge {flags} --device cuda || true"
        for tag, flags in SWEEP)
    preflight = ("python -c 'import torch;print(\"torch\",torch.__version__,\"cuda\","
                 "torch.cuda.is_available())' || pip install -q numpy")
    remote_cmd = (
        f"cd {REMOTE} && rm -f bridge.log /root/bridge.log && "
        f"(({preflight}) && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({runs})')}) "
        f"2>&1 | tee /root/bridge.log; cp /root/bridge.log {REMOTE}/bridge.log 2>/dev/null; true")

    print("=== PLAN === open-vocab probe (char vs word, lexical holdout) on H100")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (name {args.name})")
    for tag, flags in SWEEP:
        print(f"   [{tag}] thinking.lang_bridge {flags} --device cuda")
    print(f"deploy    : working-dir -> pod:{REMOTE} (no data upload; bridge generates in-code)")
    print(f"guard     : pod-side timeout {cap}s + ALWAYS-terminate; cost <= ~${cap/3600*HOURLY_USD:.2f}")
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
            sys.exit("pod never exposed SSH; terminating.")
        time.sleep(25)
        ssh = f"ssh -o StrictHostKeyChecking=no -p {port} root@{ip}"
        up = (f"COPYFILE_DISABLE=1 tar czf - --exclude './.venv*' --exclude '*/__pycache__' "
              f"--exclude './results_gpu' --exclude '*.zip' --exclude './data' --exclude '*.pt' "
              f"--exclude '*.log' --exclude './runs' --exclude './experiments' --exclude './tooling' "
              f"--exclude './artifacts' --exclude './.git' --exclude '*.tgz' -C {HERE} . "
              f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(up)
        script = remote_cmd + "; touch /root/DONE\n"
        for _ in range(5):
            subprocess.run(f"{ssh} 'cat > /root/run.sh'", shell=True, input=script, text=True, timeout=120)
            ok = subprocess.run(f"{ssh} 'test -s /root/run.sh && echo OK'", shell=True,
                                capture_output=True, text=True, timeout=120)
            if "OK" in (ok.stdout or ""):
                break
            time.sleep(10)
        else:
            raise RuntimeError("run.sh upload failed")
        sh(f"{ssh} 'rm -f /root/DONE && nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < cap:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; tail -1 /root/bridge.log'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("run complete"); break
        sh(f"{ssh} 'cd {REMOTE} && tar czf - bridge.log 2>/dev/null' | tar xzf - -C {HERE}")
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
