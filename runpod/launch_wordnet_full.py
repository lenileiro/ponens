#!/usr/bin/env python3
"""Launch the MEANING scaling sweep on a RunPod H100 (thinking/meaning.py).

Capture meaning by CONTRASTIVE pair-training over ALL WordNet signals (all POS) + a BRAIN-VERIFIED
is-a pointer probe. Local (600 concepts, CPU): contrastive retrieval ~21x chance + POS 0.70 (meaning
captured), but the verified is-a pointer is starved for parent-name signal at small scale. This sweep
SCALES concept count + encoder + steps and asks: does retrieval sharpen AND the verified is-a pointer
strengthen (esp. on UNSEEN parents -- the open-vocab-wall test)?

Safety: DRY-RUN by default unless --go. Pod-side `timeout` bounds the run; pod ALWAYS terminated in
`finally`. Auth via env RUNPOD_API_KEY (NEVER hardcoded). Code shipped from the WORKING DIR; WordNet
installed on the pod.

  Dry run (free): python runpod/launch_meaning.py
  Launch ($)    : RUNPOD_API_KEY=... python runpod/launch_meaning.py --go
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

TRAIN_STEPS = 6000


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
    return (f"python -m thinking.wordnet_full --steps {TRAIN_STEPS} --batch 256 --device cuda "
            f"--save runs/wn_full.pt --out runs/wn_full.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="wordnet-full-train")
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
    preflight = (
        "pip install -q transformers nltk && "
        "python -c \"import torch,transformers; print('torch',torch.__version__,'cuda',"
        "torch.cuda.is_available(),'transformers',transformers.__version__)\" && "
        "python -c \"import nltk; nltk.download('wordnet',quiet=True); nltk.download('omw-1.4',quiet=True); "
        "from nltk.corpus import wordnet as wn; print('wordnet', len(list(wn.all_synsets()))); "
        "from transformers import AutoModel; AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); "
        "print('minilm ok')\"")
    remote_cmd = (
        f"cd {REMOTE} && rm -f wnfull.log /root/wnfull.log && "
        f"(({preflight}) && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"mkdir -p {REMOTE}/runs && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')}) "
        f"2>&1 | tee /root/wnfull.log; "
        f"cp /root/wnfull.log {REMOTE}/wnfull.log 2>/dev/null; true")

    print("=== PLAN === MEANING scaling sweep (thinking/meaning.py) on H100, brain-verified")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (disk {args.disk}GB, name {args.name})")
    print(f"train     : fine-tune all-MiniLM on ALL WordNet def->parent pairs, {TRAIN_STEPS} steps")
    print(f"deploy    : working-dir -> pod:{REMOTE} (uncommitted ok); WordNet installed on pod")
    print(f"fetch     : wnfull.log + runs/meaning_*.json -> {HERE}/")
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
        sh(f"{ssh} 'rm -f /root/DONE /root/wnfull.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < cap:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; "
                                   f"tail -1 /root/wnfull.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("run complete")
                break
        tags = "runs/wn_full.pt runs/wn_full.json"
        fetch = (f"{ssh} 'cd {REMOTE} && tar czf - wnfull.log {tags} 2>/dev/null' "
                 f"| tar xzf - -C {HERE}")
        sh(fetch)
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
