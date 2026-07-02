#!/usr/bin/env python3
"""Launch the DEFINITION->MEANING scaling probe on a RunPod H100 (thinking/define.py).

The question: can the agent learn MEANING by READING WordNet definitions well enough to GENERALIZE to
NEVER-SEEN concepts -- crossing the concept-level open-vocab wall the symbol code could not (held-out
exact 0.000 on unseen-parent concepts)? On a handful of concepts a char-encoder just MEMORIZES (train
exact 1.0 / held-out 0.0). This sweep SCALES the concept count + model + steps and asks whether the
shared-word -> meaning RULE finally generalizes -- and whether the DEFINITION beats a NAME-only baseline
(proving the understanding comes from the gloss, not the surface name). Every conveyed fact is
brain-verified (datalog/kernel closure), so the gate is faithful meaning, not loss.

Safety: DRY-RUN by default (nothing created, no spend) unless --go. Pod-side `timeout` bounds the run
and the pod is ALWAYS terminated in `finally`. Auth via env RUNPOD_API_KEY (NEVER hardcoded). Code is
shipped from the WORKING DIR (uncommitted define.py edits ok). WordNet is installed on the pod.

  Dry run (free): python runpod/launch_define.py
  Launch ($)    : RUNPOD_API_KEY=... python runpod/launch_define.py --go
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

# (tag, input, per_cat, cap, d, steps, batch). def vs NAME-baseline at each scale = the control;
# scale (concept count + model + steps) grows down the list -> a curve for WHERE definitions generalize.
SWEEP = [
    ("def_1k",  "def",  150, 1200, 256, 12000, 256),
    ("name_1k", "name", 150, 1200, 256, 12000, 256),   # baseline: surface name only (no gloss)
    ("def_2k",  "def",  300, 2500, 384, 16000, 256),
    ("name_2k", "name", 300, 2500, 384, 16000, 256),
    ("def_3k",  "def",  600, 4000, 512, 20000, 256),   # ~3500 concepts, bigger model
    ("name_3k", "name", 600, 4000, 512, 20000, 256),
    ("both_3k", "both", 600, 4000, 512, 20000, 256),   # synonyms + gloss at full scale
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
    lines = []
    for (tag, inp, pc, cap, d, steps, batch) in SWEEP:
        lines.append(
            f"echo '=== CONFIG {tag}: input={inp} per_cat={pc} cap={cap} d{d} steps{steps} ==='; "
            f"python -m thinking.define --input {inp} --per-cat {pc} --cap {cap} --d {d} "
            f"--steps {steps} --batch {batch} --device cuda --out runs/define_{tag}.json || true")
    return " ; ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="define-meaning-probe")
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
    # define.py needs torch+numpy (base image) + nltk + the WordNet corpus. Install + download on pod.
    preflight = (
        "python -c 'import torch,numpy,sys; print(\"torch\",torch.__version__,"
        "\"cuda\",torch.cuda.is_available())' && pip install -q nltk && "
        "python -c \"import nltk; nltk.download('wordnet',quiet=True); "
        "nltk.download('omw-1.4',quiet=True); from nltk.corpus import wordnet as wn; "
        "print('wordnet ok', wn.synset('dog.n.01').definition())\"")
    remote_cmd = (
        f"cd {REMOTE} && rm -f define.log /root/define.log && "
        f"(({preflight}) && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"mkdir -p {REMOTE}/runs && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')}) "
        f"2>&1 | tee /root/define.log; "
        f"cp /root/define.log {REMOTE}/define.log 2>/dev/null; true")

    print("=== PLAN === definition->meaning scaling probe (thinking/define.py) on H100, brain-verified")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (disk {args.disk}GB, name {args.name})")
    print(f"sweep     : {len(SWEEP)} configs")
    for (tag, inp, pc, cap_, d, steps, batch) in SWEEP:
        print(f"   [{tag}] input={inp} per_cat={pc} cap={cap_} d{d} steps{steps} batch{batch}")
    print(f"deploy    : working-dir -> pod:{REMOTE} (uncommitted ok); WordNet installed on pod")
    print(f"fetch     : define.log + runs/define_*.json -> {HERE}/")
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
        sh(f"{ssh} 'rm -f /root/DONE /root/define.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < cap:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; "
                                   f"tail -1 /root/define.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("run complete")
                break
        tags = " ".join(f"runs/define_{t}.json" for (t, *_rest) in SWEEP)
        fetch = (f"{ssh} 'cd {REMOTE} && tar czf - define.log {tags} 2>/dev/null' "
                 f"| tar xzf - -C {HERE}")
        sh(fetch)
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
