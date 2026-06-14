#!/usr/bin/env python3
"""Standalone GPU launcher for the AUDIO mastery track (sing / polyglot / pronounce / mimic).

Kept separate from launch_thinking.py to avoid colliding with parallel edits there. Self-contained
modules (sing, polyglot, audio) need no data upload; say-bank modules (pronounce, mimic) get their
pre-rendered banks uploaded (pods are Linux -> no macOS `say`). Deploys a PINNED commit via
git archive (REBAL2 lesson: never tar a live tree mid-edit), runs detached, fetches runs/, always
terminates.

  RUNPOD_API_KEY=... python runpod/launch_audio.py --go
  RUNPOD_API_KEY=... python runpod/launch_audio.py --job sing-sweep --go
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from shlex import quote

REST = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
REMOTE = "/workspace/fer_relational"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def sh(cmd):
    print("  $", cmd[:160])
    return subprocess.run(cmd, shell=True).returncode


def payload(args):
    PY = "python3 -u -m thinking"
    jobs = []
    if args.job in ("all", "sing-sweep"):
        # the open question: find vowel_w where BOTH pitch and vowel stay high
        # (CPU run: vowel_w 0.5 gave pitch 0.27 / vowel 1.00 -- too high; sweep down)
        for vw in ("0.05", "0.1", "0.2"):
            jobs.append(f"{PY}.sing --steps 16000 --vowel-w {vw} "
                        f"--out runs/a6_sing_vw{vw}.json --checkpoint runs/a6_sing_vw{vw}.pt")
    if args.job in ("all", "polyglot"):
        jobs.append(f"{PY}.polyglot --steps 20000 --out runs/a4b_polyglot_gpu.json")
    if args.job in ("all", "pronounce"):
        jobs.append(f"{PY}.pronounce --train --steps 20000 --out runs/a4_pronounce_gpu.json")
    if args.job in ("all", "mimic"):
        jobs.append(f"{PY}.mimic --train --steps 16000 --out runs/a5_mimic_gpu.json")
    if args.job in ("all", "vocoder"):
        jobs.append(f"{PY}.neuralvocoder --train --steps 30000 --out runs/neural_vocoder.json --checkpoint runs/neural_vocoder.pt")
    if args.job in ("all", "vocoder-gan"):
        jobs.append(f"{PY}.vocoder_gan --train --steps 60000 --out runs/vocoder_gan.json --checkpoint runs/vocoder_gan.pt")
    if args.job == "vocoder24":
        jobs.append(f"{PY}.vocoder24 --train --steps 100000 --out runs/vocoder24.json --checkpoint runs/vocoder24.pt")
    # non-fatal chaining: one job's failure must not kill the rest
    return " ; ".join(jobs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--job", default="all",
                    choices=("all", "sing-sweep", "polyglot", "pronounce", "mimic", "vocoder", "vocoder-gan", "vocoder24"))
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE")
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--ref", default="HEAD", help="git ref to deploy (pinned)")
    ap.add_argument("--max-minutes", type=int, default=180, dest="max_minutes")
    ap.add_argument("--name", default="fer-AUDIO")
    args = ap.parse_args(argv)

    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key and args.go:
        sys.exit("export RUNPOD_API_KEY first")
    pub = os.path.expanduser("~/.ssh/id_rsa.pub")
    pubkey = open(pub).read().strip() if os.path.exists(pub) else ""
    cap = args.max_minutes * 60
    run = payload(args)
    # say-banks needed by pronounce/mimic (pods have no macOS `say`)
    need_banks = args.job in ("all", "pronounce", "mimic", "vocoder", "vocoder-gan", "vocoder24")

    setup = "pip install -q numpy tokenizers pandas pyarrow"
    remote = (f"cd {REMOTE} && rm -f /root/thinking.log && "
              f"({setup} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
              f"timeout {cap}s bash -c {quote(f'cd {REMOTE} && ({run})')}) "
              f"2>&1 | tee /root/thinking.log; cp /root/thinking.log {REMOTE}/thinking.log 2>/dev/null; true")

    print("=== PLAN === audio GPU:", args.job)
    print("jobs:\n  " + run.replace(" ; ", "\n  "))
    print(f"deploy: git archive {args.ref} -> pod:{REMOTE}" + (" + say-banks" if need_banks else ""))
    if not args.go:
        print("\n[dry-run] re-run with --go")
        return

    body = {"name": args.name, "imageName": IMAGE, "gpuTypeIds": [args.gpu], "gpuCount": 1,
            "cloudType": args.cloud, "containerDiskInGb": args.disk, "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": pubkey}}
    print("\n=== creating pod ===")
    st, pod = api("POST", "/pods", key, body)
    if st not in (200, 201):
        sys.exit(f"create failed: HTTP {st} {pod}")
    pid = pod.get("id") or pod.get("podId")
    print("pod id:", pid)
    try:
        ip = port = None
        t0 = time.time()
        while time.time() - t0 < cap:
            st, p = api("GET", f"/pods/{pid}", key)
            ip = p.get("publicIp")
            port = (p.get("portMappings") or {}).get("22")
            print(f"  status={p.get('desiredStatus')} ip={ip} port={port}")
            if p.get("desiredStatus") == "RUNNING" and ip and port:
                break
            time.sleep(12)
        if not (ip and port):
            sys.exit("pod never exposed SSH; terminating.")
        time.sleep(25)
        ssh = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p {port} root@{ip}"
        # PINNED deploy
        sh(f"git -C {quote(HERE)} archive --format=tar.gz {quote(args.ref)} "
           f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        # crossmodal.py recovered to worktree (not in HEAD) -- ship it too
        if os.path.exists(os.path.join(HERE, "thinking/crossmodal.py")):
            sh(f"tar czf - -C {quote(HERE)} thinking/crossmodal.py | {ssh} 'tar --no-same-owner -xzf - -C {REMOTE}'")
        if need_banks:
            banks = ("data/speech24k",) if args.job == "vocoder24" else ("data/pronounce", "data/mimic", "data/speech16k")
            for bank in banks:
                if os.path.isdir(os.path.join(HERE, bank)):
                    sh(f"tar czf - -C {quote(HERE)} {bank} | {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        # verified detached run-script upload (5 retries), then nohup
        script = remote + "; touch /root/DONE\n"
        for _ in range(5):
            subprocess.run(f"{ssh} 'cat > /root/run.sh'", shell=True, input=script, text=True,
                           timeout=120)
            ok = subprocess.run(f"{ssh} 'test -s /root/run.sh && echo OK'", shell=True,
                                capture_output=True, text=True)
            if "OK" in (ok.stdout or ""):
                break
        sh(f"{ssh} 'rm -f /root/DONE && nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        print(f"\npod {pid} @ {ip}:{port} -- detached. poll: ssh -p {port} root@{ip} 'tail /root/thinking.log'")
        # poll to completion, then fetch
        while time.time() - t0 < cap:
            time.sleep(60)
            try:
                d = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE'", shell=True,
                                   capture_output=True, text=True, timeout=60)
                if "DONE" in (d.stdout or ""):
                    print("run DONE")
                    break
            except subprocess.TimeoutExpired:
                continue
        sh(f"{ssh} 'cd {REMOTE} && tar czf - runs thinking.log 2>/dev/null' | tar xzf - -C {quote(HERE)}")
        print("results fetched -> runs/")
    finally:
        print("=== terminating pod ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
