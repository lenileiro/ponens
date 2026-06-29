#!/usr/bin/env python3
"""Test the SCALE hypothesis for the broad-prior span PFN: a small/CPU model STALLED meta-training on the 7-family
mix (loss ~0.55, exact ~0). TabPFN's rationale says a BROAD prior needs SCALE -- big model + many synthetic
episodes + large support. Train thinking.span_pfn at scale on a RunPod GPU (synthetic data, no upload needed) and
see whether held-in/held-out in-context exact-span recovers.

Safety: DRY-RUN by default unless --go. Pod-side timeout + ALWAYS-terminate. Auth via env RUNPOD_API_KEY (never
hardcoded). No data upload (episodes generated on-pod).

  Dry run: python runpod/launch_span_pfn.py
  Launch : RUNPOD_API_KEY=... python runpod/launch_span_pfn.py --go
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
D, LAYERS, HEADS, BS, STEPS, KSUP = 256, 6, 8, 128, 30000, 8


def api(method, path, key, body=None):
    url = REST + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
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


def build_run_cmd():
    run = (f"python -u -m thinking.span_pfn --train --d {D} --layers {LAYERS} --heads {HEADS} "
           f"--bs {BS} --steps {STEPS} --k {KSUP} --save {REMOTE}/runs/span_pfn.pt")
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA A40")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=30)
    ap.add_argument("--name", default="span-pfn-scale")
    ap.add_argument("--max-minutes", type=int, default=60)
    ap.add_argument("--print-payload", action="store_true")
    ap.add_argument("--go", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY")
    if not key and args.go:
        sys.exit("ERROR: export RUNPOD_API_KEY first (do not hardcode it).")
    pub = os.path.expanduser("~/.ssh/id_rsa.pub")
    pubkey = open(pub).read().strip() if os.path.exists(pub) else ""

    cap = args.max_minutes * 60
    run = build_run_cmd()
    remote_cmd = (
        f"cd {REMOTE} && rm -f spfn.log /root/spfn.log && mkdir -p {REMOTE}/runs && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')} "
        f"2>&1 | tee /root/spfn.log; cp /root/spfn.log {REMOTE}/spfn.log 2>/dev/null; true")

    print("=== PLAN === SCALE test: broad-prior span PFN (7 families) on GPU")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (disk {args.disk}GB, name {args.name})")
    print(f"config    : d{D} L{LAYERS} H{HEADS} bs{BS} steps{STEPS} K{KSUP} (synthetic, no data upload)")
    print(f"fetch     : spfn.log + runs/span_pfn.pt -> {HERE}/")
    print(f"guard     : pod-side timeout {cap}s + ALWAYS-terminate; SSH-wait cap {args.max_minutes}m")
    print(f"cost EST  : up to ~{args.max_minutes/60:.1f} hr * ${HOURLY_USD}/hr = ~${args.max_minutes/60*HOURLY_USD:.2f}")
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
            status = p.get("desiredStatus"); ip = p.get("publicIp")
            port = (p.get("portMappings") or {}).get("22")
            print(f"  status={status} ip={ip} port={port}")
            if status == "RUNNING" and ip and port:
                break
            time.sleep(12)
        if not (ip and port):
            sys.exit("pod never exposed SSH within cap; terminating.")
        time.sleep(25)
        ssh = f"ssh -o StrictHostKeyChecking=no -p {port} root@{ip}"
        up = (f"COPYFILE_DISABLE=1 tar czf - --exclude './.venv*' --exclude './venv*' "
              f"--exclude '*/__pycache__' --exclude '*.zip' --exclude './data' --exclude '*.pt' --exclude '*.log' "
              f"--exclude './runs' --exclude './experiments' --exclude './tooling' --exclude './artifacts' "
              f"--exclude './.git' --exclude './kaggle_data' --exclude './aqua_data' --exclude '*.tgz' -C {HERE} . "
              f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(up)
        chk = subprocess.run(f"{ssh} 'test -s {REMOTE}/thinking/span_pfn.py && echo CODE_OK'",
                             shell=True, capture_output=True, text=True, timeout=120)
        if "CODE_OK" not in (chk.stdout or ""):
            raise RuntimeError("code upload failed (span_pfn.py missing on pod)")
        print("  code upload verified")
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
        sh(f"{ssh} 'rm -f /root/DONE /root/spfn.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < cap:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; tail -1 /root/spfn.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("run complete"); break
        sh(f"{ssh} 'cd {REMOTE} && tar czf - spfn.log runs/span_pfn.pt 2>/dev/null' | tar xzf - -C {HERE}")
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
