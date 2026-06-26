#!/usr/bin/env python3
"""Scale-confirm the CAPSTONE (thinking/langcap.py) on a RunPod H100: one model that learns a variable-grammar
invented language and is robust to BOTH walls at once -- an unseen word order AND unbounded depth. On tiny d128
CPU the four corners were trained 1.000 / unseen-order 0.997 / deep(d12) 0.820 / BOTH(d12) 0.817. Question for
the GPU: does scale lift the DEEP corner (length-gen of the combined task) toward ~1.0 and hold at larger test
depths?

Each sweep config writes runs/langcap_<tag>.json with the four corners (deep test depth pushed per config).

Safety: DRY-RUN by default (no spend) unless --go. Pod-side `timeout` bounds the run; pod ALWAYS terminated in
`finally`. Auth via env RUNPOD_API_KEY (NEVER hardcoded). Upload excludes venv/ (the 30k-file dir that made
earlier uploads take ~40min) -> seconds.

  Dry run (free): python runpod/launch_langcap.py
  Launch ($)    : RUNPOD_API_KEY=... python runpod/launch_langcap.py --go
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

# Scaling sweep: (tag, d, heads, steps, batch, deep). head dim (d//heads) must be EVEN for RoPE-free Block (any ok).
SWEEP = [
    ("s", 128, 4, 25000, 64, 20),     # base scale, deeper test (d20)
    ("m", 256, 8, 50000, 64, 24),     # +width +steps, d24
    ("l", 384, 8, 80000, 96, 28),     # +more, d28
]


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


def build_sweep_cmd():
    lines = []
    for (tag, d, h, steps, batch, deep) in SWEEP:
        lines.append(
            f"echo '=== CONFIG {tag}: d{d}/h{h} steps{steps} batch{batch} deep{deep} ==='; "
            f"python -m thinking.langcap --steps {steps} --d {d} --heads {h} --bs {batch} --deep {deep} "
            f"--device cuda --out runs/langcap_{tag}.json || true")
    return " ; ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="langcap-scale")
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
        f"cd {REMOTE} && rm -f langcap.log /root/lc.log && "
        f"(({preflight}) && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"mkdir -p {REMOTE}/runs && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')}) "
        f"2>&1 | tee /root/lc.log; "
        f"cp /root/lc.log {REMOTE}/langcap.log 2>/dev/null; true")

    print("=== PLAN === langcap capstone scale-confirm (both walls at once) on H100")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (disk {args.disk}GB, name {args.name})")
    for (tag, d, h, steps, batch, deep) in SWEEP:
        print(f"   [{tag}] d{d}/h{h} steps{steps} batch{batch} deep-test-depth{deep}")
    print(f"deploy    : working-dir -> pod:{REMOTE} (venv/ excluded); synthetic data, no upload")
    print(f"fetch     : langcap.log + runs/langcap_*.json -> {HERE}/")
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
              f"--exclude '*/__pycache__' --exclude './results_gpu' --exclude '*.zip' --exclude './data' "
              f"--exclude '*.pt' --exclude '*.log' --exclude './runs' --exclude './experiments' "
              f"--exclude './tooling' --exclude './artifacts' --exclude './.git' "
              f"--exclude './kaggle_data' --exclude './aqua_data' --exclude '*.tgz' -C {HERE} . "
              f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(up)
        chk = subprocess.run(f"{ssh} 'test -s {REMOTE}/thinking/langcap.py && echo CODE_OK'",
                             shell=True, capture_output=True, text=True, timeout=120)
        if "CODE_OK" not in (chk.stdout or ""):
            raise RuntimeError("code upload failed (langcap.py missing on pod)")
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
            raise RuntimeError("run.sh upload failed after 5 attempts")
        sh(f"{ssh} 'rm -f /root/DONE /root/lc.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < cap:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; tail -1 /root/lc.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("run complete")
                break
        fetch = (f"{ssh} 'cd {REMOTE} && tar czf - langcap.log runs/langcap_s.json runs/langcap_m.json "
                 f"runs/langcap_l.json 2>/dev/null' | tar xzf - -C {HERE}")
        sh(fetch)
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
