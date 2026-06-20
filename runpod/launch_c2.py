#!/usr/bin/env python3
"""Launch the C2 SCALING PROBE on a RunPod H100: train from-scratch LMs of increasing scale on the
contamination-split real-English corpus and GATE each on the C2-style held-out reading eval
(thinking/c2_eval.py). The question this answers: does SCALE (model size + data + steps) move
discourse cohesion/coherence off chance (~0.25), or does from-scratch hit the memorization wall?

Stage 1 of thinking/C2_ROADMAP.md (path B: crack-from-scratch + scale), gated on c2_eval — NOT on
train loss (the old big-corpus run memorized: train-loss 0.002 / held-out 7.4%, no generalization gate).

Safety: DRY-RUN by default (nothing created, no spend) unless --go. Pod-side `timeout` bounds the run
and the pod is ALWAYS terminated in `finally`. Auth via env RUNPOD_API_KEY (NEVER hardcoded).
Code is shipped from the WORKING DIR (uncommitted c2_eval edits) and the gitignored data/ is uploaded
explicitly.

  Dry run (free): python runpod/launch_c2.py
  Launch ($)    : RUNPOD_API_KEY=... python runpod/launch_c2.py --go
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


# Scaling sweep: (tag, d, layers, heads, steps, batch, seq_len, extra_data). max_len fixed below.
# head dim (d//heads) must be EVEN for RoPE.  Eval items always = cosmopedia held-out (disjoint).
SWEEP = [
    ("s", 256, 4, 8, 6000, 64, 128, False),    # ~reproduce the CPU baseline, on GPU (sanity anchor)
    ("m", 512, 8, 8, 12000, 128, 192, True),   # +depth/width +data
    ("l", 768, 12, 12, 16000, 128, 256, True), # +more
]
MAX_LEN = 384  # scored-sequence cap for the discourse eval items (>= largest seq_len + option room)


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
    """Shell snippet: run each sweep config, each writing its own runs/c2_<tag>.json. Uses ';' so a
    single config failing does not abort the rest; '|| true' keeps the pipeline alive for the tee."""
    lines = []
    for (tag, d, L, h, steps, batch, seq, extra) in SWEEP:
        extra_flag = " --extra-data" if extra else ""
        lines.append(
            f"echo '=== CONFIG {tag}: d{d}/L{L}/h{h} steps{steps} batch{batch} seq{seq}"
            f"{' +extra' if extra else ''} ==='; "
            f"python -m thinking.c2_eval --steps {steps} --d {d} --layers {L} --heads {h} "
            f"--batch {batch} --seq-len {seq} --max-len {MAX_LEN} --device cuda --lr 3e-4 "
            f"--n-each 80 --vocab-cap 16000 --log-every {max(1, steps // 10)}"
            f"{extra_flag} --out runs/c2_{tag}.json || true")
    return " ; ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="c2-scaling-probe")
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
    # c2_eval only needs torch + numpy, both in the base runpod/pytorch image -> skip setup.sh
    # (its venv + cu124 wheels + wordnet steps are unneeded here and add failure surface). Just
    # preflight the imports (install numpy if somehow absent) so a missing dep fails loudly.
    preflight = ("python -c 'import torch,numpy,sys; "
                 "print(\"deps ok torch\",torch.__version__,\"cuda\",torch.cuda.is_available())' "
                 "|| pip install -q numpy")
    remote_cmd = (
        f"cd {REMOTE} && rm -f c2.log /root/c2.log && "
        f"(({preflight}) && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"mkdir -p {REMOTE}/runs && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')}) "
        f"2>&1 | tee /root/c2.log; "
        f"cp /root/c2.log {REMOTE}/c2.log 2>/dev/null; true")

    print("=== PLAN === C2 scaling probe (thinking/c2_eval.py) on H100, gated on held-out reading eval")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (disk {args.disk}GB, name {args.name})")
    print(f"sweep     : {len(SWEEP)} configs")
    for (tag, d, L, h, steps, batch, seq, extra) in SWEEP:
        print(f"   [{tag}] d{d}/L{L}/h{h} steps{steps} batch{batch} seq{seq}"
              f"{' +extra-data' if extra else ''}")
    print(f"deploy    : working-dir -> pod:{REMOTE} (uncommitted ok) + explicit data/ upload")
    print(f"fetch     : c2.log + runs/c2_*.json -> {HERE}/")
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
        # ship code from the WORKING DIR (carries uncommitted c2_eval edits); data/ is gitignored
        # and excluded here, so it is uploaded separately below.
        up = (f"COPYFILE_DISABLE=1 tar czf - --exclude './.venv*' --exclude '*/__pycache__' "
              f"--exclude './results_gpu' --exclude '*.zip' --exclude './data' --exclude '*.pt' "
              f"--exclude '*.log' --exclude './runs' --exclude './experiments' "
              f"--exclude './tooling' --exclude './artifacts' --exclude './.git' "
              f"--exclude '*.tgz' -C {HERE} . "
              f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(up)
        # explicit data upload (gitignored, excluded above): only the files the sweep reads
        data_up = (f"COPYFILE_DISABLE=1 tar czf - -C {HERE} "
                   f"data/cosmopedia_6mb.txt data/cosmopedia_4mb.txt "
                   f"data/tinystories_8mb.txt data/tinystories_4mb.txt "
                   f"| {ssh} 'mkdir -p {REMOTE}/data && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(data_up)
        # verify the corpus landed before spending GPU time
        chk = subprocess.run(f"{ssh} 'test -s {REMOTE}/data/cosmopedia_6mb.txt && echo DATA_OK'",
                             shell=True, capture_output=True, text=True, timeout=120)
        if "DATA_OK" not in (chk.stdout or ""):
            raise RuntimeError("data upload failed (cosmopedia missing on pod)")
        print("  data upload verified")
        # detached run + short-poll (a dropped SSH pipe must not take the cost guard with it)
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
        sh(f"{ssh} 'rm -f /root/DONE /root/c2.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < cap:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; "
                                   f"tail -1 /root/c2.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("run complete")
                break
        fetch = (f"{ssh} 'cd {REMOTE} && tar czf - c2.log runs/c2_s.json runs/c2_m.json "
                 f"runs/c2_l.json 2>/dev/null' | tar xzf - -C {HERE}")
        sh(fetch)
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
