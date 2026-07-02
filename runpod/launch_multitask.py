#!/usr/bin/env python3
"""Train the MULTI-TASK non-LLM span reader (thinking/multitask.py) on a RunPod GPU: ONE reader on SQuAD (QA) +
Tweet Sentiment Extraction (sentiment-span) together, so a single shipped weight generalizes across the span-task
family. Local MPS is unusable for SQuAD's long sequences (~29 s/batch); CUDA does it in minutes. Eval reports BOTH
SQuAD dev F1 and TSE dev Jaccard from the one weight.

Safety: DRY-RUN by default unless --go. Pod-side timeout + ALWAYS-terminate in finally. Auth via env RUNPOD_API_KEY
(never hardcoded). SQuAD + GloVe fetched on-pod; the TSE csv (gitignored, normally excluded) is uploaded separately.

  Dry run: python runpod/launch_multitask.py
  Launch : RUNPOD_API_KEY=... python runpod/launch_multitask.py --go
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
SQUAD = "https://rajpurkar.github.io/SQuAD-explorer/dataset"
TSE_CSV = "kaggle_data/tweet_sent/tweet_dataset.csv"
SQUAD_N, EPOCHS, HIDDEN, DIM = 0, 5, 192, 100   # 0 = ALL SQuAD (~88k); more capacity to hold both tasks


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
    prep = ("pip install -q gensim 'numpy<2' || pip install -q gensim ; "
            "mkdir -p /tmp/squad && cd /tmp/squad && "
            f"(test -s train-v1.1.json || wget -q {SQUAD}/train-v1.1.json) && "
            f"(test -s dev-v1.1.json || wget -q {SQUAD}/dev-v1.1.json) && cd " + REMOTE)
    run = (f"python -u -m thinking.multitask --train --squad-n {SQUAD_N} --epochs {EPOCHS} "
           f"--hidden {HIDDEN} --dim {DIM} --save {REMOTE}/runs/mt.pt")
    return prep + " ; " + run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA A40")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="squad-multitask")
    ap.add_argument("--max-minutes", type=int, default=60)
    ap.add_argument("--print-payload", action="store_true")
    ap.add_argument("--go", action="store_true", help="actually create the pod (spends money)")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY")
    if not key and args.go:
        sys.exit("ERROR: export RUNPOD_API_KEY first (do not hardcode it).")
    if not os.path.exists(os.path.join(HERE, TSE_CSV)):
        sys.exit(f"ERROR: {TSE_CSV} not found -- download the Kaggle TSE data first.")
    pub = os.path.expanduser("~/.ssh/id_rsa.pub")
    pubkey = open(pub).read().strip() if os.path.exists(pub) else ""

    cap = args.max_minutes * 60
    run = build_run_cmd()
    remote_cmd = (
        f"cd {REMOTE} && rm -f mt.log /root/mt.log && mkdir -p {REMOTE}/runs && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')} "
        f"2>&1 | tee /root/mt.log; cp /root/mt.log {REMOTE}/mt.log 2>/dev/null; true")

    print("=== PLAN === ONE multi-task non-LLM reader (SQuAD QA + Tweet-Sentiment span) on GPU")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (disk {args.disk}GB, name {args.name})")
    print(f"config    : SQuAD {SQUAD_N} + all TSE, {EPOCHS} epochs, hidden {HIDDEN}, GloVe-{DIM}")
    print(f"deploy    : code -> pod:{REMOTE} (venv excluded); SQuAD+GloVe on-pod; TSE csv uploaded separately")
    print(f"fetch     : mt.log + runs/mt.pt -> {HERE}/")
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
              f"--exclude '*/__pycache__' --exclude './results_gpu' --exclude '*.zip' --exclude './data' "
              f"--exclude '*.pt' --exclude '*.log' --exclude './runs' --exclude './experiments' "
              f"--exclude './tooling' --exclude './artifacts' --exclude './.git' "
              f"--exclude './kaggle_data' --exclude './aqua_data' --exclude '*.tgz' -C {HERE} . "
              f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(up)
        # upload the TSE csv (kaggle_data is excluded from the main tar)
        sh(f"tar czf - -C {os.path.join(HERE, 'kaggle_data', 'tweet_sent')} tweet_dataset.csv "
           f"| {ssh} 'mkdir -p {REMOTE}/kaggle_data/tweet_sent && tar xzf - -C {REMOTE}/kaggle_data/tweet_sent'")
        chk = subprocess.run(f"{ssh} 'test -s {REMOTE}/thinking/multitask.py && "
                             f"test -s {REMOTE}/kaggle_data/tweet_sent/tweet_dataset.csv && echo CODE_OK'",
                             shell=True, capture_output=True, text=True, timeout=120)
        if "CODE_OK" not in (chk.stdout or ""):
            raise RuntimeError("upload incomplete (multitask.py or TSE csv missing on pod)")
        print("  code + TSE data upload verified")
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
        sh(f"{ssh} 'rm -f /root/DONE /root/mt.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < cap:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; tail -1 /root/mt.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("run complete"); break
        sh(f"{ssh} 'cd {REMOTE} && tar czf - mt.log runs/mt.pt 2>/dev/null' | tar xzf - -C {HERE}")
        print("results fetched ->", HERE)
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
