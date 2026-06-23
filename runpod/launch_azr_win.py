#!/usr/bin/env python3
"""RunPod H100 runner for the COMBINED verified-synthesis solver (thinking/azr_win.py) at SCALE.

Trains the iterate-execute-residual + library + best-of-N solver with a BIG model (d>=512, many layers)
and MASSIVE batching (--bs in the hundreds -> thousands of frontier rows per forward), which is exactly the
regime where the forward becomes a large parallel matmul and an H100 pays off (CPU bench: d512/L8 B=1024 =
~3.8s/forward; H100 ~10-30ms). Pod-side `timeout` bounds the run and cleanup ALWAYS terminates the pod.
Defaults to DRY-RUN: nothing is created and no money is spent unless `--go` is passed.

Auth: export RUNPOD_API_KEY (never hardcoded).
  Dry-run (default): python runpod/launch_azr_win.py
  Launch (spends $): RUNPOD_API_KEY=... python runpod/launch_azr_win.py --go

Modeled on runpod/launch_reasoning.py (same REST endpoint, image, create/upload/detached-run/fetch/terminate).
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
RESULTS_REMOTE = "runs/azr_win.json"
SAVE_REMOTE = "runs/azr_win_solver.pt"
HOURLY_USD = 3.29


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
                payload = json.loads(txt) if txt.strip() else {}
            except json.JSONDecodeError:
                payload = {"error": txt}
            return r.status, payload
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}
    except urllib.error.URLError as e:
        return 0, {"error": f"network error: {getattr(e, 'reason', e)}"}
    except (TimeoutError, OSError) as e:
        return 0, {"error": f"error: {e}"}


def sh(cmd):
    print("  $", cmd)
    return subprocess.run(cmd, shell=True).returncode


def pod_addr(key, pid):
    """Look up an existing pod's SSH (ip, port) from the API."""
    st, p = api("GET", f"/pods/{pid}", key)
    if st != 200:
        return None, None
    return p.get("publicIp"), (p.get("portMappings") or {}).get("22")


def ssh_pull(ip, port, pairs, dest):
    """ROBUST per-file fetch over SSH (decoupled from launch/terminate, binary-safe via cat). pairs =
    list of (remote_abs_path, local_relpath). Returns the files actually retrieved (size>0)."""
    ssh = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p {port} root@{ip}"
    got = []
    for remote, rel in pairs:
        local = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        subprocess.run(f"{ssh} 'cat {shlex_quote(remote)} 2>/dev/null' > {shlex_quote(local)}",
                       shell=True)
        sz = os.path.getsize(local) if os.path.exists(local) else 0
        if sz > 0:
            got.append((rel, sz)); print(f"  fetched {rel}: {sz:,} bytes")
        else:
            if os.path.exists(local):
                os.remove(local)
            print(f"  (skip {rel}: not present yet)")
    return got


# remote artifacts to retrieve (live log lives at /root; results/model in the repo runs/ dir)
FETCH_PAIRS = [("/root/azr_win.log", "azr_win.log"),
               (f"{REMOTE}/{RESULTS_REMOTE}", RESULTS_REMOTE),
               (f"{REMOTE}/{SAVE_REMOTE}", SAVE_REMOTE)]


def build_run(args):
    """The exact `python -m thinking.azr_win ...` command run on the pod (device auto-selects cuda)."""
    grpo = f"--grpo --grpo-mode {args.grpo_mode} --G {args.G} --grpo-bs {args.grpo_bs}" if args.grpo \
        else "--no-grpo"
    return (
        f"python -m thinking.azr_win "
        f"--A {args.A} --L {args.L} --K {args.K} --concepts {args.concepts} --maxdepth {args.maxdepth} "
        f"--d {args.d} --layers {args.layers} --heads {args.heads} --bs {args.bs} --beam {args.beam} "
        f"--rounds {args.rounds} --max-macros {args.max_macros} --collect-n {args.collect_n} {grpo} "
        f"--seeds {args.seeds} --save {SAVE_REMOTE} --ckpt-every {args.ckpt_every} --out {RESULTS_REMOTE}"
    )


def main():
    ap = argparse.ArgumentParser()
    # SCALED task distribution (big enough to justify a big model) + BIG model + MASSIVE batch
    ap.add_argument("--A", type=int, default=12); ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--concepts", type=int, default=16)
    ap.add_argument("--maxdepth", type=int, default=8)
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--bs", type=int, default=512, help="tasks/round -> thousands of frontier rows/forward")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=4000); ap.add_argument("--max-macros", type=int, default=48)
    ap.add_argument("--collect-n", type=int, default=8); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="checkpoint the model every N rounds -> always fetchable, survives timeout")
    ap.add_argument("--grpo", dest="grpo", action="store_true", default=False,
                    help="add the GRPO term (default OFF: bake-off showed SFT+best-of-N is the winner)")
    ap.add_argument("--grpo-mode", choices=["process", "outcome"], default="process")
    ap.add_argument("--G", type=int, default=6); ap.add_argument("--grpo-bs", type=int, default=64)
    # cost-estimate knob (rough)
    ap.add_argument("--rounds-per-sec", type=float, default=3.0,
                    help="rough H100 rounds/sec guess for the cost ESTIMATE")
    # pod knobs
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", default="SECURE", choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--name", default="azr-win")
    ap.add_argument("--max-minutes", type=int, default=120)
    ap.add_argument("--ref", default="HEAD",
                    help="git ref to ship (pinned committed tree). Empty to ship working dir.")
    ap.add_argument("--print-payload", action="store_true")
    ap.add_argument("--go", action="store_true", help="actually create the pod (spends money)")
    ap.add_argument("--fetch", default=None, metavar="POD_ID",
                    help="SSH into an existing pod and pull results (log/json/solver) -> exit. No terminate.")
    ap.add_argument("--terminate", default=None, metavar="POD_ID",
                    help="delete a pod by id -> exit (cost cleanup).")
    args = ap.parse_args()

    assert args.d % args.heads == 0, f"--d ({args.d}) must be divisible by --heads ({args.heads})"

    key = os.environ.get("RUNPOD_API_KEY")
    if (args.go or args.fetch or args.terminate) and not key:
        sys.exit("ERROR: export RUNPOD_API_KEY first (do not hardcode it).")

    if args.terminate:
        st, _ = api("DELETE", f"/pods/{args.terminate}", key)
        print(f"terminate {args.terminate}: HTTP {st}")
        return
    if args.fetch:
        ip, port = pod_addr(key, args.fetch)
        if not (ip and port):
            sys.exit(f"pod {args.fetch} has no SSH endpoint (gone or not ready)")
        print(f"fetching from pod {args.fetch} ({ip}:{port}) -> {HERE}")
        got = ssh_pull(ip, port, FETCH_PAIRS, HERE)
        print(f"done: {len(got)} file(s) fetched (pod NOT terminated; use --terminate {args.fetch})")
        return
    pub = os.path.expanduser("~/.ssh/id_rsa.pub")
    pubkey = open(pub).read().strip() if os.path.exists(pub) else ""

    body = {"name": args.name, "imageName": IMAGE, "gpuTypeIds": [args.gpu], "gpuCount": 1,
            "cloudType": args.cloud, "containerDiskInGb": args.disk, "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": pubkey}}
    cap = args.max_minutes * 60
    run = build_run(args)
    setup = f"WORKDIR={REMOTE} bash runpod/setup.sh"
    remote_cmd = (
        f"cd {REMOTE} && rm -f azr_win.log /root/azr_win.log && "
        f"({setup} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"mkdir -p {REMOTE}/runs && "
        f"timeout {cap}s bash -c {shlex_quote(f'cd {REMOTE} && ({run})')}) "
        f"2>&1 | tee /root/azr_win.log; "
        f"cp /root/azr_win.log {REMOTE}/azr_win.log 2>/dev/null; true")

    est_hr = (args.rounds / max(0.1, args.rounds_per_sec)) / 3600.0
    print("=== PLAN === combined verified-synthesis solver (thinking/azr_win.py) on H100")
    print(f"gpu/cloud : {args.gpu} / {args.cloud}  (disk {args.disk}GB, name {args.name})")
    print(f"task dist : A={args.A} L={args.L} K={args.K} concepts={args.concepts} maxdepth={args.maxdepth}")
    print(f"model     : d={args.d} layers={args.layers} heads={args.heads}  (BIG -> matmul feeds the GPU)")
    print(f"batch     : bs={args.bs} tasks/round, beam={args.beam}  (~{args.bs * args.beam} frontier rows/fwd)")
    print(f"train     : rounds={args.rounds} collect-n={args.collect_n} grpo={args.grpo} seeds={args.seeds}")
    print(f"sync up   : {HERE}/ -> pod:{REMOTE}  (ref={args.ref or 'working-dir'})")
    print(f"fetch     : azr_win.log + {RESULTS_REMOTE} + {SAVE_REMOTE} -> {HERE}/")
    print(f"guard     : pod-side timeout {cap}s + ALWAYS-terminate; SSH-wait cap {args.max_minutes}m")
    print(f"cost EST  : ~{est_hr:.2f} hr * ${HOURLY_USD}/hr = ~${est_hr * HOURLY_USD:.2f}  "
          f"[ESTIMATE ONLY: assumes {args.rounds_per_sec:.1f} rounds/sec]")
    print("\n=== EXACT REMOTE COMMAND (under pod-side timeout) ===")
    print(f"  {run}")
    if args.print_payload:
        print("\n=== POD PAYLOAD ===\n" + remote_cmd)
    if not args.go:
        print("\n[dry-run] nothing created. Re-run with --go to launch (spends money).")
        return

    print("\n=== creating pod ===")
    st, pod = api("POST", "/pods", key, body)
    if st not in (200, 201):
        sys.exit(f"create failed: HTTP {st} {pod}")
    pid = pod.get("id") or pod.get("podId")
    print("pod id:", pid)
    t0 = time.time()
    try:
        ip = port = None
        while time.time() - t0 < args.max_minutes * 60:
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
        if args.ref:
            up = (f"git -C {shlex_quote(HERE)} archive --format=tar.gz {shlex_quote(args.ref)} "
                  f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        else:
            up = (f"COPYFILE_DISABLE=1 tar czf - --exclude './.venv*' --exclude '*/__pycache__' "
                  f"--exclude '*.pt' --exclude '*.log' --exclude './runs' --exclude './.git' "
                  f"--exclude './experiments' -C {HERE} . "
                  f"| {ssh} 'mkdir -p {REMOTE} && tar --no-same-owner -xzf - -C {REMOTE}'")
        sh(up)
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
        sh(f"{ssh} 'rm -f /root/DONE /root/azr_win.log /root/launch.out && "
           f"nohup bash /root/run.sh > /root/launch.out 2>&1 & echo detached'")
        t1 = time.time()
        while time.time() - t1 < args.max_minutes * 60:
            time.sleep(60)
            try:
                r = subprocess.run(f"{ssh} 'test -f /root/DONE && echo DONE; "
                                   f"tail -1 /root/azr_win.log 2>/dev/null'",
                                   shell=True, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                continue
            line = (r.stdout or "").strip().splitlines()
            if line:
                print(" ", line[-1][:110])
            if "DONE" in (r.stdout or ""):
                print("run complete")
                break
        got = ssh_pull(ip, port, FETCH_PAIRS, HERE)        # robust per-file pull (binary-safe, no-fail)
        print(f"results fetched -> {HERE} ({len(got)} file(s))")
    finally:
        print("=== terminating pod (cost guard) ===")
        st, _ = api("DELETE", f"/pods/{pid}", key)
        print("delete HTTP", st)


if __name__ == "__main__":
    main()
