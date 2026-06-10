#!/usr/bin/env python3
"""Cost guard: list ALL RunPod pods and TERMINATE every one. Confirms each delete (HTTP 204).

  RUNPOD_API_KEY=... python runpod/terminate_all.py            # dry-run: just LIST pods (no delete)
  RUNPOD_API_KEY=... python runpod/terminate_all.py --go       # actually terminate every pod

Key is read from the environment at runtime only -- never stored.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually terminate (default: list only)")
    args = ap.parse_args()
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("ERROR: set RUNPOD_API_KEY in the environment (runtime only).")
    st, pods = api("GET", "/pods", key)
    if st != 200:
        sys.exit(f"list failed: HTTP {st} {pods}")
    pods = pods if isinstance(pods, list) else pods.get("pods", pods.get("data", []))
    if not pods:
        print("No pods found. Nothing running -> nothing to terminate. (0 GPUs.)")
        return
    print(f"{len(pods)} pod(s):")
    for p in pods:
        print(f"  - {p.get('id')}  {p.get('name','')}  status={p.get('desiredStatus', p.get('status',''))}  "
              f"gpu={p.get('machine',{}).get('gpuTypeId', p.get('gpuTypeId',''))}")
    if not args.go:
        print("\n(dry-run) re-run with --go to terminate all of the above.")
        return
    for p in pods:
        pid = p.get("id")
        code, _ = api("DELETE", f"/pods/{pid}", key)
        print(f"  delete {pid}: HTTP {code}" + ("  OK" if code in (200, 204) else "  !! check manually"))
    st2, after = api("GET", "/pods", key)
    rem = after if isinstance(after, list) else after.get("pods", after.get("data", []))
    print(f"\nremaining pods: {len(rem)}  ({'all terminated' if not rem else 'SOME REMAIN -- check console'})")


if __name__ == "__main__":
    main()
