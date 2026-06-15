"""Local MPS fine-tune of the coding model, following FINETUNE.md + runpod/finetune_ar_local.py.

Warm-starts a pretrained multimodal coding checkpoint and continues training (SFT) on agentic /
tool-use data at a LOWER LR (the one change), then VERIFIES with the bash harness (objective
metric, not train loss). This is the Composer-style SFT phase done the repo's way: reuse the
existing `--multimodal-checkpoint` warm-start instead of training from scratch.

  PYTHONPATH=/Users/leiro/workspace/llm .venv/bin/python coding_agent/rl/finetune_coding_local.py \
      --base runs/m0_multimodal.pt --sft-manifest /tmp/devtools_sft/manifest.jsonl --steps 4000

Recipe (FINETUNE.md): load ckpt -> lower LR (1e-4 vs 3e-4) -> change ONE thing (SFT data) ->
checkpoint -> verify objective (harness pass rate), keep the best.
"""
import argparse, os, subprocess, sys, torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.join(REPO, ".venv/bin/python")


def model_arch(base_ckpt):
    """Read dim/layers/heads/max_len from the base checkpoint so the warm-started model matches
    shapes (load_state_dict needs identical shapes -- see FINETUNE.md 'Shape changes')."""
    cfg = torch.load(base_ckpt, map_location="cpu", weights_only=False).get("model_config", {})
    return (int(cfg.get("d", 768)), int(cfg.get("layers", 12)),
            int(cfg.get("heads", 12)), int(cfg.get("max_len", 256)),
            int(cfg.get("vocab_size", 32000)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="pretrained coding checkpoint to warm-start from")
    ap.add_argument("--sft-manifest", required=True, help="agentic/tool-use SFT manifest (the ONE change)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tune LR (lower than from-scratch 3e-4)")
    ap.add_argument("--out", default="runs/m0_coding_ft.pt")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    d, layers, heads, max_len, vocab = model_arch(args.base)
    print(f"warm-start arch from {args.base}: dim {d} / {layers}L / {heads}H / max_len {max_len} / vocab {vocab}")

    # ONE change = SFT data; warm-start via --multimodal-checkpoint; lower LR; objective profile manual.
    cmd = [PY, "-m", "thinking.multimodal",
           "--manifest", args.sft_manifest,
           "--multimodal-checkpoint", args.base,
           "--decode-objective", "causal", "--objective-profile", "manual",
           "--dim", str(d), "--layers", str(layers), "--heads", str(heads),
           "--max-len", str(max_len), "--max-vocab", str(vocab),
           "--lr", str(args.lr), "--steps", str(args.steps),
           "--device", args.device, "--log-every", "500",
           "--checkpoint", args.out, "--out", args.out.replace(".pt", ".json")]
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc != 0:
        sys.exit(f"fine-tune failed (rc {rc})")

    # VERIFY with the bash harness (objective metric, not train loss) -- FINETUNE.md step 5.
    print("\n=== objective verification: bash harness ===", flush=True)
    subprocess.run([PY, os.path.join(REPO, "coding_agent/harness/coding_agent.py"),
                    args.out, "--task", "all"], cwd=REPO)


if __name__ == "__main__":
    main()
