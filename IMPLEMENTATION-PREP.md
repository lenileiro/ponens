# Implementation Preparation
**Date:** 2026-06-06
**Purpose:** Ground-truth setup state and step-by-step prep to make experiments runnable. Companion to VALIDATED-PLAN.md, NEWER-TECHNIQUES.md, REASONING.md, FER-TEXT-RELATIONAL.md.

This is the "what do we actually run, and where" document. It records the verified local environment, the inspected repos, the framework decisions, and the exact setup steps for the first experiments.

---

## 1. Verified Local Environment (this machine)

| Item | Value | Implication |
|------|-------|-------------|
| Machine | Apple M1 Pro, 10 cores, **16 GB RAM** | Toy-scale only; memory is the binding constraint |
| Accelerator | Apple Metal 4 (**MPS**), **no NVIDIA/CUDA** | Heavy training is REMOTE (RunPod/Lambda per VALIDATED-PLAN §3) |
| OS | macOS (Darwin 25.4.0, arm64) | — |
| Python (system) | **3.14.3** (Homebrew) | ⚠️ Too new — most ML wheels (torch, jax) lag; DO NOT use directly |
| uv | 0.11.15 | Use uv to create pinned venvs |
| git | 2.51.1 | — |
| Free disk | ~59 GB on data volume | Fine for toy experiments + datasets; not for large checkpoints |

**Hard rule:** This box is for (a) generating datasets, (b) toy-scale model training (≤~50M params on MPS/CPU), (c) interpretability probes, and (d) authoring/orchestration. **Every real training run** (coding-LLM SFT, GRPO, the scaled FER arm) goes to rented GPUs per the validated plan.

**Python version rule:** Create a **pinned Python 3.11 or 3.12 venv per project** with `uv`. Python 3.14 will break torch/jax/transformers installs. Example: `uv venv --python 3.12`.

---

## 2. Repos Inspected (cloned to `/tmp/llm-repos/`)

Temp clones for inspection only — not committed. Re-clone into the project workspace when building.

### FER — `github.com/akarshkumar0101/fer` (the source technique)
- **Framework: JAX + Flax + Optax + EvoSax** (NOT PyTorch). `requirements.txt` is incomplete — it omits jax/flax/optax/evosax/numpy (only lists einops, jupyter, matplotlib, pandas, tqdm).
- Structure: `src/cppn.py` (Flax CPPN), `src/train_sgd.py` (SGD-fit a CPPN to a target image), `src/process_pb.py` (Picbreeder genome → layerized CPPN), `src/fer.ipynb` (main notebook), `picbreeder_genomes/`, `data/` (pre-computed, lets you regenerate visualizations without retraining).
- **Decision:** We **reuse the *method*** (per-neuron visualization, weight-sweep coherence test), **not the JAX code**. Our text experiment is built fresh in PyTorch. Run the FER notebook once (P0) to internalize the method; don't port it.

### CLUTRR — `github.com/facebookresearch/clutrr` (primary relational benchmark)
- **Pure-Python data *generator*** — pandas, `names`, tqdm, networkx, nltk. No GPU, no model. Generates relational-reasoning stories from kinship graphs.
- ⚠️ Deps pinned ancient (`pandas==0.23.4`, `names==0.3.0`) — won't install on modern Python. Need a **dedicated pinned venv (3.10/3.11)** with relaxed deps, or vendored fixes.
- Entry: `clutrr/main.py`, usage `python main.py --train_tasks 1.3 --test_tasks 1.3,1.4` (task `<id>.<relation_length>`). Task 1 = basic kinship (no noise), 2 = +supporting facts, 3 = +irrelevant facts.
- **Decision:** Use it to generate the train/test splits (train short chains k≤4, test k=5–10). This is our relational-reasoning data source.

### Still to fetch when building (not yet cloned)
- `dual-attention` (PyTorch package, Altabaa & Lafferty) — relational-attention architecture, Lever 1.
- EleutherAI `sae-auto-interp` — automated interpretability for the factored-ness probe.
- `R2E-Gym/R2E-Gym`, `agentica-org/rLLM`, `datacurve-ai/pier`, `datacurve-ai/deep-swe` — for the coding-LLM track (remote-GPU phase).

---

## 3. Framework Decisions

| Track | Framework | Where it runs |
|-------|-----------|---------------|
| FER text/relational experiment (our 2×2) | **PyTorch** + HF transformers + `dual-attention` | Local MPS (toy) → remote if scaled |
| Relational benchmark data | CLUTRR (pure Python) | Local |
| Interpretability probes (binding subspace, SAE, IRS) | PyTorch + `sae-auto-interp` + custom | Local |
| FER reference reproduction (P0 only) | JAX/Flax (upstream repo) | Local CPU |
| Coding-LLM SFT/RL (main project) | LLaMA-Factory / veRL / rLLM | **Remote GPU** (RunPod/Lambda) |
| Serving | SGLang (+ EAGLE-3, FP8) | Remote GPU |

**Why PyTorch for our experiment** (despite FER being JAX): the relational-attention architecture (`dual-attention`), HF tokenizers, and the mature interpretability tooling (SAEs, activation patching) are all PyTorch-native. We only need FER's *idea*, not its implementation.

---

## 4. Proposed Workspace Layout

```
/Users/leiro/workspace/llm/
├── VALIDATED-PLAN.md            # coding-LLM core plan
├── NEWER-TECHNIQUES.md          # Dr.GRPO/PRIME/SWE-TRACE/SGLang/EAGLE-3
├── REASONING.md                 # reasoning training + TTS
├── FER-TEXT-RELATIONAL.md       # the research spike design
├── IMPLEMENTATION-PREP.md       # this file
├── tooling/deepswe/             # existing DeepSWE harness
├── artifacts/                   # run logs, datasets, checkpoints (gitignored)
└── experiments/                 # NEW — experiment code
    └── fer_relational/
        ├── pyproject.toml       # pinned deps (uv, py3.12)
        ├── data/                # CLUTRR-generated splits
        ├── models/              # standard transformer + dual-attention variants
        ├── train.py            # train one 2×2 cell
        ├── probes/              # binding-subspace, IRS, SAE feature-splitting
        ├── eval_clutrr.py      # accuracy vs chain-length k
        └── README.md
```

---

## 5. Setup Steps (copy-paste, tailored to this machine)

### Step A — CLUTRR data-gen venv (pinned, isolated)
```bash
mkdir -p ~/workspace/llm/experiments/fer_relational && cd ~/workspace/llm/experiments/fer_relational
uv venv --python 3.11 .venv-clutrr
# CLUTRR's ancient pins need relaxing; install relaxed deps:
.venv-clutrr/bin/pip install "pandas>=1.5" names tqdm networkx nltk pyyaml sacremoses requests matplotlib
# clone clutrr into experiments and install editable:
git clone https://github.com/facebookresearch/clutrr.git
.venv-clutrr/bin/pip install -e ./clutrr
# generate splits: train short chains, test longer (the systematic-generalization axis)
cd clutrr/clutrr && ../../.venv-clutrr/bin/python main.py --train_tasks 1.2,1.3,1.4 --test_tasks 1.5,1.6,1.7,1.8,1.9,1.10
```
*If old pins still fight modern Python, vendor the generator (it's small, pure Python) and patch the 2 or 3 deprecated pandas calls.*

### Step B — Experiment venv (PyTorch + relational attention)
```bash
cd ~/workspace/llm/experiments/fer_relational
uv venv --python 3.12 .venv
.venv/bin/pip install torch torchvision  # MPS build ships in default macOS wheels
.venv/bin/pip install transformers datasets einops numpy matplotlib
.venv/bin/pip install dual-attention      # Altabaa & Lafferty relational-attention transformer
# sanity: confirm MPS is visible
.venv/bin/python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### Step C — FER reference reproduction (P0, JAX, optional but recommended)
```bash
cd /tmp/llm-repos/fer
uv venv --python 3.11 .venv-fer
.venv-fer/bin/pip install "jax[cpu]" flax optax evosax einops numpy matplotlib jupyterlab pandas tqdm
# run the notebook to see the skull weight-sweep contrast (the method we're porting)
.venv-fer/bin/jupyter lab src/fer.ipynb
```

---

## 6. First Experiment — Make the 2×2 Runnable (from FER-TEXT-RELATIONAL.md §6)

**Build order (each step gated on the previous):**

1. **P0 — Reproduce FER** (½–1 day): run `src/fer.ipynb`, confirm the evolved-vs-SGD weight-sweep contrast. Internalize the visualization + sweep method.
2. **P1 — Data** (½ day): generate CLUTRR splits (Step A). Verify train k≤4 / test k=5–10 split. Build a loader.
3. **P2 — Baseline** (2–3 days): train a small standard transformer (1–10M params) on CLUTRR on MPS; reproduce the **monotonic accuracy-drop as k grows**. Establish baseline.
4. **P3 — Probes** (3–5 days): implement the factored-ness scorers — binding-subspace rank/linearity (Feng & Steinhardt method), position-reliance, IRS, SAE feature-splitting. Validate on the baseline.
5. **P4 — The 2×2** (1–2 wks): add Dual Attention Transformer + triple-extraction auxiliary objective; run all 4 cells; produce the headline plot (factored-ness vs length-generalization).
6. **P5 — Causal test**: scramble/clamp the binding subspace; confirm generalization moves with factoring.

**Memory budget note:** at 16 GB, keep models ≤~10–20M params and batch sizes modest. If models need to grow, move P4/P5 to a rented single GPU (a few dollars/hr; see VALIDATED-PLAN §3.1 — RunPod L40S ~$0.80/hr is plenty for this).

---

## 7. Parallel Track — Coding-LLM Prep (remote, not local)

Independent of the FER spike. When ready to start the DeepSWE track (VALIDATED-PLAN §5, Phase 0):
1. Fill `tooling/deepswe/model_profile.env` (currently placeholder `your-model-id`) with a real model + provider key + backend (`--env modal/docker/daytona`).
2. `uv tool install datacurve-pier` and `git clone https://github.com/datacurve-ai/deep-swe`.
3. Run the 10-task smoke with `claude-sonnet-4-6` (expected ~32% pass@1 per leaderboard) to validate the harness.
4. This requires API credentials + a runtime backend — none configured yet. **Blocked on credentials.**

---

## 8. Open Decisions / Blockers

| Item | Status | Needs |
|------|--------|-------|
| GPU provider account (for any real training) | Not set up | User to pick RunPod/Lambda + fund |
| API keys (DeepSWE smoke) | Not set | ANTHROPIC_API_KEY or other provider |
| CLUTRR dep modernization | Known issue | Relax pins or vendor (small effort) |
| Which track first — FER spike vs DeepSWE | FER spike is cheaper/local-runnable; DeepSWE blocked on credentials | User steer (defaulting to FER spike since it's unblocked) |

---

## 9. Status Summary

- ✅ Environment profiled (M1 Pro, 16GB, MPS, no CUDA, Py3.14 → use pinned venvs)
- ✅ FER + CLUTRR repos cloned & inspected (`/tmp/llm-repos/`); frameworks identified (FER=JAX, CLUTRR=pure-Python gen)
- ✅ Framework decisions made (experiment = PyTorch + dual-attention; reuse FER method not code)
- ✅ Workspace layout + copy-paste setup steps written
- ⏭️ Next runnable step (unblocked): **P0 reproduce FER notebook + P1 generate CLUTRR data** — both local, no credentials needed
- ⛔ Blocked: anything needing rented GPUs or API keys (coding-LLM training, DeepSWE smoke)
