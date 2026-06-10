# Coding LLM — Deep-Research Validated Plan
**Synthesized:** 2026-06-06  
**Status:** Fully research-validated (4/4 research agents complete)

---

## Executive Summary

Build a coding-agent LLM that can compete on DeepSWE (the strongest current contamination-resistant benchmark). The proven open playbook already exists — we do not need to invent the architecture. Our path:

1. **Week 1–2:** Get DeepSWE infra live with an external model to validate the harness.
2. **Week 3–6:** Fine-tune Qwen3-32B on R2E-Gym with GRPO++ (the exact recipe used by DeepSWE-Preview to reach 42.2% pass@1 / 59% with TTS — fully open source).
3. **Week 7–12:** RL iteration, test-time scaling, and verifier training.
4. **Ongoing:** DeepSWE gate as the single source of truth.

Estimated compute for the RL training run: **~$37K** (64 H100s × 6 days at spot rates). A minimal 32B continued-pretraining pilot costs **~$1,400** (8 H100s × 3.6 days).

---

## 1. Benchmark Landscape (Validated, June 2026)

### 1.1 DeepSWE — Use This as Primary Target

DeepSWE is the most reliable benchmark for our purposes:
- **113 tasks** across Python, TypeScript, Go, JavaScript, Rust in 91 active OSS repos
- Written from scratch — no git-commit leakage possible (unlike SWE-bench Pro where Claude Opus cheated by reading `.git` history)
- Verification: program-based tests, 0.3% false positive rate (vs SWE-bench Pro's 8.5%)
- Tasks average 668 lines of changes across 7 files — ~5.5× harder than SWE-bench Pro

**Current leaderboard (May 26, 2026):**

| Rank | Model | Pass@1 |
|------|-------|--------|
| 1 | GPT-5.5 | **70%** |
| 2 | GPT-5.4 | 56% |
| 3 | Claude Opus 4.7 | 54% |
| 4 | **Claude Sonnet 4.6** (our current API) | **32%** |
| 5 | Gemini 3.5 Flash | 28% |
| 6 | GPT-5.4-mini | 24% |
| 11 | DeepSeek-V4-Pro | 8% |

All models run through the same scaffold: **mini-swe-agent** via **pier**. The scores isolate model capability, not scaffold differences.

**Our milestone gates:**
- Gate 1: >30% pass@1 on 10-task smoke (3 sustained runs) — proves harness works and we can reproduce
- Gate 2: >42% pass@1 on full 113 tasks — beats DeepSWE-Preview (the open-weight SOTA recipe)
- Gate 3: >54% — beats Claude Opus 4.7 (our strongest current API model)

### 1.2 SWE-bench Status (Use for Comparison Only, Not Primary)

- **SWE-bench Verified is saturated**: top score 93.9% (Claude Mythos Preview), dense cluster of 15+ models at 76–88%. Contamination likely. The `.git` loophole (reading gold solution from container) inflated Claude scores.
- **SWE-bench Pro is better**: 1,865 tasks, private repos, top score 45.9% (Claude Opus 4.5). More reliable but still less rigorous than DeepSWE.
- **Decision**: Use DeepSWE as our primary gate. Run SWE-bench Pro periodically for cross-validation.

### 1.3 LiveCodeBench (for Competitive Programming Track)

Top scores (v6, June 2026): Qwen3.7 Max at 91.6%, Kimi K2.6 at 89.6%. This benchmark is less relevant for our SWE-agent focus but useful for measuring raw code generation quality.

---

## 2. The Proven Open-Weight SWE-Agent Recipe

The following recipe is **fully open source** and was demonstrated by DeepSWE-Preview (Together AI + Agentica, July 2025):

### 2.1 DeepSWE-Preview Blueprint (Reference Implementation)

| Component | Choice | Why |
|-----------|--------|-----|
| Base model | Qwen3-32B | Extended thinking mode; strong coding baseline |
| Training data | R2E-Gym (4,500-task subset) | Procedurally generated from real commits; open source |
| RL algorithm | GRPO++ | No KL, no entropy, Clip High, leave-one-out advantage |
| Framework | rLLM (Agentica) | Open source; built for multi-turn agent RL |
| Compute | 64 H100 GPUs × 6 days | ~$37K at spot prices |
| Agent scaffold | mini-swe-agent | Standard for DeepSWE; 100 lines Python, single bash tool |

**Results:** 42.2% pass@1 on SWE-bench Verified (avg over 16 runs); 59% with hybrid test-time scaling; 71% pass@16 (oracle).

All assets are public:
- Weights: `agentica-org/DeepSWE-Preview` (Hugging Face)
- Framework: `agentica-org/rLLM` (GitHub)
- Dataset: `R2E-Gym/R2E-Gym` (GitHub + Hugging Face)
- Verifier: `DeepSWE-Verifier` (Hugging Face)

### 2.2 Supporting RL Evidence (Validates the Approach)

| System | Base Model | Algorithm | SWE-bench Verified | Compute |
|--------|-----------|-----------|-------------------|---------|
| SWE-RL (Meta, NeurIPS 2025) | Llama3-70B | GRPO (rule-based reward) | **41.0%** | Not disclosed |
| DeepSWE-Preview | Qwen3-32B | GRPO++ | **42.2% / 59% TTS** | 64 H100 × 6 days |
| Multi-turn RL (Yandex, Aug 2025) | Qwen2.5-72B | DAPO | **39.0%** | — |
| OpenHands LM 32B | Qwen2.5-Coder-32B | RL (SWE-Gym) | **37.2%** | — |
| Devstral Small (Mistral + AllHands) | 24B | RL | **46.6%** | — |
| DeepCoder-14B (Together) | DSR1-Distill-14B | GRPO+ | 60.6% LiveCodeBench | 32 H100 × 2.5 wks |
| Qwen3-Coder-480B | 480B MoE | Code RL + Agent RL | **67.0% / 69.6% @ 500 turns** | — |

**Verdict**: GRPO-family RL on executable coding environments is the standard technique. It works across model sizes (14B–480B). The recipe is proven and open.

### 2.3 R2E-Gym — Our Training Environment

R2E-Gym (UC Berkeley, COLM 2025, arXiv 2504.07164):
- **8,700 tasks** across 13 OSS repositories
- Generated procedurally via SWE-GEN (back-translation from commits — no human annotation needed)
- Each task has: Docker environment, unit tests, NL description
- Fully public: `R2E-Gym/R2E-Gym` on GitHub and Hugging Face
- Used directly by DeepSWE-Preview for training

This is our primary training environment. No need to build from scratch.

### 2.4 Test-Time Scaling (Critical for Top Scores)

Adding test-time compute consistently adds 10–20 absolute percentage points:

| System | Base Pass@1 | With TTS | Method |
|--------|------------|----------|--------|
| DeepSWE-Preview | 42.2% | **59.0%** | DeepSWE-Verifier, 16 rollouts |
| R2E-Gym agent | 34.4% | **51.0%** | Hybrid execution + execution-free verifier |
| OpenHands + critic | 60.6% | **66.4%** | TD-learned critic, N=5 rollouts |
| SWE-RM (Qwen3-Coder-Max) | 67.0% | **74.6%** | Execution-free 30B MoE verifier |

TTS is now table stakes. We need a verifier (either execution-based or the public DeepSWE-Verifier as starting point).

---

## 3. Hardware and Compute Budget

### 3.1 GPU Pricing (June 2026)

| GPU | Provider | Spot/hr | On-demand/hr |
|-----|----------|---------|--------------|
| H100 80GB SXM | RunPod | **$1.99** | $3.29 |
| H100 80GB SXM | Lambda Labs | N/A | $3.99 |
| H100 80GB SXM | CoreWeave | $2.46 | $6.16 |
| H100 80GB SXM | AWS p5 | ~$2.50 | ~$3.90 |
| H200 SXM | GMI Cloud | N/A | **$2.60** |
| B200 | RunPod | N/A | $5.89 |

**Recommended for small team:** RunPod community cloud for prototyping; Lambda Labs or CoreWeave reserved cluster for multi-day training runs.

### 3.2 Compute Budget for Our Workload

| Phase | Description | Cluster | Time | Spot Cost |
|-------|------------|---------|------|-----------|
| Phase 0: Harness validation | DeepSWE smoke with external model | 0 GPUs (API calls) | 1 week | ~$100–500 (API) |
| Phase 1: SFT pilot | 50B token continued pretraining on 7B | 8×H100 | 3.6 days | **~$1,400** |
| Phase 2: RL training | GRPO++ on 32B, 4,500 R2E-Gym tasks | 64×H100 | 6 days | **~$37,000** |
| Phase 3: Verifier training | Train execution-free verifier (30B MoE) | 16×H100 | ~5 days | **~$9,500** |
| Phase 4: Extended RL run | More tasks, more steps, larger model | 64×H100 | 14 days | **~$86,000** |
| **MVP total (Phases 0–3)** | | | ~3 weeks | **~$48,000** |

For full pretraining from scratch (7B, 1T tokens): 16×H100, ~77 days, ~$147K. **Not recommended as first step** — continued pretraining on an open-weight base is 100× more cost-effective for getting to a deployable result.

### 3.3 Chinchilla Reference

- Chinchilla-optimal tokens for 7B model: **140B tokens**
- 2026 production models train 10–100× past Chinchilla for inference efficiency (Llama-3-8B used 15T tokens)
- For us: start with continued pretraining (50B tokens) before committing to full pretraining

---

## 4. Base Model Recommendation (Fully Validated)

### 4.1 For SWE-Agent RL Training (32B scale — primary track)

**Primary recommendation: Qwen3-32B**

| Model | Used by | SWE-bench Verified | Notes |
|-------|---------|-------------------|-------|
| **Qwen3-32B** | DeepSWE-Preview | 42.2% (after GRPO++) | Extended thinking; strongest 32B for agent RL; Apache 2.0 |
| Qwen2.5-Coder-32B | OpenHands LM, Devstral, Skywork-SWE | 37–47% (after RL) | 5.5T coding tokens; most community SFT recipes |
| Devstral Small 2 (24B) | Mistral + AllHands | **68.0%** | Apache 2.0; 68% SWE-bench from a 24B model — remarkable |
| Devstral 2 (123B) | Mistral | 72.2% | Modified MIT; too large for budget-constrained training |

**32B is the sweet spot** for our compute budget:
- Inference: fits on 2×H100 80GB with 4-bit quantization (AWQ)
- Training: fits on 4–8×H100 for full-precision GRPO
- Has the most open training recipes (DeepSWE-Preview, OpenHands, Skywork-SWE all target this scale)

**Strong alternative: Devstral Small 2 (24B, Apache 2.0).** 68% SWE-bench Verified from a 24B dense model is remarkable — it was co-developed specifically for agentic coding (Mistral + All Hands AI) and is optimized for OpenHands scaffold. Worth running alongside Qwen3-32B.

### 4.2 For Smaller Pilots (7B–14B scale)

| Model | HF ID | Pretraining | Context | Notes |
|-------|-------|-------------|---------|-------|
| **Qwen3-8B-Base** | `Qwen/Qwen3-8B` | 36T tokens (inc. heavy coding) | 32K/131K | Apache 2.0; best fresh base for 8B |
| **Qwen2.5-Coder-14B** | `Qwen/Qwen2.5-Coder-14B` | 5.5T coding tokens | 131K | Most battle-tested coding SFT base |

For a budget-conscious pilot: Qwen2.5-Coder-14B on 8×H100 for SFT, then GRPO. Expect ~35–40% SWE-bench Verified with good RL training.

### 4.3 Models to Skip

- **Llama 4 Scout/Maverick**: MoE architecture complicates continued pretraining; not optimized for coding (LiveCodeBench: 32.8%/43.4% vs Qwen3 at 87%+)
- **Qwen3-Coder-480B**: 480B MoE total, 35B active — too large for budget-constrained fine-tuning
- **DeepSeek-V3 (671B/37B active)**: Excellent capability but logistically heavy for training; better as evaluation reference

---

## 5. Training Pipeline (Concrete Steps)

### Phase 0: Harness Validation (Week 1–2)

**Goal:** Prove we can run DeepSWE reliably, understand failure modes, establish baseline.

```bash
# 1. Install pier and clone DeepSWE
uv tool install datacurve-pier
git clone https://github.com/datacurve-ai/deep-swe

# 2. Configure profile with Claude Sonnet 4.6 (our current API)
cp tooling/deepswe/model_profile_template.env tooling/deepswe/model_profile.env
# Set: DEEPSWE_MODEL="claude-sonnet-4-6"
# Set: DEEPSWE_AGENT="mini-swe-agent"  
# Set: ANTHROPIC_API_KEY="..."
# Set: DEEPSWE_PIER_ARGS="--env modal --max-retries 2 --ak max_steps=96 --ak temperature=0.2"

# 3. Run 10-task smoke (3 seeds)
DEEPSWE_N_TASKS=10 DEEPSWE_SAMPLE_SEED=0 ./tooling/deepswe/run_deepswe.sh \
  --tasks-dir /path/to/deep-swe/tasks
```

**Expected result:** ~32% (matching leaderboard for claude-sonnet-4-6). If we see significantly lower, investigate harness config before proceeding.

**Gate to advance:** 3 reproducible runs at >28% pass@1 on 10-task smoke.

### Phase 1: Environment Setup and SFT Pilot (Weeks 2–4)

**Goal:** Validate the training pipeline works. Test on small scale before committing to RL.

1. Stand up 8×H100 cluster on RunPod secure cloud ($2.39/hr × 8 = $19.12/hr)
2. Download R2E-Gym dataset from Hugging Face
3. Run 50B-token SFT on Qwen2.5-Coder-32B using only successful trajectories from R2E-Gym
4. Framework: LLaMA-Factory (SFT) or axolotl — both support 32B with 8×H100 via FSDP
5. Evaluate on DeepSWE 10-task smoke; expect modest gains over base model

**Cost:** ~$1,400 for the training run.

### Phase 2: GRPO++ RL Training (Weeks 5–8)

**Goal:** Replicate DeepSWE-Preview's 42% result on our own infrastructure.

**Setup:**
```bash
# Clone rLLM (Agentica's GRPO++ framework)
git clone https://github.com/agentica-org/rLLM

# Download R2E-Gym 4,500-task subset
# (Used by DeepSWE-Preview — already processed and available)

# Configure GRPO++ training
# Key hyperparams (from DeepSWE-Preview paper):
# - No KL loss
# - No entropy loss  
# - Clip High enabled
# - Leave-one-out advantage normalization
# - Compact filtering (remove low-quality trajectories)
# - 200 RL update steps
# - Context: 32K → extended as training progresses
```

**Cluster:** 64×H100 on CoreWeave or Lambda (need InfiniBand for multi-node NCCL)
**Time:** 6 days
**Cost:** ~$37K at spot prices

**Gate:** >40% pass@1 on SWE-bench Verified 10-run average. Evaluate on DeepSWE 10-task smoke in parallel.

### Phase 3: Test-Time Scaling + Verifier (Weeks 9–12)

**Goal:** Add 10–17 absolute percentage points via inference-time compute.

1. **Use DeepSWE-Verifier as starting point** (public, from Together AI)
2. Fine-tune on our own trajectory data (successes vs failures from Phase 2 training)
3. Implement best-of-N sampling (N=4–16 rollouts per task)
4. Combine execution-based (unit tests) + execution-free (verifier) reranking
5. Evaluate on full 113-task DeepSWE suite

**Target:** >55% pass@1 on DeepSWE with TTS (surpass Claude Opus 4.7).

### Phase 4+: Scale and Iterate

- Expand R2E-Gym training to more repositories (8,700 total tasks available)
- Upgrade to Qwen3-Coder-480B if compute budget allows
- Add long-horizon agent RL (SWE-Bench-CL style)
- Evaluate on SWE-bench Pro for contamination-free validation

---

## 6. Data Strategy

### Available Open Datasets (Fully Validated)

**CORRECTION from research:** The Stack v3 does NOT exist. The Stack v2 is the current version. FineWeb-Code as a named dataset also does not exist.

| Dataset | Size | Type | Quality | Source |
|---------|------|------|---------|--------|
| **R2E-Gym** | 8,700 tasks | Executable SWE environments (Docker + unit tests) | Very high — procedurally from commits, no human annotation | `R2E-Gym/R2E-Gym` (GitHub + HF) |
| **The Stack v2** | 67.5TB raw / 32.1TB dedup / ~900B tokens | Raw code pretraining | High — 658 languages, permissive-only licenses | `bigcode/the-stack-v2` (HF) |
| **Nemotron-Pretraining-Code-v2** | 835M records / ~340B tokens | Code + synthetic transformations | High — includes QA pairs, code review, transpilation | `nvidia/Nemotron-Pretraining-Code-v2` (HF) |
| **OpenCodeInstruct** | 5M samples | SFT instruction pairs | High — execution feedback included | arXiv 2504.04030 |
| **LiteCoder-Terminal-SFT** | 11,255 trajectories | Agentic coding SFT (tool-use, editing, terminal) | Medium-high | `Lite-Coder/LiteCoder-Terminal-SFT` (HF) |
| **SWE-bench** | 2,294 tasks | GitHub issue/PR resolution | Medium — contamination risk | swe-bench.com |

### Data Pipeline for RL Training

1. **Primary:** R2E-Gym for RL rollout environments (Docker + unit tests)
2. **SFT warm-up:** OpenCodeInstruct (5M samples) for initial behavior shaping
3. **Holdout:** DeepSWE tasks must remain completely separate — never in training data

### Integrity Policy (Non-Negotiable)

- DeepSWE is the independent holdout benchmark — no DeepSWE task IDs in training data
- Run `check_deepswe_holdout_leak.py` before every training run
- Only use `--allow-training-on-deepswe` with explicit approval

---

## 7. Evaluation Strategy

### Primary (Every Gate)

| Benchmark | When | Target |
|-----------|------|--------|
| DeepSWE 10-task smoke | After every major change | >30% sustained over 3 runs |
| DeepSWE 113-task full | After each RL phase | Gate 2: >42%, Gate 3: >54% |
| SWE-bench Pro | Monthly | Cross-validation (expect ~20pp below Verified scores) |

### Secondary (Quarterly)

- LiveCodeBench: raw code generation quality
- HumanEval/MBPP: regression check
- SWE-MERA: dynamic contamination-resistant evaluation

### Not Primary (but informative)

- SWE-bench Verified: saturated, contamination risk — useful for comparison to published results only

---

## 8. Infrastructure Stack

### Training

| Layer | Choice | Rationale |
|-------|--------|-----------|
| RL framework (primary) | **veRL** (`volcengine/verl`, 21.8k stars) | Most algorithms (GRPO, DAPO, GRPO++, RLOO); proven at scale; FSDP + Megatron backends |
| RL framework (quick prototyping) | TRL (HuggingFace) | Lowest setup friction; most GRPO tutorials use it |
| GRPO++ specifically | **rLLM (Agentica)** | The exact framework used by DeepSWE-Preview; purpose-built for agentic SWE RL |
| SFT framework | **LLaMA-Factory v0.9.5** | 71.9k stars; 100+ models; day-0 support for new models; built-in GRPO via EasyR1 |
| SFT (long-context) | Axolotl | Sequence parallelism (FSDP2); better for >32K context fine-tuning |
| Pretraining (≤32 GPUs) | **FSDP2 + torchrun** | Simple, HF-native, works out of box for 7B–32B |
| Pretraining (>32 GPUs) | nanotron (HF-internal) or Megatron-LM | 3D parallelism; nanotron is cleaner Python; Megatron is most feature-complete |
| Cluster | RunPod secure cloud (prototype) → CoreWeave or Lambda (multi-node RL) | Cost vs reliability; need InfiniBand for NCCL on multi-node |
| Execution environments | Docker (via R2E-Gym) | Each RL rollout in isolated container |

### GRPO Reward Signal Guidance

**Critical validated finding:** Pass-rate rewards (partial credit per test) do NOT reliably beat binary pass/fail. Root cause: ~36.5% of tasks have higher pass-rate rewards for incorrect solutions than near-correct ones ("intra-group gradient conflict"). Source: arXiv 2605.02944, tested on Qwen3-4B and Qwen2.5-7B.

**Recommended reward stack:**
1. **Primary:** Binary unit test pass/fail (simple, no gradient conflicts, universally validated)
2. **Enhancement:** EGCA-style execution-grounded credit assignment — identifies earliest semantic divergence in execution trace, concentrates gradients there. Adds +3.1pp HumanEval, +18% wall-clock overhead. (arXiv 2603.16158)
3. **Quality:** Composite reward (format + correctness + quality score) if code quality beyond correctness matters — validated by human evaluators preferring output in 78.6% of comparisons (arXiv 2506.02211)
4. **Avoid:** Dense pass-rate partial credit as primary signal

### Inference / Serving

| Layer | Choice |
|-------|--------|
| Engine | **vLLM** (PagedAttention, continuous batching) |
| Quantization | AWQ 4-bit for 32B (fits on 2×H100, ~$6–8/hr) |
| Cost at 10K req/day | Single H100, ~$870–1,435/month |

### Benchmark Execution

| Component | Tool |
|-----------|------|
| Agent scaffold | **mini-swe-agent** (100 lines Python, single bash tool) |
| Task runner | **pier** (Harbor fork, Docker + Modal backends) |
| Evaluation harness | `tooling/deepswe/` (already built in this repo) |
| Log tracking | `artifacts/deepswe/experiment_log.csv` |

---

## 9. 16-Week Roadmap

| Phase | Weeks | Goal | Gate | Cost |
|-------|-------|------|------|------|
| **0: Harness** | 1–2 | DeepSWE smoke live with claude-sonnet-4-6 | >28% pass@1 on 10-task (3 seeds) | ~$500 API |
| **1: SFT Pilot** | 3–4 | Qwen2.5-Coder-32B SFT on R2E-Gym subset | Beats base model on smoke | ~$1,400 |
| **2: RL Training** | 5–8 | GRPO++ on 32B, 4,500 tasks | >40% SWE-bench Verified / >35% DeepSWE smoke | ~$37,000 |
| **3: TTS + Verifier** | 9–12 | Best-of-N + verifier reranking | >55% DeepSWE (beat Claude Opus 4.7) | ~$10,000 |
| **4: Scale** | 13–16 | Expand training data, longer RL | >60% DeepSWE | ~$50,000+ |

**Total MVP budget (Phases 0–3): ~$49,000**

---

## 10. Risk Register

| Risk | Severity | Mitigation |
|------|----------|-----------|
| R2E-Gym Docker environments break | Medium | Pin container versions; test locally before cluster run |
| GRPO training diverges | Medium | Use DeepSWE-Preview's exact hyperparams as starting point; monitor reward/KL |
| DeepSWE benchmark moves (new tasks added) | Low | Version-lock the task set; run against frozen snapshot for gates |
| Qwen3-32B license restrictions | Low | Verify license before deployment; Qwen2.5-Coder-32B has Apache 2.0 |
| Compute cost overrun | Medium | Start with smoke validation before committing full cluster; use spot instances |
| SWE-bench contamination invalidates results | High (for Verified) | Already mitigated — use DeepSWE as primary, SWE-bench Pro for cross-check |
| Test-time scaling too expensive to serve | Medium | Profile latency/cost at N=4 before committing to N=16; execution-free verifier as fallback |

---

## 11. Immediate Next Actions (Next 48 Hours)

1. **Configure model_profile.env** with real credentials:
   ```bash
   # Set DEEPSWE_MODEL="claude-sonnet-4-6"
   # Set ANTHROPIC_API_KEY
   # Set DEEPSWE_PIER_ARGS="--env modal --max-retries 2 --ak max_steps=96 --ak temperature=0.2"
   # Set DEEPSWE_AGENT="mini-swe-agent"
   ```

2. **Clone DeepSWE and install pier:**
   ```bash
   git clone https://github.com/datacurve-ai/deep-swe
   uv tool install datacurve-pier
   ```

3. **Run first real smoke (10 tasks, seed 0):**
   ```bash
   DEEPSWE_N_TASKS=10 DEEPSWE_SAMPLE_SEED=0 \
   ./tooling/deepswe/run_deepswe.sh --tasks-dir /path/to/deep-swe/tasks
   ```

4. **Download R2E-Gym dataset** to local storage:
   ```bash
   git clone https://github.com/R2E-Gym/R2E-Gym
   # or: huggingface-cli download R2E-Gym/R2E-Gym --repo-type dataset
   ```

5. **Clone rLLM** (GRPO++ framework):
   ```bash
   git clone https://github.com/agentica-org/rLLM
   ```

---

## 12. What We Are NOT Doing (and Why)

| Approach | Verdict | Reason |
|----------|---------|--------|
| Pretraining 7B from scratch (1T tokens) | Not first | 77 days on 16×H100, $147K — wrong first move |
| Using SWE-bench Verified as primary target | No | Saturated (93.9% top), contamination risk, `.git` loophole |
| Building custom training environments | No | R2E-Gym (8,700 tasks) already exists and is proven |
| Proprietary training data | No | R2E-Gym + OpenCodeInstruct cover the need at no cost |
| Starting with MoE architecture | No | Dense 32B first; MoE is Phase 4+ after baseline is solid |

---

## 13. Key Open-Source Assets (Everything We Need Exists)

| Asset | Repository | Used By |
|-------|-----------|---------|
| Training framework (GRPO++) | `agentica-org/rLLM` | DeepSWE-Preview |
| Training environments | `R2E-Gym/R2E-Gym` | DeepSWE-Preview, R2E-Gym paper |
| Reference model weights | `agentica-org/DeepSWE-Preview` | Starting point / target to beat |
| Execution-free verifier | `agentica-org/DeepSWE-Verifier` | TTS pipeline |
| Agent scaffold | `SWE-agent/mini-swe-agent` | All DeepSWE evaluations |
| Task runner | `datacurve-ai/pier` | DeepSWE benchmark harness |
| Benchmark tasks | `datacurve-ai/deep-swe` | Our primary eval target |
| Benchmark harness | `tooling/deepswe/` (this repo) | Already built |
| SFT dataset | OpenCodeInstruct (HF hub) | SFT warm-up stage |

---

---

## 14. Research Validation Notes

All findings validated by 4 concurrent research agents (2026-06-06):

- **Agent 1 (Benchmarks):** Confirmed DeepSWE leaderboard scores, SWE-bench saturation, mini-swe-agent architecture, pier backend list, RL training paper results
- **Agent 2 (Compute):** Confirmed GPU pricing from live provider pages; compute estimates validated against Llama-3 MFU reports and Seaweed-7B training cost reference
- **Agent 3 (RL Training):** Confirmed SWE-RL, DeepSWE-Preview, Agent-RLVR, Multi-turn DAPO results; 3-vote adversarial verification on key claims
- **Agent 4 (Base Models/Frameworks):** **Corrected** The Stack v3 claim (does not exist); confirmed Qwen3 family availability; confirmed veRL/LLaMA-Factory as dominant frameworks; validated GRPO reward signal research

**Key corrections vs existing build plans:**
1. The Stack v3 → does not exist; The Stack v2 is current (~900B tokens)
2. FineWeb-Code → does not exist as a named dataset
3. SWE-bench Verified is now effectively saturated (93.9%) — use DeepSWE as primary gate
4. Pretraining from scratch is not the right first move — GRPO++ on Qwen3-32B + R2E-Gym for $37K gets to 42% in 6 days
5. Pass-rate rewards can hurt GRPO training — use binary pass/fail as primary signal
