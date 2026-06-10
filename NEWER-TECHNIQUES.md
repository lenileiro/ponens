# Newer Techniques — Research Synthesis
**Date:** 2026-06-06  
**Status:** 3/4 research agents synthesized (serving/inference section pending)  
**Extends:** VALIDATED-PLAN.md

This document covers the cutting-edge techniques published after DeepSWE-Preview (July 2025) and updates our plan where new research changes the recommended approach.

---

## TL;DR — What Changes in Our Plan

| Area | Old Plan | Updated Recommendation |
|------|----------|----------------------|
| RL algorithm | GRPO++ | **REINFORCE++ or Dr. GRPO** (unbiased; shorter responses) |
| Reward modeling | Binary pass/fail | Binary pass/fail **+ PRIME** for dense process rewards |
| Test-time scaling | DeepSWE-Verifier (execution-based) | **CodeScaler** execution-free PRM (10× latency reduction) |
| Synthetic data | R2E-Gym only | R2E-Gym **+ adversarial test generation** (ACE/Code-A1) |
| Repo navigation | None | **RepoGraph or CGM** as plug-in (immediate gains on SWE tasks) |
| Multi-agent | Single agent | **Verifier model** alongside patch generator |
| Architecture (if pretraining) | Dense transformer | Consider **Nemotron-H hybrid Mamba** (3× faster inference) |
| Training precision | BF16 | **FP8** from the start (DeepSeek-V3 proof-of-concept) |

---

## 1. RL Algorithm Upgrades

### 1.1 Dr. GRPO — Drop-In GRPO Fix

**Paper:** arXiv:2503.20783 (COLM 2025) | Sea AI Lab / NUS  
**Two changes to vanilla GRPO:**
1. Remove standard-deviation normalization from advantage computation
2. Remove length normalization (the `1/|o_i|` divisor)

**Why this matters:** These two terms introduce systematic biases that cause GRPO to generate increasingly long responses and over-weight easy questions. Dr. GRPO makes the gradient estimator unbiased. Implementation is a 5-line change to the GRPO loss.

**Result:** 43.3% AIME 2024 from a 7B model — new SOTA at that scale. No SWE-bench numbers yet, but the bias elimination applies equally to code.

**Action:** Replace GRPO++ with Dr. GRPO in the rLLM training loop. This is a drop-in replacement.

---

### 1.2 REINFORCE++ — Global Advantage Normalization

**Paper:** arXiv:2501.03262 | Published January 2025, extensively validated  
**Core change:** Compute advantage mean/variance across the *entire training batch* rather than within per-prompt groups (which GRPO does).

**Why local normalization is broken:** Local normalization is a biased estimator that only becomes unbiased as group size per prompt → ∞. It causes advantages to explode when all outputs for a prompt get similar rewards (low within-group variance). Global normalization is nearly unbiased as batch size increases — which is the actual scaling axis.

**Variants:**
- REINFORCE++ (k≥1 samples): general-purpose
- REINFORCE++ with baseline (k>1): group-sampling variant, analogous to GRPO but globally normalized

**Action:** Test both Dr. GRPO and REINFORCE++ in smoke runs. Both address GRPO's bias; Dr. GRPO is simpler to implement, REINFORCE++ has broader validation.

---

### 1.3 RLOO — On-Policy with Leave-One-Out Baseline

**Paper:** arXiv:2402.14740 (Cohere/Google) | Used by SkyRL-Agent (SWE-bench 39.4%)  
**Key property:** Strictly on-policy — generates fresh rollouts each step, uses them once. No importance sampling corrections. LOO baseline only (no std division).

**When to prefer over GRPO/REINFORCE++:** Multi-turn agentic tasks where the off-policy nature of GRPO introduces distributional mismatch. The SkyRL-Agent paper (arXiv:2511.16108) found switching from GRPO to RLOO substantially improves long-horizon SWE agent performance with 1.55× rollout speedup via async dispatching.

**Action:** Use RLOO specifically for multi-turn agent training (Phase 2 onwards), Dr. GRPO/REINFORCE++ for single-turn code generation.

---

### 1.4 PRIME — Dense Process Rewards Without Step Annotation

**Paper:** arXiv:2502.01456 | GitHub: PRIME-RL/PRIME  
**The problem it solves:** Standard Process Reward Models (PRMs) require annotated intermediate steps — expensive and hackable. PRIME derives dense step-level rewards from outcome-only labels.

**How it works:**
- Computes implicit step rewards by comparing policy token-level log-probabilities against a reference model
- Uses a trajectory-based DPO-like objective supervised by pass/fail execution results
- The implicit PRM updates *online* as the policy improves, keeping rewards fresh
- Compatible with any advantage estimator (GRPO, RLOO, REINFORCE++)

**Results:** Starting from Qwen2.5-Math-7B-Base: +15.1% average across reasoning benchmarks over SFT baseline; Eurus-2-7B-PRIME surpasses Qwen2.5-Math-7B-Instruct on 7 benchmarks using 10% of its training data.

**Action:** Add PRIME as a reward augmentation layer on top of binary pass/fail. The implementation is a training loop wrapper, not a separate model training phase.

---

## 2. Process Reward Models for Code

### 2.1 DreamPRM-Code — Chain-of-Function PRM

**Paper:** arXiv:2512.15000 (UCSD, December 2025)  
**Key innovation:** Treats each *function* in generated code as a reasoning step (Chain-of-Function). Meta-learning-based label correction uses clean unit-test outcomes to denoise intermediate step labels via bi-level optimization.

**Result:** 80.9% pass@1 on LiveCodeBench — surpasses OpenAI o4-mini at inference time. This is the best test-time scaling result for competitive programming style tasks from an execution-free PRM.

**Why it matters for us:** This is the PRM we should train as our verifier for test-time scaling in Phase 3. It's function-granularity (not token), which maps naturally to SWE tasks where functions/files are the natural edit units.

---

### 2.2 FunPRM — Function-Level with Meta-Reward Correction

**Paper:** arXiv:2601.22249 (UCSD, January 2026) — extends DreamPRM-Code  
Adds a meta-reward correction mechanism that purifies noisy partial-solution rewards using final-solution rewards from unit testing. SOTA on LiveCodeBench and BigCodeBench.

**Action (Phase 3):** Train a function-level PRM on our RL training trajectories using DreamPRM-Code or FunPRM methodology. Use it for:
1. Best-of-N reranking during test-time scaling
2. Dense reward signal during future RL training iterations

---

### 2.3 CodeScaler — Execution-Free RL + 10× TTS Latency

**Paper:** arXiv:2602.17684 (LARK AI Lab, February 2026)  
**What it is:** An execution-free reward model for *both* RL training and test-time scaling.

**Key results:**
- +11.72 pts average on 5 coding benchmarks for Qwen3-8B-Base
- Outperforms binary execution-based RL by +1.82 pts
- Enables RL on synthetic datasets *without test cases*
- **10× latency reduction** vs. unit test approaches at inference time

**Why this matters for deployment:** Unit test execution during TTS is slow and expensive (sandboxed containers for every rollout). CodeScaler's execution-free PRM makes TTS practical in production — 10× faster means we can serve TTS without per-request container overhead.

**Action:** Train CodeScaler alongside DreamPRM-Code. Use CodeScaler for production TTS (latency-critical), DreamPRM-Code for offline evaluation (highest accuracy).

---

## 3. SWE-Specific Training Advances (2026 Papers)

### 3.1 SWE-TRACE — Rubric PRM + Heuristic TTS (April 2026)

**Paper:** arXiv:2604.14820 | SOTA result for open-weight models  
**Three innovations:**

**1. Cascaded SFT on shortest-path trajectories:** 60,000 instances curated to be token-efficient (not verbose "thinking out loud"). +4.2 pts 4B, +2.8 pts 30B, with 21-29% token reduction.

**2. Rubric-Based PRM:** A "Rubric-Agent" generates issue-specific checklists (correct file localization, edit strategy, test validation) and scores trajectories against these rubrics. The margin-separated formulation rewards *how well* the agent solved the problem, not just *whether* it passed tests. This is the key advance over binary execution-only rewards.

**3. HG-TTS (Heuristic Graph Test-Time Scaling):** The trained PRM dynamically prunes action candidates at each step instead of parallel sampling. Result: 71.2% at 36.5 min vs. 69.9% at 63.8 min with 3× more environment executions.

**Results:** SWE-TRACE-30B (Qwen3-30B-A3B): 63.5% base, **71.2% with TTS**. Beats SWE-Master-32B by +2.1 pts.

**Action:** Adopt the Rubric-PRM methodology as our Phase 3 verifier training approach. The rubric generation step is straightforward to implement on R2E-Gym tasks.

---

### 3.2 SWE-Master — Full Pipeline to 70.8% (February 2026)

**Paper:** arXiv:2602.03411 | Qwen2.5-Coder-32B base  
**Four-stage pipeline:**
1. Teacher trajectory synthesis
2. Long-horizon SFT
3. RL with real execution feedback
4. Inference design (TTS@8)

**Results:** 61.4% Pass@1 → **70.8% with TTS@8** on SWE-bench Verified.

**Key learning:** Long-horizon SFT before RL is important — it initializes the model with diverse trajectory patterns before RL refines them. This is different from DeepSWE-Preview (pure RL, no SFT first).

---

### 3.3 SWE-ZERO to SWE-HERO — NVIDIA (April 2026)

**Paper:** arXiv:2604.01496 | NVIDIA  
**Two-stage SFT recipe distilling from Qwen3-Coder-480B:**
- Stage 1 (SWE-ZERO): 300K *execution-free* trajectories for repository reasoning (teaches navigation without execution overhead)
- Stage 2 (SWE-HERO): 13K *execution-backed* refinement trajectories

**Result:** SWE-HERO-32B: 62.2% SWE-bench Verified, **44.1% SWE-bench Multilingual** (zero-shot, trained only on Python).

**Key learning:** Execution-free trajectories at scale (300K) are sufficient for the repository reasoning skills; execution is only needed for the final refinement layer. This dramatically reduces the compute needed to generate training data.

---

### 3.4 SERA — Trajectories Without Unit Tests (January 2026)

**Paper:** arXiv:2601.20789  
**Soft Verified Generation (SVG):** Creates training trajectories from any repository *without unit tests*. 200K+ synthetic trajectories.

**Result:** Qwen3-32B with SFT only: **30.0% on SWE-bench Verified**, matching Devstral-Small-2 while being fully open-source.

**Why this matters:** R2E-Gym requires repos with existing unit tests. SERA unlocks the much larger universe of repos *without* tests for trajectory generation.

---

## 4. Synthetic Data Generation — Newer Techniques

### 4.1 rStar-Coder — Mutual Consistency Verification (May 2025)

**Paper:** arXiv:2505.21297  
**The core method:**
1. Generate multiple independent solutions per problem using QWQ-32B (8-16 solutions)
2. Execute each against all test inputs
3. Correct solutions agree; incorrect solutions diverge — use supermajority vote as ground truth
4. No reference solution needed — mutual agreement is the oracle

**Three-step test case generation:**
1. Generate `generate_test_input(scale)` and `validate_test_input(input)` Python functions
2. Parameterize across 4 difficulty tiers (trivially small → large)
3. Execute and retain only constraint-valid inputs

**Results:** Qwen2.5-7B: LiveCodeBench **17.4% → 57.3%** (3.3× improvement). Qwen2.5-14B: **23.3% → 62.5%** (beats o3-mini-low).

**Action:** Use mutual consistency verification as our primary method for validating synthetic problems. Apply on top of R2E-Gym to generate additional training instances from repos without existing test suites.

---

### 4.2 GASP — Guided Asymmetric Self-Play (March 2026)

**Paper:** arXiv:2603.15957 | University of Tübingen / MPI-IS  
**The key innovation:** Ground self-play against 146 "goalpost" hard problems — tasks where pass@100 = 0 across all model seeds. The teacher generates *easier variants* of these hard problems, not arbitrary problems.

**Difficulty-band gating:** Accept generated problems only when 0.3 ≤ p ≤ 0.7 (learnable but not trivial). Reject too-easy (p > 0.7) and too-hard (p < 0.3) problems.

**Result:** GASP solves 11 previously-unsolvable goalpost questions; unguided self-play (AZR) solves 0.

**Why this is the right self-play design:** Most self-play systems fail to push the model's frontier because they generate problems at current model difficulty, not at the difficulty ceiling. GASP's goalpost grounding forces the curriculum toward the actual frontier.

**Action:** Apply GASP to generate additional training problems targeting our DeepSWE failure cases. The 113 DeepSWE tasks where we score 0 are natural goalpost problems.

---

### 4.3 Adversarial Test Generation — ACE and Code-A1 (2026)

**ACE (arXiv:2605.16299, May 2026):** Two heads — solver and adversary. Adversary generates test inputs that discriminate between correct/incorrect solutions. Trained via KTO. Results: Qwen2.5-7B LiveCodeBench **30.4% → 38.9%**.

**Code-A1 (arXiv:2603.15611, March 2026):** Two *separate* models with opposing objectives. Code LLM maximizes passing tests; Test LLM maximizes finding failures. White-box access without self-collusion (the key problem with one model doing both).

**The verification ceiling problem (arXiv:2509.20837):** When code and tests come from the same model, the verifier caps training data diversity at its own competence level. Adversarial separation is the solution.

**Action (Phase 2, alongside RL):** Train a dedicated test-generation model adversarially against our code model. Use Code-A1's two-model separation to avoid self-collusion.

---

### 4.4 Nemotron-Code-v2 Style Transformations for Pretraining

**For any pretraining runs:** Apply these transformations to The Stack v2 code:
1. **SGCR (Style-Guided Code Rewriting):** Enforce Google Python Style Guide (docstrings, type hints, naming) → +9 pts downstream (arXiv:2505.02881)
2. **SCOR (Self-Contained Optimization Rewriting):** Semantic clarity pass after SGCR → +5 additional pts
3. **Code Review Dialogues:** Generate reviewer+code-improvement pairs
4. **Student-Teacher Dialogues:** Educational explanations of code patterns
5. **Language Transpilation:** Python→C++ doubles effective C++ coverage

These transformations are straightforward to apply using Qwen3-32B as the transformation model.

---

## 5. Multi-Agent and Repo-Level Techniques

### 5.1 RepoGraph — Plug-In Repo Navigation (ICLR 2025)

**Paper:** arXiv:2410.14684  
Builds a repository-level code graph capturing cross-file dependencies, call graphs, and import relationships. Plug-in to existing SWE agents — no model changes required.

**Result:** Boosts all four tested SWE agent systems. New SOTA among open-source frameworks at time of publication.

**Action (Phase 1, immediate):** Add RepoGraph as a retrieval layer to our mini-swe-agent runs on DeepSWE. This is a preprocessing step that doesn't require model training.

---

### 5.2 CGM — Code Graph Model (May 2025)

**Paper:** arXiv:2505.16901  
Graph structure integrated directly into the LLM's attention mechanism via a specialized adapter. With Qwen2.5-72B: **43.00% on SWE-bench Lite**, #1 among open-weight, +12.33 pp over previous open-source SOTA.

**Progression:** RepoGraph (plug-in) → CGM (attention-integrated). Start with RepoGraph; upgrade to CGM when training a new model checkpoint.

---

### 5.3 AlphaCodium — Flow Engineering (Jan 2024, widely validated)

**Paper:** arXiv:2401.08500  
Two-phase flow: pre-processing (problem reflection, AI test synthesis) → iterative code generation with *actual execution* against both public and AI-generated tests.

**Result:** GPT-4 on CodeContests: **19% → 44% pass@5** (2.3× improvement).

**The key insight:** The LLM as a component in a test-driven loop, not a one-shot oracle. This is directly applicable to our agent design — our mini-swe-agent already loops, but adding AI-generated additional test cases before the loop starts is the AlphaCodium contribution.

**Action:** Add an AlphaCodium-style pre-processing phase to our agent: before the repair loop, generate 5-10 additional test cases from the problem description. Run these in the loop alongside the official tests.

---

### 5.4 Agyn — Team-Based Multi-Agent (February 2026)

**Paper:** arXiv:2602.01465  
Models an engineering org: coordinator, researcher, implementer, reviewer. Agents work in isolated sandboxes.

**Result:** 72.2% SWE-bench 500 with GPT-5 family.

**Action:** Not immediate — complex to implement. Consider for Phase 4 after single-agent baseline is solid. The reviewer agent pattern (one generates patch, one verifies) is the minimum viable version to add.

---

## 6. Architectural Innovations Worth Tracking

### 6.1 Nemotron-H — Hybrid Mamba-Transformer (April 2025, NVIDIA)

**Paper:** arXiv:2504.03624  
**Architecture:** ~92% of layers are Mamba-2 (SSM), ~8% are full self-attention, dispersed evenly.

**Key results:**
- Nemotron-H-56B matches Qwen-2.5-72B and Llama-3.1-70B on 9/17 benchmarks
- **Up to 3× faster inference throughput** vs. equivalent dense transformers
- FP8 training recipe included

**Why it matters:** If we ever train a model from scratch, the Mamba-Transformer hybrid is now production-validated at 56B scale by NVIDIA. The 3× inference speedup directly reduces serving cost.

**Action:** Not for our first iteration. Revisit when building a second-generation model from scratch.

---

### 6.2 NSA — Native Sparse Attention (February 2025, DeepSeek)

**Paper:** arXiv:2502.11089 (ACL 2025)  
End-to-end trainable sparse attention with hardware-aligned algorithm. Reduces pretraining compute while maintaining full attention quality. Two components: coarse-grained token compression + fine-grained token selection.

**Status:** Research/pretraining proposal — not yet in a deployed coding model. DeepSeek will likely use it in their next model generation.

**Action:** Follow but don't adopt yet. Watch for DeepSeek's next release to see production validation.

---

### 6.3 FP8 Training — Now Production-Ready

**Key validation:** DeepSeek-V3 (671B MoE, 14.8T tokens) trained entirely in FP8 with equivalent dynamics to BF16. Fine-grained quantization: 1×128 tiles for activations, 128×128 blocks for weights.

**FP8-RL (arXiv:2601.18150, 2026):** FP8 extended to reinforcement learning training — addresses the unique instability challenges of RL in low precision.

**Action:**
- **Immediate:** Use BF16 for our first RL training run (simpler, well-validated)
- **Phase 3+:** Switch to FP8 using NVIDIA's recipe (saves ~20-30% training compute with no quality loss)

---

## 7. Prioritized Action List for Our Plan

### Immediately Actionable (Phase 0–1)

| Action | Impact | Effort |
|--------|--------|--------|
| Add **RepoGraph** to mini-swe-agent runs | +5-12 pts on SWE tasks | Low — preprocessing step |
| **AlphaCodium-style** AI test generation before repair loop | +10-25 pts on competitive coding | Low — prompt engineering |
| Replace GRPO++ with **Dr. GRPO or REINFORCE++** in rLLM | Unbiased gradients, shorter responses | Very low — 5-line change |
| Add **RLOO** for multi-turn agent RL (Phase 2) | Better long-horizon performance | Low — already in veRL |

### Phase 2 (RL Training)

| Action | Impact | Effort |
|--------|--------|--------|
| Add **PRIME** alongside binary rewards | +15% on reasoning benchmarks | Medium — training loop wrapper |
| Use **SWE-Master pipeline** (SFT before RL) | Higher RL ceiling | Medium — adds SFT stage |
| Apply **Rubric-PRM** (SWE-TRACE approach) | +2-3 pts vs binary rewards | High — needs rubric agent |
| Add **adversarial test model** (Code-A1) | Breaks verification ceiling | High — separate model training |

### Phase 3 (TTS + Verifier)

| Action | Impact | Effort |
|--------|--------|--------|
| Train **DreamPRM-Code** (function-level PRM) | Best accuracy for TTS | Medium |
| Train **CodeScaler** (execution-free PRM) | 10× latency reduction | Medium |
| Use **GASP** with DeepSWE failures as goalposts | Targets hardest tasks | Medium |
| Apply **SERA** for trajectory generation on repos without tests | 10× more training data | Medium |

---

## 8. Updated Algorithm Recommendation

**For Phase 2 RL training, replace GRPO++ with this stack:**

```
1. Base algorithm: Dr. GRPO (or REINFORCE++)
   - Remove std normalization from advantages
   - Remove length normalization
   - Use global batch-level normalization (REINFORCE++)

2. Reward signal:
   - Primary: Binary pass/fail from unit tests
   - Augment: PRIME implicit process rewards (online PRM)
   - Add: Rubric-based scoring (from SWE-TRACE)
   
3. For multi-turn agent RL: Switch to RLOO (on-policy, no distributional mismatch)

4. Test-time scaling:
   - Accuracy-critical: DreamPRM-Code (function-level PRM)
   - Latency-critical: CodeScaler (execution-free, 10× faster)
   - Strategy: Best-of-N with PRM reranking
```

---

## 9. Key Papers Reference Table

| Paper | arXiv | Date | What It Adds |
|-------|-------|------|-------------|
| Dr. GRPO | 2503.20783 | Mar 2025 | Unbiased GRPO; remove std+length norm |
| PRIME | 2502.01456 | Feb 2025 | Dense process rewards from outcomes only |
| RLOO | 2402.14740 | Feb 2024 | On-policy LOO baseline; best for multi-turn |
| REINFORCE++ | 2501.03262 | Jan 2025 | Global batch-level advantage normalization |
| SWE-TRACE | 2604.14820 | Apr 2026 | Rubric PRM + step-level TTS (71.2%) |
| SWE-Master | 2602.03411 | Feb 2026 | SFT→RL pipeline (70.8% with TTS@8) |
| SWE-ZERO/HERO | 2604.01496 | Apr 2026 | 300K execution-free trajectories → 62.2% |
| SERA | 2601.20789 | Jan 2026 | Trajectories without unit tests |
| CodeScaler | 2602.17684 | Feb 2026 | Execution-free PRM, 10× TTS latency |
| SkyRL-Agent | 2511.16108 | Nov 2025 | Async RLOO, 39.4% from 32B |
| DreamPRM-Code | 2512.15000 | Dec 2025 | Function-level PRM, 80.9% LiveCodeBench |
| FunPRM | 2601.22249 | Jan 2026 | Meta-reward correction, SOTA LiveCodeBench |
| rStar-Coder | 2505.21297 | May 2025 | Mutual consistency verification, 418K problems |
| GASP | 2603.15957 | Mar 2026 | Self-play grounded to hard failures |
| Code-A1 | 2603.15611 | Mar 2026 | Adversarial code+test separation |
| ACE | 2605.16299 | May 2026 | Adversarial solver+adversary with KTO |
| CodeContests-O | 2601.13682 | Jan 2026 | Iterative test generation with feedback |
| RepoGraph | 2410.14684 | Oct 2024 | Repo code graph plug-in |
| CGM | 2505.16901 | May 2025 | Graph-integrated into LLM attention |
| AlphaCodium | 2401.08500 | Jan 2024 | Flow engineering, 2.3× CodeContests |
| Agyn | 2602.01465 | Feb 2026 | Team-based multi-agent SWE (72.2%) |
| Nemotron-H | 2504.03624 | Apr 2025 | Hybrid Mamba, 3× faster inference |
| NSA | 2502.11089 | Feb 2025 | Native sparse attention (trainable) |
| FP8-RL | 2601.18150 | 2026 | FP8 training for RL stability |

---

---

## 10. Serving and Inference Innovations

### 10.1 EAGLE-3 Speculative Decoding — 5–6× Speedup for Code

**Paper:** arXiv:2503.01840 (NeurIPS 2025) | GitHub: SafeAILab/EAGLE

EAGLE reuses the target model's top-layer hidden features as input to a lightweight draft head (one transformer layer + original LM head). This gives the draft head access to the target model's internal representations, yielding far higher acceptance rates (~4 tokens/step vs ~2 for vanilla speculative decoding).

**EAGLE-3 improvements over EAGLE-2:**
- Fuses low + mid + high layer features from the target model (tri-layer concatenation → FC projection)
- Eliminates feature-prediction loss; predicts tokens directly
- **Discovered a scaling law** — doubling training data keeps improving speedup (EAGLE-1/2 saturated early)

**Concrete speedups (temperature=0):**

| Model | HumanEval | MT-bench | Mean |
|-------|-----------|----------|------|
| Vicuna 13B (EAGLE-2) | 4.96× | 4.26× | 4.22× |
| Vicuna 13B (EAGLE-3) | **6.47×** | **5.58×** | **5.51×** |
| LLaMA-3.3-70B (EAGLE-3) | **4.79×** | **4.11×** | **4.12×** |
| Batch size 64 (production) | — | — | **1.38× throughput** |

**For our 32B model:** Expect 3.5–5× speedup at batch size 1 (interactive coding). Code is the best use case — highly predictable token sequences (loops, boilerplate, syntax) drive high acceptance rates.

**Action:** Enable EAGLE-3 for all interactive inference. Draft heads for Qwen3 are pre-trained and available in the SafeAILab repo.

---

### 10.2 SGLang vs. vLLM — Which to Use for Coding Agents

**Benchmark (H100, Llama 3.1 8B):**

| Metric | vLLM V1 | SGLang | Delta |
|--------|---------|--------|-------|
| Total throughput | 12,500 tok/s | 16,200 tok/s | **+29%** |
| Output throughput | 413 tok/s | 894 tok/s | **+117%** |
| TTFT (p50) | 103 ms | 79 ms | 23% faster |
| Structured JSON output | baseline | **3× faster** | 96–98% vs 90–94% compliance |

**SGLang's key advantage — RadixAttention:** KV cache stored in a radix tree keyed by token sequences. Shared prefixes (system prompt, tool schema, repo context) are automatically detected and reused. For a coding agent where every request shares a large system prompt + tool definitions, cache hit rates of **60–80%** are realistic — directly reducing TTFT and compute.

**vLLM V1's advantages:** Broader hardware support (AMD ROCm, AWS Neuron, Ascend), larger community, more mature LoRA serving. New in V1: CPU/GPU parallelism (tokenization off the critical path), prefix caching zero-overhead by default, unified token scheduler enabling chunked prefill.

**Decision:**
- **SGLang** for our coding agent (shared system prompts, structured tool outputs, interactive latency) 
- **vLLM V1** if we need AMD/non-H100 hardware or LoRA adapter serving

---

### 10.3 Chunked Prefill + Disaggregated Prefill/Decode

**Chunked prefill** (default in vLLM V1, available in SGLang): Splits large prefill requests into chunks (~512 tokens) batched with ongoing decode requests. Essential for long-context coding where a 50K-token repo context would otherwise stall all decode requests for its full duration.

**Disaggregated prefill/decode** (SGLang production, vLLM v0.7.1+ experimental): Separate GPU pools for prefill and decode connected by KV transfer. Lets you tune TTFT and ITL independently. SGLang/LMSYS deployed this for DeepSeek V3 (671B MoE) at scale, achieving 52.3K input tok/s and 22.3K output tok/s per node — 5× faster than vanilla tensor parallelism.

**Action:** Enable chunked prefill from day 1. Add disaggregated prefill/decode when traffic justifies separate GPU pools (Phase 4+).

---

### 10.4 KV Cache Quantization

**FP8 KV cache** (vLLM `--kv-cache-dtype fp8`, SGLang equivalent):
- Halves KV memory vs BF16
- Reduces ITL slope to **54% of BF16** at 7k+ token contexts
- <2 point accuracy loss on reasoning benchmarks
- **Enable for contexts ≥7K tokens on H100/A100**

**CLA (Cross-Layer Attention):** arXiv:2405.12981 — shares K/V heads across adjacent layers. Additional 2× KV cache reduction on top of GQA. Must be baked in at training time. Adopted by Apple Foundation Model, Gemma 3n. **Build CLA into any new model we train from scratch.**

**INT4 KV (emerging):** MixKVQ (arXiv:2512.19206), KVTuner (arXiv:2502.04420) show 4× reduction. Still needs per-layer calibration; use FP8 for now.

---

### 10.5 Quantization Strategy for Our 32B Model

| Hardware | Method | Quality Retention | Throughput |
|----------|--------|------------------|-----------|
| **H100/A100** | **FP8** | ~99.2% (Qwen3-32B FP8 slightly *improved* agentic bench) | **2.2× vs FP16** |
| A10G/RTX 4090 | Marlin-AWQ (4-bit) | ~95% | 741 vs 461 tok/s (+60%) |
| CPU/offload | GGUF Q5_K_M | ~97% | 93 tok/s (llama.cpp) |

**GPTQ without Marlin kernel is unacceptable** — only 277 tok/s (worse than FP16 baseline). Always use Marlin kernels with GPTQ/AWQ.

**Estimated SWE-bench quality drop from quantization:** FP8 ≈ 0 pts, AWQ 4-bit ≈ 3–7 pts, GPTQ 4-bit ≈ 5–12 pts. No published direct measurements — extrapolated from reasoning task benchmarks.

---

### 10.6 Serving Stack Summary

| Concern | Solution | Framework |
|---------|----------|-----------|
| Interactive latency (single user) | EAGLE-3 speculative decoding | SGLang + EAGLE-3 |
| Shared system prompt throughput | RadixAttention prefix cache | SGLang |
| Long-context (>7K tokens) | FP8 KV cache + chunked prefill | SGLang or vLLM V1 |
| Structured tool-call outputs | Constrained decoding | SGLang (3× faster) |
| 32B on H100 (best quality) | FP8 weights + FP8 KV | vLLM/SGLang |
| 32B on A10G (budget) | Marlin-AWQ | vLLM |
| Future MoE serving | EP + EPLB + TBO + DeepEP | SGLang |
| New model architecture | CLA (trained in) | Design choice |

**Bottom line:** Default to SGLang for our coding agent serving. Enable EAGLE-3 for interactive mode, FP8 KV cache for long-context, and RadixAttention for system prompt reuse. This stack reduces serving cost by **3–5×** vs a naive vLLM V0 baseline with FP16 weights.
