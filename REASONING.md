# Reasoning for Coding LLMs — Research Synthesis
**Date:** 2026-06-06
**Status:** Fully synthesized (4/4 reasoning research agents complete)
**Extends:** VALIDATED-PLAN.md, NEWER-TECHNIQUES.md

This document covers how reasoning ("thinking") capabilities are trained, when they help coding specifically, and how to scale reasoning at test time — translated into concrete decisions for our coding LLM.

---

## TL;DR — The Five Decisions That Matter

| Question | Answer | Evidence |
|----------|--------|----------|
| Does reasoning help our SWE target? | **Yes but modestly** (+7pp on SWE-bench), **hugely on competitive coding** (+30pp LiveCodeBench) | R1 vs V3: SWE 49.2% vs 42.0%; LiveCodeBench 65.9% vs 36.2% |
| Distill reasoning or RL it from scratch? | **Distill first, then RL** — distillation expands the ceiling, RL only amplifies what's there | arXiv 2504.13837 (NeurIPS Oral), 2505.21067 |
| How much reasoning data do we need? | **~1K–17K high-quality traces** beats 100K+ noisy ones | s1 (1K), Sky-T1 (17K, $450), Bespoke-Stratos (17K) |
| What's the optimal thinking length for SWE? | **~2K tokens** — returns saturate fast (0→2K = +3.2pp, 2K→8K = +0.4pp) | patch-budget study, Claude 4 budget sweep |
| Where do we get the biggest TTS win? | **Selection/verifier quality**, not raw sampling — random→oracle gap is 12.4pp | CodeMonkeys, S* |

**Bottom line:** Reasoning is worth adding to our plan, but it is NOT the dominant lever for SWE tasks. The dominant levers remain: (1) base model quality, (2) RL on executable environments, (3) verifier/selection quality at test time. Reasoning is a force-multiplier on competitive-coding-style subtasks and a modest boost on patch generation.

---

## 1. How Reasoning Is Trained (The Recipes)

### 1.1 DeepSeek-R1 — The Canonical 4-Stage Recipe

**arXiv:2501.12948** | Base: DeepSeek-V3-Base (671B MoE)

1. **R1-Zero (proof of concept):** Pure GRPO RL on the base model, no SFT. Proves reasoning *emerges* from RL alone — including the "aha moment" (spontaneous self-correction) and growing response length. But outputs are unreadable (language mixing).
2. **Cold-Start SFT:** Fine-tune on thousands of curated long-CoT examples. Format: `<reasoning> ... </reasoning> <summary>`. Solves readability and stabilizes subsequent RL.
3. **Reasoning RL (GRPO):** Large-scale RL on math/code/logic with **rule-based rewards only** (accuracy via execution/symbolic check + format + language consistency). Deliberately no neural reward model → prevents reward hacking.
4. **Rejection-sample SFT + General RL:** ~800K samples (600K reasoning + 200K general) from the RL checkpoint, then a final RL pass for helpfulness/harmlessness.

**Results:** AIME 79.8%, LiveCodeBench 65.9%, Codeforces 2029, SWE-bench 49.2%. Matches o1-1217.

**The "aha moment":** During R1-Zero RL, the model spontaneously learns to backtrack and re-evaluate ("Wait, wait. That's an aha moment...") because re-evaluation yields higher reward. Not programmed — emergent. Response length grows throughout training as the model learns to allocate more compute to harder problems.

### 1.2 The Cheap Distillation Path (What We Should Actually Do First)

| Model | Recipe | Cost | Result |
|-------|--------|------|--------|
| **Sky-T1-32B** (NovaSky/Berkeley) | SFT Qwen2.5-32B on 17K QwQ traces, 8×H100, 19h | **$450** | Matches o1-preview on coding |
| **Bespoke-Stratos-32B** | Same recipe, 17K curated traces | similar | LiveCodeBench-All 71.1% (beats o1-preview) |
| **s1-32B** (Stanford) | SFT on 1K curated traces + budget forcing | minimal | Beats o1-preview on MATH by 27% |
| **R1-Distill-Qwen-7B** | SFT on 800K R1 traces | — | AIME 55.5%, LiveCodeBench 49.1% (beats QwQ-32B-Preview) |

**Key finding (s1, Sky-T1, OpenThoughts):** Data quality >> quantity. 1K carefully curated reasoning traces can match expensive RL. **Teacher model quality is the single most important factor** (OpenThoughts, 1000+ controlled experiments).

### 1.3 Distillation vs. RL-from-scratch — The Settled Debate

- **arXiv:2504.13837** (NeurIPS 2025 Oral, ICML Best Paper): RLVR does **NOT create new reasoning capabilities** — it recombines patterns already in the base model. Base models with large-k sampling can match RL-trained models.
- **Distillation genuinely expands the reasoning ceiling** — the distilled pass@k curve sits above the base model's ceiling.
- **arXiv:2505.21067:** With just 920 examples, distillation beats zero-RL. Distillation introduces "flexible reasoning" (logical connectors, multi-perspective thinking, metacognition) that RL-from-scratch fails to elicit.

**Decision for us:** Distill reasoning from a strong teacher (QwQ-32B or DeepSeek-R1) into our base, THEN apply RL on top. Never RL-from-scratch for reasoning on a small model.

### 1.4 Minimum Compute — Reasoning Works at 7B–14B

| Approach | Model | Compute | Result |
|----------|-------|---------|--------|
| SimpleRL-Zoo | Qwen2.5-7B | 2×8 H100, ~15h, 8K examples | MATH 64.6%→78.2% |
| **Tina** | R1-Distill-1.5B | **$9 (LoRA, single L40S)** | AIME 43.3% (matches Sky-T1-32B) |
| rStar-Math | Qwen2.5-Math-7B | MCTS self-evolution | MATH 58.8%→90.0% |

**Caveat (arXiv:2502.12143):** Models ≤3B don't consistently benefit from *long* CoT distillation — they need shorter chains calibrated to capacity. For our 32B target this is not a concern.

---

## 2. Does Reasoning Help Coding? (The Critical Question)

### 2.1 The Uneven Picture

| Task type | Reasoning gain | Why |
|-----------|----------------|-----|
| **Competitive coding (LiveCodeBench)** | **+82% relative** (R1 65.9% vs V3 36.2%) | Long reasoning chains explore solution spaces; rewards search |
| **Codeforces rating** | 1134 → 2029 (Expert → Candidate Master) | Same — algorithmic exploration |
| **SWE-bench patch generation** | **+7pp** (R1 49.2% vs V3 42.0%) | Localization + targeted edits; reasoning saturates fast |
| **Easy problems (GSM8K-tier)** | **Negative or zero** | Overthinking wastes tokens, can regress |

**This is the single most important finding for our plan:** Our primary target (DeepSWE, a patch-generation/SWE benchmark) is in the *modest gain* category. Reasoning helps, but it's a +7pp lever, not a +30pp lever. Competitive-coding-style components (if any) benefit far more.

### 2.2 Reasoning Length for SWE — Saturates at ~2K Tokens

| Thinking budget | SWE-bench pass@1 |
|-----------------|------------------|
| 0 tokens | 40.6% |
| 2K tokens | 43.8% (+3.2pp) |
| 8K tokens | 44.2% (+0.4pp) |

Confirmed by Claude 4 Sonnet budget sweep: 8K→16K→32K = 67.3%→68.4%→68.7%. **Most gains are in the 0→2K range.** Long CoT (>10K tokens) shows an **inverted-U** — performance peaks then degrades (arXiv:2502.07266). More capable models need *shorter* chains (72B optimal ≈ 4 steps; 1.5B optimal ≈ 14 steps).

### 2.3 Where to Place Reasoning Matters More Than How Much

**"Think Anywhere" (arXiv:2603.29957):** Inject reasoning at high-entropy positions *within* code generation (assignment + return statements) rather than all upfront. Result: **+9.3pp avg** across coding benchmarks with **23% fewer tokens** than upfront GRPO reasoning. LeetCode 50.6%→69.4%.

### 2.4 Reasoning vs. Agent Loops — They Compete During Training

**Critical finding (DART, arXiv:2602.00994):** Joint RL on both reasoning AND tool-use causes **misaligned gradients** — they compete for the same parameters, degrading both. Solution: separate LoRA modules for reasoning vs. tool-use (+6.35% avg, approaches 2-agent upper bound).

**Thinking compression during coding RL (SRPO/MicroCoder-GRPO, arXiv:2603.07777):** "Math problems increase output length; coding problems decrease it" during GRPO. Coding RL naturally compresses reasoning because correct outcomes are achievable with short solutions. Counter with: conditional truncation masking, process rewards (Posterior-GRPO: +18.1% relative on LiveCodeBench).

**Agentic overthinking (arXiv:2502.08235):** On 4,018 SWE-bench trajectories — "Analysis Paralysis" (reasoning without acting), "Rogue Actions," "Premature Disengagement." Selecting low-overthinking trajectories: **+30% performance, -43% compute.**

---

## 3. Test-Time Compute Scaling (TTS)

### 3.1 The Foundational Result (arXiv:2408.03314, DeepMind/Berkeley)

- **Compute-optimal adaptive TTS** (allocate strategy per problem difficulty) beats uniform best-of-N by **4× compute efficiency** and beats a **14× larger model** at matched FLOPs — but only on difficulty bins 1–3.
- **Hard problems (bins 4–5) still need base capability** — search cannot manufacture reasoning that isn't there.
- **Strategy by difficulty:** Easy → sequential revisions. Hard → beam search with a good PRM. Best-of-N is stable but inefficient.

### 3.2 TTS for Coding Specifically (2025–2026)

| System | Method | Result |
|--------|--------|--------|
| **CodeMonkeys** (Stanford) | 10 parallel (edit,test) × 8 serial iterations, test-based selection | SWE-bench 57.4%, oracle ceiling 69.8%, ~$2,300 |
| **S\*** | Parallel + sequential + execution-grounded pairwise selection | GPT-4o-mini beats o1-preview; 3B reaches GPT-4o-mini parity |
| **SWE-Reasoner** (arXiv:2503.23803) | Internal CoT (localize→patch→verify) + external PRM/ORM | 32B: 37.6%→**46.0%** with TTS budget=8; beats o1 (45.6%) |
| **GenCluster** | Generate + behavioral cluster + rank | First IOI **gold** with open-weight model (gpt-oss-120b) |

**Coding is ideal for TTS** because execution/tests provide a binary verifier — the equivalent of Lean's formal verification for AlphaProof.

### 3.3 The Two Binding Constraints

1. **PRM quality is the ceiling.** rStar-Math: "Reward model is the dominant factor in System 2 reasoning." Bad verifiers (perplexity-scored) make search *degrade* (beam search: -5.4pp with perplexity, +8.9pp with PRM — arXiv:2603.15377). PRM false positives impose an asymptotic ceiling on best-of-N.
2. **Selection is underrated.** CodeMonkeys: random→oracle selection gap is **12.4pp** — larger than most model-quality improvements. S*'s execution-grounded pairwise selection is what lets weak models beat strong ones.

### 3.4 MCTS vs Beam Search vs Best-of-N

- **Best-of-N:** Simple, stable, log-linear scaling, saturates early on easy tasks. Plateau ~N=10-25 on HumanEval, much later on MATH.
- **Beam search:** Wins on *hard* problems with a *good* PRM; degrades on easy problems (over-optimization) and with noisy scorers (reward inversion: useful pool size n̂ ≈ 2 for perplexity, ≥4 for good PRM).
- **MCTS:** Highest accuracy with good value function, highest cost. rStar-Math (MCTS + PPM) is the strongest small-model math result (7B → MATH 90%). For SWE, the multi-stage pipeline (repo→localize→patch) benefits more from PRM than verifying just final patches.

---

## 4. Reasoning Efficiency — Avoid Overthinking

### 4.1 The Overthinking Problem

- **arXiv:2412.21187:** o1-like models produce 2-4 redundant solution rounds; easy problems disproportionately trigger elaboration.
- **arXiv:2505.00127:** Incorrect GSM8K answers averaged 3,069 tokens vs 1,375 for correct (-55%). Pearson r = -0.68 to -0.72 between length and correctness on MATH. Over 60% of questions had the correct answer in the *shortest* sampled response.

### 4.2 Budget Control Spectrum

| Method | Mechanism | Result |
|--------|-----------|--------|
| **s1 budget forcing** | Suppress `</think>`, append "Wait" (zero training) | AIME 50%→57% |
| **L1 (LCPO)** | RL-trained length adherence | 1.5B beats GPT-4o at equal length |
| **SelfBudgeter** | Self-estimate budget, budget-guided GRPO | 61% length reduction |
| **TOPS** | Train on low/med/high effort, pick shortest correct | AIME 46% at fewer tokens |
| **AdaptThink** | RL choose think/no-think per problem | -53% tokens, +2.4% acc; picks NoThink 86.9% on easy |

**Deep-thinking tokens (arXiv:2602.13517):** Reasoning *depth* (sustained revision in deep layers) correlates with accuracy (r=+0.68); token *length* anti-correlates (r=-0.59). Optimize depth, not length. "Think@n" selects high-depth samples → same accuracy at 50% compute.

---

## 5. Updated Plan — Reasoning Integration

### 5.1 Where Reasoning Fits in Our Phased Plan

| Phase | Reasoning action | Expected impact |
|-------|------------------|-----------------|
| **Phase 1 (SFT)** | Add a **reasoning distillation stage**: SFT base on 10-17K QwQ-32B/R1 coding+math traces with `<think>` format (Sky-T1 recipe, ~$450) | Sky-T1-class reasoning baseline before RL |
| **Phase 2 (RL)** | Use **separate LoRA for reasoning vs tool-use** (DART) to avoid gradient conflict. Add **process rewards** (Posterior-GRPO) to counter coding thinking-compression. Apply **conditional truncation masking** (MicroCoder-GRPO) | +18% relative on LiveCodeBench-style; preserves reasoning |
| **Phase 3 (TTS)** | This is where reasoning pays off most. Train a **PRM** (the binding constraint). Use **execution-grounded selection** (S*/CodeMonkeys). Adaptive budget per difficulty (arXiv:2408.03314) | +8-10pp on SWE (SWE-Reasoner pattern: 37.6%→46.0%) |
| **Phase 4** | Adaptive thinking budget (AdaptThink/TOPS) for production efficiency; "Think Anywhere" inline reasoning | -50% tokens at same accuracy |

### 5.2 Concrete Recipe Additions

**Reasoning distillation (new Phase 1 sub-stage):**
```
1. Generate 10-17K coding+math reasoning traces using QwQ-32B or DeepSeek-R1 as teacher
   - Rejection-sample for correctness (keep only passing solutions)
   - Domains: APPS/TACO (coding), NuminaMath (math), R2E-Gym tasks (SWE)
   - Format: <think>...</think> + readable summary
   - NOTE: For competitive coding, do NOT over-filter — OpenCodeReasoning found
     execution filtering HURT accuracy; instruction+solution diversity matters more
2. SFT Qwen3-32B for 3 epochs, lr=1e-5, batch=96 (Sky-T1 hyperparams)
3. Cost: ~$450 on 8×H100, 19 hours
```

**Reasoning-aware RL (Phase 2 modification):**
- Separate LoRA adapters: one for reasoning tokens, one for tool-use/action tokens (DART)
- Process reward on reasoning traces conditioned on task success (Posterior-GRPO)
- Conditional truncation masking to preserve long-output capability (MicroCoder-GRPO)
- Keep Dr. GRPO / REINFORCE++ as base algorithm (from NEWER-TECHNIQUES.md)

**TTS with reasoning (Phase 3 — highest ROI for reasoning):**
- Multi-stage internal reasoning: repo-understanding → fault-localization → patch → verify (SWE-Reasoner)
- Train a PRM scoring intermediate stages (this is the ceiling — invest here)
- Execution-grounded pairwise selection between candidates (S*)
- Adaptive compute: predict difficulty, allocate budget per problem (arXiv:2408.03314)
- Budget cap at ~2K reasoning tokens for patch generation (saturation point)

### 5.3 What NOT to Do (Validated Anti-Patterns)

| Anti-pattern | Why | Source |
|--------------|-----|--------|
| RL reasoning from scratch on our base | Doesn't expand ceiling, only amplifies | arXiv:2504.13837 |
| Long CoT (>2K tokens) for SWE patches | Saturates at 2K, can regress | patch budget study |
| Joint reasoning+tool-use RL without separation | Gradient conflict degrades both | DART 2602.00994 |
| Naive best-of-N with weak verifier | Search degrades without good PRM | arXiv:2603.15377 |
| Over-filtering competitive-coding SFT data | Hurts accuracy; diversity > correctness | OpenCodeReasoning |
| Uniform thinking budget on all problems | Overthinks easy, underthinks hard | arXiv:2412.21187 |

---

## 6. Open-Source Reasoning Assets

| Asset | Repo / ID | Use |
|-------|-----------|-----|
| Sky-T1 recipe + data | NovaSky-AI/SkyThought | Reasoning distillation template |
| s1K dataset | 1K curated traces (Stanford) | Minimal high-quality reasoning SFT |
| OpenThoughts3 | 1.2M traces (QwQ teacher) | Larger reasoning corpus |
| OpenCodeReasoning (OCR2) | 736K coding traces (NVIDIA) | Coding-specific reasoning SFT |
| NuminaMath | ~860K math problems+CoT | Math reasoning corpus |
| R1-Distill models | deepseek-ai/DeepSeek-R1-Distill-Qwen-{7,14,32}B | Distillation targets / baselines |
| QwQ-32B | Qwen/QwQ-32B | Teacher for trace generation (Apache 2.0) |
| OpenR / Open-Reasoner-Zero | open-source o1 replication frameworks | RL reasoning training |
| Math-Shepherd / Qwen2.5-Math-PRM | PRM training method + models | PRM for TTS |

---

## 7. Key Papers Reference Table

| Paper | arXiv | Date | Contribution |
|-------|-------|------|-------------|
| DeepSeek-R1 | 2501.12948 | Jan 2025 | 4-stage recipe, GRPO, cold-start, distillation |
| Sky-T1 | (NovaSky blog) | Jan 2025 | $450 reasoning distillation, 17K traces |
| s1: Simple TTS | 2501.19393 | Jan 2025 | 1K traces + budget forcing |
| Does RL Incentivize Reasoning? | 2504.13837 | Apr 2025 | RL amplifies, distillation expands ceiling |
| Why Distillation > Zero-RL | 2505.21067 | May 2025 | 920 examples beats RL-from-scratch |
| Scaling TTS Optimally | 2408.03314 | Aug 2024 | Adaptive TTS beats 14× larger model |
| rStar-Math | 2501.04519 | Jan 2025 | MCTS+PPM, 7B→90% MATH |
| CodeMonkeys | 2501.14723 | Jan 2025 | SWE TTS, selection is 12.4pp lever |
| S* | 2502.14382 | Feb 2025 | Execution-grounded selection for coding |
| SWE-Reasoner | 2503.23803 | Mar 2025 | Internal+external TTC, 32B→46% SWE |
| Think Anywhere | 2603.29957 | Mar 2026 | Inline reasoning at entropy bottlenecks, +9.3pp |
| DART | 2602.00994 | Feb 2026 | Reasoning/tool-use gradient conflict, separate LoRA |
| MicroCoder-GRPO/SRPO | 2603.07777 | Mar 2026 | Coding thinking-compression fixes, +17.6% |
| Posterior-GRPO | 2508.05170 | Aug 2025 | Process rewards for code, +18.1% relative |
| AdaptThink | 2505.13417 | May 2025 | Learn when to think, -53% tokens |
| Danger of Overthinking | 2502.08235 | Feb 2025 | Agentic overthinking, +30% via selection |
| Deep-Thinking Tokens | 2602.13517 | Feb 2026 | Optimize depth not length |
| Illusion of Thinking (Apple) | 2506.06941 | Jun 2025 | 3-regime complexity; LRM collapse |
| OpenThoughts | 2506.04178 | Jun 2025 | Teacher quality is #1 factor |
| QwQ-32B | (Qwen blog) | Mar 2025 | 2-stage RL, 32B matches R1 |
| Math-Shepherd | 2312.08935 | Dec 2023 | MC-estimated PRM, no human labels |
| Open-Reasoner-Zero | 2503.24290 | Mar 2025 | Vanilla PPO, no KL, 1/10 R1-Zero steps |
| TOPS | 2502.18080 | Feb 2025 | Thinking-optimal scaling, 1.3K seeds |

---

## 8. The One-Paragraph Summary

For our DeepSWE-targeted coding LLM, reasoning is a **supporting lever, not the main one**. The biggest SWE wins come from RL on executable environments and verifier/selection quality at test time (covered in VALIDATED-PLAN and NEWER-TECHNIQUES). Reasoning adds ~7pp on patch tasks and far more on competitive-coding subtasks. The right play: **(1)** cheaply distill reasoning into our base via Sky-T1-style SFT on 10-17K teacher traces (~$450) before RL; **(2)** during RL, separate reasoning from tool-use gradients (DART) and use process rewards to prevent coding's natural thinking-compression; **(3)** invest most reasoning effort in Phase-3 test-time scaling, where a good PRM + execution-grounded selection turns a 32B model from 37.6%→46% on SWE — because PRM quality and selection, not raw sampling, are the binding constraints. Cap patch-time thinking at ~2K tokens (returns saturate) and use adaptive budgeting to avoid overthinking easy problems.
