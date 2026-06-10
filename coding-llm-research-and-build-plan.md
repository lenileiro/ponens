# Deep Research: Building a State-of-the-Art Coding LLM (as of 2026-06-06)

## Objective
- Build a practical, source-first reading and design path for a state-of-the-art **coding LLM**.
- Include explicit analysis of `https://www.alphaxiv.org/abs/deepseek-v4`.
- Extend coverage beyond a single model family to compare major non-DeepSeek coding LLMs and coding-agent pipelines.
- Translate paper findings into implementation decisions: data, architecture, post-training, and evaluation.

### Expanded non-DeepSeek baseline set (for full comparison)

- [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)
- [InCoder: A Generative Model for Code Infilling and Synthesis](https://arxiv.org/abs/2204.05999)
- [DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence](https://arxiv.org/abs/2406.11931)
- [OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](https://arxiv.org/abs/2411.04905)
- [OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs](https://arxiv.org/abs/2504.04030)
- [PANGU-CODER2: Boosting Large Language Models for Code Generation](https://arxiv.org/abs/2307.14936)

### Comparison criteria

- Completion correctness, not just syntax quality.
- Edit/repair patch fidelity in context.
- Tool-use safety and deterministic behavior in agent loops.
- Repo-level long-horizon reasoning and retrieval.
- Cost/latency under deployment conditions.

## DeepSeek-V4 check (`alphaxiv.org/abs/deepseek-v4`)

- `alphaXiv abs` page is accessible for the target title and is currently a compact/interactive view.  
  URL used and verified: `https://www.alphaxiv.org/abs/deepseek-v4`.
- The technical claims are confirmed in official DeepSeek publication artifacts:
  - DeepSeek Transparency Center lists **DeepSeek-V4 New**, release date **2026-04-24** and points to the DeepSeek technical report.  
    Source: `https://www.deepseek.com/en/transparency/`
  - DeepSeek-V4 official card/open model materials report:
    - **Two models**: V4-Pro and V4-Flash
    - **Context length:** 1,000,000 tokens
    - **Core innovations:** hybrid CSA/HCA attention, manifold-constrained hyper-connections (mHC), Muon optimizer
    - **V4-Pro efficiency:** 27% single-token FLOPs and 10% KV cache vs DeepSeek-V3.2 in 1M context
    - Source: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/README.md`
- Why this matters: it is one of the first mainstream reports that simultaneously targets **large-scale coding foundation + very-long-context efficiency + MoE efficiency**, not just benchmark wins.

## Design principle summary (what current “state-of-the-art coding LLM” research implies)

1. **Data quality > parameter count for specialized coding gains**  
   Coding-specialized corpora with project/issue context and high-fidelity cleaning are still decisive.

2. **Long-context behavior is now a first-class requirement**  
   Agentic coding workflows and repository-level fixes depend on long-horizon memory and retrieval.

3. **Architecture is no longer just “bigger transformer”**  
   Papers converging on MoE routing, sparse/efficient attention, and specialized attention/normalization tricks for stability.

4. **Post-training determines production quality**  
   Inference-only improvements plateau quickly without dedicated coding/repair/SWE objectives and agent training loops.

5. **Benchmark selection must include execution, contamination control, and context reuse**  
   Static completion tests are insufficient for real coding agents.

---

## Paper map by subsystem (2024–2026 focus)

### A) Pre-training and data for coding competence

| Paper | Key claims | Why this matters for coding LLM design |
|---|---|---|
| [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814) | Uses large-scale sample generation with filtering to improve algorithmic code quality from NL prompts. | Useful historical baseline for candidate generation + test-driven filtering pipelines. |
| [InCoder: A Generative Model for Code Infilling and Synthesis](https://arxiv.org/abs/2204.05999) | Uses a code infilling objective that supports both generation and middle editing in context. | Validates edit-style objectives as a core mechanism for patch workflows. |
| [DeepSeek-Coder: When the LLM Meets Programming](https://arxiv.org/abs/2401.14196) | Trains on 2T tokens with project-level code corpus; fill-in-the-blank objective with 16k window; 1.3B–33B range. | Confirms the value of **project context + FIM objective** for edits, not only generation. |
| [StarCoder2 and The Stack v2](https://arxiv.org/abs/2402.19173) | 3.3T–4.3T code tokens, 619 languages in curated stack, + PRs/notebooks/docs; small model variants can beat larger non-specialized baselines. | Suggests strong gains from **multilingual code breadth + source attribution + quality filtering**. |
| [Code Llama](https://arxiv.org/abs/2308.12950) | Open model family with Python-specialized and instruction variants; strong FIM and infill support with 16k training contexts. | Makes a clear case for **code editing workflows**: middle infilling and structured prompts are first-class outputs. |
| [Qwen2.5-Coder technical report](https://arxiv.org/abs/2409.12186) | 5.5T pretraining tokens, 0.5B–32B variants, synthetic+cleaned mixing; performance gains across generation/completion/reasoning/repair. | Strong precedent for **scaled model series + strict data pipeline + objective mix**. |
| [rStar-Coder](https://arxiv.org/abs/2505.21297) | 418K competition-level problems + synthetic verifiable datasets with input-output checking pipeline. | Hard-code reasoning quality requires **tested generation**, not just language plausibility. |
| [OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](https://arxiv.org/abs/2411.04905) | Open-model cookbook with reproducible preprocessing and training recipes plus open multi-size checkpoints. | Strong reference if you need transparent ablation and reproducible model iteration. |
| [DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence](https://arxiv.org/abs/2406.11931) | Open-source MoE code LLM with strong performance in code intelligence benchmarks. | Non-DeepSeek-v4 control point for closed-weight parity/efficiency decisions. |

### B) Architecture and long-context scaling

| Paper | Key claims | Practical implication |
|---|---|---|
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | Economical MoE model with strong cost/performance characteristics and improved serving efficiency. | Helps isolate which gains come from sparse routing versus DeepSeek-V4-specific long-context machinery. |
| [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://www.alphaxiv.org/abs/deepseek-v4) | 1M context target, MoE, CSA/HCA, mHC, Muon. | End-to-end architecture priority for real world coding agents handling very long repository histories. |
| [StarCoder2...](https://arxiv.org/abs/2402.19173) | Demonstrates scaling from 3B→15B with broad architecture parity but improved domain-specific data design. | Architecture can be secondary if the coding corpus and objective are strong; for coding, data-first still critical. |
| [Toolformer](https://arxiv.org/abs/2302.04761) | Self-supervised tool call learning with API invocation and incorporation in token prediction. | If deploying agentic coding stack, design model to call tools under weak supervision, then harden via supervised traces. |
| [ReAct](https://arxiv.org/abs/2210.03629) | Interleaves reasoning and action in-context; improved interactive task handling. | Useful template for prompt format and training targets for agent traces. |
| [PANGU-CODER2](https://arxiv.org/abs/2307.14936) | Uses prompt and RL strategies to improve code model stability and generation quality. | Useful for lightweight quality-improvement layers on top of base checkpoints. |

### C) Post-training, alignment, and coding behavior

| Paper / system | Key claims | Practical implication |
|---|---|---|
| DeepSeek-V4 post-training details | Pretraining on 32T+ tokens, then SFT/RL-style domain cultivation and on-policy distillation (OPD-style stage). | Suggests a viable split: independent domain experts + distillation consolidation. |
| [SWE-agent](https://arxiv.org/abs/2405.15793) | Agent-computer interface materially improves SWE and HumanEvalFix task performance. | Add environment-aware action APIs early; interface decisions are model decisions by proxy. |
| [OpenHands](https://arxiv.org/abs/2407.16741) | Open platform for generalist AI software developer agents (CLI + web + sandboxed execution). | Treat your stack as **agent runtime + model**; architecture for orchestration matters as much as modeling. |
| [SWE-bench++](https://arxiv.org/abs/2512.17419) | Automated scalable repo-level benchmark generation from PRs, multilingual, 11k+ tasks, reproducible execution tasks. | For post-training, prefer this style of objective generation over static question-answer style tasks. |
| [SWE-MERA](https://arxiv.org/abs/2507.11059) | Dynamic SWE benchmark designed to reduce leakage and strengthen discrimination across recent, real-world tasks. | Useful for longitudinal measurement of true agent gains across changing issue distributions. |
| [SWE-Bench-CL](https://arxiv.org/abs/2507.00014) | Continual-learning benchmark over SWE tasks ordered by repository timeline. | Tests whether post-training can improve without catastrophic forgetting. |
| [SWE-chat](https://arxiv.org/abs/2604.20779) | Large-scale coding-agent interaction dataset with real human/agent traces and intervention points. | Essential for evaluating realism and usefulness in actual developer workflows. |
| [OpenCodeInstruct](https://arxiv.org/abs/2504.04030) | Large instruction-tuning dataset with execution feedback and curated quality controls. | A strong path for behavior shaping before expensive RL or multi-agent fine-tuning. |

### D) Evaluation framework for real coding ability (not just code completion)

| Paper | Focus | Why it is required for coding-model proof |
|---|---|---|
| [SWE-bench](https://arxiv.org/abs/2310.06770) | Repo-level bug fix tasks from real GitHub issues (2,294). | Baseline for “can edit meaningful code changes in context.” |
| [LiveCodeBench](https://arxiv.org/abs/2403.07974) | Time-robust, contamination-aware coding benchmark with execution/self-repair scope. | Prevents accidental leaderboard overfitting and stale-snapshot gaming. |
| [mHumanEval](https://arxiv.org/abs/2410.15037) | Multilingual extension of HumanEval for more language coverage and robustness checks. | Helps avoid English-only and language-specific overfitting in generation quality. |
| [HumanEval-XL](https://arxiv.org/abs/2402.16694) | Multilingual HumanEval-like benchmark for cross-lingual prompt generalization. | Useful if deployment includes multilingual developer teams or NL prompts. |
| [SWE Context Bench](https://arxiv.org/abs/2602.08316) | Context-reuse and retrieval-aware coding tasks across related issue clusters. | Essential for long-horizon agent workflows and production coding memory efficiency. |
| [Saving SWE-Bench](https://arxiv.org/abs/2510.08996) | Mutates verified-formal tasks into realistic chat/IDE interactions; shows public benchmarks can overestimate agent ability. | Your deployment-ready evaluation should include user-style interaction realism. |
| [Multi-SWE-bench](https://arxiv.org/abs/2504.02605) | Multilingual issue-resolution benchmark across common production languages. | Most coding teams are multilingual; avoid Python-only false confidence. |
| [CRUXEval](https://arxiv.org/abs/2401.03065) | Evaluates code reasoning and execution behavior with more targeted task patterns. | Flags reasoning regressions missed by pure pass@k completion checks. |
| [SecureAgentBench](https://arxiv.org/abs/2509.22097) | Measures secure code-generation quality under realistic, risk-aware coding-agent scenarios. | Required if deployment has security acceptance constraints. |
| [CoderEval](https://arxiv.org/abs/2302.00288) | Pragmatic benchmark of code generation with executable evaluation from real OSS-like prompts. | Good middle layer between HumanEval and full SWE repo tasks. |
| [SWE-WebDevBench](https://arxiv.org/abs/2605.04637) | Measures coding-agent platforms across end-to-end software delivery dimensions beyond unit tests. | Important if building app-builder assistant systems. |

## Implementation blueprint: what to build now

### 1) Data stack (foundational, highest leverage)
- Build a **multi-layer coding corpus**:
  - Layer 1: project-level source, docs, tests, PR threads.
  - Layer 2: issue resolution trajectories (candidate patches + final patches + test results).
  - Layer 3: competition-level and verified reasoning datasets (e.g., rStar-Coder style).
- Maintain strict metadata: language, repo, license, time window, test status, patch provenance.

### 2) Core model stack
- Start from an open code-specialized base if compute is constrained (e.g., Qwen2.5-Coder / Code Llama lineage style).
- If long-context and high concurrency are product requirements, design for **DeepSeek-V4 class patterns**:
  - sparse/extrinsic attention for million-token style contexts,
  - stable routing/normalization for expert models,
  - explicit quantization path (FP4/FP8-style mixed precision where feasible).

### 3) Training path (minimum viable state-of-the-art)
- Stage 1: base pretraining or continued pretraining with curated multi-language code corpora.
- Stage 2: post-training experts:
  - completion/fixing,
  - test-driven reasoning,
  - issue-context patching,
  - short-trajectory tool use.
- Stage 3: consolidation:
  - train task-specialized experts (repair, refactor, issue triage, dependency migration),
  - distill into unified model where latency/cost requires.
- Stage 4: deployment hardening:
  - robust JSON/tool-call schema,
  - deterministic parsing,
  - sandbox and safety filters before execution.

### 4) Evaluation stack for release confidence
Run four parallel harnesses every training gate:
1. **Static code quality:** HumanEval / MBPP-style completion checks.
2. **Repo-level resolution:** SWE-bench + SWE-bench++ or multilingual equivalent.
3. **Interactive agent loops:** SWE-context style and mutated chat-style tasks.
4. **Cost and reliability:** token/time/success/error taxonomies, including retry and patch rejection rates.

### 5) Architecture-level acceptance thresholds (pragmatic)
- If code-edit quality is priority: ensure FIM/edit-in-middle performance first.
- If long-horizon reliability is priority: ensure context-aware tasks remain stable above 64k/256k and retrieve useful previous fixes.
- If production constraints exist: prioritize MoE efficiency and KV cache profile over raw benchmark peak.

## Suggested 16-week research-to-MVP plan

1. **Weeks 1–4 — Corpus and baseline**
   - Assemble repo/issue/PR/Test corpus; baseline model + basic FIM and static benchmark.
2. **Weeks 5–8 — Architecture and post-training prototypes**
   - Add long-context variants; run SWE-bench and LiveCodeBench slices; add tool-action formats.
3. **Weeks 9–12 — Agent loop and context stack**
   - Build retrieval + action runtime; add Multi-SWE/Context-Bench style tests; iterate on trace quality.
4. **Weeks 13–16 — Consolidation and reproducibility**
   - Benchmark gate, failure taxonomy, eval scripts, and release notes with exact dataset/model versions.

## Key reading list (primary papers + official artifacts)
1. [DeepSeek-V4 (arXiv mirror on alphaXiv)](https://www.alphaxiv.org/abs/deepseek-v4)  
2. [DeepSeek technical report PDF (Hugging Face)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf?download=true)  
3. [DeepSeek Transparency Center](https://www.deepseek.com/en/transparency/)  
4. [DeepSeek-V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/README.md)  
5. [DeepSeek-Coder](https://arxiv.org/abs/2401.14196)  
6. [StarCoder2](https://arxiv.org/abs/2402.19173)  
7. [Code Llama](https://arxiv.org/abs/2308.12950)  
8. [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186)  
9. [rStar-Coder](https://arxiv.org/abs/2505.21297)  
10. [Toolformer](https://arxiv.org/abs/2302.04761)  
11. [ReAct](https://arxiv.org/abs/2210.03629)  
12. [SWE-agent](https://arxiv.org/abs/2405.15793)  
13. [OpenHands](https://arxiv.org/abs/2407.16741)  
14. [SWE-bench](https://arxiv.org/abs/2310.06770)  
15. [LiveCodeBench](https://arxiv.org/abs/2403.07974)  
16. [SWE-Bench++](https://arxiv.org/abs/2512.17419)  
17. [SWE Context Bench](https://arxiv.org/abs/2602.08316)  
18. [Saving SWE-Bench](https://arxiv.org/abs/2510.08996)  
19. [Multi-SWE-bench](https://arxiv.org/abs/2504.02605)  
20. [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)  
21. [InCoder: A Generative Model for Code Infilling and Synthesis](https://arxiv.org/abs/2204.05999)  
22. [DeepSeek-V2](https://arxiv.org/abs/2405.04434)  
23. [DeepSeek-Coder-V2](https://arxiv.org/abs/2406.11931)  
24. [OpenCoder](https://arxiv.org/abs/2411.04905)  
25. [OpenCodeInstruct](https://arxiv.org/abs/2504.04030)  
26. [PANGU-CODER2](https://arxiv.org/abs/2307.14936)  
27. [SWE-MERA](https://arxiv.org/abs/2507.11059)  
28. [SWE-Bench-CL](https://arxiv.org/abs/2507.00014)  
29. [SWE-chat](https://arxiv.org/abs/2604.20779)  
30. [mHumanEval](https://arxiv.org/abs/2410.15037)  
31. [HumanEval-XL](https://arxiv.org/abs/2402.16694)  
32. [CRUXEval](https://arxiv.org/abs/2401.03065)  
33. [CoderEval](https://arxiv.org/abs/2302.00288)  
34. [SecureAgentBench](https://arxiv.org/abs/2509.22097)  
35. [SWE-WebDevBench](https://arxiv.org/abs/2605.04637)
