# Build a Coding LLM From Scratch — Research-to-Implementation Plan

**Date:** 2026-06-06
**Goal:** implement our own coding-oriented language models from scratch, with a realistic path from research through MVP.

## 1) Why this plan exists
We already mapped state-of-the-art coding papers. The missing piece is now execution: selecting architecture, training budget, post-training strategy, and evaluation stack to build an **independent model family** rather than only adapting closed APIs.

### 1a) Immediate start plan (you said we have not started building yet)

Use this as the first move (48-hour baseline):

1. Choose implementation track
   - Track A (fastest): API-first benchmarking to harden agent behavior and evaluate against DeepSWE before large pretraining.
   - Track B: in-house model-from-scratch track.
   - Recommendation: start Track A, then migrate proven improvements into Track B.

2. Define first comparison pair and baseline
   - Pick one external strong model + one smaller open-weight baseline for internal reproducibility.
   - Keep this decision and all changes in `tooling/deepswe/model_profile.env` and run logs.

3. Set DeepSWE infra ready
   - `git clone https://github.com/datacurve-ai/deep-swe`
   - `uv tool install datacurve-pier`
   - `cp tooling/deepswe/model_profile_template.env tooling/deepswe/model_profile.env`
   - Fill `DEEPSWE_MODEL`, `DEEPSWE_PIER_ARGS`, provider key, and optional `DEEPSWE_AGENT` values in the profile.

4. Run first deterministic smoke
   - `DEEPSWE_N_TASKS=10 DEEPSWE_SAMPLE_SEED=0 ./tooling/deepswe/run_deepswe.sh --tasks-dir /path/to/deep-swe/tasks`
   - `cat ./artifacts/deepswe/runs/<run-id>/summary.md`
   - Repeat seeds `1` and `2` only after seed `0` is stable.

5. Lock a stable policy candidate
   - Freeze prompt/decision template, `max_steps`, retries, and timeout multipliers after 3 reproducible smoke runs.
   - Then run:
     - `./tooling/deepswe/run_deepswe_protocol.sh --tasks-dir /path/to/deep-swe/tasks --smoke-seeds "0 1 2" --smoke-target 0.30 --smoke-sustained 0.30 --smoke-require-runs 3`
     - `python3 tooling/deepswe/check_deepswe_gates.py --csv ./artifacts/deepswe/experiment_log.csv --n-tasks 10 --target-success 0.30 --sustained-target 0.30 --require-runs 3`

6. Only after the gate holds, expand to 113-task protocol and decoding-grid tuning.

### 1b) Zero-to-candidate bootstrap when you have no dataset yet

If no training corpus exists today, use this order:

1. Stand up a deterministic DeepSWE gate profile in `tooling/deepswe/model_profile.env` first.
2. Run the smoke and smoke-loop commands in section 1a until you have stable execution behavior.
3. Use `run_deepswe_zero_to_pass.sh` for the first 5 iterations with only policy knobs and no DeepSWE training data:
   - keep `--allow-training-on-deepswe` off
   - keep an external training dataset separate from DeepSWE task IDs
   - set `--build-failure-dataset` only after you explicitly permit data-source override
4. Once you have a clean external corpus (or a chosen public coding dataset), wire `--train-command` to your SFT/continual stage and keep leak checks on (default) before each iteration.

This keeps DeepSWE as a strict holdout even at day zero while you still move on policy and dataset bootstrapping.

This document gives a path to build an in-house coding model with two operating modes:

- **Research track:** maximize capability with full pretraining
- **Product track:** fastest time-to-value with modular checkpoints and iterative hardening

## 2) Core research anchors (for “from scratch” implementation)

### A) Foundational modeling papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): Transformer decoder-only architecture baseline.
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556): the modern token/data/size tradeoff baseline for pretraining planning.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361): still useful for sanity checks and extrapolation.
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053): reference for large-scale pretraining parallelism.
- [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509): sequence parallelism for long-context training efficiency.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691): core attention efficiency primitive for fast prefill/decode.

### B) Coding model benchmarks and baselines to compare against

- [DeepSeek-Coder (V1)](https://arxiv.org/abs/2401.14196): 16k context + project-level data + FIM for edits.
- [StarCoder2 and The Stack v2](https://arxiv.org/abs/2402.19173): strong open-vs-open and data transparency reference.
- [Code Llama](https://arxiv.org/abs/2308.12950): public code-adapted instruction/FIM baseline.
- [Qwen2.5-Coder technical report](https://arxiv.org/abs/2409.12186): scalable practical baseline for instruction/code tradeoffs.
- [DeepSeek-Coder-V2](https://arxiv.org/abs/2406.11931): open MoE coding benchmark point.
- [OpenCoder](https://arxiv.org/abs/2411.04905): open reproducibility/recipe-first implementation philosophy.
- [OpenCodeInstruct](https://arxiv.org/abs/2504.04030): instruction-tuning dataset template with test/execution feedback.
- [SWE-agent](https://arxiv.org/abs/2405.15793), [OpenHands](https://arxiv.org/abs/2407.16741): for agentic runtime design.

### C) Post-training and behavior papers for coding use

- [Toolformer](https://arxiv.org/abs/2302.04761): tool-call priors from weak supervision.
- [ReAct](https://arxiv.org/abs/2210.03629): reasoning/action trace format.
- [InstructGPT (RLHF)](https://arxiv.org/abs/2203.02155): practical SFT + RM + PPO flow that still informs instruction behavior.

### D) Evaluation papers for capability and realism

- [SWE-bench](https://arxiv.org/abs/2310.06770)
- [SWE-bench++](https://arxiv.org/abs/2512.17419)
- [LiveCodeBench](https://arxiv.org/abs/2403.07974)
- [SWE-context/SWE-chat](https://arxiv.org/abs/2602.08316), [SWE-MERA](https://arxiv.org/abs/2507.11059), [SWE-Bench-CL](https://arxiv.org/abs/2507.00014)
- [mHumanEval](https://arxiv.org/abs/2410.15037), [HumanEval-XL](https://arxiv.org/abs/2402.16694), [CRUXEval](https://arxiv.org/abs/2401.03065)
- [SWE-WebDevBench](https://arxiv.org/abs/2605.04637), [SecureAgentBench](https://arxiv.org/abs/2509.22097), [CoderEval](https://arxiv.org/abs/2302.00288)

## 3) Define the exact product target before training

Before a single GPU hour is spent, define:

- **Languages covered:** e.g., Python + JS + Java + Go, or multilingual.
- **Target contexts:** 16k/64k/256k (first meaningful milestone is usually 16k-32k for coding assistants).
- **Inference profile:** completion + edit + repair + agent tool-use.
- **Budget:** hardware availability and cost ceiling (can define small/medium/large model track).
- **Licensing policy:** only permissive source code and strict license metadata.

## 4) Recommended architecture for first-from-scratch stack

### Minimal but production-usable baseline

1. **Model family:** decoder-only transformer with rotary/ALiBi-like positional handling and grouped-query attention.
2. **Size:** 3B–7B for first in-house feasibility, with optional 13B/34B after baseline passes.
3. **Context:** start at 16k tokens.
4. **Objective:** next-token prediction + fill-in-the-middle task mix.
5. **Tokenizer:** byte-level BPE/Unigram tuned on combined NL/code corpus.

### Practical progression

- Phase A: train from scratch at 3B/7B (single language-family focus).
- Phase B: add multilingual support and larger context.
- Phase C: add MoE/experts for domain-specialized coding capabilities.

### Why this is realistic

Research indicates that strong data and post-training are often more cost-effective than simply scaling blindly. Even large frontier systems are now separating pretraining specialization from post-training behavior modules, which is favorable for in-house implementation.

## 5) Data plan (critical, usually 50% of outcomes)

1. **Raw sources**
   - GitHub/public code, docs, issue/PR traces, and test suites.
   - Competitive-programming and reasoning sets for synthetic pressure testing.
2. **Deduplication and filtering**
   - remove boilerplate, binaries, generated duplicate snippets, license conflicts.
   - project-aware chunking to preserve imports/references.
3. **Split strategy**
   - train/val/test by repository/time to reduce contamination.
   - strict non-overlap for SWE-style tasks and hidden evaluation set.
4. **Curriculum**
   - start with clean general code + docs, then inject high-value bugfix/patch traces.

## 6) Training pipeline (from-scratch)

### 6.1 Pretraining

- Optimizer: AdamW/Adam with modern hyperparameter recipes.
- Max sequence: 16k early, then progressively increase if loss vs cost supports it.
- Sequence-aware batching (by token count).
- Use ZeRO/3D parallelism to fit memory and maintain throughput.

### 6.2 Post-training stack

- SFT on curated coding/repair datasets.
- Instruction fine-tuning for: explain, edit, write tests, diagnose, propose patches.
- RLHF/RLAIF-style preference alignment (initially simple with expert + auto-labelers, upgrade to model-assisted preference when volume grows).
- Tool-format alignment: JSON schema, deterministic action tags, validation harness for invalid calls.

### 6.3 Safety and execution gates

- Static and run-time sanitizer checks for generated code.
- Static-analysis + unit test validation on generated patches.
- Safety policy for filesystem/network command generation.

## 7) Evaluation loop (mandatory from day 1)

Run every training gate with fixed test gates:

- **Static correctness:** HumanEval, MBPP, mHumanEval.
- **Execution correctness:** LiveCodeBench-style slices, CoderEval.
- **Repo-level:** SWE-bench / SWE-bench++ / SWE-Context.
- **Agentic:** SWE-agent-like harness and SWE-MERA/SWE-Bench-CL style dynamic checks.
- **Security/prod quality:** SecureAgentBench + mutation tests + vulnerability lints.

Track:
- pass@k, repair success rate, test pass delta, execution time, rollback/call failure ratio, token usage per successful patch.

## 8) Engineering stack (what to build in parallel)

- **Training:** PyTorch + Triton + Accelerate/DeepSpeed or Megatron path for parallelism.
- **Fine-tuning:** LLaMA-Factory/PEFT-friendly workflow for early loops.
- **Serving:** vLLM for high-throughput generation (PagedAttention, continuous batching), plus fallback batching for heavy tool workflows.
- **Data & eval infrastructure:** reproducible dataset manifests + immutable versioned artifacts.

## 9) 16-week roadmap (concrete)

### Weeks 1–2
- Finalize architecture target and evaluation definitions.
- Assemble 2–4TB clean training corpus with repository + tests lineage metadata.

### Weeks 3–6
- Pretraining pilot at chosen baseline size (3B/7B).
- Build tokenizer + baseline trainer + checkpointing/eval.

### Weeks 7–10
- Add FIM and edit-style objective mix.
- Add instruction fine-tuning and JSON/tool-call traces.

### Weeks 11–14
- RL alignment phase + bug-repair traces.
- Evaluate on SWE-bench(-lite) and LiveCodeBench slices.

### Weeks 15–16
- Hardening: security policy layer, retrieval integration, fallback strategy.
- Decide on Stage-2 scaling (MoE upgrade or longer context).

## 10) Risks and “what to avoid”

- **Compute underestimation:** frontier-scale claims are often unrealistic without serious budget.
- **Data contamination:** static benchmark leakage invalidates results fast.
- **Context mismatch:** training context too long/too short for your actual agent workload.
- **No tool grammar validation:** produces syntactically valid code but non-executable workflows.

## 11) Research counter-check ledger (evidence quality by claim)

This section flags what is strongly source-confirmed vs inferred so plan items can be de-risked before execution.

- **Strongly confirmed:**
  - Use of **ZeRO** for memory/throughput scaling is still a core recommendation; ZeRO explicitly targets memory redundancy in data/model parallel training and is shown to enable trillion-parameter-scale training patterns. ([ZeRO paper](https://arxiv.org/abs/1910.02054))
  - **DeepSpeed-Ulysses** targets long-sequence LLMs via sequence partitioning and all-to-all attention communication; abstract reports constant communication volume under proportional scaling and faster long-context training versus previous baselines. ([DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509))
  - **vLLM** inference is centered on PagedAttention and continuous batching; NVIDIA docs and vLLM docs explicitly state these are core. ([vLLM overview](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/overview.html), [vLLM API page](https://docs.vllm.ai/en/v0.10.2/api/vllm/attention/ops/paged_attn.html))
  - **SWE-bench++** is repository/PR-based and multilingual with 11,133 instances from 3,971 repos; benchmark has four pipeline stages (sourcing, environment synthesis, oracle extraction, quality assurance). ([SWE-bench++](https://arxiv.org/abs/2512.17419))
  - **SWE-context** has base + related-task sequences from GitHub issue/PR context and evaluates context reuse/ retrieval behavior. ([SWE Context Bench](https://arxiv.org/abs/2602.08316))
  - **SWE-chat** is explicitly a real-user coding-agent session corpus with interaction metrics (not just curated tasks). ([SWE-chat](https://arxiv.org/abs/2604.20779))
  - **SWE-Bench-CL** evaluates continual learning on chronologically ordered issue streams to measure retention/transfer. ([SWE-Bench-CL](https://arxiv.org/abs/2507.00014))
  - **OpenCodeInstruct** claims 5M samples with question/solution/test-cases/execution feedback and explicitly positions it as SFT dataset support. ([OpenCodeInstruct](https://arxiv.org/abs/2504.04030))
  - **SWE-Perf** extends evaluation into repository-level performance optimization with PR-derived curated tasks and executable environments. ([SWE-Perf](https://arxiv.org/abs/2507.12415))

- **Likely but needs in-house validation before hard commitments:**
  - "16k-32k first milestone" and exact 3B–7B phase sizing are planning assumptions, not direct claims from the cited papers.
  - "Strong data + post-training beats scale alone" is a strong engineering heuristic; treat as hypothesis and verify by ablation (not one paper axiom).
  - vLLM is a strong baseline, but **vAttention** is an actively published alternative with different tradeoffs; test both if concurrency/latency patterns justify it. ([vAttention](https://arxiv.org/abs/2405.04437))

## 12) Counter-check additions for coverage gaps

- Add contamination-resistant long-horizon tracking:
  - **SWE-Bench Pro** for enterprise/long-horizon issue sets and held-out/restricted repositories. ([SWE-Bench Pro](https://arxiv.org/abs/2509.16941))
  - **SWE-Bench Illusion** for memorization/contamination diagnostics on SWE tasks. ([SWE-Bench Illusion](https://arxiv.org/abs/2506.12286))
- Add performance-oriented software-agent evaluation:
  - **SWE-Perf** for optimization-aware benchmarks before shipping into production performance workflows. ([SWE-Perf](https://arxiv.org/abs/2507.12415))
- Add agent behavior diagnostics (if tool-use and handoff quality are core):
  - **SWE-MERA** for dynamic, continuously updated contamination-reduced agent tasks. ([SWE-MERA](https://arxiv.org/abs/2507.11059))
- Add architectural alternatives for long-context serving:
  - benchmark **PagedAttention** vs alternatives (including vAttention) under your exact mixed prompt/long-patch workload, instead of fixing one serving stack up-front. ([vAttention](https://arxiv.org/abs/2405.04437))

## 13) “Ready-to-start” decisions (next 24 hours)

1. Pick exact model size and context target.
2. Freeze the top-3 languages and data sources.
3. Approve legal/license matrix for all source corpora.
4. Create the first corpus pipeline and token budget forecast with Chinchilla-style compute-optimality checks.
5. Start with a 3B/7B scratch pretraining pilot and fixed SWE/LiveCodeBench smoke tests.

## 14) Reading stack to keep open during build

- DeepMind/DeepSeek/Meta papers listed in Section 2.
- DeepSpeed and Megatron docs for scaling operations.
- vLLM docs for serving scale.
- Tool-use papers (ReAct/Toolformer) for runtime alignment.
- SWE-family benchmarks, especially contamination-aware and long-horizon variants.

## 15) DeepSWE pass strategy (practical objective: DeepSWE-competent model + agent)

### DeepSWE contract (authoritative constraints)
- DeepSWE has **113 tasks** across TypeScript, Go, Python, JavaScript, and Rust in its main public benchmark.
- Task format is Harbor-like with `task.toml`, `instruction.md`, `environment/`, `tests/`, `solution/`.
- Verifiers are executable behavior tests (`tests/test.sh`), not patch-diff matching.
- Agent time budget is typically long (example task: `agent.timeout_sec = 5400`, verifier timeout `1800`); resources are constrained and isolated.
- Many tasks enforce `allow_internet = false`; runners often use fixed docker images.
- Reproducible local validation is provided via `pier run -n-tasks` and deterministic seeds.

### Baseline and open research signals
- Public DeepSWE results currently place `deepseek-v4-pro` near 8% pass@1 in the latest public slice, while frontier models are materially higher.
- `mini-swe-agent` is the public benchmark scaffold used by the published DeepSWE runs.
- Open-research direction for open-weight agents points to:
  - SWE-agent style interfaces that materially improve agent behavior on SE tasks.
  - OpenHands-style sandboxed, multimodal developer workflows.
  - R2E-Gym style environment synthesis plus hybrid inference-time scaling (execution + execution-free verifier).
  - `DeepSWE-Preview` as an example of RLAIF/RL-driven SWE-agent optimization.

### Implementation plan to move from “research prototype” to “DeepSWE pass behavior”

1) **Phase 0 — reproduce infra (week 1)**
- Clone DeepSWE and install pier.
- Run the deterministic 10-task sample with a strong external model to validate benchmark harness and metric parsing.
- Add deterministic logging for each run: task id, seed, timeout, reward, cost, pass path.

2) **Phase 1 — strengthen base policy (weeks 2-4)**
- Use a fixed scaffold and run 10-task subsets with increasing complexity.
- Prioritize patch-iteration behavior: run baseline test first, generate patch, run tests, then apply self-fix loop.
- Optimize prompt discipline:
  - explicit “inspect, locate root cause, implement minimal patch, validate, summarize risk” contract.
  - require deterministic command traces and file-level edit summaries.

3) **Phase 2 — open-weight agent stack (weeks 5-8)**
- Build training traces on large executable environments (R2E-style if available).
- Add verifier-aware training (behavior reward, not text similarity).
- Evaluate two versions:
  - shell-only agent loop (mini style),
  - function/tool-assisted loop (file edit + test + search + finish actions).
- Select winner by DeepSWE subset pass@1, not by static metrics.

4) **Phase 3 — inference-time scaling (weeks 9-12)**
- Add sample-at-k patch generation with reranking.
- Add lightweight execution-free verifier pass to filter bad candidates before full test execution.
- Evaluate cost/quality frontier; tune temperature and max steps for cost-efficiency.

5) **Phase 4 — full DeepSWE gate (week 13+)**
- Run full 113-task suite in repeated batches with identical seeds.
- Track pass@1 plus confidence interval and flake rate.
- Lock version only if reproducibility and stability are stable at target threshold.

### Acceptance threshold recommendations
- **Near-term target**: surpass `deepseek-v4-pro` on DeepSWE pass@1 on deterministic subsets.
- **Target milestone**: reach and sustain `>=30%` pass@1 on randomized 10-task subsets over three independent runs.
- **Competitive milestone**: sustained `>=50%` pass@1 on full deterministic slices with controlled variance before any productionization.
- These are practical, auditable progression gates before scaling compute.

### Next concrete actions (next 48h)
1. Add a DeepSWE tracker to the existing plan with:
   - baseline, policy version, seed, timeout, n-tasks, and pass@1.
2. Add reproducible run script for:
   - single-task smoke,
   - 10-task deterministic subset,
   - full-run wrapper.
3. Add automatic failure taxonomy extraction from verifier logs (dependency failure, timeout, assertion failure, non-deterministic flake, no-op).

### Benchmark infra added for immediate execution
- Added `tooling/deepswe/run_deepswe.sh` to run deterministic/reproducible DeepSWE slices using `pier`.
- Added `tooling/deepswe/parse_deepswe_results.py` to summarize pass/fail/unknown outcomes plus failure-class aggregation.
- Added `tooling/deepswe/README.md` with 48-hour command sequence for smoke and full-suite loops.
- Added `tooling/deepswe/append_deepswe_log.py` and updated run script to append structured experiment rows automatically (`run_id`, pass@1, fail taxonomy, and run metadata) after each benchmark run.

To use immediately:
- Set `DEEPSWE_LOG_DIR` and `DEEPSWE_AGENT` for your environment.
- Run `./tooling/deepswe/run_deepswe.sh --tasks-dir <deep-swe/tasks>` with `DEEPSWE_N_TASKS=10` first.
- Consume `summary.md` per run and track the same tuple in your experiment log:
  `(model_or_ckpt, policy_version, seed, n_tasks, pass@1, timeout_policy, token_usage_estimate)`.

### 30-day DeepSWE execution protocol

- Week 1: infra reproducibility
  - Run deterministic 10-task slice with seeds `0`, `1`, `2`.
  - Track `run_dir` for each run and triage recurring task-level failures with the failure-summarizer before model/prompt changes.
  - Verify same task ordering and stable parsing output across runs.
  - Confirm `experiment_log.csv` has one row per run with pass/fail split.

- Week 2: failure triage loop
  - For any failing task repeated >1x, run `run_deepswe_single_task.sh` and capture root cause.
  - Group fail types: timeout, dependency, permission, flake, build/syntax, network, other.
  - Freeze scaffold for 3 consecutive runs before changing prompt/tool policy.

- Weeks 3–4: controlled optimization
  - Run randomized 10-task subsets (at least 2 seeds / run) to test generalization.
  - Only keep policy changes that improve median pass@1 and reduce `unknown`.
  - Continue until target `>=30%` pass@1 on randomized 10-task splits is reached or plateaued.
  - Use `tooling/deepswe/check_deepswe_gates.py` with `--n-tasks 10` and `--sustained-target 0.30` to validate sustained gains and failure-rate floors.

- Weeks 5+: full-suite gating
  - Start 113-task full slices in duplicate with fixed seeds.
  - Use `summary_metrics.json` confidence interval as acceptance gate.
  - Promote only if pass@1 and flake/timeout rates are stable over duplicates.
  - Use `tooling/deepswe/check_deepswe_gates.py --n-tasks 113 --sustained-target 0.50` to confirm promotion readiness.
