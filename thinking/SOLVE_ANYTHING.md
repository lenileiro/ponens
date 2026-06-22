# Solve-Anything: a research direction grounded in the literature

**Goal (user's framing):** a model that solves *novel* problems by **finding the pattern/rule from the
input** — no prior knowledge of the question's shape, no reliance on recall/memorization.

This is not a toy-task framing. It is the frontier of **inductive inference**, and five parallel
literature surveys (ARC-AGI, program synthesis, universal induction, meta-learning/test-time adaptation)
**independently converged on one architecture**. This doc records that convergence and a staged,
verifiable plan that builds on assets we already have.

---

## The convergent answer

A realizable "solve-anything" learner is the **computable shadow of Solomonoff/AIXI**. The theory says it
needs exactly two things — a **simplicity prior over hypotheses** and a **search that updates on
evidence** — and that the universal version is *uncomputable*. So every real system is a bet on *which
restricted hypothesis class + which tractable search*. The four threads agree on the same bet:

1. **Hypotheses = programs** (executable, verifiable). The broadest *sound* family; makes the simplicity
   prior literal and every answer checkable. (DreamCoder, Hypothesis-Search, AZR; "code is a near-universal substrate".)
2. **Simplicity prior = description length + a LEARNED, GROWING LIBRARY of abstractions.** MDL is the
   computable surrogate for Kolmogorov complexity; a wake-sleep library (DreamCoder) makes recurring
   patterns short — the engine of *open-ended* generalization. Hand-built DSLs cap out (~20% on ARC).
3. **An amortized NEURAL PROPOSER**, meta-trained over a *distribution of fresh task-structures*, that
   proposes candidate programs from the examples — approximating the intractable posterior weighting
   (PFNs/ICL ≈ Bayes; DreamCoder recognition model; DeepCoder).
4. **A SOUND VERIFIER selects.** Non-negotiable. The checker — not model confidence — decides
   correctness, so a wrong-but-correctable proposer is still useful. (Every robust ARC method: execute
   the program on the train pairs.)
5. **PER-INSTANCE test-time search/adaptation** against the verifier — the escape hatch for genuinely
   novel structure a forward pass can't express. (Test-Time Training broke the multi-year ARC-1 stall.)
6. **A SELF-GENERATED, FRONTIER-PACED, VERIFIABLE CURRICULUM** so library + proposer co-evolve and
   novelty-handling *emerges* rather than memorizes. (MLC fresh-grammar episodes; AZR self-play; PAIRED
   regret; XLand/AdA frontier.)
7. **Honest bar: test on genuine novelty.** The ARC-AGI-1→2 cliff (frontier systems collapse to single
   digits) is the standing warning: anything that only generalizes within a tuned distribution is gaming,
   not solving. Reach is bounded — precisely — by how much of "any problem" the verifier can express.

---

## What we already have — and the missing legs

The surveys explicitly noted this direction **reinforces the repo's existing bet**:

| Component | Status in repo |
|---|---|
| Program hypothesis space | **Have** — `thinking/lang.py` (executable LOTA), the kernel |
| Sound verifier | **Have** — `datalog.py` closure + `thinking/lota_kernel.py` proof terms + BDD types (now fast) |
| Amortized neural proposer | **MISSING** — the key leg |
| Verified test-time search loop | **MISSING** |
| Library learning (growing abstractions) | **MISSING** |
| Self-generated verifiable curriculum | **MISSING** |

The toy substitution/shift experiments failed because they were a *fixed, narrow* task with no verifier
in the loop and no program search. The real mechanism is **synthesize-the-program-that-explains-the-examples,
verified** — sound by construction, and the actual meaning of "find the pattern."

---

## Staged, verifiable build plan

Each stage has a falsifiable bar; we do not advance until it's met. Built so the **verifier guarantees
soundness at every stage** (no hallucinated answers).

- **Stage 1 — Verified program synthesis from examples (no neural net yet).**
  Given input→output examples, *search* a small executable DSL for a program the verifier confirms
  reproduces ALL examples; apply to the query. Establishes the sound-search baseline (icecuber/Greenblatt
  shape, but principled). **Bar:** solves HELD-OUT program structures (never-seen rules), false=0.

- **Stage 2 — Amortized neural proposer (the missing leg).**
  Meta-train a model over a *procedurally generated distribution of programs* (fresh structure each
  episode — MLC lesson) to *propose* likely programs given the examples, guiding/replacing brute search.
  **Bar:** matches Stage-1 accuracy at a fraction of the search budget; generalizes to held-out structures.

- **Stage 3 — Library learning (DreamCoder wake-sleep).**
  Compress recurring sub-programs into new DSL primitives; retrain the proposer. **Bar:** monotonically
  rising coverage / falling description-length on a fixed held-out suite as the library grows.

- **Stage 4 — Self-generated curriculum (AZR-style self-play, zero external data).**
  The system proposes its own verifiable program-tasks at its frontier; proposer + library co-evolve.
  **Bar:** capability climbs with no human-authored tasks; verified, non-reward-hackable.

- **Stage 5 — Per-instance test-time search.**
  For hard novel instances, spend test-time compute searching/adapting against the verifier.
  **Bar:** lifts accuracy on the hardest held-out tier; ultimately an ARC-AGI-style novelty eval.

---

## Honest caveats

- This is **not** universal intelligence; the prior is our DSL and the reach is bounded by the verifier.
  We are building the *right shape* the theory demands, not claiming to solve AGI.
- Genuine novelty is hard: even frontier systems sit at single digits on ARC-AGI-2. Progress must be
  measured against *unseen task structure*, multi-seed, never a tuned distribution.
- The one non-negotiable that makes all of this sound rather than hand-wavy is the **verifier** — which
  is exactly the asset this project already treats as the trusted core.

---

# Architecture v2 — the COMPOUNDING engine (updated 2026-06-23)

After the Stage-4 self-play run (`azr.py`) **plateaued** (d1 0.97 mastered, but composition d2/d3 ≈ 0.36,
only 1 macro ever discovered, frontier stalled), a second wave of 9 literature surveys — including
**breakthrough, beyond-LLM** techniques and the **"discover a proven fact / shortcut and build on it"**
literature — converged on the precise cause and fix.

## Why the run plateaued (now over-determined)
Library/abstraction learning **harvests latent regularity; it cannot create it.** The MDL/compression
objective only fires on substructure that **recurs across solutions**. Our tasks were **uniform-random
programs** → idiosyncratic solution trees → every fragment's reuse-count ≈ 1 → library-entry cost always
exceeds the savings → **zero abstractions, by mathematical necessity** (not a tuning bug). And uniform
difficulty gives **no curriculum gradient** — solving one task makes no other reachable. *An unstructured
task distribution yields verification WITHOUT accumulation.*

## The compounding loop (what the evidence prescribes)
Capability compounds only with **a hard verifier × a reliable improvement operator × a simplicity
(reusability) pressure × a novelty (reach) pressure** — and a **structured task space**. The loop:

1. **SOLVE & LOG** over a **STRUCTURED, difficulty-graded task distribution** whose families *share
   sub-procedures* (so fragments recur) and where solving task N makes N+1 reachable (a dependency DAG).
   *This is the load-bearing change — the precondition everything else needs.*
2. **DISCOVER by compression** — periodically run an MDL / anti-unification pass (Stitch-style top-down
   branch-and-bound over solution subtrees) to extract the fragment that most reduces total description
   length across solved problems. Discovery signal = **reuse-frequency × size-saved − entry-cost**, which
   is simultaneously the finder and the overfitting guard. (NOT diversity/MI — "diversity ≠ usefulness";
   NOT raw accuracy.)
3. **VERIFY before admission** — gate each candidate abstraction through the **sound kernel/executor**:
   admit only if it type-checks / proves its spec / the executor certifies its behavior. *This is the
   lever Voyager lacks (its verifier is a fallible LLM self-check); our sound kernel turns a brittle
   self-check into a guarantee — the single hardest-to-get ingredient, which we already have.*
4. **ADD to a named, persistent, typed LIBRARY + amortize** — give the verified abstraction a name +
   signature; condition the proposer/search policy on it (DreamCoder's recognition net). A depth-2 combo
   becomes a depth-1 primitive.
5. **BUILD ON IT** — later solving composes over the enlarged primitive set; harder problems become
   reachable; their solutions feed step 2. Endogenous **curriculum**: the solver's own frontier sets
   difficulty (Polu/AlphaProof). **Novelty/irreducibility pruning** keeps the library lean (QuickSpec).

This is **ExIt's ratchet + DreamCoder/Stitch's compression-gated library + HipSpec's prove-then-assume
flywheel + AlphaProof's endogenous ladder + FunSearch's verifier-grounded discovery** — over a structured
distribution, all gated by our kernel. No single published system has all of it; our verifier puts it in reach.

## The inference substrate (beyond the transformer)
The candid beyond-LLM survey: drop KANs (hype) and "modern Hopfield" (just reframes attention). The
substrates that genuinely **discover and reuse structure** (rather than dissolve it into weights):
- **Learned term/tree REWRITING** gated by our kernel — Differentiable Tree Machine (100% OOD
  compositional generalization vs <30% for transformers) / Neuro-symbolic Rewriting. A learned
  *select → rewrite → combine* loop proposes each step; the kernel checks soundness. **Tightest fit** —
  directly extends our verified-step + LOTA-rewriting line and supplies the composition transformers lack.
- **Reasoning-as-iterative-optimization** (IRED energy-diffusion / Deep Equilibrium / recurrent-depth):
  adaptive compute *decoupled from output length* — harder instances get more steps, the **verifier**
  decides termination. Pairs with our recurrence/length-gen result.
- **The artifact is a checkable symbolic object** (symbolic-regression-as-output): the unit is a
  reusable, exactly-extrapolating term/equation, matching the kernel/`lang.py` philosophy.

## Concrete next experiment (validates the whole thesis, fixes the exact failure)
Re-run the `azr` loop with the **one change that the diagnosis demands**: a **structured task
distribution** — tasks generated by composing a small set of recurring "concept" sub-procedures at graded
depth (so solutions share substructure). Add MDL/Stitch-style abstraction with **kernel-verified library
admission** and library reuse. **Bar:** the library GROWS toward the true concept set, total solution
description-length DROPS, and **d2/d3 solve-rate climbs past 0.36 with the frontier advancing** — i.e.,
compounding emerges precisely where the uniform-random version was mathematically forbidden from it.

# The Solver — execution-guided, per-step, verified (the fix for the composition bottleneck)

The `azr_struct` run validated discovery but stalled on composition: the solver emits a whole program
blind, so depth-d is one all-or-nothing guess over an exponential, syntactically-discontinuous space and
errors compound. 4 more surveys (execution-guided / recognition-model / type-directed / retrieval-subgoal
synthesis) converge on the same fix — and it is exactly our own validated length-gen principle:
**build the solution incrementally, executing and verifying each step against a sound checkable
intermediate; never emit the whole program blind.** This turns depth-d into d single-step inductions —
and depth-1 is what our associative-lookup solver already nails at 0.998.

## Design: iterate-execute-residual (forward), verified per step
Chosen over top-down subgoal-generation (imports a 2nd hard learning problem we lack signal for) and over
making the flat inducer compose internally (no surveyed system gets that for free).

1. **Single-step proposer** = the associative-lookup solver: given the demos + the CURRENT executed
   state, propose the next op/library-macro (its winning regime). Condition it on **property signatures**
   — executable predicates comparing the current state to the demo targets ("how far am I"), the
   featurization that makes execution-guidance learnable (Odena&Sutton; BUSTLE).
2. **Execute + verify per step (forward).** Apply the op via the executor → new intermediate state for
   every demo; accept only verified-consistent steps (the SOUND gate — stronger than every PRM/self-
   critique in the literature; Lightman/Uesato justify gating every step, not just the final program).
   Recurse the inducer on (intermediate-state → goal). The interpreter manufactures the subgoals — no
   subgoal generator needed (Execution-Guided Synthesis; Write-Execute-Assess; ExeDec turns depth-3 into
   3× depth-1).
3. **Shallow beam + backtrack** (ToT/HTPS-lite): expand a few candidate first steps, keep the K whose
   executed states most reduce distance-to-goal (a Write-Execute-Assess value over states), backtrack on
   dead ends. Depth-2/3 needs only a tiny beam; recovers from a wrong first step (greedy cannot).
4. **Hindsight relabeling** (CodeIt/HER): every intermediate state the inducer actually produced becomes
   a valid (state → produced-output) single-step training pair. This supplies the *intermediate-state*
   data a single-step-only training set never contains, and is the cold-start fix for self-play sparsity
   (the exact failure mode our earlier sparse-reward runs hit).
5. **Type-pruned frontier** (Myth/Synquid/SuSLik): the kernel/BDD types restrict the next-op choices to
   the well-typed ones (a strictly stronger sound prune than the liquid/SMT types those systems use — we
   own a full dependent-type kernel). The neural proposer only ORDERS the type-legal frontier, never
   decides correctness; CEGIS = feed the verifier's failures back as constraints.
6. **Library growth, retrieval-keyed** (DreamCoder/Stitch/LILO + babble e-graph anti-unification): abstract
   frequent verified op-chains into macros, each keyed by an **embedding of its I/O behavior** so the
   associative lookup can RETRIEVE and reuse it. A recurring depth-2 chain becomes a depth-1 op next round
   → the depth ceiling rises → compounding.

## The grand unification (all session research → one architecture)
A **verified, execution-guided, bottom-up, per-step program-synthesis self-play loop**: programs are
hypotheses; the kernel both type-PRUNES and soundly VERIFIES; solving iterates a single-step associative
inducer against executed intermediates with a shallow verified beam; hindsight relabeling feeds training;
MDL/Stitch abstraction grows a retrieval-keyed library; an endogenous curriculum over a STRUCTURED task
distribution drives difficulty; the library deepens so capability compounds. Every piece is something we
have (kernel/types/executor/associative-inducer/library/self-play) or a small, well-charted add (per-step
loop, beam+value, hindsight relabel, retrieval-keying). The through-line is the principle we already
proved twice (length-gen, verified-step reasoning): **per-step against a sound checkable intermediate.**

## Minimal next build
Wire the **iterate-execute-residual** solver into `azr_struct`: replace whole-program decode with
(propose-one-op → execute → verify → recurse on residual) + a shallow beam, the associative inducer as
the single-step proposer, and hindsight relabeling of intermediates. **Bar:** depth-2/3 solve-rate climbs
past the 0.36 plateau and the frontier advances — composition as interpolation over verified single steps.

*Solver synthesis from 4 surveys (2026-06-23): Chen/Liu/Song Execution-Guided (ICLR'19); Ellis
Write-Execute-Assess (NeurIPS'19); Odena BUSTLE (ICLR'21) + CrossBeam (Shi ICLR'22) + ExeDec (Shi ICLR'24);
Odena&Sutton property signatures (ICLR'20); Butt CodeIt (ICML'24); Balog DeepCoder (ICLR'17); Devlin
RobustFill (ICML'17); Osera Myth + Polikarpova Synquid/SuSLik; Solar-Lezama Sketch/CEGIS; Willsey egg
(POPL'21) + Cao babble (POPL'23); Lample HTPS (NeurIPS'22); Polu GPT-f (2020); Czechowski kSubS; Zhou
least-to-most; Andrychowicz HER.*

---

*Synthesis of 9 parallel literature surveys, 2026-06-22/23. Beyond-LLM + discover-and-reuse sources:
Ellis DreamCoder (2006.08381) + Bowers Stitch (2211.16605) + Grand LILO (2310.19791); FunSearch (Nature
2023) + AlphaEvolve (2506.13131); AlphaProof (Nature 2025) + AlphaGeometry (Nature 2024); Polu curriculum
(2202.01344); Zhao AZR (2505.03335); Wang Voyager (2305.16291); Soulos Differentiable Tree Machine
(2306.00751) + Neuro-symbolic Rewriting (2507.19372); Du IRED (2406.11179); Bai DEQ (1909.01377);
Schmidt&Lipson natural laws (Science 2009); Tishby IB (physics/0004057).*

---

*Synthesis of 5 parallel literature surveys, 2026-06-22. Key sources: Chollet ARC-AGI (1911.01547,
2505.11831); Solomonoff/Hutter (cs/0004001, 1510.05572); Delétang "LMs are compression" (2309.10668);
Müller PFNs (2112.10510); Grau-Moya "Learning Universal Predictors" (2401.14953); Ellis DreamCoder
(2006.08381) + Grand LILO (2310.19791); Wang Hypothesis-Search (2309.05660); Akyürek ARC-TTT (2411.07279);
Li induction+transduction/BARC (2411.02272); Lake&Baroni MLC (Nature 2023); Zhao AZR (2505.03335).*
