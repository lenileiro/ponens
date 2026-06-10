# Bringing Fractured/Unified Representations to Text for Relational Reasoning
**Date:** 2026-06-06
**Status:** Research-validated design (5/5 research agents synthesized)
**Source technique:** "The Fractured Entangled Representation Hypothesis" (Kumar, Clune, Lehman, Stanley — arXiv:2505.11581, May 2025)

## The Goal (restated)

Take the FER/UFR idea — that **unified, factored** internal representations (UFR) generalize and compose better than **fractured, entangled** ones (FER) — and bring it into **text**. Train an agent that builds clean factored representations from text *understanding*, then use those representations for **relational reasoning** (reasoning about relationships between entities/concepts).

This document turns that aspiration into a concrete, falsifiable research program with a runnable first experiment.

---

## 1. The Honest Scientific Framing (read first)

The source paper is a **position paper**, not an established result. Its entire empirical demonstration is a *single CPPN image* (the Picbreeder "skull") comparing an evolved network vs. an SGD network — no text, no LLM, no quantitative fracture metric. Everything below is therefore **a hypothesis we would be testing**, not a settled technique we're applying.

Three findings from the research materially shape the bet:

1. **Counter-evidence (encouraging):** "Transformers learn factored representations" (arXiv:2602.02385, Simplex AI Safety, Feb 2026) finds transformers have an **inductive bias toward factored representations** (orthogonal subspaces) when latent factors are *conditionally independent* — and even show factoring bias early when independence is violated. **Implication:** text transformers *can* be pushed toward UFR; the lever is the **independence/structure of the training signal**, not abandoning SGD. We do NOT need evolutionary search.

2. **The phenomenon is partly already named:** The interpretability community calls FER's "entanglement" **superposition / polysemanticity** (Anthropic). FER's genuine novelty is the *causal claim* (the training process produces it) and the open-ended-search remedy — both contested. We should lean on the mature superposition/SAE tooling rather than reinvent it.

3. **The publishable gap (flagged independently by 3 agents):** **Nobody has shown that emergently-more-factored relational representations causally improve relational/systematic generalization in a text model.** That is exactly the question this program targets — it is genuinely open work, not a re-tread.

**Bottom line:** This is a real research bet with a clean falsifiable core, mature measurement tools, and a strong architectural prior. The risk is scientific (the hypothesis may be false / may reduce to "superposition"), not engineering.

---

## 2. The Core Falsifiable Hypothesis

> **H1:** A text model trained to develop **more factored** representations of entities and relations will show **better systematic relational generalization** — specifically, extrapolation to relation-composition chains *longer than any seen in training* (the CLUTRR length-generalization axis).

> **H2 (mechanism):** The factored-ness is *causal*, not merely correlated — interventions that increase factoring (architectural or training-signal) improve length generalization; interventions that scramble it degrade it.

This is testable, has a clear null (factoring uncorrelated with generalization), and a clear payoff metric (CLUTRR/REL accuracy vs. chain length k).

---

## 3. The Relational Reasoning Target

"Reasoning for relationship" maps cleanly onto the **relational/systematic-reasoning** literature. Our primary testbed:

### Primary benchmark: CLUTRR (arXiv:1908.06177)
- Reasoning about **kinship relations** from short stories. Requires (a) extracting stated relations from text + (b) inducing composition rules ("parent's parent = grandparent") to derive an *unstated* target relation.
- **The key axis:** train on short chains (k=2,3,4), **test on held-out longer chains (k up to 10)**. This is the exact "does it compose beyond training" signal we need.
- **Known result that motivates the whole bet:** a **GNN operating on the extracted relation graph generalizes far better** than text transformers; all text models degrade monotonically as k grows. A model operating over *clean factored relational structure* extrapolates; one that entangles relation-inference with surface text does not.

### Secondary benchmarks
- **REL** (arXiv:2604.12176, ICML 2026) — varies *relational* complexity while controlling entity complexity. The most on-target modern benchmark for the factored-vs-fractured question.
- **CFQ / MCD splits** (ICLR 2020) — compositional generalization (recombine known atoms in novel ways).
- **ProofWriter / ProntoQA** — multi-step logical/relational chains with gold proofs.

---

## 4. Porting the FER Methodology to Text (the measurement stack)

You cannot port FER 1:1 — a CPPN's whole output is enumerable; a transformer's is not. But each leg has a mature text analog:

| FER (original) | Text analog | Maturity |
|----------------|-------------|----------|
| Neuron = viewable image | **SAE features** + automated-interp scoring (arXiv:2410.13928); max-activating examples; Neuronpedia dashboards | **HIGH** |
| Weight sweep → semantic morph | **SAE feature clamp-sweep** (Golden-Gate-style); CAA (2312.06681) / RepE-LAT (2310.01405) as cross-checks; scored on steerability/fluency/coherence/off-target (2502.02716) | HIGH tooling, MEDIUM claim |
| "Is this one coherent concept?" | Steering-evaluation metric quartet; the *outcome* to measure, not assume | MEDIUM |
| Fracture vs factored metric | **IRS (Interventional Robustness Score)** — does changing factor A leave B unchanged?; **MIG/DCI**; **SAE feature-splitting/polysemanticity counts** as native fracture score | HIGH–MEDIUM |
| Enumerable CPPN canvas | **No clean analog** — substitute: small conditional generator with low-dim disentangled control space | LOW (research gap) |

### Relational-specific measurement (the real probe)
The strongest, most-direct factoring metric for *relations* comes from binding-mechanism interpretability:
- **Binding-subspace analysis** (Feng & Steinhardt, arXiv:2310.17191): transformers bind entities↔attributes via a **low-rank, causal "Binding ID" subspace**. We can measure: (a) the *rank/linearity* of the relation-binding subspace, (b) whether linear edits cleanly re-bind relations, (c) how much binding relies on **positional "Ordering IDs"** (arXiv:2409.05448) vs. content — positional reliance is a *fracture signature*.
- A "more factored" relational model should have a **lower-rank, more linear, more causally-clean, less position-dependent** binding subspace.

---

## 5. The Three Levers to Induce UFR in Text

The research points to three ways to push a text model toward factored relational representations, in descending order of evidence strength:

### Lever 1 — Architectural factoring (strongest evidence)
**Relational attention / Abstractors / Dual Attention Transformer** (Altabaa, Webb, Lafferty):
- **Abstractors** (arXiv:2304.00195, ICLR 2024): relational cross-attention computes a relation matrix from query·key inner products, attends over *learned input-independent symbolic values* → relational info **disentangled from object features**. Dramatic sample-efficiency gains on relational tasks.
- **Dual Attention Transformer** (arXiv:2405.16727): two head types — **sensory attention** (object features) + **relational attention** (relations) — a drop-in transformer variant (`dual-attention` package). This *architecturally enforces* the factoring FER says SGD fails to find.
- This is the **relational bottleneck** (arXiv:2309.06629) made concrete: restrict processing to relations → force abstract, symbol-like (factored) representations.

### Lever 2 — Training-signal structure (good evidence, cheap)
- **Triple-extraction auxiliary objective** (KEPLER-style, TACL 2021): jointly train language modeling + a knowledge-embedding objective over `(entity, relation, entity)` triples. Known to improve relation extraction; **untested** whether it produces a more *factored* relational subspace — that's part of our experiment.
- **Conditional-independence structure** (from arXiv:2602.02385): construct/curate training data so relational factors are conditionally independent where possible — the condition under which transformers factor cleanly.

### Lever 3 — Open-ended / diversity (weak for weights, fine for data)
- Evolutionary weight search (the paper's actual UFR cause) is **only feasible at toy scale** (NEAT/MAP-Elites over thousands–low-millions of params; QD-LLM only evolves a ~32K-param soft prompt over a frozen model). **Not a path to a real text agent.**
- The practical descendant is **diversity-driven curriculum/data** (LLM-POET, DéjàQ) and diversity-aware data selection — useful but a secondary lever here.

### Lever 3b (EVALUATED — evidence leans NEGATIVE): CEFR A1→C2 reading curriculum
Considered training the "understanding" base on a graded English curriculum (A1→C2), ordered easy-to-hard like a human learner — which is appealing because it directly operationalizes the FER paper's own "curriculum/learning-order" mitigation conjecture. **Verdict after research: the naive difficulty-ordering version is close to a known dead-end at our scale.** Record so we don't relitigate:
- **BabyLM** (10M/100M words — our exact regime): curriculum ordering was the most-tried strategy (~42% of teams) and **mostly failed**; winners used architecture (LTG/ELC-BERT). (arXiv:2504.08165)
- **Child-directed-speech study** (arXiv:2408.03617): developmental ordering had **negligible effect**; mixed corpus beat natural CDS. Near-direct refutation of "learn like a child via simple-first input."
- Large-scale curriculum (arXiv:2506.11300) mainly buys **convergence speed** (~18–45% fewer steps, ~3.5% as warmup); and **easy→hard is not reliably best** — weak models/hard tasks often prefer reverse (arXiv:2510.19099).
- "CEFR ordering → more-factored representations" is **untested** — would be novel, but nearest evidence leans negative.

**Salvageable reframe (if pursued):** (1) stage by data **type/quality = mid-training**, not difficulty — even ~1% skill coverage seeds generalization (arXiv:2512.07783); (2) treat **simplicity as a global property** (clean controlled-vocabulary input throughout), not a schedule; (3) **measure representations directly** rather than infer from scores.

**Solid resources if used:** UniversalCEFR (~505k CEFR-labeled texts), CEFR-SP (~17k sents), OneStopEnglish (CC BY-SA); XLM-R auto-CEFR classifier (~63% F1); **synthetic CEFR generation** ("Tarzan to Tolkien" arXiv:2406.03030, TinyTolkien ~2k graded stories) sidesteps copyright + volume.

**Recommendation:** Do NOT make this the main bet. At most, run it as an **ablation arm** (forward vs reverse vs shuffled) against a strong architecture baseline, expecting architecture/objective (Levers 1–2) to dominate.

#### Sub-idea: "feed a dictionary at the end of C2 to complete vocabulary" — EVALUATED, partial
Premise (a dictionary closes the vocabulary gap) rests on a wrong model of LM meaning acquisition:
- Meaning comes from **usage/distribution**, not definitions (definitions are sparse signal); coverage ≠ competence.
- **Circularity/grounding:** a monolingual dictionary defines words via other words — "learn Chinese from a Chinese-only dictionary." Helps only once a distributional base exists.
- **What's true:** dictionary definitions help as a **targeted supplement for RARE/long-tail words** (Hill et al. 2016 "Embedding the Dictionary"; Dict-BERT 2022) — tail, not core.
- **Best reframe for us:** a dictionary's high-value content is **relational** — WordNet (synonym/antonym/**hypernym**/meronym) = clean `(word, relation, word)` triples. Feed **structured lexical relations**, not flat definitions → doubles as relational scaffold + free **Lever-2 triple-extraction** signal. Strengthens Lever 2, not a curriculum capstone.

---

## 6. The Headline Experiment (runnable, toy-scale first)

A clean 2×2 that directly tests H1/H2 and fills the literature gap.

**Setup:** Small models (1M–50M params, trainable on a single GPU) on CLUTRR (+ REL as held-out transfer).

**Factors (2×2):**
- Architecture: **standard transformer** vs. **Dual Attention Transformer** (relational attention)
- Training signal: **LM only** vs. **LM + triple-extraction auxiliary objective**

**For each of the 4 cells, measure:**
1. **Outcome (the payoff):** CLUTRR accuracy as a function of chain length k, train on k≤4, test k=5…10. Plus zero-shot transfer to REL.
2. **Factored-ness (the mechanism):**
   - Binding-subspace rank + linearity + causal-edit cleanliness (Feng & Steinhardt method)
   - Position-reliance (Ordering-ID share) — lower = more factored
   - IRS on controlled relation factors
   - SAE feature-splitting / polysemanticity count over relation features
3. **The key correlation:** plot factored-ness vs. length-generalization across all cells + checkpoints. **H1 predicts a positive relationship; H2 predicts the architectural/signal interventions move both together.**

**Why this is valuable regardless of outcome:**
- If factoring → generalization holds: first causal demonstration in text; directly informs how to train better relational/coding reasoners.
- If it fails: strong evidence FER reduces to superposition / doesn't transfer to relational text — also publishable and saves the field effort.

**Cost:** Toy-scale, single-GPU, days not weeks. ~$100–500 of compute. This is a *research probe*, not a model-training megaproject.

---

## 7. Phased Plan

| Phase | Goal | Deliverable | Effort |
|-------|------|-------------|--------|
| **P0: Reproduce FER** | Run the original repo (github.com/akarshkumar0101/fer), confirm the skull weight-sweep contrast, internalize the method | Working FER visualization notebook | 1–2 days |
| **P1: Measurement harness** | Build the text-side stack: SAE features + automated interp + binding-subspace probe + IRS on a controlled relational dataset | Reusable "factored-ness scorer" for any small text model | 1 week |
| **P2: CLUTRR baselines** | Train standard transformer on CLUTRR; reproduce monotonic-degradation-with-k; establish baseline factored-ness | Baseline curves (accuracy vs k) + factored-ness scores | 3–5 days |
| **P3: The 2×2** | Add Dual Attention Transformer + triple-extraction objective; run the full grid | The headline correlation: factored-ness vs length-generalization | 1–2 weeks |
| **P4: Causal test (H2)** | Intervene — scramble the binding subspace (degrade factoring) and re-measure generalization; clamp toward factoring | Causal evidence for/against the mechanism | 1 week |
| **P5: Scale-up decision** | If P3/P4 positive, test whether the architectural/signal levers help a real small LLM (Qwen3 family) on REL/CLUTRR | Go/no-go for integrating into the main model | gated |

---

## 8. How This Connects to the Main Coding-LLM Project

This is **adjacent research, not the critical path** to DeepSWE. But there's a genuine throughline:

- Our `REASONING.md` already found that **distillation expands the reasoning ceiling while RL only amplifies existing patterns** (arXiv:2504.13837) — the *same nerve* FER hits (training process determines representational quality). FER offers a mechanistic lens on *why* distillation might transfer cleaner structure than RL.
- Coding is fundamentally **relational** — type relationships, call graphs, data-flow, variable binding. If factored relational representations help systematic generalization, that plausibly transfers to **repo-scale code reasoning** (cross-file dependency tracking is literally entity-relation reasoning). Our existing notes already flagged **RepoGraph/CGM** (graph-structured code) as a SWE lever — the same "explicit factored relational structure helps" thesis.
- **Recommendation:** Run P0–P3 as a **time-boxed research spike** in parallel with the main plan. It is cheap, falsifiable, and the binding-subspace + relational-attention findings could directly inform whether we add relational-attention heads or a code-triple-extraction auxiliary objective to our coding model. Do **not** let it block the DeepSWE critical path.

---

## 9. What's Established vs. Speculative (honesty ledger)

**Established:**
- Models over explicit factored relational structure (GNNs, abstractors, LLM+KG) generalize better on relational tasks (CLUTRR, relational seq2seq, KGQA).
- Transformers have a real, low-rank, causal **binding subspace** (partially factored relational mechanism).
- Standard transformers generalize **poorly** on compositional/length splits by default.
- Transformers have an **inductive bias toward factoring** under conditional independence (2602.02385).
- The measurement tools (SAEs, automated interp, IRS, binding-subspace probes, steering eval) are mature enough to build on today.

**Mixed / complicating:**
- LLM binding is **partly positional and multi-mechanism** (Ordering IDs, mixing mechanisms) — not a clean symbolic relation algebra; a fractured component exists even in the binding mechanism.
- SAE features are **not reliably monosemantic** ("Coffee feature activates on Coffins," arXiv:2601.03047) — so our measurement instrument is itself partly fractured. Running the probe may validate *or undermine* the monosemanticity assumption.

**Speculative (our research bet):**
- That emergently more-factored relational reps **causally** improve relational generalization in text — untested directly. **This is the experiment.**
- That triple-extraction training yields more *factored* (not just better-scoring) relational subspaces — untested.
- That the FER hypothesis transfers from toy CPPNs to relational text at all.

---

## 10. Key Assets & References

**Code:**
- FER reference implementation: `github.com/akarshkumar0101/fer` (CPPN, SGD trainer, neuron visualization, weight-sweep)
- Dual Attention Transformer: `dual-attention` package (Altabaa & Lafferty)
- CLUTRR: `github.com/facebookresearch/clutrr`
- SAE/interp tooling: Neuronpedia, EleutherAI automated-interp (`sae-auto-interp`)

**Core papers:**
- FER hypothesis — arXiv:2505.11581
- Transformers learn factored representations — arXiv:2602.02385
- Binding IDs in context — arXiv:2310.17191; Ordering IDs — arXiv:2409.05448; Mixing mechanisms — arXiv:2510.06182
- Abstractors / relational cross-attention — arXiv:2304.00195; Dual Attention Transformer — arXiv:2405.16727; Relational bottleneck — arXiv:2309.06629
- CLUTRR — arXiv:1908.06177; REL — arXiv:2604.12176; CFQ — ICLR 2020
- Automated interp at scale — arXiv:2410.13928; RepE — arXiv:2310.01405; CAA — arXiv:2312.06681; steering eval — arXiv:2502.02716; SAE steering reliability — arXiv:2601.03047
- KEPLER (triple objective) — TACL 2021
- Disentanglement metrics skepticism — Locatello et al., arXiv:1811.12359

---

## One-Paragraph Summary

Bringing FER to text means testing one clean hypothesis: **do more-factored representations of entities and relations cause better relational generalization** (extrapolating relationship chains beyond training length, à la CLUTRR)? We don't need evolutionary search — transformers already factor under the right conditions (arXiv:2602.02385). The actionable levers are **architectural** (relational-attention / Dual Attention Transformer — strongest evidence), **training-signal** (triple-extraction auxiliary objective + conditional-independence structure), and secondarily **diversity-driven data**. We port FER's "weight sweep" to **SAE feature clamping** and its "look at the neuron" to **SAE features + automated interp**, and add a relation-native factoring metric via the **binding-subspace probe**. The headline experiment is a cheap, single-GPU 2×2 (standard vs. relational-attention × LM vs. LM+triples) on CLUTRR, measuring both factored-ness and length-generalization to test whether they causally move together. This fills a genuine literature gap, is falsifiable either way, and — because code reasoning is itself relational — could feed back into the main coding-LLM plan without blocking it.
