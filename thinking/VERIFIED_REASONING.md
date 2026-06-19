# Verified-Reasoning: results, what works, and what's next

This documents the verified-reasoning line of work: a neuro-symbolic approach where a small
language model **proposes** reasoning steps and a Datalog **engine validates** them, with the
engine's proofs used as per-step supervision. All numbers below are **independently re-measured by
the lead** (agent self-reports were systematically inflated and are not trusted here).

## TL;DR

- The project had **diverged** from its original implementation: commit `3a7a03b` ("Remove legacy
  language surface stack") deleted the **verified-reasoning core** (`datalog.py` + `thinking/flow.py`
  + `thinking/verify.py`). That core was the load-bearing generalization mechanism.
- **Restored** it (`dd26572`). On controlled tasks, **proof-supervised reasoning generalizes where a
  vanilla LM (answer-only / big-corpus) memorizes and sits at chance.**
- It **scales** along depth and across a relation web, and **reasons over (templated) natural
  language**. Supported **negation** was broken and is now **fixed structurally**.
- The remaining gaps — **multi-hop reliability** and **meaning-generalization to unseen phrasings** —
  root to **base-model capacity** and are **GPU-bound**. CPU-side objective/decode tweaks were
  verified dead ends.
- This is **not C2**. It is a *verified mechanism* that generalizes and scales on controlled tasks.

## The mechanism (restored core)

```
datalog.py        Datalog engine: closure / entails (verifier oracle) / proof_tree (per-step
                  supervision) / options (choice points)
thinking/verify.py  StepChecker / GoalChecker: validates an emitted step against the closure
thinking/flow.py    FlowRuntime: model proposes a step -> checker validates -> verified facts join
                    the known set -> invalid lines rejected/resampled
ScratchpadLM      the proposer, with a POINTER head (decisive for relabeling-invariant copy/composition)
```

Training: dense supervision over an **engine-derived proof chain** (not answer-only — answer-only
supervision starves the reasoning circuit and stays at chance). The pointer head is required for
held-out entity copy/composition.

## Verified results (lead-measured, vs answer-only baseline; chance = 0.50)

### Symbolic reasoning
| task | held-out | answer-only |
|---|---|---|
| transitive, novel configs (`reason_demo.py`) | **0.82** | ~0.50 |
| depth-gen, deeper-than-trained chains (`reason_scale.py`) | **~0.90** (flat to ~2x depth) | — |
| relation web + cross-relation composition, 3-seed (`reason_web.py`) | **0.88** (INHERIT 0.81) | 0.49 |

### Reasoning over natural language
| capability | held-out | notes |
|---|---|---|
| single-hop (ISA) over English (`reason_lang.py`) | **~1.00** | solid, reproduces |
| supported negation (EXCLUDE) (`reason_lang_neg.py`) | **0.78** (was 0.23, below chance) | fixed via negation-as-failure |
| multi-hop inheritance (config-holdout) | **0.81–0.96** | seed/budget-variant |
| meaning-gen to UNSEEN phrasings (template-holdout) | **~0.72** | positive single-hop types weak; **unsolved at this scale** |

### Key structural fix: negation-as-failure
Negation was broken because a "no" answer had no learnable structure — the model defaulted to "yes".
Fix: render a **grounded, parallel** failure proof (`note no X can P . so X cannot P . answer no .`,
verified absent from the closure) + a **closed-world decoder** ("yes" must be earned by a grounded
positive proof). EXCLUDE 0.23 -> 0.78 (config and unseen-phrasing). This is the template that works.

## What did NOT work (verified negatives)
- **Self-consistency / engine-gated decode** (`reason_lang_robust.py`): marginal in lead measurement
  (multi-hop +0.05 over greedy; step-gating can *hurt*). Inference-time voting cannot fix a base that
  drifts.
- **Paraphrase-invariance** (contrastive InfoNCE aux loss, `reason_lang_inv.py`): **failed** —
  collapsed single-hop relations (train ISA 0.97 -> 0.67), GATE FAIL, no net gain.
- **Big-corpus causal LM** (pre-restoration): memorized (held-out token-acc 7.4%), did not generalize.
- The **FER architecture bet** (relational arch alone, no scaffolding): did not deliver
  sample-efficiency in 5 clean tests — the prior win came from per-step scaffolding, not the arch.

## The lesson
**Structural representation fixes generalize** (giving the model a clean way to *represent/produce*
a thing — negation-as-failure, proof chains, the pointer head). **Bolt-on objectives and
inference-time tricks do not** at this scale. The two open gaps (multi-hop reliability, meaning
generalization) need a **stronger base model** = capacity + data + steps = **GPU/scale**, applied to
this now-validated mechanism.

## Files
`datalog.py`, `thinking/flow.py`, `thinking/verify.py` (restored core);
`thinking/reason_demo.py` (transitive proof-supervision), `reason_scale.py` (depth-gen),
`reason_web.py` (relation web), `reason_lang.py` (NL single+multi-hop), `reason_lang2.py` (broader NL),
`reason_lang_neg.py` (negation-as-failure, the current best NL reasoner).
Research artifacts (verified negative/unmeasured, uncommitted): `reason_lang_robust.py` (self-consistency),
`reason_lang_inv.py` (failed invariance), `reason_lang_canon.py` (canonicalization, unmeasured).

## Next (GPU)
Train the verified mechanism (`reason_lang_neg`-style: engine proof-supervision + negation-as-failure
+ pointer head) at **real capacity** (bigger model, more data diversity, more steps, multi-seed) to
close multi-hop reliability + meaning-generalization. See `runpod/` launch prep.
