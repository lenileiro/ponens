# Toward real C2: honest scope, gap analysis, and a staged path

> Written 2026-06-20. The standing goal: the model understands what words mean and their
> associations, and can ultimately **pass a C2 exam** with wholistic understanding + write + reason.
> This document scopes what that ACTUALLY requires and where the current repo really stands — no
> toy-metric inflation.

## 1. What "real C2" actually is (the target)

CEFR **C2 = Mastery / near-native proficiency.** A real exam (Cambridge C2 Proficiency / CPE) is:
- **Reading & Use of English**: multiple-choice **lexical cloze** (fine word-choice/collocation), **open
  cloze** (grammar), **word formation**, **key-word transformations** (paraphrase under a constraint),
  **gapped text** (insert removed paragraphs — long-range discourse cohesion), **multiple matching**
  (locate/contrast information across a long text), comprehension of **implicit meaning, tone, irony**.
- **Writing**: a compulsory **summarize-and-evaluate essay** (read two texts, synthesize + argue,
  240–280 words) + one of {article, letter, report, review} — **register**, argumentation, cohesion.
- **Listening** + **Speaking** (out of scope for a text model, though we have separate ASR/TTS).

Core demands: **open-domain real English**, ~native **vocabulary** (tens of thousands incl. idiom/
collocation), **near-perfect grammar over complex syntax**, **discourse coherence over hundreds of
words**, **pragmatics/nuance**, **world knowledge**, **summarization + argumentation + register**,
**multi-text synthesis**.

## 2. Where we honestly are vs that

| capability | C2 needs | we have today | gap |
|---|---|---|---|
| Vocabulary | ~native, tens of thousands + idiom | ~hundreds of templated tokens, closed 44-entity KB | huge |
| Grammar/syntax | full complex, near-perfect | fixed template grammar (even "an horse" slips) | huge |
| Reading comprehension | nuance/implicit over long REAL texts | symbolic yes/no over hand-built facts | huge |
| Writing | coherent argumentative essays, in register | templated sentences; novel-entity 6/8, 0.73 coh | huge |
| Reasoning | robust real-world inference | verified symbolic on 44-entity KB (is-a/inherit/negation 1.00 held-out) — STRONG but closed-world | transfer unproven |
| Open-domain generalization | REQUIRED | **the binding failure** (see §3) | blocking |

## 3. The binding constraint (the honest crux)

Every clean result in this repo generalizes **because it is tiny, verifiable, and closed-world**
(small KB + Datalog-checked proofs + templated surface). C2 is the **opposite**: open-domain,
near-native, hundreds-of-words discourse.

The one time the project went open-domain — a **big-corpus causal LM** (943k-row diverse English,
dim768/12L) — it **memorized**: train token-loss 0.002 but **held-out token-acc 7.4%**, output
word-salad (see memory `dense-loss-induction` / `coding-agent-goal` / lessons.md). FER was never even
wired in (a decode-objective bug fed it constant input). **So we have NO working open-domain language
model.** That, not any of the toy axes, is what stands between us and C2.

Implication, stated plainly: **a small, from-scratch, CPU / single-H100 model will not reach C2.**
C2-level open-domain English empirically requires large-scale pretraining (billions of tokens, large
params) OR standing on a pretrained backbone. The repo's symbolic wins are real science on a
*different* axis (sample-efficient *verifiable reasoning*) — valuable, but not a path to C2 alone.

## 4. A staged, realistic path

**Stage 0 — MEASURE (do first; cheap; the thing we've never done).** Build a REAL C2-style eval
harness on **public-domain** real English (Project Gutenberg etc. — no copyrighted exam material, no
contamination): lexical cloze, gapped-text (cohesion), inference comprehension, + a writing rubric.
**Administer it to whatever model we have** to get the HONEST baseline (expect near chance). This
finally answers the hook's standing complaint ("C2 exam never administered") and replaces toy metrics
with a real yardstick.

**Stage 1 — a generalizing open-domain LM.** Fix the big-corpus memorization root cause FIRST
(diagnose the train-vs-held-out disconnect; use the `target` decode objective; clean held-out; very
likely **adopt/finetune a pretrained backbone** rather than from-scratch tiny-model). Gate on Stage-0
eval, not train loss.

**Stage 2 — graft the repo's real assets on top.** Layer the **verified-reasoning** (Datalog proof
supervision) + **content-plan→render with faithfulness checks** onto a competent LM for the parts of
C2 that are about *correctness* (inference questions, factual writing) — this is where our
sample-efficient reasoning genuinely helps.

**Stage 3 — the C2 skills.** Reading: comprehension w/ the reasoning core. Writing: summarize-and-
evaluate essays with rendering + faithfulness/coherence checks. Iterate against the Stage-0 eval.

## 5. Recommendation / first concrete move

**Build Stage 0** — the real-English, contamination-free C2-style eval harness, and administer it for
an honest baseline. It is the highest-value, on-goal, non-toy step; it's cheap (no big GPU run); and
it converts "are we at C2?" from assertion into measurement. Everything after is gated on it.

---

## 6. RESULTS LOG

### Stage 0 — DONE (2026-06-20). First real-English C2-style eval administered.
`thinking/c2_eval.py`, contamination-clean (train/held-out disjoint + dedup of corpus-duplicate
sentences). Small from-scratch LM: lexical-cloze **0.73**, discourse cohesion **0.33** / coherence
**0.30** ~ chance (0.25). Lexical knowledge emerges; cross-sentence understanding does not.

### Stage 1 probe — DONE (2026-06-20). Path B (crack-from-scratch + scale), H100 scaling probe.
`runpod/launch_c2.py`, gated on the held-out reading eval (NOT train loss). Lead-measured from fetched
JSON (chance 0.25; config l was cut by the 90-min pod cap):

| config | data | lm_loss | lexical | cohesion (gapped) | coherence (next) |
|---|---|---|---|---|---|
| s  d256/L4, 6k  | 6 MB  | 1.15 | 0.637 | 0.350 | 0.338 |
| m  d512/L8, 12k | 22 MB | ~1.1 | 0.675 | 0.325 | 0.312 |

**VERDICT: scale did not move discourse comprehension.** 2x depth/width + ~4x data + 2x steps left
cohesion/coherence FLAT (slightly down), both near chance; only lexical nudged up. Train loss fell
(model fits the text) while held-out discourse stays at chance = the **memorization-wall signature**,
now measured with a real generalization gate.

Honest caveat: 22 MB is tiny for from-scratch open-domain. This proves the *affordable* end of path B
yields NO discourse signal and did not respond to the scaling we could run; it does NOT prove no
from-scratch scale ever could. But genuinely testing B needs billions of clean tokens we don't have —
which is itself the argument for **path A (pretrained backbone)**: stand on a model that already has
open-domain language + discourse, then graft the repo's verified-reasoning + faithfulness machinery
(Stages 2-3) for the correctness/argumentation parts of C2.

**Recommended pivot: path A.** Keep `c2_eval` as the gate; the number to beat is cohesion/coherence
well above 0.25 (the from-scratch ceiling found here). C2 remains UNMET.
