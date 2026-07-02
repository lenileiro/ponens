# Extractive QA + in-context span PFN

A non-LLM research track: **read a passage, extract the answer span** — first by *runtime verified
reasoning* (WordNet-grounded, training-free), then by a *from-scratch neural reader*, and finally by
**in-context prior-fitted extraction** (TabPFN's method applied to spans) that infers the task from a
few `(text, span)` support examples with **no per-instance weight update**.

> Status (2026-06-30): every module has a passing `--selftest`; all committed on `comprehension-exam`
> (merged to `main` via PR #1). Not an LLM; no training on customer data. Honest bounds in §5.
> Product posture: **train once on public data, ship the weights, the customer never trains.**

## 1. The arc, in one table (SQuAD dev, all non-LLM)

| stage | measure | how |
|---|---|---|
| grounded rules | **F1 0.352** | WordNet answer-typing (LAT) + IDF sentence pick + NP candidates + "gravity" ranking, **training-free** |
| learned reranker | **F1 0.388** | numpy logistic-regression / GBM over symbolic features (Rajpurkar-2016 recipe), learning-to-rank |
| neural reader | **F1 0.690** (EM 0.566) | DrQA-style: GloVe + aligned-question emb + exact-match feature + BiLSTM + self-attention + bilinear span pointers |

Answer-typing alone (the "how long? → 5 business days" fix) moved rules F1 0.159 → 0.176 using **only
WordNet relations** — the `expects_quantity` attribute relation + POS-tag focus, no embeddings, no
hand-built word lists (hard project rule: never hardcode word matching/dictionaries).

## 2. One base weight (train-once, ship-weights)

A SQuAD-only reader fails zero-shot on a different span task (Tweet Sentiment Extraction, TSE:
Jaccard 0.177). Training **one** weight multi-task fixes it:

| | result |
|---|---|
| one reader on SQuAD + TSE | **SQuAD F1 0.691 + TSE Jaccard 0.420** |

The customer ships this weight and never trains; the model generalizes across the domains in its prior.

## 3. In-context span PFN (the frontier)

TabPFN / prior-fitted-network idea (Müller 2021) applied to **span extraction**: meta-train one weight
on a *prior over extraction rules*; at inference, infer the rule from `K` support `(text, span)`
examples and apply it to the query — **zero weight update, not an LLM.**

### 3.1 Synthetic proof-of-concept (`thinking/span_pfn.py`)
- single coherent family: in-context **exact-span 0.815** (support-ablated 0.420).
- broad 7-family prior: **stalls at exact 0.000** — small *and* A40-GPU scale, confirmed 3×. Abstract
  random rules over abstract tokens need true TabPFN scale.

### 3.2 Bridge to REAL text (`thinking/span_pfn_text.py`)
Real GloVe-100 tweet tokens. Made *genuinely* in-context via two modes the support must disambiguate:
`whole` (neutral → span = whole tweet) vs `phrase` (pos/neg → span = sentiment phrase). Same tweet
could be either, so the model must read the support to know which.

| setting | in-context | support-ablated | note |
|---|---|---|---|
| single-family (held-out tweets) | **0.699** | 0.429 | whole-tweet baseline ~0.40 |
| multi-task, 6 tasks, one weight (MEAN) | **0.763** | 0.398 | **broad prior does NOT stall on real text** |

Per-task (multi-task, held-out tweets) — the ablation is the lie-detector:

| task | in-context | ablated | reads as |
|---|---|---|---|
| whole | 0.938 | 0.038 | genuine in-context |
| number | 0.884 | 0.056 | genuine in-context |
| phrase | 0.434 | 0.067 | genuine in-context (fully support-dependent) |
| allcaps | 0.374 | 0.277 | weak in-context |
| after_at (@handle) | 0.980 | 0.983 | solved *without* support — trigger visible in query |
| after_hash (#word) | 0.971 | 0.966 | solved *without* support |

Tasks whose answer is ambiguous from the query collapse without support (real inference); tasks whose
trigger sits in the query get learned as unconditional skills (no ablation gap — reported, not hidden).

## 4. The key finding: in-context extraction is bounded by PRIOR COVERAGE

Leave-one-out held-out-**task** generalization (`--heldout-all`: meta-train on 5 tasks, test in-context
on the unseen 6th, defined only by its support examples):

| held-out (unseen) task | in-context | ablated |
|---|---|---|
| phrase | 0.184 | 0.114 |
| whole | 0.141 | 0.069 |
| number | 0.038 | 0.013 |
| allcaps | 0.083 | 0.038 |

**It does not generalize to a task outside the meta-training distribution** (0.04–0.18, vs 0.76–0.94
when that task *was* trained). So the strong 0.763 is **recognize-which-of-the-trained-tasks**, not
infer-any-new-rule. This is exactly TabPFN's central claim, and it mirrors the synthetic held-out-family
failure. The "define *any* extraction by examples" promise therefore reduces to a concrete,
non-magical lever: **breadth of the meta-training prior** — to cover a new task family, the prior must
contain it. Few-shot support alone will not conjure a novel rule.

## 5. Honest bounds (mapped, not hidden)

1. In-context generalization is **prior-bounded** (§4) — the roadmap to broader coverage is a wider,
   programmatically-generated family of real-text extraction tasks, not few-shot extrapolation.
2. The neural reader (F1 0.690) is strong but sub-SOTA; it is deliberately from-scratch + non-LLM.
3. Rule-based grounding is capped by WordNet coverage and the NP-chunk candidate generator.
4. Query-visible-trigger tasks are learned unconditionally, so their in-context numbers overstate the
   *inference* mechanism (the ablation column makes this explicit).

## 6. Modules

| module | role |
|---|---|
| `thinking/squadqa.py` | runtime extractive QA: IDF sentence pick, NP candidates, gravity ranking, WordNet answer-typing |
| `thinking/kb.py` | runtime KB over WordNet: `expects_quantity`, entity/supersense typing, is-a closure (no word lists) |
| `thinking/runtime_api.py` | product API: `answer(q, passage)`, `explain(q, passage)` → answer + trace |
| `thinking/squad_rank.py` | learned reranker: symbolic features + numpy logreg / sklearn GBM, learning-to-rank |
| `thinking/squad_reader.py` | DrQA-style neural reader (GloVe + BiLSTM + bilinear span pointers) |
| `thinking/tweet_sent.py` | Kaggle Tweet Sentiment Extraction loader + Jaccard |
| `thinking/multitask.py` | one reader trained on SQuAD + TSE (the shipped base weight) |
| `thinking/span_pfn.py` | in-context span PFN, synthetic prior (single-family + broad-prior scale study) |
| `thinking/span_pfn_text.py` | **in-context span PFN on REAL text** (single-family, multi-task, held-out-task) |

RunPod launchers in `runpod/` (dry-run default, pod-side timeout + always-terminate, API key env-only).

## 7. Run it

```bash
python -m thinking.span_pfn_text --selftest        # real-text in-context PFN (whole/phrase)
python -m thinking.span_pfn_text --train           # single-family, held-out tweets
python -m thinking.span_pfn_text --multitask       # 6 real tasks, one weight
python -m thinking.span_pfn_text --heldout-all     # leave-one-out generalization (the prior-coverage bound)
python -m thinking.span_pfn      --selftest        # synthetic PoC (exact-span 0.815)
python -m thinking.multitask     --selftest        # one reader on SQuAD + TSE
```
