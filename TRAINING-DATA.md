# Training Dataset — Description & Requirements
**Date:** 2026-06-06
**Scope:** Primary = the FER→relational experiment dataset (CLUTRR). Secondary = pointer to the coding-LLM datasets (already in VALIDATED-PLAN.md).
**Grounded in:** direct inspection of `github.com/facebookresearch/clutrr` (cloned to `/tmp/llm-repos/clutrr`).

---

## 1. What the Dataset Is

For the FER→text→relational experiment (see `FER-TEXT-RELATIONAL.md`), the training dataset is **CLUTRR** — *Compositional Language Understanding with Text-based Relational Reasoning* (Sinha et al., EMNLP 2019). It is **procedurally generated**, not a fixed download: a generator turns kinship graphs into short natural-language stories plus a relational query.

**Why this dataset fits our hypothesis exactly:**
- The task *requires* composing relations (the answer relation is **never stated** in the story — it must be inferred by chaining).
- It has a built-in **systematic-generalization axis**: train on short reasoning chains, test on longer ones never seen in training. This is precisely the "does a factored representation extrapolate?" signal we need.
- It is fully controllable and labeled with the underlying relation graph, so we can build the **triple-extraction auxiliary supervision** (Lever 2) for free.

---

## 2. Anatomy of One Training Example

Each example is a story + a query about an *unstated* relationship that must be inferred.

**Concrete k=2 example** (2 facts → 1 inferred relation):

> **Story:** "Bob is a son of Alice. Carol is a daughter of Bob."
> **Query:** "How is Alice related to Carol?"
> **Target:** `grand` (i.e., Carol is Alice's granddaughter)

The model must learn the composition rule `child ∘ child = grand` — the word "grandmother/granddaughter" appears **nowhere** in the story. A model that memorizes surface text fails; one that builds a factored relation representation and composes it generalizes.

**Chain length k** = number of stated facts that must be chained. A k=4 example states 4 relations and requires composing all of them to reach the target.

---

## 3. Full Schema (what the generator emits)

CLUTRR outputs a CSV; verified columns:

| Column | Meaning | Use in our experiment |
|--------|---------|----------------------|
| `id` | unique example id | bookkeeping |
| `story` | natural-language story (templated surface forms) | **model input** |
| `clean_story` | story with placeholders/cleanup | alt input |
| `query` / `text_query` | the relation being asked (e.g., "How is e_1 related to e_2?") | **model input** |
| `target` | gold relation label (e.g., `grand`, `sibling`, `in-law`) | **primary training label** |
| `text_target` | natural-language target | optional |
| `story_edges` | the underlying graph edges (entity→entity) | **triple-extraction aux label** |
| `edge_types` | relation type per edge | **triple-extraction aux label** |
| `query_edge` | the (e1,e2) pair being queried | eval |
| `proof_state` / `f_comb` | the proof trace / relation composition used | curriculum / analysis |
| `genders` | name→gender mapping | controls/probes |
| `task_name` | e.g., `1.3` (task.chain-length) | split bookkeeping |
| `task_split` | train/test marker | split |
| `syn_story`, `node_mapping` | synthetic story + node map | probes |

For the **main objective** we use `(story + query) → target`. For the **triple-extraction auxiliary objective** we use `story → {(entity, relation, entity)}` derived from `story_edges` + `edge_types`.

---

## 4. The Relation Vocabulary & Composition Rules (the reasoning to be learned)

This is the actual "knowledge" the model must internalize — small and clean, which is what makes it a good factored-representation testbed.

**Base relations** (from `relations_store.yaml`): `child` (son/daughter), `inv-child` (father/mother), `grand` (grandparent/grandchild), `sibling`, `SO` (significant-other/spouse), `in-law`, `un` (uncle/aunt) and their inverses. Surface forms have multiple paraphrases each (e.g., "e_2 is a son of e_1", "e_1 has a son called e_2", "e_2 is e_1's son").

**Composition rules** (from `rules_store.yaml` — the inference the model must induce):
```
child  ∘ child   = grand      (parent's child's child → grandchild)
child  ∘ SO      = in-law
child  ∘ sibling = child       (your child's sibling is your child)
sibling∘ sibling = sibling
sibling∘ child   = un          (sibling's child → niece/nephew, i.e. inv of uncle/aunt)
grand  ∘ sibling = grand
SO     ∘ child   = child       (spouse's child → your child)
... plus symmetric (sibling, SO) and inverse-equivalence (child↔father/mother) rules
```
There is also a `work` relation family (manager/subordinate/coworker) but we focus on **family/kinship** (Task 1).

**Why this matters for the experiment:** the rule set is a small, closed, *compositional algebra*. A Unified-Factored Representation should encode each base relation once and compose them cleanly; a Fractured representation will learn redundant per-context fragments that fail to extend to longer chains.

---

## 5. The Split Strategy (the systematic-generalization axis)

This is the heart of the experiment design:

| Split | Chain lengths (k) | Rows | Purpose |
|-------|-------------------|------|---------|
| **Train** | k = 2, 3, 4 | ~10k–50k | learn base relations + composition on short chains |
| **Test (in-dist)** | k = 2, 3, 4 | ~2k | sanity / IID accuracy |
| **Test (generalization)** | k = 5, 6, 7, 8, 9, 10 | ~1k each | **the payoff metric** — accuracy vs chain length |

- Generator config: `--train_tasks 1.2,1.3,1.4 --test_tasks 1.5,1.6,1.7,1.8,1.9,1.10`.
- Use `--holdout` / `--unique_test_pattern` so test compositions are **unseen patterns**, not memorizable.
- The headline result is **accuracy as a function of k** for each of the four 2×2 cells, cross-plotted against measured factored-ness.

**Noise variants** (optional robustness, via task id): Task 1 = clean, 2 = +supporting facts, 3 = +irrelevant facts, 4 = +disconnected facts, 5 = all. Start with Task 1 (clean) to isolate the composition signal; add noise later.

---

## 6. The Auxiliary Triple-Extraction Data (Lever 2)

To test whether a **triple-extraction training signal** induces cleaner relational representations (one of the three levers), we derive a second supervision target from the same examples:

- **Input:** `story`
- **Aux label:** the set of `(entity, relation, entity)` triples = `story_edges` zipped with `edge_types`
- **Objective:** auxiliary head/loss that reconstructs the relation graph from the text (KEPLER-style joint objective).

This requires **no extra data** — it's a free second view of the CLUTRR graph already emitted by the generator. The 2×2 toggles whether this auxiliary loss is on.

---

## 7. Volume & Compute Needed

| Resource | Requirement | Notes |
|----------|-------------|-------|
| Data volume | ~50k train + ~10k test examples | Tiny — a few hundred MB CSV. CLUTRR generation is fast (minutes). |
| Generation compute | CPU only | Pure-Python generator (pandas, names, networkx). No GPU. |
| Training compute | Local MPS for toy models (≤10–20M params); rented GPU if scaled | M1 Pro / 16GB handles the toy 2×2; see IMPLEMENTATION-PREP §6 memory note |
| Tokenizer/vocab | small — names + ~15 relation words + templates | can use a char/word tokenizer or a small BPE |

---

## 8. What's Needed to Produce It (concrete)

**Blockers:** CLUTRR's pinned deps are ancient (`pandas==0.23.4`, `names==0.3.0`) and won't install on Python 3.14. Two options:
1. **Pinned venv** (recommended first try): `uv venv --python 3.11`, install relaxed deps (`pandas>=1.5 names tqdm networkx nltk pyyaml sacremoses requests matplotlib`), `pip install -e ./clutrr`.
2. **Vendor + patch** if old pandas calls break: the generator is small/pure-Python; patch the 2–3 deprecated pandas idioms.

**Generate command** (from IMPLEMENTATION-PREP §5 Step A):
```bash
cd clutrr/clutrr
python main.py --train_tasks 1.2,1.3,1.4 --test_tasks 1.5,1.6,1.7,1.8,1.9,1.10 \
  --train_rows 4000 --test_rows 1000 --holdout
```
Output lands in `data/` as CSVs with the schema in §3.

**Then:** write a loader (`experiments/fer_relational/data/loader.py`) that yields `(story, query, target)` for the main objective and `(story, triples)` for the auxiliary objective.

---

## 9. Secondary — Coding-LLM Datasets (for the main DeepSWE track)

For completeness; fully specified in VALIDATED-PLAN.md §6 and NEWER-TECHNIQUES.md. Not needed for the FER experiment.

| Dataset | Role | Size | Source |
|---------|------|------|--------|
| **R2E-Gym** | RL training environments (Docker + unit tests) | 8,700 tasks | `R2E-Gym/R2E-Gym` |
| **The Stack v2** | code pretraining | ~900B tokens | `bigcode/the-stack-v2` |
| **Nemotron-Code-v2** | synthetic code augmentation | ~340B tokens | `nvidia/Nemotron-Pretraining-Code-v2` |
| **OpenCodeReasoning** | reasoning-trace SFT | 736K traces | NVIDIA (arXiv 2504.01943) |
| **Sky-T1 / QwQ traces** | reasoning distillation | 10–17K traces | generate via QwQ-32B teacher |
| **DeepSWE** | holdout eval ONLY — never train on it | 113 tasks | `datacurve-ai/deep-swe` |

**Integrity rule (unchanged):** DeepSWE tasks must never enter training data; run leak checks before any fine-tune.

---

## 10. Summary — What the Dataset Is and What's Needed

- **What it is:** CLUTRR — procedurally generated kinship stories where the queried relationship is *never stated* and must be inferred by composing a small, closed algebra of base relations. Each example = `(story, query) → target relation`, plus a free `(entity, relation, entity)` graph for the auxiliary objective.
- **What makes it the right choice:** the train-short/test-long chain axis directly measures systematic generalization — the exact behavior our factored-vs-fractured hypothesis predicts.
- **What's needed to produce it:** a pinned Python 3.11 venv (CLUTRR's old deps break on 3.14), the generator command above (CPU, minutes), and a small loader. ~50k/10k train/test rows, a few hundred MB, no GPU for data-gen; toy training fits on local MPS.
- **What's blocking:** only the dep-pinning fix — no credentials, no GPU, no external download required. **This is unblocked and runnable locally today.**
