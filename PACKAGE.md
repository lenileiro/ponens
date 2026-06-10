# The `thinking` package — verified thinking flows

A small LM reads natural-language facts, **thinks in a typed trace language**, and a Datalog
checker validates **every emitted line** before it commits. This document is the package
reference: architecture, the trace language, worlds and exams, the CLI, GPU operations, and the
experimental record.

## 1. Architecture

```
                 facts (NL surface)            question (NL)
                        │                          │
                        ▼                          ▼
              ┌───────────────────────────────────────────┐
              │  ScratchpadLM  (pointer head, dense aux)   │   emits one LINE at a time
              └───────────────────────────────────────────┘
                        │ line
                        ▼
              ┌───────────────────────────────────────────┐
              │  GoalChecker (forward, evidence-first)     │   validates LOCALLY:
              │  · check F   → F ∈ stated facts            │   never proves anything itself
              │  · think B so H → B ⊆ known ∧ rule(H←B)    │
              │  · arithmetic heads → builtin VERIFIES     │
              │  · duplicates rejected                     │
              └───────────────────────────────────────────┘
                 valid → commit, H joins `known`              invalid → resample (≤ retry), else
                                                              answer from the verified derivation
```

**Trace language** (every token survives the word tokenizer; premises ALWAYS precede
conclusions — at line level and trace level; violating this is answer-only supervision and does
not train):

```
check <h> <pred> <t> .                          ground a stated fact
think <B1> [and <B2> ...] so <H> .              derive H from known atoms via a rule
answer <relation | value | person> .            only valid when the derivation supports it
fact <h> <pred> <t> . ... done .                reading exercise (extraction)
write <level> <h> <pred> <t> : <sentence> .     writing exercise
compute <y2> minus <y1> : <diff> .              math drill
define <level> <word> : <definition> .          vocabulary lesson
```

**Model** (`scratchpad_model.py`): 4-layer transformer, RoPE, QK-norm with learnable per-head
attention temperature, tied embeddings, and a **pointer/copy head** (output = learned-gate
mixture of the LM softmax and last-block-head-0 attention scattered onto context token ids) —
copying is architectural, and the dense aux supervision trains exactly that head. Recurrence
(looped / HRM `hier` / TRM / mHC hyper-connections) and Ouro-style learned halting are
implemented but **off by default** (ablation: shared-block recurrence loses 0.93-vs-2.1 to
distinct layers on trace corpora at this scale). **Use `d=256` for multi-rule worlds.**

**Engine** (`datalog.py`): least-fixpoint closure with provenance (gold traces = linearized
proof trees), entailment oracle, and SLD backward chaining (`prove`/`options`) for goal-directed
worlds with distractors.

## 2. Worlds

- **`chain`** — linear `r`-chains, transitive `far` closure. The minimal benchmark; rung 0.
- **`kinship`** — CLUTRR-style families. 16 interacting rules (gendered bases → parent/sibling
  abstractions → grandparents, aunts/uncles, cousins, nephews/nieces via gender witnesses,
  in-laws, recursive `ancestor`). Questions are NL; the **answer is a relation** the flow must
  derive. Queried links are enforced **rule-only** (no stated fact ever connects the pair).
  - **deep regime** (`--deep-depth N`): N-generation spines with the full relation web as
    distractors + a consistent chronology (`born`/`died` years flow down generations).
  - **value queries**: ages at death, relative ages, hypothetical "how old in YEAR" — the
    checker *verifies* the arithmetic (builtins), it never computes answers.
  - **relativity set**: `older_by` (same fact from either frame), `who_older`/`who_younger`
    (direction flips the answer; pair order randomized).
  - **novel relations**: a rule **defined only in the question** ("a relX of someone is the P1
    of a P2 of that person — who is the relX of H?"); the skill is trained on random
    compositions, evaluated on held-out ones. Answers are nouns reached by linking.

**Entity anonymization** (`world.anonymize`, default ON): all persons renamed per-example to
slot tokens `p0..p159`. Non-negotiable — never-trained name embeddings derail the model's
structural circuits even though the pointer copies them correctly.

## 3. Language: the distilled surface bank

`thinking/distill.py` has a frontier teacher (`claude -p`) write the English; every pattern is
mechanically validated (slot discipline, charset, length) and **split train/eval** so held-out
phrasings measure language understanding. Eight education registers (preschool → scholar) form a
**training curriculum** (the ladder is climbed in cumulative mixtures before the full mix), plus
leveled word definitions — which, for relation words, are *the rules stated in English*.

```bash
python -m thinking.cli train ...           # uses surfaces.json automatically when present
python -m thinking.distill --go            # regenerate the bank (frontier calls)
python -m thinking.distill --definitions   # add/refresh leveled definitions
```

## 4. Exams (the report card)

| exam | command fragment | measures |
|---|---|---|
| reasoning | `--mode verified/free --hops ...` | derivation accuracy by depth |
| composition holdout | `--split holdout` | relations never queried in training |
| novel relations | `--split novel` | rules that exist only in the question |
| reading | `--mode extract` | NL → canonical facts, F1 |
| self-grounded reasoning | `--mode self` | reasoning over the model's OWN extraction |
| writing | `--mode write` | express a fact at an education level |
| math | `--mode math` | subtraction on (almost surely) unseen operand pairs |
| vocabulary | `--mode define` | state a word's definition at a level |
| language understanding | `--phrasings eval` | HELD-OUT surface patterns |
| names | `--train-names` | train-pool vs held-out entities (harness discriminator) |

## 5. CLI

```bash
python -m thinking.cli selftest      # the correctness core: checker, parsers, aux alignment,
                                     # bank coverage, gold traces for every query type
python -m thinking.cli train  --world kinship [--simple|--bank|--canon] [--deep-depth N]
                              [--dim 256] [--steps N] [--batch N] [--no-loop] [--neg] --out RUN
python -m thinking.cli eval   RUN [--mode ...] [--split iid|holdout|novel] [--hops 2,3]
                              [--phrasings train|eval] [--level preschool..scholar|mix] [--n N]
python -m thinking.cli demo   RUN [--k N]
python -m thinking.cli induce [--out rules.json]    # discover the rules from raw observations
python -m thinking.cli ablate                       # recurrence ablation harness
python -m thinking.verbalize  [--corpus tinystories[,cosmopedia]] --out V.pt  # expressive output
```

Run directories are self-contained (`config.json`, `model.pt` with auto-captured model config,
`results.json` accumulating tagged eval cells).

Training mixture (configurable fractions): reasoning traces + reading + writing + math drills +
vocabulary + novel-relation problems, with **rolling pool refresh** (the generator is infinite —
fixed pools memorize and collapse) and an example-packed, never-truncate-mid-example batcher.

## 6. GPU operations

```bash
RUNPOD_API_KEY=... python runpod/launch_thinking.py --stair [--bank] [--canon] \
    [--stair-world chain|kinship] [--dim 256] [--train-steps N] --fast --go
```

Tar-over-ssh sync, pod-side `timeout`, always-terminate in `finally`. Operational rules learned
the hard way: logs tee to the pod's **local disk** (`/workspace` is a network volume that stalls
streaming writes); after killing a launcher, **verify pod state via the REST API** (a hung SSH
also blocks the cleanup that would terminate the pod); pre-flight any data-shape change locally
(`build_examples` dry-run) before it bills a pod.

## 7. Experimental record (the staircase)

Validated bottom-up, one variable per run, after a nine-run zero streak taught us not to stack:

| rung | config | result |
|---|---|---|
| 0 | chain, production trainer, train names | 0.95/0.80 verified k=2/4 |
| 0b | + entity anonymization | **1.00 / 1.00** (held-out = train by construction) |
| A | kinship, built-in NL, d=128 | reads + validates, think-grammar fails |
| A-d256 | + capacity | **1.00 / 0.95 free k=3** |
| B | + 8-register bank + curriculum (20k steps) | 0.45 trained / **0.40 held-out phrasings** |
| B6 | + budget (50k steps) | in flight |

Root causes found en route (now defaults/tests): unseen-name embeddings → anonymization;
head-first lines → premises-first; fixed-pool epochs → rolling refresh; d=128 → d=256;
mHC zero-init write gate (block got zero gradient); duplicate-line loops → checker dedup.

## 8. Roadmap

Rung C: deep trees at scale. Rung D: the full exam battery in one model. Then: digit-level
arithmetic traces (atomic-token math can't emit unseen differences), the verbalizer span-copy
fix (BPE fragments names), model-proposed rule induction (the enumerate-and-test learner already
hits 0.99 — the model should propose the candidates), and the self-verify flywheel: verified
traces become the next generation's training data.
