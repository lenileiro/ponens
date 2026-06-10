# Grounding & Closure — "English is just language rules"

## Thesis

Teaching English = teaching a **closed, self-grounding web of relational rules**, not next-token
statistics. Every word is *defined by its relations to other words*, recursively, until the web bottoms
out in a few axiomatic **primitives**. A dog is a mammal; a mammal is explained; and every word used to
explain it is explained too — down to the roots. This is exactly what a dictionary is, taken seriously.

Validated cornerstone (see memory `stage1-multi-relation`, `english-chat-pipeline`): when the per-step
recall is made *exact* via aux attention-supervision, externally-verified relational chaining generalizes
in depth and across relation types. So the research question is **how much of English reduces to verified
reasoning over a closed relational vocabulary** — and where it stops.

## The five design requirements (from the user) and how each is met

1. **"english is just language rules"** → the model reads relational facts and reasons+verifies; no LM
   backbone. (`relational_loop.py`)
2. **"more expressive, >10 facts per sentence"** → DENSE sentences: facts grouped by subject into
   multi-clause sentences (`relations.dense_tokens`), e.g. *"a dog is a mammal , has fur , can run ,
   seems warm , and sits in the house ; a mammal is a animal , ..."* (10 facts, 1 sentence).
3. **"explain what mammal is, unpack every describing word"** → RECURSIVE GROUNDING: every tail word must
   appear as a head (be defined), enforced to closure. (`relations.undefined_words` / `closure_report`)
4. **"more descriptive and scientific"** → harvest the REAL scientific taxonomy + glosses from WordNet,
   and (later) frontier-paraphrase into rich scientific prose. (`wordnet_kb.py`, `gen_corpus.py`)
5. **"explain everything including the RELATIONS — what does 'opposes' mean?"** → the 8 relations are
   themselves defined (`relations.REL_GLOSS`, `REL_MARKER`); marker words `is-a` to the `relation`
   primitive, so even the function words ground out.

## Architecture

```
WordNet (real lexical DB)
   │  wordnet_kb.build_kb()  -- PROCESS + EXTEND the existing dictionary.py
   ▼
relation tuples  isa / has_part / part_of / antonym  (+ glosses)         [SCIENTIFIC, CLOSED at 'thing']
   │  relations.py = schema (8 relations + inference RULES) + closure machinery + primitives
   ▼
CLOSED relational ontology  (closure_report -> 0 undefined; every word grounds to a primitive)
   │  ├─ dense_tokens()  -> expressive >10-fact sentences (the reading surface)
   │  └─ gen_corpus.py   -> frontier (Claude) paraphrases each true tuple into diverse SCIENTIFIC English
   ▼                          (surface-only; tuples stay the ground truth -> Datalog oracle sound)
relational_loop.py
   READ dense English -> recall ONE typed hop (aux-supervised, CE->~0) -> chain VERIFIED hops (Datalog)
   -> DISCOVER entailed facts at depths/compositions never trained.
```

## Files

| file | role | status |
|---|---|---|
| `relations.py` | 8-relation schema + inference `RULES` + `dense_tokens` + **closure machinery** (`PRIMITIVES`, `undefined_words`, `closure_report`, `CLOSED_ONTOLOGY`, `REL_GLOSS`) | ✅ closed (0 undefined) |
| `wordnet_kb.py` | PROCESS+EXTEND dictionary.py/WordNet → isa-chain/has_part/part_of/antonym tuples + glosses, recursively closed | ✅ 1672 facts / 1019 words from 88 seeds |
| `relational_loop.py` | dense multi-relation aux-supervised recall + verified typed chaining; depth + composition tests | ✅ dense runs; deep-decay open (below) |
| `gen_corpus.py` | frontier scientific-paraphrase of `CLOSED_ONTOLOGY` (dry-run default, oracle-sound) | ⏳ built, `--go` not run |
| `dictionary.py` | (existing) WordNet A1 genus projection + **sense disambiguation** (reuse for clean chains) | reuse |

## Primitives (grounding roots)
`thing` (top of objects, = WordNet `entity`), `quality` (properties), `part`, `action`, `relation`.
Everything else is defined via relations to these. The web is **circular** below the primitives (words
defined by words) — correct for a thesis-pure relational system; a small primitive set is the only axiom.

## Open issues / next steps
- **Dense deep-decay (honest):** dense (>10-fact) sentences make per-step recall harder (eval recall
  0.80–1.00 vs 1.00 simple); deep reasoning decays (backbone d=7 ≈ 0.3–0.45 vs flat-1.00 on simple).
  Cause: imperfect per-step under long-range binding → compounding. Fix: more training + a length/density
  curriculum to push dense per-step back to ~1.00 (same lever that fixed the simple case).
- **WordNet sense quality:** raw first-synset chains are noisy (`car → container`, `dolphin → (none)`).
  Reuse `dictionary.py`'s `CATEGORY_GENUS` / `_ground_noun` sense logic for clean scientific chains.
- **has_prop / capable_of from WordNet:** add via attribute/`also_see` + verb-genus (currently isa /
  has_part / part_of / antonym only).
- **Frontier closure expansion:** `gen_corpus` can recursively DEFINE any remaining `undefined_words`
  (worklist) — surface-only, validated before entering the oracle.
- **Scale:** train on the WordNet-derived closed KB at GPU scale (`runpod/launch_relational.py`, TODO).
```
