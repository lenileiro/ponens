# Ponens — the `thinking` package reference

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
For multimodal bridge runs, `ScratchpadLM.forward(..., prefix=embeddings)` can prepend continuous
image/audio embeddings before the token stream on **non-pointer** models; pointer models reject
prefixes because the copy distribution is defined over discrete context token ids.

**Engine** (`datalog.py`): least-fixpoint closure with provenance (gold traces = linearized
proof trees), entailment oracle, and SLD backward chaining (`prove`/`options`) for goal-directed
worlds with distractors.

**Verified decode** (`flow.py`), in the order things happen:

1. The model emits a greedy line; the checker validates it.
2. A rejected line is **resampled at temp 0.5** from the model's own distribution. Resampling is
   **prefix-pruned**: a trie over all checker-valid lines (every premise order — the checker
   accepts any body permutation, and a stricter trie silently collapses acceptance) aborts an
   attempt at the first token that cannot extend to a valid line. Identical acceptance set to
   full-line propose-then-reject; failure costs ~2 tokens instead of 18, which funds a 4× retry
   budget.
3. If every retry fails, **skip-don't-die**: the model's own rejected line advances the
   *context* (uncommitted to the checker) and decoding continues. This is load-bearing —
   regenerating from the same clean prefix just reproduces the same line until the budget dies
   (removing it collapsed verified k=3 from 0.95 to 0.10; restored in `13c5bc3`).
4. Every rejection is classified (`reject_causes` in eval output: syntax / check-not-edb /
   premise-unknown / no-rule / goal-anchor / duplicate / builtin-reject) — the taxonomy says
   where per-line error actually lives before you optimize it.

Two stronger interventions were built and measured, and both **lose** to rejection sampling on
a gold-trace-trained policy (forced valid-but-foreign lines push generation off-distribution and
the damage compounds): `--mode masked` (masked repair of rejected lines, 0.50) and
`decode="masked-full"` (every token masked to valid prefixes, 0.40) vs verified 0.80–0.95. Kept
as diagnostics. `decode="ranker"` scores the full legal-action frontier and records the oracle
action's rank — measures trace-policy quality in isolation from syntax failures.

**Representation probes** (`thinking probe`) are the FER/UFR guardrail from arXiv:2505.11581:
they are separate from answer accuracy. The probe teacher-forces verified traces, captures
proof-line hidden states, and reports same-rule reuse, cross-depth reuse, depth-index leakage,
and a heuristic `ufr_score`/`verdict`. A run can solve depth-30 rollout while still being
fractured if rule reuse is weak or tied to line position; the GPU learning-curve summary now
keeps those fields beside `deep_eval_*` accuracy.

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

`thinking.text` is the separate Text-0 rung for language understanding beyond the local kinship
surface bank. It consumes web-backed semantic records, not next-token corpora: each example pairs
natural language with a canonical meaning target, and the model must decode `extract fact ...`
traces. The first importer is SNLI premise/hypothesis inference; the target is the human label
(`entailment`, `contradiction`, or `neutral`) rendered as a canonical `nli` fact. The MultiNLI
importer uses the same canonical target across broader genres, so the text rung can learn from
more language domains without adding English rules. The HANS importer adds controlled
adversarial NLI examples for lexical-overlap, subsequence, and constituent heuristics; HANS
`non-entailment` maps to SNLI-compatible `neutral` by default so checkpoints can be scored
across both datasets. The grounded importer samples transcript-only descriptions from
`thinking.multimodal` and asks that module to render the canonical multimodal trace;
`thinking.text` parses that trace into target facts, so the text rung does not duplicate or
hard-code the image/audio fact schema. The evaluator reports teacher-forced fact accuracy,
direct semantic-head fact accuracy, free decoded fact F1/exact match, per-dataset and
per-heuristic buckets, paraphrase groups when the dataset supplies them, explicit counterfactual
records when supplied, and an NLI artifact control that scores hypothesis-only and premise-only
ablations. Large mixed-language runs can cap teacher/semantic eval with `--fact-n`,
`--kind-fact-n`, and `--artifact-n`; sampled reports explicitly mark `sampled: true`.

```bash
python -m thinking.text --selftest
python -m thinking.text --import-snli --snli-zip /private/tmp/snli_1.0.zip \
    --snli-train 20000 --snli-eval 1000 --seed 11 --out data/text_snli.jsonl
python -m thinking.text --import-mnli --mnli-zip /private/tmp/multinli_1.0.zip \
    --mnli-train 20000 --mnli-eval 2000 --seed 19 --out data/text_mnli.jsonl
python -m thinking.text --data data/text_snli.jsonl --steps 1500 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 --free-n 200 --max-new 20 \
    --out runs/text_snli.json --checkpoint runs/text_snli.pt
python -m thinking.text --import-hans --hans-train 6000 --hans-eval 3000 \
    --seed 13 --out data/text_hans.jsonl
python -m thinking.text --data data/text_hans.jsonl --eval-checkpoint runs/text_snli.pt \
    --free-n 200 --max-new 20 --out runs/text_snli_on_hans.json
python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --steps 1500 --batch 64 --d 192 --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 200 --max-new 20 \
    --out runs/text_snli_hans.json --checkpoint runs/text_snli_hans.pt
python -m thinking.text --import-grounded --grounded-train 8000 \
    --grounded-eval 1500 --grounded-counterfactual 100 --seed 17 \
    --out data/text_grounded.jsonl
python -m thinking.text --data data/text_grounded.jsonl --steps 600 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 100 --paraphrase-n 50 --counterfactual-n 50 --max-new 32 \
    --out runs/text_grounded.json --checkpoint runs/text_grounded.pt
python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --data data/text_grounded.jsonl --steps 1200 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 \
    --free-n 100 --paraphrase-n 30 --counterfactual-n 30 --max-new 32 \
    --out runs/text_snli_hans_grounded.json --checkpoint runs/text_snli_hans_grounded.pt
python -m thinking.text --data data/text_snli.jsonl --data data/text_hans.jsonl \
    --data data/text_grounded.jsonl --steps 1200 --batch 64 --d 192 \
    --layers 4 --heads 6 --semantic-w 0.75 --balance-by kind \
    --free-n 80 --kind-free-n 5 --paraphrase-n 20 --counterfactual-n 20 --max-new 32 \
    --out runs/text_snli_hans_grounded_balanced.json \
    --checkpoint runs/text_snli_hans_grounded_balanced.pt
python -m thinking.text --data data/text_snli.jsonl --data data/text_mnli.jsonl \
    --data data/text_hans.jsonl --data data/text_grounded.jsonl \
    --steps 120 --batch 48 --d 96 --layers 3 --heads 4 --semantic-w 0.75 \
    --balance-by kind --fact-n 240 --kind-fact-n 40 --artifact-n 240 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_snli_mnli_hans_grounded_smoke.json \
    --checkpoint runs/text_snli_mnli_hans_grounded_smoke.pt
python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_smoke.pt \
    --steps 40 --batch 32 --study-lr 0.0005 --semantic-w 0.75 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_smoke.json
python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_semantic_replay_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 80 --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_semantic_replay_smoke.json
python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_error_replay_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 40 --study-rounds 2 --study-strategy errors \
    --study-probe-n 512 --study-hard-max 256 \
    --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_error_replay_smoke.json
python -m thinking.text --data data/text_mnli.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_mnli_select_both_replay_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 40 --study-rounds 2 --study-strategy errors \
    --study-probe-n 512 --study-hard-max 256 --study-select-best \
    --study-score-metric both --study-retention-w 2.0 \
    --batch 32 --study-lr 0.0005 --decode-w 0 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_mnli_select_both_replay_smoke.json
python -m thinking.text --reading-data NEWER-TECHNIQUES.md \
    --steps 4 --batch 16 --d 96 --layers 2 --heads 4 \
    --latent-concept-slots 6 --latent-concept-layers 1 \
    --reading-max-tokens 96 --reading-min-tokens 8 --reading-eval-n 32 \
    --reading-study-strategy graph --reading-study-probe-n 48 \
    --reading-study-hard-max 24 --reading-study-refresh-steps 1 \
    --reading-memory-size 64 --reading-memory-w 0.05 \
    --reading-association-w 0.05 --reading-association-transitive-steps 3 \
    --reading-association-transitive-w 0.25 \
    --reading-composition-w 0.1 --reading-composition-transitive-steps 3 \
    --reading-composition-transitive-w 0.25 \
    --reading-graph-predict-w 0.2 --reading-graph-predict-transitive-steps 3 \
    --reading-graph-predict-transitive-w 0.25 \
    --reading-context-target-w 0.1 --reading-neighborhood-w 0.05 \
    --reading-neighborhood-batch 8 --reading-neighborhood-probe-n 32 \
    --reading-transition-w 0.05 --reading-transition-batch 8 \
    --reading-cluster-w 0.05 --reading-cluster-batch 16 \
    --reading-cluster-probe-n 32 \
    --out runs/text_raw_reading_graph_study_smoke.json \
    --checkpoint runs/text_raw_reading_graph_study_smoke.pt
python -m thinking.multimodal --steps 3 --batch 4 --dim 96 \
    --layers 1 --heads 4 --eval-n 20 --free-n 0 --counterfactual-n 0 \
    --free-counterfactual-n 0 --latent-concept-slots 6 \
    --latent-concept-memory-size 64 --latent-concept-memory-w 0.05 \
    --latent-concept-association-w 0.05 \
    --latent-concept-association-transitive-steps 3 \
    --latent-concept-association-transitive-w 0.25 \
    --latent-concept-composition-w 0.05 \
    --latent-concept-composition-transitive-steps 3 \
    --latent-concept-composition-transitive-w 0.25 \
    --latent-concept-graph-predict-w 0.1 \
    --latent-concept-graph-predict-transitive-steps 3 \
    --latent-concept-graph-predict-transitive-w 0.25 \
    --latent-concept-neighborhood-w 0.05 \
    --latent-concept-transition-w 0.05 --latent-concept-cluster-w 0.05 \
    --study-strategy graph --study-probe-n 8 --study-hard-max 4 \
    --study-refresh-steps 1 \
    --text-checkpoint runs/text_raw_reading_graph_study_smoke.pt \
    --out runs/m0_multimodal_graph_study_smoke.json \
    --checkpoint runs/m0_multimodal_graph_study_smoke.pt
```

Current local Text-0 SNLI baseline (20k train / 1k dev, 1.5k steps, d=192, 4 layers, 6 heads,
semantic auxiliary classification weight 0.75, learned attention readout per canonical fact key)
does **not** gate: free decoded exact/F1 is **0.515** on a 200-example free-decode sample,
teacher-forced fact-value accuracy is **0.519**, and the direct semantic head is **0.519**.
The artifact control is now more meaningful: hypothesis-only decoded accuracy is **0.408**
(`full - hypothesis_only = 0.111`), while hypothesis-only semantic-head accuracy is **0.401**
(`semantic full - hypothesis_only = 0.118`). This is useful as a measurement floor and shows the
model is extracting some premise-hypothesis signal, not a language-mastery claim.

HANS now makes that failure explicit. The SNLI checkpoint scored on HANS without retraining gets
only **0.423** teacher-forced fact accuracy, **0.429** semantic-head accuracy, and **0.430** free
decoded F1 on a 200-example sample; hypothesis-only accuracy is higher than full-input accuracy,
so this checkpoint is still shortcut-sensitive. A HANS-only control run (6k train / 3k eval,
1k steps) does gate at **0.911** teacher-forced, **0.909** semantic-head, and **0.905** sampled
free decoded F1, with lexical-overlap the weakest bucket (**0.812** teacher-forced). The mixed
SNLI+HANS run (26k train / 4k eval, 1.5k steps) is the current honest language-rung baseline:
it improves overall semantic-head accuracy to **0.722** and keeps constituent/subsequence HANS
strong, but it still fails the gate (**0.625** sampled free decoded F1, SNLI dev **0.486**
semantic-head, lexical-overlap HANS **0.595** semantic-head). This is progress toward
shortcut-resistant language understanding, not mastery.

Grounded transcript language is now a separate text target tied to the existing image/audio
world rather than another NLI label. The grounded-only run (8k train / 1.6k eval, 600 steps)
gates: **0.9895** teacher-forced fact accuracy, **0.9968** semantic-head accuracy, **0.996**
sampled free decoded F1, **0.88** sampled paraphrase consistency, and **0.996** sampled
counterfactual F1. The unified SNLI+HANS+grounded run (34k train / 5.6k eval, 1.2k steps)
improves global teacher-forced/semantic accuracy to **0.874 / 0.894** and keeps grounded
semantic accuracy high (**0.981**), but it still fails the gate: sampled free F1 is **0.742**,
sampled paraphrase consistency is **0.467**, SNLI dev semantic-head is **0.522**, and HANS
lexical-overlap semantic-head remains **0.594**. The next text rung should scale shortcut-
resistant web data and add richer semantic corpora (e.g. MultiNLI/CFQ/GeoQuery and raw-reading
material) plus stronger paraphrase and counterfactual splits.

Kind-balanced mixed training is a useful curriculum control, not a rulebook. With
`--balance-by kind`, the mixed SNLI+HANS+grounded run improves global sampled free F1 to
**0.8125** and lifts HANS lexical-overlap semantic-head accuracy to **0.786**, while keeping
grounded semantic accuracy high (**0.975**). It still fails the language gate: SNLI dev
semantic-head accuracy drops to **0.413** and sampled paraphrase consistency is only **0.65**.
This says the next step is broader language supervision and better paraphrase/counterfactual
coverage, not hand-coded English rules.

The first MultiNLI integration is now wired and measured as a smoke, not as a mastery run. A
20k-train / 2k-dev MultiNLI sample imports from the official NYU zip into
`data/text_mnli.jsonl` across ten genres. A small SNLI+MultiNLI+HANS+grounded smoke
(120 steps, d=96, sampled eval) completes end to end and writes a checkpoint before evaluation;
it does **not** gate (**0.324** sampled teacher-forced, **0.591** sampled semantic-head). The
useful outcome is infrastructure: wider web-backed semantic data, sampled large-set evaluation,
and no hard-coded English rules.

The first reading-task update path is also wired. `--study-checkpoint` loads an existing text
checkpoint, expands token embeddings and semantic heads for new reading-task vocabulary/facts,
evaluates before, fine-tunes on the reading records (plus optional `--study-replay-data`), saves
to a new checkpoint, and evaluates after. `--decode-w 0` gives a semantic-only study mode for
understanding updates that do not train the canonical trace decoder on the same step. A 40-step
full-loss MultiNLI study smoke expanded the balanced SNLI+HANS+grounded checkpoint from
**11,251** to **34,017** text tokens and improved sampled MultiNLI semantic-head accuracy from
**0.317** to **0.392**, while teacher-forced accuracy fell. The safer semantic-only replay smoke
with grounded replay improved sampled MultiNLI semantic-head accuracy from **0.342** to
**0.375**, nudged teacher-forced from **0.350** to **0.358**, and preserved grounded replay
semantic accuracy at **0.988**. `--study-strategy errors` adds an explicit self-study loop:
each round samples train records, mines examples where the semantic head is wrong, and trains on
those hard examples plus replay. In the current smoke it mined ~60% hard examples, improved
sampled MultiNLI teacher-forced accuracy from **0.350** to **0.400**, barely moved semantic-head
accuracy (**0.317** to **0.325**), and preserved grounded replay semantic accuracy at **0.988**.
`--study-select-best` now evaluates each study round and can restore the best weights using
semantic, teacher-forced, combined, or bottleneck scoring. The current combined-score smoke
(`--study-score-metric both --study-retention-w 2.0`) selected round 2 only after both sampled
MultiNLI teacher-forced accuracy (**0.350** to **0.408**) and semantic-head accuracy
(**0.333** to **0.383**) improved, with a small grounded replay drop (**0.988** to **0.983** for
both heads). It still fails the gate; this is the weight-update mechanism for future reading
curricula, not language mastery.

Raw reading is now the active reading-task path. `--reading-data` accepts JSON, JSONL, or TXT
chunks and builds schema-free text windows without answer labels or task-specific heads. Training
creates two corrupted views of each window, updates persistent latent concept memory, mines
association/composition graphs, and can choose hard examples by graph-prediction surprise.
`--reading-study-strategy graph` waits until memory and relations exist, then refreshes the study
pool from nonzero graph scores. Selection is driven by retrieval, context, neighborhood, cluster,
and graph-prediction metrics rather than a task-specific answer harness. The old task-specific
reader/import/candidate path has been removed from the code and is no longer supported here.

A local MPS smoke on `NEWER-TECHNIQUES.md` ran real optimizer steps through this path. Graph-study
refresh produced nonzero mean scores at steps 2-4 (**0.320**, **0.341**, **0.290**), final
`graph_predict_loss` was **0.384**, `memory_active` reached **64**, and the association graph held
**698** active edges. `cluster_acc` moved from **0.000** to **1.000** on the sampled probe. This is
real weight movement and graph self-study, not a claim of language mastery.

The upstream multimodal bridge now inherits the same latent concept machinery. Loading the text
checkpoint into a multimodal MPS smoke copied **28** token embeddings, **14** text tensors, and
**37** latent tensors, then used `--study-strategy graph` on multimodal concept states. Graph-study
mean scores were **0.537**, **0.205**, and **0.241** across the three steps; final
`latent_graph_predict_loss` was **0.249**, `latent_memory_active` reached **64**, and the latent
association graph held **700** active edges. The important result is architectural: text-side
self-study memory, relation graphs, and graph-prediction surprise are now available upstream to
multimodal learning.

The architecture now has a generic schema concept head in `thinking.concepts.SchemaConceptHead`.
Instead of adding another task-specific rule path, text records expose `(slot, predicate) -> value`
concepts through learned key queries and learned value embeddings. `TextFactLM` trains this with
`--fact-concept-w` and reports `fact_concept_head`; checkpoint expansion copies learned concept
queries/value embeddings by symbolic schema identity, not by array position. The multimodal
bridge now uses the same schema concept head for color/shape/pitch/timbre/envelope concept tokens,
so concept transfer, agreement, and distillation operate on the same architecture as text reading.
A concept-only SNLI check with decoder and old semantic-head losses disabled
(`--decode-w 0 --semantic-w 0 --fact-concept-w 1`) reached sampled `fact_concept_head` **0.405**
after 40 local MPS steps. This is not language mastery; it is the model-side insertion point for
understanding-oriented training rather than a harness-specific branch.

The same concept head now has a schema-generic contrastive geometry objective. Text reading runs
can add `--fact-concept-contrast-w` so records sharing a data-supplied concept value pull the same
slot states together while other values for that slot separate; reports expose
`fact_concept_geometry` nearest-same accuracy and same-vs-different cosine margin. The multimodal
bridge exposes the same mechanism as `--concept-contrast-w`, so color/shape/sound factors and text
facts improve through one learned concept geometry rather than through task-specific handling.
RunPod forwards the multimodal form as `--multimodal-concept-contrast-w` and
`--multimodal-concept-contrast-temperature`.

The geometry head also supports learned per-value prototypes. Text runs expose
`--fact-concept-prototype-w` and `--fact-concept-prototype-spread-w`; multimodal runs expose the
parallel `--concept-prototype-w` and `--concept-prototype-spread-w`. These are schema-generic:
targets come from data-supplied factor ids, while the loss only sees key-local prototype classes
and an anti-collapse margin. RunPod forwards these as `--multimodal-concept-prototype-w`,
`--multimodal-concept-prototype-spread-w`, and
`--multimodal-concept-prototype-spread-margin`.

Multimodal also mirrors the newer text concept plumbing with `--concept-prefix`,
`--concept-centroid-w`, and `--concept-state-spread-w`. The prefix path prepends schema-key
concept states to the decoder, while centroid/state-spread objectives use only data-supplied
factor ids to improve reusable geometry. RunPod forwards the same controls as
`--multimodal-concept-prefix`, `--multimodal-concept-centroid-*`, and
`--multimodal-concept-state-spread-*`.

## 3b. Image grounding: synthetic visual factors first

`thinking/vision.py` is the Image-0/Image-1 rung for applying the FER hypothesis to pixels
without losing checkability. It generates small colored-shape scenes with exact canonical facts
(`fact p0 color red .`, `fact p0 shape circle .`, `fact p0 left_of p1 .`), trains a compact
vision encoder to recover color/shape factors, and reports visual FER/UFR probes:
same-color reuse across shapes, same-shape reuse across colors, color-shape leakage, and a
`ufr_score`/`verdict`. The encoder now has two architecture choices:

- `shared`: one representation with separate color/shape heads, kept as the historical baseline.
- `bottleneck`: explicit color and shape subspaces plus a decorrelation loss, so diagnostics can
  inspect the actual spaces that feed the factor heads.

```bash
python -m thinking.vision --selftest
python -m thinking.vision --train --steps 2000 --batch 64 --dim 64 \
    --out runs/vision_object_encoder.pt
python -m thinking.vision --train --arch bottleneck --steps 2000 --batch 64 --dim 64 \
    --out runs/vision_bottleneck.pt
python -m thinking.image2 --steps 400 --seeds 0,1,2 --out runs/image2_bottleneck.json
python -m thinking.image_flow --train --steps 400 --out runs/image_flow.pt
python -m thinking.image_latent --train --ae-steps 400 --flow-steps 400 \
    --out runs/image_latent_flow.pt
python -m thinking.image_latent --train --flow-arch dit --ae-steps 400 --flow-steps 400 \
    --cond-drop 0.1 --cfg-scale 1.5 --cfg-rescale 0.7 \
    --sample-steps 8 --flow-semantic-w 0.25 \
    --out runs/image_latent_dit.pt
python -m thinking.image_latent --train --cond-mode text --flow-arch mmdit \
    --ae-steps 400 --flow-steps 400 --cond-drop 0.1 --cfg-scale 1.5 \
    --cfg-rescale 0.7 \
    --sample-steps 8 --flow-semantic-w 0.25 --time-sampling logit-normal \
    --flow-ema-decay 0.999 --ae-intervention-w 0.1 --ae-factor-orth-w 0.05 \
    --dit-mlp swiglu \
    --flow-noise-coupling sliced_ot --flow-noise-coupling-projections 4 \
    --semantic-guidance-w 2.0 --sample-churn 0.05 --sample-churn-interval 0.0,0.8 \
    --out runs/image_latent_mmdit_text.pt
python -m thinking.image_fetch --source text-to-image-2m-512-2m \
    --max-records 1024 --image-dir data/images/web_fetch \
    --manifest data/images/train_web.jsonl --root data/images \
    --report-out runs/image_fetch_report.json
python -m thinking.image_caption --manifest data/images/train_web.jsonl \
    --root data/images --backend hf --model Salesforce/blip-image-captioning-large \
    --mode replace --batch 16 --device cuda \
    --out data/images/train_web_captioned.jsonl \
    --report-out runs/image_caption_report.json
python -m thinking.image_score --manifest data/images/train_web_captioned.jsonl \
    --root data/images --backend stats --image-size 256 \
    --out data/images/train_web_scored.jsonl \
    --sidecar-out data/images/train_web_quality_scores.jsonl \
    --report-out runs/image_score_report.json
python -m thinking.image_embed --manifest data/images/train_web_scored.jsonl \
    --root data/images --backend hf --model google/siglip-base-patch16-224 \
    --features both --text-embed-mode both \
    --text-sequence-model google-t5/t5-base --batch 64 --device cuda \
    --out data/images/embeddings.jsonl \
    --report-out runs/image_embed_report.json
python -m thinking.image_data --manifest data/images/train_web_scored.jsonl \
    --root data/images --min-side 256 --max-aspect 2.0 \
    --max-nsfw 0.2 --max-watermark 0.5 --min-image-text-cosine 0.15 \
    --max-image-duplicate-cosine 0.985 \
    --embedding-manifest data/images/embeddings.jsonl --embedding-key image \
    --min-caption-tokens 3 --write-filtered data/images/train_clean.jsonl \
    --report-out runs/image_manifest_report.json
python -m thinking.image_latent --train --cond-mode text --flow-arch mmdit \
    --image-manifest data/images/train_clean.jsonl --image-root data/images \
    --caption-cond-source auto \
    --image-quality-weight 1.5 \
    --size 64 --ae-arch residual --latent-downsample 8 --latent-max-tokens 128 \
    --dit-mlp swiglu --latent-patch-size 2 \
    --ae-recon-loss hybrid --ae-grad-w 0.1 --ae-ms-w 0.1 --ae-fft-w 0.05 \
    --image-text-align-w 0.1 --flow-text-align-w 0.05 --text-embed-dim 128 \
    --image-feature-align-w 0.1 --flow-feature-align-w 0.05 \
    --image-feature-embed-dim 128 \
    --flow-repa-w 0.05 --flow-repa-mode auto \
    --flow-self-repa-w 0.05 --flow-self-repa-mode auto \
    --flow-sra-w 0.05 --flow-sra-mode both --flow-sra-time-gap 0.25 \
    --ae-accum-steps 2 --flow-accum-steps 2 --grad-clip 1.0 \
    --flow-cache-latents --flow-cache-dir runs/image_manifest_cache \
    --flow-cache-shard-size 2048 --flow-cache-batch 32 --flow-cache-dtype bf16 \
    --flow-cache-max-loaded-shards 4 \
    --ae-steps 400 --flow-steps 400 --sample-steps 8 \
    --flow-consistency-w 0.05 --flow-endpoint-w 0.1 \
    --flow-distill-steps 20000 --flow-guidance-distill-w 0.1 \
    --flow-guidance-distill-cfg-scale 1.5 \
    --flow-noise-coupling sliced_ot --flow-noise-coupling-projections 4 \
    --sample-grid-out runs/image_manifest_grid.ppm \
    --out runs/image_manifest_mmdit.pt
python -m thinking.image_latent --eval-checkpoint runs/image_manifest_mmdit.pt \
    --eval-image-manifest data/images/train_clean.jsonl --eval-image-root data/images \
    --eval-image-split eval --size 64 --cfg-scales 1.0,1.25,1.5,2.0 \
    --cfg-rescales 0.0,0.7 --sample-steps-list 4,8,16 --sample-churns 0.0,0.05 \
    --eval-seeds 1,2,3 \
    --sample-grid-out runs/image_manifest_eval_grid.ppm \
    --sample-manifest-out data/images/generated_captioned.jsonl \
    --eval-out runs/image_manifest_mmdit_sweep.json
python -m thinking.image_embed --manifest data/images/generated_captioned.jsonl \
    --root data/images --backend hf --model google/siglip-base-patch16-224 \
    --features both --text-embed-mode pooled --batch 64 --device cuda \
    --out data/images/generated_embeddings.jsonl \
    --report-out runs/generated_image_embed_report.json
python -m thinking.image_eval \
    --real-manifest data/images/train_clean.jsonl \
    --generated-manifest data/images/generated_captioned.jsonl \
    --generated-embedding-sidecar data/images/generated_embeddings.jsonl \
    --embedding-key image --max-records 2048 \
    --report-out runs/image_eval_report.json
python -m thinking.image_score --manifest data/images/generated_captioned.jsonl \
    --root data/images --backend ensemble --technical-w 0.3 \
    --external-sidecar data/images/generated_reward_scores.jsonl \
    --external-score-field reward_score --external-w 0.7 \
    --out data/images/generated_scored.jsonl \
    --report-out runs/generated_score_report.json
python -m thinking.image_preferences --manifest data/images/generated_scored.jsonl \
    --root data/images --group-by prompt_id,prompt,caption --mode top-bottom \
    --min-score-gap 0.05 --out data/images/generated_preferences.jsonl \
    --report-out runs/generated_preferences_report.json
python -m thinking.image_latent --eval-checkpoint runs/image_latent_dit.pt \
    --cfg-scales 1.0,1.25,1.5,2.0 --cfg-rescales 0.0,0.7 \
    --sample-steps-list 4,8,16 \
    --eval-seeds 1,2,3 --roundtrip-samples 2 --eval-out runs/image_latent_dit_sweep.json
python -m thinking.audio --steps 500 --seeds 0,1,2 --out runs/audio1_fer.json
python -m thinking.multimodal --steps 400 --out runs/m0_multimodal.json
RUNPOD_API_KEY=... python runpod/launch_thinking.py --vision --vision-arch bottleneck \
    --image2 --image-flow --image-latent --image-latent-arch dit \
    --image-cond-drop 0.1 --image-cfg-scale 1.5 --image-cfg-rescale 0.7 \
    --image-sample-steps 8 \
    --image-flow-semantic-w 0.25 --image-eval-sweep \
    --audio --multimodal --fast --go
RUNPOD_API_KEY=... python runpod/launch_thinking.py --image-latent --image-latent-arch mmdit \
    --image-cond-mode text --image-dit-head-width-mult 2 \
    --image-cond-drop 0.1 --image-cfg-scale 1.5 --image-cfg-rescale 0.7 \
    --image-sample-steps 8 --image-flow-semantic-w 0.25 \
    --image-flow-consistency-w 0.05 --image-flow-endpoint-w 0.1 \
    --image-time-sampling logit-normal --image-flow-ema-decay 0.999 \
    --image-dit-mlp swiglu \
    --image-latent-normalize channel --image-latent-stat-samples 1024 \
    --image-cfg-interval 0.0,0.8 --image-semantic-guidance-interval 0.0,0.75 \
    --image-ae-intervention-w 0.1 --image-ae-factor-orth-w 0.05 \
    --image-semantic-guidance-w 2.0 --image-semantic-guidance-sweep 0.0,1.0,2.0 \
    --image-sample-churn 0.05 --image-sample-churns 0.0,0.05 \
    --image-sample-methods euler,heun --image-cfg-rescale-sweep 0.0,0.7 \
    --image-sample-grid \
    --image-eval-sweep --fast --go
RUNPOD_API_KEY=... python runpod/launch_thinking.py --upload-image-data --image-embed \
    --image-embed-model google/siglip-base-patch16-224 \
    --image-embed-text-mode tokens \
    --image-clean-min-side 256 --image-clean-max-aspect 2.0 \
    --image-latent --image-latent-arch mmdit \
    --image-cond-mode text --image-dit-head-width-mult 2 --image-dit-qk-norm \
    --image-dit-attn-impl sdpa --image-dit-pos-embed rope2d --image-dit-mlp swiglu \
    --image-manifest data/images/train.jsonl --image-root data/images \
    --image-caption-cond-source auto \
    --image-crop-mode pad --image-hflip-prob 0.5 \
    --image-quality-weight 1.5 \
    --image-size 128x192 --image-size-buckets 128x128,128x192,192x128 \
    --image-ae-arch residual --image-latent-downsample 8 \
    --image-latent-patch-size 2 \
    --image-latent-max-tokens 384 \
    --image-ae-recon-loss hybrid --image-ae-grad-w 0.1 --image-ae-ms-w 0.1 \
    --image-ae-fft-w 0.05 \
    --image-text-align-w 0.1 --image-flow-text-align-w 0.05 \
    --image-text-embed-dim 128 \
    --image-feature-align-w 0.1 --image-flow-feature-align-w 0.05 \
    --image-feature-embed-dim 128 \
    --image-flow-repa-w 0.05 --image-flow-repa-steps 20000 \
    --image-flow-repa-mode auto \
    --image-flow-repa-embed-dim 128 \
    --image-flow-self-repa-w 0.05 --image-flow-self-repa-steps 20000 \
    --image-flow-self-repa-mode auto \
    --image-flow-self-repa-embed-dim 128 \
    --image-flow-sra-w 0.05 --image-flow-sra-steps 20000 \
    --image-flow-sra-mode both --image-flow-sra-time-gap 0.25 \
    --image-quality-score-w 0.1 --image-flow-quality-score-w 0.1 \
    --image-quality-score-steps 2000 \
    --image-quality-score-rank-w 0.05 --image-flow-quality-rank-w 0.05 \
    --image-ae-accum-steps 2 --image-flow-accum-steps 2 \
    --image-train-precision bf16 --image-grad-clip 1.0 \
    --image-flow-cache-latents --image-flow-cache-dir runs/image_manifest_cache \
    --image-flow-cache-shard-size 2048 --image-flow-cache-batch 64 \
    --image-flow-cache-dtype bf16 --image-flow-cache-max-loaded-shards 4 \
    --image-sample-steps 8 --image-flow-consistency-w 0.05 \
    --image-flow-endpoint-w 0.1 --image-flow-noise-coupling sliced_ot \
    --image-flow-noise-coupling-projections 4 \
    --image-time-sampling logit-normal --image-time-shift 1.25 \
    --image-time-shift-mode dim --image-time-shift-ref-dim 1024 \
    --image-flow-loss-weight min-snr-v --image-flow-loss-weight-gamma 5.0 \
    --image-cfg-rescale 0.7 --image-cfg-rescale-sweep 0.0,0.7 \
    --image-latent-normalize channel --image-latent-stat-samples 4096 \
    --image-eval-sweep --image-eval-split eval --image-sample-grid --fast --go
```

`thinking.image2` is the head-aware FER experiment: shared factored heads vs explicit
bottleneck vs one joint color×shape classifier on held-out color/shape combinations. It reports
both the embedding probe and the factor-space probe, fixing the Image-1 blind spot where
the embedding layer missed where factorization lived.

`thinking.image_flow` is the first generation scaffold. It trains a small fact-conditioned
rectified-flow model in pixel space:

```
canonical visual facts -> condition vector -> velocity field -> image
```

`thinking.image_latent` is the next scale rung:

```
image -> semantic AE latent -> fact-conditioned latent velocity field -> decoder -> image
```

The autoencoder is deliberately semantic, not reconstruction-only: it predicts color/shape facts
from the compressed latent while reconstructing the image. That keeps the latent aligned with the
same factorization requirement as the FER probes and gives us the right insertion point for a
future DiT/MMDiT backbone.

This follows the direction used by modern image systems: latent diffusion reduces generation
cost by operating in compressed spaces (LDM, arXiv:2112.10752), diffusion transformers replace
the U-Net backbone at scale (DiT, arXiv:2212.09748), and current high-resolution systems use
rectified-flow transformer variants (Stable Diffusion 3, arXiv:2403.03206) or linear-attention
efficiency paths (Sana, arXiv:2410.10629). Newer RAE-style work pushes the same point harder:
the autoencoder latent should be semantically rich enough for DiT training, not just a lossy
pixel codec. Our implementation is still synthetic, but the path now matches that shape.

Current finding: the H100 Image-1 baseline reaches **1.00 color / 1.00 shape accuracy** after
2k steps, but the visual factor probe still reports `high_fer_risk` (`ufr_score=0.0`) because
color and shape remain strongly entangled. A two-arm held-out-combo experiment also shows the
factored head generalizes better than a joint color×shape classifier (≈0.30 vs 0.00 holdout),
while the embedding-layer probe alone misses where that factorization lives. So for image, as
for trace reasoning, label accuracy is not enough; probes must inspect the right layer/head.

Image-2 GPU update (H100, 1000 steps, seed 0): the explicit bottleneck keeps **1.00 seen
accuracy**, reaches **0.84 held-out color/shape combo accuracy** vs **0.61** for shared factored
heads and **0.00** for the joint classifier, and cuts factor-space leakage from **0.75 → 0.10**.
The factor-space probe now tracks behavior (`best_factor_space_arm = bottleneck`), while the
shared embedding probe still reports `ufr_score=0.0` because the concatenated representation is
still entangled. `thinking.image_flow` also trains the first fact-conditioned rectified-flow
scaffold to `velocity_mse ≈ 0.29` after 1000 steps; this proves the generation path is wired, not
that image quality is solved.

A separate tracked Image-2 sweep (`runs/image2_fer.json`, 600 steps × 3 seeds) records the
stronger intervention target: held-out combo accuracy moves **0.65 → 0.97 → 1.00** across
factored, swap, and subspace arms, and the linear-decodability probe ranks the arms in the same
order as behavior. Treat that as the next reproducibility target for the public command path.

Image-3 GPU update (H100, 1000 AE steps + 1000 latent-flow steps): `thinking.image_latent`
with the conv velocity baseline reaches `recon_mse=0.015`, latent color accuracy **1.00**, latent
shape accuracy **0.86**, `latent_velocity_mse=1.05`, and center-target sample MSE **0.044**. This
proves the semantic latent path can preserve canonical factors before moving transport training
off pixels. The module now also has `--flow-arch dit`, a tiny patch-token DiT velocity field over
the same semantic latents. That is the toy-scale version of the architecture we need to scale:
semantic image latents + fact/text conditioning + rectified-flow transformer.

Image-4 GPU update (H100, 1000 AE steps + 1000 DiT-flow steps): the patch-token DiT flow keeps the
same semantic AE quality (`recon_mse=0.015`, color **1.00**, shape **0.86**) and improves the
transport objective vs the conv baseline: `latent_velocity_mse` **1.05 → 0.75** and center-target
sample MSE **0.044 → 0.020**. This makes DiT the default scale direction; the conv flow remains a
small baseline.

The image generator now also has the first generation-faithfulness check: every color×shape
condition is sampled, decoded back to pixels, re-encoded by the semantic AE, and scored for
round-trip color/shape accuracy. The same code path supports classifier-free condition dropout
and guided latent sampling (`--cond-drop`, `--cfg-scale`, `--sample-steps`), so the GPU rung can
measure whether stronger conditioning improves generated-image facts rather than only lowering a
single target-image MSE.

Image-5 GPU update (H100, 1000 AE steps + 1000 DiT-flow steps, `cond_drop=0.1`,
`cfg_scale=1.5`, 8 sample steps): guided latent DiT keeps the earlier reconstruction quality
(`recon_mse=0.015`, color **1.00**, shape **0.86**) and center-target sample MSE remains strong
at **0.0198**. The new round-trip metric is the important finding: generated samples decode back
to the requested color **0.87**, shape **0.67**, and both facts **0.53** across all 15 color×shape
conditions. That means the architecture is now measuring image fact-faithfulness, and the next
image work should attack generated shape fidelity rather than only optimizing latent velocity MSE.

The next image intervention is a REPA-style semantic endpoint loss for the flow: the predicted
clean latent endpoint is read by the frozen semantic AE fact heads and optimized against the
conditioning facts (`--flow-semantic-w`). This is not a hard-coded shape rule; it is a generic
alignment pressure from condition slots to the representation heads already used for evaluation.
It follows the recent lesson that image generation quality depends on semantic representations
inside the denoising/flow model, not only lower transport MSE.

Image-6 GPU update (H100, same 1000+1000 DiT setup, `flow_semantic_w=0.25`): semantic endpoint
alignment fixes the shape-fidelity bottleneck measured by Image-5. Round-trip generated samples
now decode to color **0.87**, shape **1.00**, and both facts **0.87** across all 15 color×shape
conditions, up from **0.87 / 0.67 / 0.53**. The tradeoff is small but real: latent velocity MSE
is **0.78** vs **0.77**, center-target sample MSE moves **0.0198 → 0.0251**, and conditional
sample MSE moves **0.0739 → 0.0848**. For this foundation, the result says representation
alignment is buying semantic faithfulness at a modest pixel-MSE cost.

Image-7 robust checkpoint sweep (`--eval-checkpoint`, no retraining): sampler choice is now
measurable as a Pareto problem without rerunning training. On the Image-6 checkpoint, sweeping
CFG `{1.0,1.25,1.5,2.0}` and steps `{4,8,16}` across eval seeds `{1,2,3}` with two samples per
condition finds the best aggregate fact-faithfulness at `cfg=1.5, steps=4`: color **0.967 ±
0.027**, shape **0.944 ± 0.016**, both facts **0.911 ± 0.031**, and conditional sample MSE
**0.065**. Higher guidance (`cfg=2.0, steps=4`) slightly improves shape mean (**0.944**) but
hurts color and pixel distance (`both=0.900`, MSE **0.095**). The earlier one-seed `cfg=2.0`
winner was therefore too brittle; robust selection points to moderate CFG plus few Euler steps.

Image-8 starts the language-to-image path without hard-coding generator rules: the latent flow can
now take `--cond-mode text`, where a learned prompt encoder maps prompt tokens into the continuous
condition vector consumed by the DiT velocity field. Canonical facts still supervise semantic
endpoint alignment during training, but checkpoint save/load and sampler sweeps now preserve the
prompt vocabulary and text encoder, so inference no longer needs a hand-built fact vector.

Image-9 adds `--flow-arch crossdit`: image latent tokens now cross-attend to prompt/fact condition
tokens instead of receiving only one pooled condition vector. This is still toy-scale, but it moves
the scaffold toward the token-to-token conditioning pattern used by modern rectified-flow T2I
transformers and gives us a concrete place to plug in richer text encoders later.

Image-10 adds paper-aligned timestep sampling for rectified-flow training: `--time-sampling
logit-normal` biases interpolation times toward intermediate noise scales while keeping `uniform`
available as the baseline. The report records the sampled time mean/std in `last_flow`, so GPU
sweeps can compare schedule effects against sampler/CFG effects.

Image-11 adds `--flow-arch mmdit`: a tiny dual-stream multimodal DiT flow with separate image and
condition-token projections, joint attention, and modality-specific feed-forwards. Unlike the
one-way `crossdit` scaffold, prompt/fact tokens can also absorb image-token state inside each block,
matching the bidirectional multimodal token-mixing direction of modern rectified-flow T2I systems.

Image-12 adds condition-adaptive modulation to MM-DiT blocks: the time/text condition vector now
modulates the attention and feed-forward normalizations for both image and condition streams. The
adapter is zero-initialized, so it starts as the previous MM-DiT block and learns stronger
conditioning without destabilizing checkpoint-smoke training.

Image-13 adds AdaLN-style residual gates to the adaptive MM-DiT branches. Attention and feed-forward
updates in both streams now have condition-dependent gates. The flow loader is strict about MM-DiT
gate weights so checkpoint state matches the active architecture exactly.

Image-14 adds EMA checkpointing/evaluation for the latent flow and learned text conditioner. Train
reports can evaluate the averaged weights, checkpoints save both raw and EMA states, and
`--eval-checkpoint` prefers EMA when available. Use `--checkpoint-weight-mode raw` for raw-weight
comparisons.

Image-15 fixes the short-run EMA failure found on the first MMDiT text GPU smoke: a target decay of
0.999 made the averaged checkpoint lag raw weights badly at 400 steps. EMA now uses a generic warmup
schedule that ramps toward the requested decay, records the effective decay in reports/checkpoints,
and keeps `--no-ema-warmup` / `--image-no-ema-warmup` for exact-decay ablations.

Image-16 makes raw-vs-EMA evaluation measured instead of assumed. Train reports and
`--eval-checkpoint` now support `raw`, `ema`, and `auto` weight modes; `auto` evaluates the available
weight candidates and selects by the existing semantic round-trip / MSE sweep score. This prevents a
short-run EMA checkpoint from hiding a faster-learning raw model while still allowing EMA to win when
the metrics support it.

Image-17 adds a FER/UFR latent intervention diagnostic. The probe learns fact-value prototype
directions from encoded examples, edits one requested fact in latent space, decodes/re-reads the
image, and reports target-change accuracy plus collateral stability. It is data-derived rather than
a renderer rule. On the fetched 400-step MMDiT text H100 checkpoint, a 16-sample smoke reports
`latent_intervention_score=0.408`: color directions are strong, while shape edits still disturb color
too often. That gives the next representation-quality target beyond output MSE and round-trip score.

Image-18 adds an optional semantic-AE intervention training loss (`--ae-intervention-w`; RunPod:
`--image-ae-intervention-w`). It uses batch-derived fact prototype deltas, supervises edited target
facts, and penalizes collateral drift after decode/re-encode. A local 60/10-step smoke moved
`latent_intervention_score` from **0.079 -> 0.102** with similar reconstruction MSE, so this is now
the recommended next H100 knob while remaining opt-in for ablations.

Image-19 H100 update (800 AE + 800 MMDiT text-flow steps, `ae_intervention_w=0.1`): the intervention
regularizer is a real representation win. The train report reaches `latent_intervention_score=0.705`
with target image-edit accuracy **0.820** and collateral stability **0.859**, up from the previous
fetched 400-step diagnostic score **0.408**. The checkpoint sweep selects EMA at `cfg=1.5, steps=4`
and reports color **1.00**, shape/both **0.778 ± 0.137**, and conditional sample MSE **0.056**.
Shape edits are still the weak axis, but the image stack now has a generic learned pressure that
improves editable semantic latents rather than only lowering reconstruction or flow MSE.

Image-20 adds a generic latent factor-orthogonality diagnostic and optional training loss
(`--ae-factor-orth-w`; RunPod: `--image-ae-factor-orth-w`). It estimates fact-value prototype
subspaces from the batch and penalizes squared cosine overlap between different fact predicates, so
the pressure is over `FACT_VOCAB` groups rather than color/shape rules. A 60/10-step local smoke
reduced eval `latent_factor_orth_loss` from **0.156 -> 0.131** with comparable reconstruction MSE;
the tiny intervention score did not move, so this is a representation-health knob to combine with
the intervention loss on the next H100 run, not yet a standalone quality claim.

Image-21 adds sampling-time semantic guidance (`--semantic-guidance-w`; RunPod:
`--image-semantic-guidance-w`). It is classifier-style guidance, but the classifier is the learned
semantic AE itself: each Euler step can take a normalized gradient through either latent heads or
decode/re-read heads toward the requested canonical facts. On the fetched H100 MMDiT text checkpoint,
decoded guidance at `w=2.0`, `cfg=1.5`, `steps=4`, seeds `{1,2,3}` moves generated shape/both
round-trip accuracy from **0.800 -> 0.978** while color stays **1.00**; conditional sample MSE moves
from **0.0509 -> 0.0562**. This directly attacks the remaining shape-fidelity gap without adding
renderer rules.

Image-22 makes semantic guidance a measured sweep axis (`--semantic-guidance-weights`; RunPod:
`--image-semantic-guidance-sweep`) rather than a fixed post-hoc knob. On the fetched H100 checkpoint,
the same three-seed sweep over `{0, 1, 2}` selects `sem=2`: both-fact accuracy moves
**0.800 -> 0.933 -> 0.978** as guidance increases, while conditional MSE moves
**0.0509 -> 0.0522 -> 0.0562**. This turns semantic guidance into a Pareto selection problem beside
CFG and sampler steps.

Image-23 makes the rectified-flow ODE solver a measured sweep axis (`--sample-methods`; RunPod:
`--image-sample-methods`) with Euler and Heun methods. On the fetched H100 checkpoint, the joint
`{euler, heun} × {sem=0,1,2}` sweep still selects `euler;sem=2`: Heun reaches the same both-fact
accuracy at each semantic-guidance level but has higher conditional MSE on this straightened toy
flow. That keeps Euler as the current default while giving future less-straight flows a fair solver
comparison.

Image-24 adds dependency-free PPM sample-grid export (`--sample-grid-out`; RunPod:
`--image-sample-grid`). The grid uses the same selected sampler/guidance settings as checkpoint
evaluation, so fact metrics and visual artifacts can be audited together. This is necessary for
image-generation work: the fact heads can say shape is correct, but the artifact makes pixel-level
failures visible.

Image-25 adds a lightweight wide DiT velocity head (`--dit-head-width-mult`; RunPod:
`--image-dit-head-width-mult`) for DiT, Cross-DiT, and MM-DiT flows. The default is the compact
linear head; setting the multiplier above 1 gives the transformer a wider projection head for
semantically rich latents. This follows the RAE/REPA direction: improve the model's capacity to
operate in representation space without adding renderer-specific rules.

Image-26 adds checkpointed latent normalization (`--latent-normalize none|global|channel|auto`;
RunPod: `--image-latent-normalize`). The flow can now train and sample in normalized AE latent
coordinates while semantic losses, guidance, decoding, and visual grids still operate in raw AE
latent space. This is a generic scale fix for semantically rich/high-dimensional latents: it
stabilizes coordinate scale without hard-coding visual factors or renderer rules. The default is
`auto`, which preserves `none` for synthetic factor runs and resolves to channel statistics for
real-image/external-VAE runs.

Image-27 adds guidance intervals (`--cfg-interval`, `--semantic-guidance-interval`; RunPod:
`--image-cfg-interval`, `--image-semantic-guidance-interval`). CFG and semantic AE guidance can
now be active over selected rectified-flow time ranges. This follows the REPA/guidance-interval
direction: guidance becomes a measured sampler schedule rather than a hard-coded every-step push,
and it stays generic over conditions.

Image-28 adds an optional endpoint-consistency regularizer (`--flow-consistency-w`; RunPod:
`--image-flow-consistency-w`) and eval metrics (`latent_endpoint_mse`,
`latent_endpoint_consistency_mse`, `latent_endpoint_time_gap`). The loss asks two different flow
times on the same noise/data path to predict the same clean latent endpoint. This targets the
few-step image-generation bottleneck without adding renderer-specific grammar: if the path is
more self-consistent, Euler/Heun sweeps should need fewer steps to preserve facts and pixels.

Image-29 starts the real image-text data bridge. `thinking.image_data` reads JSONL/CSV/TSV
captioned-image manifests, supports dependency-free PPM fixtures locally, and uses Pillow for
JPEG/PNG/WebP on GPU boxes. `thinking.image_latent --image-manifest ... --cond-mode text` now
trains the same semantic AE + text-conditioned latent flow on captioned images instead of only
synthetic color/shape renders. Manifest JSONL rows need `image`/`path` plus `caption`/`text`, with
optional `split`, `aesthetic`, `width`, and `height`. This is the necessary data-plane step toward
SOTA-quality images: synthetic factors remain the controllable probe, but real image/caption data
is now a first-class training source.

Image-30 makes real image-text checkpoint evaluation explicit. `thinking.image_latent
--eval-checkpoint ... --eval-image-manifest ... --eval-image-split eval` now runs CFG/step/method
sweeps on held-out captioned records and reports `caption_sample_mse`, reconstruction, velocity,
and endpoint-consistency metrics. Manifest checkpoints no longer silently fall back to the
synthetic color/shape evaluator, and RunPod `--image-eval-sweep` now uses the held-out manifest
path when `--image-manifest` is present.

Image-31 adds manifest QA and cleaned-manifest export. `python -m thinking.image_data --manifest
...` reports split counts, extension counts, caption length stats, dimensions/aspect stats,
duplicates, missing/corrupt files, low-resolution rows, and rejection examples, then can write a
filtered JSONL with `--write-filtered`. This makes the real image data path auditable before GPU
training, which is a prerequisite for scaling toward high-quality image generation rather than
training on silent path/caption/data-quality failures.

Image-32 adds a residual representation autoencoder path. `thinking.image_latent --ae-arch
residual --latent-downsample 8` uses residual encoder/decoder stages, records the actual latent
grid and token count, reloads from checkpoint metadata, and replaces hard-coded 4x latent
assumptions with the AE's own latent shape. `--size` and `--latent-max-tokens` are now first-class
CLI/RunPod controls, so real-image runs can move beyond 32px while keeping DiT/MM-DiT token counts
explicit.

Image-33 adds dependency-free AE reconstruction-quality controls. `--ae-recon-loss
mse|l1|hybrid`, `--ae-grad-w`, `--ae-ms-w`, `--ae-fft-w`, and `--ae-latent-reg-w` let real-image
runs optimize pixel fidelity, image gradients, multi-scale structure, frequency-spectrum detail,
and latent magnitude without adding LPIPS or discriminator dependencies. This is not a replacement
for full perceptual/adversarial AE training, but it moves the local/RPU smoke path away from plain
MSE compression and records the loss configuration in reports/checkpoints.

Image-34 adds GPU training-scale controls for latent image runs. `--ae-accum-steps`,
`--flow-accum-steps`, `--train-precision fp32|bf16|fp16`, and `--grad-clip` make the per-step
effective batch explicit, enable CUDA AMP for bf16/fp16, and clip gradients after accumulation.
CPU tests stay fp32 even if bf16/fp16 is requested, while H100 RunPod jobs can now use bf16 and
larger effective batches without changing the model objective.

Image-35 adds optional cached-latent flow training for manifest runs. `--flow-cache-latents`
encodes the post-AE training manifest once, stores raw AE latents and caption token IDs on CPU,
uses the cache for latent normalization stats, and trains the flow from cached latents instead of
reloading images and re-running `ae.encode` every microstep. This follows the latent-diffusion
separation between compression and generative-model training while keeping the cache opt-in.
`--flow-cache-dtype fp32|bf16|fp16` stores cached latents in reduced precision when requested; sampled
training latents are cast back to float32 for stable CPU and CUDA AMP execution. Cache use is
auditable via `flow_cache_*` report/checkpoint fields.

Image-36 adds a generic caption-image alignment objective for real-image manifests.
`--image-text-align-w` trains a contrastive bridge between AE latents and caption conditions during
compression, while `--flow-text-align-w` applies the same paired caption signal to predicted clean
flow endpoints. Eval/checkpoint sweeps now report caption retrieval and generated-caption
retrieval accuracy when the aligner is present. This gives real-image runs a prompt-alignment
metric and loss without adding color/shape grammar or renderer-specific labels.

Image-37 adds manifest-level precomputed text embedding conditioning. Rows may now include
`text_embedding` / `caption_embedding` / `embedding` arrays, and `--caption-cond-source
tokens|embedding|auto` chooses whether real-image training uses the local token encoder or projects
those external embeddings into the conditioning stream. The embedding path now also accepts
`text_embedding_sequence` token matrices, giving Cross-DiT/MM-DiT flows a full external text-token
stream instead of only a pooled caption vector. This is the hook for CLIP/T5/SigLIP-style frozen
text features: the model can move toward stronger prompt understanding without baking in a specific
provider or adding renderer-specific rules.

Image-38 adds disk-backed manifest latent caching. `--flow-cache-dir` writes AE latents,
caption token IDs or text embeddings, and shard metadata to disk, then flow training samples from
those shards instead of keeping the full cache resident in CPU RAM. Reports/checkpoints now expose
`flow_cache_backend`, `flow_cache_dir`, `flow_cache_shards`, shard size, latent dtype, and byte
count. This keeps the latent-diffusion separation practical for larger real-image manifests:
encode pixels once, train the rectified-flow transformer from reusable latent shards, and avoid a
hidden RAM ceiling.

Image-39 adds manifest-level visual embedding alignment. Rows may include `image_embedding` /
`visual_embedding` / `vision_embedding` / `clip_image_embedding` / `dino_embedding` arrays, and
`--image-feature-align-w` plus `--flow-feature-align-w` contrastively align AE latents and
predicted clean flow endpoints against those external image features. This is the generic RAE-style
hook: plug in DINO/SigLIP/CLIP/MAE features from a separate preprocessing job, train the generator
latents to preserve high-level visual semantics, and keep the core model free of provider-specific
or dataset-specific labels.

Image-40 makes that preprocessing path concrete. `thinking.image_data` now accepts
`--embedding-manifest`, `--embedding-key image|caption|basename`, and `--embedding-overwrite` so an
external feature job can write a sidecar JSONL/CSV/TSV and the validator can join text/image
embeddings into the cleaned manifest before training. The join report records matched/missing
rows, duplicate sidecar keys, written/preserved embeddings, and embedding dimensions, which keeps
CLIP/SigLIP/DINO/MAE preprocessing auditable and reproducible instead of implicit notebook state.

Image-41 adds the generic preprocessing job itself. `thinking.image_embed` reads the same manifest
format and writes sidecar JSONL rows with `text_embedding` and/or `image_embedding`; local smoke
tests use a deterministic stats backend, while GPU jobs can use `--backend hf --model ...` for
CLIP/SigLIP text-image encoders or DINO/MAE-style image encoders with `--features image`. With
`--text-embed-mode tokens`, the sidecar writes nested `text_embedding_sequence` rows and the
generator conditions on those token streams directly. The generator stays model-agnostic: feature
extraction, manifest QA, and training remain separate auditable stages.

Image-42 wires that path into the GPU launcher. `runpod/launch_thinking.py --image-embed` now runs
`thinking.image_embed`, merges the sidecar through `thinking.image_data`, writes
`runs/image_train_clean.jsonl`, and points manifest training/eval at that cleaned file. Full venv
runs install `transformers`/`accelerate` only when the HF embedding backend is requested, while
`--fast` dry runs can use the deterministic stats backend for smoke validation.

Image-43 adds the missing dataset transport step for real-image GPU jobs. `--upload-image-data`
uploads repo-relative `--image-root` and `--image-manifest` paths to the pod before preprocessing,
so the pinned-code deploy can still train from local `data/images/...` manifests without committing
large image corpora into git. Absolute manifest/root paths remain reserved for datasets already
mounted on the pod.

Image-44 adds explicit manifest image augmentation. `thinking.image_latent` now accepts
`--image-crop-mode center|random|none|pad` and `--image-hflip-prob`, and RunPod passes the same
flags through. Eval remains deterministic, but real-image AE/flow training, latent-stat estimation,
and latent cache construction can now sample random square crops and horizontal flips. The settings
are recorded in train reports/checkpoints so caption-sensitive runs can keep flips at zero.

Image-45 adds rectified-flow timestep shifting for high-resolution latent image runs.
`thinking.image_latent --time-shift` shifts training samples and sampler nodes toward the noisy
part of the data-time path while keeping `1.0` as the original schedule. Checkpoint eval uses
the saved shift by default, `--sample-time-shift` can override it for sweeps, and RunPod exposes the
training knob as `--image-time-shift`.

Image-46 adds optional per-head QK RMSNorm to the custom MM-DiT attention path. Use
`thinking.image_latent --dit-qk-norm` or RunPod `--image-dit-qk-norm` for high-resolution MM-DiT
runs; checkpoint metadata records the flag so qk-normalized checkpoints reload with the same
attention modules, while existing checkpoints keep the default non-normalized architecture.

Image-47 adds an MM-DiT attention implementation selector. `--dit-attn-impl
manual|sdpa|linear|auto` keeps the original explicit attention path available, uses PyTorch
scaled-dot-product attention for fused/flash kernels when requested, enables an opt-in linear
attention approximation for longer latent token grids, or falls back automatically. RunPod exposes
the same setting as `--image-dit-attn-impl`, and checkpoint metadata records the selected
implementation.

Image-48 adds a REPA-style hidden-state representation alignment path for real-image latent flows.
`--flow-repa-w` trains the flow transformer's noisy-step image-token states against generic
manifest `image_embedding` rows, separately from endpoint feature alignment, and
`--flow-repa-steps` can restrict that auxiliary pressure to early training. `--flow-repa-mode
pooled|token|both|auto` additionally uses manifest `image_embedding_sequence` rows for token-level
hidden/visual representation matching when available. RunPod exposes the same settings as
`--image-flow-repa-w`, `--image-flow-repa-steps`, `--image-flow-repa-embed-dim`, and
`--image-flow-repa-mode`; disk latent caches now carry pooled image embeddings and token sequences
when endpoint feature alignment or REPA needs them.

Image-49 adds optional Min-SNR-style velocity loss weighting for rectified-flow training.
`--flow-loss-weight none|min-snr-v|soft-min-snr-v` keeps the exact original objective by default,
but can reweight per-example velocity MSE by the data-time SNR with batch-mean normalization.
Reports/checkpoints record the weighting mode, gamma, and observed weight range; RunPod exposes the
same controls as `--image-flow-loss-weight`, `--image-flow-loss-weight-gamma`, and
`--image-no-flow-loss-weight-normalize`.

Image-50 adds sampler-side CFG rescale. `--cfg-rescale` rescales guided latent velocities toward
the conditional velocity standard deviation before blending, while `--cfg-rescales` makes it a
checkpoint-sweep axis. RunPod exposes `--image-cfg-rescale` and
`--image-cfg-rescale-sweep`. Default `0.0` preserves the original CFG behavior.

Image-51 adds aspect-preserving manifest padding. `--image-crop-mode pad` resizes each image to fit
the requested canvas and pads the remaining area instead of square-cropping or distorting it. This
is the minimal data-pipeline step toward mixed-aspect high-resolution training: captioned objects
stay in frame while the current fixed-shape AE/flow code can still batch tensors locally and on
RunPod.

Image-52 adds geometry-aware latent DiT positions. `--dit-pos-embed learned|sincos2d|rope2d`
keeps learned 1D tables by default, but high-resolution DiT/CrossDiT/MM-DiT runs can use
deterministic 2D sinusoidal image-token positions, and MM-DiT can use 2D axial rotary image-token
Q/K positions with `rope2d`. RunPod exposes this as `--image-dit-pos-embed`. This removes a
fixed-token-table assumption from the transformer path and is a better default for resolution and
aspect-ratio scaling experiments.

Image-53 adds rectangular manifest resolution support. `thinking.image_latent --size` and RunPod
`--image-size` now accept `HxW` as well as square `SIZE`; checkpoints and reports preserve
`image_h`/`image_w`, and latent token limits use the real `latent_h * latent_w` grid. Synthetic
factor renders remain square-only and fail early if given a rectangular size, while real-image
manifest runs can pair `--image-crop-mode pad` with rectangular canvases.

Image-54 adds aspect-bucketed manifest training. `thinking.image_latent --size-buckets
128x128,128x192,192x128` keeps each batch on one efficient canvas while letting different batches
use different aspect ratios; RunPod exposes the same setting as `--image-size-buckets`. Reports
record the bucket list, per-bucket sample counts, missing manifest dimensions, and max train latent
tokens. Cached flow training now stores one fixed-shape subcache per bucket, so the recommended
RunPod recipe can combine aspect buckets with `--image-flow-cache-latents`.

Image-55 adds a generated text-image embedding score for real-image eval. When manifest rows carry
same-width external `text_embedding` and `image_embedding` vectors, checkpoint sweeps now report
`generated_external_text_image_score_cos` and retrieval accuracies by projecting generated image
latents through the learned image-feature bridge and comparing them to the caption embedding. This
keeps CLIP/SigLIP-style prompt alignment measurable without loading the external encoder during
eval.

Image-56 adds quality-weighted manifest sampling. `thinking.image_latent
--image-quality-weight W` samples rows from normalized `aesthetic` / `score` / `quality` metadata
instead of treating all kept rows equally; `W=0` is the exact uniform behavior. The same
weighting now reaches AE batches, latent-stat estimation, and memory/disk/bucketed flow latent
caches, with reports/checkpoints exposing `image_quality_*` and `flow_cache_weighted` fields. This
is a generic data-quality scaling lever: cleaned manifests can prefer better scored images without
hard-coding visual labels, caption grammar, or dataset-specific rules.

Image-57 adds dimension-aware rectified-flow time shifting for RAE-style latents.
`--time-shift-mode dim` scales the requested `--time-shift` by
`(latent_ch * latent_h * latent_w / --time-shift-ref-dim) ** --time-shift-dim-power`, using the
largest active aspect bucket. Reports/checkpoints now record requested shift, effective shift,
latent effective dimension, reference dimension, scale, and mode. This keeps the manual schedule
available while making high-dimensional semantic latents less dependent on hand-retuned constants.

Image-58 adds an optional pretrained latent-autoencoder bridge. `thinking.image_latent
--ae-arch hf-vae --ae-hf-model MODEL_ID` loads a frozen Diffusers `AutoencoderKL` and trains the
same text-conditioned flow on its latents instead of always learning a tiny local AE first.
`--ae-hf-subfolder` supports repos that store the VAE under `vae/`, and
`--ae-hf-scaling-factor` can override the model config when needed. Checkpoints store the model
reference and avoid serializing the frozen pretrained VAE weights, while local semantic/residual
AE runs stay dependency-free. RunPod installs `diffusers` only when `--image-ae-arch hf-vae` is
selected. This is the practical hook for SD-style VAE latents today and RAE-style pretrained
latent spaces when exported through the same AutoencoderKL interface.

Image-59 adds opt-in own-model endpoint distillation for the latent flow. After normal flow
training, `--flow-distill-steps N` freezes a teacher snapshot of the same model (`raw`, `ema`, or
`auto`) and trains the student to match the teacher's cleaner-time endpoint prediction from a
noisier point on the same rectified-flow path. The objective is latent/condition generic: it works
for fact, text, manifest, cached, local-AE, and pretrained-AE paths without caption grammar or
object-rule code. Reports/checkpoints record `flow_distill_steps_run`, `flow_distill_teacher_used`,
and `last_distill.distill_endpoint_mse`; RunPod exposes the same controls as
`--image-flow-distill-steps`, `--image-flow-distill-w`, `--image-flow-distill-time-gap`, and
`--image-flow-distill-teacher`.

Image-59 also supports guidance distillation. `--flow-guidance-distill-w` adds a frozen-teacher
CFG endpoint target inside the same post-flow distillation loop, with
`--flow-guidance-distill-cfg-scale` and `--flow-guidance-distill-cfg-rescale` controlling the
teacher guidance. This lets the conditional student absorb some sampling-time CFG behavior without
adding caption-specific rules. RunPod exposes the same controls as
`--image-flow-guidance-distill-w`, `--image-flow-guidance-distill-cfg-scale`, and
`--image-flow-guidance-distill-cfg-rescale`.

Image-60 lets sample grids use arbitrary prompt text even when a manifest run trained on external
caption embeddings. `thinking.image_latent --eval-checkpoint CKPT --sample-grid-out grid.ppm
--sample-prompts "a glass house at sunrise; a red canoe on a lake"` now routes prompts through
the checkpoint's token conditioner or, for `caption_cond_source=embedding`, through a live
`--prompt-embed-backend stats|hf` text embedder before the learned projection. The HF path uses
the same model family as the manifest sidecar (`--prompt-embed-model MODEL_ID`), validates the
embedding width against `text_embedding_in_dim`, and is exposed on RunPod as
`--image-sample-prompts` plus `--image-prompt-embed-*`. This closes the gap between embedding-based
training and free-form prompt inspection after GPU runs.

Image-61 broadens the rectified-flow solver sweep beyond Euler/Heun with `midpoint` and `rk4`
sample methods. The midpoint solver adds a second-order Runge-Kutta trajectory check that can beat
Euler on less-straight flows without changing training; RK4 is a higher-NFE diagnostic for finding
whether sample quality is integration-limited. RunPod now accepts `--image-sample-method midpoint`
or `rk4`, and its default sweep compares `euler,heun,midpoint` while leaving Euler as the single-run
default.

Image-62 makes timestep placement a first-class sampler axis. `thinking.image_latent` now accepts
`--sample-schedule linear|quadratic|sqrt|cosine` and checkpoint sweeps can pass
`--sample-schedules ...`; sample-grid metadata records the selected schedule. This lets GPU runs
measure whether quality improves by spending more solver evaluations near noisy states, clean
states, or both endpoints, instead of assuming the shifted linear schedule is always optimal.
RunPod forwards the same controls as `--image-sample-schedule` and `--image-sample-schedules`.

Image-63 adds generic negative-conditioned CFG for prompt sample grids. `thinking.image_latent
--sample-prompts ... --sample-negative-prompts ...` now uses the same token or live embedding
conditioner as the positive prompts to build the CFG baseline, instead of forcing every guided
sample to compare only against a zero vector. One negative prompt is broadcast across all prompts;
otherwise the negative prompt count must match the prompt count. RunPod exposes the same control as
`--image-sample-negative-prompts`, and sample-grid metadata records whether guidance used a zero or
negative-prompt baseline.

Image-64 adds best-of-N prompt sample selection. `--sample-candidates-per-prompt K` draws K samples
per arbitrary prompt and, when the checkpoint has the generic image/text aligner from real-image
training, keeps the candidate with the highest learned image-text cosine score. Checkpoints without
that scorer use the generic finite/collapse/luminance health score instead of silently keeping the
first candidate. RunPod forwards the same knob as `--image-sample-candidates-per-prompt`, so GPU
prompt grids can use alignment- or health-guided inspection without changing training or adding
prompt-specific rules.

Image-65 adds sampling-time text-alignment guidance for prompt grids. `--sample-text-guidance-w`
uses the checkpoint's learned image/text aligner as a differentiable latent reward during sampling:
after each active solver step, the latent is nudged along the normalized gradient that increases the
prompt/image alignment score. This is generic scorer guidance, not a renderer rule, and defaults to
off. RunPod forwards `--image-sample-text-guidance-w` and
`--image-sample-text-guidance-interval` for GPU prompt-grid runs.

Image-66 adds a manifest quality scorer. `--image-quality-score-w` trains a latent scorer from
generic `aesthetic` / `score` / `quality` manifest metadata, `--flow-quality-score-w` can preserve
that score on predicted flow endpoints, and prompt grids can use
`--sample-quality-guidance-w` to nudge samples toward higher learned quality. Checkpoints save the
scorer separately from the flow, eval reports include generated quality-score means, and RunPod
exposes the same training/guidance knobs. This turns quality metadata into a reusable model signal,
not only a row-sampling weight. `--quality-score-steps` / RunPod
`--image-quality-score-steps` can also train this scorer as a standalone latent phase.

Image-67 makes latent normalization safe by default for scale runs. `--latent-normalize auto`
now resolves to `channel` when training on image manifests or an external HF VAE, and to `none`
for synthetic factor-only runs. Reports/checkpoints store both `latent_normalize_requested` and
the effective `latent_normalize`, so GPU runs can tell whether a model sampled from standardized
latent coordinates. RunPod now defaults `--image-latent-normalize auto`.

Image-68 adds token-level external text conditioning for real-image manifests. Manifest rows and
embedding sidecars may include nested `text_embedding_sequence` / `text_token_embeddings` arrays;
`PrecomputedTextConditioner` projects `B x T x D` inputs into masked condition tokens for
Cross-DiT/MM-DiT while still producing a pooled vector for global conditioning and retrieval
metrics. `thinking.image_embed --text-embed-mode tokens` emits those sequences, and RunPod exposes
it as `--image-embed-text-mode tokens`. This closes a major prompt-understanding gap versus modern
text-to-image systems without tying the model to a specific tokenizer or provider.

Image-69 hardens token-sequence conditioning for cached flow training. In-memory latent caches now
pad variable-length `text_embedding_sequence` chunks before concatenation, and memory/disk cache
reports include `flow_cache_text_embedding_*` shape fields. This keeps large manifest runs from
failing only after a cache chunk happens to contain longer captions than earlier chunks, and makes
GPU reports show whether training used pooled vectors or token streams.

Image-70 improves prompt-grid selection for real-image runs. `--sample-candidates-per-prompt`
still works without task-specific rules, but embedding-conditioned prompt grids now reuse compatible
external image-feature alignment as a candidate scorer alongside the learned text aligner. This
means CLIP/SigLIP-style sidecars can help choose the best generated candidate for a prompt even
when the strongest training signal came from image-feature or REPA alignment rather than the
internal caption aligner alone.

Image-71 adds sampling-time external feature guidance. Prompt grids can now pass
`--sample-feature-guidance-w` / `--sample-feature-guidance-interval`; when the live prompt
embedding shares the sidecar image-feature dimension, the sampler nudges generated latents toward
that external text/image feature target using the checkpoint's `image_feature_aligner`. This turns
CLIP/SigLIP-style sidecars into both a training/eval signal and an inference-time guidance signal,
without adding caption- or domain-specific rules.

Image-72 makes manifest checkpoint eval tune the same inference guidance path instead of relying on
one-off prompt grids. `thinking.image_latent --eval-checkpoint ... --eval-image-manifest ...` now
accepts `--eval-text-guidance-weights`, `--eval-feature-guidance-weights`, and
`--eval-quality-guidance-weights`; aggregate rows and best-selection keys include those axes.
RunPod exposes the same knobs as `--image-eval-text-guidance-sweep`,
`--image-eval-feature-guidance-sweep`, and `--image-eval-quality-guidance-sweep`, with launcher-side
validation. This gives real-image runs a generic way to measure whether text, external-feature, or
quality guidance improves sample metrics before scaling the setting on GPU.

Image-73 adds generic manifest source-mixture weighting. Manifest rows now preserve a `source`
field (or `dataset` / `domain` / `collection` / `bucket` aliases), summaries report source counts,
and `thinking.image_latent --image-source-weights "curated=2.0,text_render=1.5,*=0.5"` multiplies
those weights with existing quality weights for AE training, latent-stat estimation, flow training,
and latent-cache sampling. RunPod exposes the same setting as `--image-source-weights`. This is the
data-composition lever we need for SOTA-scale image training: we can rebalance curated photos,
synthetic/diagram/text-rendering subsets, and noisy web data without adding domain-specific model
rules.

Image-74 makes aspect-bucket training respect that data mixture. Multi-resolution manifest runs now
choose size buckets by total bucket record weight when quality or source weights are active, while
unweighted runs keep the previous uniform bucket sampling. Bucketed latent caches use the same
`weight_sum` rule at sample time, and reports expose `size_bucket_sampling_mode`,
`size_bucket_weight_sums`, `size_bucket_sampling_probs`, and cache bucket sampling probabilities.
This prevents rectangular training from silently undoing curated/source/quality weighting just
because high-weight records cluster in one aspect ratio.

Image-75 adds generated-vs-real feature distribution metrics for manifest eval. When an
`image_feature_aligner` is available, generated samples and their paired real sidecar image
features are compared in the learned external-feature space with matched cosine/L2, mean-gap,
covariance Frobenius distance, Frechet-style distance, and RBF MMD. These metrics aggregate in
checkpoint sweeps and participate in best-run selection as lower-is-better distribution distances.
This gives guidance/source/quality sweeps a generic quality signal closer to FID/KID-style model
selection without depending on a hard-coded dataset or a heavyweight eval package.

Image-76 extends that same feature-space eval with diversity and coverage checks. Manifest sweeps
now report generated-vs-real pairwise spread, diversity ratio, nearest-real/nearest-generated
distances, and support precision/recall at data-derived feature radii. These metrics catch early
mode collapse or over-narrow guidance: a checkpoint can match prompts pairwise but still fail to
cover the real sidecar feature support. Best-run selection now prefers high support precision/recall
and a generated/real diversity ratio near one, while keeping the metrics generic to any external
image embedding sidecar.

Image-77 decouples generated manifest eval count from sampler step count. `thinking.image_latent
--eval-checkpoint ... --eval-generated-samples N` now controls how many generated images feed the
caption retrieval, feature-distribution, diversity, coverage, and quality metrics for each sweep
row; `0` uses `min(batch,sample_steps)` for smoke behavior. Generation is chunked by the
normal eval batch size, and RunPod exposes the same control as `--image-eval-generated-samples`.
This makes one-step solver smokes useful without weakening full-sample quality metrics, and lets
GPU sweeps spend eval budget on statistically meaningful generated sets.

Image-78 adds opt-in timestep curriculum training for latent rectified flows. Recent flow-matching
training recipes use non-uniform timestep sampling early, then switch toward uniform coverage for
later refinement; `thinking.image_latent --time-sampling logit-normal
--time-curriculum-frac 0.5` now uses logit-normal samples for the first half of flow training and
uniform samples afterward. Reports/checkpoints record the requested fraction, switch step, final
sampling mode, and last active training mode; RunPod exposes the same lever as
`--image-time-curriculum-frac`. Default `0` preserves the static timestep sampler.

Image-79 makes MM-DiT branch gates strict architecture state. The adaptive branches use the
AdaLN-zero-style multiplier directly, reports/checkpoints record `zero_residual_gating`, and the
flow loader requires gate weights to match the active architecture.

Image-80 adds opt-in activation checkpointing for latent DiT/CrossDiT/MM-DiT flow blocks.
`thinking.image_latent --flow-checkpoint-blocks` recomputes transformer blocks during backward to
reduce activation memory for deeper or higher-resolution image runs. The state dict is unchanged;
reports and checkpoints record `flow_checkpoint_blocks` and `activation_checkpointing`, and RunPod
exposes the same training knob as `--image-flow-checkpoint-blocks`.

Image-81 adds opt-in stochastic sampler churn for latent rectified-flow generation.
`thinking.image_latent --sample-churn 0.05 --sample-churn-interval 0.0,0.8` injects
deterministic per-step latent noise during the selected flow-time interval, while the default
`--sample-churn 0` keeps deterministic ODE sampling. Checkpoint sweeps can pass
`--sample-churns ...`; reports, aggregate keys, sample grids, checkpoints, and RunPod expose the
same fields so deterministic and stochastic candidate settings are directly comparable.

Image-82 adds opt-in SwiGLU feed-forward blocks for the scalable CrossDiT/MM-DiT latent flow
path. `thinking.image_latent --dit-mlp swiglu` replaces the GELU MLP inside CrossDiT/MM-DiT
blocks with a parameter-comparable gated feed-forward layer; plain `dit` keeps the existing Torch
encoder layer. Reports/checkpoints record `dit_mlp` and `uses_swiglu_mlp`, and RunPod exposes the
same setting as `--image-dit-mlp swiglu`.

Image-83 adds latent patch tokens for transformer flows. `--latent-patch-size 2` groups each
2x2 latent neighborhood into one DiT/CrossDiT/MM-DiT token, reducing attention length while
unpatchifying the predicted velocity back to the full latent grid before decoding. Reports now
separate `latent_cells` from active `latent_tokens`, record `latent_patch_size`, and use the
patched token count for `latent_max_tokens` validation. RunPod exposes this as
`--image-latent-patch-size`.

Image-84 batches classifier-free guidance inside the latent sampler. When `cfg_scale != 1`,
the conditional and unconditional velocity queries are concatenated into one model forward and
split afterward, preserving the same guidance math while reducing sampler launches for high-step
MM-DiT runs.

Image-85 adds generated sample health metrics to catch visual collapse. Eval rows and sample-grid
metadata now report finite/non-finite fractions, luminance mean/std, RGB std, dynamic range,
low/high luminance fractions, collapsed-sample fraction, and a generic `sample_health_score`.
Checkpoint and sampler selection keys include these fields after semantic/feature metrics, so
flat black or numerically unstable outputs are penalized without adding renderer- or
prompt-specific rules. Prompt-grid best-of-N selection now uses the same generic health signal
whenever no learned text/feature scorer is available.

Image-86 adds sampler trajectory tracing and generic stability controls. Eval rows and sample-grid
metadata now report latent/velocity finite fractions, non-finite step counts, RMS/max-absolute
trajectory bounds, and stabilization events. `--sample-finite-guard` zeroes non-finite
latent/velocity entries, `--sample-velocity-clip` clips per-sample velocity RMS, and
`--sample-latent-clip` clamps latent magnitude; RunPod exposes the same controls with
`--image-sample-*`. These are generic numerical controls, not renderer or prompt rules.

Image-86 also fixes the root cause of the first black MM-DiT prompt grids: token negative prompts
with only out-of-vocabulary words were becoming all-padding rows, which can make transformer
attention return NaNs. New synthetic prompt vocabs include `<unk>`, and tokenization falls back to
a non-pad unknown id when loading a checkpoint vocab that has no explicit `<unk>`. A local
negative-prompt MM-DiT eval now stays finite without the finite guard
(`sample_grid_nonfinite_frac=0.0`, `sample_grid_trace_velocity_nonfinite_steps=0`), though the
actual image quality is still noisy and needs further training/architecture work.

Image-87 adds an opt-in clean-endpoint auxiliary objective for latent rectified flow.
`--flow-endpoint-w` directly penalizes the predicted clean latent endpoint
`zt + (1-t) * velocity` against the training latent, while preserving the main velocity loss and
existing timestep weighting. Reports/checkpoints record `flow_endpoint_w`, and `last_flow` always
reports weighted/unweighted `flow_endpoint_target_mse` so GPU runs can tell whether the learned
vector field is improving the decode endpoint or only reducing velocity MSE. RunPod exposes the
same objective as `--image-flow-endpoint-w`.

Image-88 adds a dependency-free frequency-spectrum AE reconstruction term. `--ae-fft-w` compares
log FFT magnitudes with a mild high-frequency weight, so residual/HF-free autoencoder runs can
penalize blurred spectra in addition to pixel, gradient, and multi-scale losses. RunPod exposes the
same control as `--image-ae-fft-w`; reports/checkpoints record `ae_fft_w`, and `last_ae` includes
`recon_fft_l1` whenever the term is active.

Image-89 adds optional source-noise coupling for latent flow matching. `--flow-noise-coupling
sliced_ot` sorts each clean-latent batch and Gaussian source batch along one or more random latent
projections, then pairs matching ranks from the projection with the lowest pair MSE before forming
the rectified-flow target. `--flow-noise-coupling-projections K` trades extra cheap dot products for
better pairings; if none beats the original random pairing, training keeps the original source
noise. This sliced-OT approximation preserves the Gaussian source marginal while reducing
arbitrary source/data pairing noise; `last_flow` reports before/after pair MSE, projection count,
selected projection, and whether coupling was active/accepted. RunPod exposes the same mode as
`--image-flow-noise-coupling sliced_ot --image-flow-noise-coupling-projections K`.

Image-90 adds reduced-precision latent-cache storage for real-image flow training.
`--flow-cache-dtype bf16|fp16` stores memory and disk cached AE latents in the requested dtype while
sampling them back as float32 for the flow. This cuts cache RAM/disk/IO pressure for external-VAE
or high-resolution manifests without changing the training objective. RunPod exposes the same knob
as `--image-flow-cache-dtype`.

Image-91 makes disk latent caches actually reusable. When `--flow-cache-dir` already contains a
`meta.json`, training now fingerprints the selected manifest rows, image file stats, crop/flip
settings, cache dtype, conditioning source, and AE identity/state. Matching caches are loaded
directly and report `flow_cache_reused=true`; mismatches rebuild the cache instead of silently
using stale latents.

Image-92 adds bounded in-process shard caching for disk latent caches. `--flow-cache-max-loaded-shards
N` keeps up to N recently sampled `.pt` latent shards loaded in CPU RAM, using LRU eviction, instead
of `torch.load`-ing every shard on every sampled batch. Reports/checkpoints expose
`flow_cache_loaded_shards`, `flow_cache_shard_loads`, and hit/miss counters, and RunPod exposes the
same knob as `--image-flow-cache-max-loaded-shards`. Set it to fit available CPU RAM; `0` preserves
the previous streaming behavior.

Image-93 lets quality-aware flow training scale with latent caches. When
`--flow-quality-score-w` is active, memory and disk flow caches now store normalized quality targets
and masks from manifest `aesthetic` / `score` / `quality` metadata, and sampled cache payloads feed
those targets into the frozen quality scorer loss. Reports expose
`flow_cache_has_quality_targets`, so cached real-image runs can combine reduced-precision latent
shards, bounded shard loading, and quality-preserving flow training instead of falling back to
per-step image decode.

Image-94 decouples quality-scorer learning from AE training. `--quality-score-steps N` runs a
standalone scorer pretrain phase after AE freezing and latent-cache construction, using cached
latents plus cached quality targets when available, or live no-grad AE encodes otherwise. This lets
external/frozen VAE runs use `--flow-quality-score-w` without also spending AE loss weight on
`--image-quality-score-w`. Reports/checkpoints expose `quality_score_steps`,
`quality_score_steps_run`, and `last_quality_score`; RunPod forwards the same control as
`--image-quality-score-steps`.

Image-95 adds generic quality preference ranking. `--image-quality-rank-w`,
`--quality-score-rank-w`, and `--flow-quality-rank-w` build same-batch preference pairs from
normalized manifest `aesthetic` / `score` / `quality` values, train the scorer to rank better rows
above worse rows, and preserve that ordering on predicted flow endpoints. The loss is pairwise and
schema-free: it consumes only score ordering, not caption grammar, hand-written visual rules, or
dataset-specific labels. RunPod exposes the same controls as `--image-quality-rank-w`,
`--image-quality-score-rank-w`, `--image-flow-quality-rank-w`, and
`--image-quality-rank-margin`.

Image-96 makes prompt-grid best-of-N selection quality-aware. When
`--sample-candidates-per-prompt K` is greater than one, selection now includes the checkpoint's
learned quality scorer in addition to compatible text and external-feature aligners. Each scorer is
normalized within a prompt's K candidates before averaging, so cosine aligners and scalar quality
scores can contribute without hand-tuned scale constants. Reports expose the component count,
selected candidate indices, and selected component scores.

Image-97 applies the same best-of-K reranking to held-out manifest generation eval. Use
`--eval-generated-candidates-per-prompt K` with `--eval-generated-samples N` to draw K candidates
per caption, rerank with the checkpoint's generic learned text aligner, external feature aligner,
and quality scorer when available, and keep one selected image per caption for metrics. Reports now
separate selected sample count from `generated_eval_raw_candidates`, preserve selected candidate
indices, include the composite selection score, and aggregate candidate K without mixing K=1 and
K>1 rows. RunPod exposes the same control as
`--image-eval-generated-candidates-per-prompt`.

Image-98 replaces zero-only text CFG dropout with a learned null condition for token and
precomputed-embedding conditioners. Text training still uses the same `--cond-drop` knob, but
dropped rows now train a real unconditional payload (`null_vec` plus null condition tokens), and
sampling uses that learned null branch whenever no negative prompt is supplied. Auxiliary
semantic/text alignment losses use an explicit dropout mask, so the learned null vector can become
nonzero without being mistaken for an active caption. Reports expose `conditioner_learned_null`
and `cfg_uncond_default`, while old checkpoints without null parameters still load with the new
null state initialized from defaults.

Image-99 adds a CFG++-style sampler mode for latent rectified-flow generation. `--cfg-mode
standard|cfgpp` keeps the existing guided velocity path by default, while `cfgpp` mixes the guided
clean endpoint direction with the unconditional source direction as
`(1 - t) * v_guided + t * v_uncond`. This is generic to any condition payload, including
learned-null text CFG, negative
prompts, and external caption embeddings. Checkpoint sweeps can compare modes with `--cfg-modes
standard,cfgpp`; rows, aggregates, sample-grid metadata, checkpoint metadata, and RunPod
(`--image-cfg-mode`, `--image-cfg-modes`) all record the selected mode.

Image-100 adds no-extra-data self-REPA for latent rectified-flow training. `--flow-self-repa-w`
aligns the flow transformer's noisy-step image-token states to the same sample's clean AE latent
patches using paired cosine regression, so the signal is linear in token count and does not require
manifest image embeddings, grammar rules, or extra labels. `--flow-self-repa-mode pooled|token|both|auto`
uses pooled latent targets, patch-token targets, or both; `auto` selects both because latent patch
tokens are always present. `--flow-self-repa-steps` can keep the pressure to early training, and
`--flow-self-repa-embed-dim` controls the auxiliary projection width. Checkpoints save and reload the
auxiliary aligner explicitly. RunPod exposes the same generic path as
`--image-flow-self-repa-w`, `--image-flow-self-repa-steps`, `--image-flow-self-repa-embed-dim`, and
`--image-flow-self-repa-mode`.

Image-101 adds no-extra-component self-representation alignment for transformer latent flows.
`--flow-sra-w` runs a detached teacher pass at a cleaner timestep on the same rectified-flow path
and aligns the noisy-step hidden image tokens toward that cleaner hidden representation. This is
generic over synthetic factors, caption manifests, cached latents, and HF-VAE latents because it
uses only the model's own hidden states and the sampled flow path, not image embeddings or
renderer-specific labels. `--flow-sra-mode pooled|token|both|auto` controls pooled vs per-patch
alignment, `--flow-sra-time-gap` controls how far the teacher step moves toward clean data, and
`--flow-sra-steps` can restrict the signal to early training. RunPod exposes the same path as
`--image-flow-sra-w`, `--image-flow-sra-steps`, `--image-flow-sra-time-gap`, and
`--image-flow-sra-mode`.

Image-102 adds a reproducible web-data bootstrap for image generation. `thinking.image_fetch`
streams captioned WebDataset tar shards into local image files plus the existing JSONL manifest
format, stopping at `--max-records` so smoke runs do not accidentally pull a full corpus. The
scale source is `text-to-image-2m-512-2m`, discovered from the Hugging Face dataset tree at fetch
time; `text-to-image-2m-1024-10k` remains available for high-resolution smoke fetches. RunPod can
run the fetch as part of one command with `--image-fetch`, `--image-fetch-source`,
`--image-fetch-max-records`, `--image-fetch-dir`, and `--image-fetch-manifest`; downstream
embedding, manifest QA, latent caching, and flow training then consume that fetched manifest
instead of a hand-prepared local dataset.

Image-103 adds a RunPod quality preset for the real-image path. `runpod/launch_thinking.py
--image-quality-preset web-hf-vae` expands to a full web-data training profile: fetch from the
512px text-to-image 2M shard set, compute HF text/image embeddings with SigLIP plus token-level T5
text sequences, clean the manifest, train the flow in a frozen SDXL VAE latent space, use MM-DiT
with rope2d/SwiGLU/QK norm/linear attention, cache bf16 latents, enable self-REPA and SRA, and
write prompt grids. The preset intentionally leaves quality-score/ranking losses off because the
built-in web shards have captions but no aesthetic score metadata.

Image-104 makes web-data QA dependency-free and score-aware. `thinking.image_data` now reads
JPEG/PNG/WebP/BMP dimensions from headers before falling back to Pillow, so local and pod-side
manifest cleaning can enforce `--min-side` / `--max-aspect` without optional image libraries.
`thinking.image_fetch` writes width/height from source metadata or image headers, and both fetch
and QA preserve generic `aesthetic`, `nsfw`, and `watermark` scores when datasets provide them.
`--max-nsfw` and `--max-watermark` filter numeric metadata thresholds without caption grammar or
dataset-specific visual labels; RunPod forwards the same controls as `--image-clean-max-nsfw` and
`--image-clean-max-watermark`, and the `web-hf-vae` preset applies conservative defaults.

Image-105 adds generic image-text alignment filtering for real-image data. After
`thinking.image_embed` writes pooled text and image embeddings, `thinking.image_data
--min-image-text-cosine T` rejects rows whose paired embedding cosine is too low and reports score
distribution, missing/mismatch counts, and sampled reject examples. The filter also works with
manifests that already carry pooled embeddings or sequence embeddings that can be mean-pooled. This
implements a CLIP/DataComp-style data quality gate without prompt grammar, hand-written visual
classes, or source-specific captions; RunPod exposes it as
`--image-clean-min-image-text-cosine`, and the `web-hf-vae` preset starts with a conservative
`0.15` threshold.

Image-106 upgrades external text conditioning for real-image MM-DiT runs. `thinking.image_embed
--text-embed-mode both` now writes both pooled `text_embedding` rows for alignment filtering and
token-level `text_embedding_sequence` rows for the generator. `caption_cond_source=auto` then
chooses embedding conditioning and feeds token sequences to CrossDiT/MM-DiT while preserving pooled
text features for image-text cosine filtering and prompt-grid guidance. Prompt grids now request
token-aware live prompt embeddings when the checkpoint flow consumes condition tokens, so sampling
matches the training conditioning path. RunPod's `web-hf-vae` preset uses this combined mode by
default.

Image-107 adds a second real web-data ingestion path for image training. `thinking.image_fetch
--source diffusiondb-2m` downloads DiffusionDB zip parts into the same generic image/caption JSONL
manifest as the WebDataset path, using each part's JSON metadata for prompts and optionally merging
`metadata.parquet` via `--diffusiondb-metadata` when `pyarrow` is available. The emitted rows still
use only generic fields (`caption`, `width`, `height`, `nsfw`, `watermark`, `aesthetic`), so the
downstream cleaner and model keep the no-hardcoded-rules contract. RunPod exposes the path through
`--image-fetch-source diffusiondb-2m`, `--image-fetch-diffusiondb-start-part`,
`--image-fetch-diffusiondb-end-part`, and `--image-fetch-diffusiondb-metadata`.

Image-108 adds hybrid text-encoder preprocessing for real-image generation. `thinking.image_embed
--text-sequence-model MODEL` keeps the primary HF vision-language encoder responsible for
`image_embedding` and pooled `text_embedding` rows, while a separate HF text encoder writes the
token-level `text_embedding_sequence` used by CrossDiT/MM-DiT conditioning. This mirrors the
multi-encoder shape used by modern text-to-image systems without tying the code to a prompt grammar
or a single provider; pooled VLM dimensions and token text-encoder dimensions may differ, and
training uses the token-sequence width for conditioning while preserving pooled rows for
alignment/filtering. Prompt grids expose matching
`--prompt-embed-text-sequence-model` / RunPod
`--image-prompt-embed-text-sequence-model` controls so live prompts use the same token width as
training; the `web-hf-vae` preset defaults this path to `google-t5/t5-base` and keeps SigLIP for
pooled alignment/filtering.

Image-109 adds generic semantic near-duplicate filtering for image manifests. After embedding
merge, `thinking.image_data --max-image-duplicate-cosine T` uses deterministic random-hyperplane
LSH over normalized image embeddings, then performs exact cosine checks only within candidate
buckets. This keeps the curation pass roughly linear for large manifests while removing repeated or
near-repeated visual rows; when generic `aesthetic`/`quality` metadata is present, the filter keeps
the higher-quality duplicate rather than simply the first row. Reports expose LSH settings,
comparison counts, bucket/candidate stats, missing-embedding diagnostics, and duplicate examples.
RunPod forwards the same controls as `--image-clean-max-image-duplicate-cosine`,
`--image-clean-dedupe-lsh-*`, and `--image-clean-dedupe-keep-first`; `web-hf-vae` enables a
conservative `0.985` duplicate cosine threshold by default.

Image-110 adds optional recaptioning as a generic data-plane stage. `thinking.image_caption` reads
the same image manifest format, generates `generated_caption` rows with either a dependency-free
stats backend or a Hugging Face image-captioning model, preserves `original_caption`, and writes a
new manifest via `--mode replace|append|fill-empty|sidecar`. This targets the DALL-E 3/PixArt-style
lesson that prompt following improves when image/text pairs carry dense, image-grounded captions,
without adding caption grammar or visual-label rules to the generator. RunPod exposes it as
`--image-caption` between fetch/upload and embedding, so downstream SigLIP/T5 embeddings,
alignment filtering, semantic deduplication, latent caching, and flow training all consume the
caption-improved manifest.

Image-111 adds checkpoint-independent image-generation evaluation. `thinking.image_eval` compares
any real/generated manifest pair that carries generic `image_embedding` and optional
`text_embedding` fields, then reports Fréchet-style embedding distance, RBF-MMD, diversity ratio,
nearest-neighbor support precision/recall, and CLIPScore-style image/text retrieval. This gives
the web-data and generated-sample loop an offline quality gate before expensive training or GPU
sweeps, and it is encoder-agnostic: SigLIP, CLIP, DINO, or future multimodal embedding sidecars
use the same manifest contract.

Image-112 closes that eval loop for actual model samples. `thinking.image_latent
--sample-manifest-out generated.jsonl` now writes each generated grid sample as an individual PPM
plus a captioned manifest row with the conditioning prompt/caption, sampler settings, and
best-of-N selection metadata. The manifest can be fed directly to `thinking.image_embed` and then
`thinking.image_eval`, so GPU prompt grids and checkpoint sweeps produce metric-ready artifacts
instead of only a visual contact sheet. RunPod exposes the same path as
`--image-sample-manifest-out` / `--image-sample-image-dir`. `--image-eval-generated` then embeds
that generated manifest and runs `thinking.image_eval` against the cleaned/effective reference
manifest, writing `*_embeddings.jsonl`, `*_embed_report.json`, and `*_eval_report.json` by
default. `thinking.image_eval` now reports a composite `image_eval_score` over distribution
distance, support precision/recall, diversity parity, and image-text alignment; optional
`--min-score`, `--min-support-*`, `--min-image-text-cos`, `--max-frechet`, and `--max-mmd-rbf`
thresholds produce `image_eval_gate_*` fields, and `--fail-on-gate` turns them into a hard CI/GPU
gate. The `web-hf-vae` preset enables this loop automatically, so each GPU quality run returns
both image artifacts and comparable generated-vs-reference metrics.

Image-113 adds a generic quality/preference scoring data stage. `thinking.image_score` reads a
captioned manifest, writes a scored manifest with `quality_score`/`aesthetic` metadata, and can
also emit a score-only sidecar. The dependency-free `stats` backend is for local smoke tests and
technical image-health filtering; the `external` and `ensemble` paths merge arbitrary JSONL/CSV/TSV
scores from preference/reward models without hardcoding ImageReward, PickScore, HPS, or future
reward APIs into model training. RunPod exposes this as `--image-score`, and `web-hf-vae` now
enables it before embedding/cleaning with modest quality sampling, quality-head, ranking, and
quality-guidance weights, so the existing generic training hooks can use preference metadata when
it is available.

Image-114 adds the preference-pair artifact needed for DPO-style image tuning. `thinking.image_preferences`
groups a scored manifest by generic prompt fields (`preference_group`, `prompt_id`, `prompt`, or
`caption` by default), selects chosen/rejected image pairs by `quality_score` or any requested
reward field, and writes JSONL rows with prompt, chosen image, rejected image, scores, and score
gap. This is the bridge from generated best-of-N candidates or human/reward-model annotations to
future preference optimization without coupling the repo to a specific reward model API.

## 3c. Multimodal bridge: image + audio into the same trace language

`thinking.audio` transposes the Image-2 FER setup to sound: pitch × timbre × envelope are rendered
as waveforms and log-spectrograms, with held-out factor combinations and the same factored/swap/
subspace intervention ladder. The target facts use the same h/pred/t convention:
`fact a0 pitch n440 .`, `fact a0 timbre saw .`, `fact a0 env decay .`.

`thinking.multimodal` is M-0: one non-pointer `ScratchpadLM` receives continuous image patch
prefixes, audio spectrogram prefixes, and transcript prefixes from
`data/multimodal_transcripts.json`, then emits one canonical extraction trace:

```
[IMG prefix][AUD prefix][TXT prefix] extract
  fact p0 color red . fact p0 shape circle .
  fact a0 pitch n440 . fact a0 timbre saw . fact a0 env decay . done .
```

This is the multimodal version of the project invariant: every modality grounds into canonical
facts before the checker/reasoner sees it. Image generation stays connected through the same
semantic latent path (`image -> latent -> facts/generation`) rather than becoming a detached
pixel model.

M-1 architecture update: the multimodal bridge now inherits the upstream reader improvements
instead of freezing M-0 as a toy fixed prefix. `thinking.multimodal` exposes decoder width/depth,
image/audio/text prefix token counts, residual sensory trunks, trunk width/depth, deeper transcript
encoders, modality dropout, and a cross-mode factor-value agreement loss. Image prefixes use a
near-square adaptive pool as token count grows; audio keeps frequency-preserving bands. Reports and
checkpoints record the full architecture, and RunPod forwards the same levers through
`--multimodal-trunk-arch`, `--multimodal-img-tokens`, `--multimodal-txt-tokens`,
`--multimodal-dropout`, and `--multimodal-agreement-w`.

M-2 upstream concept update: multimodal now uses shared concept fusion as the single upstream
path. Learned concept tokens mix with image/audio/text prefixes before the trace decoder, and an
auxiliary concept head exposes the same factor schema. `--concept-w` trains those upstream concept
tokens directly, `--concept-agreement-w` aligns their factor distributions across full,
sensor-only, and text-only paths, and `--concept-distill-w` distills the full multimodal concept
distribution into partial modes. This lets the multimodal bridge use the same idea as the text
self-teaching work: preserve and transfer model-derived concept distributions upstream, before the
canonical trace decoder has to emit tokens. Reports now include `concept_head`, `concept_gate`,
reader-prefix token count, and total prefix token count, and RunPod exposes the active controls as
`--multimodal-concept-tokens`, `--multimodal-fusion-layers`, `--multimodal-concept-w`,
`--multimodal-concept-agreement-w`, and `--multimodal-concept-distill-w`.

M-3 carries the text rank-retention idea into the multimodal concept fusion path.
`--concept-rank-distill-w` preserves full-path concept winners and target-over-alternative margins
inside sensor-only and text-only modes, but only on rows where the full multimodal path already
predicts the labeled factor value correctly. This is not a renderer rule or a hand-authored English
fact: the teacher signal is the model's detached full-path concept distribution, filtered by the
batch labels that already supervise the trace. RunPod exposes the same lever as
`--multimodal-concept-rank-distill-w` with `--multimodal-concept-rank-distill-margin`.

M-4 moves the latest text-study transfer idea into the upstream multimodal architecture.
`--concept-transfer-w` aligns sensor-only and text-only concept state vectors toward detached
full-path concept states on rows where the full path is already correct, preferring current
partial-mode errors when they exist. The stable side is detached, so the hard partial reader moves
toward the discovered concept without dragging the full reader backward. This is vector-level
concept transfer before decoding, not token prediction and not a hard-coded fact table. RunPod
exposes the same path as `--multimodal-concept-transfer-w` and
`--multimodal-concept-transfer-margin`.

M-5 carries the text concept-geometry objective into multimodal fusion. `--concept-contrast-w`
uses the shared schema head's projected concept states and clusters examples with the same
data-supplied factor value while separating other values for that factor. Reports now include
`concept_geometry` for full, text-only, and sensor-only modes with nearest-same accuracy and
same-vs-different cosine margins. RunPod exposes the same controls as
`--multimodal-concept-contrast-w` and `--multimodal-concept-contrast-temperature`.

M-6 carries the newer text prototype objective into multimodal fusion. `--concept-prototype-w`
classifies projected concept states against learned per-key value prototypes, while
`--concept-prototype-spread-w` keeps those prototypes from collapsing inside each schema key. This
keeps the reusable concept geometry anchored even when a batch has few same-value pairs for the
contrastive loss. RunPod exposes the same controls as `--multimodal-concept-prototype-w`,
`--multimodal-concept-prototype-spread-w`, and
`--multimodal-concept-prototype-spread-margin`.

M-7 carries the latest text concept-prefix and batch-geometry improvements into the multimodal
decoder path. `--concept-prefix` prepends schema-key concept states computed by the generic
`SchemaConceptHead` to the decoder prefix, so the trace decoder can consume reusable factors
directly instead of relying only on auxiliary losses over hidden tokens. `--concept-centroid-w`
aligns states to same-value centroids discovered inside the current batch, and
`--concept-state-spread-w` keeps those projected states from collapsing. RunPod exposes this as
`--multimodal-concept-prefix`, `--multimodal-concept-centroid-w`, and
`--multimodal-concept-state-spread-w` with matching temperature, margin, variance, and covariance
controls.

M-8 carries the schema-free latent concept slots from `thinking.text` into the multimodal bridge.
`--latent-concept-slots` adds trainable concept queries that read the fused image/audio/text prefix,
and `--latent-concept-w` aligns those slots across full, sensor-only, and text-only views with
VICReg-style invariance, variance, and covariance terms. This is deliberately grammar-free: the
loss only sees paired views of the same example, not factor names or value rules. The training loop
now extracts latent slots from the existing `mode_bundle` prefixes, so the loss does not re-run the
image/audio/text encoders. RunPod exposes the same controls as
`--multimodal-latent-concept-slots`, `--multimodal-latent-concept-w`, and the matching
`--multimodal-latent-concept-*-w` term weights.

M-4 real local run (`runs/m0_multimodal_transfer_real.json`, 240 steps on MPS) gates both the
decoder and upstream concept head with `--concept-transfer-w 0.1`: `gate=true`,
`concept_gate=true`, token loss **0.059**, concept loss **0.193**, transfer loss **0.527**. Full
teacher-forced accuracy is color **1.00**, shape **0.97**, pitch **0.93**, timbre **0.99**, env
**1.00**; sensor-only is **1.00 / 1.00 / 0.83 / 0.99 / 1.00**; text-only held-out phrasings are
**0.77 / 0.80 / 0.64 / 0.95 / 1.00**. Free text-only exact is **0.35**, and free counterfactual
mean target/collateral are **0.83 / 0.80**. So the upstream transfer path works and improves the
concept/sensor bridge enough to gate, but it is not a language-mastery claim: free exact and some
text-only factors still need better retention/curriculum work.

The adaptive text study loop above is intentionally upstream of this multimodal bridge. When the
text reader fails shortcut controls, the next study attempt can focus the generated evidence
routes that failed, while the selector blocks regressions. Multimodal can then consume a reader
whose concept distributions are being improved by measured failures and reusable prototype
controls instead of by hard-coded English facts or hand-written rules.

M-0 local update (240 steps, value-token weighted loss, all-mode ablations every step): the
tracked run now gates three paths separately. Teacher-forced full multimodal accuracy is color
**1.00**, shape **1.00**, pitch **0.79**, timbre **0.94**, envelope **1.00**. Text-only held-out
transcript phrasings, with image/audio zeroed, score **0.92 / 0.97 / 0.48 / 0.91 / 0.95**. Free
greedy text-only extraction reaches **0.80 exact trace-fact match** over the small eval sample.
The stronger semantic gate is counterfactual text intervention: edit exactly one described factor
in a held-out transcript, keep image/audio zeroed, and require only the matching canonical fact to
move. Teacher-forced value-position intervention has mean target accuracy **0.92** and collateral
stability **0.93**. The free greedy version, where the model must emit the edited fact trace from
`extract` alone, reaches mean target accuracy **0.95**, collateral stability **0.90**, and exact
all-fact match **0.62**. This does not prove broad natural-language understanding; it proves that
the bridge can learn a configurable transcript surface bank as a sensory input and bind edited
words to canonical facts rather than merely predicting the next caption token.

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
python -m thinking.cli eval   RUN [--mode free|verified|masked|path|extract|self|write|math|
                              define] [--split iid|holdout|novel] [--hops 2,3]
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
| B6 | + budget (50k steps, fixed-phase curriculum) | 0.45 — UNCHANGED: phase pools memorize |
| B7b | bank, NO curriculum, always-fresh pool | **0.75 trained / 0.90 held-out phrasings** |
| C4 | + deep-20 trees, verified-decode fix | verified k=20 **0.45** (was ~0), free 0.50; shallow starved by 50% deep mix |
| C6d | + contrastive questions (3/tree) + goal anchor | k=2 OFF ZERO (0.10-0.20); verified k=3 **0.75** |
| L (rope) | trained depth<=6, eval 6/10/20/40 | 0.70/0.40/0.17/0.00 — no cliff; per-line error compounds |
| L (nope) | same, no positions | 0.40/0.60/0.00/0.00 — position mode not decisive |
| CONS | contrastive 0.9, 36k steps (2x C6d) | k=2 **0.45-0.55**, verified k=3 **0.95** / free **1.00**; deep dipped (k=20 free 0.45) — trio inflation diluted effective deep share to ~0.20 |
| C5 | deep_frac 0.3 rebalance (18k) | **verified k=3 0.60 > free 0.55**; deep k=20 0.40-0.45; held-out phrasings ≈ trained |
| TR1 | trace-rank objective, depth-16 train -> depth-30 eval | constrained decode **1.00 acc / 1.00 valid**; ranker top1 0.61, top5 0.80, MRR 0.70 over 16 candidates; decode ~4.2 ms/example |
| LC1 | 500/1000-step learning curve, aux vs noaux, eval depth-30 | all arms reached **1.00 acc / 1.00 valid**; aux@1000 hit top1/MRR **1.00/1.00**; noaux solved rollout but had weak rule geometry (FER margin 0.02-0.06 vs aux 0.62-0.68) |

Latest finding: trace-rank improves usable trace execution before we add any hand-coded
domain rules. The model is learning to select legal next actions from a generic frontier
more reliably; the auxiliary rule losses are not needed for this small depth-30 accuracy
check, but they keep the internal rule space separated enough to support scaling.

Root causes found en route (now defaults/tests): unseen-name embeddings → anonymization;
head-first lines → premises-first; fixed-pool epochs → rolling refresh; d=128 → d=256;
mHC zero-init write gate (block got zero gradient); duplicate-line loops → checker dedup;
fixed-phase curricula memorize → always-fresh mixed pools (`--no-curriculum`);
verified-mode drops → skip-don't-die + temp-0.5 resampling.

## 8. Roadmap

Rung C: deep trees at scale. Rung D: the full exam battery in one model. Then: digit-level
arithmetic traces (atomic-token math can't emit unseen differences), the verbalizer span-copy
fix (BPE fragments names), model-proposed rule induction (the enumerate-and-test learner already
hits 0.99 — the model should propose the candidates), and the self-verify flywheel: verified
traces become the next generation's training data.
