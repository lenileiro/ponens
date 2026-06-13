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
python -m thinking.text --import-squad --squad-train 5000 --squad-eval 1000 \
    --squad-max-context-tokens 160 --squad-max-question-tokens 40 \
    --squad-max-answer-tokens 8 --seed 23 --out data/text_squad.jsonl
python -m thinking.text --import-squad --squad-train 800 --squad-eval 200 \
    --squad-max-context-tokens 96 --squad-max-question-tokens 32 \
    --squad-max-answer-tokens 5 --seed 29 --out data/text_squad_smoke.jsonl
python -m thinking.text --import-squad --squad-choice-only --squad-choice-n 4 \
    --squad-train 1200 --squad-eval 300 --squad-max-context-tokens 128 \
    --squad-max-question-tokens 32 --squad-max-answer-tokens 6 --seed 31 \
    --out data/text_squad_choice_smoke.jsonl
python -m thinking.text --import-squad --squad-choice-only --squad-choice-n 4 \
    --squad-choice-swap-negatives 1 --squad-train 1800 --squad-eval 450 \
    --squad-max-context-tokens 128 --squad-max-question-tokens 32 \
    --squad-max-answer-tokens 6 --seed 37 \
    --out data/text_squad_choice_neg_smoke.jsonl
python -m thinking.text --import-squad --squad-choice-only --squad-choice-n 4 \
    --squad-choice-swap-negatives 1 --squad-choice-absent-negatives 1 \
    --squad-train 2400 --squad-eval 600 --squad-max-context-tokens 128 \
    --squad-max-question-tokens 32 --squad-max-answer-tokens 6 --seed 41 \
    --out data/text_squad_choice_absent_neg_smoke.jsonl
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
python -m thinking.text --data data/text_squad_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_control_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 12 --study-rounds 1 --study-strategy errors \
    --study-probe-n 128 --study-hard-max 96 --study-select-best \
    --study-score-metric both --study-retention-w 2.0 --study-control-w 2.0 \
    --batch 24 --study-lr 0.0005 --decode-w 0.25 --semantic-w 1.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 10 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 36 \
    --out runs/text_study_squad_control_smoke.json
python -m thinking.text --data data/text_squad_choice_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_all_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 80 --study-rounds 1 --study-strategy all --study-select-best \
    --study-score-metric both --study-retention-w 2.0 --study-control-w 2.0 \
    --batch 32 --study-lr 0.0005 --decode-w 0.25 --semantic-w 1.0 \
    --balance-by kind --fact-n 120 --kind-fact-n 20 --artifact-n 120 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_all_smoke.json
python -m thinking.text --data data/text_squad_choice_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_neg_guard_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 100 --study-rounds 1 --study-strategy all --study-select-best \
    --study-score-metric both --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0.25 \
    --semantic-w 1.0 --balance-by kind --fact-n 160 --kind-fact-n 30 \
    --artifact-n 160 --free-n -1 --paraphrase-n -1 --counterfactual-n -1 \
    --max-new 24 --out runs/text_study_squad_choice_neg_guard_smoke.json
python -m thinking.text --data data/text_squad_choice_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_seeded_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 100 --study-rounds 1 --study-strategy all --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0.25 \
    --semantic-w 0.5 --choice-w 1.0 --balance-by kind --fact-n 160 \
    --kind-fact-n 30 --artifact-n 160 --free-n -1 --paraphrase-n -1 \
    --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_seeded_smoke.json
python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_absent_answerw4_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 100 --study-rounds 1 --study-strategy all --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 4.0 --choice-none-w 1.0 \
    --balance-by kind --fact-n 160 --kind-fact-n 30 --artifact-n 160 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_absent_answerw4_smoke.json
python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_evidence_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy all --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 20 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_evidence_2round_smoke.json
python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_errors_kindguard_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy errors --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 20 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_errors_kindguard_2round_smoke.json
python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_candidate_ctx_pair_controlguard_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy errors --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --choice-pair-w 1.0 --choice-pair-margin 0.0 --balance-by kind \
    --fact-n 80 --kind-fact-n 20 --artifact-n 80 --free-n -1 \
    --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_candidate_ctx_pair_controlguard_2round_smoke.json
python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_candidate_ctx_pair_ctrlcontrast_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy errors --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --choice-pair-w 1.0 --choice-pair-margin 0.0 --choice-control-w 0 \
    --choice-control-contrast-w 1.0 --choice-control-margin 0.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 20 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_candidate_ctx_pair_ctrlcontrast_2round_smoke.json
python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_contextloc025_pair_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy errors --study-select-best \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --choice-context-w 0.25 --choice-pair-w 1.0 --choice-pair-margin 0.0 \
    --choice-control-w 0 --choice-control-contrast-w 0 --choice-control-margin 0.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 20 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_contextloc025_pair_2round_smoke.json
python -m thinking.text --data data/text_squad_choice_absent_neg_smoke.jsonl \
    --study-checkpoint runs/text_snli_hans_grounded_balanced.pt \
    --study-out-checkpoint runs/text_study_squad_choice_answerability_candctx025_confirm1_qctx_2round_smoke.pt \
    --study-replay-data data/text_grounded.jsonl \
    --steps 10 --study-rounds 2 --study-strategy errors --study-select-best --study-confirm-n 1 \
    --study-score-metric choice --study-retention-w 2.0 --study-control-w 2.0 \
    --study-kind-w 1.0 --batch 32 --study-lr 0.0005 --decode-w 0 \
    --semantic-w 0 --choice-w 1.0 --choice-answer-w 1.0 --choice-none-w 1.0 \
    --choice-answerability-w 0.5 --choice-answerability-control-w 0.5 \
    --choice-answerability-contrast-w 0.5 --choice-answerability-contrast-margin 0.0 \
    --choice-answerability-pair-w 0.5 --choice-answerability-pair-margin 0.0 \
    --choice-context-w 0.25 \
    --choice-candidate-context-w 0.25 --choice-candidate-context-margin 0.0 \
    --choice-question-context-w 0.25 \
    --choice-question-context-contrast-w 0.10 --choice-question-context-margin 0.0 \
    --choice-pair-w 1.0 --choice-pair-margin 0.0 \
    --choice-control-w 0 --choice-control-contrast-w 0 --choice-control-margin 0.0 \
    --balance-by kind --fact-n 80 --kind-fact-n 20 --artifact-n 80 \
    --free-n -1 --paraphrase-n -1 --counterfactual-n -1 --max-new 24 \
    --out runs/text_study_squad_choice_answerability_candctx025_confirm1_qctx_2round_smoke.json
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
resistant web data and add richer semantic datasets (e.g. MultiNLI/CFQ/GeoQuery/SQuAD-style
answer facts) plus stronger paraphrase and counterfactual splits.

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

SQuAD v1.1 is now a second real reading-task source. `--import-squad` downloads the official
context/question/answer JSON, windows each passage around the labeled answer span, and can render
either extractive answer facts (`answer length`, `a000 answer_token`, ...) or contrastive
multiple-choice facts (`answer choice c00x`) whose candidates are shuffled real answers from the
same paragraph. The importer writes filtering/windowing stats so skipped long-answer examples are
visible instead of hidden. Evaluation now also includes QA controls: full context+question must
beat question-only, context-only, and same-paragraph question-swap inputs, otherwise the study
selector receives a shortcut-control penalty via `--study-control-w`. `--study-kind-w` penalizes
updates that improve the aggregate score by sacrificing an entire eval kind, and the study selector
now refuses non-positive total scores unless `--study-allow-negative-score` is set.

A local 800/200 extractive SQuAD smoke expanded the balanced text checkpoint from **11,251** to
**20,633** text tokens and from **26** to **1,760** fact values. After a 12-step hard-example
replay study, sampled SQuAD semantic-head accuracy rose from **0.049** to **0.117**, but
teacher-forced accuracy fell (**0.019** to **0.004**) and QA ablation gaps remained below the 0.05
gate. The contrastive choice smoke is more tractable: four-choice SQuAD imports add only **4**
fact values. An 80-step all-record replay study improved sampled teacher-forced choice accuracy
from **0.000** to **0.283**, kept grounded replay gated, and scored positive before kind-collapse
guarding, but still failed understanding controls (`full - context_only = 0.000`, question-swap
teacher gap **0.000**, semantic question-swap gap **0.026**).

Question-swap negatives add an explicit data-derived `answer choice none` target for swapped
questions whose answers are not in the candidate list. The first 100-step negative-swap study
learned the negative class while collapsing positive SQuAD choices (`squad_choice` sampled
teacher/semantic **0.000**, `squad_choice_swap_negative` **1.000**). With `--study-kind-w 1.0`
and the positive-score guard, the same collapsed round is rejected (`score_allowed=false`,
selected round **0**, checkpoint deltas **0.000**), preserving replay and refusing a misleading
update.

The candidate-aware choice head is now separate from the fixed semantic value head: it scores each
record's candidate spans plus a learned `none` option, uses a target-balanced positive/none loss
(`--choice-w`), and can be selected with `--study-score-metric choice`. Checkpoint expansion and
eval now seed missing/new weights from `--seed`, so repeated study commands are reproducible. The
seeded 100-step SQuAD choice smoke improved aggregate candidate-head accuracy from **0.188** to
**0.463**, but the selector still rejected the update (`score_allowed=false`, selected round
**0**) because `squad_choice` fell to **0.033** while `squad_choice_swap_negative` reached
**0.833**, grounded replay teacher accuracy dropped from **0.983** to **0.922**, and QA choice
control gaps stayed below threshold. This proves the architecture can train a data-derived
candidate scorer and reject a shortcut update; it is still not language understanding.

Answer-absent negatives are now another data-derived SQuAD choice control. They keep the original
context and question but replace the choices with same-paragraph answer candidates that exclude the
gold answer, so `answer choice none` cannot be learned solely from a swapped-question surface.
The new 2.4k/600 smoke writes positives, swapped-question negatives, and answer-absent negatives.
A 100-step choice-only run with extra positive pressure (`--choice-answer-w 4.0`) still failed to
beat the guarded baseline: the candidate-head round scored **0.106** aggregate, kept
`squad_choice` at **0.233**, and only weakly learned answerability (`squad_choice_absent_negative`
**0.067**, `squad_choice_swap_negative` **0.133**). The selected round stayed **0**. This rules
out simple scalar loss weighting as the fix.

The candidate scorer now uses an evidence threshold instead of a static `none` vector. The question
span attends over context tokens, each candidate span is scored against both the question and the
attended context, and `none` wins only when no candidate clears the learned threshold. That is a
better architecture for answerability, but the current small study still rejects training updates:
the 100-step evidence run learned both negative kinds perfectly while collapsing real choices
(`squad_choice` **0.000**, absent/swap negatives **1.000**), and the lighter 2-round/10-step
self-study run selected round **0** over both trained rounds. Its untrained expanded head was more
balanced on the sampled eval (`squad_choice` **0.250**, absent **0.150**, swap **0.400**), while
both trained rounds again drove `squad_choice` to **0.000**. The guarded self-study machinery is
doing its job; the next text rung needs richer positive supervision or a stronger candidate
matching objective before accepting reading-task weight updates.

The choice objective now combines candidate ranking for real answers with a learned threshold for
answerability, and `--study-strategy errors` is choice-aware when `--study-score-metric choice` is
selected. That means reading-task self-study mines the candidate-head records the model actually
misses instead of using semantic-head errors as a proxy. On the 2-round SQuAD choice/absent/swap
smoke, round 1 mined **1,776 / 2,400** choice errors and round 2 mined **1,407 / 2,400**. Aggregate
choice accuracy could rise from **0.338** to **0.625**, and the QA choice control gaps cleared the
0.05 thresholds, but the real-answer kind still regressed (`squad_choice` **0.250** to **0.150**)
while negative kinds dominated (absent **0.850**, swap **0.900**). The selector now compares each
eval kind against the pre-study baseline, records `kind_regressions`, and blocks a round when the
kind guard is enabled. The same smoke therefore keeps `selected_round` **0** with checkpoint deltas
**0.000**. This is a stronger self-teaching guard, not language mastery; the remaining gap is a
positive candidate-matching objective that can improve real answers without leaning on `none`.

The next candidate objective is paired rather than label-count based. SQuAD positives are paired
with their dataset-derived answer-absent or swapped-question negatives through `base_id`; the
paired loss (`--choice-pair-w`) asks the positive answer evidence to outrank the paired `none`
record's candidate evidence. The choice scorer also now retrieves context evidence per candidate
instead of sharing one question-attended context vector across every option. That made the initial
expanded head less lucky on the sampled SQuAD choice eval (**0.150** aggregate), but after 10
choice-error steps the candidate head reached **0.400** without regressing `squad_choice`
(**0.200** before and after; absent/swap negatives **0.450**/**0.450**). The update still failed
understanding controls (`qa_choice_full_minus_question_only` **0.0375**,
`qa_choice_full_minus_context_only` **-0.0750**, `qa_choice_full_minus_question_swap` **-0.0385**).
Selection now records `control_failures` and, when `--study-control-w` is enabled, treats failed
shortcut controls as hard blockers instead of only score penalties. The strict-control smoke keeps
`selected_round` **0** and checkpoint deltas **0.000**. This is the right failure: the model can
fit more of the task, but it is still not allowed to call that understanding until it beats the
ablations.

Training-side QA controls are now generated from the same data, not from English rules.
`--choice-control-w` samples question-only/context-only ablations from positive choice records and
trains them as `answer choice none`; `--choice-control-contrast-w` instead keeps the original target
and asks the full context+question target evidence to outrank the ablated target evidence. Both are
guarded by the same held-out ablation checks. In the 2-round smoke, blunt ablation-none training
overcorrected into `none`: with weight **1.0**, aggregate choice reached **0.588** but
`squad_choice` fell from **0.200** to **0.050**; with weight **0.25**, aggregate reached **0.500**
but `squad_choice` still fell to **0.050**. The contrastive control loss avoided the blunt `none`
target, but still did not solve the shortcut: one round preserved `squad_choice` at **0.200** while
choice controls became worse (`question_only` gap **-0.4875**, `context_only` **-0.2000**), and the
other improved negatives while leaving all control gaps negative. All control-training smokes keep
`selected_round` **0**. The next model-side gap is not more scalar loss pressure; it needs a
stronger mechanism for binding question predicates to matching context spans before answerability
training can be trusted.

The choice head now has an explicit data-derived context localization loss (`--choice-context-w`).
For positive SQuAD choice records, the target choice tokens are located in the context token stream,
and the question+candidate context attention is trained to place probability mass on that answer
span. This is not an English rule; it is span supervision extracted from the reading record. With
`--choice-context-w 1.0`, the model improved negatives while preserving `squad_choice` in the first
round, but still failed context-only and question-swap controls. With the lighter
`--choice-context-w 0.25`, round 1 was the cleanest result so far: sampled choice accuracy improved
from **0.150** to **0.4125**, `squad_choice` stayed at **0.200**, answer-absent rose to **0.450**,
and swapped-question negatives rose to **0.400**. The selector still rejected it because held-out
controls remained below threshold (`context_only` gap **-0.0375**, question-swap gap **-0.0385**).
Adding same-paragraph question-swap contrast on top made the small head worse, not better. The
current evidence says span localization helps candidate evidence, but the architecture still needs
a stronger question-conditioned binding mechanism before a reading update should be accepted.

The scorer now includes that stronger binding path: candidate context evidence is gated by a
question-to-context attention distribution instead of letting the context stand in for the question
when a control ablates it. `--choice-question-context-w` trains that question-only attention to
locate the data-labeled answer span, and `--choice-question-context-contrast-w` asks the full
question's answer-span attention to outrank a same-context swapped question's attention to the
original answer. This is the first objective to pass all held-out choice shortcut controls in a
round: with `--choice-context-w 0.25 --choice-question-context-w 0.25
--choice-question-context-contrast-w 0.10`, round 1 reached sampled choice accuracy **0.425** and
control gaps `question_only` **0.100**, `context_only` **0.1625**, `question_swap` **0.0769**.
The selector still rejected the update because the real-answer `squad_choice` kind regressed
from **0.200** to **0.150** while the negative kinds improved (**0.700** / **0.750**). Raising
positive answer weight to **2.0** flipped the failure: `squad_choice` rose to **0.250**, but both
negative kinds collapsed to **0.000** and controls failed. The current bottleneck is therefore not
a missing scalar weight; it is preserving positive and negative answerability simultaneously while
maintaining the new binding controls.

Two retention/calibration probes make that bottleneck sharper. `--study-anchor-correct-per-kind`
mixes currently-correct train records back into error-mined self-study batches, so the model
teaches itself from misses without forgetting examples it already handles. On the qctx
swap-contrast smoke, 32 anchors per kind were correctly added (**96** anchors total; **32** each
for `squad_choice`, answer-absent, and swapped-question negatives), but the accepted checkpoint
still stayed at round **0**: round 1 reached choice accuracy **0.450** while `squad_choice`
regressed to **0.100** and question-swap control fell to **0.000**. `--choice-answer-margin`
adds a margin between positive evidence and the learned `none` threshold without changing class
weights; at margin **0.5**, round 1 reached **0.2375** choice accuracy and kept negatives usable
(**0.400** / **0.500**), but `squad_choice` still regressed to **0.150** and question-only /
question-swap controls remained just under threshold (**0.0375** / **0.0385**). The evidence now
points away from more sampling or scalar loss knobs and toward a representation that models
answerability as a joint relation between question predicate, context span, and candidate span.

The current scorer also adds a small learned context-mass term: if the question-to-context
distribution places more mass on a candidate's matching context span than a uniform distribution
would, the candidate logit gets learned support. Answer-absent negatives now still supervise the
question-to-context distribution toward the true answer span, so "none" is learned as an uncovered
answer relation instead of as missing answer evidence. With this path and the same qctx contrast,
round 2 reached choice accuracy **0.4125**, preserved all three choice kinds
(`squad_choice` **0.400**, answer-absent **0.500**, swapped-question **0.550**), and retained
replay (**0.965** semantic), but controls failed again (`question_only` **-0.0125**,
`question_swap` **-0.0769**), so the selector kept round **0**. Adding
`--choice-control-contrast-w 0.25` did not rescue it: the best round stayed rejected, with
question-only and context-only gaps negative in round 1 and all three shortcut controls negative
by round 2. That makes the next architectural target clearer: the model needs an answerability
head that separates "the question finds an answer span" from "this candidate covers that span",
rather than pushing both jobs through the candidate logits alone.

That answerability head is now implemented as a learned coverage verifier. It builds a
question-selected context vector, compares it to each candidate span through separate learned
projections, trains positive/none answerability with `--choice-answerability-w`, trains generated
question-only/context-only/swapped-question none controls with `--choice-answerability-control-w`,
and trains the full question's covered-span score to outrank the same-context swapped question
with `--choice-answerability-contrast-w`. This is still data-derived supervision: the labels come
from the imported reading records and generated counterfactual records, not from fixed English
facts. On the non-confirmed two-round smoke, the selector accepted round **2** for the first time
in this SQuAD-choice line: choice accuracy rose from **0.1875** to **0.325**, replay stayed high
(**0.97125**, a **0.005** drop from reference), all held-out choice shortcut controls passed
(`question_only` **0.1125**, `context_only` **0.200**, `question_swap` **0.1154**), and no primary
kind regressed (`squad_choice` **0.300**, answer-absent **0.600**, swapped-question negatives
**0.400**).

The selector now has an optional confirmation gate: `--study-confirm-n` reruns selection on
additional held-out evaluation seeds and compares each candidate against the initial checkpoint on
the same sample. With `--study-confirm-n 1`, the same training run was correctly rejected: primary
round 2 still passed all controls, but confirmation seed **7936** found a `squad_choice` regression
from **0.250** to **0.100** while the negative kinds improved (**0.550** / **0.550**), producing
score **-0.0525** and keeping selected round **0**. This is not language mastery, but it is a
stronger reading-update gate: a candidate update must now retain prior grounded facts, pass
shortcut controls, and preserve answerable examples across independent held-out samples before it
can be accepted.

The answerability head now also has a paired contrast (`--choice-answerability-pair-w`) over
SQuAD records that share a `base_id`: answer-present examples must score above their
answer-absent or swapped-question partners. This is the answerability analogue of
`--choice-pair-w`, and it uses only imported record structure. In the confirmation-gated smoke,
the pair loss optimized (`ans-pair` fell from **0.648** in round 1 to **0.129** in round 2), but
the selector still kept `selected_round` **0**. Primary round 2 reached **0.2625** sampled choice
accuracy with `squad_choice` **0.300**, absent **0.350**, and swap **0.500**, but confirmation
seed **7936** exposed a `squad_choice` regression from **0.250** to **0.050** and a failed
question-swap control. The learned objective is aligned, but it is not sufficient; the next text
step needs candidate-span binding that keeps real answers stable under independent held-out
samples.

Candidate-span binding is now explicit via `--choice-candidate-context-w`: for answer-present
SQuAD choice records, the correct candidate must localize the answer span better than the
distractor candidates. This is still span supervision from the imported reading record, not a
hand-authored English rule. With weight **0.25**, the confirmation-gated smoke improved the failure
mode: primary round 2 moved `squad_choice` to **0.450**, and confirmation seed **7936** preserved
`squad_choice` at its **0.250** baseline with no kind regressions. The update is still rejected
because controls failed (`question_only` **0.0250**, `question_swap` **-0.0769**), so
`selected_round` remains **0**. That narrows the remaining gap: the model can now retain answerable
choices better, but it must still prove that the answer depends on both the actual question and
the actual context before a reading update can be accepted.

The final evaluated choice head now has optional direct supervision too:
`--choice-final-w` trains `model.choice_logits` after answerability is added, and
`--choice-final-control-w` trains generated question-only/context-only/swapped-question controls
as `none` through those same final logits. This closes an instrumentation gap: earlier losses
trained raw candidate evidence and answerability separately, while evaluation used their sum. The
first probes show why the gate is still needed. At control weight **0.25**, aggregate sampled
choice rose to **0.550**, but `squad_choice` collapsed to **0.000** and confirmation seed **7936**
regressed from **0.250** to **0.000**. At control weight **0.05**, round 2 kept better aggregate
choice (**0.4875**) and held primary control gaps closer, but confirmation still regressed
`squad_choice` from **0.250** to **0.050** while negatives dominated (**0.750** /
**0.700**). Both runs keep `selected_round` **0**. The useful result is diagnostic: final-logit
control pressure reaches the evaluated head, but the next accepted update needs a
positive-preserving calibration mechanism rather than stronger `none` pressure.

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
    --semantic-guidance-w 2.0 \
    --out runs/image_latent_mmdit_text.pt
python -m thinking.image_embed --manifest data/images/train.jsonl \
    --root data/images --backend hf --model google/siglip-base-patch16-224 \
    --features both --batch 64 --device cuda --out data/images/embeddings.jsonl \
    --report-out runs/image_embed_report.json
python -m thinking.image_data --manifest data/images/train.jsonl \
    --root data/images --min-side 256 --max-aspect 2.0 \
    --embedding-manifest data/images/embeddings.jsonl --embedding-key image \
    --min-caption-tokens 3 --write-filtered data/images/train_clean.jsonl \
    --report-out runs/image_manifest_report.json
python -m thinking.image_latent --train --cond-mode text --flow-arch mmdit \
    --image-manifest data/images/train_clean.jsonl --image-root data/images \
    --caption-cond-source auto \
    --image-quality-weight 1.5 \
    --size 64 --ae-arch residual --latent-downsample 8 --latent-max-tokens 128 \
    --ae-recon-loss hybrid --ae-grad-w 0.1 --ae-ms-w 0.1 \
    --image-text-align-w 0.1 --flow-text-align-w 0.05 --text-embed-dim 128 \
    --image-feature-align-w 0.1 --flow-feature-align-w 0.05 \
    --image-feature-embed-dim 128 \
    --ae-accum-steps 2 --flow-accum-steps 2 --grad-clip 1.0 \
    --flow-cache-latents --flow-cache-dir runs/image_manifest_cache \
    --flow-cache-shard-size 2048 --flow-cache-batch 32 \
    --ae-steps 400 --flow-steps 400 --sample-steps 8 \
    --flow-consistency-w 0.05 --sample-grid-out runs/image_manifest_grid.ppm \
    --out runs/image_manifest_mmdit.pt
python -m thinking.image_latent --eval-checkpoint runs/image_manifest_mmdit.pt \
    --eval-image-manifest data/images/train_clean.jsonl --eval-image-root data/images \
    --eval-image-split eval --size 64 --cfg-scales 1.0,1.25,1.5,2.0 \
    --cfg-rescales 0.0,0.7 --sample-steps-list 4,8,16 --eval-seeds 1,2,3 \
    --sample-grid-out runs/image_manifest_eval_grid.ppm \
    --eval-out runs/image_manifest_mmdit_sweep.json
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
    --image-flow-consistency-w 0.05 \
    --image-time-sampling logit-normal --image-flow-ema-decay 0.999 \
    --image-latent-normalize channel --image-latent-stat-samples 1024 \
    --image-cfg-interval 0.0,0.8 --image-semantic-guidance-interval 0.0,0.75 \
    --image-ae-intervention-w 0.1 --image-ae-factor-orth-w 0.05 \
    --image-semantic-guidance-w 2.0 --image-semantic-guidance-sweep 0.0,1.0,2.0 \
    --image-sample-methods euler,heun --image-cfg-rescale-sweep 0.0,0.7 \
    --image-sample-grid \
    --image-eval-sweep --fast --go
RUNPOD_API_KEY=... python runpod/launch_thinking.py --upload-image-data --image-embed \
    --image-embed-model google/siglip-base-patch16-224 \
    --image-clean-min-side 256 --image-clean-max-aspect 2.0 \
    --image-latent --image-latent-arch mmdit \
    --image-cond-mode text --image-dit-head-width-mult 2 --image-dit-qk-norm \
    --image-dit-attn-impl sdpa --image-dit-pos-embed sincos2d \
    --image-manifest data/images/train.jsonl --image-root data/images \
    --image-caption-cond-source auto \
    --image-crop-mode pad --image-hflip-prob 0.5 \
    --image-quality-weight 1.5 \
    --image-size 128x192 --image-size-buckets 128x128,128x192,192x128 \
    --image-ae-arch residual --image-latent-downsample 8 \
    --image-latent-max-tokens 384 \
    --image-ae-recon-loss hybrid --image-ae-grad-w 0.1 --image-ae-ms-w 0.1 \
    --image-text-align-w 0.1 --image-flow-text-align-w 0.05 \
    --image-text-embed-dim 128 \
    --image-feature-align-w 0.1 --image-flow-feature-align-w 0.05 \
    --image-feature-embed-dim 128 \
    --image-flow-repa-w 0.05 --image-flow-repa-steps 20000 \
    --image-flow-repa-embed-dim 128 \
    --image-ae-accum-steps 2 --image-flow-accum-steps 2 \
    --image-train-precision bf16 --image-grad-clip 1.0 \
    --image-flow-cache-latents --image-flow-cache-dir runs/image_manifest_cache \
    --image-flow-cache-shard-size 2048 --image-flow-cache-batch 64 \
    --image-sample-steps 8 --image-flow-consistency-w 0.05 \
    --image-time-sampling logit-normal --image-time-shift 1.25 \
    --image-time-shift-mode dim --image-time-shift-ref-dim 1024 \
    --image-flow-loss-weight min-snr-v --image-flow-loss-weight-gamma 5.0 \
    --image-cfg-rescale 0.7 --image-cfg-rescale-sweep 0.0,0.7 \
    --image-latent-normalize channel --image-latent-stat-samples 4096 \
    --image-eval-sweep --image-eval-split eval --image-sample-grid --fast --go
```

`thinking.image2` is the head-aware FER experiment: shared factored heads vs explicit
bottleneck vs one joint color×shape classifier on held-out color/shape combinations. It reports
both the old embedding probe and the new factor-space probe, fixing the Image-1 blind spot where
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
The factor-space probe now tracks behavior (`best_factor_space_arm = bottleneck`), while the old
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
updates in both streams now have condition-dependent gates, and old MM-DiT checkpoints without gate
weights are tolerated by the loader with zero-initialized identity-centered gates.

Image-14 adds EMA checkpointing/evaluation for the latent flow and learned text conditioner. Train
reports can evaluate the averaged weights, checkpoints save both raw and EMA states, and
`--eval-checkpoint` prefers EMA when available while keeping `--no-ema-checkpoint` for raw-weight
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
`--image-dit-head-width-mult`) for DiT, Cross-DiT, and MM-DiT flows. The default remains the
old linear head for checkpoint compatibility; setting the multiplier above 1 gives the transformer
a wider projection head for semantically rich latents. This follows the RAE/REPA direction: improve
the model's capacity to operate in representation space without adding renderer-specific rules.

Image-26 adds checkpointed latent normalization (`--latent-normalize none|global|channel`;
RunPod: `--image-latent-normalize`). The flow can now train and sample in normalized AE latent
coordinates while semantic losses, guidance, decoding, and visual grids still operate in raw AE
latent space. This is a generic scale fix for semantically rich/high-dimensional latents: it
stabilizes coordinate scale without hard-coding visual factors or renderer rules, and old
checkpoints default to `none`.

Image-27 adds guidance intervals (`--cfg-interval`, `--semantic-guidance-interval`; RunPod:
`--image-cfg-interval`, `--image-semantic-guidance-interval`). CFG and semantic AE guidance can
now be active over selected rectified-flow time ranges while defaults preserve old always-on
behavior. This follows the REPA/guidance-interval direction: guidance becomes a measured sampler
schedule rather than a hard-coded every-step push, and it stays generic over conditions.

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
mse|l1|hybrid`, `--ae-grad-w`, `--ae-ms-w`, and `--ae-latent-reg-w` let real-image runs optimize
pixel fidelity, image gradients, multi-scale structure, and latent magnitude without adding LPIPS
or discriminator dependencies. This is not a replacement for full perceptual/adversarial AE
training, but it moves the local/RPU smoke path away from plain MSE compression and records the
loss configuration in reports/checkpoints.

Image-34 adds GPU training-scale controls for latent image runs. `--ae-accum-steps`,
`--flow-accum-steps`, `--train-precision fp32|bf16|fp16`, and `--grad-clip` make the per-step
effective batch explicit, enable CUDA AMP for bf16/fp16, and clip gradients after accumulation.
CPU tests stay fp32 even if bf16/fp16 is requested, while H100 RunPod jobs can now use bf16 and
larger effective batches without changing the model objective.

Image-35 adds optional cached-latent flow training for manifest runs. `--flow-cache-latents`
encodes the post-AE training manifest once, stores raw AE latents and caption token IDs on CPU,
uses the cache for latent normalization stats, and trains the flow from cached latents instead of
reloading images and re-running `ae.encode` every microstep. This follows the latent-diffusion
separation between compression and generative-model training while keeping the cache opt-in and
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
those external embeddings into the conditioning stream. This is the hook for CLIP/T5/SigLIP-style
frozen text features: the model can move toward stronger prompt understanding without baking in a
specific provider or adding renderer-specific rules.

Image-38 adds disk-backed manifest latent caching. `--flow-cache-dir` writes AE latents,
caption token IDs or text embeddings, and shard metadata to disk, then flow training samples from
those shards instead of keeping the full cache resident in CPU RAM. Reports/checkpoints now expose
`flow_cache_backend`, `flow_cache_dir`, `flow_cache_shards`, shard size, and byte count. This keeps
the latent-diffusion separation practical for larger real-image manifests: encode pixels once,
train the rectified-flow transformer from reusable latent shards, and avoid a hidden RAM ceiling.

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
CLIP/SigLIP text-image encoders or DINO/MAE-style image encoders with `--features image`. The
generator stays model-agnostic: feature extraction, manifest QA, and training remain separate
auditable stages.

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
part of the data-time path while keeping `1.0` as the exact legacy schedule. Checkpoint eval uses
the saved shift by default, `--sample-time-shift` can override it for sweeps, and RunPod exposes the
training knob as `--image-time-shift`.

Image-46 adds optional per-head QK RMSNorm to the custom MM-DiT attention path. Use
`thinking.image_latent --dit-qk-norm` or RunPod `--image-dit-qk-norm` for high-resolution MM-DiT
runs; checkpoint metadata records the flag so qk-normalized checkpoints reload with the same
attention modules, while legacy checkpoints keep the default non-normalized architecture.

Image-47 adds an MM-DiT attention implementation selector. `--dit-attn-impl manual|sdpa|auto`
keeps the original explicit attention path available, uses PyTorch scaled-dot-product attention for
fused/flash kernels when requested, or falls back automatically. RunPod exposes the same setting as
`--image-dit-attn-impl`, and checkpoint metadata records the selected implementation.

Image-48 adds a REPA-style hidden-state representation alignment path for real-image latent flows.
`--flow-repa-w` trains the flow transformer's noisy-step image-token states against generic
manifest `image_embedding` rows, separately from endpoint feature alignment, and
`--flow-repa-steps` can restrict that auxiliary pressure to early training. RunPod exposes the same
settings as `--image-flow-repa-w`, `--image-flow-repa-steps`, and
`--image-flow-repa-embed-dim`; disk latent caches now carry image embeddings when either endpoint
feature alignment or REPA needs them.

Image-49 adds optional Min-SNR-style velocity loss weighting for rectified-flow training.
`--flow-loss-weight none|min-snr-v|soft-min-snr-v` keeps the exact legacy objective by default,
but can reweight per-example velocity MSE by the data-time SNR with batch-mean normalization.
Reports/checkpoints record the weighting mode, gamma, and observed weight range; RunPod exposes the
same controls as `--image-flow-loss-weight`, `--image-flow-loss-weight-gamma`, and
`--image-no-flow-loss-weight-normalize`.

Image-50 adds sampler-side CFG rescale. `--cfg-rescale` rescales guided latent velocities toward
the conditional velocity standard deviation before blending, while `--cfg-rescales` makes it a
checkpoint-sweep axis. RunPod exposes `--image-cfg-rescale` and
`--image-cfg-rescale-sweep`. Default `0.0` preserves legacy CFG behavior.

Image-51 adds aspect-preserving manifest padding. `--image-crop-mode pad` resizes each image to fit
the requested canvas and pads the remaining area instead of square-cropping or distorting it. This
is the minimal data-pipeline step toward mixed-aspect high-resolution training: captioned objects
stay in frame while the current fixed-shape AE/flow code can still batch tensors locally and on
RunPod.

Image-52 adds geometry-aware latent DiT positions. `--dit-pos-embed learned|sincos2d` keeps legacy
learned 1D tables by default, but new high-resolution DiT/CrossDiT/MM-DiT runs can use deterministic
2D sinusoidal image-token positions. RunPod exposes this as `--image-dit-pos-embed`. This removes a
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
instead of treating all kept rows equally; `W=0` is the exact legacy uniform behavior. The same
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

Image-60 lets sample grids use arbitrary prompt text even when a manifest run trained on external
caption embeddings. `thinking.image_latent --eval-checkpoint CKPT --sample-grid-out grid.ppm
--sample-prompts "a glass house at sunrise; a red canoe on a lake"` now routes prompts through
the checkpoint's token conditioner or, for `caption_cond_source=embedding`, through a live
`--prompt-embed-backend stats|hf` text embedder before the learned projection. The HF path uses
the same model family as the manifest sidecar (`--prompt-embed-model MODEL_ID`), validates the
embedding width against `text_embedding_in_dim`, and is exposed on RunPod as
`--image-sample-prompts` plus `--image-prompt-embed-*`. This closes the gap between embedding-based
training and free-form prompt inspection after GPU runs.

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
