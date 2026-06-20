# LOTA — a small, expressive Language Of Thought & Action (v1)

*Working name; rename freely.* Reference implementation: `thinking/lang.py`
(`python -m thinking.lang --selftest | --demo | --eval "(...)"`).

## Why this exists (the foundation step)

We measured (see `C2_ROADMAP.md §6`) that a from-scratch model learns English **vocabulary** but not
**cross-sentence meaning**, and scale didn't fix it. Meanwhile every result in this repo that
*generalized* was **typed, compositional, and engine-verifiable**. Conclusion: make the generalizing
thing the substrate. Give the agent **its own language** first — small, expressive, and **executable**.
Natural language then becomes a **mapping** problem, decomposed into three learnable + checkable steps:

```
   natural language  --parse-->  LOTA  --reason (proof-checked)-->  LOTA  --generate-->  natural language
                                  ^ the agent thinks, knows, and ACTS here ^
```

## Design (locked)

- **Grounding: unified knowledge + action.** Events subsume both facts ("a robin can fly") and acts
  ("pick up the block"). One language to think *and* do.
- **Syntax: S-expressions.** Regular, unambiguous, trivial to parse — and easy for a small model to
  emit token-by-token and for an engine to check.
- **Expressivity: full** — relations, events, actions, `not/and/or/if/iff`, `forall/exists` (with
  restrictors), modality (`can/must/may`), attitudes (`believe/want/goal`), tense/aspect modifiers.
- **Small via composition.** ~8 construct families; everything else is a user predicate or entity.
  Expressivity comes from *composing* the primitives, not from a big vocabulary.
- **Executable / verifiable.** Everything desugars to one first-order core over ground atoms,
  evaluated against a **world** (atoms + Datalog rules) reusing `datalog.py` — the verified engine
  whose closed-world negation + proofs already generalized in our experiments.

## Surface grammar (informal)

```
form    := (pred term*)                         ; predication (positional args)
         | (event TYPE :role term ...)          ; event: reified, ∃ an event entity
         | (do    TYPE :role term ...)          ; action: an event the agent executes
         | (not form) | (and form+) | (or form+) | (if form form) | (iff form form)
         | (forall BIND form) | (exists BIND form)
         | (can form) | (must form) | (may form)
         | (believe :agent term :that form) | (want :agent term :that form) | (goal form)
BIND    := ?v | (?v KIND)                       ; optional restrictor: (forall (?x bird) ...)
term    := name | ?var | "literal" | (DET KIND :role term ...)
DET     := a | an | some | the | all | every | each | no
:role   := :agent :patient :recipient :loc :time :tense :aspect :color ...   (open set)
```

## Semantics (v1, honest about depth)

Desugar → first-order core, then `evaluate` against the world (`atoms` + Datalog `rules`), **closed
world**:

- **Predication** `(isa robin bird)` → atom; true iff in the Datalog **closure** (so derived facts
  count: `(animal robin)` via `bird→animal`).
- **Determiner NPs** are **quantifier-raised**: `(a mouse)`→ `∃x. mouse(x) ∧ …`; `(all bird)`→
  `∀x. bird(x) → …`; `(no cat)` → `¬∃x. cat(x) ∧ …`. `(the …)` is existential in v1 (uniqueness not
  enforced yet).
- **Events** `(event chase :agent (the cat) :patient (a mouse))` →
  `∃e. chase(e) ∧ agent(e,x) ∧ cat(x) ∧ patient(e,y) ∧ mouse(y)`.
- **Quantifiers** range over the world's **domain** (constants in the closure). `not` = negation-as-
  failure (unprovable ⇒ false).
- **Modality** — the one place we go *open-world*, deliberately:
  - `must φ` = φ is **provable** (necessity ≈ known-true).
  - `can φ` / `may φ` = φ is **not explicitly ruled out** (possibility). Explicit impossibility of an
    atom `(p a…)` is the fact `(~p a…)`. So a *merely unknown* φ is still **possible** even though its
    plain `(not φ)` is true by failure — that contrast is exactly why modality earns its place.
- **Attitudes** (`believe/want/goal`) are **structural** in v1 (is φ a registered goal/belief?), not
  yet a possible-worlds semantics.
- **Tense/aspect** (`:tense past`, `:aspect prog`) are **recorded** on the event but not truth-
  evaluated yet.

## Actions (grounding)

`(do TYPE :role filler …)` executes via a **schema** = precondition + add/delete effects:

```
pickup: pre (clear ?patient)   add (holding ?agent ?patient)   del (clear ?patient)
```

`execute("(do pickup :agent self :patient blockA)", world)` checks the precondition against the world,
then mutates it. This is where symbols become **grounded** in what the agent can do; swap the toy
schemas for real tool calls and the same language drives an agent.

## What's deliberately deferred to v2

- definite-reference uniqueness for `(the …)`; richer scope control for nested quantifiers.
- deep possible-worlds semantics for modality + attitudes; tense/time evaluation against a timeline.
- a **type system with declared predicate signatures** (v1 checks structure/category, not types).
- the **learning bridge**: train the model to (a) parse NL→LOTA and (b) generate LOTA→NL, gated by
  round-trip + `evaluate` faithfulness (this is the payoff that makes LOTA the road to C2).

## Status

v1 parses, type-checks (structural), desugars, evaluates (relations, quantifiers, negation, modality,
events, attitudes), and executes actions — all covered by `--selftest` (passing) and `--demo`.
It is a real, executable agent language, not a sketch. C2 remains unmet; this is the substrate
underneath the path to it.
