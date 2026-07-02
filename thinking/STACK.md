# The verified-reasoning + typed-language stack

A self-contained, in-house, dependency-free stack where an agent **understands language, reasons with
machine-checkable proofs, and answers with a certificate** — built by borrowing the *designs* of mature
systems (Lean's proof kernel; Elixir's set-theoretic types) and realizing them small.

> Status (2026-06-20): every module below has a passing `--selftest`; all committed on
> `comprehension-exam`. **C2 is NOT met** — this is the *substrate* beneath it. Honest gaps in §7.

## 1. The pipeline (end to end)

```
English question
  ──parse (learned NL→LOTA)──▶ LOTA goal
  ──proof SEARCH (datalog, untrusted) + NEURAL guidance──▶ derivation
  ──proof CHECK (sound kernel, trusted)──▶ answer + machine-checkable proof
```

Held-out (unseen entity combinations): **parse 1.000, end-to-end 1.000** — every "yes" carries a
kernel-checked proof; every "no" is sound. (`thinking/pipeline.py`)

## 2. Modules

| module | role | key result (lead-measured) |
|---|---|---|
| `kernel.py` | **sound dependent type-theory kernel** — the only trusted code; proofs are terms; checking = type-checking; predicative universes; + logic prelude (False/And/Or/Eq/Ex) | proves A→B→A, And-commutativity, Eq-refl, ∃-intro; rejects bogus proofs |
| `bdd_types.py` | **set-theoretic category types** (ROBDD, hash-consed); union/intersect/negate; subtyping via emptiness over the isa taxonomy | bird⊥fish, robin≤animal; cross-checks **380/380** vs the kernel prover |
| `lang.py` | **LOTA** — small, expressive, *executable* agent language (S-expr; events/actions/quantifiers/modality/attitudes) desugaring to a first-order core | selftest: relations, quantifiers, negation-as-failure, events, actions |
| `lota_kernel.py` | **elaborate LOTA → kernel** (datalog searches, kernel checks); atoms/and/if/∃/∀, negation via exclusion axioms, modality | multi-hop isa + inherited props are kernel-checked terms |
| `prover.py` | **learned prover** (proposes proof terms; kernel gates) + **kernel-guided search** + **neural branch-ranker** | search 1.000 (sound 0.000 on false); ranker **564× fewer** nodes |
| `pipeline.py` | **NL question → kernel-verified answer**; paraphrase-robust front-end | held-out parse 1.000 / end-to-end 1.000 |
| `lang_bridge.py` | NL↔LOTA mapping; holdout regimes (combo / lexical / structural) | recombination **1.000**; novel words **0.000** (open-vocab wall) |
| `c2_eval.py` | real-English (contamination-clean) C2-style reading eval = the honest yardstick | small LM: lexical 0.73, discourse ≈ chance |
| `datalog.py` | Datalog engine: closure / entails / proof_tree (the untrusted proof *search*) | restored core; negation generalizes 1.00 held-out |

Supporting: `reason_realtext.py` (real-meaning KB + reasoning), `reason_write.py` (reason→render),
`write_lm.py` (char writer). RunPod launchers in `runpod/` (dry-run default, pod-side timeout +
always-terminate, key env-only).

**Sibling research track — extractive QA + in-context span PFN** (`EXTRACTIVE-QA-PFN.md`): the same
"model proposes, verifier bounds" posture applied to answer-span extraction — WordNet-grounded runtime
QA (F1 0.352, training-free) → from-scratch neural reader (F1 0.690) → one shipped base weight
(SQuAD 0.691 + TSE 0.420) → TabPFN-style in-context span PFN on real text (multi-task MEAN 0.763; the
honest finding: in-context generalization is **bounded by prior coverage**).

## 3. The trust architecture (borrowed from Lean)

The **kernel is the only trusted code.** Anything that *produces* terms — an elaborator, a tactic, or
a **neural model** — is **untrusted** and re-checked. This is the project thesis ("model proposes,
engine verifies") and it **structurally eliminates the hollow-verifier bug** that plagued earlier work
(closed-world "not in the set", or a semantic-equivalence metric that didn't discriminate): a claim is
true *only* with a proof term the kernel accepts.

- **Curry-Howard**: proposition = type, proof = term, checking a proof = type-checking. Implication and
  ∀ are both the dependent function type Π.
- **Inductive logic** (the `False/And/Or/Eq/Ex` prelude): standard constructor/eliminator types as
  sound axioms — full logic for *proof-checking* (no iota-computation; that's the deferred deep layer).

## 4. Two type schools, composing

- **Dependent types** (`kernel.py`) — for **proofs / correctness**.
- **Set-theoretic types** (`bdd_types.py`, the Elixir/BDD line) — for **subtyping / data**: ROBDD with
  hash-consing (structural dedup), subtyping `A ≤ B ⟺ A∧¬B = ∅`, emptiness grounded in the taxonomy.

**They compose:** set-theoretic types prove category **disjointness** (`bird ⊥ fish`), which licenses
kernel **exclusion axioms** (`∀x, isa x A → isa x B → False`), which let the kernel prove **negations**
as real *disproofs* — `(not (isa robin fish))` → `fun h => excl_bird_fish robin ⟨robin:bird⟩ h : False`
— replacing closed-world negation-as-failure with a kernel-checked disproof.

## 5. Neuro-symbolic search (the payoff)

`datalog` = proof **search** (untrusted) · `kernel` = proof **check** (trusted) · the model = **branch
ranker** (untrusted heuristic). On the proving task: kernel-guided search is sound (0.000 on false
goals) and complete-for-KB (1.000 on derivable); a small trained ranker cuts the search **2466 → 4.4
nodes (~564×)** with **soundness unchanged** — a bad ranking costs time, never correctness. This is the
clean division of labor: soundness from the kernel, completeness from search, efficiency from the model.

## 6. Modality + quantifiers

- `must φ` = kernel-provable (necessity, real proof); `can/may φ` = not-refutable (consistency
  *decision*; impossibility carries a kernel disproof). Full possible-worlds modal proofs deferred.
- `∃` = witness from the KB domain + `Ex_intro`; restricted `∀` = And-fold of implication proofs
  (closed-domain), honestly labeled.

## 7. Honest gaps (mapped, not hidden)

1. **Open-vocabulary NL — the wall to C2.** The compositional *mechanism* generalizes (recombination of
   seen words 1.000), but **novel words fail** (word-level 0.000). **GPU from-scratch probe (H100, 20k
   steps, char vs word):** char-level 0.013 / word 0.000 — char produces valid *structure* but the
   wrong novel word. **Verdict: from-scratch open-vocab does not generalize even with char + scale →
   C2 needs a pretrained BACKBONE** (path A). Empirical, not assumed.
2. **Structural systematicity** — unseen *constructions* don't generalize from scratch (SCAN-like).
3. **Deep modal logic** (possible-worlds proof terms) and **iota-computational inductives** (Nat,
   arithmetic) deferred.
4. **Lazy/eager BDD optimizations** (the Elixir posts' perf layer) — implemented (`empty_eager`) but a
   *scale-only* win; no benefit at toy size (honest stress finding).

## 8. C2 status

Real C2 = open-domain, near-native reading + writing + reasoning. We built the **verified-reasoning +
typed-language substrate** and an honest **C2 yardstick** (`c2_eval`, `C2_ROADMAP.md`). The one route
left to C2 is path A (a pretrained backbone as the NL front-end over this verified core); every
from-scratch avenue is empirically closed. **C2 remains unmet.**

## 9. Run it

```
for m in kernel bdd_types lang lota_kernel prover pipeline lang_bridge c2_eval; do
  python -m thinking.$m --selftest
done
python -m thinking.lota_kernel --demo     # LOTA surface → kernel-checked proof
python -m thinking.prover      --search    # kernel-guided search + neural guidance (564×)
python -m thinking.bdd_types   --cross     # set-theoretic vs kernel: 380/380 agree
```
