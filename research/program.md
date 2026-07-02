# Autonomous research loop — brain-grounded comprehension

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch): an agent improves a
single file under a fixed budget against a single metric. Here the target is **our in-house, LLM-free,
brain-grounded comprehension** — no pretrained models, no gradient training required.

## The task
Given a WordNet concept's **definition (gloss)**, identify its **is-a parent**. You improve the ranking
of candidate parents; the **brain verifies** every proposal and **picks the most specific provable
ancestor**. The brain guarantees correctness (it never accepts a false is-a), so your only job is to
make the right parent rank high and surface more true ancestors.

## What you may edit
- **`research/strategy.py`** — `propose(gloss, brain) -> [parent synset-names, best-first]`. THIS ONLY.

## What is FIXED (do not modify)
- **`research/harness.py`** — loads the held-out set, applies the brain gate, computes the metric.
- The brain, the dataset, the eval.

## The single metric
`exact-parent` accuracy over **all** held-out concepts (`exact / N`, printed by the harness) — the
end-to-end number: the strategy must both SURFACE a true ancestor (coverage) and have the brain pick the
exact direct parent. Higher is better. Secondary (reported): `coverage` (fraction with ≥1 true ancestor
surfaced). Note `metric ≈ coverage × precision-when-covered`, so raise either to improve.

## Hard constraints (a run is INVALID otherwise)
- **No cheating**: use ONLY `gloss` + the read-only `brain` API. You may NOT read the held concept's
  true parent/ancestors (the harness doesn't give them to you).
- **`false` must stay 0.000** — the brain gate enforces this; don't try to bypass it.
- **Budget**: the full eval must finish within the wall-clock budget (default 180s).

## How to run an experiment
```
python -m research.harness                 # scores research/strategy.py on the metric
python -m research.harness --max-concepts 20000 --seed 0
```
Record the metric, keep changes that improve it, revert ones that don't (autoresearch style).

## Ideas to try (the research)
- Better candidate generation: lemma substrings/morphology ("canine" ↔ "canid"), multi-word lemmas,
  the head noun of the gloss ("a **breed** of dog" → dog), synonyms of gloss words.
- Better ranking: prefer candidates whose own gloss overlaps the concept's gloss; weight by where the
  word appears in the gloss; combine depth with lexical-match strength.
- Use the brain's structure: among lexically-matched candidates, prefer ones connected in the taxonomy.
- Expand coverage without adding false positives (the brain gate keeps you honest).

The baseline (`baseline-lexical`) scores **metric ≈ 0.48** (exact over all held) at **coverage ≈ 0.71**
(so ~0.68 exact among the concepts it covers). Beat it — raise coverage and/or precision.
