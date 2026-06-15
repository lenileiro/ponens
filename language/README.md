# language

General-language track of the "general model, frontier-on-coding" goal: the model must learn
English meanings/associations, holistic understanding, writing, and **reasoning**, toward C2.

- `lang_exam.py` — read English (bidirectional masked reader, `ScratchpadLM causal=False`) from a
  tiny self-made corpus of LEARNABLE patterns, then **probe ON THE FLY** (novel generated items,
  no fixed test set / no contamination) for mastery + reasoning: subject-verb agreement, a/an,
  conditional tense, comprehension, and size-inference (reasoning). Answers by COMPREHENSION
  (fill [MASK] from both sides), not left-to-right next-token. **On-the-fly mastery: 1.00**
  (chance 0.50) across all categories incl. reasoning.

## Benchmark gate (held-out, contamination-free — analogous to datacurve for coding)
No single public "C2 exam" LLM benchmark exists, so use a mapped suite + on-the-fly probing primary:
- Comprehension/exam-style: **RACE**, **BELEBELE**
- Knowledge+reasoning (C2 breadth): **MMLU-Pro**, **BBH**, GPQA
- Grammar/proficiency (CEFR): **BLiMP**/**CoLA**, CEFR-J/Words-CEFR vocab
- Inference: **ANLI**
- Instruction/writing: **IFEval** + LLM-judge rubric
Honest: C2 + frontier-coding in one small model is scale-bound; these are demonstrations of the
comprehension/reasoning mechanism. Primary eval = on-the-fly mastery probe (per user); benchmarks
secondary, never trained on.
