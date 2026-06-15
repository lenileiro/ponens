# coding_agent

In-house coding-agent program: train a model to **write code, debug/fix bugs, and use tools
(bash/git/docker) like a software engineer**, evaluated through a bash harness, with the end goal
of passing the [datacurve.ai](https://datacurve.ai/) SWE benchmark **without test contamination**.

## Layout

### `data/` — corpus builders (continuation manifests for `thinking.multimodal`, causal next-token)
- `build_codemix.py` — code-heavy mix (per-file contiguous windows, `source`+`chunk_index` meta so
  the trainer auto-detects `causal`; `target == next.text`).
- `build_bigcorpus.py` — multi-GB streaming version (low local RAM); ship via the launcher's
  `--multimodal-upload-manifest` (robust sharded scp).
- `build_toolformat.py` — bash-ReAct protocol transcripts (write code → run → fix), grounded.
- `build_devtools.py` — **tool/computer-use mastery**: git/docker/pkg-mgr/build-test/edit/process +
  multi-step end-to-end workflows (fix-test→commit, branch→write→test→push, read-traceback→patch).
- `build_chartool.py` — compact char-level executable task↔command-coupled transcripts.
- `build_induction.py` — in-context **rule induction** blocks (unnamed rule shown by examples).

### `harness/` — the acceptance rig (ReAct loop, sandboxed bash + success checks)
- `coding_agent.py` — load a checkpoint, generate `$ ` commands, **execute in a sandbox tmpdir
  (timeout + denylist)**, feed output back, score write/debug/fix tasks. Causal inference
  (prompt in `ids`, empty prefix). Char-aware decode.
- `char_test.py`, `induction_test.py` — task-family + rule-induction evals.

### `experiments/` — rule-finding / length-generalization (the project's "find rules" thesis)
- `scan_lengthgen.py` — running-sum scan; **recurrence (loop_inject + windowed attn + per-loop
  deep supervision) length-generalizes** (train len 4–16 → len 24 ≈ train accuracy).
- `iter_lengthgen.py`, `parity_recur.py` — iterated-computation / parity length-gen probes.

### `rl/` — RL-in-harness (the agentic phase, per Cursor Composer 2)
- `rl_harness.py` — **Dr.GRPO with VERIFIABLE rewards**: sample K rollouts → execute → reward =
  correct output → advantage = reward − group_mean (no std-normalize). Warmup SFT for a non-chance
  base, then RL lifts harness success.

### `runpod/` — H100 launch templates (run from repo root; need `RUNPOD_API_KEY` in env)
- `launch_full.sh` / `launch_scale.sh` / `launch_coding.sh` / `launch_smoke.sh` — multimodal causal
  training configs (dim/batch/steps, `--decode-objective causal`, `--max-vocab`, repetition-
  unlikelihood, `--multimodal-upload-manifest` for multi-GB).

## Key findings (so far)
- Causal next-token over contiguous code windows fixed an earlier seq2seq **memorization collapse**.
- Small char model: **9/9** on the narrow write/run/debug/fix harness; in-context induction 0.95 on
  trained rules, but novel-rule (0.25) / length-gen (0.13) are walls **unless** recurrence is used.
- **Recurrence finds rules with iterated structure** (scan length-gen 0.13→0.90, scales with
  compute) but not global one-shot transforms.
- **Composer 2 recipe** (continued-pretrain a frontier base → RL-in-harness) is the realistic path
  to frontier SWE; our from-scratch small model is a research-faithful track. Tool-use + harness +
  contamination-free eval are validated; the binding constraint is model/data **scale**.
