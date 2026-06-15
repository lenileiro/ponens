# Fine-tuning guide

How to continue-train (fine-tune) a saved checkpoint in this repo instead of training from scratch.
Worked example: fixing the autoregressive TTS model's free-running drift. The same recipe applies to
any module that saves a `{"state_dict": ...}` checkpoint.

## When to fine-tune (vs. train from scratch)

Fine-tune when a checkpoint is *mostly right* and you want to change one thing cheaply:

- **Fix a behavior gap** the original objective didn't cover (our case: the AR TTS spoke perfectly
  *teacher-forced* but drifted in *free-running* inference — exposure bias).
- **Adapt to new data / a new speaker / a new style** without re-learning everything.
- **Add an auxiliary loss or robustness trick** (input noise, a new regularizer).

Train from scratch instead when the architecture changes shape (e.g. the TTS reduction factor
`r=1 -> r=3` changes the output layer, so the old checkpoint can't load — see "Shape changes" below).

## The recipe (5 steps)

1. **Load** the checkpoint into a matching architecture.
2. **Lower the learning rate** (typically 3–10× below the from-scratch LR; we use `1e-4` vs `3e-4`).
   A high LR on a converged model erases what it learned.
3. **Change exactly one thing** — the new data, the new loss term, or the robustness trick. Keep the
   rest of the training loop identical so you can attribute the result.
4. **Checkpoint periodically** so a long run is never lost, and so you can pick the best intermediate.
5. **Verify with an objective metric**, not training loss — for TTS that's a Whisper ASR round-trip
   (word error rate), not the mel L1.

## Worked example: `runpod/finetune_ar_local.py`

Fixes free-running drift in the AR TransformerTTS by continuing training with **teacher-forcing input
noise** (`NOISE` in `thinking/tts_ar.py`). Noise on the previous-frame input teaches the model to
recover from its own imperfect predictions, which is exactly what happens at inference.

```bash
# Local, on the Mac GPU (MPS) or CPU — no cloud needed. PYTHONPATH so `thinking` resolves.
PYTHONPATH=/Users/leiro/workspace/llm \
  .venv/bin/python -u runpod/finetune_ar_local.py --steps 12000
```

What it does, in order:
- `thinking.tts_ar.R = 1` — match the reduction factor the existing `runs/tts_ar.pt` was trained with
  (the live module default is `R=3`; the saved checkpoint is `r=1`, so we override before importing
  the batcher/model so shapes line up).
- `TransformerTTS(r=1)` + `load_state_dict(torch.load("runs/tts_ar.pt"))`.
- AdamW at `lr=1e-4` (fine-tune LR), batch 16, `NOISE`-augmented teacher forcing (active because
  `model.training` is True — see `TransformerTTS.decode`).
- Saves to `runs/tts_ar_ft.pt` every 2000 steps and at the end.
- `asr_check()` runs **free-running** `model.infer()` on held-out `say_test.PROMPTS`, vocodes with
  `runs/realvoice.pt`, and prints the Whisper WER — the real pass/fail signal.

Key knobs (edit in `runpod/finetune_ar_local.py` / `thinking/tts_ar.py`):

| Knob | Where | Meaning |
|------|-------|---------|
| `--steps` | CLI | fine-tune length (12k ≈ 1.3 h on MPS @ ~0.4 s/step) |
| `lr` | `finetune()` | fine-tune learning rate (default `1e-4`) |
| `NOISE` | `tts_ar.py` | std of teacher-forcing input noise (default `0.25`) — the robustness fix |
| `R` | `tts_ar.py` | reduction factor; must match the checkpoint being loaded |
| `ckpt` / `out` | `finetune()` | source checkpoint / fine-tuned output path |

## Fine-tuning on GPU instead of local

When the RunPod balance is funded, run the from-scratch / longer job through the launcher
(`runpod/launch_audio.py`, job `tts-ar`) which deploys the pinned git HEAD, uploads
`runs/realvoice.pt`, trains, and fetches `runs/`. To make it a *fine-tune* rather than from-scratch,
add a `--resume runs/tts_ar.pt` path to `thinking/tts_ar.py:train()` (load before the loop) and a
lower LR; the launcher job line lives in `runpod/launch_audio.py:payload()`.

## Shape changes (why some checkpoints can't be fine-tuned directly)

`load_state_dict` requires identical tensor shapes. If you changed:
- the **reduction factor** `r` (output layer is `Linear(d, r*n_bins)`),
- the **hidden width** `d`, layer count, or head count,

then the old checkpoint won't load and you must train from scratch (or write a partial-load that
copies only the matching tensors). This is why the `r=3` drift-fix is a *fresh* GPU run, while the
`r=1` noise-robustness adaptation is a *fine-tune* of the existing checkpoint.

## Verification checklist

- [ ] Fine-tune loss starts near the original checkpoint's loss (confirms it loaded, LR isn't wrecking it).
- [ ] The objective metric improves on the **target gap** (free-running WER down), not just train loss.
- [ ] No regression on what already worked (teacher-forced quality, alignment focus).
- [ ] Keep the best intermediate checkpoint, not necessarily the last.
