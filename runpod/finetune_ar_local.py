"""Local MPS fine-tune: add input-noise robustness to the already-intelligible AR TTS checkpoint to
fix FREE-RUNNING drift (exposure bias) without a from-scratch GPU run. Loads runs/tts_ar.pt (r=1),
continues training with teacher-forcing input noise, checkpoints to runs/tts_ar_ft.pt, and verifies
free-running synthesis with a Whisper ASR round-trip.

  .venv/bin/python runpod/finetune_ar_local.py --steps 12000
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import thinking.tts_ar as T
T.R = 1                                                   # match the existing r=1 checkpoint
from thinking.tts_ar import TransformerTTS, _batch
from thinking.tts import load, guided_attention_loss, encode_text


def finetune(steps=12000, lr=1e-4, batch=16, ckpt="runs/tts_ar.pt", out="runs/tts_ar_ft.pt"):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    data = load(); td = data[:int(len(data) * 0.95)]
    m = TransformerTTS(r=1).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    rng = np.random.default_rng(0); t0 = time.time()
    for st in range(1, steps + 1):
        idt, mel, stop, il, ml, gl = _batch(td, rng, batch, dev)
        co, re, sl, al = m.decode(idt, mel)               # NOISE applied (model.training)
        mm = (torch.arange(mel.shape[2], device=dev)[None] < ml[:, None]).float()
        l_mel = sum((F.l1_loss(p, mel, reduction="none").mean(1) * mm).sum() / mm.sum() for p in (co, re))
        l_stop = F.binary_cross_entropy_with_logits(sl, stop)
        l_ga = sum(guided_attention_loss(a[b:b + 1], int(il[b]), int(gl[b]))
                   for a in al for b in range(batch)) / (batch * len(al))
        (l_mel + l_stop + 5.0 * l_ga).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
        if st % 500 == 0:
            print(f"  ft {st}/{steps} mel {l_mel.item():.3f} ga {l_ga.item():.4f} "
                  f"({(time.time() - t0) / st:.2f}s/st)", flush=True)
        if st % 2000 == 0:
            torch.save({"state_dict": m.state_dict()}, out)
    torch.save({"state_dict": m.state_dict()}, out)
    print(f"saved {out}", flush=True)
    return m, data


def asr_check(m, data, n=6):
    from transformers import pipeline
    from thinking.realvoice import Vocos, write_wav
    from thinking.say_test import wer, PROMPTS
    dev = next(m.parameters()).device
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=-1)
    voc = Vocos(d=512).to(dev)
    voc.load_state_dict(torch.load("runs/realvoice.pt", map_location=dev)["state_dict"]); voc.eval()
    m.eval(); wers = []
    for i, txt in enumerate(PROMPTS[:n]):
        ids = torch.tensor([encode_text(txt)], device=dev)
        with torch.no_grad():
            spec = m.infer(ids)                            # FREE-RUNNING
            wav = voc(spec)[0].cpu().numpy()
        p = f"data/synth/FT_say_{i}.wav"; write_wav(p, wav)
        hyp = asr(p)["text"].strip(); e = wer(txt, hyp); wers.append(e)
        print(f"  [{i}] WER {e:.2f}  ref {txt!r}\n        asr {hyp[:70]!r}", flush=True)
    print(f"\nFREE-RUNNING mean WER {np.mean(wers):.3f}  (lower=better; <0.4 ~ intelligible)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    args = ap.parse_args()
    m, data = finetune(steps=args.steps)
    asr_check(m, data)


if __name__ == "__main__":
    main()
